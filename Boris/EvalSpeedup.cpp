#include "stdafx.h"
#include "EvalSpeedup.h"

#if defined(MODULE_COMPILATION_DEMAG) || defined(MODULE_COMPILATION_SDEMAG) || defined(MODULE_COMPILATION_ATOM_DIPOLEDIPOLE)

#include "VEC_MeshTransfer.h"

BError EvalSpeedup::Initialize_EvalSpeedup(
	//pre-calculated self demag coefficient
	DBL3 selfDemagCoeff_,
	//what speedup factor (polynomial order) is being used?
	int evaluation_speedup_factor,
	//cell-size and rect for Hdemag cuVECs
	DBL3 h, Rect meshRect,
	std::function<bool(VEC<DBL3>&)>& initialize_mesh_transfer)
{
	BError error(CLASS_STR(EvalSpeedup));

	selfDemagCoeff = selfDemagCoeff_;

	//make sure to allocate memory for Hdemag if we need it
	if (evaluation_speedup_factor >= 6) { if (!Hdemag6.resize(h, meshRect) || !initialize_mesh_transfer(Hdemag6)) return error(BERROR_OUTOFMEMORY_CRIT); }
	else Hdemag6.clear();

	if (evaluation_speedup_factor >= 5) { if (!Hdemag5.resize(h, meshRect) || !initialize_mesh_transfer(Hdemag5)) return error(BERROR_OUTOFMEMORY_CRIT); }
	else Hdemag5.clear();

	if (evaluation_speedup_factor >= 4) { if (!Hdemag4.resize(h, meshRect) || !initialize_mesh_transfer(Hdemag4)) return error(BERROR_OUTOFMEMORY_CRIT); }
	else Hdemag4.clear();

	if (evaluation_speedup_factor >= 3) { if (!Hdemag3.resize(h, meshRect) || !initialize_mesh_transfer(Hdemag3)) return error(BERROR_OUTOFMEMORY_CRIT); }
	else Hdemag3.clear();

	if (evaluation_speedup_factor >= 2) { if (!Hdemag2.resize(h, meshRect) || !initialize_mesh_transfer(Hdemag2)) return error(BERROR_OUTOFMEMORY_CRIT); }
	else Hdemag2.clear();

	if (evaluation_speedup_factor >= 1) { if (!Hdemag.resize(h, meshRect) || !initialize_mesh_transfer(Hdemag)) return error(BERROR_OUTOFMEMORY_CRIT); }
	else Hdemag.clear();

	num_Hdemag_saved = 0;

	return error;
}

void EvalSpeedup::Initialize_EvalSpeedup_Mode_Atom(VEC<DBL3>& M_VEC, VEC<DBL3>& H_VEC)
{
	eval_speedup_mode = EVALSPEEDUP_MODE_ATOM;

	pM_VEC = &M_VEC;

	pH_VEC = &H_VEC;

	//make everything else nullptr, zero size
	pM_VEC_VC = nullptr;
	pM2_VEC_VC = nullptr;

	pH2_VEC = nullptr;
}

void EvalSpeedup::Initialize_EvalSpeedup_Mode_FM(VEC_VC<DBL3>& M_VEC_VC, VEC<DBL3>& H_VEC)
{
	eval_speedup_mode = EVALSPEEDUP_MODE_FM;

	pM_VEC_VC = &M_VEC_VC;
	pH_VEC = &H_VEC;

	//make everything else nullptr, zero size
	pM_VEC = nullptr;

	pM2_VEC_VC = nullptr;

	pH2_VEC = nullptr;
}

void EvalSpeedup::Initialize_EvalSpeedup_Mode_AFM(
	VEC_VC<DBL3>& M_VEC_VC, VEC_VC<DBL3>& M2_VEC_VC,
	VEC<DBL3>& H_VEC, VEC<DBL3>& H2_VEC)
{
	eval_speedup_mode = EVALSPEEDUP_MODE_AFM;

	pM_VEC_VC = &M_VEC_VC;
	pM2_VEC_VC = &M2_VEC_VC;

	pH_VEC = &H_VEC;
	pH2_VEC = &H2_VEC;

	//make everything else nullptr, zero size
	pM_VEC = nullptr;
}

void EvalSpeedup::UpdateConfiguration_EvalSpeedup(void)
{
	Hdemag.clear();
	Hdemag2.clear();
	Hdemag3.clear();
	Hdemag4.clear();
	Hdemag5.clear();
	Hdemag6.clear();

	num_Hdemag_saved = 0;
}

//check if speedup should be done (true) or not (false)
//if true, then caller should then run the method below (UpdateField_EvalSpeedup) instead of its no speedup computation
bool EvalSpeedup::Check_if_EvalSpeedup(int eval_speedup_factor, bool check_step_update)
{
	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////// NO SPEEDUP //////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	//No Speedup computation if:
	//1. speedup disabled, or
	// 
	//2a. we don't yet have enough previous stored evaluations for the given speedup factor (polynomial extrapolation order), and
	//2b. not being asked to compute one (check_step_update false), and also:
	//2c. speedup factor is greater than step type
	//Reason for 2c is for step type always compute and store a previous evaluation, since this could be used for demag time-step greater than ODE time-step
	if (!eval_speedup_factor || (num_Hdemag_saved < eval_speedup_factor && !check_step_update && eval_speedup_factor > 1)) {

		//return false, so caller can run demag computation without speedup
		return false;
	}

	return true;
}

void EvalSpeedup::UpdateField_EvalSpeedup(
	int eval_speedup_factor, bool check_step_update,
	double eval_step_time,
	std::function<void(VEC<DBL3>&)>& do_evaluation,
	std::function<void(void)>& do_transfer_in, std::function<void(VEC<DBL3>&)>& do_transfer_out)
{
	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////// EVAL SPEEDUP /////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	//update if required by ODE solver or if we don't have enough previous evaluations saved to extrapolate
	if (check_step_update || num_Hdemag_saved < eval_speedup_factor) {

		VEC<DBL3>* pHdemag;

		if (num_Hdemag_saved < eval_speedup_factor) {

			//don't have enough evaluations, so save next one
			switch (num_Hdemag_saved)
			{
			default:
			case 0:
				pHdemag = &Hdemag;
				time_demag1 = eval_step_time;
				break;
			case 1:
				pHdemag = &Hdemag2;
				time_demag2 = eval_step_time;
				break;
			case 2:
				pHdemag = &Hdemag3;
				time_demag3 = eval_step_time;
				break;
			case 3:
				pHdemag = &Hdemag4;
				time_demag4 = eval_step_time;
				break;
			case 4:
				pHdemag = &Hdemag5;
				time_demag5 = eval_step_time;
				break;
			case 5:
				pHdemag = &Hdemag6;
				time_demag6 = eval_step_time;
				break;
			}

			num_Hdemag_saved++;
		}
		else {

			//have enough evaluations saved, so just cycle between them now

			switch (eval_speedup_factor) {

			case EVALSPEEDUP_QUINTIC:
			{

				//1, 2, 3, 4, 5, 6 -> next is 1
				if (time_demag6 > time_demag5 && time_demag5 > time_demag4 && time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

					pHdemag = &Hdemag;
					time_demag1 = eval_step_time;
				}
				//2, 3, 4, 5, 6, 1 -> next is 2
				else if (time_demag1 > time_demag2) {

					pHdemag = &Hdemag2;
					time_demag2 = eval_step_time;
				}
				//3, 4, 5, 6, 1, 2 -> next is 3
				else if (time_demag2 > time_demag3) {

					pHdemag = &Hdemag3;
					time_demag3 = eval_step_time;
				}
				//4, 5, 6, 1, 2, 3 -> next is 4
				else if (time_demag3 > time_demag4) {

					pHdemag = &Hdemag4;
					time_demag4 = eval_step_time;
				}
				//5, 6, 1, 2, 3, 4 -> next is 5
				else if (time_demag4 > time_demag5) {

					pHdemag = &Hdemag5;
					time_demag5 = eval_step_time;
				}
				else {

					pHdemag = &Hdemag6;
					time_demag6 = eval_step_time;
				}
			}
			break;

			case EVALSPEEDUP_QUARTIC:
			{

				//1, 2, 3, 4, 5 -> next is 1
				if (time_demag5 > time_demag4 && time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

					pHdemag = &Hdemag;
					time_demag1 = eval_step_time;
				}
				//2, 3, 4, 5, 1 -> next is 2
				else if (time_demag1 > time_demag2) {

					pHdemag = &Hdemag2;
					time_demag2 = eval_step_time;
				}
				//3, 4, 5, 1, 2 -> next is 3
				else if (time_demag2 > time_demag3) {

					pHdemag = &Hdemag3;
					time_demag3 = eval_step_time;
				}
				//4, 5, 1, 2, 3 -> next is 4
				else if (time_demag3 > time_demag4) {

					pHdemag = &Hdemag4;
					time_demag4 = eval_step_time;
				}
				else {

					pHdemag = &Hdemag5;
					time_demag5 = eval_step_time;
				}
			}
			break;

			case EVALSPEEDUP_CUBIC:
			{

				//1, 2, 3, 4 -> next is 1
				if (time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

					pHdemag = &Hdemag;
					time_demag1 = eval_step_time;
				}
				//2, 3, 4, 1 -> next is 2
				else if (time_demag1 > time_demag2) {

					pHdemag = &Hdemag2;
					time_demag2 = eval_step_time;
				}
				//3, 4, 1, 2 -> next is 3
				else if (time_demag2 > time_demag3) {

					pHdemag = &Hdemag3;
					time_demag3 = eval_step_time;
				}
				else {

					pHdemag = &Hdemag4;
					time_demag4 = eval_step_time;
				}
			}
			break;

			case EVALSPEEDUP_QUADRATIC:
			{

				//1, 2, 3 -> next is 1
				if (time_demag3 > time_demag2 && time_demag2 > time_demag1) {

					pHdemag = &Hdemag;
					time_demag1 = eval_step_time;
				}
				//2, 3, 1 -> next is 2
				else if (time_demag3 > time_demag2 && time_demag1 > time_demag2) {

					pHdemag = &Hdemag2;
					time_demag2 = eval_step_time;
				}
				//3, 1, 2 -> next is 3, leading to 1, 2, 3 again
				else {

					pHdemag = &Hdemag3;
					time_demag3 = eval_step_time;
				}
			}
			break;

			case EVALSPEEDUP_LINEAR:
			{

				//1, 2 -> next is 1
				if (time_demag2 > time_demag1) {

					pHdemag = &Hdemag;
					time_demag1 = eval_step_time;
				}
				//2, 1 -> next is 2, leading to 1, 2 again
				else {

					pHdemag = &Hdemag2;
					time_demag2 = eval_step_time;
				}
			}
			break;

			default:
			case EVALSPEEDUP_STEP:
			{
				pHdemag = &Hdemag;
			}
			break;
			};
		}

		//do evaluation as implemented by caller, with any transfer in required first
		do_transfer_in();
		do_evaluation(*pHdemag);

		//subtract self demag from *pHDemag (unless it's the step method)
		switch (eval_speedup_mode) {

		case EVALSPEEDUP_MODE_FM:
		{
			//add contribution to Heff
#pragma omp parallel for
			for (int idx = 0; idx < pHdemag->linear_size(); idx++) {

				(*pH_VEC)[idx] += (*pHdemag)[idx];
				//subtract self demag contribution (unless it's the step method): we'll add in again for the new magnetization, so it least the self demag is exact
				if (eval_speedup_factor > 1) (*pHdemag)[idx] -= (selfDemagCoeff & (*pM_VEC_VC)[idx]);
			}
		}
			break;

		case EVALSPEEDUP_MODE_AFM:
		{
			//add contribution to Heff and Heff2
#pragma omp parallel for
			for (int idx = 0; idx < pHdemag->linear_size(); idx++) {

				(*pH_VEC)[idx] += (*pHdemag)[idx];
				(*pH2_VEC)[idx] += (*pHdemag)[idx];
				//subtract self demag contribution (unless it's the step method): we'll add in again for the new magnetization, so it least the self demag is exact
				if (eval_speedup_factor > 1) (*pHdemag)[idx] -= (selfDemagCoeff & ((*pM_VEC_VC)[idx] + (*pM2_VEC_VC)[idx]) / 2);
			}
		}
			break;

		case EVALSPEEDUP_MODE_ATOM:
			//transfer field to atomistic mesh, then subtract self contribution from evaluation for later use
			do_transfer_out(*pHdemag);
			//subtract self demag contribution (unless it's the step method): we'll add in again for the new magnetization, so it least the self demag is exact
			if (eval_speedup_factor > 1) {
#pragma omp parallel for
				for (int idx = 0; idx < pHdemag->linear_size(); idx++) {

					(*pHdemag)[idx] -= (selfDemagCoeff & (*pM_VEC)[idx]);
				}
			}
			break;
		};
	}
	else {

		//transfer data in if needed, depending on mode set
		//in any case this is not needed for step method, since no self demag correction is used for step method
		if (eval_speedup_factor > 1) {

			switch (eval_speedup_mode) {

			case EVALSPEEDUP_MODE_FM:
			case EVALSPEEDUP_MODE_AFM:
				//not needed here
				break;

			case EVALSPEEDUP_MODE_ATOM:
				do_transfer_in();
				break;
			};
		}

		//not required to update, and we have enough previous evaluations: use previous Hdemag saves to extrapolate for current evaluation

		double a1 = 1.0, a2 = 0.0, a3 = 0.0, a4 = 0.0, a5 = 0.0, a6 = 0.0;
		double time = eval_step_time;

		switch (eval_speedup_factor) {

		case EVALSPEEDUP_QUINTIC:
		{
			a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4) * (time_demag1 - time_demag5) * (time_demag1 - time_demag6));
			a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4) * (time_demag2 - time_demag5) * (time_demag2 - time_demag6));
			a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4) * (time_demag3 - time_demag5) * (time_demag3 - time_demag6));
			a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag5) * (time - time_demag6) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3) * (time_demag4 - time_demag5) * (time_demag4 - time_demag6));
			a5 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag6) / ((time_demag5 - time_demag1) * (time_demag5 - time_demag2) * (time_demag5 - time_demag3) * (time_demag5 - time_demag4) * (time_demag5 - time_demag6));
			a6 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag6 - time_demag1) * (time_demag6 - time_demag2) * (time_demag6 - time_demag3) * (time_demag6 - time_demag4) * (time_demag6 - time_demag5));

			switch (eval_speedup_mode) {

			case EVALSPEEDUP_MODE_FM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + Hdemag6[idx] * a6 + (selfDemagCoeff & (*pM_VEC_VC)[idx]);
				}
			}
				break;

			case EVALSPEEDUP_MODE_AFM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					DBL3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + Hdemag6[idx] * a6 + (selfDemagCoeff & ((*pM_VEC_VC)[idx] + (*pM2_VEC_VC)[idx]) / 2);
					(*pH_VEC)[idx] += Hdemag_value;
					(*pH2_VEC)[idx] += Hdemag_value;
				}
			}
				break;

			case EVALSPEEDUP_MODE_ATOM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + Hdemag6[idx] * a6 + (selfDemagCoeff & (*pM_VEC)[idx]);
				}

				do_transfer_out(*pH_VEC);
			}
				break;
			};
		}
		break;

		case EVALSPEEDUP_QUARTIC:
		{
			a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4) * (time_demag1 - time_demag5));
			a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4) * (time_demag2 - time_demag5));
			a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) * (time - time_demag5) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4) * (time_demag3 - time_demag5));
			a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag5) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3) * (time_demag4 - time_demag5));
			a5 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) / ((time_demag5 - time_demag1) * (time_demag5 - time_demag2) * (time_demag5 - time_demag3) * (time_demag5 - time_demag4));

			switch (eval_speedup_mode) {

			case EVALSPEEDUP_MODE_FM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + (selfDemagCoeff & (*pM_VEC_VC)[idx]);
				}
			}
				break;

			case EVALSPEEDUP_MODE_AFM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					DBL3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + (selfDemagCoeff & ((*pM_VEC_VC)[idx] + (*pM2_VEC_VC)[idx]) / 2);
					(*pH_VEC)[idx] += Hdemag_value;
					(*pH2_VEC)[idx] += Hdemag_value;
				}
			}
				break;

			case EVALSPEEDUP_MODE_ATOM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + (selfDemagCoeff & (*pM_VEC)[idx]);
				}

				do_transfer_out(*pH_VEC);
			}
				break;
			};
		}
		break;

		case EVALSPEEDUP_CUBIC:
		{
			a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4));
			a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4));
			a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4));
			a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3));

			switch (eval_speedup_mode) {

			case EVALSPEEDUP_MODE_FM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + (selfDemagCoeff & (*pM_VEC_VC)[idx]);
				}
			}
				break;

			case EVALSPEEDUP_MODE_AFM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					DBL3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + (selfDemagCoeff & ((*pM_VEC_VC)[idx] + (*pM2_VEC_VC)[idx]) / 2);
					(*pH_VEC)[idx] += Hdemag_value;
					(*pH2_VEC)[idx] += Hdemag_value;
				}
			}
				break;

			case EVALSPEEDUP_MODE_ATOM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + (selfDemagCoeff & (*pM_VEC)[idx]);
				}

				do_transfer_out(*pH_VEC);
			}
				break;
			};
		}
		break;

		case EVALSPEEDUP_QUADRATIC:
		{

			if (time_demag2 != time_demag1 && time_demag2 != time_demag3 && time_demag1 != time_demag3) {

				a1 = (time - time_demag2) * (time - time_demag3) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3));
				a2 = (time - time_demag1) * (time - time_demag3) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3));
				a3 = (time - time_demag1) * (time - time_demag2) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2));
			}

			switch (eval_speedup_mode) {

			case EVALSPEEDUP_MODE_FM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + (selfDemagCoeff & (*pM_VEC_VC)[idx]);
				}
			}
				break;

			case EVALSPEEDUP_MODE_AFM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					DBL3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + (selfDemagCoeff & ((*pM_VEC_VC)[idx] + (*pM2_VEC_VC)[idx]) / 2);
					(*pH_VEC)[idx] += Hdemag_value;
					(*pH2_VEC)[idx] += Hdemag_value;
				}
			}
				break;

			case EVALSPEEDUP_MODE_ATOM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + (selfDemagCoeff & (*pM_VEC)[idx]);
				}

				do_transfer_out(*pH_VEC);
			}
				break;
			};
		}
		break;

		case EVALSPEEDUP_LINEAR:
		{

			if (time_demag2 != time_demag1) {

				a1 = (time - time_demag2) / (time_demag1 - time_demag2);
				a2 = (time - time_demag1) / (time_demag2 - time_demag1);
			}

			switch (eval_speedup_mode) {

			case EVALSPEEDUP_MODE_FM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + (selfDemagCoeff & (*pM_VEC_VC)[idx]);
				}
			}
				break;

			case EVALSPEEDUP_MODE_AFM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					DBL3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + (selfDemagCoeff & ((*pM_VEC_VC)[idx] + (*pM2_VEC_VC)[idx]) / 2);
					(*pH_VEC)[idx] += Hdemag_value;
					(*pH2_VEC)[idx] += Hdemag_value;
				}
			}
				break;

			case EVALSPEEDUP_MODE_ATOM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + (selfDemagCoeff & (*pM_VEC)[idx]);
				}

				do_transfer_out(*pH_VEC);
			}
				break;
			};
		}
		break;

		case EVALSPEEDUP_STEP:
		{
			// no need to add self demag for step method
			switch (eval_speedup_mode) {

			case EVALSPEEDUP_MODE_FM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					(*pH_VEC)[idx] += Hdemag[idx];
				}
			}
				break;

			case EVALSPEEDUP_MODE_AFM:
			{
#pragma omp parallel for
				for (int idx = 0; idx < Hdemag.linear_size(); idx++) {

					DBL3 Hdemag_value = Hdemag[idx];
					(*pH_VEC)[idx] += Hdemag_value;
					(*pH2_VEC)[idx] += Hdemag_value;
				}
			}
				break;

			case EVALSPEEDUP_MODE_ATOM:
			{
				do_transfer_out(Hdemag);
			}
				break;
			};
		}
		break;
		}
	}
}

#endif
