#include "stdafx.h"
#include "EvalSpeedupCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_DEMAG) || defined(MODULE_COMPILATION_SDEMAG) || defined(MODULE_COMPILATION_ATOM_DIPOLEDIPOLE)

#include "VEC_MeshTransfer.h"

BError EvalSpeedupCUDA::Initialize_EvalSpeedup(
	//pre-calculated self demag coefficient
	cuReal3 selfDemagCoeff_cpu,
	//what speedup factor (polynomial order) is being used?
	int evaluation_speedup_factor,
	//cell-size and rect for Hdemag cuVECs
	cuReal3 h, cuRect meshRect,
	//if transfer to other H cuVECs is required from Hdemag cuVECs then set it here, with transfer info pre-calculated
	std::vector<mcu_VEC(cuReal3)*> pVal_to_H, std::vector<mcu_VEC(cuReal3)*> pVal_to_H2, Transfer<DBL3>* ptransfer_info_cpu,
	std::vector<mcu_VEC(cuReal3)*> pVal2_to_H, Transfer<DBL3>* ptransfer2_info_cpu)
{
	BError error(CLASS_STR(EvalSpeedupCUDA));

	selfDemagCoeff.from_cpu(selfDemagCoeff_cpu);

	//make sure to allocate memory for Hdemag if we need it
	if (evaluation_speedup_factor >= 6) { if (!Hdemag6.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	else Hdemag6.clear();

	if (evaluation_speedup_factor >= 5) { if (!Hdemag5.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	else Hdemag5.clear();

	if (evaluation_speedup_factor >= 4) { if (!Hdemag4.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	else Hdemag4.clear();

	if (evaluation_speedup_factor >= 3) { if (!Hdemag3.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	else Hdemag3.clear();

	if (evaluation_speedup_factor >= 2) { if (!Hdemag2.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	else Hdemag2.clear();

	if (evaluation_speedup_factor >= 1) { if (!Hdemag.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	else Hdemag.clear();

	if (!error && pVal_to_H.size() && ptransfer_info_cpu) {

		if (!pVal_to_H2.size()) {

			if (evaluation_speedup_factor >= 1) if (!Hdemag.copy_transfer_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal_to_H, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (evaluation_speedup_factor >= 2) if (!Hdemag2.copy_transfer_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal_to_H, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (evaluation_speedup_factor >= 3) if (!Hdemag3.copy_transfer_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal_to_H, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (evaluation_speedup_factor >= 4) if (!Hdemag4.copy_transfer_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal_to_H, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (evaluation_speedup_factor >= 5) if (!Hdemag5.copy_transfer_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal_to_H, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (evaluation_speedup_factor >= 6) if (!Hdemag6.copy_transfer_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal_to_H, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		}
		else {

			//This is used if antiferromagnetic meshes are present
			if (evaluation_speedup_factor >= 1) if (!Hdemag.copy_transfer_info_averagedinputs_duplicatedoutputs<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, {}, pVal_to_H, pVal_to_H2, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (evaluation_speedup_factor >= 2) if (!Hdemag2.copy_transfer_info_averagedinputs_duplicatedoutputs<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, {}, pVal_to_H, pVal_to_H2, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (evaluation_speedup_factor >= 3) if (!Hdemag3.copy_transfer_info_averagedinputs_duplicatedoutputs<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, {}, pVal_to_H, pVal_to_H2, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (evaluation_speedup_factor >= 4) if (!Hdemag4.copy_transfer_info_averagedinputs_duplicatedoutputs<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, {}, pVal_to_H, pVal_to_H2, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (evaluation_speedup_factor >= 5) if (!Hdemag5.copy_transfer_info_averagedinputs_duplicatedoutputs<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, {}, pVal_to_H, pVal_to_H2, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (evaluation_speedup_factor >= 6) if (!Hdemag6.copy_transfer_info_averagedinputs_duplicatedoutputs<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, {}, pVal_to_H, pVal_to_H2, *ptransfer_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		}
	}

	if (!error && pVal2_to_H.size() && ptransfer2_info_cpu) {

		//additional transfer (e.g. both atomistic and (anti)ferromagnetic meshes present)
		if (evaluation_speedup_factor >= 1) if (!Hdemag.copy_transfer2_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal2_to_H, *ptransfer2_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		if (evaluation_speedup_factor >= 2) if (!Hdemag2.copy_transfer2_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal2_to_H, *ptransfer2_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		if (evaluation_speedup_factor >= 3) if (!Hdemag3.copy_transfer2_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal2_to_H, *ptransfer2_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		if (evaluation_speedup_factor >= 4) if (!Hdemag4.copy_transfer2_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal2_to_H, *ptransfer2_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		if (evaluation_speedup_factor >= 5) if (!Hdemag5.copy_transfer2_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal2_to_H, *ptransfer2_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		if (evaluation_speedup_factor >= 6) if (!Hdemag6.copy_transfer2_info<cuVEC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal2_to_H, *ptransfer2_info_cpu)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
	}

	num_Hdemag_saved = 0;

	return error;
}

void EvalSpeedupCUDA::Initialize_EvalSpeedup_Mode_Atom(mcu_VEC(cuReal3)& M_cuVEC, mcu_VEC(cuReal3)& H_cuVEC)
{
	eval_speedup_mode = EVALSPEEDUP_MODE_ATOM;

	pM_cuVEC = &M_cuVEC;
	pH_cuVEC = &H_cuVEC;

	//make everything else nullptr, zero size
	pM_cuVEC_VC = nullptr;
	pM2_cuVEC_VC = nullptr;

	pH2_cuVEC = nullptr;
}

void EvalSpeedupCUDA::Initialize_EvalSpeedup_Mode_FM(mcu_VEC_VC(cuReal3)& M_cuVEC_VC, mcu_VEC(cuReal3)& H_cuVEC)
{
	eval_speedup_mode = EVALSPEEDUP_MODE_FM;

	pM_cuVEC_VC = &M_cuVEC_VC;
	pH_cuVEC = &H_cuVEC;

	//make everything else nullptr, zero size
	pM_cuVEC = nullptr;
	pM2_cuVEC_VC = nullptr;
	pH2_cuVEC = nullptr;
}

void EvalSpeedupCUDA::Initialize_EvalSpeedup_Mode_AFM(
	mcu_VEC_VC(cuReal3)& M_cuVEC_VC, mcu_VEC_VC(cuReal3)& M2_cuVEC_VC,
	mcu_VEC(cuReal3)& H_cuVEC, mcu_VEC(cuReal3)& H2_cuVEC)
{
	eval_speedup_mode = EVALSPEEDUP_MODE_AFM;

	pM_cuVEC_VC = &M_cuVEC_VC;
	pM2_cuVEC_VC = &M2_cuVEC_VC;

	pH_cuVEC = &H_cuVEC;
	pH2_cuVEC = &H2_cuVEC;

	//make everything else nullptr, zero size
	pM_cuVEC = nullptr;
}

void EvalSpeedupCUDA::UpdateConfiguration_EvalSpeedup(void)
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
bool EvalSpeedupCUDA::Check_if_EvalSpeedup(int eval_speedup_factor, bool check_step_update)
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

void EvalSpeedupCUDA::UpdateField_EvalSpeedup(
	int eval_speedup_factor, bool check_step_update,
	double eval_step_time,
	std::function<void(mcu_VEC(cuReal3)&)>& do_evaluation,
	std::function<void(void)>& do_transfer_in, std::function<void(mcu_VEC(cuReal3)&)>& do_transfer_out)
{
	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////// EVAL SPEEDUP /////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	//update if required by ODE solver or if we don't have enough previous evaluations saved to extrapolate
	if (check_step_update || num_Hdemag_saved < eval_speedup_factor) {

		mcu_VEC(cuReal3)* pHdemag;

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
			if (eval_speedup_factor > 1) EvalSpeedup_AddField_SubSelf_FM(*pHdemag);
			else EvalSpeedup_AddField_FM();
			break;

		case EVALSPEEDUP_MODE_AFM:
			if (eval_speedup_factor > 1) EvalSpeedup_AddField_SubSelf_AFM(*pHdemag);
			else EvalSpeedup_AddField_AFM();
			break;

		case EVALSPEEDUP_MODE_ATOM:
			//transfer field to atomistic mesh, then subtract self contribution from evaluation for later use
			do_transfer_out(*pHdemag);
			if (eval_speedup_factor > 1) EvalSpeedup_SubSelf(*pHdemag);
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

		cuBReal a1 = 1.0, a2 = 0.0, a3 = 0.0, a4 = 0.0, a5 = 0.0, a6 = 0.0;
		cuBReal time = eval_step_time;

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
				EvalSpeedup_AddExtrapField_AddSelf_FM(a1, a2, a3, a4, a5, a6);
				break;

			case EVALSPEEDUP_MODE_AFM:
				EvalSpeedup_AddExtrapField_AddSelf_AFM(a1, a2, a3, a4, a5, a6);
				break;

			case EVALSPEEDUP_MODE_ATOM:
				//construct effective field approximation
				EvalSpeedup_SetExtrapField_AddSelf(a1, a2, a3, a4, a5, a6);
				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				do_transfer_out(*pH_cuVEC);
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
				EvalSpeedup_AddExtrapField_AddSelf_FM(a1, a2, a3, a4, a5);
				break;

			case EVALSPEEDUP_MODE_AFM:
				EvalSpeedup_AddExtrapField_AddSelf_AFM(a1, a2, a3, a4, a5);
				break;

			case EVALSPEEDUP_MODE_ATOM:
				//construct effective field approximation
				EvalSpeedup_SetExtrapField_AddSelf(a1, a2, a3, a4, a5);
				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				do_transfer_out(*pH_cuVEC);
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
				EvalSpeedup_AddExtrapField_AddSelf_FM(a1, a2, a3, a4);
				break;

			case EVALSPEEDUP_MODE_AFM:
				EvalSpeedup_AddExtrapField_AddSelf_AFM(a1, a2, a3, a4);
				break;

			case EVALSPEEDUP_MODE_ATOM:
				//construct effective field approximation
				EvalSpeedup_SetExtrapField_AddSelf(a1, a2, a3, a4);
				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				do_transfer_out(*pH_cuVEC);
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
				EvalSpeedup_AddExtrapField_AddSelf_FM(a1, a2, a3);
				break;

			case EVALSPEEDUP_MODE_AFM:
				EvalSpeedup_AddExtrapField_AddSelf_AFM(a1, a2, a3);
				break;

			case EVALSPEEDUP_MODE_ATOM:
				//construct effective field approximation
				EvalSpeedup_SetExtrapField_AddSelf(a1, a2, a3);
				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				do_transfer_out(*pH_cuVEC);
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
				EvalSpeedup_AddExtrapField_AddSelf_FM(a1, a2);
				break;

			case EVALSPEEDUP_MODE_AFM:
				EvalSpeedup_AddExtrapField_AddSelf_AFM(a1, a2);
				break;

			case EVALSPEEDUP_MODE_ATOM:
				//construct effective field approximation
				EvalSpeedup_SetExtrapField_AddSelf(a1, a2);
				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				do_transfer_out(*pH_cuVEC);
				break;
			};
		}
		break;

		case EVALSPEEDUP_STEP:
		{
			// no need to add self demag for step method
			switch (eval_speedup_mode) {

			case EVALSPEEDUP_MODE_FM:
				EvalSpeedup_AddField_FM();
				break;

			case EVALSPEEDUP_MODE_AFM:
				EvalSpeedup_AddField_AFM();
				break;

			case EVALSPEEDUP_MODE_ATOM:
				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				do_transfer_out(Hdemag);
				break;
			};
		}
		break;
		}
	}
}

mcu_VEC(cuReal3)* EvalSpeedupCUDA::UpdateField_EvalSpeedup_MConv_Start(int eval_speedup_factor, bool check_step_update, double eval_step_time)
{
	mcu_VEC(cuReal3)* pHdemag = nullptr;

	if (check_step_update || num_Hdemag_saved < eval_speedup_factor) {

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
	}

	return pHdemag;
}

void EvalSpeedupCUDA::UpdateField_EvalSpeedup_MConv_Finish(int eval_speedup_factor, bool do_transfer, mcu_VEC(cuReal3)* pHdemag, mcu_VEC(cuReal3)& transfer)
{
	//subtract self demag from *pHDemag (unless it's the step method)
	switch (eval_speedup_mode) {

	case EVALSPEEDUP_MODE_FM:
		//copy H to mesh Heff before subtracting self
		if (!do_transfer) {

			if (eval_speedup_factor > 1) EvalSpeedup_AddField_SubSelf_FM(*pHdemag);
			else EvalSpeedup_AddField_FM();
		}
		//just subtract self since transfer to mesh Heff has already been done
		else EvalSpeedup_SubSelf(*pHdemag, transfer);
		break;

	case EVALSPEEDUP_MODE_AFM:
		if (!do_transfer) {

			if (eval_speedup_factor > 1) EvalSpeedup_AddField_SubSelf_AFM(*pHdemag);
			else EvalSpeedup_AddField_AFM();
		}
		else EvalSpeedup_SubSelf(*pHdemag, transfer);
		break;

	case EVALSPEEDUP_MODE_ATOM:
		//subtract self contribution from evaluation for later use (for atomistic meshes do_transfer is forced, so atomistic mesh Heff already has the right contribution)
		if (eval_speedup_factor > 1) EvalSpeedup_SubSelf(*pHdemag, transfer);
		break;
	};
}

void EvalSpeedupCUDA::UpdateField_EvalSpeedup_MConv_Extrap(
	int eval_speedup_factor, double eval_step_time,
	mcu_VEC(cuReal3)* ptransfer)
{
	//not required to update, and we have enough previous evaluations: use previous Hdemag saves to extrapolate for current evaluation

	if (ptransfer) {

		switch (eval_speedup_mode) {

		case EVALSPEEDUP_MODE_FM:
			ptransfer->transfer_in();
			break;

		case EVALSPEEDUP_MODE_AFM:
			ptransfer->transfer_in_averaged();
			break;

		case EVALSPEEDUP_MODE_ATOM:
			ptransfer->transfer_in();
			break;
		};
	}

	cuBReal a1 = 1.0, a2 = 0.0, a3 = 0.0, a4 = 0.0, a5 = 0.0, a6 = 0.0;
	cuBReal time = eval_step_time;

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
			//no transfer required : build extrapolation field with self contribution directly using Heff and M
			if (!ptransfer) EvalSpeedup_AddExtrapField_AddSelf_FM(a1, a2, a3, a4, a5, a6);
			else {

				//build extrapolation field into *ptransfer, given that *ptransfer initially contains self contribution
				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3, a4, a5, a6);
				ptransfer->transfer_out();
			}
			break;

		case EVALSPEEDUP_MODE_AFM:
			if (!ptransfer) EvalSpeedup_AddExtrapField_AddSelf_AFM(a1, a2, a3, a4, a5, a6);
			else {

				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3, a4, a5, a6);
				ptransfer->transfer_out_duplicated();
			}
			break;

		case EVALSPEEDUP_MODE_ATOM:
			//transfer mode is forced for atomistic meshes
			if (ptransfer) {

				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3, a4, a5, a6);
				ptransfer->transfer_out();
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
			//no transfer required : build extrapolation field with self contribution directly using Heff and M
			if (!ptransfer) EvalSpeedup_AddExtrapField_AddSelf_FM(a1, a2, a3, a4, a5);
			else {

				//build extrapolation field into *ptransfer, given that *ptransfer initially contains self contribution
				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3, a4, a5);
				ptransfer->transfer_out();
			}
			break;

		case EVALSPEEDUP_MODE_AFM:
			if (!ptransfer) EvalSpeedup_AddExtrapField_AddSelf_AFM(a1, a2, a3, a4, a5);
			else {

				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3, a4, a5);
				ptransfer->transfer_out_duplicated();
			}
			break;

		case EVALSPEEDUP_MODE_ATOM:
			//transfer mode is forced for atomistic meshes
			if (ptransfer) {

				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3, a4, a5);
				ptransfer->transfer_out();
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
			//no transfer required : build extrapolation field with self contribution directly using Heff and M
			if (!ptransfer) EvalSpeedup_AddExtrapField_AddSelf_FM(a1, a2, a3, a4);
			else {

				//build extrapolation field into *ptransfer, given that *ptransfer initially contains self contribution
				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3, a4);
				ptransfer->transfer_out();
			}
			break;

		case EVALSPEEDUP_MODE_AFM:
			if (!ptransfer) EvalSpeedup_AddExtrapField_AddSelf_AFM(a1, a2, a3, a4);
			else {

				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3, a4);
				ptransfer->transfer_out_duplicated();
			}
			break;

		case EVALSPEEDUP_MODE_ATOM:
			//transfer mode is forced for atomistic meshes
			if (ptransfer) {

				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3, a4);
				ptransfer->transfer_out();
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
			//no transfer required : build extrapolation field with self contribution directly using Heff and M
			if (!ptransfer) EvalSpeedup_AddExtrapField_AddSelf_FM(a1, a2, a3);
			else {

				//build extrapolation field into *ptransfer, given that *ptransfer initially contains self contribution
				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3);
				ptransfer->transfer_out();
			}
			break;

		case EVALSPEEDUP_MODE_AFM:
			if (!ptransfer) EvalSpeedup_AddExtrapField_AddSelf_AFM(a1, a2, a3);
			else {

				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3);
				ptransfer->transfer_out_duplicated();
			}
			break;

		case EVALSPEEDUP_MODE_ATOM:
			//transfer mode is forced for atomistic meshes
			if (ptransfer) {

				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2, a3);
				ptransfer->transfer_out();
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
			//no transfer required : build extrapolation field with self contribution directly using Heff and M
			if (!ptransfer) EvalSpeedup_AddExtrapField_AddSelf_FM(a1, a2);
			else {

				//build extrapolation field into *ptransfer, given that *ptransfer initially contains self contribution
				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2);
				ptransfer->transfer_out();
			}
			break;

		case EVALSPEEDUP_MODE_AFM:
			if (!ptransfer) EvalSpeedup_AddExtrapField_AddSelf_AFM(a1, a2);
			else {

				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2);
				ptransfer->transfer_out_duplicated();
			}
			break;

		case EVALSPEEDUP_MODE_ATOM:
			//transfer mode is forced for atomistic meshes
			if (ptransfer) {

				EvalSpeedup_SetExtrapField_AddSelf_MConv(ptransfer, a1, a2);
				ptransfer->transfer_out();
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
			if (!ptransfer) EvalSpeedup_AddField_FM();
			else Hdemag.transfer_out();
			break;

		case EVALSPEEDUP_MODE_AFM:
			if (!ptransfer) EvalSpeedup_AddField_AFM();
			else Hdemag.transfer_out_duplicated();
			break;

		case EVALSPEEDUP_MODE_ATOM:
			//transfer mode is forced for atomistic meshes
			Hdemag.transfer_out();
			break;
		};
	}
	break;
	}
}

#endif

#endif
