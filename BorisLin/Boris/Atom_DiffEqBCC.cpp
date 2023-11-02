#include "stdafx.h"
#include "Atom_DiffEqBCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC

#include "Atom_Mesh_BCC.h"
#include "Atom_MeshParamsControl_Multi.h"

Atom_DifferentialEquationBCC::Atom_DifferentialEquationBCC(Atom_Mesh_BCC*paMesh):
	Atom_DifferentialEquation(paMesh)
{
	error_on_create = UpdateConfiguration(UPDATECONFIG_FORCEUPDATE);
}

Atom_DifferentialEquationBCC::~Atom_DifferentialEquationBCC()
{
}

//---------------------------------------- OTHERS

//Restore magnetization after a failed step for adaptive time-step methods
void Atom_DifferentialEquationBCC::RestoreMoments(void)
{
#pragma omp parallel for
	for (int idx = 0; idx < paMesh->n.dim(); idx++) {

		paMesh->M1[idx] = sM1[idx];
		paMesh->M1(1, idx) = sM1(1, idx);
	}
}

//Save current moments in sM VECs (e.g. useful to reset dM / dt calculation)
void Atom_DifferentialEquationBCC::SaveMoments(void)
{
#if COMPILECUDA == 1
	if (pameshODECUDA) {

		pameshODECUDA->SaveMoments();

		return;
	}
#endif

#pragma omp parallel for
	for (int idx = 0; idx < paMesh->n.dim(); idx++) {

		sM1[idx] = paMesh->M1[idx];
		sM1(1, idx) = paMesh->M1(1, idx);
	}
}

//renormalize vectors to set moment length value (which could have a spatial variation)
void Atom_DifferentialEquationBCC::RenormalizeMoments(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {
#pragma omp parallel for
		for (int idx = 0; idx < paMesh->M1.linear_size(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				double mu_s = paMesh->mu_s(sidx);
				paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s);

				if (mu_s) paMesh->M1(sidx, idx).renormalize(mu_s);
			}
		}
	}
}

//---------------------------------------- SET-UP METHODS

BError Atom_DifferentialEquationBCC::AllocateMemory(void)
{
	BError error(CLASS_STR(Atom_DifferentialEquationBCC));

	//first make sure everything not needed is cleared
	CleanupMemory();

	if (!sM1.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);

	switch (evalMethod) {

	case EVAL_EULER:
		break;

	case EVAL_TEULER:
	case EVAL_AHEUN:
		break;

	case EVAL_ABM:
		if (!sEval0.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval1.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		break;

	case EVAL_RK4:
		if (!sEval0.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval1.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval2.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		break;

	case EVAL_RK23:
		if (!sEval0.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval1.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval2.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		break;

	case EVAL_RKF45:
		if (!sEval0.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval1.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval2.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval3.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval4.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		break;

	case EVAL_RKCK45:
		if (!sEval0.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval1.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval2.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval3.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval4.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		break;

	case EVAL_RKDP54:
		if (!sEval0.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval1.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval2.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval3.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval4.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval5.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		break;

	case EVAL_RKF56:
		if (!sEval0.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval1.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval2.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval3.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval4.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval5.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		if (!sEval6.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		break;

	case EVAL_SD:
		if (!sEval0.resize(paMesh->n)) return error(BERROR_OUTOFMEMORY_CRIT);
		break;
	}

	//For stochastic equations must also allocate memory for thermal VECs

	switch (setODE) {

	case ODE_SLLG:
	case ODE_SLLGSTT:
	case ODE_SLLGSA:
		if (!H_Thermal.resize(paMesh->h, paMesh->meshRect)) return error(BERROR_OUTOFMEMORY_CRIT);
		break;
	}

	//make sure all VECs have 2 sub-lattices with BCC structure
	sM1.set_number_of_sublattices(paMesh->M1.get_sublattice_vectors());

	sEval0.set_number_of_sublattices(paMesh->M1.get_sublattice_vectors());
	sEval1.set_number_of_sublattices(paMesh->M1.get_sublattice_vectors());
	sEval2.set_number_of_sublattices(paMesh->M1.get_sublattice_vectors());
	sEval3.set_number_of_sublattices(paMesh->M1.get_sublattice_vectors());
	sEval4.set_number_of_sublattices(paMesh->M1.get_sublattice_vectors());
	sEval5.set_number_of_sublattices(paMesh->M1.get_sublattice_vectors());
	sEval6.set_number_of_sublattices(paMesh->M1.get_sublattice_vectors());

	H_Thermal.set_number_of_sublattices(paMesh->M1.get_sublattice_vectors());

	//----------------------- CUDA mirroring

#if COMPILECUDA == 1
	if (pameshODECUDA) {

		if (!error) error = pameshODECUDA->AllocateMemory();
	}
#endif

	return error;
}

void Atom_DifferentialEquationBCC::CleanupMemory(void)
{
	//Only clear vectors not used for current evaluation method
	sM1.clear();

	if (evalMethod != EVAL_RK4 &&
		evalMethod != EVAL_ABM &&
		evalMethod != EVAL_RK23 &&
		evalMethod != EVAL_RKF45 &&
		evalMethod != EVAL_RKCK45 &&
		evalMethod != EVAL_RKDP54 &&
		evalMethod != EVAL_RKF56 &&
		evalMethod != EVAL_SD) {

		sEval0.clear();
	}

	if (evalMethod != EVAL_RK4 &&
		evalMethod != EVAL_ABM &&
		evalMethod != EVAL_RK23 &&
		evalMethod != EVAL_RKF45 &&
		evalMethod != EVAL_RKCK45 &&
		evalMethod != EVAL_RKF56 &&
		evalMethod != EVAL_RKDP54) {

		sEval1.clear();
	}

	if (evalMethod != EVAL_RK4 &&
		evalMethod != EVAL_RK23 &&
		evalMethod != EVAL_RKF45 &&
		evalMethod != EVAL_RKCK45 &&
		evalMethod != EVAL_RKDP54 &&
		evalMethod != EVAL_RKF56) {

		sEval2.clear();
	}

	if (evalMethod != EVAL_RK4 &&
		evalMethod != EVAL_RKF45 &&
		evalMethod != EVAL_RKCK45 &&
		evalMethod != EVAL_RKDP54 &&
		evalMethod != EVAL_RKF56) {

		sEval3.clear();
		sEval4.clear();
	}

	if (evalMethod != EVAL_RKDP54 &&
		evalMethod != EVAL_RKF56) {

		sEval5.clear();
	}

	if (evalMethod != EVAL_RKF56) {

		sEval6.clear();
	}

	//For thermal vecs only clear if not used for current set ODE
	if (setODE != ODE_SLLG && 
		setODE != ODE_SLLGSTT &&
		setODE != ODE_SLLGSA) {

		H_Thermal.clear();
	}

	//----------------------- CUDA mirroring

#if COMPILECUDA == 1
	if (pameshODECUDA) { pameshODECUDA->CleanupMemory(); }
#endif
}


BError Atom_DifferentialEquationBCC::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Atom_DifferentialEquationBCC));

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHCHANGE, UPDATECONFIG_ODE_SOLVER)) {

		error = AllocateMemory();
	}

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_ODE_MOVEMESH)) {

	}

	if (cfgMessage == UPDATECONFIG_PARAMVALUECHANGED_MLENGTH) RenormalizeMoments();

	//----------------------- CUDA mirroring

#if COMPILECUDA == 1
	if (pameshODECUDA) {

		if (!error) error = pameshODECUDA->UpdateConfiguration(cfgMessage);
	}
#endif

	return error;
}

//switch CUDA state on/off
BError Atom_DifferentialEquationBCC::SwitchCUDAState(bool cudaState)
{
	BError error(CLASS_STR(Atom_DifferentialEquationBCC));

#if COMPILECUDA == 1

	//are we switching to cuda?
	if (cudaState) {

		if (!pameshODECUDA) {

			pameshODECUDA = new Atom_DifferentialEquationBCCCUDA(this);
			error = pameshODECUDA->Error_On_Create();
		}
	}
	else {

		//cuda switched off so delete cuda module object
		if (pameshODECUDA) delete pameshODECUDA;
		pameshODECUDA = nullptr;
	}

#endif

	return error;
}

//---------------------------------------- GETTERS

//return dM by dT - should only be used when evaluation sequence has ended (TimeStepSolved() == true)
DBL3 Atom_DifferentialEquationBCC::dMdt(int idx)
{
	return (paMesh->M1[idx] - sM1[idx]) / dT_last;
}

#endif
