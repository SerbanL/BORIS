#include "stdafx.h"
#include "DiffEqFMCUDA.h"

#if COMPILECUDA == 1
#ifdef MESH_COMPILATION_FERROMAGNETIC

#include "DiffEqFM.h"

#include "Mesh_Ferromagnetic.h"
#include "Mesh_FerromagneticCUDA.h"

DifferentialEquationFMCUDA::DifferentialEquationFMCUDA(DifferentialEquation *pmeshODE) :
	DifferentialEquationCUDA(pmeshODE),
	cuDiffEq(mGPU)
{
	error_on_create = AllocateMemory(true);

	cuDiffEq.set_pointers(this);
	SetODEMethodPointers();
}

DifferentialEquationFMCUDA::~DifferentialEquationFMCUDA()
{
	//If called_from_destructor is true then do not attempt to transfer data to cpu where this is held in the derived class of DifferentialEquation
	//This derived class will have already destructed so attempting to copy over data to it is an invalid action and can crash the program.
	//This can happen when a mesh is deleted with CUDA switched on
	//It's also possible this destructor was called simply due to switching CUDA off, in which case called_from_destructor should be false
	CleanupMemory(!pmeshODE->called_from_destructor);
}

//---------------------------------------- SET-UP METHODS  : DiffEqCUDA.cpp and DiffEqCUDA.cu

//allocate memory depending on set evaluation method - also cleans up previously allocated memory by calling CleanupMemory();
BError DifferentialEquationFMCUDA::AllocateMemory(bool copy_from_cpu)
{
	BError error(CLASS_STR(DifferentialEquationFMCUDA));

	//first make sure everything not needed is cleared
	CleanupMemory();
	
	if (!sM1.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
	//copy values from cpu : it's possible the user switches to CUDA during a simulation and expects it to continue as normal
	//some evaluation methods need saved data (sM1, sEval... etc.) to carry over.
	else if (copy_from_cpu) sM1.copy_from_cpuvec(pmeshODE->sM1);

	switch (pmeshODE->evalMethod) {

	case EVAL_EULER:
		break;

	case EVAL_AHEUN:
	case EVAL_TEULER:
		break;

	case EVAL_RK4:
		if (!sEval0.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval0.copy_from_cpuvec(pmeshODE->sEval0);

		if (!sEval1.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval1.copy_from_cpuvec(pmeshODE->sEval1);

		if (!sEval2.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval2.copy_from_cpuvec(pmeshODE->sEval2);
		break;

	case EVAL_ABM:
		if (!sEval0.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval0.copy_from_cpuvec(pmeshODE->sEval0);

		if (!sEval1.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval1.copy_from_cpuvec(pmeshODE->sEval1);
		break;

	case EVAL_RK23:
		if (!sEval0.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval0.copy_from_cpuvec(pmeshODE->sEval0);

		if (!sEval1.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval1.copy_from_cpuvec(pmeshODE->sEval1);

		if (!sEval2.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval2.copy_from_cpuvec(pmeshODE->sEval2);
		break;

	case EVAL_RKF45:
		if (!sEval0.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if(copy_from_cpu) sEval0.copy_from_cpuvec(pmeshODE->sEval0);

		if (!sEval1.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval1.copy_from_cpuvec(pmeshODE->sEval1);

		if (!sEval2.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval2.copy_from_cpuvec(pmeshODE->sEval2);

		if (!sEval3.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval3.copy_from_cpuvec(pmeshODE->sEval3);

		if (!sEval4.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval4.copy_from_cpuvec(pmeshODE->sEval4);
		break;

	case EVAL_RKCK45:
		if (!sEval0.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval0.copy_from_cpuvec(pmeshODE->sEval0);

		if (!sEval1.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval1.copy_from_cpuvec(pmeshODE->sEval1);

		if (!sEval2.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval2.copy_from_cpuvec(pmeshODE->sEval2);

		if (!sEval3.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval3.copy_from_cpuvec(pmeshODE->sEval3);

		if (!sEval4.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval4.copy_from_cpuvec(pmeshODE->sEval4);
		break;

	case EVAL_RKDP54:
		if (!sEval0.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval0.copy_from_cpuvec(pmeshODE->sEval0);

		if (!sEval1.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval1.copy_from_cpuvec(pmeshODE->sEval1);

		if (!sEval2.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval2.copy_from_cpuvec(pmeshODE->sEval2);

		if (!sEval3.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval3.copy_from_cpuvec(pmeshODE->sEval3);

		if (!sEval4.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval4.copy_from_cpuvec(pmeshODE->sEval4);

		if (!sEval5.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval5.copy_from_cpuvec(pmeshODE->sEval5);
		break;

	case EVAL_RKF56:
		if (!sEval0.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval0.copy_from_cpuvec(pmeshODE->sEval0);

		if (!sEval1.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval1.copy_from_cpuvec(pmeshODE->sEval1);

		if (!sEval2.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval2.copy_from_cpuvec(pmeshODE->sEval2);

		if (!sEval3.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval3.copy_from_cpuvec(pmeshODE->sEval3);

		if (!sEval4.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval4.copy_from_cpuvec(pmeshODE->sEval4);

		if (!sEval5.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval5.copy_from_cpuvec(pmeshODE->sEval5);

		if (!sEval6.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval6.copy_from_cpuvec(pmeshODE->sEval6);
		break;

	case EVAL_SD:
		if (!sEval0.resize((cuSZ3)pMesh->n)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) sEval0.copy_from_cpuvec(pmeshODE->sEval0);
		break;
	}

	//For stochastic equations must also allocate memory for thermal VECs

	bool prng_used = false;

	switch (pmeshODE->setODE) {

	case ODE_SLLG:
	case ODE_SLLGSTT:
	case ODE_SLLGSA:
		if (!H_Thermal.resize((cuReal3)pMesh->h_s, (cuRect)pMesh->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) H_Thermal.copy_from_cpuvec(pmeshODE->H_Thermal);
		prng_used = true;
		break;

	case ODE_SLLB:
	case ODE_SLLBSTT:
	case ODE_SLLBSA:
		if (!H_Thermal.resize((cuReal3)pMesh->h_s, (cuRect)pMesh->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) H_Thermal.copy_from_cpuvec(pmeshODE->H_Thermal);

		if (!Torque_Thermal.resize((cuReal3)pMesh->h_s, (cuRect)pMesh->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (copy_from_cpu) Torque_Thermal.copy_from_cpuvec(pmeshODE->Torque_Thermal);
		prng_used = true;
		break;
	}

	if (prng_used) {

		//initialize the pseudo-random number generator with a seed and memory size - recommended use kernel size divided by 128
		if (prng.initialize((pMesh->prng_seed == 0 ? GetSystemTickCount() : pMesh->prng_seed), pMesh->n_s.dim(), (pMesh->prng_seed == 0 ? 0 : 1)) != cudaSuccess) error(BERROR_OUTOFGPUMEMORY_NCRIT);
	}

	return error;
}

//deallocate memory before re-allocating it (depending on evaluation method previously allocated memory might not be used again, so need clean-up before)
void DifferentialEquationFMCUDA::CleanupMemory(bool copy_to_cpu)
{
	//copy values to cpu before erasing : it's possible the user switches CUDA off during a simulation and expects it to continue as normal
	//some evaluation methods need saved data (sM1, sEval... etc.) to carry over.
	//CleanupMemory will be called by destructor in this case, so before destroying gpu data copy it over to cpu if possible
	//CleanupMemory may also be called in other circumstances, in particular from the cpu version of CleanupMemory, after having cleaned cpu vecs, thus in this case the copy methods will not run.

	//Only clear vectors not used for current evaluation method
	if (copy_to_cpu && sM1.size_cpu() == pmeshODE->sM1.size()) sM1.copy_to_cpuvec(pmeshODE->sM1);
	sM1.clear();

	if (
		pmeshODE->evalMethod != EVAL_RK4 &&
		pmeshODE->evalMethod != EVAL_ABM &&
		pmeshODE->evalMethod != EVAL_RK23 &&
		pmeshODE->evalMethod != EVAL_RKF45 &&
		pmeshODE->evalMethod != EVAL_RKCK45 &&
		pmeshODE->evalMethod != EVAL_RKDP54 &&
		pmeshODE->evalMethod != EVAL_RKF56 &&
		pmeshODE->evalMethod != EVAL_SD) {

		if (copy_to_cpu && sEval0.size_cpu() == pmeshODE->sEval0.size()) sEval0.copy_to_cpuvec(pmeshODE->sEval0);
		sEval0.clear();
	}

	if (pmeshODE->evalMethod != EVAL_RK4 &&
		pmeshODE->evalMethod != EVAL_ABM &&
		pmeshODE->evalMethod != EVAL_RK23 &&
		pmeshODE->evalMethod != EVAL_RKF45 &&
		pmeshODE->evalMethod != EVAL_RKCK45 &&
		pmeshODE->evalMethod != EVAL_RKF56 &&
		pmeshODE->evalMethod != EVAL_RKDP54) {

		if (copy_to_cpu && sEval1.size_cpu() == pmeshODE->sEval1.size()) sEval1.copy_to_cpuvec(pmeshODE->sEval1);
		sEval1.clear();
	}

	if (pmeshODE->evalMethod != EVAL_RK4 &&
		pmeshODE->evalMethod != EVAL_RK23 &&
		pmeshODE->evalMethod != EVAL_RKF45 &&
		pmeshODE->evalMethod != EVAL_RKCK45 &&
		pmeshODE->evalMethod != EVAL_RKDP54 &&
		pmeshODE->evalMethod != EVAL_RKF56) {

		if (copy_to_cpu && sEval2.size_cpu() == pmeshODE->sEval2.size()) sEval2.copy_to_cpuvec(pmeshODE->sEval2);
		sEval2.clear();
	}

	if (pmeshODE->evalMethod != EVAL_RK4 &&
		pmeshODE->evalMethod != EVAL_RKF45 &&
		pmeshODE->evalMethod != EVAL_RKCK45 &&
		pmeshODE->evalMethod != EVAL_RKDP54 &&
		pmeshODE->evalMethod != EVAL_RKF56) {

		if (copy_to_cpu && sEval3.size_cpu() == pmeshODE->sEval3.size()) sEval3.copy_to_cpuvec(pmeshODE->sEval3);
		sEval3.clear();

		if (copy_to_cpu && sEval4.size_cpu() == pmeshODE->sEval4.size()) sEval4.copy_to_cpuvec(pmeshODE->sEval4);
		sEval4.clear();
	}

	if (pmeshODE->evalMethod != EVAL_RKDP54 &&
		pmeshODE->evalMethod != EVAL_RKF56) {

		if (copy_to_cpu && sEval5.size_cpu() == pmeshODE->sEval5.size()) sEval5.copy_to_cpuvec(pmeshODE->sEval5);
		sEval5.clear();
	}

	if (pmeshODE->evalMethod != EVAL_RKF56) {

		if (copy_to_cpu && sEval6.size_cpu() == pmeshODE->sEval6.size()) sEval6.copy_to_cpuvec(pmeshODE->sEval6);
		sEval6.clear();
	}

	//For thermal vecs only clear if not used for current set ODE
	if (pmeshODE->setODE != ODE_SLLG &&
		pmeshODE->setODE != ODE_SLLGSTT &&
		pmeshODE->setODE != ODE_SLLB &&
		pmeshODE->setODE != ODE_SLLBSTT &&
		pmeshODE->setODE != ODE_SLLGSA &&
		pmeshODE->setODE != ODE_SLLBSA) {

		if (copy_to_cpu && H_Thermal.size_cpu() == pmeshODE->H_Thermal.size()) H_Thermal.copy_to_cpuvec(pmeshODE->H_Thermal);
		H_Thermal.clear();
	}

	if (pmeshODE->setODE != ODE_SLLB &&
		pmeshODE->setODE != ODE_SLLBSTT &&
		pmeshODE->setODE != ODE_SLLBSA) {

		if (copy_to_cpu && Torque_Thermal.size_cpu() == pmeshODE->Torque_Thermal.size()) Torque_Thermal.copy_to_cpuvec(pmeshODE->Torque_Thermal);
		Torque_Thermal.clear();
	}

	prng.clear();
}


BError DifferentialEquationFMCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(DifferentialEquationFMCUDA));

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHDELETED)) {

		//if a mesh is deleted then a DifferentialEquationCUDA object can be deleted
		//this results in deletion of static data in ODECommonCUDA
		//whilst the static data is remade by UpdateConfiguration in ODECommonCUDA following this, our ManagedDiffEqFMCUDA object now has pointers which are not linked correctly, so need to update them
		cuDiffEq.set_pointers(this);
	}

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHCHANGE, UPDATECONFIG_ODE_SOLVER)) {

		error = AllocateMemory();
	}

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_ODE_MOVEMESH)) {

		/*
		//REMOVED : not really necessary, especially if you have ends coupled to dipoles, which freeze end cells only
		if (!error) {

			//set skip cells flags for moving mesh if enabled
			if (pmeshODE->moving_mesh) {

				Rect mesh_rect = pMesh->GetMeshRect();

				DBL3 end_size = mesh_rect.size() & DBL3(MOVEMESH_ENDRATIO, 1, 1);

				Rect end_rect_left = Rect(mesh_rect.s, mesh_rect.s + end_size);
				Rect end_rect_right = Rect(mesh_rect.e - end_size, mesh_rect.e);

				pMeshCUDA->M()->set_skipcells((cuRect)end_rect_left);
				pMeshCUDA->M()->set_skipcells((cuRect)end_rect_right);
			}
			else {

				pMeshCUDA->M()->clear_skipcells();
			}
		}
		*/
	}

	if (cfgMessage == UPDATECONFIG_PARAMVALUECHANGED_MLENGTH) RenormalizeMagnetization();

	return error;
}

#endif
#endif