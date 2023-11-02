#include "stdafx.h"
#include "DiffEqAFMCUDA.h"

#if COMPILECUDA == 1
#ifdef MESH_COMPILATION_ANTIFERROMAGNETIC

#include "ManagedDiffEqAFMCUDA.h"

BError ManagedDiffEqAFMCUDA::set_pointers(DifferentialEquationAFMCUDA* pDiffEqCUDA, int device_idx)
{
	BError error(__FUNCTION__);

	//Pointers to data in ODECommonCUDA

	if (set_gpu_value(pdT, pDiffEqCUDA->pdT->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pdT_last, pDiffEqCUDA->pdT_last->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	
	if (set_gpu_value(pmxh, pDiffEqCUDA->pmxh->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pmxh_av, pDiffEqCUDA->pmxh_av->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pavpoints, pDiffEqCUDA->pavpoints->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(pdmdt, pDiffEqCUDA->pdmdt->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pdmdt_av, pDiffEqCUDA->pdmdt_av->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pavpoints2, pDiffEqCUDA->pavpoints2->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	
	if (set_gpu_value(plte, pDiffEqCUDA->plte->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	
	if (set_gpu_value(prenormalize, pDiffEqCUDA->prenormalize->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	
	if (set_gpu_value(psolve_spin_current, pDiffEqCUDA->psolve_spin_current->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	
	if (set_gpu_value(psetODE, pDiffEqCUDA->psetODE->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	
	if (set_gpu_value(palternator, pDiffEqCUDA->palternator->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(pdelta_M_sq, pDiffEqCUDA->pdelta_M_sq->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pdelta_G_sq, pDiffEqCUDA->pdelta_G_sq->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pdelta_M_dot_delta_G, pDiffEqCUDA->pdelta_M_dot_delta_G->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(pdelta_M2_sq, pDiffEqCUDA->pdelta_M2_sq->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pdelta_G2_sq, pDiffEqCUDA->pdelta_G2_sq->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pdelta_M2_dot_delta_G2, pDiffEqCUDA->pdelta_M2_dot_delta_G2->get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	//Pointers to data in DifferentialEquationCUDA

	if (set_gpu_value(psM1, pDiffEqCUDA->sM1.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psM1_2, pDiffEqCUDA->sM1_2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(psEval0, pDiffEqCUDA->sEval0.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval1, pDiffEqCUDA->sEval1.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval2, pDiffEqCUDA->sEval2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval3, pDiffEqCUDA->sEval3.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval4, pDiffEqCUDA->sEval4.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval5, pDiffEqCUDA->sEval5.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval6, pDiffEqCUDA->sEval6.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(psEval0_2, pDiffEqCUDA->sEval0_2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval1_2, pDiffEqCUDA->sEval1_2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval2_2, pDiffEqCUDA->sEval2_2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval3_2, pDiffEqCUDA->sEval3_2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval4_2, pDiffEqCUDA->sEval4_2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval5_2, pDiffEqCUDA->sEval5_2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(psEval6_2, pDiffEqCUDA->sEval6_2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(pH_Thermal, pDiffEqCUDA->H_Thermal.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pTorque_Thermal, pDiffEqCUDA->Torque_Thermal.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(pH_Thermal_2, pDiffEqCUDA->H_Thermal_2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pTorque_Thermal_2, pDiffEqCUDA->Torque_Thermal_2.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	
	//Managed cuda mesh pointer so all mesh data can be accessed in device code

	if (set_gpu_value(pcuMesh, pDiffEqCUDA->pMeshCUDA->cuMesh.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	
	return error;
}

#endif
#endif


