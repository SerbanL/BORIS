#include "stdafx.h"
#include "DiffEq_CommonCUDA.h"
#include "DiffEq_Common.h"

#if COMPILECUDA == 1

//-----------------------------------Managed DiffEq and CPU version pointer

mcu_obj<ManagedDiffEq_CommonCUDA, ManagedDiffEqPolicy_CommonCUDA>* ODECommonCUDA::pcuDiffEq = nullptr;

ODECommon* ODECommonCUDA::pODE = nullptr;
//-----------------------------------Equation

mcu_val<int>* ODECommonCUDA::psetODE = nullptr;

//-----------------------------------Evaluation method modifiers

mcu_val<bool>* ODECommonCUDA::prenormalize = nullptr;

//-----------------------------------Steepest Descent Solver

mcu_val<cuBReal>* ODECommonCUDA::pdelta_M_sq = nullptr;
mcu_val<cuBReal>* ODECommonCUDA::pdelta_G_sq = nullptr;
mcu_val<cuBReal>* ODECommonCUDA::pdelta_M_dot_delta_G = nullptr;
mcu_val<cuBReal>* ODECommonCUDA::pdelta_M2_sq = nullptr;
mcu_val<cuBReal>* ODECommonCUDA::pdelta_G2_sq = nullptr;
mcu_val<cuBReal>* ODECommonCUDA::pdelta_M2_dot_delta_G2 = nullptr;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ODECommonCUDA::ODECommonCUDA(ODECommon *pODE_) :
	ODECommon_BaseCUDA(pODE_)
{
	if (!pODE) pODE = pODE_;

	AllocateStaticData();
}

ODECommonCUDA::~ODECommonCUDA()
{
	if (pODE) {

		//delete these to prevent memory leaks
		//Note since all these are static, they will be deleted in all meshes holding a ODECommonCUDA
		//Thus if you delete a mesh, you'll need to remake these and also set pointers in cuDiffEq : do it in UpdateConfiguration
		
		if (pcuDiffEq) { delete pcuDiffEq; pcuDiffEq = nullptr; }

		if (prenormalize) { delete  prenormalize; prenormalize = nullptr; }

		if (psetODE) { delete psetODE; psetODE = nullptr; }

		if (pdelta_M_sq) { delete pdelta_M_sq; pdelta_M_sq = nullptr; }
		if (pdelta_G_sq) { delete pdelta_G_sq; pdelta_G_sq = nullptr; }
		if (pdelta_M_dot_delta_G) { delete pdelta_M_dot_delta_G; pdelta_M_dot_delta_G = nullptr; }

		if (pdelta_M2_sq) { delete pdelta_M2_sq; pdelta_M2_sq = nullptr; }
		if (pdelta_G2_sq) { delete pdelta_G2_sq; pdelta_G2_sq = nullptr; }
		if (pdelta_M2_dot_delta_G2) { delete pdelta_M2_dot_delta_G2; pdelta_M2_dot_delta_G2 = nullptr; }
	}
}

//Allocate memory for all static data; deletion only happens in the destructor, however allocation can also be triggered by UpdateConfiguration since the static data can be deleted by another instance which inherits same static data
void ODECommonCUDA::AllocateStaticData(void)
{
	ODECommon_BaseCUDA::AllocateStaticData();

	if (!prenormalize) prenormalize = new mcu_val<bool>(mGPU);

	if (!psetODE) psetODE = new mcu_val<int>(mGPU);

	if (!pdelta_M_sq) pdelta_M_sq = new mcu_val<cuBReal>(mGPU);
	if (!pdelta_G_sq) pdelta_G_sq = new mcu_val<cuBReal>(mGPU);
	if (!pdelta_M_dot_delta_G) pdelta_M_dot_delta_G = new mcu_val<cuBReal>(mGPU);

	if (!pdelta_M2_sq) pdelta_M2_sq = new mcu_val<cuBReal>(mGPU);
	if (!pdelta_G2_sq) pdelta_G2_sq = new mcu_val<cuBReal>(mGPU);
	if (!pdelta_M2_dot_delta_G2) pdelta_M2_dot_delta_G2 = new mcu_val<cuBReal>(mGPU);

	if (!pcuDiffEq) pcuDiffEq = new mcu_obj<ManagedDiffEq_CommonCUDA, ManagedDiffEqPolicy_CommonCUDA>(mGPU);

	//setup the managed cuDiffEq object with pointers to all required data for differential equations calculations
	pcuDiffEq->set_pointers(this);

	SyncODEValues();
}

BError ODECommonCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(__FUNCTION__);

	//this could have been deleted by a destructor in another instance which inherits from ODECommonCUDA : try to remake it.
	//this will only happen when a mesh is deleted
	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHDELETED)) {

		AllocateStaticData();
	}
	
	return error;
}

void ODECommonCUDA::Get_SD_Solver_BB_Values(
	double* pdelta_M_sq_cpu, double* pdelta_G_sq_cpu, double* pdelta_M_dot_delta_G_cpu,
	double* pdelta_M2_sq_cpu, double* pdelta_G2_sq_cpu, double* pdelta_M2_dot_delta_G2_cpu)
{
	*pdelta_M_sq_cpu = pdelta_M_sq->to_cpu_sum();
	*pdelta_G_sq_cpu = pdelta_G_sq->to_cpu_sum();
	*pdelta_M_dot_delta_G_cpu = pdelta_M_dot_delta_G->to_cpu_sum();

	*pdelta_M2_sq_cpu = pdelta_M2_sq->to_cpu_sum();
	*pdelta_G2_sq_cpu = pdelta_G2_sq->to_cpu_sum();
	*pdelta_M2_dot_delta_G2_cpu = pdelta_M2_dot_delta_G2->to_cpu_sum();
}

void ODECommonCUDA::SyncODEValues(void)
{
	ODECommon_BaseCUDA::SyncODEValues();

	prenormalize->from_cpu(pODE->renormalize);

	psetODE->from_cpu(pODE->setODE);
}

#endif