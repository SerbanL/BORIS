#include "stdafx.h"
#include "Atom_DiffEq_CommonCUDA.h"
#include "Atom_DiffEq_Common.h"

#if COMPILECUDA == 1

//-----------------------------------Managed DiffEq and CPU version pointer

mcu_obj<ManagedAtom_DiffEq_CommonCUDA, ManagedAtom_DiffEqPolicy_CommonCUDA>* Atom_ODECommonCUDA::pcuaDiffEq = nullptr;

Atom_ODECommon* Atom_ODECommonCUDA::pODE = nullptr;

//-----------------------------------Equation

mcu_val<int>* Atom_ODECommonCUDA::psetODE = nullptr;

//-----------------------------------Evaluation method modifiers

mcu_val<bool>* Atom_ODECommonCUDA::prenormalize = nullptr;

//-----------------------------------Steepest Descent Solver

mcu_val<cuBReal>* Atom_ODECommonCUDA::pdelta_M_sq = nullptr;
mcu_val<cuBReal>* Atom_ODECommonCUDA::pdelta_G_sq = nullptr;
mcu_val<cuBReal>* Atom_ODECommonCUDA::pdelta_M_dot_delta_G = nullptr;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Atom_ODECommonCUDA::Atom_ODECommonCUDA(Atom_ODECommon *pODE_) :
	ODECommon_BaseCUDA(pODE_)
{
	if (!pODE) pODE = pODE_;

	AllocateStaticData();
}

Atom_ODECommonCUDA::~Atom_ODECommonCUDA()
{
	if (pODE) {

		//delete these to prevent memory leaks
		//Note since all these are static, they will be deleted in all meshes holding a ODECommonCUDA
		//Thus if you delete a mesh, you'll need to remake these and also set pointers in cuDiffEq : do it in UpdateConfiguration
		
		if (pcuaDiffEq) { delete pcuaDiffEq; pcuaDiffEq = nullptr; }

		if (prenormalize) { delete prenormalize; prenormalize = nullptr; }

		if (psetODE) { delete psetODE; psetODE = nullptr; }

		if (pdelta_M_sq) { delete pdelta_M_sq; pdelta_M_sq = nullptr; }
		if (pdelta_G_sq) { delete pdelta_G_sq; pdelta_G_sq = nullptr; }
		if (pdelta_M_dot_delta_G) { delete pdelta_M_dot_delta_G; pdelta_M_dot_delta_G = nullptr; }
	}
}

//Allocate memory for all static data; deletion only happens in the destructor, however allocation can also be triggered by UpdateConfiguration since the static data can be deleted by another instance which inherits same static data
void Atom_ODECommonCUDA::AllocateStaticData(void)
{
	ODECommon_BaseCUDA::AllocateStaticData();

	if (!prenormalize) prenormalize = new mcu_val<bool>(mGPU);

	if (!psetODE) psetODE = new mcu_val<int>(mGPU);

	if (!pdelta_M_sq) pdelta_M_sq = new mcu_val<cuBReal>(mGPU);
	if (!pdelta_G_sq) pdelta_G_sq = new mcu_val<cuBReal>(mGPU);
	if (!pdelta_M_dot_delta_G) pdelta_M_dot_delta_G = new mcu_val<cuBReal>(mGPU);

	if (!pcuaDiffEq) pcuaDiffEq = new mcu_obj<ManagedAtom_DiffEq_CommonCUDA, ManagedAtom_DiffEqPolicy_CommonCUDA>(mGPU);

	//setup the managed cuDiffEq object with pointers to all required data for differential equations calculations
	pcuaDiffEq->set_pointers(this);

	SyncODEValues();
}

BError Atom_ODECommonCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(__FUNCTION__);

	//this could have been deleted by a destructor in another instance which inherits from ODECommonCUDA : try to remake it.
	//this will only happen when a mesh is deleted
	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHDELETED)) {

		AllocateStaticData();
	}

	return error;
}

void Atom_ODECommonCUDA::Get_SD_Solver_BB_Values(double* pdelta_M_sq_cpu, double* pdelta_G_sq_cpu, double* pdelta_M_dot_delta_G_cpu)
{
	*pdelta_M_sq_cpu = pdelta_M_sq->to_cpu_sum();
	*pdelta_G_sq_cpu = pdelta_G_sq->to_cpu_sum();
	*pdelta_M_dot_delta_G_cpu = pdelta_M_dot_delta_G->to_cpu_sum();
}

void Atom_ODECommonCUDA::SyncODEValues(void)
{
	ODECommon_BaseCUDA::SyncODEValues();

	prenormalize->from_cpu(pODE->renormalize);

	psetODE->from_cpu(pODE->setODE);
}

#endif