#include "stdafx.h"
#include "MeshBaseCUDA.h"
#include "MeshBase.h"
#include "SuperMesh.h"
#include "BorisLib.h"

#if COMPILECUDA == 1

MeshBaseCUDA::MeshBaseCUDA(MeshBase* pMeshBase_) :
	meshRect(pMeshBase_->meshRect),
	n(pMeshBase_->n), h(pMeshBase_->h),
	n_e(pMeshBase_->n_e), h_e(pMeshBase_->h_e),
	n_t(pMeshBase_->n_t), h_t(pMeshBase_->h_t),
	n_m(pMeshBase_->n_m), h_m(pMeshBase_->h_m),
	aux_real(mGPU), aux_real3(mGPU), aux_int(mGPU),
	profile_storage_sca(mGPU), profile_storage_vec(mGPU),
	auxVEC_cuReal3(mGPU), auxVEC_cuBReal(mGPU),
	prng(mGPU, GetSystemTickCount(), pMeshBase_->n.dim() / (128 * mGPU.get_num_devices())),
	mc_acceptance_rate(mGPU), cmc_n(mGPU), cmc_M(mGPU),
	V(mGPU), elC(mGPU), E(mGPU), S(mGPU),
	Temp(mGPU), Temp_l(mGPU),
	u_disp(mGPU), strain_diag(mGPU), strain_odiag(mGPU)
{
	pMeshBase = pMeshBase_;
}

MeshBaseCUDA::~MeshBaseCUDA()
{
	//copy any extracted profiles to cpu versions (e.g. in case an averaged profile was being extracted, then cuda 0 used before reading the profile out)
	if (profile_storage_sca.size()) {

		pMeshBase->profile_storage_dbl.resize(profile_storage_sca.size());
		profile_storage_sca.copy_to_vector(pMeshBase->profile_storage_dbl);
	}

	if (profile_storage_vec.size()) {

		pMeshBase->profile_storage_dbl3.resize(profile_storage_vec.size());
		profile_storage_vec.copy_to_vector(pMeshBase->profile_storage_dbl3);
	}
}

//----------------------------------- MESH INFO GET/SET METHODS

int MeshBaseCUDA::GetMeshType(void)
{
	return (int)pMeshBase->GetMeshType();
}

//search save data list (saveDataList) for given dataID set for this mesh. Return true if found and its rectangle is not Null; else return false.
bool MeshBaseCUDA::IsOutputDataSet_withRect(int datumId)
{
	return pMeshBase->IsOutputDataSet_withRect(datumId);
}

//return true if data is set (with any rectangle)
bool MeshBaseCUDA::IsOutputDataSet(int datumId)
{
	return pMeshBase->IsOutputDataSet(datumId);
}

//check if given stage is set
bool MeshBaseCUDA::IsStageSet(int stageType)
{
	return pMeshBase->IsStageSet(stageType);
}

//set computefields_if_MC flag on SuperMesh
void MeshBaseCUDA::Set_Force_MonteCarlo_ComputeFields(bool status)
{
	pMeshBase->Set_Force_MonteCarlo_ComputeFields(status);
}

//others

bool MeshBaseCUDA::is_atomistic(void)
{
	return pMeshBase->is_atomistic();
}

//----------------------------------- VALUE GETTERS

//check if the ODECommon::available flag is true (ode step solved)
bool MeshBaseCUDA::CurrentTimeStepSolved(void)
{
	return pMeshBase->pSMesh->CurrentTimeStepSolved();
}

//check evaluation speedup flag in ODECommon
int MeshBaseCUDA::GetEvaluationSpeedup(void)
{
	return pMeshBase->pSMesh->GetEvaluationSpeedup();
}

//check in ODECommon the type of field update we need to do depending on the ODE evaluation step
bool MeshBaseCUDA::Check_Step_Update(void)
{
	return pMeshBase->pSMesh->Check_Step_Update();
}

//get total time with evaluation step resolution level
cuBReal MeshBaseCUDA::Get_EvalStep_Time(void)
{
	return pMeshBase->pSMesh->Get_EvalStep_Time();
}

cuBReal MeshBaseCUDA::GetStageTime(void)
{
	return pMeshBase->pSMesh->GetStageTime();
}

int MeshBaseCUDA::GetStageStep(void)
{
	return pMeshBase->pSMesh->stage_step.minor;
}

cuBReal MeshBaseCUDA::GetTimeStep(void)
{
	return pMeshBase->pSMesh->GetTimeStep();
}

//----------------------------------- DISPLAY-ASSOCIATED GET/SET METHODS

//Get settings for module display data 
//Return module displaying its effective field (MOD_ALL means total Heff)
int MeshBaseCUDA::Get_Module_Heff_Display(void)
{
	return pMeshBase->Get_Module_Heff_Display();
}

int MeshBaseCUDA::Get_ActualModule_Heff_Display(void)
{
	return pMeshBase->Get_ActualModule_Heff_Display();
}

//Return module displaying its energy density spatial variation (MOD_ERROR means none set)
int MeshBaseCUDA::Get_Module_Energy_Display(void)
{
	return pMeshBase->Get_Module_Energy_Display();
}

int MeshBaseCUDA::Get_ActualModule_Energy_Display(void)
{
	return pMeshBase->Get_ActualModule_Energy_Display();
}

#endif