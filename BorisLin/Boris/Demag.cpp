#include "stdafx.h"
#include "Demag.h"
#include "SuperMesh.h"

#ifdef MODULE_COMPILATION_DEMAG

#include "SimScheduleDefs.h"

#include "Mesh.h"

#if COMPILECUDA == 1
#include "DemagMCUDA.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////

Demag::Demag(Mesh *pMesh_) : 
	Modules(),
	Convolution<Demag, DemagKernel>(pMesh_->GetMeshSize(), pMesh_->GetMeshCellsize()),
	ProgramStateNames(this, {VINFO(demag_pbc_images)}, {}),
	EvalSpeedup()
{
	pMesh = pMesh_;

	Uninitialize();

	error_on_create = Convolution_Error_on_Create();

	//-------------------------- Is CUDA currently enabled?

	//If cuda is enabled we also need to make the cuda module version
	if (pMesh->cudaEnabled) {

		if (!error_on_create) error_on_create = SwitchCUDAState(true);
	}
}

Demag::~Demag()
{
	//when deleting the Demag module any pbc settings should no longer take effect in this mesh
	//thus must clear pbc flags in M

	pMesh->M.set_pbc(0, 0, 0);
	pMesh->M2.set_pbc(0, 0, 0);

	//same for the CUDA version if we are in cuda mode
#if COMPILECUDA == 1
	if (pModuleCUDA) {

		pMesh->pMeshCUDA->M.copyflags_from_cpuvec(pMesh->M);
	}
#endif
}

BError Demag::Initialize(void) 
{	
	BError error(CLASS_STR(Demag));

	if (!initialized) {
		
		error = Calculate_Demag_Kernels();

		if (!error) initialized = true;
	}

	if (pMesh->pSMesh->GetEvaluationSpeedup()) {

		std::function<bool(VEC<DBL3>&)> initialize_mesh_transfer = [&](VEC<DBL3>& H) -> bool {

			//no mesh transfer needed here
			return true;
		};

		EvalSpeedup::Initialize_EvalSpeedup(
			DemagTFunc().SelfDemag_PBC(pMesh->h, pMesh->n, demag_pbc_images),
			pMesh->pSMesh->GetEvaluationSpeedup(),
			pMesh->h, pMesh->meshRect,
			initialize_mesh_transfer);

		if (pMesh->GetMeshType() == MESH_FERROMAGNETIC) {

			EvalSpeedup::Initialize_EvalSpeedup_Mode_FM(pMesh->M, pMesh->Heff);
		}
		else if (pMesh->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

			EvalSpeedup::Initialize_EvalSpeedup_Mode_AFM(pMesh->M, pMesh->M2, pMesh->Heff, pMesh->Heff2);
		}
	}

	EvalSpeedup::num_Hdemag_saved = 0;

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		pMesh->h, pMesh->meshRect, 
		(MOD_)pMesh->Get_Module_Heff_Display() == MOD_DEMAG || pMesh->IsOutputDataSet_withRect(DATA_E_DEMAG) || pMesh->IsStageSet(SS_MONTECARLO),
		(MOD_)pMesh->Get_Module_Energy_Display() == MOD_DEMAG || pMesh->IsOutputDataSet_withRect(DATA_E_DEMAG) || pMesh->IsStageSet(SS_MONTECARLO));
	if (error) initialized = false;

	//if a Monte Carlo stage is set then we need to compute fields
	if (pMesh->IsStageSet(SS_MONTECARLO)) pMesh->Set_Force_MonteCarlo_ComputeFields(true);

	return error;
}

BError Demag::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Demag));

	//only need to uninitialize if n or h have changed, or pbc settings have changed
	if (!CheckDimensions(pMesh->n, pMesh->h, demag_pbc_images) || cfgMessage == UPDATECONFIG_DEMAG_CONVCHANGE || cfgMessage == UPDATECONFIG_MESHCHANGE) {
		
		Uninitialize();

		//Set convolution dimensions for embedded multiplication and required PBC conditions
		error = SetDimensions(pMesh->n, pMesh->h, true, demag_pbc_images);

		EvalSpeedup::UpdateConfiguration_EvalSpeedup();
	}

	//------------------------ CUDA UpdateConfiguration if set

#if COMPILECUDA == 1
	if (pModuleCUDA) {

		if (!error) error = pModuleCUDA->UpdateConfiguration(cfgMessage);
	}
#endif

	return error;
}

//Set PBC
BError Demag::Set_PBC(INT3 demag_pbc_images_)
{
	BError error(__FUNCTION__);

	demag_pbc_images = demag_pbc_images_;

	pMesh->Set_Magnetic_PBC(demag_pbc_images);

	//update will be needed if pbc settings have changed
	error = UpdateConfiguration(UPDATECONFIG_DEMAG_CONVCHANGE);

	return error;
}

BError Demag::MakeCUDAModule(void)
{
	BError error(CLASS_STR(Demag));

#if COMPILECUDA == 1

	if (pMesh->pMeshCUDA) {

		//Note : it is posible pMeshCUDA has not been allocated yet, but this module has been created whilst cuda is switched on. This will happen when a new mesh is being made which adds this module by default.
		//In this case, after the mesh has been fully made, it will call SwitchCUDAState on the mesh, which in turn will call this SwitchCUDAState method; then pMeshCUDA will not be nullptr and we can make the cuda module version
		pModuleCUDA = new DemagMCUDA(pMesh->pMeshCUDA);
		error = pModuleCUDA->Error_On_Create();
	}

#endif

	return error;
}

double Demag::UpdateField(void) 
{
	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////// NO SPEEDUP //////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (!pMesh->pSMesh->GetEvaluationSpeedup() || (num_Hdemag_saved < pMesh->pSMesh->GetEvaluationSpeedup() && !pMesh->pSMesh->Check_Step_Update())) {

		//don't use evaluation speedup, so no need to use Hdemag (this won't have memory allocated anyway) - or else we are using speedup but don't yet have enough previous evaluations at steps where we should be extrapolating

		//convolute and get "energy" value
		if (pMesh->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

			if (Module_Heff.linear_size()) energy = Convolute_AveragedInputs_DuplicatedOutputs(pMesh->M, pMesh->M2, pMesh->Heff, pMesh->Heff2, false, &Module_Heff, &Module_energy);
			else energy = Convolute_AveragedInputs_DuplicatedOutputs(pMesh->M, pMesh->M2, pMesh->Heff, pMesh->Heff2, false);
		}
		else {

			if (Module_Heff.linear_size()) energy = Convolute(pMesh->M, pMesh->Heff, false, &Module_Heff, &Module_energy);
			else energy = Convolute(pMesh->M, pMesh->Heff, false);
		}

		//finish off energy value
		if (pMesh->M.get_nonempty_cells()) energy *= -MU0 / (2 * pMesh->M.get_nonempty_cells());
		else energy = 0;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////// EVAL SPEEDUP /////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////
	
	else {

		std::function<void(VEC<DBL3>&)> do_evaluation = [&](VEC<DBL3>& H) -> void {

			//do evaluation
			if (pMesh->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

				if (Module_Heff.linear_size()) energy = Convolute_AveragedInputs(pMesh->M, pMesh->M2, H, true, &Module_Heff, &Module_energy);
				else energy = Convolute_AveragedInputs(pMesh->M, pMesh->M2, H, true);
			}
			else {

				if (Module_Heff.linear_size()) energy = Convolute(pMesh->M, H, true, &Module_Heff, &Module_energy);
				else energy = Convolute(pMesh->M, H, true);
			}

			//finish off energy value
			if (pMesh->M.get_nonempty_cells()) energy *= -MU0 / (2 * pMesh->M.get_nonempty_cells());
			else energy = 0;
		};

		std::function<void(void)> do_transfer_in = [&](void) -> void {

			//empty, as there's nothing to transfer in
		};

		std::function<void(VEC<DBL3>&)> do_transfer_out = [&](VEC<DBL3>& H) -> void {

			//empty, as there's nothing to transfer out
		};

		EvalSpeedup::UpdateField_EvalSpeedup(
			pMesh->pSMesh->GetEvaluationSpeedup(), pMesh->pSMesh->Check_Step_Update(),
			pMesh->pSMesh->Get_EvalStep_Time(),
			do_evaluation,
			do_transfer_in, do_transfer_out);
	}

	return energy;
}

//-------------------Energy methods

//FM mesh
double Demag::Get_EnergyChange(int spin_index, DBL3 Mnew)
{
	//Module_Heff needs to be calculated (done during a Monte Carlo simulation, where this method would be used)
	if (Module_Heff.linear_size()) {

		//do not divide by 2 as we are not double-counting here
		if (Mnew != DBL3()) return -pMesh->h.dim() * MU0 * Module_Heff[pMesh->M.cellidx_to_position(spin_index)] * (Mnew - pMesh->M[spin_index]);
		else return -pMesh->h.dim() * MU0 * Module_Heff[pMesh->M.cellidx_to_position(spin_index)] * pMesh->M[spin_index];
	}
	else return 0.0;
}

//AFM mesh
DBL2 Demag::Get_EnergyChange(int spin_index, DBL3 Mnew_A, DBL3 Mnew_B)
{
	//Module_Heff needs to be calculated (done during a Monte Carlo simulation, where this method would be used)
	if (Module_Heff.linear_size()) {

		DBL3 M = (pMesh->M[spin_index] + pMesh->M2[spin_index]) / 2;
		DBL3 Mnew = (Mnew_A + Mnew_B) / 2;

		double energy_ = 0.0;

		//do not divide by 2 as we are not double-counting here
		if (Mnew_A != DBL3() && Mnew_B != DBL3()) {

			energy_ = -pMesh->h.dim() * MU0 * Module_Heff[pMesh->M.cellidx_to_position(spin_index)] * (Mnew - M);
		}
		else {

			energy_ = -pMesh->h.dim() * MU0 * Module_Heff[pMesh->M.cellidx_to_position(spin_index)] * M;
		}

		return DBL2(energy_, energy_);
	}
	else return DBL2();
}

#endif



