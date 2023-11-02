#include "stdafx.h"
#include "Atom_Demag.h"
#include "SuperMesh.h"

#if defined(MODULE_COMPILATION_DEMAG) && ATOMISTIC == 1

#include "SimScheduleDefs.h"

#include "Atom_Mesh.h"

#if COMPILECUDA == 1
#include "Atom_DemagMCUDA.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////

Atom_Demag::Atom_Demag(Atom_Mesh *paMesh_) :
	Modules(),
	Convolution<Atom_Demag, DemagKernel>(paMesh_->n_dm, paMesh_->h_dm),
	ProgramStateNames(this, { VINFO(demag_pbc_images) }, {}),
	EvalSpeedup()
{
	paMesh = paMesh_;

	Uninitialize();

	error_on_create = Convolution_Error_on_Create();

	//-------------------------- Is CUDA currently enabled?

	//If cuda is enabled we also need to make the cuda module version
	if (paMesh->cudaEnabled) {

		if (!error_on_create) error_on_create = SwitchCUDAState(true);
	}
}

Atom_Demag::~Atom_Demag()
{
	//when deleting the Demag module any pbc settings should no longer take effect in this mesh
	//thus must clear pbc flags in M1

	paMesh->M1.set_pbc(0, 0, 0);

	//might not need to keep computing fields : if we do then the module which requires it will set the flag back to true on initialization
	paMesh->Set_Force_MonteCarlo_ComputeFields(false);

	//same for the CUDA version if we are in cuda mode
#if COMPILECUDA == 1
	if (pModuleCUDA) {

		paMesh->paMeshCUDA->M1.copyflags_from_cpuvec(paMesh->M1);
	}
#endif
}

//Initialize mesh transfer from atomistic mesh to micromagnetic mesh for demag field computation
BError Atom_Demag::Initialize_Mesh_Transfer(void)
{
	BError error(CLASS_STR(Atom_Demag));

	if (!M.resize(paMesh->h_dm, paMesh->meshRect)) return error(BERROR_OUTOFMEMORY_CRIT);
	if (!Hd.resize(paMesh->h_dm, paMesh->meshRect)) return error(BERROR_OUTOFMEMORY_CRIT);

	if (!M.Initialize_MeshTransfer({ &paMesh->M1 }, {}, MESHTRANSFERTYPE_WDENSITY, MUB)) return error(BERROR_OUTOFMEMORY_CRIT);
	if (!Hd.Initialize_MeshTransfer({}, { &paMesh->Heff1 }, MESHTRANSFERTYPE_ENLARGED)) return error(BERROR_OUTOFMEMORY_CRIT);

	if (paMesh->pSMesh->GetEvaluationSpeedup()) {

		std::function<bool(VEC<DBL3>&)> initialize_mesh_transfer = [&](VEC<DBL3>& H) -> bool {

			if (!H.Initialize_MeshTransfer({}, { &paMesh->Heff1 }, MESHTRANSFERTYPE_ENLARGED)) return false;
			else return true;
		};

		EvalSpeedup::Initialize_EvalSpeedup(
			DemagTFunc().SelfDemag_PBC(paMesh->h_dm, paMesh->n_dm, demag_pbc_images),
			paMesh->pSMesh->GetEvaluationSpeedup(),
			paMesh->h_dm, paMesh->meshRect,
			initialize_mesh_transfer
		);

		EvalSpeedup::Initialize_EvalSpeedup_Mode_Atom(M, Hd);
	}

	return error;
}

BError Atom_Demag::Initialize(void)
{
	BError error(CLASS_STR(Atom_Demag));

	if (!initialized) {

		error = Calculate_Demag_Kernels();

		error = Initialize_Mesh_Transfer();

		if (!error) initialized = true;
	}

	EvalSpeedup::num_Hdemag_saved = 0;

	//need to calculate non-empty cells here so we don't waste time during computations (M is a VEC, not a VEC_VC, which means non-empty cells need to be calculated on every call)
	M.transfer_in();
	non_empty_cells = M.get_nonempty_cells();

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		paMesh->h_dm, paMesh->meshRect, 
		(MOD_)paMesh->Get_Module_Heff_Display() == MOD_DEMAG || paMesh->IsOutputDataSet_withRect(DATA_E_DEMAG) || paMesh->IsStageSet(SS_MONTECARLO),
		(MOD_)paMesh->Get_Module_Energy_Display() == MOD_DEMAG || paMesh->IsOutputDataSet_withRect(DATA_E_DEMAG) || paMesh->IsStageSet(SS_MONTECARLO));
	if (error)	initialized = false;

	//if a Monte Carlo stage is set then we need to compute fields
	if (paMesh->IsStageSet(SS_MONTECARLO)) paMesh->Set_Force_MonteCarlo_ComputeFields(true);

	return error;
}

BError Atom_Demag::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Atom_Demag));

	//only need to uninitialize if n or h have changed, or pbc settings have changed
	if (!CheckDimensions(paMesh->n_dm, paMesh->h_dm, demag_pbc_images) || cfgMessage == UPDATECONFIG_MESHCHANGE || cfgMessage == UPDATECONFIG_DEMAG_CONVCHANGE) {

		Uninitialize();

		//Set convolution dimensions for embedded multiplication and required PBC conditions
		error = SetDimensions(paMesh->n_dm, paMesh->h_dm, true, demag_pbc_images);

		Hd.clear();
		M.clear();

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
BError Atom_Demag::Set_PBC(INT3 demag_pbc_images_)
{
	BError error(__FUNCTION__);

	demag_pbc_images = demag_pbc_images_;

	paMesh->Set_Magnetic_PBC(demag_pbc_images);

	//update will be needed if pbc settings have changed
	error = UpdateConfiguration(UPDATECONFIG_DEMAG_CONVCHANGE);

	return error;
}

BError Atom_Demag::MakeCUDAModule(void)
{
	BError error(CLASS_STR(Atom_Demag));

#if COMPILECUDA == 1

	if (paMesh->paMeshCUDA) {

		//Note : it is posible pMeshCUDA has not been allocated yet, but this module has been created whilst cuda is switched on. This will happen when a new mesh is being made which adds this module by default.
		//In this case, after the mesh has been fully made, it will call SwitchCUDAState on the mesh, which in turn will call this SwitchCUDAState method; then pMeshCUDA will not be nullptr and we can make the cuda module version
		pModuleCUDA = new Atom_DemagMCUDA(paMesh->paMeshCUDA);
		error = pModuleCUDA->Error_On_Create();
	}

#endif

	return error;
}

double Atom_Demag::UpdateField(void)
{
	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////// NO SPEEDUP //////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (!paMesh->pSMesh->GetEvaluationSpeedup() || (num_Hdemag_saved < paMesh->pSMesh->GetEvaluationSpeedup() && !paMesh->pSMesh->Check_Step_Update())) {

		//don't use evaluation speedup

		//transfer magnetic moments to magnetization mesh, converting from moment to magnetization in the process
		M.transfer_in();

		//convolute and get "energy" value
		if (Module_Heff.linear_size()) energy = Convolute(M, Hd, true, &Module_Heff, &Module_energy);
		else energy = Convolute(M, Hd, true);

		//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
		Hd.transfer_out();

		//finish off energy value
		if (non_empty_cells) energy *= -MU0 / (2 * non_empty_cells);
		else energy = 0;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////// EVAL SPEEDUP /////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	else {

		std::function<void(VEC<DBL3>&)> do_evaluation = [&](VEC<DBL3>& H) -> void {

			//convolute and get "energy" value
			if (Module_Heff.linear_size()) energy = Convolute(M, H, true, &Module_Heff, &Module_energy);
			else energy = Convolute(M, H, true);

			//finish off energy value
			if (non_empty_cells) energy *= -MU0 / (2 * non_empty_cells);
			else energy = 0;
		};

		std::function<void(void)> do_transfer_in = [&](void) -> void {

			M.transfer_in();
		};

		std::function<void(VEC<DBL3>&)> do_transfer_out = [&](VEC<DBL3>& H) -> void {

			H.transfer_out();
		};

		EvalSpeedup::UpdateField_EvalSpeedup(
			paMesh->pSMesh->GetEvaluationSpeedup(), paMesh->pSMesh->Check_Step_Update(),
			paMesh->pSMesh->Get_EvalStep_Time(),
			do_evaluation,
			do_transfer_in, do_transfer_out);
	}

	return energy;
}

//-------------------Energy methods

//For simple cubic mesh spin_index coincides with index in M1
double Atom_Demag::Get_EnergyChange(int spin_index, DBL3 Mnew)
{
	//Energy at spin i is then E_i = -mu0 * Hd_i * mu_i, where mu_i is the magnetic moment, Hd_i is the dipole-dipole field at spin i. 
	//Note, no division by 2: this only comes in the total energy since there we consider pairs twice.

	//Module_Heff needs to be calculated (done during a Monte Carlo simulation, where this method would be used)
	if (Module_Heff.linear_size()) {

		if (Mnew != DBL3()) return -MUB_MU0 * Module_Heff[paMesh->M1.cellidx_to_position(spin_index)] * (Mnew - paMesh->M1[spin_index]);
		else return -MUB_MU0 * Module_Heff[paMesh->M1.cellidx_to_position(spin_index)] * paMesh->M1[spin_index];
	}
	else return 0.0;
}

#endif


