#include "stdafx.h"
#include "SuperMesh.h"

//---------------------------------------------------------IMPORTANT CONTROL METHODS

BError SuperMesh::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SuperMesh));
	
#if COMPILECUDA == 1
	//Commands are executed on newly spawned threads, so if cuda is on and we are not using device 0 (default device) we must switch to required device, otherwise 0 will be used
	if (cudaEnabled && cudaDeviceSelect != 0) {

		int device = 0;
		cudaGetDevice(&device);
		if (device != cudaDeviceSelect) cudaSetDevice(cudaDeviceSelect);
	}
#endif

	//1. Magnetic super-mesh - also construct the entire super-mesh rectangle

	//calculate magnetic super-mesh from currently set meshes and super-mesh cellsize
	sMeshRect_fm = Rect();

	sMeshRect = Rect();

	//identify all existing magnetic meshes with magnetic dynamics computations enabled (thus both micromagnetic and atomistic meshes)
	for (int idx = 0; idx < (int)pMesh.size(); idx++) {

		if (pMesh[idx]->MComputation_Enabled() && !pMesh[idx]->Is_Dormant()) {

			//build super-mesh rectangle to include it
			sMeshRect_fm += pMesh[idx]->GetMeshRect();
		}

		sMeshRect += pMesh[idx]->GetMeshRect();
	}

	//extract numbers of cells, and adjust cell-size so it divides the super-mesh rectangle exactly
	n_fm = round(sMeshRect_fm / h_fm);
	if (n_fm.x == 0) n_fm.x = 1;
	if (n_fm.y == 0) n_fm.y = 1;
	if (n_fm.z == 0) n_fm.z = 1;
	if (!sMeshRect_fm.IsNull()) h_fm = sMeshRect_fm / n_fm;

	//2. Electric super-mesh

	//calculate electric super-mesh from currently set meshes and super-mesh cellsize
	sMeshRect_e = Rect();

	//identify all existing magnetic meshes
	for (int idx = 0; idx < (int)pMesh.size(); idx++) {

		if (pMesh[idx]->EComputation_Enabled() && !pMesh[idx]->Is_Dormant()) {

			//build super-mesh rectangle to include it
			sMeshRect_e += pMesh[idx]->GetMeshRect();
		}
	}

	//extract numbers of cells, and adjust cell-size so it divides the super-mesh rectangle exactly
	n_e = round(sMeshRect_e / h_e);
	if (n_e.x <= 1) n_e.x = 2;
	if (n_e.y <= 1) n_e.y = 2;
	if (n_e.z <= 1) n_e.z = 2;
	if (!sMeshRect_e.IsNull()) h_e = sMeshRect_e / n_e;
	
	///////////////////////////////////////////////////////
	//Update configuration for ODE Solvers
	///////////////////////////////////////////////////////

	//important this is called on ODECommon instance before any Mesh UpdateConfiguration, as this in turn will call UpdateConfiguration on DifferentialEquation objects which inherit from ODECommon
	//Any important changes must first be established in ODECommon, as some DifferentialEquation data depends on data in ODECommon
	//There is one important example of this : deleting a mesh when CUDA is enabled can result in deletion of static data in ODECommonCUDA (must do this to prevent GPU memory leaks)
	//Remaining meshes will need these static data, so the UpdateConfiguration method will remake it in ODECommonCUDA
	//However there are GPU pointers to these static data held in instances which inherit from ODECommonCUDA (e.g. DifferentialEquationFMCUDA has a ManagedDiffEqFMCUDA object which holds pointers to these static data to be used in kernel calls).
	//Thus these objects must also be remade so the pointers are set correctly (call ManagedDiffEqFMCUDA set_pointers method); We can only do this if the static data has been re-allocated first, which happens in ODECommonCUDA, hence the priority.
	if (!error) error = odeSolver.UpdateConfiguration(cfgMessage);
	
	//same thing for atomistic ODE
	if (!error) error = atom_odeSolver.UpdateConfiguration(cfgMessage);

	///////////////////////////////////////////////////////
	//Update configuration for meshes and their modules
	///////////////////////////////////////////////////////

	for (int idx = 0; idx < (int)pMesh.size(); idx++) {

		if (!error) error = pMesh[idx]->UpdateConfiguration(cfgMessage);
	}

	///////////////////////////////////////////////////////
	//Update configuration for super-mesh modules
	///////////////////////////////////////////////////////

	for (int idx = 0; idx < (int)pSMod.size(); idx++) {

		if (!error) error = pSMod[idx]->UpdateConfiguration(cfgMessage);
	}

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHSHAPECHANGE, UPDATECONFIG_MESHCHANGE, UPDATECONFIG_MESHADDED, UPDATECONFIG_MESHDELETED, UPDATECONFIG_SWITCHCUDASTATE)) {

		//Check if we need to couple ferromagnetic meshes to dipoles
		if (!error) CoupleToDipoles();
	}

#if COMPILECUDA == 1
	gpuMemFree_MB = cudaMemGetFree() / (1024 * 1024);
	gpuMemTotal_MB = cudaMemGetTotal() / (1024 * 1024);
#endif

	cpuMemFree_MB = MemGetFreeMB() / (1024 * 1024);
	cpuMemTotal_MB = MemGetTotalMB() / (1024 * 1024);
	
	return error;
}

void SuperMesh::UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage)
{
	///////////////////////////////////////////////////////
	//Update configuration for meshes and their modules
	///////////////////////////////////////////////////////

	for (int idx = 0; idx < (int)pMesh.size(); idx++) {

		pMesh[idx]->UpdateConfiguration_Values(cfgMessage);
	}

	///////////////////////////////////////////////////////
	//Update configuration for super-mesh modules
	///////////////////////////////////////////////////////

	for (int idx = 0; idx < (int)pSMod.size(); idx++) {

		pSMod[idx]->UpdateConfiguration_Values(cfgMessage);
	}

	///////////////////////////////////////////////////////
	//Update configuration for ODE Solvers
	///////////////////////////////////////////////////////

	odeSolver.UpdateConfiguration_Values(cfgMessage);

	atom_odeSolver.UpdateConfiguration_Values(cfgMessage);
}


//switch CUDA state on/off; when switching on set selected cuda device number (from 0 up)
//Enforce these rules here:
//1. If CUDA is on, then before device can be changed turn cuda off (free memory for currently selected device), reset device.
//2. If CUDA is off, when turning on set device, then allocate memory.
BError SuperMesh::SwitchCUDAState(bool cudaState, std::vector<int> devices)
{
	BError error(CLASS_STR(SuperMesh));

#if COMPILECUDA == 1

	auto update_configuration = [&](bool __cudaState, BError& error) -> BError {
		
		//Switch for ODECommon
		if (!error) error = odeSolver.SwitchCUDAState(__cudaState);
		
		//Switch for Atom_ODECommon
		if (!error) error = atom_odeSolver.SwitchCUDAState(__cudaState);
		
		//Switch for Meshes and their modules
		for (int idx = 0; idx < (int)pMesh.size(); idx++) {

			if (!error) error = pMesh[idx]->SwitchCUDAState(__cudaState);
		}

		//Switch for super-mesh modules
		for (int idx = 0; idx < (int)pSMod.size(); idx++) {

			if (!error) error = pSMod[idx]->SwitchCUDAState(__cudaState);
		}

		//make sure configuration is updated for the new mode
		error = UpdateConfiguration(UPDATECONFIG_SWITCHCUDASTATE);

		return error;
	};

	//call only if CUDA currently on
	auto turn_cuda_off_for_current_device = [&](BError& error) -> BError {

		if (pSMeshCUDA) delete pSMeshCUDA;
		pSMeshCUDA = nullptr;

		error = update_configuration(false, error);

		if (!error) cudaEnabled = false;

		return error;
	};

	//call only if CUDA currently off
	auto turn_cuda_on_for_selected_device = [&](std::vector<int> devices, BError& error) -> BError {

		error = select_cuda_devices(devices);
		if (error) return error;

		if (!pSMeshCUDA) {

			pSMeshCUDA = new SuperMeshCUDA(this);
		}
		
		//Monte-Carlo serial mode not possible with cuda on
		Set_MonteCarlo_Serial(false, superMeshHandle);

		error = update_configuration(true, error);

		if (!error) cudaEnabled = true;

		return error;
	};

	if (cudaState) {

		//switching CUDA on

		//If currently off, then proceed
		if (!cudaEnabled) error = turn_cuda_on_for_selected_device(devices, error);
		else {

			//currently on, so this action only makes sense if switching device; if not switching device then action is pointless so do nothing
			if (!mGPU.configuration_already_set(devices)) {

				//In this case, first switch off for current device
				error = turn_cuda_off_for_current_device(error);
				//Now it's off, so if all fine then proceed to switch device
				if (!error) error = turn_cuda_on_for_selected_device(devices, error);
			}
		}
	}
	else {

		//switching CUDA off - just do it!
		error = turn_cuda_off_for_current_device(error);

		error = select_cuda_devices(devices);
	}

	gpuMemFree_MB = cudaMemGetFree() / (1024 * 1024);
	gpuMemTotal_MB = cudaMemGetTotal() / (1024 * 1024);

	cpuMemFree_MB = MemGetFreeMB() / (1024 * 1024);
	cpuMemTotal_MB = MemGetTotalMB() / (1024 * 1024);

#endif

	return error;
}

//couple magnetic meshes to any touching dipole meshes, setting interface cell values and flags
//this is used for moving mesh algorithm, in particular for domain wall simulations, where effectively an infinite wire is achieved by removing end magnetic charges, and setting exchange coupling to dipole magnetization direction.
//Magnetic charges remove using stray field from dipoles; "exchange coupling" achieved by setting skip cells at the ends of the wire: i.e. cells which are not updated by the ODE solver, and have magnetization direction along the dipole direction 
void SuperMesh::CoupleToDipoles(void)
{
	for (int idx = 0; idx < (int)pMesh.size(); idx++) {

		pMesh[idx]->CoupleToDipoles(coupled_dipoles); 
	}
}