#include "stdafx.h"
#include "SDemagCUDA_Demag.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SDEMAG

#include "SDemagMCUDA_Demag_single.h"

#include "SDemag.h"
#include "SDemag_Demag.h"
#include "MeshDefs.h"
#include "SuperMesh.h"

SDemagCUDA_Demag::SDemagCUDA_Demag(MeshBaseCUDA* pMeshBaseCUDA_, SDemag_Demag *pSDemag_Demag_) :
	ModulesCUDA(),
	EvalSpeedupCUDA(),
	transfer(mGPU), 
	transfer_Module_Heff(mGPU), transfer_Module_energy(mGPU),
	energy_density_weight(mGPU)
{
	Uninitialize();

	/////////////////////////////////////////////////////////////
	//Setup pointers

	pMeshBaseCUDA = pMeshBaseCUDA_;

	if (!pMeshBaseCUDA->is_atomistic()) pMeshCUDA = dynamic_cast<MeshCUDA*>(pMeshBaseCUDA);
	else paMeshCUDA = dynamic_cast<Atom_MeshCUDA*>(pMeshBaseCUDA);

	pSDemag_Demag = pSDemag_Demag_;

	/////////////////////////////////////////////////////////////
	//make SDemagMCUDA_Demag_single objects

	pDemagMCUDA.resize(mGPU.get_num_devices());

	//don't use the mGPU for loop construct since mGPU could change whilst making objects below
	for (int idx_from = 0; idx_from < mGPU.get_num_devices(); idx_from++) {

		M_Input_transfer.push_back(std::vector<mGPU_Transfer<cuReal3>*>());
		M_Input_transfer_half.push_back(std::vector<mGPU_Transfer<cuBHalf>*>());

		xFFT_Data_transfer.push_back(std::vector<mGPU_Transfer<cuBComplex>*>());
		xFFT_Data_transfer_half.push_back(std::vector<mGPU_Transfer<cuBHalf>*>());

		xIFFT_Data_transfer.push_back(std::vector<mGPU_Transfer<cuBComplex>*>());
		xIFFT_Data_transfer_half.push_back(std::vector<mGPU_Transfer<cuBHalf>*>());

		Out_Data_transfer.push_back(std::vector<mGPU_Transfer<cuReal3>*>());
		Out_Data_transfer_half.push_back(std::vector<mGPU_Transfer<cuBHalf>*>());

		for (int idx_to = 0; idx_to < mGPU.get_num_devices(); idx_to++) {

			M_Input_transfer[idx_from].push_back(new mGPU_Transfer<cuReal3>(mGPU));
			M_Input_transfer_half[idx_from].push_back(new mGPU_Transfer<cuBHalf>(mGPU));

			xFFT_Data_transfer[idx_from].push_back(new mGPU_Transfer<cuBComplex>(mGPU));
			xFFT_Data_transfer_half[idx_from].push_back(new mGPU_Transfer<cuBHalf>(mGPU));

			xIFFT_Data_transfer[idx_from].push_back(new mGPU_Transfer<cuBComplex>(mGPU));
			xIFFT_Data_transfer_half[idx_from].push_back(new mGPU_Transfer<cuBHalf>(mGPU));

			Out_Data_transfer[idx_from].push_back(new mGPU_Transfer<cuReal3>(mGPU));
			Out_Data_transfer_half[idx_from].push_back(new mGPU_Transfer<cuBHalf>(mGPU));
		}
	}

	for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

		mGPU.select_device(idx);
		//make single demag module on given device
		pDemagMCUDA[idx] = new SDemagMCUDA_Demag_single(this, idx);
	}
}

SDemagCUDA_Demag::~SDemagCUDA_Demag() 
{
	for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

		mGPU.select_device(idx);
		if (pDemagMCUDA[idx]) delete pDemagMCUDA[idx];
		pDemagMCUDA[idx] = nullptr;

		for (int idx_to = 0; idx_to < mGPU.get_num_devices(); idx_to++) {

			if (M_Input_transfer[idx][idx_to]) delete M_Input_transfer[idx][idx_to];
			if (M_Input_transfer_half[idx][idx_to]) delete M_Input_transfer_half[idx][idx_to];

			if (xFFT_Data_transfer[idx][idx_to]) delete xFFT_Data_transfer[idx][idx_to];
			if (xFFT_Data_transfer_half[idx][idx_to]) delete xFFT_Data_transfer_half[idx][idx_to];

			if (xIFFT_Data_transfer[idx][idx_to]) delete xIFFT_Data_transfer[idx][idx_to];
			if (xIFFT_Data_transfer_half[idx][idx_to]) delete xIFFT_Data_transfer_half[idx][idx_to];

			if (Out_Data_transfer[idx][idx_to]) delete Out_Data_transfer[idx][idx_to];
			if (Out_Data_transfer_half[idx][idx_to]) delete Out_Data_transfer_half[idx][idx_to];
		}
	}
}

//Get Ms0 value for mesh holding this module
double SDemagCUDA_Demag::Get_Ms0(void)
{
	if (!pMeshBaseCUDA->is_atomistic()) {

		return pSDemag_Demag->pMesh->Ms.get0();
	}
	else {
		
		return pSDemag_Demag->paMesh->Show_Ms();
	}
}

void SDemagCUDA_Demag::Uninitialize(void)
{
	initialized = false;

	for (int idx = 0; idx < pDemagMCUDA.size(); idx++) {

		pDemagMCUDA[idx]->initialized = false;
	}
}

bool SDemagCUDA_Demag::Submodules_Initialized(void)
{
	bool all_initialized = true;

	for (int idx = 0; idx < pDemagMCUDA.size(); idx++) {

		all_initialized &= pDemagMCUDA[idx]->initialized;
	}

	return all_initialized;
}

//this is called from SDemagCUDA so this mesh module can set convolution sizes
BError SDemagCUDA_Demag::Set_Convolution_Dimensions(cuBReal h_max, cuSZ3 n_common, cuINT3 pbc, std::vector<cuRect>& Rect_collection, int mesh_idx)
{
	BError error(CLASS_STR(SDemagCUDA_Demag));

	/////////////////////////////////////////////////////////////
	//Do we need mesh transfer? If so allocate memory for transfer.

	//convolution rectangle for this module
	cuRect convolution_rect = Rect_collection[mesh_idx];

	//common discretisation cellsize (may differ in thickness in 2D mode)
	cuReal3 h_common = convolution_rect / n_common;

	if (!pMeshBaseCUDA->is_atomistic() && convolution_rect == (cuRect)pSDemag_Demag->meshRect && h_common == (cuReal3)pSDemag_Demag->h) {

		//no transfer required (always force transfer for atomistic meshes)
		do_transfer = false;
		transfer.clear();
	}
	else {

		do_transfer = true;

		//set correct size for transfer cuVEC
		if (!transfer.resize(h_common, convolution_rect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
	}

	//number of macrocells along x must be greater or equal to number of devices used (x partitioning used)
	if ((do_transfer ? transfer.n.x : pMeshCUDA->M.n.x) < mGPU.get_num_devices()) {

		Uninitialize();
		return error(BERROR_MGPU_XCELLS);
	}

	/////////////////////////////////////////////////////////////
	//Setup convolution dimensions

	//h_convolution may differ from h_common in 2D mode
	cuReal3 h_convolution = Rect_collection[mesh_idx] / n_common;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (!pDemagMCUDA[mGPU]->CheckDimensions(n_common, h_convolution, pbc)) {

			//the convolution x region is determined by the M mcuVEC partitioning along the x axis
			cuBox dbox = (do_transfer ? transfer.device_box(mGPU) : pMeshCUDA->M.device_box(mGPU));
			cuINT2 xRegion = cuINT2(dbox.s.x, dbox.e.x);

			//Set convolution dimensions and required PBC conditions
			error = pDemagMCUDA[mGPU]->SetDimensions(n_common, h_convolution, CONV_MULTIMESH, pbc, xRegion, { mGPU, mGPU.get_num_devices() });
		}
	}

	//set all rect collections
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
		
		error = pDemagMCUDA[mGPU]->Set_Rect_Collection(Rect_collection, Rect_collection[mesh_idx], h_max, mesh_idx);
	}

	return error;
}

BError SDemagCUDA_Demag::Initialize(void)
{
	BError error(CLASS_STR(SDemagCUDA_Demag));

	//no energy density contribution here
	ZeroEnergy();

	//pointer to cpu SDemag object
	SDemag* pSDemag = pSDemag_Demag->pSDemag;

	//pointer to gpu SDemagCUDA object
	SDemagCUDA* pSDemagCUDA = dynamic_cast<SDemagCUDA*>(pSDemag->pModuleCUDA);

	//Make sure SDemagCUDA is initialized - this will allocate convolution sizes
	if (!pSDemagCUDA->IsInitialized()) error = pSDemagCUDA->Initialize();
	if (error) return error;

	//convolution rectangle for this module
	cuRect convolution_rect = (cuRect)pSDemag->get_convolution_rect(pSDemag_Demag);

	//common discretisation cellsize (may differ in thickness in 2D mode)
	cuReal3 h_common = convolution_rect / (cuSZ3)pSDemag->n_common;

	if (!initialized) {

		/////////////////////////////////////////////////////////////
		//Dimensions

		//initialize each DemagMCUDA_single object (one per device) - kernel collections are calculated here
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = pDemagMCUDA[mGPU]->Initialize(pSDemagCUDA->kernel_collection[mGPU], (do_transfer ? transfer.n.z : pMeshCUDA->M.n.z), pSDemag_Demag->pSDemag->pSMesh->Get_Kernel_Initialize_on_GPU());
		}

		if (!Submodules_Initialized()) return error;

		/////////////////////////////////////////////////////////////
		//Transfer objects

		if (mGPU.get_num_devices() > 1) {

			for (int idx_from = 0; idx_from < mGPU.get_num_devices(); idx_from++) {
				for (int idx_to = 0; idx_to < mGPU.get_num_devices(); idx_to++) {

					if (idx_to == idx_from) continue;

					if (!mGPU.get_halfprecision_transfer()) {

						M_Input_transfer[idx_from][idx_to]->set_transfer_size(pDemagMCUDA[idx_from]->Real_xRegion[idx_to]->size());
						M_Input_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDemagMCUDA[idx_from]->Real_xRegion[idx_to]->get_array());
						M_Input_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDemagMCUDA[idx_to]->Real_yRegion[idx_from]->get_array());

						xFFT_Data_transfer[idx_from][idx_to]->set_transfer_size(pDemagMCUDA[idx_from]->Complex_yRegion[idx_to]->size());
						xFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDemagMCUDA[idx_from]->Complex_yRegion[idx_to]->get_array());
						xFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDemagMCUDA[idx_to]->Complex_xRegion[idx_from]->get_array());

						xIFFT_Data_transfer[idx_from][idx_to]->set_transfer_size(pDemagMCUDA[idx_from]->Complex_xRegion[idx_to]->size());
						xIFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDemagMCUDA[idx_from]->Complex_xRegion[idx_to]->get_array());
						xIFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDemagMCUDA[idx_to]->Complex_yRegion[idx_from]->get_array());

						Out_Data_transfer[idx_from][idx_to]->set_transfer_size(pDemagMCUDA[idx_from]->Real_yRegion[idx_to]->size());
						Out_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDemagMCUDA[idx_from]->Real_yRegion[idx_to]->get_array());
						Out_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDemagMCUDA[idx_to]->Real_xRegion[idx_from]->get_array());
					}
					else {

						M_Input_transfer_half[idx_from][idx_to]->set_transfer_size(pDemagMCUDA[idx_from]->Real_xRegion_half[idx_to]->size());
						M_Input_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDemagMCUDA[idx_from]->Real_xRegion_half[idx_to]->get_array());
						M_Input_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDemagMCUDA[idx_to]->Real_yRegion_half[idx_from]->get_array());

						xFFT_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pDemagMCUDA[idx_from]->Complex_yRegion_half[idx_to]->size());
						xFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDemagMCUDA[idx_from]->Complex_yRegion_half[idx_to]->get_array());
						xFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDemagMCUDA[idx_to]->Complex_xRegion_half[idx_from]->get_array());

						xIFFT_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pDemagMCUDA[idx_from]->Complex_xRegion_half[idx_to]->size());
						xIFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDemagMCUDA[idx_from]->Complex_xRegion_half[idx_to]->get_array());
						xIFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDemagMCUDA[idx_to]->Complex_yRegion_half[idx_from]->get_array());

						Out_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pDemagMCUDA[idx_from]->Real_yRegion_half[idx_to]->size());
						Out_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDemagMCUDA[idx_from]->Real_yRegion_half[idx_to]->get_array());
						Out_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDemagMCUDA[idx_to]->Real_xRegion_half[idx_from]->get_array());
					}
				}
			}
		}

		int non_empty_cells = 0;

		if (!do_transfer) {

			non_empty_cells = pMeshCUDA->M.get_nonempty_cells_cpu();

			//initialize eval speedup if needed
			if (pMeshBaseCUDA->GetEvaluationSpeedup()) {

				EvalSpeedupCUDA::Initialize_EvalSpeedup(
					DemagTFunc().SelfDemag_PBC(h_common, (cuSZ3)pSDemag->n_common, pSDemag->demag_pbc_images),
					pMeshBaseCUDA->GetEvaluationSpeedup(),
					h_common, convolution_rect);
			}
		}
		else {

			/////////////////////////////////////////////////////////////
			// Make mesh transfers in transfer

			//Now copy mesh transfer object to cuda version
			std::vector<mcu_VEC_VC(cuReal3)*> pVal_from, pVal_from2;
			std::vector<mcu_VEC(cuReal3)*> pVal_to, pVal_to2;

			std::vector<mcu_VEC_VC(cuReal3)*> pVal_afrom;
			std::vector<mcu_VEC(cuReal3)*> pVal_ato;

			if (!pMeshBaseCUDA->is_atomistic()) {

				pVal_from.push_back(&pMeshCUDA->M);
				pVal_to.push_back(&pMeshCUDA->Heff);
				pVal_from2.push_back(&pMeshCUDA->M2);
				pVal_to2.push_back(&pMeshCUDA->Heff2);
			}
			else {

				pVal_afrom.push_back(&paMeshCUDA->M1);
				pVal_ato.push_back(&paMeshCUDA->Heff1);
			}

			//now calculate in cpu memory
			pSDemag_Demag->Initialize_Mesh_Transfer();

			///////////////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////// FERROMAGNETIC MESH /////////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////////////////////

			if (pMeshBaseCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

				if (!transfer.copy_transfer_info<cuVEC_VC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>
					(pVal_from, pVal_to, pSDemag_Demag->transfer.get_transfer())) return error(BERROR_OUTOFGPUMEMORY_CRIT);

				//initialize eval speedup if needed
				if (pMeshBaseCUDA->GetEvaluationSpeedup()) {

					EvalSpeedupCUDA::Initialize_EvalSpeedup(
						DemagTFunc().SelfDemag_PBC(h_common, (cuSZ3)pSDemag->n_common, pSDemag->demag_pbc_images),
						pMeshBaseCUDA->GetEvaluationSpeedup(),
						h_common, convolution_rect,
						pVal_to, std::vector<mcu_VEC(cuReal3)*>{}, &pSDemag_Demag->transfer.get_transfer());
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////// ANTIFERROMAGNETIC MESH ///////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////////////////////

			else if (pMeshBaseCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

				if (!transfer.copy_transfer_info_averagedinputs_duplicatedoutputs<cuVEC_VC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>
					(pVal_from, pVal_from2, pVal_to, pVal_to2, pSDemag_Demag->transfer.get_transfer())) return error(BERROR_OUTOFGPUMEMORY_CRIT);

				//initialize eval speedup if needed
				if (pMeshBaseCUDA->GetEvaluationSpeedup()) {

					EvalSpeedupCUDA::Initialize_EvalSpeedup(
						DemagTFunc().SelfDemag_PBC(h_common, (cuSZ3)pSDemag->n_common, pSDemag->demag_pbc_images),
						pMeshBaseCUDA->GetEvaluationSpeedup(),
						h_common, convolution_rect,
						pVal_to, pVal_to2, &pSDemag_Demag->transfer.get_transfer());
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////// ATOMISTIC MESH ///////////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////////////////////

			else {

				if (!transfer.copy_transfer_info<cuVEC_VC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>
					(pVal_afrom, pVal_ato, pSDemag_Demag->transfer.get_transfer())) return error(BERROR_OUTOFGPUMEMORY_CRIT);

				//initialize eval speedup if needed
				if (pMeshBaseCUDA->GetEvaluationSpeedup()) {

					EvalSpeedupCUDA::Initialize_EvalSpeedup(
						DemagTFunc().SelfDemag_PBC(h_common, (cuSZ3)pSDemag->n_common, pSDemag->demag_pbc_images),
						pMeshBaseCUDA->GetEvaluationSpeedup(),
						h_common, convolution_rect,
						pVal_ato, std::vector<mcu_VEC(cuReal3)*>{}, & pSDemag_Demag->transfer.get_transfer());
				}
			}

			non_empty_cells = pSDemag_Demag->transfer.get_nonempty_cells();
		}
		
		//setup energy density weights
		if (pSDemagCUDA->total_nonempty_volume) {

			energy_density_weight.from_cpu((cuBReal)(non_empty_cells * h_common.dim() / pSDemagCUDA->total_nonempty_volume));
		}
		
		if (pMeshBaseCUDA->GetEvaluationSpeedup()) {

			if (pMeshCUDA) {

				if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

					EvalSpeedupCUDA::Initialize_EvalSpeedup_Mode_FM(pMeshCUDA->M, pMeshCUDA->Heff);
				}
				else if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					EvalSpeedupCUDA::Initialize_EvalSpeedup_Mode_AFM(pMeshCUDA->M, pMeshCUDA->M2, pMeshCUDA->Heff, pMeshCUDA->Heff2);
				}
			}
			//transfer is forced for atomistic meshes
			else {

				EvalSpeedupCUDA::Initialize_EvalSpeedup_Mode_Atom(transfer, transfer);
			}
		}

		initialized = true;
	}

	EvalSpeedupCUDA::num_Hdemag_saved = 0;

	//Make sure display data has memory allocated (or freed) as required : mirror cpu module version
	error = pSDemag_Demag->Initialize_Module_Display();
	if (error) { initialized = false; return error(BERROR_OUTOFMEMORY_CRIT); }

	if (pSDemag_Demag->Get_Module_Heff().linear_size()) {

		error = Update_Module_Display_VECs((cuReal3)pMeshBaseCUDA->h, (cuRect)pMeshBaseCUDA->meshRect, true, true);
		if (error) return error(BERROR_OUTOFGPUMEMORY_CRIT);

		set_SDemag_DemagCUDA_pointers();
	}

	//allocate memory and copy transfer info for module display transfer objects if needed
	if (pSDemag_Demag->transfer_Module_Heff.linear_size()) {

		if (!transfer_Module_Heff.resize(h_common, convolution_rect)) { initialized = false; return error(BERROR_OUTOFGPUMEMORY_CRIT); }

		std::vector<mcu_VEC(cuReal3)*> pVal_to;

		if (!pMeshBaseCUDA->is_atomistic()) {

			pVal_to.push_back(&(*(pSDemag_Demag->pMesh))(MOD_SDEMAG_DEMAG)->Get_Module_HeffCUDA());
		}
		else {

			pVal_to.push_back(&(*(pSDemag_Demag->paMesh))(MOD_SDEMAG_DEMAG)->Get_Module_HeffCUDA());
		}

		if (!transfer_Module_Heff.copy_transfer_info<cuVEC_VC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>({}, pVal_to, pSDemag_Demag->transfer_Module_Heff.get_transfer())) {
			
			initialized = false; return error(BERROR_OUTOFGPUMEMORY_CRIT);
		}
	}

	if (pSDemag_Demag->transfer_Module_energy.linear_size()) {

		if (!transfer_Module_energy.resize(h_common, convolution_rect)) { initialized = false; return error(BERROR_OUTOFGPUMEMORY_CRIT); }

		std::vector<mcu_VEC(cuBReal)*> pVal_to;

		if (!pMeshBaseCUDA->is_atomistic()) {

			pVal_to.push_back(&(*(pSDemag_Demag->pMesh))(MOD_SDEMAG_DEMAG)->Get_Module_EnergyCUDA());
		}
		else {

			pVal_to.push_back(&(*(pSDemag_Demag->paMesh))(MOD_SDEMAG_DEMAG)->Get_Module_EnergyCUDA());
		}

		if (!transfer_Module_energy.copy_transfer_info<cuVEC_VC<cuBReal>, cuVEC<cuBReal>, Transfer<double>>({}, pVal_to, pSDemag_Demag->transfer_Module_energy.get_transfer())) {
			
			initialized = false; return error(BERROR_OUTOFGPUMEMORY_CRIT); 
		}
	}

	return error;
}

BError SDemagCUDA_Demag::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SDemagCUDA_Demag));
	
	//just mirror the initialization flag in the cpu version module
	if (!pSDemag_Demag->IsInitialized()) Uninitialize();

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_DEMAG_CONVCHANGE)) Uninitialize();

	//update configuration for all submodules
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		error = pDemagMCUDA[mGPU]->UpdateConfiguration(cfgMessage);
	}

	//if memory needs to be allocated for Hdemag, it will be done through Initialize 
	if (!initialized) {
		
		EvalSpeedupCUDA::UpdateConfiguration_EvalSpeedup();
	}

	return error;
}

void SDemagCUDA_Demag::UpdateField(void)
{
	//demag field update done through the supermesh module.
	//here we need to zero the module display objects in case they are used : if we have to transfer data into them from the display transfer objects, this is done by adding.
	//cannot set output when transferring since we can have multiple transfer objects contributing to the display objects
	ZeroModuleVECs();

	//same goes for the total energy
	ZeroEnergy();
}

#endif

#endif