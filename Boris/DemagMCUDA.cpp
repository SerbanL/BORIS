#include "stdafx.h"
#include "DemagMCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_DEMAG

#include "DemagMCUDA_single.h"

#include "SimScheduleDefs.h"

#include "MeshCUDA.h"
#include "Mesh.h"
#include "DataDefs.h"
#include "SuperMesh.h"

DemagMCUDA::DemagMCUDA(MeshCUDA* pMeshCUDA_) :
	ModulesCUDA(),
	Hdemag(mGPU), Hdemag2(mGPU), Hdemag3(mGPU), Hdemag4(mGPU), Hdemag5(mGPU), Hdemag6(mGPU),
	selfDemagCoeff(mGPU)
{
	Uninitialize();

	pMeshCUDA = pMeshCUDA_;

	//make DemagMCUDA_single objects
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
		pDemagMCUDA[idx] = new DemagMCUDA_single(pMeshCUDA, this, idx);
	}
}

DemagMCUDA::~DemagMCUDA() 
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

BError DemagMCUDA::Initialize(void)
{
	BError error(CLASS_STR(DemagMCUDA));

	//this module only works with subvec axis x (multiple devices)
	if (mGPU.get_num_devices() > 1 && mGPU.get_subvec_axis() != 0) {

		Uninitialize();
		return error(BERROR_MGPU_MUSTBEXAXIS);
	}

	if (!initialized) {

		/////////////////////////////////////////////////////////////
		//Dimensions
		
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			if (!pDemagMCUDA[mGPU]->CheckDimensions(pMeshCUDA->n, pMeshCUDA->h, pMeshCUDA->M.Get_PBC())) {

				//the convolution x region is determined by the M mcuVEC partitioning along the x axis
				cuBox dbox = pMeshCUDA->M.device_box(mGPU);
				cuINT2 xRegion = cuINT2(dbox.s.x, dbox.e.x);

				//Set convolution dimensions and required PBC conditions
				error = pDemagMCUDA[mGPU]->SetDimensions(pMeshCUDA->n, pMeshCUDA->h, CONV_SINGLEMESH, pMeshCUDA->M.Get_PBC(), xRegion, { mGPU, mGPU.get_num_devices() });
			}
		}
		
		//initialize each DemagMCUDA_single object (one per device) - kernel collections are calculated here
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = pDemagMCUDA[mGPU]->Initialize();
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

		/////////////////////////////////////////////////////////////
		//Eval speedup

		selfDemagCoeff.from_cpu(DemagTFunc().SelfDemag_PBC(pMeshCUDA->h, pMeshCUDA->n, pMeshCUDA->M.Get_PBC()));

		//make sure to allocate memory for Hdemag if we need it
		if (pMeshCUDA->GetEvaluationSpeedup() >= 6) { if (!Hdemag6.resize(pMeshCUDA->h, pMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag6.clear();

		if (pMeshCUDA->GetEvaluationSpeedup() >= 5) { if (!Hdemag5.resize(pMeshCUDA->h, pMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag5.clear();

		if (pMeshCUDA->GetEvaluationSpeedup() >= 4) { if (!Hdemag4.resize(pMeshCUDA->h, pMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag4.clear();

		if (pMeshCUDA->GetEvaluationSpeedup() >= 3) { if (!Hdemag3.resize(pMeshCUDA->h, pMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag3.clear();

		if (pMeshCUDA->GetEvaluationSpeedup() >= 2) { if (!Hdemag2.resize(pMeshCUDA->h, pMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag2.clear();

		if (pMeshCUDA->GetEvaluationSpeedup() >= 1) { if (!Hdemag.resize(pMeshCUDA->h, pMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag.clear();

		if (!error) initialized = true;
	}

	num_Hdemag_saved = 0;

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		(cuReal3)pMeshCUDA->h, (cuRect)pMeshCUDA->meshRect, 
		(MOD_)pMeshCUDA->Get_Module_Heff_Display() == MOD_DEMAG || pMeshCUDA->IsOutputDataSet_withRect(DATA_E_DEMAG) || pMeshCUDA->IsStageSet(SS_MONTECARLO),
		(MOD_)pMeshCUDA->Get_Module_Energy_Display() == MOD_DEMAG || pMeshCUDA->IsOutputDataSet_withRect(DATA_E_DEMAG) || pMeshCUDA->IsStageSet(SS_MONTECARLO));
	if (error) initialized = false;

	if (initialized) set_DemagCUDA_pointers();

	//if a Monte Carlo stage is set then we need to compute fields
	if (pMeshCUDA->IsStageSet(SS_MONTECARLO)) pMeshCUDA->Set_Force_MonteCarlo_ComputeFields(true);

	return error;
}

bool DemagMCUDA::Submodules_Initialized(void)
{
	bool all_initialized = true;

	for (int idx = 0; idx < pDemagMCUDA.size(); idx++) {

		all_initialized &= pDemagMCUDA[idx]->initialized;
	}

	return all_initialized;
}

BError DemagMCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(DemagMCUDA));

	//update configuration for all submodules
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		error = pDemagMCUDA[mGPU]->UpdateConfiguration(cfgMessage);
	}

	//unititialize this module also if not all submodules are still all initialized
	if (!Submodules_Initialized()) {

		Uninitialize();

		//if memory needs to be allocated for Hdemag, it will be done through Initialize 
		Hdemag.clear();
		Hdemag2.clear();
		Hdemag3.clear();
		Hdemag4.clear();
		Hdemag5.clear();
		Hdemag6.clear();
	}

	num_Hdemag_saved = 0;

	return error;
}

void DemagMCUDA::UpdateField(void)
{	
	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////// NO SPEEDUP //////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (!pMeshCUDA->GetEvaluationSpeedup() || (num_Hdemag_saved < pMeshCUDA->GetEvaluationSpeedup() && !pMeshCUDA->Check_Step_Update())) {

		if (pMeshCUDA->CurrentTimeStepSolved()) ZeroEnergy();

		///////////////////////////////////////////////////////////////////////////////////////////////
		//MULTIPLE DEVICES DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////
		if (mGPU.get_num_devices() > 1) {

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Copy M data to linear regions so we can transfer
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				pDemagMCUDA[mGPU]->Copy_M_Input_xRegion(mGPU.get_halfprecision_transfer());
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////
			//Transfer data between devices before x FFT
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			if (mGPU.get_halfprecision_transfer()) {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						M_Input_transfer_half[device_from][device_to]->transfer(device_to, device_from);
					}
				}
				mGPU.synchronize();
			}
			else {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						M_Input_transfer[device_from][device_to]->transfer(device_to, device_from);
					}
				}
				mGPU.synchronize();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Forward x FFT for all devices (first step)
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

					if (!mGPU.get_halfprecision_transfer())
						pDemagMCUDA[mGPU]->ForwardFFT_mGPU_first(pMeshCUDA->M.get_deviceobject(mGPU), pDemagMCUDA[mGPU]->Real_yRegion_arr, pDemagMCUDA[mGPU]->Complex_yRegion_arr);
					else 
						pDemagMCUDA[mGPU]->ForwardFFT_mGPU_first(
							pMeshCUDA->M.get_deviceobject(mGPU), 
							pDemagMCUDA[mGPU]->Real_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization_M, 
							pDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
				}

				else if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					if (!mGPU.get_halfprecision_transfer())
						pDemagMCUDA[mGPU]->ForwardFFT_AveragedInputs_mGPU_first(pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU), pDemagMCUDA[mGPU]->Real_yRegion_arr, pDemagMCUDA[mGPU]->Complex_yRegion_arr);
					else
						pDemagMCUDA[mGPU]->ForwardFFT_AveragedInputs_mGPU_first(
							pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU), 
							pDemagMCUDA[mGPU]->Real_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization_M,
							pDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
				}
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////
			//Transfer data between devices after x FFT
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			if (mGPU.get_halfprecision_transfer()) {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						xFFT_Data_transfer_half[device_from][device_to]->transfer(device_to, device_from);
					}
				}
				mGPU.synchronize();
			}
			else {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						xFFT_Data_transfer[device_from][device_to]->transfer(device_to, device_from);
					}
				}
				mGPU.synchronize();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Forward FFT for all devices (last step)
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (!mGPU.get_halfprecision_transfer())
					pDemagMCUDA[mGPU]->ForwardFFT_mGPU_last(pDemagMCUDA[mGPU]->Complex_xRegion_arr);
				else 
					pDemagMCUDA[mGPU]->ForwardFFT_mGPU_last(pDemagMCUDA[mGPU]->Complex_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////
			//Kernel multiplications
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				pDemagMCUDA[mGPU]->KernelMultiplication();
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////
			//Inverse FFT for all devices
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (!mGPU.get_halfprecision_transfer())
					pDemagMCUDA[mGPU]->InverseFFT_mGPU_first(pDemagMCUDA[mGPU]->Complex_xRegion_arr);
				else
					pDemagMCUDA[mGPU]->InverseFFT_mGPU_first(pDemagMCUDA[mGPU]->Complex_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////
			//Transfer data between devices before x IFFT
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			if (mGPU.get_halfprecision_transfer()) {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						xIFFT_Data_transfer_half[device_from][device_to]->transfer(device_to, device_from);
					}
				}
				mGPU.synchronize();
			}
			else {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						xIFFT_Data_transfer[device_from][device_to]->transfer(device_to, device_from);
					}
				}
				mGPU.synchronize();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//x IFFT
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (!mGPU.get_halfprecision_transfer())
					pDemagMCUDA[mGPU]->InverseFFT_mGPU_last(pDemagMCUDA[mGPU]->Complex_yRegion_arr, pDemagMCUDA[mGPU]->Real_yRegion_arr);
				else
					pDemagMCUDA[mGPU]->InverseFFT_mGPU_last(pDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization, pDemagMCUDA[mGPU]->Real_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////
			//Transfer data between devices before finishing
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			if (mGPU.get_halfprecision_transfer()) {
				
				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						Out_Data_transfer_half[device_from][device_to]->transfer(device_to, device_from);
					}
				}
				mGPU.synchronize();
			}
			else {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						Out_Data_transfer[device_from][device_to]->transfer(device_to, device_from);
					}
				}
				mGPU.synchronize();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Finish convolution, setting output in Heff
			///////////////////////////////////////////////////////////////////////////////////////////////
			
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {
					
					if (!mGPU.get_halfprecision_transfer()) {

						if (Module_Heff.linear_size_cpu()) {

							pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_arr,
								pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->Heff.get_deviceobject(mGPU),
								energy(mGPU), pMeshCUDA->CurrentTimeStepSolved(), false,
								Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
						}
						else {

							pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_arr,
								pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->Heff.get_deviceobject(mGPU),
								energy(mGPU), pMeshCUDA->CurrentTimeStepSolved(), false);
						}
					}
					else {

						if (Module_Heff.linear_size_cpu()) {

							pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
								pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->Heff.get_deviceobject(mGPU),
								energy(mGPU), pMeshCUDA->CurrentTimeStepSolved(), false,
								Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
						}
						else {

							pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
								pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->Heff.get_deviceobject(mGPU),
								energy(mGPU), pMeshCUDA->CurrentTimeStepSolved(), false);
						}
					}
				}
				else if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					if (!mGPU.get_halfprecision_transfer()) {

						if (Module_Heff.linear_size_cpu())
							pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_DuplicatedOutputs_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_arr,
								pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU),
								pMeshCUDA->Heff.get_deviceobject(mGPU), pMeshCUDA->Heff2.get_deviceobject(mGPU),
								energy(mGPU), pMeshCUDA->CurrentTimeStepSolved(), false,
								Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
						else
							pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_DuplicatedOutputs_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_arr,
								pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU),
								pMeshCUDA->Heff.get_deviceobject(mGPU), pMeshCUDA->Heff2.get_deviceobject(mGPU),
								energy(mGPU), pMeshCUDA->CurrentTimeStepSolved(), false);
					}
					else {

						if (Module_Heff.linear_size_cpu())
							pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_DuplicatedOutputs_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
								pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU),
								pMeshCUDA->Heff.get_deviceobject(mGPU), pMeshCUDA->Heff2.get_deviceobject(mGPU),
								energy(mGPU), pMeshCUDA->CurrentTimeStepSolved(), false,
								Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
						else
							pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_DuplicatedOutputs_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
								pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU),
								pMeshCUDA->Heff.get_deviceobject(mGPU), pMeshCUDA->Heff2.get_deviceobject(mGPU),
								energy(mGPU), pMeshCUDA->CurrentTimeStepSolved(), false);
					}
				}
			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////
		//SINGLE DEVICE DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////
		else {

			if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

				if (Module_Heff.linear_size_cpu())
					pDemagMCUDA[0]->Convolute(
						pMeshCUDA->M.get_deviceobject(0), pMeshCUDA->Heff.get_deviceobject(0),
						energy(0), pMeshCUDA->CurrentTimeStepSolved(), false,
						&Module_Heff.get_deviceobject(0), &Module_energy.get_deviceobject(0));
				else
					pDemagMCUDA[0]->Convolute(
						pMeshCUDA->M.get_deviceobject(0), pMeshCUDA->Heff.get_deviceobject(0),
						energy(0), pMeshCUDA->CurrentTimeStepSolved(), false);
			}

			else if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

				if (Module_Heff.linear_size_cpu())
					pDemagMCUDA[0]->Convolute_AveragedInputs_DuplicatedOutputs(
						pMeshCUDA->M.get_deviceobject(0), pMeshCUDA->M2.get_deviceobject(0), pMeshCUDA->Heff.get_deviceobject(0), pMeshCUDA->Heff2.get_deviceobject(0),
						energy(0), pMeshCUDA->CurrentTimeStepSolved(), false,
						&Module_Heff.get_deviceobject(0), &Module_energy.get_deviceobject(0));
				else
					pDemagMCUDA[0]->Convolute_AveragedInputs_DuplicatedOutputs(
						pMeshCUDA->M.get_deviceobject(0), pMeshCUDA->M2.get_deviceobject(0), pMeshCUDA->Heff.get_deviceobject(0), pMeshCUDA->Heff2.get_deviceobject(0),
						energy(0), pMeshCUDA->CurrentTimeStepSolved(), false);
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////// EVAL SPEEDUP /////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	else {

		//use evaluation speedup method (Hdemag will have memory allocated - this was done in the Initialize method)

		//update if required by ODE solver or if we don't have enough previous evaluations saved to extrapolate
		if (pMeshCUDA->Check_Step_Update() || num_Hdemag_saved < pMeshCUDA->GetEvaluationSpeedup()) {

			mcu_VEC(cuReal3)* pHdemag;

			if (num_Hdemag_saved < pMeshCUDA->GetEvaluationSpeedup()) {

				//don't have enough evaluations, so save next one
				switch (num_Hdemag_saved)
				{
				case 0:
					pHdemag = &Hdemag;
					time_demag1 = pMeshCUDA->Get_EvalStep_Time();
					break;
				case 1:
					pHdemag = &Hdemag2;
					time_demag2 = pMeshCUDA->Get_EvalStep_Time();
					break;
				case 2:
					pHdemag = &Hdemag3;
					time_demag3 = pMeshCUDA->Get_EvalStep_Time();
					break;
				case 3:
					pHdemag = &Hdemag4;
					time_demag4 = pMeshCUDA->Get_EvalStep_Time();
					break;
				case 4:
					pHdemag = &Hdemag5;
					time_demag5 = pMeshCUDA->Get_EvalStep_Time();
					break;
				case 5:
					pHdemag = &Hdemag6;
					time_demag6 = pMeshCUDA->Get_EvalStep_Time();
					break;
				}

				num_Hdemag_saved++;
			}
			else {

				//have enough evaluations saved, so just cycle between them now

				//QUINTIC
				if (pMeshCUDA->GetEvaluationSpeedup() == 6) {

					//1, 2, 3, 4, 5, 6 -> next is 1
					if (time_demag6 > time_demag5 && time_demag5 > time_demag4 && time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

						pHdemag = &Hdemag;
						time_demag1 = pMeshCUDA->Get_EvalStep_Time();
					}
					//2, 3, 4, 5, 6, 1 -> next is 2
					else if (time_demag1 > time_demag2) {

						pHdemag = &Hdemag2;
						time_demag2 = pMeshCUDA->Get_EvalStep_Time();
					}
					//3, 4, 5, 6, 1, 2 -> next is 3
					else if (time_demag2 > time_demag3) {

						pHdemag = &Hdemag3;
						time_demag3 = pMeshCUDA->Get_EvalStep_Time();
					}
					//4, 5, 6, 1, 2, 3 -> next is 4
					else if (time_demag3 > time_demag4) {

						pHdemag = &Hdemag4;
						time_demag4 = pMeshCUDA->Get_EvalStep_Time();
					}
					//5, 6, 1, 2, 3, 4 -> next is 5
					else if (time_demag4 > time_demag5) {

						pHdemag = &Hdemag5;
						time_demag5 = pMeshCUDA->Get_EvalStep_Time();
					}
					else {

						pHdemag = &Hdemag6;
						time_demag6 = pMeshCUDA->Get_EvalStep_Time();
					}
				}
				//QUARTIC
				else if (pMeshCUDA->GetEvaluationSpeedup() == 5) {

					//1, 2, 3, 4, 5 -> next is 1
					if (time_demag5 > time_demag4 && time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

						pHdemag = &Hdemag;
						time_demag1 = pMeshCUDA->Get_EvalStep_Time();
					}
					//2, 3, 4, 5, 1 -> next is 2
					else if (time_demag1 > time_demag2) {

						pHdemag = &Hdemag2;
						time_demag2 = pMeshCUDA->Get_EvalStep_Time();
					}
					//3, 4, 5, 1, 2 -> next is 3
					else if (time_demag2 > time_demag3) {

						pHdemag = &Hdemag3;
						time_demag3 = pMeshCUDA->Get_EvalStep_Time();
					}
					//4, 5, 1, 2, 3 -> next is 4
					else if (time_demag3 > time_demag4) {

						pHdemag = &Hdemag4;
						time_demag4 = pMeshCUDA->Get_EvalStep_Time();
					}
					else {

						pHdemag = &Hdemag5;
						time_demag5 = pMeshCUDA->Get_EvalStep_Time();
					}
				}
				//CUBIC
				else if (pMeshCUDA->GetEvaluationSpeedup() == 4) {

					//1, 2, 3, 4 -> next is 1
					if (time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

						pHdemag = &Hdemag;
						time_demag1 = pMeshCUDA->Get_EvalStep_Time();
					}
					//2, 3, 4, 1 -> next is 2
					else if (time_demag1 > time_demag2) {

						pHdemag = &Hdemag2;
						time_demag2 = pMeshCUDA->Get_EvalStep_Time();
					}
					//3, 4, 1, 2 -> next is 3
					else if (time_demag2 > time_demag3) {

						pHdemag = &Hdemag3;
						time_demag3 = pMeshCUDA->Get_EvalStep_Time();
					}
					else {

						pHdemag = &Hdemag4;
						time_demag4 = pMeshCUDA->Get_EvalStep_Time();
					}
				}
				//QUADRATIC
				else if (pMeshCUDA->GetEvaluationSpeedup() == 3) {

					//1, 2, 3 -> next is 1
					if (time_demag3 > time_demag2 && time_demag2 > time_demag1) {

						pHdemag = &Hdemag;
						time_demag1 = pMeshCUDA->Get_EvalStep_Time();
					}
					//2, 3, 1 -> next is 2
					else if (time_demag1 > time_demag2) {

						pHdemag = &Hdemag2;
						time_demag2 = pMeshCUDA->Get_EvalStep_Time();
					}
					//3, 1, 2 -> next is 3, leading to 1, 2, 3 again
					else {

						pHdemag = &Hdemag3;
						time_demag3 = pMeshCUDA->Get_EvalStep_Time();
					}
				}
				//LINEAR
				else if (pMeshCUDA->GetEvaluationSpeedup() == 2) {

					//1, 2 -> next is 1
					if (time_demag2 > time_demag1) {

						pHdemag = &Hdemag;
						time_demag1 = pMeshCUDA->Get_EvalStep_Time();
					}
					//2, 1 -> next is 2, leading to 1, 2 again
					else {

						pHdemag = &Hdemag2;
						time_demag2 = pMeshCUDA->Get_EvalStep_Time();
					}
				}
				//STEP
				else {

					pHdemag = &Hdemag;
				}
			}

			//do evaluation
			ZeroEnergy();
			
			///////////////////////////////////////////////////////////////////////////////////////////////
			//MULTIPLE DEVICES DEMAG
			///////////////////////////////////////////////////////////////////////////////////////////////
			if (mGPU.get_num_devices() > 1) {

				///////////////////////////////////////////////////////////////////////////////////////////////
				//Copy M data to linear regions so we can transfer
				///////////////////////////////////////////////////////////////////////////////////////////////

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					pDemagMCUDA[mGPU]->Copy_M_Input_xRegion(mGPU.get_halfprecision_transfer());
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				//Transfer data between devices before x FFT
				///////////////////////////////////////////////////////////////////////////////////////////////
				
				if (mGPU.get_halfprecision_transfer()) {

					for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
						for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

							if (device_to == device_from) continue;
							M_Input_transfer_half[device_from][device_to]->transfer(device_to, device_from);
						}
					}
					mGPU.synchronize();
				}
				else {
					for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
						for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

							if (device_to == device_from) continue;
							M_Input_transfer[device_from][device_to]->transfer(device_to, device_from);
						}
					}
					mGPU.synchronize();
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				//Forward x FFT for all devices (first step)
				///////////////////////////////////////////////////////////////////////////////////////////////

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

						if (!mGPU.get_halfprecision_transfer())
							pDemagMCUDA[mGPU]->ForwardFFT_mGPU_first(pMeshCUDA->M.get_deviceobject(mGPU), pDemagMCUDA[mGPU]->Real_yRegion_arr, pDemagMCUDA[mGPU]->Complex_yRegion_arr);
						else
							pDemagMCUDA[mGPU]->ForwardFFT_mGPU_first(
								pMeshCUDA->M.get_deviceobject(mGPU),
								pDemagMCUDA[mGPU]->Real_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization_M,
								pDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
					}

					else if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						if (!mGPU.get_halfprecision_transfer())
							pDemagMCUDA[mGPU]->ForwardFFT_AveragedInputs_mGPU_first(pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU), pDemagMCUDA[mGPU]->Real_yRegion_arr, pDemagMCUDA[mGPU]->Complex_yRegion_arr);
						else
							pDemagMCUDA[mGPU]->ForwardFFT_AveragedInputs_mGPU_first(
								pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU),
								pDemagMCUDA[mGPU]->Real_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization_M,
								pDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
					}
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				//Transfer data between devices after x FFT
				///////////////////////////////////////////////////////////////////////////////////////////////
				
				if (mGPU.get_halfprecision_transfer()) {

					for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
						for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

							if (device_to == device_from) continue;
							xFFT_Data_transfer_half[device_from][device_to]->transfer(device_to, device_from);
						}
					}
					mGPU.synchronize();
				}
				else {

					for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
						for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

							if (device_to == device_from) continue;
							xFFT_Data_transfer[device_from][device_to]->transfer(device_to, device_from);
						}
					}
					mGPU.synchronize();
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				//Forward FFT for all devices (last step)
				///////////////////////////////////////////////////////////////////////////////////////////////

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					if (!mGPU.get_halfprecision_transfer())
						pDemagMCUDA[mGPU]->ForwardFFT_mGPU_last(pDemagMCUDA[mGPU]->Complex_xRegion_arr);
					else
						pDemagMCUDA[mGPU]->ForwardFFT_mGPU_last(pDemagMCUDA[mGPU]->Complex_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				//Kernel multiplications
				///////////////////////////////////////////////////////////////////////////////////////////////

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					pDemagMCUDA[mGPU]->KernelMultiplication();
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				//Inverse FFT for all devices
				///////////////////////////////////////////////////////////////////////////////////////////////

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					if (!mGPU.get_halfprecision_transfer())
						pDemagMCUDA[mGPU]->InverseFFT_mGPU_first(pDemagMCUDA[mGPU]->Complex_xRegion_arr);
					else
						pDemagMCUDA[mGPU]->InverseFFT_mGPU_first(pDemagMCUDA[mGPU]->Complex_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				//Transfer data between devices before x IFFT
				///////////////////////////////////////////////////////////////////////////////////////////////

				if (mGPU.get_halfprecision_transfer()) {

					for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
						for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

							if (device_to == device_from) continue;
							xIFFT_Data_transfer_half[device_from][device_to]->transfer(device_to, device_from);
						}
					}
					mGPU.synchronize();
				}
				else {

					for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
						for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

							if (device_to == device_from) continue;
							xIFFT_Data_transfer[device_from][device_to]->transfer(device_to, device_from);
						}
					}
					mGPU.synchronize();
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				//x IFFT
				///////////////////////////////////////////////////////////////////////////////////////////////

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					if (!mGPU.get_halfprecision_transfer())
						pDemagMCUDA[mGPU]->InverseFFT_mGPU_last(pDemagMCUDA[mGPU]->Complex_yRegion_arr, pDemagMCUDA[mGPU]->Real_yRegion_arr);
					else
						pDemagMCUDA[mGPU]->InverseFFT_mGPU_last(pDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization, pDemagMCUDA[mGPU]->Real_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				//Transfer data between devices before finishing
				///////////////////////////////////////////////////////////////////////////////////////////////
				
				if (mGPU.get_halfprecision_transfer()) {

					for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
						for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

							if (device_to == device_from) continue;
							Out_Data_transfer_half[device_from][device_to]->transfer(device_to, device_from);
						}
					}
					mGPU.synchronize();
				}
				else {

					for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
						for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

							if (device_to == device_from) continue;
							Out_Data_transfer[device_from][device_to]->transfer(device_to, device_from);
						}
					}
					mGPU.synchronize();
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				//Finish convolution, setting output in Heff
				///////////////////////////////////////////////////////////////////////////////////////////////

				if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

					for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

						if (!mGPU.get_halfprecision_transfer()) {

							if (Module_Heff.linear_size_cpu()) {

								pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pDemagMCUDA[mGPU]->Real_xRegion_arr,
									pMeshCUDA->M.get_deviceobject(mGPU), pHdemag->get_deviceobject(mGPU),
									energy(mGPU), true, true,
									Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
							}
							else {

								pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pDemagMCUDA[mGPU]->Real_xRegion_arr,
									pMeshCUDA->M.get_deviceobject(mGPU), pHdemag->get_deviceobject(mGPU),
									energy(mGPU), true, true);
							}
						}
						else {
						
							if (Module_Heff.linear_size_cpu()) {

								pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
									pMeshCUDA->M.get_deviceobject(mGPU), pHdemag->get_deviceobject(mGPU),
									energy(mGPU), true, true,
									Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
							}
							else {

								pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
									pMeshCUDA->M.get_deviceobject(mGPU), pHdemag->get_deviceobject(mGPU),
									energy(mGPU), true, true);
							}
						}
					}

					//add contribution to Heff and subtract self demag from *pHDemag
					Demag_EvalSpeedup_AddField_SubSelf(pMeshCUDA->Heff, *pHdemag, pMeshCUDA->M);
				}
				else if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

						if (!mGPU.get_halfprecision_transfer()) {

							if (Module_Heff.linear_size_cpu())
								pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_SingleOutput_mGPU_finish(
									pDemagMCUDA[mGPU]->Real_xRegion_arr,
									pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU),
									pHdemag->get_deviceobject(mGPU),
									energy(mGPU), true, true,
									Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
							else
								pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_SingleOutput_mGPU_finish(
									pDemagMCUDA[mGPU]->Real_xRegion_arr,
									pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU),
									pHdemag->get_deviceobject(mGPU),
									energy(mGPU), true, true);
						}
						else {

							if (Module_Heff.linear_size_cpu())
								pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_SingleOutput_mGPU_finish(
									pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
									pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU),
									pHdemag->get_deviceobject(mGPU),
									energy(mGPU), true, true,
									Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
							else
								pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_SingleOutput_mGPU_finish(
									pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
									pMeshCUDA->M.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU),
									pHdemag->get_deviceobject(mGPU),
									energy(mGPU), true, true);
						}
					}

					//add contribution to Heff and Heff2 and subtract self demag from *pHDemag
					Demag_EvalSpeedup_AddField_SubSelf(pMeshCUDA->Heff, pMeshCUDA->Heff2, *pHdemag, pMeshCUDA->M, pMeshCUDA->M2);
				}
			}
			///////////////////////////////////////////////////////////////////////////////////////////////
			//SINGLE DEVICE DEMAG
			///////////////////////////////////////////////////////////////////////////////////////////////
			else {

				if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

					if (Module_Heff.linear_size_cpu())
						pDemagMCUDA[0]->Convolute(
							pMeshCUDA->M.get_deviceobject(0), pHdemag->get_deviceobject(0),
							energy(0), true, true,
							&Module_Heff.get_deviceobject(0), &Module_energy.get_deviceobject(0));
					else
						pDemagMCUDA[0]->Convolute(
							pMeshCUDA->M.get_deviceobject(0), pHdemag->get_deviceobject(0),
							energy(0), true, true);

					//add contribution to Heff and subtract self demag from *pHDemag
					Demag_EvalSpeedup_AddField_SubSelf(pMeshCUDA->Heff, *pHdemag, pMeshCUDA->M);
				}

				else if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {
					
					if (Module_Heff.linear_size_cpu())
						pDemagMCUDA[0]->Convolute_AveragedInputs(
							pMeshCUDA->M.get_deviceobject(0), pMeshCUDA->M2.get_deviceobject(0), pHdemag->get_deviceobject(0),
							energy(0), true, true,
							&Module_Heff.get_deviceobject(0), &Module_energy.get_deviceobject(0));
					else
						pDemagMCUDA[0]->Convolute_AveragedInputs(
							pMeshCUDA->M.get_deviceobject(0), pMeshCUDA->M2.get_deviceobject(0), pHdemag->get_deviceobject(0),
							energy(0), true, true);

					//add contribution to Heff and Heff2 and subtract self demag from *pHDemag
					Demag_EvalSpeedup_AddField_SubSelf(pMeshCUDA->Heff, pMeshCUDA->Heff2, *pHdemag, pMeshCUDA->M, pMeshCUDA->M2);
				}
			}
		}
		else {

			//not required to update, and we have enough previous evaluations: use previous Hdemag saves to extrapolate for current evaluation

			cuBReal a1 = 1.0, a2 = 0.0, a3 = 0.0, a4 = 0.0, a5 = 0.0, a6 = 0.0;
			cuBReal time = pMeshCUDA->Get_EvalStep_Time();

			//QUINTIC
			if (pMeshCUDA->GetEvaluationSpeedup() == 6) {

				a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4) * (time_demag1 - time_demag5) * (time_demag1 - time_demag6));
				a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4) * (time_demag2 - time_demag5) * (time_demag2 - time_demag6));
				a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4) * (time_demag3 - time_demag5) * (time_demag3 - time_demag6));
				a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag5) * (time - time_demag6) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3) * (time_demag4 - time_demag5) * (time_demag4 - time_demag6));
				a5 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag6) / ((time_demag5 - time_demag1) * (time_demag5 - time_demag2) * (time_demag5 - time_demag3) * (time_demag5 - time_demag4) * (time_demag5 - time_demag6));
				a6 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag6 - time_demag1) * (time_demag6 - time_demag2) * (time_demag6 - time_demag3) * (time_demag6 - time_demag4) * (time_demag6 - time_demag5));

				if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					//add contribution to Heff and Heff2, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, pMeshCUDA->Heff2, a1, a2, a3, a4, a5, a6, pMeshCUDA->M, pMeshCUDA->M2);
				}
				else {

					//add contribution to Heff, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, a1, a2, a3, a4, a5, a6, pMeshCUDA->M);
				}
			}
			//QUARTIC
			else if (pMeshCUDA->GetEvaluationSpeedup() == 5) {

				a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4) * (time_demag1 - time_demag5));
				a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4) * (time_demag2 - time_demag5));
				a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) * (time - time_demag5) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4) * (time_demag3 - time_demag5));
				a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag5) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3) * (time_demag4 - time_demag5));
				a5 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) / ((time_demag5 - time_demag1) * (time_demag5 - time_demag2) * (time_demag5 - time_demag3) * (time_demag5 - time_demag4));

				if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					//add contribution to Heff and Heff2, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, pMeshCUDA->Heff2, a1, a2, a3, a4, a5, pMeshCUDA->M, pMeshCUDA->M2);
				}
				else {

					//add contribution to Heff, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, a1, a2, a3, a4, a5, pMeshCUDA->M);
				}
			}
			//CUBIC
			else if (pMeshCUDA->GetEvaluationSpeedup() == 4) {

				a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4));
				a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4));
				a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4));
				a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3));

				if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					//add contribution to Heff and Heff2, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, pMeshCUDA->Heff2, a1, a2, a3, a4, pMeshCUDA->M, pMeshCUDA->M2);
				}
				else {

					//add contribution to Heff, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, a1, a2, a3, a4, pMeshCUDA->M);
				}
			}
			//QUADRATIC
			else if (pMeshCUDA->GetEvaluationSpeedup() == 3) {

				a1 = (time - time_demag2) * (time - time_demag3) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3));
				a2 = (time - time_demag1) * (time - time_demag3) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3));
				a3 = (time - time_demag1) * (time - time_demag2) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2));

				if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					//add contribution to Heff and Heff2, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, pMeshCUDA->Heff2, a1, a2, a3, pMeshCUDA->M, pMeshCUDA->M2);
				}
				else {

					//add contribution to Heff, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, a1, a2, a3, pMeshCUDA->M);
				}
			}
			//LINEAR
			else if (pMeshCUDA->GetEvaluationSpeedup() == 2) {

				a1 = (time - time_demag2) / (time_demag1 - time_demag2);
				a2 = (time - time_demag1) / (time_demag2 - time_demag1);

				if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					//add contribution to Heff and Heff2, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, pMeshCUDA->Heff2, a1, a2, pMeshCUDA->M, pMeshCUDA->M2);
				}
				else {

					//add contribution to Heff, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, a1, a2, pMeshCUDA->M);
				}
			}
			//STEP
			else {

				if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					//add contribution to Heff and Heff2, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, pMeshCUDA->Heff2, pMeshCUDA->M, pMeshCUDA->M2);
				}
				else {

					//add contribution to Heff, together with self demag
					Demag_EvalSpeedup_AddExtrapField_AddSelf(pMeshCUDA->Heff, pMeshCUDA->M);
				}
			}
		}
	}
}

#endif

#endif