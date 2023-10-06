#include "stdafx.h"
#include "SDemagCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SDEMAG

#include "SuperMesh.h"
#include "SDemag.h"

#include "SDemagCUDA_Demag.h"
#include "SDemagMCUDA_Demag_single.h"

//called from Initialize if using multiconvolution demag
BError SDemagCUDA::Initialize_MConv_Demag(void)
{
	BError error(CLASS_STR(SDemagCUDA));

	//in multi-layered convolution mode must make sure all convolution sizes are set correctly, and rect collections also set
	//SDemag_Demag modules are initialized before SDemag, so they must check if SDemag is not initialized, in which case must call this
	//This will happen in the first SDemag_Demag module to initialize, so after that everything is set correctly to calculate kernels

	/////////////////////////////////////////////////////////////
	//Calculate convolution rectangles collection

	//update common discretisation if needed
	if (pSDemag->use_default_n) pSDemag->set_default_n_common();

	//make sure Rect_collection is correct
	pSDemag->set_Rect_collection();

	//rectangle collection copy from SDemag
	Rect_collection.resize(pSDemag->Rect_collection.size());

	for (int idx = 0; idx < pSDemag->Rect_collection.size(); idx++) {

		Rect_collection[idx] = (cuRect)pSDemag->Rect_collection[idx];
	}

	/////////////////////////////////////////////////////////////
	//Setup convolution dimensions in individual mesh modules

	cuBReal h_max = (cuBReal)pSDemag->get_maximum_cellsize();

	for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

		error = pSDemagCUDA_Demag[idx]->Set_Convolution_Dimensions(h_max, (cuSZ3)pSDemag->n_common, pSDemag->Get_PBC(), Rect_collection, idx);
	}

	//calculate total_nonempty_volume from all meshes participating in convolution
	total_nonempty_volume = 0.0;

	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled() && !(*pSMesh)[idx]->Get_Demag_Exclusion()) {

			total_nonempty_volume += pSMesh->pMesh[idx]->Get_NonEmpty_Magnetic_Volume();
		}
	}

	//now everything is set correctly, ready to calculate demag kernel collections

	return error;
}

//called from UpdateConfiguration if using multiconvolution demag
BError SDemagCUDA::UpdateConfiguration_MConv_Demag(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SDemagCUDA));

	//rebuild kernel collection, and pSDemagCUDA_Demag
	pSDemagCUDA_Demag.resize(pSDemag->pSDemag_Demag.size());

	kernel_collection.resize(mGPU.get_num_devices());
	FFT_Spaces_x_Input.resize(mGPU.get_num_devices());
	FFT_Spaces_y_Input.resize(mGPU.get_num_devices());
	FFT_Spaces_z_Input.resize(mGPU.get_num_devices());
	for (int gpu_idx = 0; gpu_idx < mGPU.get_num_devices(); gpu_idx++) {

		FFT_Spaces_x_Input[gpu_idx].resize(pSDemag->pSDemag_Demag.size());
		FFT_Spaces_y_Input[gpu_idx].resize(pSDemag->pSDemag_Demag.size());
		FFT_Spaces_z_Input[gpu_idx].resize(pSDemag->pSDemag_Demag.size());
		kernel_collection[gpu_idx].resize(pSDemag->pSDemag_Demag.size());
	}

	for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

		pSDemagCUDA_Demag[idx] = dynamic_cast<SDemagCUDA_Demag*>(pSDemag->pSDemag_Demag[idx]->pModuleCUDA);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			kernel_collection[mGPU][idx] = dynamic_cast<DemagKernelCollectionCUDA*>(pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]);

			FFT_Spaces_x_Input[mGPU][idx] = pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Get_Input_Scratch_Space_x();
			FFT_Spaces_y_Input[mGPU][idx] = pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Get_Input_Scratch_Space_y();
			FFT_Spaces_z_Input[mGPU][idx] = pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Get_Input_Scratch_Space_z();
		}
	}

	//mirror SDemag initialization flag
	initialized &= pSDemag->IsInitialized();

	//If SDemagCUDA or any SDemagCUDA_Demag modules are uninitialized, then Uninitialize all SDemagCUDA_Demag modules
	for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

		initialized &= pSDemagCUDA_Demag[idx]->IsInitialized();
	}

	if (!initialized) UninitializeAll();

	return error;
}

//called from UpdateField if using multiconvolution demag
void SDemagCUDA::UpdateField_MConv_Demag(void)
{
	bool eval_speedup = EvalSpeedupCUDA::Check_if_EvalSpeedup(pSMesh->GetEvaluationSpeedup(), pSMesh->Check_Step_Update());

	//for all meshes perform first part of convolution
	std::function<void(void)> start_convolution_all = [&](void) -> void {

		///////////////////////////////////////////////////////////////////////////////////////////////
		//MULTIPLE DEVICES DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (mGPU.get_num_devices() > 1) {

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Copy M data to linear regions so we can transfer
			///////////////////////////////////////////////////////////////////////////////////////////////

			//first transfer to common meshing
			for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

				if (pSDemagCUDA_Demag[idx]->do_transfer) {

					if (pSDemagCUDA_Demag[idx]->pMeshBaseCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) pSDemagCUDA_Demag[idx]->transfer.transfer_in_averaged();
					else pSDemagCUDA_Demag[idx]->transfer.transfer_in();
				}
			}

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

					pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Copy_M_Input_xRegion(mGPU.get_halfprecision_transfer());
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Transfer data between devices before x FFT
			///////////////////////////////////////////////////////////////////////////////////////////////

			if (mGPU.get_halfprecision_transfer()) {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;

						for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

							pSDemagCUDA_Demag[idx]->M_Input_transfer_half[device_from][device_to]->transfer(device_to, device_from);
						}
					}
				}
				mGPU.synchronize();
			}
			else {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

							pSDemagCUDA_Demag[idx]->M_Input_transfer[device_from][device_to]->transfer(device_to, device_from);
						}
					}
				}
				mGPU.synchronize();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Forward x FFT for all devices (first step)
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

					if (!mGPU.get_halfprecision_transfer()) {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->ForwardFFT_mGPU_first(
							(pSDemagCUDA_Demag[idx]->do_transfer ? pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(mGPU) : pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU)),
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_yRegion_arr,
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Complex_yRegion_arr);
					}
					else {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->ForwardFFT_mGPU_first(
							(pSDemagCUDA_Demag[idx]->do_transfer ? pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(mGPU) : pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU)),
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_yRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization_M,
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization);
					}
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Transfer data between devices after x FFT
			///////////////////////////////////////////////////////////////////////////////////////////////

			if (mGPU.get_halfprecision_transfer()) {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;

						for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

							pSDemagCUDA_Demag[idx]->xFFT_Data_transfer_half[device_from][device_to]->transfer(device_to, device_from);
						}
					}
				}
				mGPU.synchronize();
			}
			else {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;

						for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

							pSDemagCUDA_Demag[idx]->xFFT_Data_transfer[device_from][device_to]->transfer(device_to, device_from);
						}
					}
				}
				mGPU.synchronize();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Forward FFT for all devices (last step)
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

					if (!mGPU.get_halfprecision_transfer()) {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->ForwardFFT_mGPU_last(pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Complex_xRegion_arr);
					}
					else {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->ForwardFFT_mGPU_last(
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Complex_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization);
					}
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Kernel multiplications
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

					pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->KernelMultiplication_MultipleInputs(FFT_Spaces_x_Input[mGPU], FFT_Spaces_y_Input[mGPU], FFT_Spaces_z_Input[mGPU]);
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Inverse FFT for all devices
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

					if (!mGPU.get_halfprecision_transfer()) {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_first(pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Complex_xRegion_arr);
					}
					else {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_first(
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Complex_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization);
					}
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Transfer data between devices before x IFFT
			///////////////////////////////////////////////////////////////////////////////////////////////

			if (mGPU.get_halfprecision_transfer()) {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;

						for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

							pSDemagCUDA_Demag[idx]->xIFFT_Data_transfer_half[device_from][device_to]->transfer(device_to, device_from);
						}
					}
				}
				mGPU.synchronize();
			}
			else {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;

						for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

							pSDemagCUDA_Demag[idx]->xIFFT_Data_transfer[device_from][device_to]->transfer(device_to, device_from);
						}
					}
				}
				mGPU.synchronize();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//x IFFT
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

					if (!mGPU.get_halfprecision_transfer()) {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_last(
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Complex_yRegion_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_yRegion_arr);
					}
					else {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_last(
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_yRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization);
					}
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Transfer data between devices before finishing
			///////////////////////////////////////////////////////////////////////////////////////////////

			if (mGPU.get_halfprecision_transfer()) {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;

						for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

							pSDemagCUDA_Demag[idx]->Out_Data_transfer_half[device_from][device_to]->transfer(device_to, device_from);
						}
					}
				}
				mGPU.synchronize();
			}
			else {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;

						for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

							pSDemagCUDA_Demag[idx]->Out_Data_transfer[device_from][device_to]->transfer(device_to, device_from);
						}
					}
				}
				mGPU.synchronize();
			}
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		//SINGLE DEVICE DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////
		else {

			//Forward FFT for all magnetic meshes
			for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

				///////////////////////////////////////////////////////////////////////////////////////////////
				//////////////////////////////////// ANTIFERROMAGNETIC MESH ///////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////////////

				if (pSDemagCUDA_Demag[idx]->pMeshBaseCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					if (pSDemagCUDA_Demag[idx]->do_transfer) {

						//transfer from M to common meshing
						pSDemagCUDA_Demag[idx]->transfer.transfer_in_averaged();

						//do forward FFT
						pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->ForwardFFT(pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(0));
					}
					else {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->ForwardFFT_AveragedInputs(pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(0), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(0));
					}
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////// OTHER MAGNETIC MESH /////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////////////

				else {

					if (pSDemagCUDA_Demag[idx]->do_transfer) {

						//transfer from M to common meshing
						pSDemagCUDA_Demag[idx]->transfer.transfer_in();

						//do forward FFT
						pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->ForwardFFT(pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(0));
					}
					else {

						//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh
						pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->ForwardFFT(pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(0));
					}
				}
			}

			//Kernel multiplications for multiple inputs.
			for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

				pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->KernelMultiplication_MultipleInputs(FFT_Spaces_x_Input[0], FFT_Spaces_y_Input[0], FFT_Spaces_z_Input[0]);
			}
		}
	};

	//finish convolution for mesh with index idx
	std::function<double(mcu_VEC(cuReal3)&, int)> finish_convolution_mesh = [&](mcu_VEC(cuReal3)& H, int idx) -> double {

		double energy_cpu = 0.0;

		///////////////////////////////////////////////////////////////////////////////////////////////
		//MULTIPLE DEVICES DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (mGPU.get_num_devices() > 1) {

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Finish convolution, setting output
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				///////////////////////////////////////////////////////////////////////////////////////////////
				//////////////////////////////////// ANTIFERROMAGNETIC MESH ///////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////////////

				if (pSDemagCUDA_Demag[idx]->pMeshBaseCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					if (pSDemagCUDA_Demag[idx]->do_transfer) {

						if (!mGPU.get_halfprecision_transfer()) {

							if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_arr,
									pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true,
									&pSDemagCUDA_Demag[idx]->transfer_Module_Heff.get_deviceobject(mGPU), &pSDemagCUDA_Demag[idx]->transfer_Module_energy.get_deviceobject(mGPU));

								pSDemagCUDA_Demag[idx]->transfer_Module_Heff.transfer_out();
								pSDemagCUDA_Demag[idx]->transfer_Module_energy.transfer_out();
							}
							else {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_arr,
									pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true);
							}
						}
						else {

							if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
									pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true,
									&pSDemagCUDA_Demag[idx]->transfer_Module_Heff.get_deviceobject(mGPU), &pSDemagCUDA_Demag[idx]->transfer_Module_energy.get_deviceobject(mGPU));

								pSDemagCUDA_Demag[idx]->transfer_Module_Heff.transfer_out();
								pSDemagCUDA_Demag[idx]->transfer_Module_energy.transfer_out();
							}
							else {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
									pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true);
							}
						}
					}
					else {

						if (!eval_speedup) {

							if (!mGPU.get_halfprecision_transfer()) {

								if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_DuplicatedOutputs_mGPU_finish(
										pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_arr,
										pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff2.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), false,
										&pSDemagCUDA_Demag[idx]->Module_Heff.get_deviceobject(mGPU), &pSDemagCUDA_Demag[idx]->Module_energy.get_deviceobject(mGPU));
								}
								else {

									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_DuplicatedOutputs_mGPU_finish(
										pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_arr,
										pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff2.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), false);
								}
							}
							else {

								if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_DuplicatedOutputs_mGPU_finish(
										pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
										pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff2.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), false,
										&pSDemagCUDA_Demag[idx]->Module_Heff.get_deviceobject(mGPU), &pSDemagCUDA_Demag[idx]->Module_energy.get_deviceobject(mGPU));
								}
								else {

									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_DuplicatedOutputs_mGPU_finish(
										pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
										pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff2.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), false);
								}
							}
						}
						else {

							if (!mGPU.get_halfprecision_transfer()) {

								if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_SingleOutput_mGPU_finish(
										pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_arr,
										pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(mGPU),
										H.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true,
										&pSDemagCUDA_Demag[idx]->Module_Heff.get_deviceobject(mGPU), &pSDemagCUDA_Demag[idx]->Module_energy.get_deviceobject(mGPU));
								}
								else {

									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_SingleOutput_mGPU_finish(
										pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_arr,
										pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(mGPU),
										H.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true);
								}
							}
							else {

								if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_SingleOutput_mGPU_finish(
										pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
										pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(mGPU),
										H.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true,
										&pSDemagCUDA_Demag[idx]->Module_Heff.get_deviceobject(mGPU), &pSDemagCUDA_Demag[idx]->Module_energy.get_deviceobject(mGPU));
								}
								else {

									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_AveragedInputs_SingleOutput_mGPU_finish(
										pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
										pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(mGPU),
										H.get_deviceobject(mGPU),
										pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true);
								}
							}
						}
					}
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////// OTHER MAGNETIC MESH /////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////////////

				else {

					if (pSDemagCUDA_Demag[idx]->do_transfer) {

						if (!mGPU.get_halfprecision_transfer()) {

							if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_arr,
									pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true,
									&pSDemagCUDA_Demag[idx]->transfer_Module_Heff.get_deviceobject(mGPU), &pSDemagCUDA_Demag[idx]->transfer_Module_energy.get_deviceobject(mGPU));

								pSDemagCUDA_Demag[idx]->transfer_Module_Heff.transfer_out();
								pSDemagCUDA_Demag[idx]->transfer_Module_energy.transfer_out();
							}
							else {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_arr,
									pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true);
							}
						}
						else {

							if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
									pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true,
									&pSDemagCUDA_Demag[idx]->transfer_Module_Heff.get_deviceobject(mGPU), &pSDemagCUDA_Demag[idx]->transfer_Module_energy.get_deviceobject(mGPU));

								pSDemagCUDA_Demag[idx]->transfer_Module_Heff.transfer_out();
								pSDemagCUDA_Demag[idx]->transfer_Module_energy.transfer_out();
							}
							else {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
									pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), true);
							}
						}
					}
					else {

						if (!mGPU.get_halfprecision_transfer()) {

							if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_arr,
									pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), eval_speedup,
									&pSDemagCUDA_Demag[idx]->Module_Heff.get_deviceobject(mGPU), &pSDemagCUDA_Demag[idx]->Module_energy.get_deviceobject(mGPU));
							}
							else {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_arr,
									pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), eval_speedup);
							}
						}
						else {

							if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
									pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), eval_speedup,
									&pSDemagCUDA_Demag[idx]->Module_Heff.get_deviceobject(mGPU), &pSDemagCUDA_Demag[idx]->Module_energy.get_deviceobject(mGPU));
							}
							else {

								pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
									pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagCUDA_Demag[idx]->pDemagMCUDA[mGPU]->normalization,
									pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
									pSDemagCUDA_Demag[idx]->energy(mGPU), pSMesh->CurrentTimeStepSolved(), eval_speedup);
							}
						}
					}
				}
			}

			//transfer to Heff in each mesh
			if (pSDemagCUDA_Demag[idx]->do_transfer) {

				if (pSDemagCUDA_Demag[idx]->pMeshBaseCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) H.transfer_out_duplicated();
				else H.transfer_out();
			}

			if (pSMesh->CurrentTimeStepSolved()) energy_cpu = pSDemagCUDA_Demag[idx]->energy.to_cpu_sum() * pSDemag->pSDemag_Demag[idx]->energy_density_weight;
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		//SINGLE DEVICE DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////

		else {

			//Inverse FFT
			
			///////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////// ANTIFERROMAGNETIC MESH ///////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////////////////////

			if (pSDemagCUDA_Demag[idx]->pMeshBaseCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

				if (pSDemagCUDA_Demag[idx]->do_transfer) {

					//do inverse FFT and accumulate energy
					if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->InverseFFT(
							pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(0),
							H.get_deviceobject(0),
							pSDemagCUDA_Demag[idx]->energy(0), pSMesh->CurrentTimeStepSolved(), true,
							&pSDemagCUDA_Demag[idx]->transfer_Module_Heff.get_deviceobject(0), &pSDemagCUDA_Demag[idx]->transfer_Module_energy.get_deviceobject(0));

						pSDemagCUDA_Demag[idx]->transfer_Module_Heff.transfer_out();
						pSDemagCUDA_Demag[idx]->transfer_Module_energy.transfer_out();
					}
					else {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->InverseFFT(
							pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(0),
							H.get_deviceobject(0),
							pSDemagCUDA_Demag[idx]->energy(0),
							pSMesh->CurrentTimeStepSolved(), true);
					}

					//transfer to Heff in each mesh
					H.transfer_out_duplicated();
				}
				else {

					if (!eval_speedup) {

						if (pSDemag->pSDemag_Demag[idx]->Module_Heff.linear_size()) {

							//do inverse FFT and accumulate energy. Add to Heff in each mesh
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->InverseFFT_AveragedInputs_DuplicatedOutputs(
								pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(0), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(0),
								pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff.get_deviceobject(0), pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff2.get_deviceobject(0),
								pSDemagCUDA_Demag[idx]->energy(0), pSMesh->CurrentTimeStepSolved(), false,
								&pSDemagCUDA_Demag[idx]->Module_Heff.get_deviceobject(0), &pSDemagCUDA_Demag[idx]->Module_energy.get_deviceobject(0));
						}
						else {

							//do inverse FFT and accumulate energy. Add to Heff in each mesh
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->InverseFFT_AveragedInputs_DuplicatedOutputs(
								pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(0), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(0),
								pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff.get_deviceobject(0), pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff2.get_deviceobject(0),
								pSDemagCUDA_Demag[idx]->energy(0),
								pSMesh->CurrentTimeStepSolved(), false);
						}
					}
					else {

						if (pSDemag->pSDemag_Demag[idx]->Module_Heff.linear_size()) {

							//do inverse FFT and accumulate energy. Add to Heff in each mesh
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->InverseFFT_AveragedInputs(
								pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(0), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(0),
								H.get_deviceobject(0),
								pSDemagCUDA_Demag[idx]->energy(0), pSMesh->CurrentTimeStepSolved(), true,
								&pSDemagCUDA_Demag[idx]->Module_Heff.get_deviceobject(0), &pSDemagCUDA_Demag[idx]->Module_energy.get_deviceobject(0));
						}
						else {

							//do inverse FFT and accumulate energy. Add to Heff in each mesh
							pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->InverseFFT_AveragedInputs(
								pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(0), pSDemagCUDA_Demag[idx]->pMeshCUDA->M2.get_deviceobject(0),
								H.get_deviceobject(0),
								pSDemagCUDA_Demag[idx]->energy(0),
								pSMesh->CurrentTimeStepSolved(), true);
						}
					}
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			///////////////////////////////////// OTHER MAGNETIC MESH /////////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////////////////////

			else {

				if (pSDemagCUDA_Demag[idx]->do_transfer) {

					//do inverse FFT and accumulate energy
					if (pSDemag->pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->InverseFFT(
							pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(0),
							H.get_deviceobject(0),
							pSDemagCUDA_Demag[idx]->energy(0),
							pSMesh->CurrentTimeStepSolved(), true,
							&pSDemagCUDA_Demag[idx]->transfer_Module_Heff.get_deviceobject(0), &pSDemagCUDA_Demag[idx]->transfer_Module_energy.get_deviceobject(0));

						pSDemagCUDA_Demag[idx]->transfer_Module_Heff.transfer_out();
						pSDemagCUDA_Demag[idx]->transfer_Module_energy.transfer_out();
					}
					else {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->InverseFFT(
							pSDemagCUDA_Demag[idx]->transfer.get_deviceobject(0),
							H.get_deviceobject(0),
							pSDemagCUDA_Demag[idx]->energy(0),
							pSMesh->CurrentTimeStepSolved(), true);
					}

					//transfer to Heff in each mesh
					H.transfer_out();
				}
				else {

					//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh

					//do inverse FFT and accumulate energy. Add to Heff in each mesh
					if (pSDemag->pSDemag_Demag[idx]->Module_Heff.linear_size()) {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->InverseFFT(
							pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(0),
							H.get_deviceobject(0),
							pSDemagCUDA_Demag[idx]->energy(0),
							pSMesh->CurrentTimeStepSolved(), eval_speedup,
							&pSDemagCUDA_Demag[idx]->Module_Heff.get_deviceobject(0), &pSDemagCUDA_Demag[idx]->Module_energy.get_deviceobject(0));
					}
					else {

						pSDemagCUDA_Demag[idx]->pDemagMCUDA[0]->InverseFFT(
							pSDemagCUDA_Demag[idx]->pMeshCUDA->M.get_deviceobject(0),
							H.get_deviceobject(0),
							pSDemagCUDA_Demag[idx]->energy(0),
							pSMesh->CurrentTimeStepSolved(), eval_speedup);
					}
				}
			}

			if (pSMesh->CurrentTimeStepSolved()) energy_cpu = pSDemagCUDA_Demag[idx]->energy.to_cpu_sum() * pSDemag->pSDemag_Demag[idx]->energy_density_weight;
		}

		return energy_cpu;
	};

	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////// NO SPEEDUP //////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (!eval_speedup) {

		start_convolution_all();

		double energy_cpu = 0.0;
		for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

			if (pSDemagCUDA_Demag[idx]->do_transfer) energy_cpu += finish_convolution_mesh(pSDemagCUDA_Demag[idx]->transfer, idx);
			//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh
			else energy_cpu += finish_convolution_mesh(pSDemagCUDA_Demag[idx]->pMeshCUDA->Heff, idx);
		}

		if (pSMesh->CurrentTimeStepSolved()) energy.from_cpu(energy_cpu);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////// EVAL SPEEDUP /////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	else {

		//update if required by ODE solver or if we don't have enough previous evaluations saved to extrapolate
		if (pSMesh->Check_Step_Update() || num_Hdemag_saved < pSMesh->GetEvaluationSpeedup()) {

			start_convolution_all();

			double energy_cpu = 0.0;
			for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

				mcu_VEC(cuReal3)* pHdemag = pSDemagCUDA_Demag[idx]->UpdateField_EvalSpeedup_MConv_Start(pSMesh->GetEvaluationSpeedup(), pSMesh->Check_Step_Update(), pSMesh->Get_EvalStep_Time());

				if (pHdemag) {

					energy_cpu += finish_convolution_mesh(*pHdemag, idx);

					pSDemagCUDA_Demag[idx]->UpdateField_EvalSpeedup_MConv_Finish(pSMesh->GetEvaluationSpeedup(), pSDemagCUDA_Demag[idx]->do_transfer, pHdemag, pSDemagCUDA_Demag[idx]->transfer);
				}
			}

			if (pSMesh->CurrentTimeStepSolved()) energy.from_cpu(energy_cpu);

			num_Hdemag_saved = pSDemagCUDA_Demag[0]->num_Hdemag_saved;
		}
		//not required to update, and we have enough previous evaluations: use previous Hdemag saves to extrapolate for current evaluation
		else {

			for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

				pSDemagCUDA_Demag[idx]->UpdateField_EvalSpeedup_MConv_Extrap(
					pSMesh->GetEvaluationSpeedup(), pSMesh->Get_EvalStep_Time(),
					(pSDemagCUDA_Demag[idx]->do_transfer ? &pSDemagCUDA_Demag[idx]->transfer : nullptr));
			}
		}
	}
}

#endif

#endif