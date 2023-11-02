#include "stdafx.h"
#include "OerstedCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_OERSTED

#include "OerstedMCUDA_single.h"

#include "SuperMesh.h"
#include "Oersted.h"

OerstedCUDA::OerstedCUDA(SuperMesh* pSMesh_, Oersted* pOersted_) :
	ModulesCUDA(),
	sm_Vals(mGPU)
{
	Uninitialize();

	pSMesh = pSMesh_;
	pOersted = pOersted_;

	//set from cpu version of sm_vals
	if (!sm_Vals.set_from_cpuvec(pOersted->sm_Vals)) error_on_create(BERROR_OUTOFGPUMEMORY_CRIT);

	//make OerstedMCUDA_single objects
	pOerstedMCUDA.resize(mGPU.get_num_devices());

	//don't use the mGPU for loop construct since mGPU could change whilst making objects below
	for (int idx_from = 0; idx_from < mGPU.get_num_devices(); idx_from++) {

		J_Input_transfer.push_back(std::vector<mGPU_Transfer<cuReal3>*>());
		J_Input_transfer_half.push_back(std::vector<mGPU_Transfer<cuBHalf>*>());

		xFFT_Data_transfer.push_back(std::vector<mGPU_Transfer<cuBComplex>*>());
		xFFT_Data_transfer_half.push_back(std::vector<mGPU_Transfer<cuBHalf>*>());

		xIFFT_Data_transfer.push_back(std::vector<mGPU_Transfer<cuBComplex>*>());
		xIFFT_Data_transfer_half.push_back(std::vector<mGPU_Transfer<cuBHalf>*>());

		Out_Data_transfer.push_back(std::vector<mGPU_Transfer<cuReal3>*>());
		Out_Data_transfer_half.push_back(std::vector<mGPU_Transfer<cuBHalf>*>());

		for (int idx_to = 0; idx_to < mGPU.get_num_devices(); idx_to++) {

			J_Input_transfer[idx_from].push_back(new mGPU_Transfer<cuReal3>(mGPU));
			J_Input_transfer_half[idx_from].push_back(new mGPU_Transfer<cuBHalf>(mGPU));

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
		pOerstedMCUDA[idx] = new OerstedMCUDA_single(this, idx);
	}
}

OerstedCUDA::~OerstedCUDA()
{
	//copy values back to cpu version
	if (Holder_Module_Available()) {

		sm_Vals.copy_to_cpuvec(pOersted->sm_Vals);
	}

	for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

		mGPU.select_device(idx);
		if (pOerstedMCUDA[idx]) delete pOerstedMCUDA[idx];
		pOerstedMCUDA[idx] = nullptr;

		for (int idx_to = 0; idx_to < mGPU.get_num_devices(); idx_to++) {

			if (J_Input_transfer[idx][idx_to]) delete J_Input_transfer[idx][idx_to];
			if (J_Input_transfer_half[idx][idx_to]) delete J_Input_transfer_half[idx][idx_to];

			if (xFFT_Data_transfer[idx][idx_to]) delete xFFT_Data_transfer[idx][idx_to];
			if (xFFT_Data_transfer_half[idx][idx_to]) delete xFFT_Data_transfer_half[idx][idx_to];

			if (xIFFT_Data_transfer[idx][idx_to]) delete xIFFT_Data_transfer[idx][idx_to];
			if (xIFFT_Data_transfer_half[idx][idx_to]) delete xIFFT_Data_transfer_half[idx][idx_to];

			if (Out_Data_transfer[idx][idx_to]) delete Out_Data_transfer[idx][idx_to];
			if (Out_Data_transfer_half[idx][idx_to]) delete Out_Data_transfer_half[idx][idx_to];
		}
	}
}

BError OerstedCUDA::Initialize(void)
{
	BError error(CLASS_STR(OerstedCUDA));

	//this module only works with subvec axis x (multiple devices)
	if (mGPU.get_num_devices() > 1 && mGPU.get_subvec_axis() != 0) {

		Uninitialize();
		return error(BERROR_MGPU_MUSTBEXAXIS);
	}

	//not counting this to the total energy density for now
	ZeroEnergy();

	if (!initialized) {

		if (!sm_Vals.resize(pSMesh->h_e, pSMesh->sMeshRect_e)) return error(BERROR_OUTOFGPUMEMORY_CRIT);

		//number of cells along x must be greater or equal to number of devices used (x partitioning used)
		if (sm_Vals.n.x < mGPU.get_num_devices()) {

			Uninitialize();
			return error(BERROR_MGPU_XCELLS);
		}

		/////////////////////////////////////////////////////////////
		//Dimensions

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			if (!pOerstedMCUDA[mGPU]->CheckDimensions(sm_Vals.n, sm_Vals.h, cuINT3())) {

				//the convolution x region is determined by the mcuVEC partitioning along the x axis
				cuBox dbox = sm_Vals.device_box(mGPU);
				cuINT2 xRegion = cuINT2(dbox.s.x, dbox.e.x);

				//Set convolution dimensions and required PBC conditions
				error = pOerstedMCUDA[mGPU]->SetDimensions(sm_Vals.n, sm_Vals.h, CONV_SINGLEMESH, cuINT3(), xRegion, {mGPU, mGPU.get_num_devices()});
			}
		}

		//initialize each DemagMCUDA_single object (one per device) - kernel collections are calculated here
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = pOerstedMCUDA[mGPU]->Initialize();
		}

		if (!Submodules_Initialized()) return error;

		/////////////////////////////////////////////////////////////
		// Make mesh transfers in sm_Vals

		//always recalculate the mesh transfer as some meshes could have changed
		std::vector< VEC<DBL3>* > pVal_from_cpu_E;
		std::vector< VEC<double>* > pVal_from_cpu_elC;
		std::vector< VEC<DBL3>* > pVal_to_cpu;

		//array of pointers to input meshes (M) and oputput meshes (Heff) to transfer from and to
		std::vector<mcu_VEC_VC(cuReal3)*> pVal_from_E;
		std::vector<mcu_VEC_VC(cuBReal)*> pVal_from_elC;
		std::vector<mcu_VEC(cuReal3)*> pVal_to;

		//identify all existing output magnetic meshes (magnetic computation enabled), and input electric meshes
		for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

			if ((*pSMesh)[idx]->EComputation_Enabled()) {

				pVal_from_cpu_E.push_back(&((*pSMesh)[idx]->E));
				pVal_from_E.push_back(&(*pSMesh)[idx]->pMeshBaseCUDA->E);

				pVal_from_cpu_elC.push_back(&((*pSMesh)[idx]->elC));
				pVal_from_elC.push_back(&(*pSMesh)[idx]->pMeshBaseCUDA->elC);
			}

			if ((*pSMesh)[idx]->MComputation_Enabled()) {

				if (!(*pSMesh)[idx]->is_atomistic()) {

					pVal_to_cpu.push_back(&(dynamic_cast<Mesh*>((*pSMesh)[idx])->Heff));
					pVal_to.push_back(&dynamic_cast<Mesh*>((*pSMesh)[idx])->pMeshCUDA->Heff);
				}
				else {

					pVal_to_cpu.push_back(&(dynamic_cast<Atom_Mesh*>((*pSMesh)[idx])->Heff1));
					pVal_to.push_back(&dynamic_cast<Atom_Mesh*>((*pSMesh)[idx])->paMeshCUDA->Heff1);
				}
			}

			if ((*pSMesh)[idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

				pVal_to_cpu.push_back(&(dynamic_cast<Mesh*>((*pSMesh)[idx])->Heff2));
				pVal_to.push_back(&dynamic_cast<Mesh*>((*pSMesh)[idx])->pMeshCUDA->Heff2);
			}
		}

		//Initialize the mesh transfer object.
		if (!pOersted->sm_Vals.Initialize_MeshTransfer_MultipliedInputs(pVal_from_cpu_E, pVal_from_cpu_elC, pVal_to_cpu, MESHTRANSFERTYPE_WEIGHTED)) return error(BERROR_OUTOFMEMORY_CRIT);

		//Now copy mesh transfer object to cuda version
		if (!sm_Vals.copy_transfer_info_multipliedinputs<cuVEC_VC<cuReal3>, cuVEC_VC<cuBReal>, cuVEC<cuReal3>, Transfer<DBL3>>
			(pVal_from_E, pVal_from_elC, pVal_to, pOersted->sm_Vals.get_transfer())) return error(BERROR_OUTOFGPUMEMORY_CRIT);

		/////////////////////////////////////////////////////////////
		//Transfer objects

		if (mGPU.get_num_devices() > 1) {

			for (int idx_from = 0; idx_from < mGPU.get_num_devices(); idx_from++) {
				for (int idx_to = 0; idx_to < mGPU.get_num_devices(); idx_to++) {

					if (idx_to == idx_from) continue;

					if (!mGPU.get_halfprecision_transfer()) {

						J_Input_transfer[idx_from][idx_to]->set_transfer_size(pOerstedMCUDA[idx_from]->Real_xRegion[idx_to]->size());
						J_Input_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pOerstedMCUDA[idx_from]->Real_xRegion[idx_to]->get_array());
						J_Input_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pOerstedMCUDA[idx_to]->Real_yRegion[idx_from]->get_array());

						xFFT_Data_transfer[idx_from][idx_to]->set_transfer_size(pOerstedMCUDA[idx_from]->Complex_yRegion[idx_to]->size());
						xFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pOerstedMCUDA[idx_from]->Complex_yRegion[idx_to]->get_array());
						xFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pOerstedMCUDA[idx_to]->Complex_xRegion[idx_from]->get_array());

						xIFFT_Data_transfer[idx_from][idx_to]->set_transfer_size(pOerstedMCUDA[idx_from]->Complex_xRegion[idx_to]->size());
						xIFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pOerstedMCUDA[idx_from]->Complex_xRegion[idx_to]->get_array());
						xIFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pOerstedMCUDA[idx_to]->Complex_yRegion[idx_from]->get_array());

						Out_Data_transfer[idx_from][idx_to]->set_transfer_size(pOerstedMCUDA[idx_from]->Real_yRegion[idx_to]->size());
						Out_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pOerstedMCUDA[idx_from]->Real_yRegion[idx_to]->get_array());
						Out_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pOerstedMCUDA[idx_to]->Real_xRegion[idx_from]->get_array());
					}
					else {

						J_Input_transfer_half[idx_from][idx_to]->set_transfer_size(pOerstedMCUDA[idx_from]->Real_xRegion_half[idx_to]->size());
						J_Input_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pOerstedMCUDA[idx_from]->Real_xRegion_half[idx_to]->get_array());
						J_Input_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pOerstedMCUDA[idx_to]->Real_yRegion_half[idx_from]->get_array());

						xFFT_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pOerstedMCUDA[idx_from]->Complex_yRegion_half[idx_to]->size());
						xFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pOerstedMCUDA[idx_from]->Complex_yRegion_half[idx_to]->get_array());
						xFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pOerstedMCUDA[idx_to]->Complex_xRegion_half[idx_from]->get_array());

						xIFFT_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pOerstedMCUDA[idx_from]->Complex_xRegion_half[idx_to]->size());
						xIFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pOerstedMCUDA[idx_from]->Complex_xRegion_half[idx_to]->get_array());
						xIFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pOerstedMCUDA[idx_to]->Complex_yRegion_half[idx_from]->get_array());

						Out_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pOerstedMCUDA[idx_from]->Real_yRegion_half[idx_to]->size());
						Out_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pOerstedMCUDA[idx_from]->Real_yRegion_half[idx_to]->get_array());
						Out_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pOerstedMCUDA[idx_to]->Real_xRegion_half[idx_from]->get_array());
					}
				}
			}
		}

		if (!error) initialized = true;
	}

	return error;
}

bool OerstedCUDA::Submodules_Initialized(void)
{
	bool all_initialized = true;

	for (int idx = 0; idx < pOerstedMCUDA.size(); idx++) {

		all_initialized &= pOerstedMCUDA[idx]->initialized;
	}

	return all_initialized;
}

BError OerstedCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(OerstedCUDA));

	//update configuration for all submodules
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		error = pOerstedMCUDA[mGPU]->UpdateConfiguration(cfgMessage);
	}

	//uninitialize this module also if not all submodules are still all initialized
	if (!Submodules_Initialized() || sm_Vals.n != pSMesh->n_e || sm_Vals.h != pSMesh->h_e || sm_Vals.rect != pSMesh->sMeshRect_e) {

		Uninitialize();
	}

	return error;
}

void OerstedCUDA::UpdateField(void)
{
	//only recalculate Oersted field if there was a significant change in current density (judged based on transport solver iterations)
	if (pSMesh->CallModuleMethod(&STransport::Transport_Recalculated) || !pOersted->oefield_computed) {

		//transfer values from invidual Jc meshes to sm_Vals
		sm_Vals.transfer_in_multiplied();

		//only need energy after ode solver step finished
		if (pSMesh->CurrentTimeStepSolved()) ZeroEnergy();

		///////////////////////////////////////////////////////////////////////////////////////////////
		//MULTIPLE DEVICES CONVOLUTION
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (mGPU.get_num_devices() > 1) {

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Copy J data to linear regions so we can transfer
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				pOerstedMCUDA[mGPU]->Copy_J_Input_xRegion(mGPU.get_halfprecision_transfer());
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Transfer data between devices before x FFT
			///////////////////////////////////////////////////////////////////////////////////////////////

			if (mGPU.get_halfprecision_transfer()) {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						J_Input_transfer_half[device_from][device_to]->transfer(device_to, device_from);
					}
				}
				mGPU.synchronize();
			}
			else {

				for (int device_from = 0; device_from < mGPU.get_num_devices(); device_from++) {
					for (int device_to = 0; device_to < mGPU.get_num_devices(); device_to++) {

						if (device_to == device_from) continue;
						J_Input_transfer[device_from][device_to]->transfer(device_to, device_from);
					}
				}
				mGPU.synchronize();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Forward x FFT for all devices (first step)
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (!mGPU.get_halfprecision_transfer())
					pOerstedMCUDA[mGPU]->ForwardFFT_mGPU_first(sm_Vals.get_deviceobject(mGPU), pOerstedMCUDA[mGPU]->Real_yRegion_arr, pOerstedMCUDA[mGPU]->Complex_yRegion_arr);
				else
					pOerstedMCUDA[mGPU]->ForwardFFT_mGPU_first(
						sm_Vals.get_deviceobject(mGPU),
						pOerstedMCUDA[mGPU]->Real_yRegion_half_arr, pOerstedMCUDA[mGPU]->normalization_J,
						pOerstedMCUDA[mGPU]->Complex_yRegion_half_arr, pOerstedMCUDA[mGPU]->normalization);
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
					pOerstedMCUDA[mGPU]->ForwardFFT_mGPU_last(pOerstedMCUDA[mGPU]->Complex_xRegion_arr);
				else
					pOerstedMCUDA[mGPU]->ForwardFFT_mGPU_last(pOerstedMCUDA[mGPU]->Complex_xRegion_half_arr, pOerstedMCUDA[mGPU]->normalization);
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Kernel multiplications
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				pOerstedMCUDA[mGPU]->KernelMultiplication();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Inverse FFT for all devices
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (!mGPU.get_halfprecision_transfer())
					pOerstedMCUDA[mGPU]->InverseFFT_mGPU_first(pOerstedMCUDA[mGPU]->Complex_xRegion_arr);
				else
					pOerstedMCUDA[mGPU]->InverseFFT_mGPU_first(pOerstedMCUDA[mGPU]->Complex_xRegion_half_arr, pOerstedMCUDA[mGPU]->normalization);
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
					pOerstedMCUDA[mGPU]->InverseFFT_mGPU_last(pOerstedMCUDA[mGPU]->Complex_yRegion_arr, pOerstedMCUDA[mGPU]->Real_yRegion_arr);
				else
					pOerstedMCUDA[mGPU]->InverseFFT_mGPU_last(pOerstedMCUDA[mGPU]->Complex_yRegion_half_arr, pOerstedMCUDA[mGPU]->normalization, pOerstedMCUDA[mGPU]->Real_yRegion_half_arr, pOerstedMCUDA[mGPU]->normalization);
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
			//Finish convolution, setting output in Hd
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (!mGPU.get_halfprecision_transfer()) {

					pOerstedMCUDA[mGPU]->InverseFFT_mGPU_finish(
						pOerstedMCUDA[mGPU]->Real_xRegion_arr,
						sm_Vals.get_deviceobject(mGPU), sm_Vals.get_deviceobject(mGPU),
						energy(mGPU), pSMesh->CurrentTimeStepSolved(), true);
				}
				else {

					pOerstedMCUDA[mGPU]->InverseFFT_mGPU_finish(
						pOerstedMCUDA[mGPU]->Real_xRegion_half_arr, pOerstedMCUDA[mGPU]->normalization,
						sm_Vals.get_deviceobject(mGPU), sm_Vals.get_deviceobject(mGPU),
						energy(mGPU), pSMesh->CurrentTimeStepSolved(), true);
				}
			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////
		//SINGLE DEVICE DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////
		else {

			pOerstedMCUDA[0]->Convolute(
				sm_Vals.get_deviceobject(0), sm_Vals.get_deviceobject(0),
				energy(0), pSMesh->CurrentTimeStepSolved(), true);
		}

		//transfer to individual Heff meshes
		sm_Vals.transfer_out();

		pOersted->oefield_computed = true;
	}
	else {

		//transfer to individual Heff meshes
		sm_Vals.transfer_out();
	}
}

#endif

#endif