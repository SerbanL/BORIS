#include "stdafx.h"
#include "SDemagCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SDEMAG

#include "SDemagMCUDA_single.h"

#include "SuperMesh.h"
#include "SDemag.h"

//construct objects for supermesh demag
void SDemagCUDA::Make_SMesh_Demag(void)
{
	//only make this if not already constructed
	if (pSDemagMCUDA.size() == mGPU.get_num_devices()) return;

	//at this point pSDemagMCUDA size should be zero, but if not, destroy everything first before remaking
	if (pSDemagMCUDA.size()) Clear_SMesh_Demag();

	//make SDemagMCUDA_single objects
	pSDemagMCUDA.resize(mGPU.get_num_devices());

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
		pSDemagMCUDA[idx] = new SDemagMCUDA_single(this, idx);
	}
}

//destruct objects for supermesh demag
void SDemagCUDA::Clear_SMesh_Demag(void)
{
	for (int idx = 0; idx < pSDemagMCUDA.size(); idx++) {

		mGPU.select_device(idx);
		if (pSDemagMCUDA[idx]) delete pSDemagMCUDA[idx];
		pSDemagMCUDA[idx] = nullptr;

		for (int idx_to = 0; idx_to < M_Input_transfer.size(); idx_to++) {

			if (M_Input_transfer[idx][idx_to]) delete M_Input_transfer[idx][idx_to];
			if (M_Input_transfer_half[idx][idx_to]) delete M_Input_transfer_half[idx][idx_to];

			if (xFFT_Data_transfer[idx][idx_to]) delete xFFT_Data_transfer[idx][idx_to];
			if (xFFT_Data_transfer_half[idx][idx_to]) delete xFFT_Data_transfer_half[idx][idx_to];

			if (xIFFT_Data_transfer[idx][idx_to]) delete xIFFT_Data_transfer[idx][idx_to];
			if (xIFFT_Data_transfer_half[idx][idx_to]) delete xIFFT_Data_transfer_half[idx][idx_to];

			if (Out_Data_transfer[idx][idx_to]) delete Out_Data_transfer[idx][idx_to];
			if (Out_Data_transfer_half[idx][idx_to]) delete Out_Data_transfer_half[idx][idx_to];
		}

		M_Input_transfer[idx].clear();
		M_Input_transfer_half[idx].clear();

		xFFT_Data_transfer[idx].clear();
		xFFT_Data_transfer_half[idx].clear();

		Out_Data_transfer[idx].clear();
		Out_Data_transfer_half[idx].clear();
	}

	pSDemagMCUDA.clear();

	M_Input_transfer.clear();
	M_Input_transfer_half.clear();

	xFFT_Data_transfer.clear();
	xFFT_Data_transfer_half.clear();

	Out_Data_transfer.clear();
	Out_Data_transfer_half.clear();

	sm_Vals.clear();
}

bool SDemagCUDA::SDemagCUDA_Submodules_Initialized(void)
{
	bool all_initialized = true;

	for (int idx = 0; idx < pSDemagMCUDA.size(); idx++) {

		all_initialized &= pSDemagMCUDA[idx]->initialized;
	}

	return all_initialized;
}

//called from Initialize if using SMesh demag
BError SDemagCUDA::Initialize_SMesh_Demag(void)
{
	BError error(CLASS_STR(SDemagCUDA));

	//make sure Supermesh demag is actually created
	Make_SMesh_Demag();

	if (!sm_Vals.resize(pSMesh->h_fm, pSMesh->sMeshRect_fm)) return error(BERROR_OUTOFGPUMEMORY_CRIT);

	//number of cells along x must be greater or equal to number of devices used (x partitioning used)
	if (sm_Vals.n.x < mGPU.get_num_devices()) {

		Uninitialize();
		return error(BERROR_MGPU_XCELLS);
	}

	
	/////////////////////////////////////////////////////////////
	//Eval speedup

	/////////////////////////////////////////////////////////////
	// Make mesh transfers in sm_Vals

	//value used by SDemagMCUDA_single modules for half-precision transfer data normalization (set as largest Ms from all participating meshes)
	normalization_Ms = 1.0;

	//array of pointers to input meshes (M) and oputput meshes (Heff) to transfer from and to
	std::vector<mcu_VEC_VC(cuReal3)*> pVal_from, pVal_from2;
	std::vector<mcu_VEC(cuReal3)*> pVal_to, pVal_to2;
	//atomistic meshes input / output
	std::vector<mcu_VEC_VC(cuReal3)*> pVal_afrom;
	std::vector<mcu_VEC(cuReal3)*> pVal_ato;

	//identify all existing magnetic meshes (magnetic computation enabled)
	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled()) {

			if (!(*pSMesh)[idx]->is_atomistic()) {

				//micromagnetic mesh

				pVal_from.push_back(&dynamic_cast<Mesh*>((*pSMesh)[idx])->pMeshCUDA->M);
				pVal_to.push_back(&dynamic_cast<Mesh*>((*pSMesh)[idx])->pMeshCUDA->Heff);

				pVal_from2.push_back(&dynamic_cast<Mesh*>((*pSMesh)[idx])->pMeshCUDA->M2);
				pVal_to2.push_back(&dynamic_cast<Mesh*>((*pSMesh)[idx])->pMeshCUDA->Heff2);

				if ((*pSMesh)[idx]->GetMeshType() == MESH_FERROMAGNETIC) {

					cuBReal meshMs = dynamic_cast<Mesh*>((*pSMesh)[idx])->Ms.get0();
					normalization_Ms = (normalization_Ms > meshMs ? normalization_Ms : meshMs);
				}
				else if ((*pSMesh)[idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					cuBReal meshMs = dynamic_cast<Mesh*>((*pSMesh)[idx])->Ms_AFM.get0().norm();
					normalization_Ms = (normalization_Ms > meshMs ? normalization_Ms : meshMs);
				}
			}
			else {

				//atomistic mesh

				pVal_afrom.push_back(&dynamic_cast<Atom_Mesh*>((*pSMesh)[idx])->paMeshCUDA->M1);
				pVal_ato.push_back(&dynamic_cast<Atom_Mesh*>((*pSMesh)[idx])->paMeshCUDA->Heff1);

				cuBReal meshMs = dynamic_cast<Atom_Mesh*>((*pSMesh)[idx])->Show_Ms();
				normalization_Ms = (normalization_Ms > meshMs ? normalization_Ms : meshMs);
			}
		}
	}

	//Initialize the cpu mesh transfer object - note, SDemag::initialized is set true, but SDemag is not properly initialized - hence need the SDemagCUDA destructor to uninitialize SDemag
	error = pSDemag->Initialize_Mesh_Transfer();
	if (error) return error;

	if (pVal_from.size()) {

		///////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////// ALL FERROMAGNETIC MESHES //////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (!pSDemag->antiferromagnetic_meshes_present) {

			//Now copy mesh transfer object to cuda version
			if (!sm_Vals.copy_transfer_info<cuVEC_VC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>
				(pVal_from, pVal_to, pSDemag->sm_Vals.get_transfer())) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////// AT LEAST ONE ANTIFERROMAGNETIC MESH ///////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		else {

			//Now copy mesh transfer object to cuda version
			if (!sm_Vals.copy_transfer_info_averagedinputs_duplicatedoutputs<cuVEC_VC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>
				(pVal_from, pVal_from2, pVal_to, pVal_to2, pSDemag->sm_Vals.get_transfer())) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////// ATOMISTIC MESHES //////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (pVal_afrom.size()) {

		//Now copy mesh transfer object to cuda version
		if (!sm_Vals.copy_transfer2_info<cuVEC_VC<cuReal3>, cuVEC<cuReal3>, Transfer<DBL3>>
			(pVal_afrom, pVal_ato, pSDemag->sm_Vals.get_transfer2())) return error(BERROR_OUTOFGPUMEMORY_CRIT);
	}

	//initialize eval speedup if needed
	if (pSMesh->GetEvaluationSpeedup()) {

		EvalSpeedupCUDA::Initialize_EvalSpeedup(
			DemagTFunc().SelfDemag_PBC(pSMesh->h_fm, pSMesh->n_fm, pSDemag->Get_PBC()),
			pSMesh->GetEvaluationSpeedup(),
			pSMesh->h_fm, pSMesh->sMeshRect_fm,
			pVal_to, (pSDemag->antiferromagnetic_meshes_present ? pVal_to2 : std::vector<mcu_VEC(cuReal3)*>{}), &pSDemag->sm_Vals.get_transfer(),
			(pVal_afrom.size() ? pVal_ato : std::vector<mcu_VEC(cuReal3)*>{}), (pVal_afrom.size() ? &pSDemag->sm_Vals.get_transfer2() : nullptr));

		EvalSpeedupCUDA::Initialize_EvalSpeedup_Mode_Atom(sm_Vals, sm_Vals);
	}

	/////////////////////////////////////////////////////////////
	//Dimensions

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (!pSDemagMCUDA[mGPU]->CheckDimensions(pSMesh->n_fm, pSMesh->h_fm, pSDemag->Get_PBC())) {

			//the convolution x region is determined by the M mcuVEC partitioning along the x axis
			cuBox dbox = sm_Vals.device_box(mGPU);
			cuINT2 xRegion = cuINT2(dbox.s.x, dbox.e.x);

			//Set convolution dimensions and required PBC conditions
			error = pSDemagMCUDA[mGPU]->SetDimensions(pSMesh->n_fm, pSMesh->h_fm, CONV_SINGLEMESH, pSDemag->Get_PBC(), xRegion, { mGPU, mGPU.get_num_devices() });
		}
	}

	//initialize each DemagMCUDA_single object (one per device) - kernel collections are calculated here
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		error = pSDemagMCUDA[mGPU]->Initialize();
	}

	if (!SDemagCUDA_Submodules_Initialized()) return error;

	/////////////////////////////////////////////////////////////
	//Transfer objects

	if (mGPU.get_num_devices() > 1) {

		for (int idx_from = 0; idx_from < mGPU.get_num_devices(); idx_from++) {
			for (int idx_to = 0; idx_to < mGPU.get_num_devices(); idx_to++) {

				if (idx_to == idx_from) continue;

				if (!mGPU.get_halfprecision_transfer()) {

					M_Input_transfer[idx_from][idx_to]->set_transfer_size(pSDemagMCUDA[idx_from]->Real_xRegion[idx_to]->size());
					M_Input_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pSDemagMCUDA[idx_from]->Real_xRegion[idx_to]->get_array());
					M_Input_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pSDemagMCUDA[idx_to]->Real_yRegion[idx_from]->get_array());

					xFFT_Data_transfer[idx_from][idx_to]->set_transfer_size(pSDemagMCUDA[idx_from]->Complex_yRegion[idx_to]->size());
					xFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pSDemagMCUDA[idx_from]->Complex_yRegion[idx_to]->get_array());
					xFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pSDemagMCUDA[idx_to]->Complex_xRegion[idx_from]->get_array());

					xIFFT_Data_transfer[idx_from][idx_to]->set_transfer_size(pSDemagMCUDA[idx_from]->Complex_xRegion[idx_to]->size());
					xIFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pSDemagMCUDA[idx_from]->Complex_xRegion[idx_to]->get_array());
					xIFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pSDemagMCUDA[idx_to]->Complex_yRegion[idx_from]->get_array());

					Out_Data_transfer[idx_from][idx_to]->set_transfer_size(pSDemagMCUDA[idx_from]->Real_yRegion[idx_to]->size());
					Out_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pSDemagMCUDA[idx_from]->Real_yRegion[idx_to]->get_array());
					Out_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pSDemagMCUDA[idx_to]->Real_xRegion[idx_from]->get_array());
				}
				else {

					M_Input_transfer_half[idx_from][idx_to]->set_transfer_size(pSDemagMCUDA[idx_from]->Real_xRegion_half[idx_to]->size());
					M_Input_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pSDemagMCUDA[idx_from]->Real_xRegion_half[idx_to]->get_array());
					M_Input_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pSDemagMCUDA[idx_to]->Real_yRegion_half[idx_from]->get_array());

					xFFT_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pSDemagMCUDA[idx_from]->Complex_yRegion_half[idx_to]->size());
					xFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pSDemagMCUDA[idx_from]->Complex_yRegion_half[idx_to]->get_array());
					xFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pSDemagMCUDA[idx_to]->Complex_xRegion_half[idx_from]->get_array());

					xIFFT_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pSDemagMCUDA[idx_from]->Complex_xRegion_half[idx_to]->size());
					xIFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pSDemagMCUDA[idx_from]->Complex_xRegion_half[idx_to]->get_array());
					xIFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pSDemagMCUDA[idx_to]->Complex_yRegion_half[idx_from]->get_array());

					Out_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pSDemagMCUDA[idx_from]->Real_yRegion_half[idx_to]->size());
					Out_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pSDemagMCUDA[idx_from]->Real_yRegion_half[idx_to]->get_array());
					Out_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pSDemagMCUDA[idx_to]->Real_xRegion_half[idx_from]->get_array());
				}
			}
		}
	}

	return error;
}

//called from UpdateConfiguration if using Smesh demag
BError SDemagCUDA::UpdateConfiguration_SMesh_Demag(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SDemagCUDA));

	//update configuration for all submodules
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		error = pSDemagMCUDA[mGPU]->UpdateConfiguration(cfgMessage);
	}

	//uninitialize this module also if not all submodules are still all initialized
	if (!SDemagCUDA_Submodules_Initialized()) {

		Uninitialize();
	}

	//num_Hdemag_saved = 0;

	return error;
}

//called from UpdateField if using Smesh demag
void SDemagCUDA::UpdateField_SMesh_Demag(void)
{
	std::function<void(mcu_VEC(cuReal3)&)> do_evaluation = [&](mcu_VEC(cuReal3)& H) -> void {

		if (pSMesh->CurrentTimeStepSolved()) ZeroEnergy();

		///////////////////////////////////////////////////////////////////////////////////////////////
		//MULTIPLE DEVICES DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (mGPU.get_num_devices() > 1) {

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Copy M data to linear regions so we can transfer
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				pSDemagMCUDA[mGPU]->Copy_M_Input_xRegion(mGPU.get_halfprecision_transfer());
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

				if (!mGPU.get_halfprecision_transfer())
					pSDemagMCUDA[mGPU]->ForwardFFT_mGPU_first(sm_Vals.get_deviceobject(mGPU), pSDemagMCUDA[mGPU]->Real_yRegion_arr, pSDemagMCUDA[mGPU]->Complex_yRegion_arr);
				else
					pSDemagMCUDA[mGPU]->ForwardFFT_mGPU_first(
						sm_Vals.get_deviceobject(mGPU),
						pSDemagMCUDA[mGPU]->Real_yRegion_half_arr, pSDemagMCUDA[mGPU]->normalization_M,
						pSDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pSDemagMCUDA[mGPU]->normalization);
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
					pSDemagMCUDA[mGPU]->ForwardFFT_mGPU_last(pSDemagMCUDA[mGPU]->Complex_xRegion_arr);
				else
					pSDemagMCUDA[mGPU]->ForwardFFT_mGPU_last(pSDemagMCUDA[mGPU]->Complex_xRegion_half_arr, pSDemagMCUDA[mGPU]->normalization);
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Kernel multiplications
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				pSDemagMCUDA[mGPU]->KernelMultiplication();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Inverse FFT for all devices
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (!mGPU.get_halfprecision_transfer())
					pSDemagMCUDA[mGPU]->InverseFFT_mGPU_first(pSDemagMCUDA[mGPU]->Complex_xRegion_arr);
				else
					pSDemagMCUDA[mGPU]->InverseFFT_mGPU_first(pSDemagMCUDA[mGPU]->Complex_xRegion_half_arr, pSDemagMCUDA[mGPU]->normalization);
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
					pSDemagMCUDA[mGPU]->InverseFFT_mGPU_last(pSDemagMCUDA[mGPU]->Complex_yRegion_arr, pSDemagMCUDA[mGPU]->Real_yRegion_arr);
				else
					pSDemagMCUDA[mGPU]->InverseFFT_mGPU_last(pSDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pSDemagMCUDA[mGPU]->normalization, pSDemagMCUDA[mGPU]->Real_yRegion_half_arr, pSDemagMCUDA[mGPU]->normalization);
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
			//Finish convolution, setting output
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (!mGPU.get_halfprecision_transfer()) {

					pSDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
						pSDemagMCUDA[mGPU]->Real_xRegion_arr,
						sm_Vals.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
						energy(mGPU), pSMesh->CurrentTimeStepSolved(), true);
				}
				else {

					pSDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
						pSDemagMCUDA[mGPU]->Real_xRegion_half_arr, pSDemagMCUDA[mGPU]->normalization,
						sm_Vals.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
						energy(mGPU), pSMesh->CurrentTimeStepSolved(), true);
				}
			}
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		//SINGLE DEVICE DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////

		else {

			pSDemagMCUDA[0]->Convolute(
				sm_Vals.get_deviceobject(0), H.get_deviceobject(0),
				energy(0), pSMesh->CurrentTimeStepSolved(), true);
		}
	};

	std::function<void(void)> do_transfer_in = [&](void) -> void {

		///////////////////////////////////////////////////////////////////////////////////////////////
		//Transfer in to supermesh
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (!pSDemag->antiferromagnetic_meshes_present) {

			//transfer values from invidual M meshes to sm_Vals
			if (pSDemag->sm_Vals.size_transfer_in()) sm_Vals.transfer_in();
			//transfer from atomistic mesh (if any) - clear input only if there was no transfer from micromagnetic meshes, else add in
			if (pSDemag->sm_Vals.size_transfer2_in()) sm_Vals.transfer2_in(pSDemag->sm_Vals.size_transfer_in() == 0);
		}
		else {

			//transfer values from invidual M meshes to sm_Vals
			if (pSDemag->sm_Vals.size_transfer_in()) sm_Vals.transfer_in_averaged();
			//transfer from atomistic mesh (if any) - clear input only if there was no transfer from micromagnetic meshes, else add in
			if (pSDemag->sm_Vals.size_transfer2_in()) sm_Vals.transfer2_in(pSDemag->sm_Vals.size_transfer_in() == 0);
		}
	};

	std::function<void(mcu_VEC(cuReal3)&)> do_transfer_out = [&](mcu_VEC(cuReal3)& H) -> void {

		///////////////////////////////////////////////////////////////////////////////////////////////
		//Transfer out from supermesh to individual mesh effective fields
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (!pSDemag->antiferromagnetic_meshes_present) {

			//transfer to individual Heff meshes (micromagnetic and atomistc meshes)
			if (pSDemag->sm_Vals.size_transfer_out()) H.transfer_out();
			if (pSDemag->sm_Vals.size_transfer2_out()) H.transfer2_out();
		}
		else {

			//transfer to individual Heff meshes (micromagnetic and atomistc meshes)
			if (pSDemag->sm_Vals.size_transfer_out()) H.transfer_out_duplicated();
			if (pSDemag->sm_Vals.size_transfer2_out()) H.transfer2_out();
		}
	};

	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////// NO SPEEDUP //////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (!EvalSpeedupCUDA::Check_if_EvalSpeedup(pSMesh->GetEvaluationSpeedup(), pSMesh->Check_Step_Update())) {

		do_transfer_in();

		do_evaluation(sm_Vals);

		do_transfer_out(sm_Vals);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////// EVAL SPEEDUP /////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	else {

		EvalSpeedupCUDA::UpdateField_EvalSpeedup(
			pSMesh->GetEvaluationSpeedup(), pSMesh->Check_Step_Update(),
			pSMesh->Get_EvalStep_Time(),
			do_evaluation,
			do_transfer_in, do_transfer_out);
	}
}

#endif

#endif