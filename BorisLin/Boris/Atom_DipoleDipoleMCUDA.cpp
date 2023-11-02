#include "stdafx.h"
#include "Atom_DipoleDipoleMCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_ATOM_DIPOLEDIPOLE) && ATOMISTIC == 1

#include "Atom_DipoleDipoleMCUDA_single.h"

#include "SimScheduleDefs.h"

#include "Atom_MeshCUDA.h"
#include "Atom_Mesh.h"
#include "DataDefs.h"
#include "SuperMesh.h"

Atom_DipoleDipoleMCUDA::Atom_DipoleDipoleMCUDA(Atom_MeshCUDA* paMeshCUDA_) :
	ModulesCUDA(),
	EvalSpeedupCUDA(),
	M(mGPU), Hd(mGPU)
{
	Uninitialize();

	paMeshCUDA = paMeshCUDA_;

	//make DipoleDipoleMCUDA_single objects
	pDipoleDipoleMCUDA.resize(mGPU.get_num_devices());

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
		pDipoleDipoleMCUDA[idx] = new Atom_DipoleDipoleMCUDA_single(paMeshCUDA, this, idx);
	}
}

Atom_DipoleDipoleMCUDA::~Atom_DipoleDipoleMCUDA() 
{
	for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

		mGPU.select_device(idx);
		if (pDipoleDipoleMCUDA[idx]) delete pDipoleDipoleMCUDA[idx];
		pDipoleDipoleMCUDA[idx] = nullptr;

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

BError Atom_DipoleDipoleMCUDA::Initialize(void)
{
	BError error(CLASS_STR(Atom_DipoleDipoleMCUDA));

	//this module only works with subvec axis x (multiple devices)
	if (mGPU.get_num_devices() > 1 && mGPU.get_subvec_axis() != 0) {

		Uninitialize();
		return error(BERROR_MGPU_MUSTBEXAXIS);
	}

	if (!initialized) {

		using_macrocell = (paMeshCUDA->h_dm != paMeshCUDA->h);

		if (using_macrocell) {

			//M and Hd used only if using macrocell
			if (!M.resize(paMeshCUDA->h_dm, paMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (!Hd.resize(paMeshCUDA->h_dm, paMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		}
		else {

			M.clear();
			Hd.clear();
		}

		//number of cells along x must be greater or equal to number of devices used (x partitioning used)
		if (paMeshCUDA->n_dm.x < mGPU.get_num_devices()) {

			Uninitialize();
			return error(BERROR_MGPU_XCELLS);
		}

		/////////////////////////////////////////////////////////////
		//Dimensions

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			if (!pDipoleDipoleMCUDA[mGPU]->CheckDimensions(paMeshCUDA->n_dm, paMeshCUDA->h_dm, paMeshCUDA->M1.Get_PBC())) {

				//the convolution x region is determined by the M mcuVEC partitioning along the x axis
				cuBox dbox = (using_macrocell ? M.device_box(mGPU) : paMeshCUDA->M1.device_box(mGPU));
				cuINT2 xRegion = cuINT2(dbox.s.x, dbox.e.x);

				//Set convolution dimensions and required PBC conditions
				error = pDipoleDipoleMCUDA[mGPU]->SetDimensions(paMeshCUDA->n_dm, paMeshCUDA->h_dm, CONV_SINGLEMESH, paMeshCUDA->M1.Get_PBC(), xRegion, { mGPU, mGPU.get_num_devices() });
			}
		}
		
		//initialize each DipoleDipoleMCUDA_single object (one per device) - kernel collections are calculated here
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = pDipoleDipoleMCUDA[mGPU]->Initialize();
		}

		if (!Submodules_Initialized()) return error;
		
		/////////////////////////////////////////////////////////////
		//Transfer objects

		if (mGPU.get_num_devices() > 1) {

			for (int idx_from = 0; idx_from < mGPU.get_num_devices(); idx_from++) {
				for (int idx_to = 0; idx_to < mGPU.get_num_devices(); idx_to++) {

					if (idx_to == idx_from) continue;

					if (!mGPU.get_halfprecision_transfer()) {

						M_Input_transfer[idx_from][idx_to]->set_transfer_size(pDipoleDipoleMCUDA[idx_from]->Real_xRegion[idx_to]->size());
						M_Input_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDipoleDipoleMCUDA[idx_from]->Real_xRegion[idx_to]->get_array());
						M_Input_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDipoleDipoleMCUDA[idx_to]->Real_yRegion[idx_from]->get_array());

						xFFT_Data_transfer[idx_from][idx_to]->set_transfer_size(pDipoleDipoleMCUDA[idx_from]->Complex_yRegion[idx_to]->size());
						xFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDipoleDipoleMCUDA[idx_from]->Complex_yRegion[idx_to]->get_array());
						xFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDipoleDipoleMCUDA[idx_to]->Complex_xRegion[idx_from]->get_array());

						xIFFT_Data_transfer[idx_from][idx_to]->set_transfer_size(pDipoleDipoleMCUDA[idx_from]->Complex_xRegion[idx_to]->size());
						xIFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDipoleDipoleMCUDA[idx_from]->Complex_xRegion[idx_to]->get_array());
						xIFFT_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDipoleDipoleMCUDA[idx_to]->Complex_yRegion[idx_from]->get_array());

						Out_Data_transfer[idx_from][idx_to]->set_transfer_size(pDipoleDipoleMCUDA[idx_from]->Real_yRegion[idx_to]->size());
						Out_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDipoleDipoleMCUDA[idx_from]->Real_yRegion[idx_to]->get_array());
						Out_Data_transfer[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDipoleDipoleMCUDA[idx_to]->Real_xRegion[idx_from]->get_array());
					}
					else {

						M_Input_transfer_half[idx_from][idx_to]->set_transfer_size(pDipoleDipoleMCUDA[idx_from]->Real_xRegion_half[idx_to]->size());
						M_Input_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDipoleDipoleMCUDA[idx_from]->Real_xRegion_half[idx_to]->get_array());
						M_Input_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDipoleDipoleMCUDA[idx_to]->Real_yRegion_half[idx_from]->get_array());

						xFFT_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pDipoleDipoleMCUDA[idx_from]->Complex_yRegion_half[idx_to]->size());
						xFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDipoleDipoleMCUDA[idx_from]->Complex_yRegion_half[idx_to]->get_array());
						xFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDipoleDipoleMCUDA[idx_to]->Complex_xRegion_half[idx_from]->get_array());

						xIFFT_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pDipoleDipoleMCUDA[idx_from]->Complex_xRegion_half[idx_to]->size());
						xIFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDipoleDipoleMCUDA[idx_from]->Complex_xRegion_half[idx_to]->get_array());
						xIFFT_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDipoleDipoleMCUDA[idx_to]->Complex_yRegion_half[idx_from]->get_array());

						Out_Data_transfer_half[idx_from][idx_to]->set_transfer_size(pDipoleDipoleMCUDA[idx_from]->Real_yRegion_half[idx_to]->size());
						Out_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_from, pDipoleDipoleMCUDA[idx_from]->Real_yRegion_half[idx_to]->get_array());
						Out_Data_transfer_half[idx_from][idx_to]->setup_device_memory_handle(idx_to, pDipoleDipoleMCUDA[idx_to]->Real_xRegion_half[idx_from]->get_array());
					}
				}
			}
		}
		
		/////////////////////////////////////////////////////////////
		//Eval speedup

		if (paMeshCUDA->GetEvaluationSpeedup()) {

			EvalSpeedupCUDA::Initialize_EvalSpeedup(
				(using_macrocell ? DipoleDipoleTFunc().SelfDemag_PBC(paMeshCUDA->h_dm, paMeshCUDA->n_dm, paMeshCUDA->M1.Get_PBC()) : DBL3()),
				paMeshCUDA->GetEvaluationSpeedup(),
				paMeshCUDA->h_dm, paMeshCUDA->meshRect);

			if (using_macrocell) EvalSpeedupCUDA::Initialize_EvalSpeedup_Mode_Atom(M, Hd);
			else EvalSpeedupCUDA::Initialize_EvalSpeedup_Mode_FM(paMeshCUDA->M1, paMeshCUDA->Heff1);
		}

		if (!error) initialized = true;
	}

	EvalSpeedupCUDA::num_Hdemag_saved = 0;

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		paMeshCUDA->h_dm, (cuRect)paMeshCUDA->meshRect,
		(MOD_)paMeshCUDA->Get_Module_Heff_Display() == MOD_ATOM_DIPOLEDIPOLE || paMeshCUDA->IsOutputDataSet_withRect(DATA_E_DEMAG) || paMeshCUDA->IsStageSet(SS_MONTECARLO),
		(MOD_)paMeshCUDA->Get_Module_Energy_Display() == MOD_ATOM_DIPOLEDIPOLE || paMeshCUDA->IsOutputDataSet_withRect(DATA_E_DEMAG) || paMeshCUDA->IsStageSet(SS_MONTECARLO));
	if (error)	initialized = false;

	if (initialized) set_Atom_DipoleDipoleCUDA_pointers();

	//if a Monte Carlo stage is set then we need to compute fields
	if (paMeshCUDA->IsStageSet(SS_MONTECARLO)) paMeshCUDA->Set_Force_MonteCarlo_ComputeFields(true);

	return error;
}

bool Atom_DipoleDipoleMCUDA::Submodules_Initialized(void)
{
	bool all_initialized = true;

	for (int idx = 0; idx < pDipoleDipoleMCUDA.size(); idx++) {

		all_initialized &= pDipoleDipoleMCUDA[idx]->initialized;
	}

	return all_initialized;
}

BError Atom_DipoleDipoleMCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Atom_DipoleDipoleMCUDA));

	//update configuration for all submodules
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		error = pDipoleDipoleMCUDA[mGPU]->UpdateConfiguration(cfgMessage);
	}

	//uninitialize this module also if not all submodules are still all initialized
	if (!Submodules_Initialized()) {

		Uninitialize();
		EvalSpeedupCUDA::UpdateConfiguration_EvalSpeedup();
	}

	return error;
}

void Atom_DipoleDipoleMCUDA::UpdateField(void)
{
	bool eval_speedup = EvalSpeedupCUDA::Check_if_EvalSpeedup(paMeshCUDA->GetEvaluationSpeedup(), paMeshCUDA->Check_Step_Update());

	std::function<void(mcu_VEC(cuReal3)&)> do_evaluation = [&](mcu_VEC(cuReal3)& H) -> void {

		if (paMeshCUDA->CurrentTimeStepSolved()) ZeroEnergy();

		///////////////////////////////////////////////////////////////////////////////////////////////
		//MULTIPLE DEVICES DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////
		if (mGPU.get_num_devices() > 1) {

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Copy M data to linear regions so we can transfer
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				pDipoleDipoleMCUDA[mGPU]->Copy_M_Input_xRegion(mGPU.get_halfprecision_transfer());
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
					pDipoleDipoleMCUDA[mGPU]->ForwardFFT_mGPU_first(
						(using_macrocell ? M.get_deviceobject(mGPU) : paMeshCUDA->M1.get_deviceobject(mGPU)),
						pDipoleDipoleMCUDA[mGPU]->Real_yRegion_arr, pDipoleDipoleMCUDA[mGPU]->Complex_yRegion_arr);
				else
					pDipoleDipoleMCUDA[mGPU]->ForwardFFT_mGPU_first(
						(using_macrocell ? M.get_deviceobject(mGPU) : paMeshCUDA->M1.get_deviceobject(mGPU)),
						pDipoleDipoleMCUDA[mGPU]->Real_yRegion_half_arr, pDipoleDipoleMCUDA[mGPU]->normalization_M,
						pDipoleDipoleMCUDA[mGPU]->Complex_yRegion_half_arr, pDipoleDipoleMCUDA[mGPU]->normalization);
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
					pDipoleDipoleMCUDA[mGPU]->ForwardFFT_mGPU_last(pDipoleDipoleMCUDA[mGPU]->Complex_xRegion_arr);
				else
					pDipoleDipoleMCUDA[mGPU]->ForwardFFT_mGPU_last(pDipoleDipoleMCUDA[mGPU]->Complex_xRegion_half_arr, pDipoleDipoleMCUDA[mGPU]->normalization);
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Kernel multiplications
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				pDipoleDipoleMCUDA[mGPU]->KernelMultiplication();
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			//Inverse FFT for all devices
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (!mGPU.get_halfprecision_transfer())
					pDipoleDipoleMCUDA[mGPU]->InverseFFT_mGPU_first(pDipoleDipoleMCUDA[mGPU]->Complex_xRegion_arr);
				else
					pDipoleDipoleMCUDA[mGPU]->InverseFFT_mGPU_first(pDipoleDipoleMCUDA[mGPU]->Complex_xRegion_half_arr, pDipoleDipoleMCUDA[mGPU]->normalization);
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
					pDipoleDipoleMCUDA[mGPU]->InverseFFT_mGPU_last(pDipoleDipoleMCUDA[mGPU]->Complex_yRegion_arr, pDipoleDipoleMCUDA[mGPU]->Real_yRegion_arr);
				else
					pDipoleDipoleMCUDA[mGPU]->InverseFFT_mGPU_last(pDipoleDipoleMCUDA[mGPU]->Complex_yRegion_half_arr, pDipoleDipoleMCUDA[mGPU]->normalization, pDipoleDipoleMCUDA[mGPU]->Real_yRegion_half_arr, pDipoleDipoleMCUDA[mGPU]->normalization);
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

				if (!mGPU.get_halfprecision_transfer()) {

					if (Module_Heff.linear_size_cpu()) {

						pDipoleDipoleMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDipoleDipoleMCUDA[mGPU]->Real_xRegion_arr,
							(using_macrocell ? M.get_deviceobject(mGPU) : paMeshCUDA->M1.get_deviceobject(mGPU)), H.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), using_macrocell || eval_speedup,
							Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
					}
					else {

						pDipoleDipoleMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDipoleDipoleMCUDA[mGPU]->Real_xRegion_arr,
							(using_macrocell ? M.get_deviceobject(mGPU) : paMeshCUDA->M1.get_deviceobject(mGPU)), H.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), using_macrocell || eval_speedup);
					}
				}
				else {

					if (Module_Heff.linear_size_cpu()) {

						pDipoleDipoleMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDipoleDipoleMCUDA[mGPU]->Real_xRegion_half_arr, pDipoleDipoleMCUDA[mGPU]->normalization,
							(using_macrocell ? M.get_deviceobject(mGPU) : paMeshCUDA->M1.get_deviceobject(mGPU)), H.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), using_macrocell || eval_speedup,
							Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
					}
					else {

						pDipoleDipoleMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDipoleDipoleMCUDA[mGPU]->Real_xRegion_half_arr, pDipoleDipoleMCUDA[mGPU]->normalization,
							(using_macrocell ? M.get_deviceobject(mGPU) : paMeshCUDA->M1.get_deviceobject(mGPU)), H.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), using_macrocell || eval_speedup);
					}
				}
			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////
		//SINGLE DEVICE DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////
		else {

			if (Module_Heff.linear_size_cpu())
				pDipoleDipoleMCUDA[0]->Convolute(
					(using_macrocell ? M.get_deviceobject(0) : paMeshCUDA->M1.get_deviceobject(0)), H.get_deviceobject(0),
					energy(0), paMeshCUDA->CurrentTimeStepSolved(), using_macrocell || eval_speedup,
					&Module_Heff.get_deviceobject(0), &Module_energy.get_deviceobject(0));
			else
				pDipoleDipoleMCUDA[0]->Convolute(
					(using_macrocell ? M.get_deviceobject(0) : paMeshCUDA->M1.get_deviceobject(0)), H.get_deviceobject(0),
					energy(0), paMeshCUDA->CurrentTimeStepSolved(), using_macrocell || eval_speedup);
		}
	};

	std::function<void(void)> do_transfer_in = [&](void) -> void {

		//transfer magnetic moments to macrocell
		if (using_macrocell) Transfer_Moments_to_Macrocell();
	};

	std::function<void(mcu_VEC(cuReal3)&)> do_transfer_out = [&](mcu_VEC(cuReal3)& H) -> void {

		//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the macrocell receive the same field
		if (using_macrocell) Transfer_DipoleDipole_Field(H);
	};

	if (!eval_speedup) {

		///////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////// NO SPEEDUP //////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		//transfer into M
		do_transfer_in();

		//convolution
		do_evaluation(using_macrocell ? Hd : paMeshCUDA->Heff1);

		//transfer from Hd to Heff
		if (using_macrocell) do_transfer_out(Hd);
	}

	else {

		///////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////// EVAL SPEEDUP /////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		EvalSpeedupCUDA::UpdateField_EvalSpeedup(
			paMeshCUDA->GetEvaluationSpeedup(), paMeshCUDA->Check_Step_Update(),
			paMeshCUDA->Get_EvalStep_Time(),
			do_evaluation,
			do_transfer_in, do_transfer_out);
	}
}

#endif

#endif
