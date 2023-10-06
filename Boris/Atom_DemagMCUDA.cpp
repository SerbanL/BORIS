#include "stdafx.h"
#include "Atom_DemagMCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_DEMAG) && ATOMISTIC == 1

#include "Atom_DemagMCUDA_single.h"

#include "SimScheduleDefs.h"

#include "Atom_MeshCUDA.h"
#include "Atom_Mesh.h"
#include "DataDefs.h"
#include "SuperMesh.h"

Atom_DemagMCUDA::Atom_DemagMCUDA(Atom_MeshCUDA* paMeshCUDA_) :
	ModulesCUDA(),
	M(mGPU), Hd(mGPU),
	EvalSpeedupCUDA()
{
	Uninitialize();

	paMeshCUDA = paMeshCUDA_;

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
		pDemagMCUDA[idx] = new Atom_DemagMCUDA_single(paMeshCUDA, this, idx);
	}
}

Atom_DemagMCUDA::~Atom_DemagMCUDA() 
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

BError Atom_DemagMCUDA::Initialize(void)
{
	BError error(CLASS_STR(Atom_DemagMCUDA));

	//this module only works with subvec axis x (multiple devices)
	if (mGPU.get_num_devices() > 1 && mGPU.get_subvec_axis() != 0) {

		Uninitialize();
		return error(BERROR_MGPU_MUSTBEXAXIS);
	}

	if (!initialized) {

		if (!M.resize(paMeshCUDA->h_dm, paMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		if (!Hd.resize(paMeshCUDA->h_dm, paMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);

		//number of macrocells along x must be greater or equal to number of devices used (x partitioning used)
		if (M.n.x < mGPU.get_num_devices()) {

			Uninitialize();
			return error(BERROR_MGPU_XCELLS);
		}

		/////////////////////////////////////////////////////////////
		//Dimensions

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			if (!pDemagMCUDA[mGPU]->CheckDimensions(paMeshCUDA->n_dm, paMeshCUDA->h_dm, paMeshCUDA->M1.Get_PBC())) {

				//the convolution x region is determined by the M mcuVEC partitioning along the x axis
				cuBox dbox = M.device_box(mGPU);
				cuINT2 xRegion = cuINT2(dbox.s.x, dbox.e.x);

				//Set convolution dimensions and required PBC conditions
				error = pDemagMCUDA[mGPU]->SetDimensions(paMeshCUDA->n_dm, paMeshCUDA->h_dm, CONV_SINGLEMESH, paMeshCUDA->M1.Get_PBC(), xRegion, { mGPU, mGPU.get_num_devices() });
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

		//initialize eval speedup if needed
		if (paMeshCUDA->GetEvaluationSpeedup()) {

			EvalSpeedupCUDA::Initialize_EvalSpeedup(
				DemagTFunc().SelfDemag_PBC(paMeshCUDA->h_dm, paMeshCUDA->n_dm, paMeshCUDA->M1.Get_PBC()),
				paMeshCUDA->GetEvaluationSpeedup(),
				paMeshCUDA->h_dm, paMeshCUDA->meshRect);

			EvalSpeedupCUDA::Initialize_EvalSpeedup_Mode_Atom(M, Hd);
		}

		if (!error) initialized = true;
	}
	
	EvalSpeedupCUDA::num_Hdemag_saved = 0;

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		(cuReal3)paMeshCUDA->h_dm, (cuRect)paMeshCUDA->meshRect, 
		(MOD_)paMeshCUDA->Get_Module_Heff_Display() == MOD_DEMAG || paMeshCUDA->IsOutputDataSet_withRect(DATA_E_DEMAG) || paMeshCUDA->IsStageSet(SS_MONTECARLO),
		(MOD_)paMeshCUDA->Get_Module_Energy_Display() == MOD_DEMAG || paMeshCUDA->IsOutputDataSet_withRect(DATA_E_DEMAG) || paMeshCUDA->IsStageSet(SS_MONTECARLO));
	if (error)	initialized = false;

	if (initialized) set_Atom_DemagCUDA_pointers();

	//if a Monte Carlo stage is set then we need to compute fields
	if (paMeshCUDA->IsStageSet(SS_MONTECARLO)) paMeshCUDA->Set_Force_MonteCarlo_ComputeFields(true);

	return error;
}

bool Atom_DemagMCUDA::Submodules_Initialized(void)
{
	bool all_initialized = true;

	for (int idx = 0; idx < pDemagMCUDA.size(); idx++) {

		all_initialized &= pDemagMCUDA[idx]->initialized;
	}

	return all_initialized;
}

BError Atom_DemagMCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Atom_DemagMCUDA));

	//update configuration for all submodules
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		error = pDemagMCUDA[mGPU]->UpdateConfiguration(cfgMessage);
	}

	//uninitialize this module also if not all submodules are still all initialized
	if (!Submodules_Initialized()) {

		Uninitialize();
		EvalSpeedupCUDA::UpdateConfiguration_EvalSpeedup();
	}

	return error;
}

void Atom_DemagMCUDA::UpdateField(void)
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

				if (!mGPU.get_halfprecision_transfer())
					pDemagMCUDA[mGPU]->ForwardFFT_mGPU_first(M.get_deviceobject(mGPU), pDemagMCUDA[mGPU]->Real_yRegion_arr, pDemagMCUDA[mGPU]->Complex_yRegion_arr);
				else
					pDemagMCUDA[mGPU]->ForwardFFT_mGPU_first(
						M.get_deviceobject(mGPU),
						pDemagMCUDA[mGPU]->Real_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization_M,
						pDemagMCUDA[mGPU]->Complex_yRegion_half_arr, pDemagMCUDA[mGPU]->normalization);
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
			//Finish convolution, setting output in Hd
			///////////////////////////////////////////////////////////////////////////////////////////////

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				if (!mGPU.get_halfprecision_transfer()) {

					if (Module_Heff.linear_size_cpu()) {

						pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDemagMCUDA[mGPU]->Real_xRegion_arr,
							M.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), true,
							Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
					}
					else {

						pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDemagMCUDA[mGPU]->Real_xRegion_arr,
							M.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), true);
					}
				}
				else {

					if (Module_Heff.linear_size_cpu()) {

						pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
							M.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), true,
							Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
					}
					else {

						pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
							M.get_deviceobject(mGPU), H.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), true);
					}
				}
			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////
		//SINGLE DEVICE DEMAG
		///////////////////////////////////////////////////////////////////////////////////////////////
		else {

			if (Module_Heff.linear_size_cpu())
				pDemagMCUDA[0]->Convolute(
					M.get_deviceobject(0), H.get_deviceobject(0),
					energy(0), paMeshCUDA->CurrentTimeStepSolved(), true,
					&Module_Heff.get_deviceobject(0), &Module_energy.get_deviceobject(0));
			else
				pDemagMCUDA[0]->Convolute(
					M.get_deviceobject(0), H.get_deviceobject(0),
					energy(0), paMeshCUDA->CurrentTimeStepSolved(), true);
		}
	};

	std::function<void(void)> do_transfer_in = [&](void) -> void {

		//transfer magnetic moments to magnetization mesh, converting from moment to magnetization in the process
		Transfer_Moments_to_Magnetization();
	};

	std::function<void(mcu_VEC(cuReal3)&)> do_transfer_out = [&](mcu_VEC(cuReal3)& H) -> void {

		//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
		Transfer_Demag_Field(H);
	};

	if (!eval_speedup) {

		///////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////// NO SPEEDUP //////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		//transfer into M
		do_transfer_in();

		//convolution
		do_evaluation(Hd);

		//transfer from Hd to Heff
		do_transfer_out(Hd);
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
