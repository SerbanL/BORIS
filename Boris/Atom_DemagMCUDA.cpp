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
	Hdemag(mGPU), Hdemag2(mGPU), Hdemag3(mGPU), Hdemag4(mGPU), Hdemag5(mGPU), Hdemag6(mGPU),
	selfDemagCoeff(mGPU)
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

		//there must be an integer number of spins in each macrocell
		for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

			cuSZ3 dn = paMeshCUDA->M1.device_n(idx);
			cuSZ3 dn_m = M.device_n(idx);

			if (dn.x % dn_m.x || dn.y % dn_m.y || dn.z % dn_m.z) {

				Uninitialize();
				return error(BERROR_ATOMDMCELL);
			}
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

		/////////////////////////////////////////////////////////////
		//Eval speedup

		selfDemagCoeff.from_cpu(DemagTFunc().SelfDemag_PBC(M.h, M.n, paMeshCUDA->M1.Get_PBC()));

		//make sure to allocate memory for Hdemag if we need it
		if (paMeshCUDA->GetEvaluationSpeedup() >= 6) { if (!Hdemag6.resize(M.h, paMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag6.clear();

		if (paMeshCUDA->GetEvaluationSpeedup() >= 5) { if (!Hdemag5.resize(M.h, paMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag5.clear();

		if (paMeshCUDA->GetEvaluationSpeedup() >= 4) { if (!Hdemag4.resize(M.h, paMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag4.clear();

		if (paMeshCUDA->GetEvaluationSpeedup() >= 3) { if (!Hdemag3.resize(M.h, paMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag3.clear();

		if (paMeshCUDA->GetEvaluationSpeedup() >= 2) { if (!Hdemag2.resize(M.h, paMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag2.clear();

		if (paMeshCUDA->GetEvaluationSpeedup() >= 1) { if (!Hdemag.resize(M.h, paMeshCUDA->meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		else Hdemag.clear();

		if (!error) initialized = true;
	}

	num_Hdemag_saved = 0;
	
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

void Atom_DemagMCUDA::UpdateField(void)
{
	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////// NO SPEEDUP //////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (!paMeshCUDA->GetEvaluationSpeedup() || (num_Hdemag_saved < paMeshCUDA->GetEvaluationSpeedup() && !paMeshCUDA->Check_Step_Update())) {
		
		//transfer magnetic moments to magnetization mesh, converting from moment to magnetization in the process
		Transfer_Moments_to_Magnetization();

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
							M.get_deviceobject(mGPU), Hd.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), true,
							Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
					}
					else {

						pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDemagMCUDA[mGPU]->Real_xRegion_arr,
							M.get_deviceobject(mGPU), Hd.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), true);
					}
				}
				else {

					if (Module_Heff.linear_size_cpu()) {

						pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
							M.get_deviceobject(mGPU), Hd.get_deviceobject(mGPU),
							energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), true,
							Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
					}
					else {

						pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
							pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
							M.get_deviceobject(mGPU), Hd.get_deviceobject(mGPU),
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
					M.get_deviceobject(0), Hd.get_deviceobject(0),
					energy(0), paMeshCUDA->CurrentTimeStepSolved(), true,
					&Module_Heff.get_deviceobject(0), &Module_energy.get_deviceobject(0));
			else
				pDemagMCUDA[0]->Convolute(
					M.get_deviceobject(0), Hd.get_deviceobject(0),
					energy(0), paMeshCUDA->CurrentTimeStepSolved(), true);
		}

		//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
		Transfer_Demag_Field(Hd);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////// EVAL SPEEDUP /////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////
	
	else {

		//use evaluation speedup method (Hdemag will have memory allocated - this was done in the Initialize method)

		//update if required by ODE solver or if we don't have enough previous evaluations saved to extrapolate
		if (paMeshCUDA->Check_Step_Update() || num_Hdemag_saved < paMeshCUDA->GetEvaluationSpeedup()) {

			mcu_VEC(cuReal3)* pHdemag;

			if (num_Hdemag_saved < paMeshCUDA->GetEvaluationSpeedup()) {

				//don't have enough evaluations, so save next one
				switch (num_Hdemag_saved)
				{
				case 0:
					pHdemag = &Hdemag;
					time_demag1 = paMeshCUDA->Get_EvalStep_Time();
					break;
				case 1:
					pHdemag = &Hdemag2;
					time_demag2 = paMeshCUDA->Get_EvalStep_Time();
					break;
				case 2:
					pHdemag = &Hdemag3;
					time_demag3 = paMeshCUDA->Get_EvalStep_Time();
					break;
				case 3:
					pHdemag = &Hdemag4;
					time_demag4 = paMeshCUDA->Get_EvalStep_Time();
					break;
				case 4:
					pHdemag = &Hdemag5;
					time_demag5 = paMeshCUDA->Get_EvalStep_Time();
					break;
				case 5:
					pHdemag = &Hdemag6;
					time_demag6 = paMeshCUDA->Get_EvalStep_Time();
					break;
				}

				num_Hdemag_saved++;
			}
			else {

				//have enough evaluations saved, so just cycle between them now

				//QUINTIC
				if (paMeshCUDA->GetEvaluationSpeedup() == 6) {

					//1, 2, 3, 4, 5, 6 -> next is 1
					if (time_demag6 > time_demag5 && time_demag5 > time_demag4 && time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

						pHdemag = &Hdemag;
						time_demag1 = paMeshCUDA->Get_EvalStep_Time();
					}
					//2, 3, 4, 5, 6, 1 -> next is 2
					else if (time_demag1 > time_demag2) {

						pHdemag = &Hdemag2;
						time_demag2 = paMeshCUDA->Get_EvalStep_Time();
					}
					//3, 4, 5, 6, 1, 2 -> next is 3
					else if (time_demag2 > time_demag3) {

						pHdemag = &Hdemag3;
						time_demag3 = paMeshCUDA->Get_EvalStep_Time();
					}
					//4, 5, 6, 1, 2, 3 -> next is 4
					else if (time_demag3 > time_demag4) {

						pHdemag = &Hdemag4;
						time_demag4 = paMeshCUDA->Get_EvalStep_Time();
					}
					//5, 6, 1, 2, 3, 4 -> next is 5
					else if (time_demag4 > time_demag5) {

						pHdemag = &Hdemag5;
						time_demag5 = paMeshCUDA->Get_EvalStep_Time();
					}
					else {

						pHdemag = &Hdemag6;
						time_demag6 = paMeshCUDA->Get_EvalStep_Time();
					}
				}
				//QUARTIC
				else if (paMeshCUDA->GetEvaluationSpeedup() == 5) {

					//1, 2, 3, 4, 5 -> next is 1
					if (time_demag5 > time_demag4 && time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

						pHdemag = &Hdemag;
						time_demag1 = paMeshCUDA->Get_EvalStep_Time();
					}
					//2, 3, 4, 5, 1 -> next is 2
					else if (time_demag1 > time_demag2) {

						pHdemag = &Hdemag2;
						time_demag2 = paMeshCUDA->Get_EvalStep_Time();
					}
					//3, 4, 5, 1, 2 -> next is 3
					else if (time_demag2 > time_demag3) {

						pHdemag = &Hdemag3;
						time_demag3 = paMeshCUDA->Get_EvalStep_Time();
					}
					//4, 5, 1, 2, 3 -> next is 4
					else if (time_demag3 > time_demag4) {

						pHdemag = &Hdemag4;
						time_demag4 = paMeshCUDA->Get_EvalStep_Time();
					}
					else {

						pHdemag = &Hdemag5;
						time_demag5 = paMeshCUDA->Get_EvalStep_Time();
					}
				}
				//CUBIC
				else if (paMeshCUDA->GetEvaluationSpeedup() == 4) {

					//1, 2, 3, 4 -> next is 1
					if (time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

						pHdemag = &Hdemag;
						time_demag1 = paMeshCUDA->Get_EvalStep_Time();
					}
					//2, 3, 4, 1 -> next is 2
					else if (time_demag1 > time_demag2) {

						pHdemag = &Hdemag2;
						time_demag2 = paMeshCUDA->Get_EvalStep_Time();
					}
					//3, 4, 1, 2 -> next is 3
					else if (time_demag2 > time_demag3) {

						pHdemag = &Hdemag3;
						time_demag3 = paMeshCUDA->Get_EvalStep_Time();
					}
					else {

						pHdemag = &Hdemag4;
						time_demag4 = paMeshCUDA->Get_EvalStep_Time();
					}
				}
				//QUADRATIC
				else if (paMeshCUDA->GetEvaluationSpeedup() == 3) {

					//1, 2, 3 -> next is 1
					if (time_demag3 > time_demag2 && time_demag2 > time_demag1) {

						pHdemag = &Hdemag;
						time_demag1 = paMeshCUDA->Get_EvalStep_Time();
					}
					//2, 3, 1 -> next is 2
					else if (time_demag3 > time_demag2 && time_demag1 > time_demag2) {

						pHdemag = &Hdemag2;
						time_demag2 = paMeshCUDA->Get_EvalStep_Time();
					}
					//3, 1, 2 -> next is 3, leading to 1, 2, 3 again
					else {

						pHdemag = &Hdemag3;
						time_demag3 = paMeshCUDA->Get_EvalStep_Time();
					}
				}
				//LINEAR
				else if (paMeshCUDA->GetEvaluationSpeedup() == 2) {

					//1, 2 -> next is 1
					if (time_demag2 > time_demag1) {

						pHdemag = &Hdemag;
						time_demag1 = paMeshCUDA->Get_EvalStep_Time();
					}
					//2, 1 -> next is 2, leading to 1, 2 again
					else {

						pHdemag = &Hdemag2;
						time_demag2 = paMeshCUDA->Get_EvalStep_Time();
					}
				}
				//STEP
				else {

					pHdemag = &Hdemag;
				}
			}

			//do evaluation
			ZeroEnergy();

			//transfer magnetic moments to magnetization mesh, converting from moment to magnetization in the process
			Transfer_Moments_to_Magnetization();

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
								M.get_deviceobject(mGPU), pHdemag->get_deviceobject(mGPU),
								energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), true,
								Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
						}
						else {

							pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_arr,
								M.get_deviceobject(mGPU), pHdemag->get_deviceobject(mGPU),
								energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), true);
						}
					}
					else {

						if (Module_Heff.linear_size_cpu()) {

							pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
								M.get_deviceobject(mGPU), pHdemag->get_deviceobject(mGPU),
								energy(mGPU), paMeshCUDA->CurrentTimeStepSolved(), true,
								Module_Heff.get_managed_object(mGPU), Module_energy.get_managed_object(mGPU));
						}
						else {

							pDemagMCUDA[mGPU]->InverseFFT_mGPU_finish(
								pDemagMCUDA[mGPU]->Real_xRegion_half_arr, pDemagMCUDA[mGPU]->normalization,
								M.get_deviceobject(mGPU), pHdemag->get_deviceobject(mGPU),
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
						M.get_deviceobject(0), pHdemag->get_deviceobject(0),
						energy(0), paMeshCUDA->CurrentTimeStepSolved(), true,
						&Module_Heff.get_deviceobject(0), &Module_energy.get_deviceobject(0));
				else
					pDemagMCUDA[0]->Convolute(
						M.get_deviceobject(0), pHdemag->get_deviceobject(0),
						energy(0), paMeshCUDA->CurrentTimeStepSolved(), true);
			}

			//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
			Transfer_Demag_Field(*pHdemag);

			//subtract self demag from *pHDemag
			Atom_Demag_EvalSpeedup_SubSelf(*pHdemag);
		}
		else {

			//not required to update, and we have enough previous evaluations: use previous Hdemag saves to extrapolate for current evaluation

			//transfer magnetic moments to magnetization mesh, converting from moment to magnetization in the process
			Transfer_Moments_to_Magnetization();

			cuBReal a1 = 1.0, a2 = 0.0, a3 = 0.0, a4 = 0.0, a5 = 0.0, a6 = 0.0;
			cuBReal time = paMeshCUDA->Get_EvalStep_Time();

			//QUINTIC
			if (paMeshCUDA->GetEvaluationSpeedup() == 6) {

				a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4) * (time_demag1 - time_demag5) * (time_demag1 - time_demag6));
				a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4) * (time_demag2 - time_demag5) * (time_demag2 - time_demag6));
				a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4) * (time_demag3 - time_demag5) * (time_demag3 - time_demag6));
				a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag5) * (time - time_demag6) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3) * (time_demag4 - time_demag5) * (time_demag4 - time_demag6));
				a5 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag6) / ((time_demag5 - time_demag1) * (time_demag5 - time_demag2) * (time_demag5 - time_demag3) * (time_demag5 - time_demag4) * (time_demag5 - time_demag6));
				a6 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag6 - time_demag1) * (time_demag6 - time_demag2) * (time_demag6 - time_demag3) * (time_demag6 - time_demag4) * (time_demag6 - time_demag5));

				//construct effective field approximation
				Atom_Demag_EvalSpeedup_SetExtrapField_AddSelf(Hd, a1, a2, a3, a4, a5, a6);

				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				Transfer_Demag_Field(Hd);
			}
			//QUARTIC
			else if (paMeshCUDA->GetEvaluationSpeedup() == 5) {

				a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4) * (time_demag1 - time_demag5));
				a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4) * (time_demag2 - time_demag5));
				a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) * (time - time_demag5) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4) * (time_demag3 - time_demag5));
				a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag5) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3) * (time_demag4 - time_demag5));
				a5 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) / ((time_demag5 - time_demag1) * (time_demag5 - time_demag2) * (time_demag5 - time_demag3) * (time_demag5 - time_demag4));

				//construct effective field approximation
				Atom_Demag_EvalSpeedup_SetExtrapField_AddSelf(Hd, a1, a2, a3, a4, a5);

				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				Transfer_Demag_Field(Hd);
			}
			//CUBIC
			else if (paMeshCUDA->GetEvaluationSpeedup() == 4) {

				a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4));
				a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4));
				a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4));
				a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3));

				//construct effective field approximation
				Atom_Demag_EvalSpeedup_SetExtrapField_AddSelf(Hd, a1, a2, a3, a4);

				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				Transfer_Demag_Field(Hd);
			}
			//QUADRATIC
			else if (paMeshCUDA->GetEvaluationSpeedup() == 3) {

				if (time_demag2 != time_demag1 && time_demag2 != time_demag3 && time_demag1 != time_demag3) {

					a1 = (time - time_demag2) * (time - time_demag3) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3));
					a2 = (time - time_demag1) * (time - time_demag3) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3));
					a3 = (time - time_demag1) * (time - time_demag2) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2));
				}

				//construct effective field approximation
				Atom_Demag_EvalSpeedup_SetExtrapField_AddSelf(Hd, a1, a2, a3);

				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				Transfer_Demag_Field(Hd);
			}
			//LINEAR
			else if (paMeshCUDA->GetEvaluationSpeedup() == 2) {

				if (time_demag2 != time_demag1) {

					a1 = (time - time_demag2) / (time_demag1 - time_demag2);
					a2 = (time - time_demag1) / (time_demag2 - time_demag1);
				}

				//construct effective field approximation
				Atom_Demag_EvalSpeedup_SetExtrapField_AddSelf(Hd, a1, a2);

				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				Transfer_Demag_Field(Hd);
			}
			//STEP
			else {

				//construct effective field approximation
				Atom_Demag_EvalSpeedup_SetExtrapField_AddSelf(Hd);

				//transfer demagnetising field to atomistic mesh effective field : all atomistic cells within the larger micromagnetic cell receive the same field
				Transfer_Demag_Field(Hd);
			}
		}
	}
}

#endif

#endif
