#pragma once

#include "mcuVEC.h"

#include "cuVEC_VC_halo.cuh"

//------------------------------------------------------------------- RUNTIME : EXCHANGE

//exchange values in all halos
template <typename VType, typename MType>
template <typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::exchange_halos(bool force_exchange)
{
	//only use if halos applicable. 
	//Also, when using UVA there's no need to exchange halos as we can read directly using pointers set through set_halo_conditions
	//However NOTE : UVA can be slower than halo exchange mechanism, especially with more than 2 GPUs
	
	if (halo_depth == 0 || mGPU.get_num_devices() == 1 || !n.dim()) return;

	//additionally if uva is enabled we must synchronize at this point, as this method will be called before launching a kernel using uva to access memory on other devices so must guarantee no data race
	//The exception is when we want to force halo exchange for special-use cases (e.g. shifting algorithm)
	if (!force_exchange && mGPU.is_uva_enabled()) {

		mGPU.synchronize();
		return;
	}

	bool pbc_along_halo = ((pbc_x && halo_flag == NF2_HALOX) || (pbc_y && halo_flag == NF2_HALOY) || (pbc_z && halo_flag == NF2_HALOZ));

	//1. in each device extract values to temporary halos
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (mGPU != mGPU.get_num_devices() - 1 || pbc_along_halo) {

			//extraction to halotemp_n : all devices except last one, unless pbc is along halo
			mng(mGPU)->extract_halo(halo_flag_n, halo_quant_n_transf.get_transfer_size());
		}

		if (mGPU != 0 || pbc_along_halo) {

			//extraction to halotemp_p : all devices except first one, unless pbc is along halo
			mng(mGPU)->extract_halo(halo_flag_p, halo_quant_p_transf.get_transfer_size());
		}
	}

	//must synchronize before transfer else we'll end up with data race
	mGPU.synchronize();

	//2a. transfer from temporary to runtime halos between devices : n halos
	for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

		//transfer halotemp_n from this device to halo_n of next device
		if (idx < mGPU.get_num_devices() - 1) halo_quant_n_transf.transfer(idx + 1, 0, idx, 1);
		//if pbc along halo and this is last device, then transfer the extract halotemp_n to halo_n of first device
		else if (pbc_along_halo) halo_quant_n_transf.transfer(0, 0, mGPU.get_num_devices() - 1, 1);
	}

	//2b. transfer from temporary to runtime halos between devices : p halos
	for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

		//transfer halotemp_p from this device to halo_p of previous device
		if (idx > 0) halo_quant_p_transf.transfer(idx - 1, 0, idx, 1);
		//if pbc along halo and this is first device, then transfer the extract halotemp_p to halo_p of last device
		else if (pbc_along_halo) halo_quant_p_transf.transfer(mGPU.get_num_devices() - 1, 0, 0, 1);
	}

	//must synchronize after transfer else we'll end up with data race
	mGPU.synchronize();
}