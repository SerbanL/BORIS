#pragma once

#include "mcuVEC.h"

//------------------------------------------------------------------- ALLOCATION

//allocate memory for halos
template <typename VType, typename MType>
template <typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::allocate_halos(void)
{
	//only use if halos applicable
	if (halo_depth == 0 || mGPU.get_num_devices() == 1) return;

	//NOTE : halo flags must be set even if using UVA. Only difference is halos not exchanged at runtime.

	int num_halo_cells = 0;

	//now determine halo direction and number of cells we need to allocate for halos
	
	//x direction
	if (mGPU.get_subvec_axis() == 0 || (mGPU.get_subvec_axis() == -1 && n.x >= n.y && n.x > n.z)) {

		halo_flag_n = NF2_HALONX; halo_flag_p = NF2_HALOPX; halo_flag = NF2_HALOX;
		num_halo_cells = halo_depth * n.y * n.z;
	}

	//y direction
	else if (mGPU.get_subvec_axis() == 1 || (mGPU.get_subvec_axis() == -1 && n.y > n.z)) {

		halo_flag_n = NF2_HALONY; halo_flag_p = NF2_HALOPY; halo_flag = NF2_HALOY;
		num_halo_cells = halo_depth * n.x * n.z;
	}
	
	//z direction
	else {

		halo_flag_n = NF2_HALONZ; halo_flag_p = NF2_HALOPZ; halo_flag = NF2_HALOZ;
		num_halo_cells = halo_depth * n.x * n.y;
	}

	bool reallocated = false;

	//allocate memory (if needed)
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (phalo_ngbr_p[mGPU] == nullptr) { phalo_ngbr_p[mGPU] = new cu_arr<int>(num_halo_cells); reallocated = true; }
		if (phalo_ngbr_n[mGPU] == nullptr) { phalo_ngbr_n[mGPU] = new cu_arr<int>(num_halo_cells); reallocated = true; }

		if (phalo_ngbr_p[mGPU]->size() != num_halo_cells) { phalo_ngbr_p[mGPU]->resize(num_halo_cells); reallocated = true; }
		if (phalo_ngbr_n[mGPU]->size() != num_halo_cells) { phalo_ngbr_n[mGPU]->resize(num_halo_cells); reallocated = true; }
	}

	if (reallocated) {

		//setup synchronous transfer for ngbr flags (slower than asynch, but this is initialization only - required to avoid data race conditions).
		halo_ngbr_n_transf.set_async_transfer(false);
		halo_ngbr_p_transf.set_async_transfer(false);

		//setup memory transfer objects
		halo_ngbr_n_transf.set_transfer_size(num_halo_cells);
		halo_ngbr_p_transf.set_transfer_size(num_halo_cells);
		halo_quant_n_transf.set_transfer_size(num_halo_cells);
		halo_quant_p_transf.set_transfer_size(num_halo_cells);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			halo_ngbr_n_transf.setup_device_memory_handle(mGPU, phalo_ngbr_n[mGPU]->get_array());
			halo_ngbr_p_transf.setup_device_memory_handle(mGPU, phalo_ngbr_p[mGPU]->get_array());
		}
	}
}

//------------------------------------------------------------------- SETUP

//coordinate ngbr flag exchanges to set halo conditions in managed devices
template <typename VType, typename MType>
template <typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::set_halo_conditions(void)
{
	//only use if halos applicable
	if (halo_depth == 0 || mGPU.get_num_devices() == 1) return;

	//NOTE : halo flags must be set even if using UVA. Only difference is halos not exchanged at runtime.

	bool pbc_along_halo = ((pbc_x && halo_flag == NF2_HALOX) || (pbc_y && halo_flag == NF2_HALOY) || (pbc_z && halo_flag == NF2_HALOZ));

	//for each device copy ngbr flags region from adjacent device, then set halo flags
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		//cells box on n and p sides; halo flags on n and p sides.
		cuBox box_n, box_p;

		switch (halo_flag) {

		case NF2_HALOX:

			//box on p side (next device)
			if (mGPU + 1 < mGPU.get_num_devices()) box_p = cuBox(1, pn_d[mGPU + 1].y, pn_d[mGPU + 1].z);
			//for last device get box from first device if pbc
			else if (pbc_x) box_p = cuBox(1, pn_d[0].y, pn_d[0].z);

			//box on n side (previous device)
			if (mGPU > 0) box_n = cuBox(pn_d[mGPU - 1].x - 1, 0, 0, pn_d[mGPU - 1].x, pn_d[mGPU - 1].y, pn_d[mGPU - 1].z);
			//for first device get box from last device if pbc
			else if (pbc_x) box_n = cuBox(pn_d[mGPU.get_num_devices() - 1].x - 1, 0, 0, pn_d[mGPU.get_num_devices() - 1].x, pn_d[mGPU.get_num_devices() - 1].y, pn_d[mGPU.get_num_devices() - 1].z);
			break;
		case NF2_HALOY:
			//box on p side (next device)
			if (mGPU + 1 < mGPU.get_num_devices()) box_p = cuBox(pn_d[mGPU + 1].x, 1, pn_d[mGPU + 1].z);
			//for last device get box from first device if pbc
			else if (pbc_y) box_p = cuBox(pn_d[0].x, 1, pn_d[0].z);

			//box on n side (previous device)
			if (mGPU > 0) box_n = cuBox(0, pn_d[mGPU - 1].y - 1, 0, pn_d[mGPU - 1].x, pn_d[mGPU - 1].y, pn_d[mGPU - 1].z);
			//for first device get box from last device if pbc
			else if (pbc_y) box_n = cuBox(0, pn_d[mGPU.get_num_devices() - 1].y - 1, 0, pn_d[mGPU.get_num_devices() - 1].x, pn_d[mGPU.get_num_devices() - 1].y, pn_d[mGPU.get_num_devices() - 1].z);
			break;
		case NF2_HALOZ:
			//box on p side (next device)
			if (mGPU + 1 < mGPU.get_num_devices()) box_p = cuBox(pn_d[mGPU + 1].x, pn_d[mGPU + 1].y, 1);
			//for last device get box from first device if pbc
			else if (pbc_z) box_p = cuBox(pn_d[0].x, pn_d[0].y, 1);

			//box on n side (previous device)
			if (mGPU > 0) box_n = cuBox(0, 0, pn_d[mGPU - 1].z - 1, pn_d[mGPU - 1].x, pn_d[mGPU - 1].y, pn_d[mGPU - 1].z);
			//for first device get box from last device if pbc
			else if (pbc_z) box_n = cuBox(0, 0, pn_d[mGPU.get_num_devices() - 1].z - 1, pn_d[mGPU.get_num_devices() - 1].x, pn_d[mGPU.get_num_devices() - 1].y, pn_d[mGPU.get_num_devices() - 1].z);
			break;
		}

		//get ngbr flags from device on p side (if possible)
		//to do this: 1) select next device, 2) copy flags region from that device in linear storage space here, 3) transfer to current device in one operation
		if (mGPU + 1 < mGPU.get_num_devices()) {

			//1)
			mGPU.select_next_device();

			//2)
			mng(mGPU + 1)->extract_ngbrFlags(box_p, *phalo_ngbr_p[mGPU + 1]);

			//3) transfer phalo_ngbr_p[mGPU + 1] to phalo_ngbr_p[mGPU]
			halo_ngbr_p_transf.transfer(mGPU, mGPU + 1);
		}
		//for last device, if pbc along halo direction then extract from first device and transfer to last device
		else if (pbc_along_halo) {

			//1) select first device
			mGPU.select_device(0);

			//2)
			mng(0)->extract_ngbrFlags(box_p, *phalo_ngbr_p[0]);

			//3) transfer phalo_ngbr_p[0] to phalo_ngbr_p[mGPU]
			halo_ngbr_p_transf.transfer(mGPU, 0);
		}

		//get ngbr flags from device on n side (if possible)
		if (mGPU > 0) {

			//1)
			mGPU.select_previous_device();

			//2)
			mng(mGPU - 1)->extract_ngbrFlags(box_n, *phalo_ngbr_n[mGPU - 1]);

			//3) transfer phalo_ngbr_n[mGPU - 1] to phalo_ngbr_n[mGPU]
			halo_ngbr_n_transf.transfer(mGPU, mGPU - 1);
		}
		//for first device, if pbc along halo direction then extract from last device and transfer to first device
		else if (pbc_along_halo) {

			//1)
			mGPU.select_device(mGPU.get_num_devices() - 1);

			//2)
			mng(mGPU.get_num_devices() - 1)->extract_ngbrFlags(box_n, *phalo_ngbr_n[mGPU.get_num_devices() - 1]);

			//3) transfer phalo_ngbr_n[mGPU.get_num_devices() - 1] to phalo_ngbr_n[mGPU]
			halo_ngbr_n_transf.transfer(mGPU, mGPU.get_num_devices() - 1);
		}

		//set flags in current device
		mGPU.select_current_device();
		if (mGPU < mGPU.get_num_devices() - 1 || pbc_along_halo) mng(mGPU)->set_halo_conditions(*phalo_ngbr_p[mGPU], halo_flag_p, halo_depth);
		if (mGPU > 0 || pbc_along_halo) mng(mGPU)->set_halo_conditions(*phalo_ngbr_n[mGPU], halo_flag_n, halo_depth);

		//now that required halos have been allocated and halo flags set, configure transfer object
		//transfers : halo arrays are base case, halotemp arrays added as extra case
		halo_quant_n_transf.setup_device_memory_handle_managed(mGPU, mng(mGPU)->halo_n_ref());
		if (!halo_quant_n_transf.get_num_extra_transfers(mGPU)) halo_quant_n_transf.add_extra_device_memory_handle_managed(mGPU, mng(mGPU)->halotemp_n_ref());
		else halo_quant_n_transf.setup_extra_device_memory_handle_managed(mGPU, 1, mng(mGPU)->halotemp_n_ref());

		halo_quant_p_transf.setup_device_memory_handle_managed(mGPU, mng(mGPU)->halo_p_ref());
		if (!halo_quant_p_transf.get_num_extra_transfers(mGPU)) halo_quant_p_transf.add_extra_device_memory_handle_managed(mGPU, mng(mGPU)->halotemp_p_ref());
		else halo_quant_p_transf.setup_extra_device_memory_handle_managed(mGPU, 1, mng(mGPU)->halotemp_p_ref());
	}

	//if UVA possible then set pUVA_haloVEC_... pointers as needed. Must be at the end since calling set_halo_conditions clears the flags below.
	if (mGPU.is_uva_enabled()) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			//left if possible
			if (mGPU > 0) mng(mGPU)->set_pUVA_haloVEC_left(mng.get_managed_object(mGPU - 1));

			//right if possible
			if (mGPU + 1 < mGPU.get_num_devices()) mng(mGPU)->set_pUVA_haloVEC_right(mng.get_managed_object(mGPU + 1));

			//pbc along halo direction?
			if (pbc_along_halo) {

				//first device will see on its left the last device cuVEC - i.e. pbc
				if (mGPU == 0) mng(mGPU)->set_pUVA_haloVEC_left(mng.get_managed_object(mGPU.get_num_devices() - 1));

				//last device will see on its right the first device cuVEC - i.e. pbc
				if (mGPU == mGPU.get_num_devices() - 1) mng(mGPU)->set_pUVA_haloVEC_right(mng.get_managed_object(0));
			}
		}
	}
}