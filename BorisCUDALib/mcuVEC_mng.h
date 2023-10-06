#pragma once

#include "mcuVEC.h"
#include "mcuVEC_MeshTransfer.h"
#include "cuVEC_mcuVEC.h"
#include "cuVEC_VC_mcuVEC.h"

//--------------------------------------------CONSTRUCTORS : mcuVEC_mng.h

//constructor for this policy class : void
template <typename VType, typename MType>
mcuVEC<VType, MType>::mcuVEC(mcu_obj<MType, mcuVEC<VType, MType>>& mng_, mGPUConfig& mGPU_) :
	mng(mng_),
	mGPU(mGPU_),
	halo_ngbr_n_transf(mGPU_),
	halo_ngbr_p_transf(mGPU_),
	halo_quant_n_transf(mGPU_),
	halo_quant_p_transf(mGPU_),
	profile_aux(mGPU_),
	profile_component_aux(mGPU_),
	histogram_transf(mGPU_)
{
	pn_d = new cuSZ3[mGPU.get_num_devices()];
	prect_d = new cuRect[mGPU.get_num_devices()];
	pbox_d = new cuBox[mGPU.get_num_devices()];

	//storage vectors for communication between devices
	phalo_ngbr_n.assign(mGPU.get_num_devices(), nullptr);
	phalo_ngbr_p.assign(mGPU.get_num_devices(), nullptr);

	//histogram extra handles
	histogram_base_aux.resize(mGPU.get_num_devices());

	for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

		histogram_transf.add_extra_device_memory_handle(0, histogram_base_aux[idx].get_array());
	}

	for (int idx = 0; idx < histogram_base_aux.size(); idx++) {

		histogram_base_aux_col.push_back(histogram_base_aux[idx].get_managed_array());
	}
}

//assignment operator
template <typename VType, typename MType>
mcuVEC<VType, MType>& mcuVEC<VType, MType>::operator=(const mcuVEC<VType, MType>& copyThis)
{
	clear_memory_aux();

	pn_d = new cuSZ3[mGPU.get_num_devices()];
	prect_d = new cuRect[mGPU.get_num_devices()];
	pbox_d = new cuBox[mGPU.get_num_devices()];

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		pn_d[mGPU] = copyThis.pn_d[mGPU];
		prect_d[mGPU] = copyThis.prect_d[mGPU];
		pbox_d[mGPU] = copyThis.pbox_d[mGPU];

		if (copyThis.phalo_ngbr_n[mGPU]) phalo_ngbr_n[mGPU] = new cu_arr<int>(copyThis.phalo_ngbr_n[mGPU]->size());
		if (copyThis.phalo_ngbr_p[mGPU]) phalo_ngbr_p[mGPU] = new cu_arr<int>(copyThis.phalo_ngbr_p[mGPU]->size());
	}

	n = copyThis.n;
	h = copyThis.h;
	rect = copyThis.rect; set_collection_origin();

	halo_depth = copyThis.halo_depth;
	halo_flag_n = copyThis.halo_flag_n;
	halo_flag_p = copyThis.halo_flag_p;
	halo_flag = copyThis.halo_flag;

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	return *this;
}

//make man_mcuVEC - call when constructing policy (i.e. one of the above methods)
template <typename VType, typename MType>
void mcuVEC<VType, MType>::construct_managed(void)
{
	man_mcuVEC.resize(mGPU.get_num_devices());
	man_mcuVEC_Base.resize(mGPU.get_num_devices());

	//make a cu_obj-managed mcuVEC_Managed object for each device so we can pass it into CUDA kernels
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		man_mcuVEC[mGPU] = new cu_obj<mcuVEC_Managed<MType, VType>>();
		man_mcuVEC_Base[mGPU] = new cu_obj<mcuVEC_Managed<cuVEC<VType>, VType>>();
	}

	//setup cuVEC(_VC) pointers in each one so we can use UVA to access them in CUDA kernels
	//go through each entry in man_mcuVEC in the outer loop
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		//for each entry in man_mcuVEC (i.e. cu-obj managed mcuVEC_Managed<MType, VType>>) setup pointers to all devices, so have inner loop going through all
		for (int device_idx = 0; device_idx < mGPU.get_num_devices(); device_idx++) {

			(*man_mcuVEC[mGPU])()->set_pointers(mGPU.get_num_devices(), device_idx, mng.get_managed_object(device_idx));
			//here conversion from cuVEC_VC to cuVEC is done
			(*man_mcuVEC_Base[mGPU])()->set_pointers(mGPU.get_num_devices(), device_idx, (cuVEC<VType>*&)mng.get_managed_object(device_idx));
		}
	}

	//now for each device sub-cuVEC store pointer to respective man_mcuVEC
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
		
		mng(mGPU)->set_pmcuVEC(man_mcuVEC[mGPU]->get_managed_object(), man_mcuVEC_Base[mGPU]->get_managed_object());
	}
}

//destroy man_mcuVEC - call when destructing this object
template <typename VType, typename MType>
void mcuVEC<VType, MType>::destruct_managed(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		delete man_mcuVEC[mGPU];
		man_mcuVEC[mGPU] = nullptr;

		delete man_mcuVEC_Base[mGPU];
		man_mcuVEC_Base[mGPU] = nullptr;
	}
}

//--------------------------------------------man_mcuVEC Auxiliary : mcuVEC_mng.h

template <typename VType, typename MType>
void mcuVEC<VType, MType>::synch_dimensions(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*man_mcuVEC[mGPU])()->synch_dimensions(mGPU.get_num_devices(), n, h, rect, pn_d, prect_d, pbox_d);
		(*man_mcuVEC_Base[mGPU])()->synch_dimensions(mGPU.get_num_devices(), n, h, rect, pn_d, prect_d, pbox_d);
	}
}

//--------------------------------------------SIZING : mcuVEC_mng.h

//resize number of cells and assign value, breaking up space amongst devices
template <typename VType, typename MType>
bool mcuVEC<VType, MType>::assign(cuSZ3 new_n, VType value)
{
	//nothing to do if dimensions already set, apart from assigning value
	if (new_n == n) {

		set(value);
		return true;
	}

	//get required n values for all devices, including last one which could differ in size from the rest
	std::pair<cuSZ3, cuSZ3> nd = get_devices_n_values(new_n);

	//starting point for box of first device (relative to cuVEC), i.e. zero.
	cuSZ3 box_start = cuINT3();

	//set sizes for all devices
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (mGPU < mGPU.get_num_devices() - 1) {

			if (!mng(mGPU)->assign(nd.first, value)) return false;
			pn_d[mGPU] = nd.first;

			//setup box with cell coordinates relative to entire cuVEC
			pbox_d[mGPU] = cuBox(box_start, box_start + nd.first);
			box_start += nd.first & cu_normalize(new_n - pbox_d[mGPU].e);
		}
		else {

			if (!mng(mGPU)->assign(nd.second, value)) return false;
			pn_d[mGPU] = nd.second;

			//setup box with cell coordinates relative to entire cuVEC
			pbox_d[mGPU] = cuBox(box_start, box_start + nd.second);
		}
	}

	//new values
	n = new_n;
	rect = cuRect(); set_collection_origin();
	h = cuReal3();

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	return true;
}

//resize number of cells, breaking up space amongst devices
template <typename VType, typename MType>
bool mcuVEC<VType, MType>::resize(cuSZ3 new_n)
{
	//nothing to do if dimensions already set
	if (new_n == n) return true;

	//get required n values for all devices, including last one which could differ in size from the rest
	std::pair<cuSZ3, cuSZ3> nd = get_devices_n_values(new_n);

	//starting point for box of first device (relative to cuVEC), i.e. zero.
	cuSZ3 box_start = cuINT3();

	//set sizes for all devices
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (mGPU < mGPU.get_num_devices() - 1) {

			if (!mng(mGPU)->resize(nd.first)) return false;
			pn_d[mGPU] = nd.first;

			//setup box with cell coordinates relative to entire cuVEC
			pbox_d[mGPU] = cuBox(box_start, box_start + nd.first);
			box_start += nd.first & cu_normalize(new_n - pbox_d[mGPU].e);
		}
		else {

			if (!mng(mGPU)->resize(nd.second)) return false;
			pn_d[mGPU] = nd.second;

			//setup box with cell coordinates relative to entire cuVEC
			pbox_d[mGPU] = cuBox(box_start, box_start + nd.second);
		}
	}

	//new values
	n = new_n;
	rect = cuRect(); set_collection_origin();
	h = cuReal3();

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	return true;
}

//special resizing mode where the device dimensions are specified externally 
//device_dimension specifies dimensions for all but possibly last device
//last device dimension adjusted to make up any difference
template <typename VType, typename MType>
bool mcuVEC<VType, MType>::resize(cuSZ3 new_n, cuSZ3 device_dimension)
{
	//nothing to do if dimensions already set with matching device dimension
	if (new_n == n && pn_d[0] == device_dimension) return true;

	//get required n values for all devices, including last one which could differ in size from the rest
	std::pair<cuSZ3, cuSZ3> nd = get_devices_n_values(new_n, device_dimension);
	//if  nd.second then device_dimension passed in is too large, or else we are using only one GPU
	if (mGPU.get_num_devices() > 1 && nd.second == cuSZ3()) return false;

	//starting point for box of first device (relative to cuVEC), i.e. zero.
	cuSZ3 box_start = cuINT3();

	//set sizes for all devices
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (mGPU < mGPU.get_num_devices() - 1) {

			if (!mng(mGPU)->resize(nd.first)) return false;
			pn_d[mGPU] = nd.first;

			//setup box with cell coordinates relative to entire cuVEC
			pbox_d[mGPU] = cuBox(box_start, box_start + nd.first);
			box_start += nd.first & cu_normalize(new_n - pbox_d[mGPU].e);
		}
		else {

			if (!mng(mGPU)->resize(nd.second)) return false;
			pn_d[mGPU] = nd.second;

			//setup box with cell coordinates relative to entire cuVEC
			pbox_d[mGPU] = cuBox(box_start, box_start + nd.second);
		}
	}

	//new values
	n = new_n;
	rect = cuRect(); set_collection_origin();
	h = cuReal3();

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	return true;
}

//set dimensions and assign value, breaking up space amongst devices
template <typename VType, typename MType>
bool mcuVEC<VType, MType>::assign(cuReal3 new_h, cuRect new_rect, VType value)
{
	//nothing to do if dimensions already set, apart from assigning value
	if (new_h == h && new_rect == rect) {

		set(value);
		return true;
	}

	//get required n
	cuSZ3 new_n = get_n_from_h_and_rect(new_h, new_rect);
	//now adjust h if needed
	new_h = new_rect / new_n;

	//get required n values for all devices, including last one which could differ in size from the rest
	std::pair<cuSZ3, cuSZ3> nd = get_devices_n_values(new_n);

	//starting point for rectangle of first device
	cuReal3 rect_start = new_rect.s;

	//starting point for box of first device (relative to cuVEC), i.e. zero.
	cuSZ3 box_start = cuINT3();

	//set dimensions for all devices
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (mGPU < mGPU.get_num_devices() - 1) {

			//rectangle in absolute coordinates for this device
			cuRect device_rect = cuRect(nd.first & new_h) + rect_start;

			if (!mng(mGPU)->assign(new_h, device_rect, value)) return false;
			pn_d[mGPU] = nd.first;
			prect_d[mGPU] = device_rect;

			//get start of next device rectangle (new_rect.e - device_rect.e is non-zero in exactly one dimension, except for last device)
			rect_start += (device_rect.e - device_rect.s) & cu_normalize(new_rect.e - device_rect.e);

			//setup box with cell coordinates relative to entire cuVEC
			pbox_d[mGPU] = cuBox(box_start, box_start + nd.first);
			box_start += nd.first & cu_normalize(new_n - pbox_d[mGPU].e);
		}
		else {

			//rectangle in absolute coordinates for this device
			cuRect device_rect = cuRect(nd.second & new_h) + rect_start;

			if (!mng(mGPU)->assign(new_h, device_rect, value)) return false;
			pn_d[mGPU] = nd.second;
			prect_d[mGPU] = device_rect;

			//setup box with cell coordinates relative to entire cuVEC
			pbox_d[mGPU] = cuBox(box_start, box_start + nd.second);
		}
	}

	//new values
	n = new_n;
	h = new_h;
	rect = new_rect; set_collection_origin();

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	return true;
}

//set dimensions and assign value, breaking up space amongst devices
template <typename VType, typename MType>
bool mcuVEC<VType, MType>::resize(cuReal3 new_h, cuRect new_rect)
{
	//nothing to do if dimensions already set
	if (new_h == h && new_rect == rect) return true;

	//get required n
	cuSZ3 new_n = get_n_from_h_and_rect(new_h, new_rect);
	//now adjust h if needed
	new_h = new_rect / new_n;

	//get required n values for all devices, including last one which could differ in size from the rest
	std::pair<cuSZ3, cuSZ3> nd = get_devices_n_values(new_n);

	//starting point for rectangle of first device
	cuReal3 rect_start = new_rect.s;

	//starting point for box of first device (relative to cuVEC), i.e. zero.
	cuSZ3 box_start = cuINT3();

	//set dimensions for all devices
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (mGPU < mGPU.get_num_devices() - 1) {

			//rectangle in absolute coordinates for this device
			cuRect device_rect = cuRect(nd.first & new_h) + rect_start;

			if (!mng(mGPU)->resize(new_h, device_rect)) return false;
			pn_d[mGPU] = nd.first;
			prect_d[mGPU] = device_rect;

			//get start of next device rectangle (new_rect.e - device_rect.e is non-zero in exactly one dimension, except for last device)
			
			rect_start += (device_rect.e - device_rect.s) & cu_normalize(new_rect.e - device_rect.e);

			//setup box with cell coordinates relative to entire cuVEC
			pbox_d[mGPU] = cuBox(box_start, box_start + nd.first);
			box_start += nd.first & cu_normalize(new_n - pbox_d[mGPU].e);
		}
		else {

			//rectangle in absolute coordinates for this device
			cuRect device_rect = cuRect(nd.second & new_h) + rect_start;

			if (!mng(mGPU)->resize(new_h, device_rect)) return false;
			pn_d[mGPU] = nd.second;
			prect_d[mGPU] = device_rect;

			//setup box with cell coordinates relative to entire cuVEC
			pbox_d[mGPU] = cuBox(box_start, box_start + nd.second);
		}
	}

	//new values
	n = new_n;
	h = new_h;
	rect = new_rect; set_collection_origin();

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	return true;
}

template <typename VType, typename MType>
void mcuVEC<VType, MType>::clear(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		pn_d[mGPU] = cuSZ3();
		prect_d[mGPU] = cuRect();
		pbox_d[mGPU] = cuBox();
		mng(mGPU)->clear();
	}

	n = cuSZ3();
	rect = cuRect(); set_collection_origin();

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	clear_transfer();
	clear_transfer2();
}

//set rect start (i.e. shift the entire rectangle to align with given absolute starting coordinates
template <typename VType, typename MType>
void mcuVEC<VType, MType>::set_rect_start(const cuReal3& rect_start)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		prect_d[mGPU] += (rect_start - rect.s);
		mng(mGPU)->set_rect_start(prect_d[mGPU].s);
	}

	rect += (rect_start - rect.s);
	set_collection_origin();

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();
}

template <typename VType, typename MType>
void mcuVEC<VType, MType>::shift_rect_start(const cuReal3& shift)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		prect_d[mGPU] += shift;
		mng(mGPU)->shift_rect_start(shift);
	}

	rect += shift;
	set_collection_origin();

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();
}

//--------------------------------------------COPY TO / FROM VEC : mcuVEC_mng.h

//copy everything from a VEC - type must be convertible. Return false if failed (memory could not be allocated)
template <typename VType, typename MType>
template <typename cpuVEC>
bool mcuVEC<VType, MType>::set_from_cpuvec(cpuVEC& vec)
{
	if (!vec.linear_size()) {

		clear();
		return true;
	}

	//set required dimensions
	if (!resize(vec.h, vec.rect)) return false;
	
	//now copy values and flags
	return copy_from_cpuvec(vec);	
}

//copy everything to a VEC - type must be convertible. Return false if failed (memory could not be allocated)
template <typename VType, typename MType>
template <typename cpuVEC>
bool mcuVEC<VType, MType>::set_cpuvec(cpuVEC& vec)
{
	bool success = true;

	//First allocate memory and set dimensions
	if (!vec.resize(h, rect)) return false;

	//auxiliary subVEC
	cpuVEC svec;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		//get values from gpu into svec
		success &= mng(mGPU)->set_cpuvec(svec);

		//copy subvector into larger vec in the correct place
		vec.copy_values(svec, prect_d[mGPU] - rect.s);
	}

	//pbcs must be handled now if VEC_VC
	set_pbc_to(vec);

	return success;
}

//faster version of set_from_cpuvec, where it is assumed the cpu vec already has the same sizes as this cuVEC
template <typename VType, typename MType>
template <typename cpuVEC>
bool mcuVEC<VType, MType>::copy_from_cpuvec(cpuVEC& vec)
{
	if (!vec.linear_size()) return true;

	bool success = true;

	//now copy data (and flags if applicable)
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		success &= mng(mGPU)->copy_from_cpuvec(vec.subvec(pbox_d[mGPU]));
	}

	//pbcs must be handled now if VEC_VC
	set_pbc_from(vec);

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	if (!success) clear();
	return success;
}

//faster version of set_cpuvec, where it is assumed the cpu vec already has the same sizes as this cuVEC
template <typename VType, typename MType>
template <typename cpuVEC>
bool mcuVEC<VType, MType>::copy_to_cpuvec(cpuVEC& vec)
{
	bool success = true;
	
	//auxiliary subVEC
	cpuVEC svec;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
		
		//get values from gpu into svec
		success &= mng(mGPU)->set_cpuvec(svec);

		//copy subvector into larger vec in the correct place
		vec.copy_values(svec, prect_d[mGPU] - rect.s);
	}

	//pbcs must be handled now if VEC_VC
	set_pbc_to(vec);

	return success;
}

//copy values from a std::vector (cpu memory)
template <typename VType, typename MType>
template <typename SType>
bool mcuVEC<VType, MType>::copy_from_vector(std::vector<SType> vec)
{
	bool success = true;

	if (vec.size() != n.dim()) return false;

	//vec is linear memory which maps onto the larger cuVEC. Thus when copying to each gpu extract a std::vector by using the correct strides
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		cuSZ3 s = pbox_d[mGPU].s;
		cuSZ3 e = pbox_d[mGPU].e;
		success &= mng(mGPU)->copy_from_vector(subvec(vec, s.i, s.j, s.k, e.i, e.j, e.k, n.i, n.j, n.k));
	}

	return success;
}

//copy values to a std::vector (cpu memory)
template <typename VType, typename MType>
template <typename SType>
bool mcuVEC<VType, MType>::copy_to_vector(std::vector<SType>& vec)
{
	if (vec.size() != n.dim()) return false;

	//auxiliary subvector
	std::vector<SType> svec;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		//make sure svec has correct size, then copy from respective gpu
		if (!malloc_vector(svec, pn_d[mGPU].dim())) return false;
		mng(mGPU)->copy_to_vector(svec);

		//copy svec to vec with strided copy
		cuSZ3 s = pbox_d[mGPU].s;
		cuSZ3 e = pbox_d[mGPU].e;
		strided_copy(vec, svec, s.i, s.j, s.k, e.i, e.j, e.k, n.i, n.j, n.k);
	}

	return true;
}

//copy flags only from vec_vc, where sizes must match. Applicable to cuVEC_VC only
template <typename VType, typename MType>
template <typename cpuVEC_VC, typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
bool mcuVEC<VType, MType>::copyflags_from_cpuvec(cpuVEC_VC& vec_vc)
{
	bool success = true;

	//now copy data (and flags if applicable)
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		success &= mng(mGPU)->copyflags_from_cpuvec(vec_vc.subvec(pbox_d[mGPU]));
	}

	//pbcs must be handled now if VEC_VC
	set_pbc_from(vec_vc);

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	if (!success) clear();
	return success;
}

//--------------------------------------------COPY TO ANOTHER cuVEC : mcuVEC_mng.h

//extract values from this and place them in cuvec : both must have same rectangle, but can differ in cuVEC<VType>::h - cuvec.h <= this->cuVEC<VType>::h needed (and hence cuVEC<VType>::n); e.g. this method allows extraction of a coarser cuvec.
template <typename VType, typename MType>
void mcuVEC<VType, MType>::extract_cuvec(mcu_obj<MType, mcuVEC<VType, MType>>& mcuvec, cuBReal multiplier)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		mng(mGPU)->extract_cuvec(mcuvec.device_size(mGPU), mcuvec.get_deviceobject(mGPU), multiplier);
	}
}

//--------------------------------------------SIZING cuVEC_VC specific : mcuVEC_mng.h

//resize and set shape using linked vec. Applicable to cuVEC_VC only
template <typename VType, typename MType>
template <typename LVType, typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
bool mcuVEC<VType, MType>::resize(cuSZ3 new_n, mcu_obj<cuVEC_VC<LVType>, mcuVEC<LVType, cuVEC_VC<LVType>>>& linked_vec)
{
	//first set all correct dimensions
	if (!resize(new_n)) return false;

	//now go over each device and set shape (call resize on each, even though required dimensions already set; it will set shape from linked vec)
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (!mng(mGPU)->resize(pn_d[mGPU], linked_vec.get_deviceobject(mGPU))) return false;
	}

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	return true;
}

//resize and set shape using linked vec. Applicable to cuVEC_VC only
template <typename VType, typename MType>
template <typename LVType, typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
bool mcuVEC<VType, MType>::resize(cuReal3 new_h, cuRect new_rect, mcu_obj<cuVEC_VC<LVType>, mcuVEC<LVType, cuVEC_VC<LVType>>>& linked_vec)
{
	//first set all correct dimensions
	if (!resize(new_h, new_rect)) return false;

	//now go over each device and set shape (call resize on each, even though required dimensions already set; it will set shape from linked vec)
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (!mng(mGPU)->resize(h, prect_d[mGPU], linked_vec.get_deviceobject(mGPU))) return false;
	}

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	return true;
}

//set value and shape from linked vec
template <typename VType, typename MType>
template <typename LVType, typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
bool mcuVEC<VType, MType>::assign(cuSZ3 new_n, VType value, mcu_obj<cuVEC_VC<LVType>, mcuVEC<LVType, cuVEC_VC<LVType>>>& linked_vec)
{
	//first set all correct dimensions
	if (!resize(new_n)) return false;

	//now go over each device and set shape (call resize on each, even though required dimensions already set; it will set shape from linked vec)
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (!mng(mGPU)->assign(pn_d[mGPU], value, linked_vec.get_deviceobject(mGPU))) return false;
	}

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	return true;
}

//set value and shape from linked vec
template <typename VType, typename MType>
template <typename LVType, typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
bool mcuVEC<VType, MType>::assign(cuReal3 new_h, cuRect new_rect, VType value, mcu_obj<cuVEC_VC<LVType>, mcuVEC<LVType, cuVEC_VC<LVType>>>& linked_vec)
{
	//first set all correct dimensions
	if (!resize(new_h, new_rect)) return false;

	//now go over each device and set shape (call resize on each, even though required dimensions already set; it will set shape from linked vec)
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (!mng(mGPU)->assign(h, prect_d[mGPU], value, linked_vec.get_deviceobject(mGPU))) return false;
	}

	//synch dimensions in man_mcuVEC since they have changed here
	synch_dimensions();

	//allocate halo spaces
	allocate_halos();

	//set halo flags in managed devices
	set_halo_conditions();

	//setup histogram transfer data
	setup_histogram_transfers();

	return true;
}
