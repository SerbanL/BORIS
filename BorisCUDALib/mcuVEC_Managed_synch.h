#pragma once

#include "mcuVEC_Managed.h"

//--------------------------------------------mcuVEC Synchronization

template <typename MType, typename VType>
__host__ void mcuVEC_Managed<MType, VType>::synch_dimensions(
	int num_devices_,
	const cuSZ3& n_, const cuReal3& h_, const cuRect& rect_, 
	cuSZ3*& pn_d_, cuRect*& prect_d_, cuBox* pbox_d_)
{
	//overall logical cuVEC dimensions
	set_gpu_value(n, n_);
	set_gpu_value(h, h_);
	set_gpu_value(rect, rect_);

	//dimensions on each device
	cpu_to_gpu_managed(pn_d, pn_d_, num_devices_);
	cpu_to_gpu_managed(prect_d, prect_d_, num_devices_);
	cpu_to_gpu_managed(pbox_d, pbox_d_, num_devices_);
}