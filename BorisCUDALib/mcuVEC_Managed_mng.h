#pragma once

#include "mcuVEC_Managed.h"

//--------------------------------------------CONSTRUCTORS : mcuVEC_Managed_mng.h

//void constructor
template <typename MType, typename VType>
__host__ void mcuVEC_Managed<MType, VType>::construct_cu_obj(void)
{
	nullgpuptr(ppcuvec);

	nullgpuptr(pn_d);
	nullgpuptr(prect_d);
	nullgpuptr(pbox_d);

	set_gpu_value(num_devices, (int)0);
}

template <typename MType, typename VType>
__host__ void mcuVEC_Managed<MType, VType>::destruct_cu_obj(void)
{
	if (get_gpu_value(num_devices)) {

		gpu_free_managed(ppcuvec);
		nullgpuptr(ppcuvec);

		gpu_free_managed(pn_d);
		nullgpuptr(pn_d);

		gpu_free_managed(prect_d);
		nullgpuptr(prect_d);

		gpu_free_managed(pbox_d);
		nullgpuptr(pbox_d);
	}
}

template <typename MType, typename VType>
__host__ void mcuVEC_Managed<MType, VType>::set_pointers(int num_devices_, int device_idx, MType*& pcuvec)
{
	if (!get_gpu_value(num_devices)) {

		gpu_alloc_managed(ppcuvec, num_devices_);
		
		gpu_alloc_managed(pn_d, num_devices_);
		gpu_alloc_managed(prect_d, num_devices_);
		gpu_alloc_managed(pbox_d, num_devices_);

		set_gpu_value(num_devices, num_devices_);
	}

	cpu_to_gpu_managed(ppcuvec, &pcuvec, 1, device_idx);
}
