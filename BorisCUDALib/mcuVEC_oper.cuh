#pragma once

#include "mcuVEC.h"

#include "cuVEC_VC_oper.cuh"

//------------------------------------------------------------------- COPY VALUES

//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions
//can specify destination and source rectangles in relative coordinates
//this is intended for VECs where copy_this cellsize is much larger than that in this VEC, and instead of setting all values the same, thermalize_func generator will generate values
//e.g. this is useful for copying values from a micromagnetic mesh into an atomistic mesh, where the atomistic spins are generated according to a distribution setup in obj.thermalize_func
//obj.thermalize_func returns the value to set, and takes parameters VType (value in the larger cell from copy_this which is being copied), and int, int (index of larger cell from copy_this which is being copied, and index of destination cell)
//index in copy_this is for the entire mcuVEC, whilst index of destination cell is relative to respective device
//NOTE : can only be called in cu files (where it is possible to include mcuVEC_oper.cuh), otherwise explicit template parameters would have to be declared, which is too restrictive.
template <typename VType, typename MType>
template <typename Class_Thermalize, class Class_ThermalizePolicy, typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::copy_values_thermalize(mcu_obj<cuVEC_VC<VType>, mcuVEC<VType, cuVEC_VC<VType>>>& copy_this, mcu_obj<Class_Thermalize, Class_ThermalizePolicy>& obj, cuBox cells_box_dst, cuBox cells_box_src, mcu_obj<cuBorisRand<>, mcuBorisRand>& prng, bool recalculate_flags)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		cuBox cells_box_dst_device = cells_box_dst.get_intersection(pbox_d[mGPU]);
		if (!cells_box_dst_device.IsNull()) {

			//destination box for each device (intersection of device box with cells_box_dst) but must be relative to each device
			//also pass in source and destination boxes for entire mcuVECs
			mng(mGPU)->copy_values_thermalize_mcuVEC(copy_this.get_deviceobject(mGPU), obj.get_deviceobject(mGPU), cells_box_dst_device - pbox_d[mGPU].s, cells_box_dst, cells_box_src, prng.get_deviceobject(mGPU), recalculate_flags);
		}
	}

	if (recalculate_flags) set_halo_conditions();
}