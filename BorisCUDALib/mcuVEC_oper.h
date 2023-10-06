#pragma once

#include "mcuVEC.h"

#include "mcu_prng.h"

//------------------------------------------------------------------- SET

//set value in all cells
template <typename VType, typename MType>
void mcuVEC<VType, MType>::set(VType value)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		mng(mGPU)->set(pn_d[mGPU].dim(), value);
	}
}

//------------------------------------------------------------------- RENORMALIZE

//re-normalize all non-zero values to have the new magnitude (multiply by new_norm and divide by current magnitude)
template <typename VType, typename MType>
template <typename PType>
void mcuVEC<VType, MType>::renormalize(PType new_norm)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		mng(mGPU)->renormalize(pn_d[mGPU].dim(), new_norm);
	}
}

//------------------------------------------------------------------- SETNONEMPTY

//exactly the same as assign value - do not use assign as it is slow (sets flags). Applicable to cuVEC_VC only
template <typename VType, typename MType>
template <typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::setnonempty(VType value)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		mng(mGPU)->setnonempty(value);
	}
}

//------------------------------------------------------------------- SET RECT NONEMPTY

//set value in non-empty cells only in given rectangle (relative coordinates). Applicable to cuVEC_VC only
template <typename VType, typename MType>
template <typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::setrectnonempty(const cuRect& rectangle, VType value)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		mng(mGPU)->setrectnonempty(rectangle.get_intersection(prect_d[mGPU] - rect.s) + rect.s - prect_d[mGPU].s, value);
	}
}

//------------------------------------------------------------------- COPY VALUES

//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions; from flags only copy the shape but not the boundary condition values or anything else - these are reset
template <typename VType, typename MType>
void mcuVEC<VType, MType>::copy_values(mcu_obj<cuVEC<VType>, mcuVEC<VType, cuVEC<VType>>>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		cuBox cells_box_dst_device = cells_box_dst.get_intersection(pbox_d[mGPU]);
		if (!cells_box_dst_device.IsNull()) {

			//destination box for each device (intersection of device box with cells_box_dst) but must be relative to each device
			//also pass in source and destination boxes for entire mcuVECs
			mng(mGPU)->copy_values_mcuVEC(copy_this.get_deviceobject(mGPU), cells_box_dst_device - pbox_d[mGPU].s, cells_box_dst, cells_box_src, multiplier);
		}
	}
}


template <typename VType, typename MType>
template <typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::copy_values(mcu_obj<cuVEC_VC<VType>, mcuVEC<VType, cuVEC_VC<VType>>>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		cuBox cells_box_dst_device = cells_box_dst.get_intersection(pbox_d[mGPU]);
		if (!cells_box_dst_device.IsNull()) {
			
			//destination box for each device (intersection of device box with cells_box_dst) but must be relative to each device
			//also pass in source and destination boxes for entire mcuVECs
			mng(mGPU)->copy_values_mcuVEC(copy_this.get_deviceobject(mGPU), cells_box_dst_device - pbox_d[mGPU].s, cells_box_dst, cells_box_src, multiplier, recalculate_flags);
		}
	}

	if (recalculate_flags) set_halo_conditions();
}

//------------------------------------------------------------------- SHIFT

//shift all the values in this cuVEC by the given delta (units same as cuVEC<VType>::h). Shift values in given shift_rect (absolute coordinates). Applicable to cuVEC_VC only
template <typename VType, typename MType>
template <typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::shift_x(cuBReal delta, cuRect shift_rect, bool recalculate_flags)
{
	bool shift_again = false;

	do {

		shift_again = false;

		//before attempting shifts, must exchange halos as these are used for shifting values at boundaries for each gpu
		//for shift_x we only need to do this if sub-rectangles are arranged along x
		//must force halo exchange in case UVA is being used (can't use UVA with shift algorithm - without special modification - due to data races)
		if (halo_flag == NF2_HALOX) exchange_halos(true);

		//now we can do the shifts
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			//must force single cell shifts if multiple devices are being used, since halos must be exchanged after each step
			shift_again |= mng(mGPU)->shift_x(pn_d[mGPU].dim(), delta, shift_rect.get_intersection(prect_d[mGPU]), recalculate_flags, mGPU.get_num_devices() > 1);
		}

		//this was banked in shift_debt so set it to zero if performing another cell shift
		if (shift_again) {

			delta = 0.0;
			//also need to set halo flags again as these are used
			set_halo_conditions();
		}

	} while (shift_again);

	if (recalculate_flags) set_halo_conditions();
}
