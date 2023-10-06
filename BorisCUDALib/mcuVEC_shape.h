#pragma once

#include "mcuVEC.h"

//------------------------------------------------------------------- SETBOX

//set value in box
template <typename VType, typename MType>
void mcuVEC<VType, MType>::setbox(cuBox box, VType value)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		mng(mGPU)->setbox(box.get_intersection(pbox_d[mGPU]) - pbox_d[mGPU].s, value);
	}

	//set halo flags in managed devices (need to set halo flags again since shape could have changed)
	set_halo_conditions();
}

//------------------------------------------------------------------- SETRECT

//set value in rectangle (i.e. in cells intersecting the rectangle), where the rectangle is relative to this cuVEC's rectangle.
template <typename VType, typename MType>
void mcuVEC<VType, MType>::setrect(const cuRect& rectangle, VType value)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		mng(mGPU)->setrect(rectangle.get_intersection(prect_d[mGPU] - rect.s) + rect.s - prect_d[mGPU].s, value);
	}

	//set halo flags in managed devices (need to set halo flags again since shape could have changed)
	set_halo_conditions();
}

//------------------------------------------------------------------- DELRECT

//delete rectangle, where the rectangle is relative to this VEC's rectangle, by setting empty cell values - all cells become empty cells. Applicable to cuVEC_VC only
template <typename VType, typename MType>
template <typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::delrect(cuRect rectangle, bool recalculate_flags)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		mng(mGPU)->delrect(rectangle.get_intersection(prect_d[mGPU] - rect.s) + rect.s - prect_d[mGPU].s, recalculate_flags);
	}

	//set halo flags in managed devices (need to set halo flags again since shape could have changed)
	if (recalculate_flags) set_halo_conditions();
}

//------------------------------------------------------------------- BITMAP MASK

//mask values in cells using bitmap image : white -> empty cells. black -> keep values. Apply mask up to given z depth number of cells depending on grayscale value (zDepth, all if 0). Applicable to cuVEC_VC only
template <typename VType, typename MType>
template <typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
bool mcuVEC<VType, MType>::apply_bitmap_mask(std::vector<unsigned char>& bitmap, int zDepth)
{
	bool success = true;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		cuBox box = pbox_d[mGPU];
		std::vector<unsigned char> sub_bitmap = subvec(bitmap, box.s.i, box.s.j, 0, box.e.i, box.e.j, 1, n.i, n.j, 1);

		success &= mng(mGPU)->apply_bitmap_mask(sub_bitmap, zDepth);
	}

	//set halo flags in managed devices (need to set halo flags again since shape could have changed)
	set_halo_conditions();

	return success;
}