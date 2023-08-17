#pragma once

#include "VEC_VC.h"

//--------------------------------------------EXTRACT A LINE PROFILE

//extract profile in profile_storage temporary vector, returned through reference: extract starting at start in the direction end - step, with given step; use average to extract profile with given stencil, excluding zero points (assumed empty)
//all coordinates are relative positions
template <typename VType>
std::vector<VType>& VEC_VC<VType>::extract_profile(DBL3 start, DBL3 end, double step, DBL3 stencil)
{
	if (step > 0) {

		size_t size = round((end - start).norm() / step) + 1;
		if (VEC<VType>::line_profile_storage.size() != size && !malloc_vector(VEC<VType>::line_profile_storage, size)) {

			VEC<VType>::line_profile_storage.clear();
			return VEC<VType>::line_profile_storage;
		}

		DBL3 meshDim = VEC<VType>::rect.size();

#pragma omp parallel for
		for (int idx = 0; idx < size; idx++) {

			//position wrapped-around
			DBL3 position = (start + (double)idx * step * (end - start).normalized()) % meshDim;

			VType value = VEC_VC<VType>::average_nonempty(Rect(position - stencil / 2, position + stencil / 2));
			VEC<VType>::line_profile_storage[idx] = value;
		}
	}
	else VEC<VType>::line_profile_storage.clear();

	return VEC<VType>::line_profile_storage;
}