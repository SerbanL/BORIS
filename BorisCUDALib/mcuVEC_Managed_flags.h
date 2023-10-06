#pragma once

#include "mcuVEC_Managed.h"

//--------------------------------------------FLAG CHECKING

template <typename MType, typename VType>
__device__ bool mcuVEC_Managed<MType, VType>::is_empty(const cuRect& rectangle) const
{
	cuBox cells = box_from_rect_max(rectangle);

	for (int i = (cells.s.x >= 0 ? cells.s.x : 0); i < (cells.e.x <= n.x ? cells.e.x : n.x); i++) {
		for (int j = (cells.s.y >= 0 ? cells.s.y : 0); j < (cells.e.y <= n.y ? cells.e.y : n.y); j++) {
			for (int k = (cells.s.z >= 0 ? cells.s.z : 0); k < (cells.e.z <= n.z ? cells.e.z : n.z); k++) {

				if (!is_empty(cuINT3(i, j, k))) return false;
			}
		}
	}

	return true;
}