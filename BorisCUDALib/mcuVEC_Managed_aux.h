#pragma once

#include "mcuVEC_Managed.h"

//--------------------------------------------AUXILIARY

//take a global index (as would be used to index the entire logical cuVEC), and convert it to an index for the sub-cuVEC (return value) which is on device
//the device value is set in device passed through reference, and is also used as a hint to speed-up calculation (when called repeatedly, it's likely same device will be used many times in a row)
template <typename MType, typename VType>
__device__ size_t mcuVEC_Managed<MType, VType>::global_idx_to_device_idx(int i_global, int j_global, int k_global, int& device) const
{
	//check hint device first
	if (pbox_d[device].contains(i_global, j_global, k_global)) {

		int i_device = i_global - pbox_d[device].s.i;
		int j_device = j_global - pbox_d[device].s.j;
		int k_device = k_global - pbox_d[device].s.k;

		return i_device + j_device * pn_d[device].x + k_device * pn_d[device].x * pn_d[device].y;
	}

	//not in hint device. find which of pbox_d contains it
	for (int device_idx = 0; device_idx < num_devices; device_idx++) {

		if (device_idx == device) continue;

		if (pbox_d[device_idx].contains(i_global, j_global, k_global)) {

			//update hint device
			device = device_idx;

			int i_device = i_global - pbox_d[device].s.i;
			int j_device = j_global - pbox_d[device].s.j;
			int k_device = k_global - pbox_d[device].s.k;

			return i_device + j_device * pn_d[device].x + k_device * pn_d[device].x * pn_d[device].y;
		}
	}

	return 0;
}

template <typename MType, typename VType>
__device__ size_t mcuVEC_Managed<MType, VType>::global_idx_to_device_idx(const cuINT3& ijk_global, int& device) const
{
	//check hint device first
	if (pbox_d[device].contains(ijk_global)) {

		cuINT3 ijk_dev = ijk_global - pbox_d[device].s;

		return ijk_dev.i + ijk_dev.j * pn_d[device].x + ijk_dev.k * pn_d[device].x * pn_d[device].y;
	}

	//not in hint device. find which of pbox_d contains it
	for (int device_idx = 0; device_idx < num_devices; device_idx++) {

		if (device_idx == device) continue;

		if (pbox_d[device_idx].contains(ijk_global)) {

			//update hint device
			device = device_idx;

			cuINT3 ijk_dev = ijk_global - pbox_d[device].s;

			return ijk_dev.i + ijk_dev.j * pn_d[device].x + ijk_dev.k * pn_d[device].x * pn_d[device].y;
		}
	}

	return 0;
}

//rel_pos is relative to entire cuVEC. Find containing device and return position relative to that device.
template <typename MType, typename VType>
__device__ cuReal3 mcuVEC_Managed<MType, VType>::global_pos_to_device_pos(const cuReal3& rel_pos, int& device) const
{
	//check hint device first
	if (prect_d[device].contains(rel_pos + rect.s)) {

		return rel_pos + rect.s - prect_d[device].s;
	}

	//not in hint device. find which of pbox_d contains it
	for (int device_idx = 0; device_idx < num_devices; device_idx++) {

		if (device_idx == device) continue;

		if (prect_d[device_idx].contains(rel_pos + rect.s)) {

			//update hint device
			device = device_idx;

			return rel_pos + rect.s - prect_d[device_idx].s;
		}
	}

	return cuReal3();
}

//get index of cell which contains position (absolute value, not relative to start of rectangle), capped to mesh size
template <typename MType, typename VType>
__device__ cuINT3 mcuVEC_Managed<MType, VType>::cellidx_from_position(const cuReal3& absolute_position) const
{
	cuINT3 ijk = cuINT3(
		(int)cu_floor_epsilon((absolute_position.x - rect.s.x) / h.x),
		(int)cu_floor_epsilon((absolute_position.y - rect.s.y) / h.y),
		(int)cu_floor_epsilon((absolute_position.z - rect.s.z) / h.z));

	if (ijk.i < 0) ijk.i = 0;
	if (ijk.j < 0) ijk.j = 0;
	if (ijk.k < 0) ijk.k = 0;

	if (ijk.i > n.x) ijk.i = n.x;
	if (ijk.j > n.y) ijk.j = n.y;
	if (ijk.k > n.z) ijk.k = n.z;

	return ijk;
}

//extract box of cells intersecting with the given rectangle (rectangle is in absolute coordinates). Cells in box : from and including start, up to but not including end; Limited to cuVEC sizes.
template <typename MType, typename VType>
__device__ cuBox mcuVEC_Managed<MType, VType>::box_from_rect_max(const cuRect& rectangle) const
{
	if (!rectangle.intersects(rect)) return cuBox();

	//get start point. this will be limited to 0 to n (inclusive)
	cuINT3 start = cellidx_from_position(rectangle.s);

	//the Rect could be a plane rectangle on one of the surfaces of this mesh, so adjust start point for this
	if (start.x >= n.x) start.x = n.x - 1;
	if (start.y >= n.y) start.y = n.y - 1;
	if (start.z >= n.z) start.z = n.z - 1;

	//get end point. this will be limited to 0 to n (inclusive)
	cuINT3 end = cellidx_from_position(rectangle.e);

	cuReal3 snap = (h & end) + rect.s;

	//add 1 since end point must be included in the box, unless the rectangle end point is already at the end of a cell
	if ((cuIsNZ(snap.x - rectangle.e.x) || start.x == end.x) && end.x < n.x) end.x++;
	if ((cuIsNZ(snap.y - rectangle.e.y) || start.y == end.y) && end.y < n.y) end.y++;
	if ((cuIsNZ(snap.z - rectangle.e.z) || start.z == end.z) && end.z < n.z) end.z++;

	return cuBox(start, end);
}

//extract box of cells completely included in the given rectangle (rectangle is in absolute coordinates).
template <typename MType, typename VType>
__device__ cuBox mcuVEC_Managed<MType, VType>::box_from_rect_min(const cuRect& rectangle) const
{
	if (!rectangle.intersects(rect)) return cuBox();

	//get i,j,k indexes of cells containing the start and end points of mesh_intersection
	cuINT3 start = cellidx_from_position(rectangle.s);
	cuINT3 end = cellidx_from_position(rectangle.e);

	//adjust start so that Box(start, end) represents the set of cells completely included in mesh_intersection
	cuRect cell_rect = get_cellrect(start);
	if (!rectangle.contains(cell_rect)) start += cuINT3(1, 0, 0);

	cell_rect = get_cellrect(start);
	if (!rectangle.contains(cell_rect)) start += cuINT3(-1, 1, 0);

	cell_rect = get_cellrect(start);
	if (!rectangle.contains(cell_rect)) start += cuINT3(0, -1, 1);

	cell_rect = get_cellrect(start);
	if (!rectangle.contains(cell_rect)) start += cuINT3(1, 1, 0);

	return cuBox(start, end);
}