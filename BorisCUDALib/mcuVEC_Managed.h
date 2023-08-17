#pragma once

#include "cuVEC_VC.h"

//This is a cu-obj managed object containing cuVEC(_VC) across multiple GPUs
//It holds pointers to cuVEC(_VC) across all devices available, which are managed by the mcuVEC policy class
//It can be passed to CUDA kernels, where access to memory relies on UVA being available

//Whilst the cuVEC(_VC) objects managed by mcuVEC are only small parts (one for each device) of an entire logical cuVEC, mcuVEC_Managed appears to each CUDA kernel as this entire logical cuVEC, even though it is spread across multiple devices
//Thus mcuVEC_Managed is useful when we need to access data and methods without having to think about which device to use
//e.g. for use with CMBND where access to values in the secondary cuVEC cannot be guaranteed to be on same device
//e.g. also useful for weighted_average, where the averaging region can span 2 or more devices etc.

//Using the separate cuVECs for each device in general is the most efficient way, so only use mcuVEC_Managed when needed.

//NOTE that mcuVEC_Managed is not stand-alone, but it's actually held in mcuVEC also (and managed by it), so in host code you just declare a mcuVEC.

template <typename MType, typename VType>
class mcuVEC_Managed
{
private:

	//array of pointers to cuVECs all in GPU memory (only one address will be on same device, the rest are on other devices, but can access them using UVA)
	MType** ppcuvec;

	//dimensions and rectangles of each cuVEC for each device
	cuSZ3* pn_d;
	cuRect* prect_d;

	//box for each device (box relative to entire cuVEC, i.e. cell start and end box coordinates make sense in cuBox(n), where n dimensions of entire cuVEC)
	cuBox* pbox_d;

	//number of devices available (i.e. size of above arrays)
	int num_devices;

public:

	//----------------- (cuVEC)

	//overall dimensions along x, y and z of the quantity
	cuSZ3 n;

	//cellsize of structured mesh
	cuReal3 h;

	//overall rectangle, same units as h. VEC has n number of cells, so n * h gives the rect dimensions
	cuRect rect;

private:

	//--------------------------------------------AUXILIARY : mcuVEC_Managed_aux.h

	//take a global index (as would be used to index the entire logical cuVEC), and convert it to an index for the sub-cuVEC (return value) which is on device
	//the device value is set in device passed through reference, and is also used as a hint to speed-up calculation (when called repeatedly, it's likely same device will be used many times in a row)
	__device__ size_t global_idx_to_device_idx(int i_global, int j_global, int k_global, int& device) const;
	__device__ size_t global_idx_to_device_idx(const cuINT3& ijk_global, int& device) const;

	//rel_pos is relative to entire cuVEC. Find containing device and return position relative to that device.
	__device__ cuReal3 global_pos_to_device_pos(const cuReal3& rel_pos, int& device) const;

public:

	//--------------------------------------------CONSTRUCTORS : cu_obj "managed constructors" only. Real constructors are never called since you should never make a real instance of a cuVEC. : mcuVEC_Managed_mng.h

	//void constructor
	__host__ void construct_cu_obj(void);

	__host__ void destruct_cu_obj(void);

	__host__ void set_pointers(int num_devices, int device_idx, MType*& pcuvec);

	//--------------------------------------------mcuVEC Synchronization

	__host__ void synch_dimensions(
		int num_devices_,
		const cuSZ3& n_, const cuReal3& h_, const cuRect& rect_,
		cuSZ3*& pn_d_, cuRect*& prect_d_, cuBox* pbox_d_);

	//--------------------------------------------INDEXING

	__device__ VType& operator[](const cuINT3& ijk)
	{
		int device = 0;
		size_t dev_idx = global_idx_to_device_idx(ijk, device);
		return (*ppcuvec[device])[dev_idx];
	}

	__device__ VType& operator[](const cuReal3& rel_pos)
	{
		int device = 0;
		cuReal3 devrel_pos = global_pos_to_device_pos(rel_pos, device);
		return (*ppcuvec[device])[devrel_pos];
	}

	//--------------------------------------------FLAG CHECKING

	__device__ bool is_empty(const cuINT3& ijk) const
	{
		int device = 0;
		size_t dev_idx = global_idx_to_device_idx(ijk, device);
		return ppcuvec[device]->is_empty(dev_idx);
	}

	__device__ bool is_empty(const cuReal3& rel_pos) const
	{
		int device = 0;
		cuReal3 devrel_pos = global_pos_to_device_pos(rel_pos, device);
		return ppcuvec[device]->is_empty(devrel_pos);
	}

	//--------------------------------------------GETTERS : mcuVEC_Managed_aux.h

	//get index of cell which contains position (absolute value, not relative to start of rectangle), capped to mesh size
	__device__ cuINT3 cellidx_from_position(const cuReal3& absolute_position)  const;

	//get cell rectangle (absolute values, not relative to start of mesh rectangle) for cell with index ijk
	__device__ cuRect get_cellrect(const cuINT3& ijk)  const { return cuRect(rect.s + (h & ijk), rect.s + (h & ijk) + h); }

	//extract box of cells intersecting with the given rectangle (rectangle is in absolute coordinates). Cells in box : from and including start, up to but not including end; Limited to cuVEC sizes.
	__device__ cuBox box_from_rect_max(const cuRect& rectangle) const;

	//extract box of cells completely included in the given rectangle (rectangle is in absolute coordinates).
	__device__ cuBox box_from_rect_min(const cuRect& rectangle)  const;

	//--------------------------------------------AVERAGING OPERATIONS : mcuVEC_Managed_avg.cuh

	//full average in given rectangle (relative coordinates to entire cuVEC).
	__device__ VType average(const cuRect& rectangle);

	//get weighted average around coord in given stencil, where coord is relative to entire cuVEC
	__device__ VType weighted_average(const cuReal3& coord, const cuReal3& stencil);

	//--------------------------------------------TESTING

	__device__ MType& get_deviceobject(int device) { return *ppcuvec[device]; }
};
