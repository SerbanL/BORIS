#pragma once

#include "cuVEC_VC.h"
#include "cuFuncs_Math.h"
#include "launchers.h"
#include "Reduction.cuh"
#include "mcuVEC_Managed.h"

//--------------------------------------------MULTIPLE ENTRIES SETTERS - OTHERS

//------------------------------------------------------------------- SETNONEMPTY

template <typename VType>
__global__ void setnonempty_kernel(cuSZ3& n, int*& ngbrFlags, VType*& quantity, VType value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n.dim()) {

		if (ngbrFlags[idx] & NF_NOTEMPTY) {

			quantity[idx] = value;
		}
	}
}

template void cuVEC_VC<float>::setnonempty(float value);
template void cuVEC_VC<double>::setnonempty(double value);

template void cuVEC_VC<cuFLT3>::setnonempty(cuFLT3 value);
template void cuVEC_VC<cuDBL3>::setnonempty(cuDBL3 value);

template <typename VType>
__host__ void cuVEC_VC<VType>::setnonempty(VType value)
{
	setnonempty_kernel <<< (get_gpu_value(cuVEC<VType>::n).dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> > (cuVEC<VType>::n, ngbrFlags, cuVEC<VType>::quantity, value);
}

//------------------------------------------------------------------- SETRECTNONEMPTY

template <typename VType>
__global__ void setrectnonempty_kernel(cuSZ3& n, int*& ngbrFlags, VType*& quantity, cuBox box, VType value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x*n.y));

	if (idx < n.dim() && box.contains(ijk)) {

		if (ngbrFlags[idx] & NF_NOTEMPTY) {

			quantity[idx] = value;
		}
	}
}

template void cuVEC_VC<float>::setrectnonempty(const cuRect& rectangle, float value);
template void cuVEC_VC<double>::setrectnonempty(const cuRect& rectangle, double value);

template void cuVEC_VC<cuFLT3>::setrectnonempty(const cuRect& rectangle, cuFLT3 value);
template void cuVEC_VC<cuDBL3>::setrectnonempty(const cuRect& rectangle, cuDBL3 value);

//set value in non-empty cells only in given rectangle (relative coordinates)
template <typename VType>
__host__ void cuVEC_VC<VType>::setrectnonempty(const cuRect& rectangle, VType value)
{
	cuBox box = cuVEC<VType>::box_from_rect_max_cpu(rectangle + get_gpu_value(cuVEC<VType>::rect).s);

	setrectnonempty_kernel <<< (get_gpu_value(cuVEC<VType>::n).dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (cuVEC<VType>::n, ngbrFlags, cuVEC<VType>::quantity, box, value);
}

//------------------------------------------------------------------- RENORMALIZE (cuVEC_VC)

template <typename VType, typename PType>
__global__ void cuvec_vc_renormalize_kernel(cuSZ3& n, int*& ngbrFlags, VType*& quantity, PType new_norm)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n.dim()) {

		PType curr_norm = cu_GetMagnitude(quantity[idx]);

		if ((ngbrFlags[idx] & NF_NOTEMPTY) && cuIsNZ(curr_norm)) {

			quantity[idx] *= new_norm / curr_norm;
		}
	}
}

template void cuVEC_VC<float>::renormalize(size_t arr_size, float new_norm);
template void cuVEC_VC<double>::renormalize(size_t arr_size, double new_norm);

template void cuVEC_VC<cuFLT3>::renormalize(size_t arr_size, float new_norm);
template void cuVEC_VC<cuDBL3>::renormalize(size_t arr_size, double new_norm);

//re-normalize all non-zero values to have the new magnitude (multiply by new_norm and divide by current magnitude)
template <typename VType>
template <typename PType>
__host__ void cuVEC_VC<VType>::renormalize(size_t arr_size, PType new_norm)
{
	cuvec_vc_renormalize_kernel <<< (arr_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (cuVEC<VType>::n, ngbrFlags, cuVEC<VType>::quantity, new_norm);
}

//------------------------------------------------------------------- COPY VALUES (DIRECT VERSION)

template <typename VType>
__global__ void copy_values_kernel(cuVEC_VC<VType>& to_this, cuVEC_VC<VType>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier)
{
	int idx_box_dst = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 dst_n = cells_box_dst.size();
	cuRect srcRect = cuRect(cells_box_src.s & copy_this.h, cells_box_src.e & copy_this.h);
	cuRect dstRect = cuRect(cells_box_dst.s & to_this.h, cells_box_dst.e & to_this.h);
	cuReal3 lRatio = dstRect.size() / srcRect.size();

	if (idx_box_dst < dst_n.dim()) {

		int i = idx_box_dst % dst_n.i;
		int j = (idx_box_dst / dst_n.i) % dst_n.j;
		int k = idx_box_dst / (dst_n.i * dst_n.j);

		int idx_out = (i + cells_box_dst.s.i) + (j + cells_box_dst.s.j) * to_this.n.x + (k + cells_box_dst.s.k) * to_this.n.x * to_this.n.y;

		//destination cell rectangle
		cuRect dst_cell_rect_rel = to_this.get_cellrect(idx_out) - to_this.rect.s - dstRect.s;

		//now map this to source rectangle
		cuRect src_cell_rect_rel = cuRect(dst_cell_rect_rel.s & lRatio, dst_cell_rect_rel.e & lRatio) + srcRect.s;

		if (idx_out < to_this.n.dim()) {

			to_this[idx_out] = copy_this.average(src_cell_rect_rel) * multiplier;

			if (copy_this.is_empty(src_cell_rect_rel + copy_this.rect.s)) to_this.mark_empty(idx_out);
			else to_this.mark_not_empty(idx_out);
		}
	}
}

template void cuVEC_VC<float>::copy_values(cuVEC_VC<float>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags);
template void cuVEC_VC<double>::copy_values(cuVEC_VC<double>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags);

template void cuVEC_VC<cuFLT3>::copy_values(cuVEC_VC<cuFLT3>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags);
template void cuVEC_VC<cuDBL3>::copy_values(cuVEC_VC<cuDBL3>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags);

//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions; from flags only copy the shape but not the boundary condition values or anything else - these are reset
template <typename VType>
__host__ void cuVEC_VC<VType>::copy_values(cuVEC_VC<VType>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags)
{
	copy_values_kernel <<< (cells_box_dst.size().dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (*this, copy_this, cells_box_dst, cells_box_src, multiplier);

	//recalculate neighbor flags : current shape is maintained.
	if (recalculate_flags) set_ngbrFlags();
}

//------------------------------------------------------------------- COPY VALUES (MCUVEC VERSION)

template <typename VType>
__global__ void copy_values_mcuVEC_kernel(cuVEC_VC<VType>& to_this, cuVEC_VC<VType>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier)
{
	int idx_box_dst = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 dst_n = cells_box_dst_device.size();
	cuRect srcRect = cuRect(cells_box_src.s & copy_this.h, cells_box_src.e & copy_this.h);
	cuRect dstRect = cuRect(cells_box_dst.s & to_this.h, cells_box_dst.e & to_this.h);
	cuReal3 lRatio = dstRect.size() / srcRect.size();

	if (idx_box_dst < dst_n.dim()) {

		int i = idx_box_dst % dst_n.i;
		int j = (idx_box_dst / dst_n.i) % dst_n.j;
		int k = idx_box_dst / (dst_n.i * dst_n.j);

		int idx_out = (i + cells_box_dst.s.i) + (j + cells_box_dst.s.j) * to_this.n.x + (k + cells_box_dst.s.k) * to_this.n.x * to_this.n.y;

		//destination cell rectangle relative to destination rectangle of entire mcuVEC
		cuRect dst_cell_rect_rel = to_this.get_cellrect(idx_out) - to_this.origin - dstRect.s;

		//now map this to source rectangle of entire mcuVEC
		cuRect src_cell_rect_rel = cuRect(dst_cell_rect_rel.s & lRatio, dst_cell_rect_rel.e & lRatio) + srcRect.s;

		if (idx_out < to_this.n.dim()) {

			to_this[idx_out] = copy_this.mcuvec().average(src_cell_rect_rel) * multiplier;

			if (copy_this.mcuvec().is_empty(src_cell_rect_rel + copy_this.origin)) to_this.mark_empty(idx_out);
			else to_this.mark_not_empty(idx_out);
			
		}
	}
}

//special modification of copy_values method above, where we use copy_this.mcuvec() instead of copy_this directly
//this should only be used by mcuVEC
template void cuVEC_VC<float>::copy_values_mcuVEC(cuVEC_VC<float>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags);
template void cuVEC_VC<double>::copy_values_mcuVEC(cuVEC_VC<double>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags);

template void cuVEC_VC<cuFLT3>::copy_values_mcuVEC(cuVEC_VC<cuFLT3>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags);
template void cuVEC_VC<cuDBL3>::copy_values_mcuVEC(cuVEC_VC<cuDBL3>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags);

//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions; from flags only copy the shape but not the boundary condition values or anything else - these are reset
template <typename VType>
__host__ void cuVEC_VC<VType>::copy_values_mcuVEC(cuVEC_VC<VType>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier, bool recalculate_flags)
{
	copy_values_mcuVEC_kernel <<< (cells_box_dst_device.size().dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (*this, copy_this, cells_box_dst_device, cells_box_dst, cells_box_src, multiplier);

	//recalculate neighbor flags : current shape is maintained.
	if (recalculate_flags) set_ngbrFlags();
}

//------------------------------------------------------------------- COPY VALUES THERMALIZE (DIRECT VERSION)

template <typename VType, typename Class_Thermalize>
__global__ void copy_values_thermalize_kernel(cuVEC_VC<VType>& to_this, cuVEC_VC<VType>& copy_this, Class_Thermalize& obj, cuBox cells_box_dst, cuBox cells_box_src, cuBorisRand<>& prng)
{
	int idx_box_dst = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 dst_n = cells_box_dst.size();
	cuSZ3 src_n = cells_box_src.size();

	cuReal3 sourceIdx = (cuReal3)src_n / dst_n;

	if (idx_box_dst < dst_n.dim()) {

		int i = idx_box_dst % dst_n.i;
		int j = (idx_box_dst / dst_n.i) % dst_n.j;
		int k = idx_box_dst / (dst_n.i * dst_n.j);

		int _x = (int)floor(i * sourceIdx.x);
		int _y = (int)floor(j * sourceIdx.y);
		int _z = (int)floor(k * sourceIdx.z);

		int idx_out = (i + cells_box_dst.s.i) + (j + cells_box_dst.s.j) * to_this.n.x + (k + cells_box_dst.s.k) * to_this.n.x * to_this.n.y;
		int idx_in = (_x + cells_box_src.s.i) + (_y + cells_box_src.s.j) * copy_this.n.x + (_z + cells_box_src.s.k) * (copy_this.n.x * copy_this.n.y);

		if (idx_out < to_this.n.dim() && idx_in < copy_this.n.dim()) {

			to_this[idx_out] = obj.thermalize_func(copy_this[idx_in], idx_in, idx_out, prng);

			if (copy_this.is_empty(idx_in)) to_this.mark_empty(idx_out);
			else to_this.mark_not_empty(idx_out);
		}
	}
}

//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions
//can specify destination and source rectangles in relative coordinates
//this is intended for VECs where copy_this cellsize is much larger than that in this VEC, and instead of setting all values the same, thermalize_func generator will generate values
//e.g. this is useful for copying values from a micromagnetic mesh into an atomistic mesh, where the atomistic spins are generated according to a distribution setup in obj.thermalize_func
//obj.thermalize_func returns the value to set, and takes parameters VType (value in the larger cell from copy_this which is being copied), and int, int (index of larger cell from copy_this which is being copied, and index of destination cell)
//NOTE : can only be called in cu files (where it is possible to include cuVEC_VC_oper.cuh), otherwise explicit template parameters would have to be declared, which is too restrictive.
template <typename VType>
template <typename Class_Thermalize>
__host__ void cuVEC_VC<VType>::copy_values_thermalize(cuVEC_VC<VType>& copy_this, Class_Thermalize& obj, cuBox cells_box_dst, cuBox cells_box_src, cuBorisRand<>& prng, bool recalculate_flags)
{
	copy_values_thermalize_kernel <<< (cells_box_dst.size().dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (*this, copy_this, obj, cells_box_dst, cells_box_src, prng);

	//recalculate neighbor flags : current shape is maintained.
	if (recalculate_flags) set_ngbrFlags();
}

//------------------------------------------------------------------- COPY VALUES THERMALIZE (MCUVEC VERSION)

template <typename VType, typename Class_Thermalize>
__global__ void copy_values_thermalize_mcuVEC_kernel(cuVEC_VC<VType>& to_this, cuVEC_VC<VType>& copy_this, Class_Thermalize& obj, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBorisRand<>& prng)
{
	int idx_box_dst = blockIdx.x * blockDim.x + threadIdx.x;

	mcuVEC_Managed<cuVEC_VC<VType>, VType>& copy_this_mcuvec = copy_this.mcuvec();

	cuSZ3 dst_n = cells_box_dst_device.size();
	//source rectangle of entire mcuVEC
	cuRect srcRect = cuRect(cells_box_src.s & copy_this_mcuvec.h, cells_box_src.e & copy_this_mcuvec.h);
	//destination rectangle of entire mcuVEC
	cuRect dstRect = cuRect(cells_box_dst.s & to_this.h, cells_box_dst.e & to_this.h);
	cuReal3 lRatio = dstRect.size() / srcRect.size();

	if (idx_box_dst < dst_n.dim()) {

		//i, j, k on this device
		int i = (idx_box_dst % dst_n.i) + cells_box_dst_device.s.i;
		int j = ((idx_box_dst / dst_n.i) % dst_n.j) + cells_box_dst_device.s.j;
		int k = idx_box_dst / (dst_n.i * dst_n.j) + cells_box_dst_device.s.k;

		//linear output index on this device
		int idx_out = i + j * to_this.n.x + k * to_this.n.x * to_this.n.y;

		//destination cell position relative to destination rectangle of entire mcuVEC
		cuReal3 dst_cell_pos_rel = to_this.position_from_cellidx(cuINT3(i, j, k)) - to_this.mcuvec().rect.s - dstRect.s;

		//now map this to source rectangle of entire mcuVEC
		cuReal3 src_cell_pos_rel = (dst_cell_pos_rel & lRatio) + srcRect.s;

		//linear input index in entire source mcuVEC
		int idx_in = copy_this_mcuvec.position_to_cellidx(src_cell_pos_rel);

		if (idx_out < to_this.n.dim() && idx_in < copy_this_mcuvec.n.dim()) {

			to_this[idx_out] = obj.thermalize_func(copy_this_mcuvec[src_cell_pos_rel], idx_in, idx_out, prng);

			if (copy_this_mcuvec.is_empty(src_cell_pos_rel)) to_this.mark_empty(idx_out);
			else to_this.mark_not_empty(idx_out);
		}
	}
}

//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions
//can specify destination and source rectangles in relative coordinates
//this is intended for VECs where copy_this cellsize is much larger than that in this VEC, and instead of setting all values the same, thermalize_func generator will generate values
//e.g. this is useful for copying values from a micromagnetic mesh into an atomistic mesh, where the atomistic spins are generated according to a distribution setup in obj.thermalize_func
//obj.thermalize_func returns the value to set, and takes parameters VType (value in the larger cell from copy_this which is being copied), and int, int (index of larger cell from copy_this which is being copied, and index of destination cell)
//NOTE : can only be called in cu files (where it is possible to include cuVEC_VC_oper.cuh), otherwise explicit template parameters would have to be declared, which is too restrictive.
template <typename VType>
template <typename Class_Thermalize>
__host__ void cuVEC_VC<VType>::copy_values_thermalize_mcuVEC(cuVEC_VC<VType>& copy_this, Class_Thermalize& obj, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBorisRand<>& prng, bool recalculate_flags)
{
	copy_values_thermalize_mcuVEC_kernel <<< (cells_box_dst_device.size().dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (*this, copy_this, obj, cells_box_dst_device, cells_box_dst, cells_box_src, prng);

	//recalculate neighbor flags : current shape is maintained.
	if (recalculate_flags) set_ngbrFlags();
}

//------------------------------------------------------------------- SHIFT : x

template <typename VType>
__global__ void shift_x_left1_kernel(cuRect shift_rect, cuVEC_VC<VType>& vec)
{
	__shared__ VType shared_memory[CUDATHREADS];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = vec.n;
	cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

	cuBox shift_box = vec.box_from_rect_min(shift_rect);
	//shift_box.e.x - 2 takes value from shift_box.e.x - 1
	shift_box.e.x--;

	if (idx < n.dim()) {

		if (vec.is_not_empty(idx)) shared_memory[threadIdx.x] = vec[idx];
		else shared_memory[threadIdx.x] = VType();
	}
	else shared_memory[threadIdx.x] = VType();

	//all values in this block must be transferred to shared memory before proceeding
	__syncthreads();

	//shift values within the box using shared_memory (both destination and source must not be empty).
	//We cannot shift the first element in this block since it must go to the block before this - store it in aux_block_values for later.
	//Similarly we cannot write the last element in this block since it needs a value from the next block.
	if (threadIdx.x == 0) vec.aux_block_values_ref()[blockIdx.x] = shared_memory[0];

	//special treatment for cells at +x boundary, if halos are being used : get value from halo
	if (ijk.i == n.x - 1 && vec.halo_p_ref()) {

		if (vec.ngbrFlags2_ref() && (vec.ngbrFlags2_ref()[idx] & NF2_HALOPX)) {

			vec[idx] = vec.halo_p_ref()[ijk.j + ijk.k * n.y];
			vec.mark_not_empty(idx);
		}
		else {

			vec[idx] = VType();
			vec.mark_empty(idx);
		}
	}
	else if (shift_box.contains(ijk)) {

		if (threadIdx.x < CUDATHREADS - 1) {

			vec[idx] = shared_memory[threadIdx.x + 1];

			//important to shift shape as well (but check for empty cell by value, otherwise we have to shift ngbrFlags as well)
			if (vec[idx] == VType()) vec.mark_empty(idx);
			else vec.mark_not_empty(idx);
		}
	}
}

template <typename VType>
__global__ void shift_x_left1_stitch_kernel(cuRect shift_rect, cuVEC_VC<VType>& vec)
{
	//index in aux_block_values
	int aux_blocks_idx = blockIdx.x * blockDim.x + threadIdx.x;

	//index in vec : aux_block_values stored block beginning values which must be shifted to the cell to the left
	int cell_idx = aux_blocks_idx * CUDATHREADS - 1;

	cuSZ3 n = vec.n;
	cuINT3 ijk = cuINT3(cell_idx % n.x, (cell_idx / n.x) % n.y, cell_idx / (n.x * n.y));

	cuBox shift_box = vec.box_from_rect_min(shift_rect);
	shift_box.e.x--;

	if (shift_box.contains(ijk)) {

		vec[cell_idx] = vec.aux_block_values_ref()[aux_blocks_idx];

		//important to shift shape as well (but check for empty cell by value, otherwise we have to shift ngbrFlags as well)
		if (vec[cell_idx] == VType()) vec.mark_empty(cell_idx);
		else vec.mark_not_empty(cell_idx);
	}
}

template <typename VType>
__global__ void shift_x_right1_kernel(cuRect shift_rect, cuVEC_VC<VType>& vec)
{
	__shared__ VType shared_memory[CUDATHREADS];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = vec.n;
	cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

	cuBox shift_box = vec.box_from_rect_min(shift_rect);
	//shift_box.s.x + 1 takes value from shift_box.s.x
	shift_box.s.x++;

	if (idx < n.dim()) {

		if (vec.is_not_empty(idx)) shared_memory[threadIdx.x] = vec[idx];
		else shared_memory[threadIdx.x] = VType();
	}
	else shared_memory[threadIdx.x] = VType();

	//all values in this block must be transferred to shared memory before proceeding
	__syncthreads();

	//shift values within the box using shared_memory (both destination and source must not be empty).
	//We cannot shift the last element in this block since it must go to the block after this - store it in aux_block_values for later.
	//Similarly we cannot write the first element in this block since it needs a value from the previous block.
	if (threadIdx.x == CUDATHREADS - 1) vec.aux_block_values_ref()[blockIdx.x] = shared_memory[CUDATHREADS - 1];

	//special treatment for cells at -x boundary, if halos are being used : get value from halo
	if (ijk.i == 0 && vec.halo_n_ref()) {

		if (vec.ngbrFlags2_ref() && (vec.ngbrFlags2_ref()[idx] & NF2_HALONX)) {

			vec[idx] = vec.halo_n_ref()[ijk.j + ijk.k * n.y];
			vec.mark_not_empty(idx);
		}
		else {

			vec[idx] = VType();
			vec.mark_empty(idx);
		}
	}
	else if (shift_box.contains(ijk)) {

		if (threadIdx.x > 0) {

			vec[idx] = shared_memory[threadIdx.x - 1];

			//important to shift shape as well (but check for empty cell by value, otherwise we have to shift ngbrFlags as well)
			if (vec[idx] == VType()) vec.mark_empty(idx);
			else vec.mark_not_empty(idx);
		}
	}
}

template <typename VType>
__global__ void shift_x_right1_stitch_kernel(cuRect shift_rect, cuVEC_VC<VType>& vec)
{
	//index in aux_block_values
	int aux_blocks_idx = blockIdx.x * blockDim.x + threadIdx.x;

	//index in vec : aux_block_values stored block ending values which must be shifted to the cell to the right
	int cell_idx = aux_blocks_idx * CUDATHREADS + CUDATHREADS;

	cuSZ3 n = vec.n;
	cuINT3 ijk = cuINT3(cell_idx % n.x, (cell_idx / n.x) % n.y, cell_idx / (n.x*n.y));

	cuBox shift_box = vec.box_from_rect_min(shift_rect);
	shift_box.s.x++;

	if (shift_box.contains(ijk)) {

		vec[cell_idx] = vec.aux_block_values_ref()[aux_blocks_idx];

		//important to shift shape as well (but check for empty cell by value, otherwise we have to shift ngbrFlags as well)
		if (vec[cell_idx] == VType()) vec.mark_empty(cell_idx);
		else vec.mark_not_empty(cell_idx);
	}
}

template bool cuVEC_VC<float>::shift_x(size_t size, cuBReal delta, cuRect shift_rect, bool recalculate_flags, bool force_single_cell_shift);
template bool cuVEC_VC<double>::shift_x(size_t size, cuBReal delta, cuRect shift_rect, bool recalculate_flags, bool force_single_cell_shift);

template bool cuVEC_VC<cuFLT3>::shift_x(size_t size, cuBReal delta, cuRect shift_rect, bool recalculate_flags, bool force_single_cell_shift);
template bool cuVEC_VC<cuDBL3>::shift_x(size_t size, cuBReal delta, cuRect shift_rect, bool recalculate_flags, bool force_single_cell_shift);

//shift all the values in this VEC by the given delta (units same as h). Shift values in given shift_rect (absolute coordinates).
//the shift may require performing multiple single cell shifts, which are done unless force_single_cell_shift is true
//in this case the function return true if further single cell shifts are required but were not executed since force_single_cell_shift = true
//this is used by an external coordinating algorithm (mcuVEC). Only a single cell shift is performed, and the rest is banked in shift_debt.
//thus the coordinating algorithm needs to call shift_x again, but with delta = 0, since the shift_debt will be used instead.
//repeat until shift_debt is exhausted and function returns false.
template <typename VType>
__host__ bool cuVEC_VC<VType>::shift_x(size_t size, cuBReal delta, cuRect shift_rect, bool recalculate_flags, bool force_single_cell_shift)
{
	bool require_further_shifts = false;

	cuReal3 shift_debt_cpu = get_gpu_value(shift_debt);
	cuReal3 h_cpu = get_gpu_value(cuVEC<VType>::h);

	if ((int)round(fabs(shift_debt_cpu.x + delta) / h_cpu.x) == 0) {

		//total shift not enough : bank it and return
		shift_debt_cpu.x += delta;
		set_gpu_value(shift_debt, shift_debt_cpu);
		return require_further_shifts;
	}

	//only shift an integer number of cells : there might be a sub-cellsize remainder so just bank it to be used next time
	int cells_shift = (int)round((shift_debt_cpu.x + delta) / h_cpu.x);
	if (force_single_cell_shift) {

		if (cells_shift < -1) { cells_shift = -1;  require_further_shifts = true; }
		if (cells_shift > +1) { cells_shift = +1;  require_further_shifts = true; }
	}

	shift_debt_cpu.x -= h_cpu.x * cells_shift - delta;
	set_gpu_value(shift_debt, shift_debt_cpu);
	
	if (cells_shift < 0) {

		//only shift one cell at a time - for a moving mesh algorithm it would be very unusual to have to shift by more than one cell at a time if configured properly (mesh trigger from finest mesh)
		//one-call shift routines for cells_shift > 1 are not straight-forward so not worth implementing for now
		while (cells_shift < 0) {

			shift_x_left1_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (shift_rect, *this);

			size_t stitch_size = (size + CUDATHREADS) / CUDATHREADS;
			shift_x_left1_stitch_kernel <<< (stitch_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (shift_rect, *this);
			
			cells_shift++;
		}
	}
	else {

		while (cells_shift > 0) {

			shift_x_right1_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (shift_rect, *this);

			size_t stitch_size = (size + CUDATHREADS) / CUDATHREADS;
			shift_x_right1_stitch_kernel <<< (stitch_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (shift_rect, *this);

			cells_shift--;
		}
	}

	//shape could have changed, so must recalculate shape flags
	if (recalculate_flags) set_ngbrFlags();

	return require_further_shifts;
}

//------------------------------------------------------------------- SHIFT : y