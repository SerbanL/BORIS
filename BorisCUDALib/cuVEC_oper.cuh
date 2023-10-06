#pragma once

#include "cuVEC.h"
#include "cuFuncs_Math.h"
#include "launchers.h"
#include "Reduction.cuh"

//------------------------------------------------------------------- SETBOX

template <typename VType>
__global__ void setbox_kernel(cuSZ3& n, cuBox box, VType value, VType*& quantity)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x*n.y));

	if (idx < n.dim()) {

		if (box.contains(ijk)) quantity[idx] = value;
	}
}

template void cuVEC<float>::setbox(cuBox box, float value);
template void cuVEC<double>::setbox(cuBox box, double value);

template void cuVEC<cuFLT3>::setbox(cuBox box, cuFLT3 value);
template void cuVEC<cuDBL3>::setbox(cuBox box, cuDBL3 value);

template void cuVEC<cuFLT4>::setbox(cuBox box, cuFLT4 value);
template void cuVEC<cuDBL4>::setbox(cuBox box, cuDBL4 value);

template void cuVEC<cuReIm3>::setbox(cuBox box, cuReIm3 value);

template <typename VType>
__host__ void cuVEC<VType>::setbox(cuBox box, VType value)
{
	setbox_kernel <<< (get_gpu_value(n).dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (n, box, value, quantity);
}

//------------------------------------------------------------------- SET

template <typename VType>
__global__ void set_kernel(size_t size, VType value, VType*& quantity)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {

		quantity[idx] = value;
	}
}

template void cuVEC<float>::set(size_t size, float value);
template void cuVEC<double>::set(size_t size, double value);

template void cuVEC<cuFLT3>::set(size_t size, cuFLT3 value);
template void cuVEC<cuDBL3>::set(size_t size, cuDBL3 value);

template void cuVEC<cuFLT4>::set(size_t size, cuFLT4 value);
template void cuVEC<cuDBL4>::set(size_t size, cuDBL4 value);

template void cuVEC<cuReIm3>::set(size_t size, cuReIm3 value);

template <typename VType>
__host__ void cuVEC<VType>::set(size_t size, VType value)
{
	set_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (size, value, quantity);
}

//------------------------------------------------------------------- RENORMALIZE

template <typename VType, typename PType>
__global__ void renormalize_kernel(cuSZ3& n, VType*& quantity, PType new_norm)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n.dim()) {

		PType curr_norm = cu_GetMagnitude(quantity[idx]);

		if (cuIsNZ(curr_norm)) quantity[idx] *= new_norm / curr_norm;
	}
}

template void cuVEC<float>::renormalize(size_t arr_size, float new_norm);
template void cuVEC<double>::renormalize(size_t arr_size, double new_norm);

template void cuVEC<cuFLT3>::renormalize(size_t arr_size, float new_norm);
template void cuVEC<cuDBL3>::renormalize(size_t arr_size, double new_norm);

template void cuVEC<cuFLT4>::renormalize(size_t arr_size, float new_norm);
template void cuVEC<cuDBL4>::renormalize(size_t arr_size, double new_norm);

template <typename VType>
template <typename PType>
__host__ void cuVEC<VType>::renormalize(size_t arr_size, PType new_norm)
{
	renormalize_kernel <<< (arr_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (n, quantity, new_norm);
}

//------------------------------------------------------------------- COPY VALUES (DIRECT VERSION)

template <typename VType>
__global__ void copy_values_cuvec_kernel(cuVEC<VType>& to_this, cuVEC<VType>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier)
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
		}
	}
}

template void cuVEC<float>::copy_values(cuVEC<float>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier);
template void cuVEC<double>::copy_values(cuVEC<double>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier);

template void cuVEC<cuFLT3>::copy_values(cuVEC<cuFLT3>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier);
template void cuVEC<cuDBL3>::copy_values(cuVEC<cuDBL3>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier);

//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions
template <typename VType>
__host__ void cuVEC<VType>::copy_values(cuVEC<VType>& copy_this, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier)
{
	copy_values_cuvec_kernel <<< (cells_box_dst.size().dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (*this, copy_this, cells_box_dst, cells_box_src, multiplier);
}

//------------------------------------------------------------------- COPY VALUES (MCUVEC VERSION)

template <typename VType>
__global__ void copy_values_cuvec_mcuVEC_kernel(cuVEC<VType>& to_this, cuVEC<VType>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier)
{
	int idx_box_dst = blockIdx.x * blockDim.x + threadIdx.x;

	mcuVEC_Managed<cuVEC<VType>, VType>& copy_this_mcuvec = copy_this.mcuvec();

	cuSZ3 dst_n = cells_box_dst_device.size();
	//source rectangle of entire mcuVEC
	cuRect srcRect = cuRect(cells_box_src.s & copy_this_mcuvec.h, cells_box_src.e & copy_this_mcuvec.h);
	//destination rectangle of entire mcuVEC
	cuRect dstRect = cuRect(cells_box_dst.s & to_this.h, cells_box_dst.e & to_this.h);
	cuReal3 lRatio = dstRect.size() / srcRect.size();

	if (idx_box_dst < dst_n.dim()) {

		//i, j, k on this device
		int i = idx_box_dst % dst_n.i;
		int j = (idx_box_dst / dst_n.i) % dst_n.j;
		int k = idx_box_dst / (dst_n.i * dst_n.j);

		//linear output index on this device
		int idx_out = (i + cells_box_dst_device.s.i) + (j + cells_box_dst_device.s.j) * to_this.n.x + (k + cells_box_dst_device.s.k) * to_this.n.x * to_this.n.y;

		//destination cell rectangle relative to destination rectangle of entire mcuVEC
		cuRect dst_cell_rect_rel = to_this.get_cellrect(idx_out) - to_this.mcuvec().rect.s - dstRect.s;

		//now map this to source rectangle of entire mcuVEC
		cuRect src_cell_rect_rel = cuRect(dst_cell_rect_rel.s & lRatio, dst_cell_rect_rel.e & lRatio) + srcRect.s;

		if (idx_out < to_this.n.dim()) {

			to_this[idx_out] = copy_this_mcuvec.average(src_cell_rect_rel) * multiplier;
		}
	}
}

template void cuVEC<float>::copy_values_mcuVEC(cuVEC<float>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier);
template void cuVEC<double>::copy_values_mcuVEC(cuVEC<double>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier);

template void cuVEC<cuFLT3>::copy_values_mcuVEC(cuVEC<cuFLT3>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier);
template void cuVEC<cuDBL3>::copy_values_mcuVEC(cuVEC<cuDBL3>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier);

//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions
template <typename VType>
__host__ void cuVEC<VType>::copy_values_mcuVEC(cuVEC<VType>& copy_this, cuBox cells_box_dst_device, cuBox cells_box_dst, cuBox cells_box_src, cuBReal multiplier)
{
	copy_values_cuvec_mcuVEC_kernel <<< (cells_box_dst_device.size().dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (*this, copy_this, cells_box_dst_device, cells_box_dst, cells_box_src, multiplier);
}