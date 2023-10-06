#pragma once

#include "cuVEC_VC.h"
#include "launchers.h"

////////////////////////////////// FULL REDUCTION KERNEL : 1) internal storage, 2) external storage

template <typename VType>
__global__ void extract_profilevalues_reduction_kernel(
	cuVEC_VC<VType>& cuvec_vc,
	cuReal3& start, cuReal3& end, cuBReal& step, cuReal3& stencil, int& num_points, size_t*& line_profile_avpoints,
	VType* line_profile, 
	int profile_idx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	VType value = 0.0;
	bool include_in_reduction = false;

	if (idx < num_points) {

		//profile point position
		cuReal3 position = start + (cuBReal)profile_idx * step * (end - start).normalized();

		//dimensions in number of cells, of averaging box
		cuINT3 n = stencil / cuvec_vc.h;
		//i, j, k indexes in averaging box
		int i = idx % n.x;
		int j = (idx / n.x) % n.y;
		int k = idx / (n.x * n.y);

		//position in mesh for this kernel thread - this is where we have to get value from for reduction
		cuReal3 ker_position = (position - stencil / 2 + (cuReal3(i + 0.5, j + 0.5, k + 0.5) & cuvec_vc.h)) % cuvec_vc.rect.size();

		if (cuvec_vc.is_not_empty(ker_position)) {

			value = cuvec_vc[ker_position];
			include_in_reduction = true;
		}
	}

	reduction_avg(0, 1, &value, line_profile[profile_idx], line_profile_avpoints[profile_idx], include_in_reduction);
}

template <typename VType>
__global__ void extract_profilevalues_reduction_kernel(
	cuVEC_VC<VType>& cuvec_vc, 
	cuReal3& start, cuReal3& end, cuBReal& step, cuReal3& stencil, int& num_points, size_t*& line_profile_avpoints, 
	int profile_idx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	VType value = 0.0;
	bool include_in_reduction = false;

	if (idx < num_points) {

		//profile point position
		cuReal3 position = start + (cuBReal)profile_idx * step * (end - start).normalized();

		//dimensions in number of cells, of averaging box
		cuINT3 n = stencil / cuvec_vc.h;
		//i, j, k indexes in averaging box
		int i = idx % n.x;
		int j = (idx / n.x) % n.y;
		int k = idx / (n.x * n.y);

		//position in mesh for this kernel thread - this is where we have to get value from for reduction
		cuReal3 ker_position = (position - stencil / 2 + (cuReal3(i + 0.5, j + 0.5, k + 0.5) & cuvec_vc.h)) % cuvec_vc.rect.size();

		if (cuvec_vc.is_not_empty(ker_position)) {

			value = cuvec_vc[ker_position];
			include_in_reduction = true;
		}
	}

	reduction_avg(0, 1, &value, cuvec_vc.get_line_profile()[profile_idx], line_profile_avpoints[profile_idx], include_in_reduction);
}

////////////////////////////////// FULL PROFILE KERNEL : 1) internal storage, 2) external storage

template <typename VType>
__global__ void extract_profilevalues_cuarr_kernel(
	cuVEC_VC<VType>& cuvec_vc,
	cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil, 
	VType* profile, size_t profile_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < profile_size) {

		cuReal3 position = (start + (cuBReal)idx * step * (end - start).normalized()) % cuvec_vc.rect.size();

		profile[idx] = cuvec_vc.average_nonempty(cuRect(position - stencil / 2, position + stencil / 2));
	}
}

template <typename VType>
__global__ void extract_profilevalues_kernel(
	cuVEC_VC<VType>& cuvec_vc, 
	cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil, size_t& line_profile_component_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < line_profile_component_size) {

		cuReal3 position = (start + (cuBReal)idx * step * (end - start).normalized()) % cuvec_vc.rect.size();

		cuvec_vc.get_line_profile()[idx] = cuvec_vc.average_nonempty(cuRect(position - stencil / 2, position + stencil / 2));
	}
}

template bool cuVEC_VC<float>::extract_profile(cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil, cu_arr<float>& profile_gpu);
template bool cuVEC_VC<double>::extract_profile(cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil, cu_arr<double>& profile_gpu);

template bool cuVEC_VC<cuFLT3>::extract_profile(cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil, cu_arr<cuFLT3>& profile_gpu);
template bool cuVEC_VC<cuDBL3>::extract_profile(cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil, cu_arr<cuDBL3>& profile_gpu);

template <typename VType>
__host__ bool cuVEC_VC<VType>::extract_profile(cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil, cu_arr<VType>& profile_gpu)
{
	if (step > 0) {

		size_t size = round((end - start).norm() / step) + 1;

		//make sure memory is allocated correctly for auxiliary arrays
		if (!cuVEC<VType>::allocate_profile_component_memory(size)) return false;

		//make sure profile_gpu has correct size
		if (!profile_gpu.resize(size)) return false;

		cuReal3 h_cpu = get_gpu_value(cuVEC<VType>::h);
		cuINT3 nstencil = stencil / h_cpu;
		int num_stencil_points = nstencil.dim();

		//if stencil has more points than the profile, then better to launch multiple reduction kernels : one per each profile point
		if (num_stencil_points > size) {

			//set data in gpu memory once so we don't have to do it every iteration
			cu_obj<cuReal3> start_gpu;
			start_gpu.from_cpu(start);
			cu_obj<cuReal3> end_gpu;
			end_gpu.from_cpu(end);
			cu_obj<cuBReal> step_gpu;
			step_gpu.from_cpu(step);
			cu_obj<cuReal3> stencil_gpu;
			stencil_gpu.from_cpu(stencil);
			cu_obj<int> num_stencil_points_gpu;
			num_stencil_points_gpu.from_cpu(num_stencil_points);

			//zero values for reduction
			zero_profilevalues_cuarr_kernel<VType> <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(profile_gpu, cuVEC<VType>::line_profile_avpoints, cuVEC<VType>::line_profile_component_size);

			//extract
			for (int idx = 0; idx < size; idx++) {

				extract_profilevalues_reduction_kernel<VType> <<< (num_stencil_points + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(*this, start_gpu, end_gpu, step_gpu, stencil_gpu, num_stencil_points_gpu, cuVEC<VType>::line_profile_avpoints, profile_gpu, idx);
			}

			//divide by number of averaging points to finish off
			finish_profileaveragevalues_cuarr_kernel<VType> <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(profile_gpu, cuVEC<VType>::line_profile_avpoints, cuVEC<VType>::line_profile_component_size);
		}
		else {

			//extract profile
			extract_profilevalues_cuarr_kernel<VType> <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(*this, start, end, step, stencil, profile_gpu, profile_gpu.size());
		}

		//all done
		return true;
	}
	else return false;
}

template bool cuVEC_VC<float>::extract_profile(cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil);
template bool cuVEC_VC<double>::extract_profile(cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil);

template bool cuVEC_VC<cuFLT3>::extract_profile(cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil);
template bool cuVEC_VC<cuDBL3>::extract_profile(cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil);

//as above, but only store profile in internal memory (line_profile) so we can read it out later as needed
template <typename VType>
__host__ bool cuVEC_VC<VType>::extract_profile(cuReal3 start, cuReal3 end, cuBReal step, cuReal3 stencil)
{
	if (step > 0) {

		size_t size = round((end - start).norm() / step) + 1;

		//make sure memory is allocated correctly for auxiliary arrays
		if (!cuVEC<VType>::allocate_profile_component_memory(size)) return false;

		cuReal3 h_cpu = get_gpu_value(cuVEC<VType>::h);
		cuINT3 nstencil = stencil / h_cpu;
		int num_stencil_points = nstencil.dim();

		//if stencil has more points than the profile, then better to launch multiple reduction kernels : one per each profile point
		if (num_stencil_points > size) {

			//set data in gpu memory once so we don't have to do it every iteration
			cu_obj<cuReal3> start_gpu;
			start_gpu.from_cpu(start);
			cu_obj<cuReal3> end_gpu;
			end_gpu.from_cpu(end);
			cu_obj<cuBReal> step_gpu;
			step_gpu.from_cpu(step);
			cu_obj<cuReal3> stencil_gpu;
			stencil_gpu.from_cpu(stencil);
			cu_obj<int> num_stencil_points_gpu;
			num_stencil_points_gpu.from_cpu(num_stencil_points);

			//zero values for reduction
			zero_profilevalues_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(cuVEC<VType>::line_profile, cuVEC<VType>::line_profile_avpoints, cuVEC<VType>::line_profile_component_size);

			//extract
			for (int idx = 0; idx < size; idx++) {

				extract_profilevalues_reduction_kernel <<< (num_stencil_points + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(*this, start_gpu, end_gpu, step_gpu, stencil_gpu, num_stencil_points_gpu, cuVEC<VType>::line_profile_avpoints, idx);
			}

			//divide by number of averaging points to finish off
			finish_profileaveragevalues_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(cuVEC<VType>::line_profile, cuVEC<VType>::line_profile_avpoints, cuVEC<VType>::line_profile_component_size);
		}
		else {

			//extract profile
			extract_profilevalues_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(*this, start, end, step, stencil, cuVEC<VType>::line_profile_component_size);
		}

		//all done
		return true;
	}
	else return false;
}