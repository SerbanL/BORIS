#include "MeshBaseCUDA.h"

#if COMPILECUDA == 1

__global__ void Zero_aux_values_kernels(cuBReal& aux_real, cuReal3& aux_real3, size_t& aux_int)
{
	if (threadIdx.x == 0) aux_real = 0.0;
	if (threadIdx.x == 1) aux_real3 = cuReal3();
	if (threadIdx.x == 2) aux_int = 0.0;
}

//zero all single aux avalues
void MeshBaseCUDA::Zero_aux_values(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Zero_aux_values_kernels <<< 1, CUDATHREADS >>> (aux_real(mGPU), aux_real3(mGPU), aux_int(mGPU));
	}
}

//----------------------------------- DISPLAY-ASSOCIATED GET/SET METHODS AUXILIARY

__global__ void average_mesh_profile_kernel(size_t size, cuBReal* profile_storage_sca, cuVEC<cuBReal>& cuvec_sca, int num_profile_averages)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size && idx < cuvec_sca.get_line_profile_size()) {
		
		if (num_profile_averages) profile_storage_sca[idx] = (profile_storage_sca[idx] * num_profile_averages + cuvec_sca.get_line_profile()[idx]) / (num_profile_averages + 1);
		//if num_profile_averages == 0, it's possible profile_storage_sca is not initialized and currently storing a nan
		else profile_storage_sca[idx] = cuvec_sca.get_line_profile()[idx];
	}
}

__global__ void average_mesh_profile_kernel(size_t size, cuBReal* profile_storage_sca, cuVEC_VC<cuBReal>& cuvecvc_sca, int num_profile_averages)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size && idx < cuvecvc_sca.get_line_profile_size()) {

		if (num_profile_averages) profile_storage_sca[idx] = (profile_storage_sca[idx] * num_profile_averages + cuvecvc_sca.get_line_profile()[idx]) / (num_profile_averages + 1);
		else profile_storage_sca[idx] = cuvecvc_sca.get_line_profile()[idx];
	}
}

__global__ void average_mesh_profile_kernel(size_t size, cuReal3* profile_storage_vec, cuVEC<cuReal3>& cuvec_vec, int num_profile_averages)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size && idx < cuvec_vec.get_line_profile_size()) {

		if (num_profile_averages) profile_storage_vec[idx] = (profile_storage_vec[idx] * num_profile_averages + cuvec_vec.get_line_profile()[idx]) / (num_profile_averages + 1);
		else profile_storage_vec[idx] = cuvec_vec.get_line_profile()[idx];
	}
}

__global__ void average_mesh_profile_kernel(size_t size, cuReal3* profile_storage_vec, cuVEC_VC<cuReal3>& cuvecvc_vec, int num_profile_averages)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size && idx < cuvecvc_vec.get_line_profile_size()) {

		if (num_profile_averages) profile_storage_vec[idx] = (profile_storage_vec[idx] * num_profile_averages + cuvecvc_vec.get_line_profile()[idx]) / (num_profile_averages + 1);
		else profile_storage_vec[idx] = cuvecvc_vec.get_line_profile()[idx];
	}
}

//average into profile_storage_sca / profile_storage_vec
void MeshBaseCUDA::average_mesh_profile(mcu_VEC(cuBReal)& cuvec_sca, int& num_profile_averages)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		average_mesh_profile_kernel <<< (profile_storage_sca.size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(profile_storage_sca.size(mGPU), profile_storage_sca.get_cu_arr(mGPU), cuvec_sca.get_deviceobject(mGPU), num_profile_averages);
	}

	num_profile_averages++;
}

void MeshBaseCUDA::average_mesh_profile(mcu_VEC_VC(cuBReal)& cuvecvc_sca, int& num_profile_averages)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		average_mesh_profile_kernel <<< (profile_storage_sca.size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(profile_storage_sca.size(mGPU), profile_storage_sca.get_cu_arr(mGPU), cuvecvc_sca.get_deviceobject(mGPU), num_profile_averages);

	}

	num_profile_averages++;
}

void MeshBaseCUDA::average_mesh_profile(mcu_VEC(cuReal3)& cuvec_vec, int& num_profile_averages)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		average_mesh_profile_kernel <<< (profile_storage_vec.size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(profile_storage_vec.size(mGPU), profile_storage_vec.get_cu_arr(mGPU), cuvec_vec.get_deviceobject(mGPU), num_profile_averages);
	}

	num_profile_averages++;
}

void MeshBaseCUDA::average_mesh_profile(mcu_VEC_VC(cuReal3)& cuvecvc_vec, int& num_profile_averages)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		average_mesh_profile_kernel <<< (profile_storage_vec.size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(profile_storage_vec.size(mGPU), profile_storage_vec.get_cu_arr(mGPU), cuvecvc_vec.get_deviceobject(mGPU), num_profile_averages);
	}

	num_profile_averages++;
}

#endif

