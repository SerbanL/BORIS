#include "MeshCUDA.h"

#if COMPILECUDA == 1

#include "BorisCUDALib.cuh"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void GetTopologicalCharge_Kernel(ManagedMeshCUDA& cuMesh, cuRect rectangle, cuBReal& Q)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal Q_ = 0.0;

	cuReal3 pos;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			pos = M.cellidx_to_position(idx);

			cuBReal M_mag = M[idx].norm();

			cuReal33 M_grad = M.grad_neu(idx);

			cuReal3 dm_dx = M_grad.x / M_mag;
			cuReal3 dm_dy = M_grad.y / M_mag;

			Q_ = (M[idx] / M_mag) * (dm_dx ^ dm_dy) * M.h.x * M.h.y / (4 * (cuBReal)PI * M.n.z);
		}
	}

	reduction_sum(0, 1, &Q_, Q, rectangle.contains(pos));
}

//get topological charge using formula Q = Integral(m.(dm/dx x dm/dy) dxdy) / 4PI
cuBReal MeshCUDA::GetTopologicalCharge(cuRect rectangle)
{
	if (rectangle.IsNull()) rectangle = meshRect;

	Zero_aux_values();

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		GetTopologicalCharge_Kernel <<< (M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (cuMesh.get_deviceobject(mGPU), rectangle, aux_real(mGPU));
	}

	return aux_real.to_cpu_sum();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Compute_TopoChargeDensity_Kernel(ManagedMeshCUDA& cuMesh, cuVEC<cuBReal>& auxVEC_cuBReal)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			cuBReal M_mag = M[idx].norm();

			cuReal33 M_grad = M.grad_neu(idx);

			cuReal3 dm_dx = M_grad.x / M_mag;
			cuReal3 dm_dy = M_grad.y / M_mag;

			auxVEC_cuBReal[idx] = (M[idx] / M_mag) * (dm_dx ^ dm_dy) * M.h.x * M.h.y / (4 * (cuBReal)PI * M.n.z);
		}
		else auxVEC_cuBReal[idx] = 0.0;
	}
}

//compute topological charge density spatial dependence and have it available in auxVEC_cuBReal
//Use formula Qdensity = m.(dm/dx x dm/dy) / 4PI
void MeshCUDA::Compute_TopoChargeDensity(void)
{
	auxVEC_cuBReal.resize(h, meshRect);

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Compute_TopoChargeDensity_Kernel <<< (M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (cuMesh.get_deviceobject(mGPU), auxVEC_cuBReal.get_deviceobject(mGPU));
	}
}

#endif