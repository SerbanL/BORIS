#include "Atom_Mesh_CubicCUDA.h"

#if COMPILECUDA == 1

#ifdef MESH_COMPILATION_ATOM_CUBIC

#include "Reduction.cuh"

#include "Atom_MeshParamsControlCUDA.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void GetTopologicalCharge_Cubic_Kernel(ManagedAtom_MeshCUDA& cuaMesh, cuRect rectangle, cuBReal& Q)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal Q_ = 0.0;
	bool include_in_reduction = false;

	cuReal3 pos;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			pos = M1.cellidx_to_position(idx);

			cuBReal Mnorm = M1[idx].norm();

			cuReal33 M_grad = M1.grad_neu(idx);

			cuReal3 dm_dx = M_grad.x / Mnorm;
			cuReal3 dm_dy = M_grad.y / Mnorm;

			Q_ = (M1[idx] / Mnorm) * (dm_dx ^ dm_dy) * M1.h.x * M1.h.y / (4 * (cuBReal)PI * M1.n.z);

			include_in_reduction = true;
		}
	}

	reduction_sum(0, 1, &Q_, Q, include_in_reduction && rectangle.contains(pos));
}

//get topological charge using formula Q = Integral(m.(dm/dx x dm/dy) dxdy) / 4PI
cuBReal Atom_Mesh_CubicCUDA::GetTopologicalCharge(cuRect rectangle)
{
	if (rectangle.IsNull()) rectangle = meshRect;

	Zero_aux_values();

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		GetTopologicalCharge_Cubic_Kernel <<< (M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (cuaMesh.get_deviceobject(mGPU), rectangle, aux_real(mGPU));
	}

	return aux_real.to_cpu_sum();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Compute_TopoChargeDensity_Cubic_Kernel(ManagedAtom_MeshCUDA& cuaMesh, cuVEC<cuBReal>& auxVEC_cuBReal)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			cuBReal Mnorm = M1[idx].norm();

			cuReal33 M_grad = M1.grad_neu(idx);

			cuReal3 dm_dx = M_grad.x / Mnorm;
			cuReal3 dm_dy = M_grad.y / Mnorm;

			auxVEC_cuBReal[idx] = (M1[idx] / Mnorm) * (dm_dx ^ dm_dy) * M1.h.x * M1.h.y / (4 * (cuBReal)PI * M1.n.z);
		}
		else auxVEC_cuBReal[idx] = 0.0;
	}
}

//compute topological charge density spatial dependence and have it available in auxVEC_cuBReal
//Use formula Qdensity = m.(dm/dx x dm/dy) / 4PI
void Atom_Mesh_CubicCUDA::Compute_TopoChargeDensity(void)
{
	auxVEC_cuBReal.resize(h, meshRect);

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Compute_TopoChargeDensity_Cubic_Kernel <<< (M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (cuaMesh.get_deviceobject(mGPU), auxVEC_cuBReal.get_deviceobject(mGPU));
	}
}

#endif

#endif