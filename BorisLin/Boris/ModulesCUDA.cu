#include "ModulesCUDA.h"

#if COMPILECUDA == 1

#include "Reduction.cuh"

__global__ void ZeroEnergy_kernel(cuBReal& energy, cuReal3& torque, size_t& points_count)
{
	if (threadIdx.x == 0) energy = 0.0;
	if (threadIdx.x == 1) torque = 0.0;
	if (threadIdx.x == 2) points_count = 0;
}

void ModulesCUDA::ZeroEnergy(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		ZeroEnergy_kernel <<< 1, CUDATHREADS >>> (energy(mGPU), torque(mGPU), points_count(mGPU));
	}
}

__global__ void ZeroModuleVECs_kernel(cuVEC<cuReal3>& Module_Heff, cuVEC<cuReal3>& Module_Heff2, cuVEC<cuBReal>& Module_energy, cuVEC<cuBReal>& Module_energy2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Module_Heff.linear_size()) Module_Heff[idx] = cuReal3();
	if (idx < Module_Heff2.linear_size()) Module_Heff2[idx] = cuReal3();
	if (idx < Module_energy.linear_size()) Module_energy[idx] = 0.0;
	if (idx < Module_energy2.linear_size()) Module_energy2[idx] = 0.0;
}

void ModulesCUDA::ZeroModuleVECs(void)
{
	//This method is used at runtime, so better use a single kernel launch rather than zeroing them separately

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		ZeroModuleVECs_kernel <<< (Module_Heff.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(Module_Heff.get_deviceobject(mGPU), Module_Heff2.get_deviceobject(mGPU), Module_energy.get_deviceobject(mGPU), Module_energy2.get_deviceobject(mGPU));
	}
}

//-------------------------- Torque

__global__ void CalculateTorque_kernel(cuVEC_VC<cuReal3>& M, cuVEC<cuReal3>& Module_Heff, cuRect avRect, cuReal3& torque, size_t& points_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3& n = M.n;

	cuReal3 torque_ = cuReal3();
	bool include_in_average = false;

	if (idx < M.linear_size()) {

		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x*n.y));

		if (M.box_from_rect_max(avRect + M.rect.s).contains(ijk) && M.is_not_empty(ijk)) {

			torque_ = M[ijk] ^ Module_Heff[ijk];
			include_in_average = true;
		}
	}

	//need the idx < n.dim() check before cuvec.is_not_empty(ijk) to avoid bad memory access
	reduction_avg(0, 1, &torque_, torque, points_count, include_in_average);
}

//return cross product of M with Module_Heff, averaged in given rect (relative)
cuReal3 ModulesCUDA::CalculateTorque(mcu_VEC_VC(cuReal3)& M, cuRect& avRect)
{
	if (!Module_Heff.linear_size_cpu()) return cuReal3();

	ZeroEnergy();

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		CalculateTorque_kernel <<< (M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(M.get_deviceobject(mGPU), Module_Heff.get_deviceobject(mGPU), avRect, torque(mGPU), points_count(mGPU));
	}

	size_t points_count_cpu = points_count.to_cpu_sum();

	if (points_count_cpu) return torque.to_cpu_sum() / points_count_cpu;
	else return cuReal3();
}

#endif