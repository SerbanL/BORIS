#include "Atom_DiffEqCubicCUDA.h"

#if COMPILECUDA == 1
#ifdef MESH_COMPILATION_ATOM_CUBIC

#include "cu_prng.cuh"

#include "Atom_MeshParamsControlCUDA.h"

//---------------------------------------- OTHER CALCULATION METHODS : GENERATE THERMAL cuVECs

//----------------------------------------

__global__ void GenerateThermalField_Kernel(cuBorisRand<>& prng, ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh, cuBReal& deltaT)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal grel = cuaMesh.pgrel->get0();

	if (idx < cuaDiffEq.pH_Thermal->linear_size() && cuIsNZ(grel)) {

		if (cuaMesh.pM1->is_not_empty(idx) && !cuaMesh.pM1->is_skipcell(idx)) {

			cuReal3 h = cuaDiffEq.pH_Thermal->h;

			cuBReal Temperature;

			if (cuaMesh.pTemp->linear_size()) {

				//get temperature at centre of idx M cell
				Temperature = (*cuaMesh.pTemp)[cuaDiffEq.pH_Thermal->cellidx_to_position(idx)];
			}
			else Temperature = (*cuaMesh.pbase_temperature);

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal s_eff = *cuaMesh.ps_eff;
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.ps_eff, s_eff);

			//do not include any damping here - this will be included in the stochastic equations
			cuBReal Hth_const = s_eff * sqrt(2 * (cuBReal)BOLTZMANN * Temperature / ((cuBReal)MUB_MU0 * GAMMA * grel * mu_s * deltaT));
			
			(*cuaDiffEq.pH_Thermal)[idx] = Hth_const * cuReal3(prng.rand_gauss(0, 1), prng.rand_gauss(0, 1), prng.rand_gauss(0, 1));
		}
	}
}

//called when using stochastic equations
void Atom_DifferentialEquationCubicCUDA::GenerateThermalField_CUDA(mcu_val<cuBReal>& deltaT)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		GenerateThermalField_Kernel <<< (H_Thermal.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(prng.get_deviceobject(mGPU), cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU), deltaT(mGPU));
	}
}

#endif
#endif