#include "DiffEqFMCUDA.h"

#if COMPILECUDA == 1
#ifdef ODE_EVAL_COMPILATION_EULER
#ifdef MESH_COMPILATION_FERROMAGNETIC

#include "MeshParamsControlCUDA.h"

#include "Reduction.cuh"

//defines evaluation methods kernel launchers

//----------------------------------------- EVALUATIONS: Euler

__global__ void RunEuler_Kernel_withReductions(ManagedDiffEqFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuReal3 mxh = cuReal3();
	cuReal3 dmdt = cuReal3();
	bool include_in_average = false;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			//Save current magnetization
			(*cuDiffEq.psM1)[idx] = (*cuMesh.pM)[idx];

			if (!cuMesh.pM->is_skipcell(idx)) {

				//obtain average normalized torque term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				mxh = ((*cuMesh.pM)[idx] ^ (*cuMesh.pHeff)[idx]) / (Mnorm * Mnorm);
				include_in_average = true;

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx);

				//Now estimate magnetization for the next time step
				(*cuMesh.pM)[idx] += rhs * dT;

				if (*cuDiffEq.prenormalize) {

					cuBReal Ms = *cuMesh.pMs;
					cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms);
					(*cuMesh.pM)[idx].renormalize(Ms);
				}

				//obtain maximum normalized dmdt term
				dmdt = ((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);
			}
			else {

				cuBReal Ms = *cuMesh.pMs;
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms);
				(*cuMesh.pM)[idx].renormalize(Ms);		//re-normalize the skipped cells no matter what - temperature can change
			}
		}
	}

	//only reduce for dmdt (and mxh) if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_avg(0, 1, &mxh, *cuDiffEq.pmxh_av, *cuDiffEq.pavpoints, include_in_average);
		reduction_avg(0, 1, &dmdt, *cuDiffEq.pdmdt_av, *cuDiffEq.pavpoints2, include_in_average);
	}
}

__global__ void RunEuler_Kernel(ManagedDiffEqFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			//Save current magnetization
			(*cuDiffEq.psM1)[idx] = (*cuMesh.pM)[idx];

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx);

				//Now estimate magnetization for the next time step
				(*cuMesh.pM)[idx] += rhs * dT;

				if (*cuDiffEq.prenormalize) {

					cuBReal Ms = *cuMesh.pMs;
					cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms);
					(*cuMesh.pM)[idx].renormalize(Ms);
				}
			}
			else {

				cuBReal Ms = *cuMesh.pMs;
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms);
				(*cuMesh.pM)[idx].renormalize(Ms);		//re-normalize the skipped cells no matter what - temperature can change
			}
		}
	}
}

//----------------------------------------- DifferentialEquationCUDA Launchers

//EULER

void DifferentialEquationFMCUDA::RunEuler(bool calculate_mxh, bool calculate_dmdt)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		if (calculate_mxh || calculate_dmdt) {

			RunEuler_Kernel_withReductions <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}
		else {

			RunEuler_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}
	}
}

#endif
#endif
#endif