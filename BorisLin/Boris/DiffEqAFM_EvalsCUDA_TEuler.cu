#include "DiffEqAFMCUDA.h"

#if COMPILECUDA == 1
#ifdef ODE_EVAL_COMPILATION_TEULER
#ifdef MESH_COMPILATION_ANTIFERROMAGNETIC

#include "MeshParamsControlCUDA.h"

#include "Reduction.cuh"

//defines evaluation methods kernel launchers

//----------------------------------------- EVALUATIONS : Trapezoidal Euler

__global__ void RunTEuler_Step0_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuReal3 mxh = cuReal3();
	bool include_in_average = false;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			//obtain average normalized torque term
			cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
			mxh = ((*cuMesh.pM)[idx] ^ (*cuMesh.pHeff)[idx]) / (Mnorm * Mnorm);
			include_in_average = true;

			//Save current magnetization for the next step
			(*cuDiffEq.psM1)[idx] = (*cuMesh.pM)[idx];
			(*cuDiffEq.psM1_2)[idx] = (*cuMesh.pM2)[idx];

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs2;
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, rhs2);

				//Now estimate magnetization for the next time step
				(*cuMesh.pM)[idx] += rhs * dT;
				(*cuMesh.pM2)[idx] += rhs2 * dT;
			}
		}
	}

	//only reduce for mxh if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_avg(0, 1, &mxh, *cuDiffEq.pmxh_av, *cuDiffEq.pavpoints, include_in_average);
	}
}

__global__ void RunTEuler_Step0_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			//Save current magnetization for the next step
			(*cuDiffEq.psM1)[idx] = (*cuMesh.pM)[idx];
			(*cuDiffEq.psM1_2)[idx] = (*cuMesh.pM2)[idx];

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs2;
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, rhs2);

				//Now estimate magnetization for the next time step
				(*cuMesh.pM)[idx] += rhs * dT;
				(*cuMesh.pM2)[idx] += rhs2 * dT;
			}
		}
	}
}

__global__ void RunTEuler_Step1_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuReal3 dmdt = cuReal3();
	bool include_in_average = false;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs2;
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, rhs2);

				//Now estimate magnetization using the second trapezoidal Euler step equation
				(*cuMesh.pM)[idx] = ((*cuDiffEq.psM1)[idx] + (*cuMesh.pM)[idx] + rhs * dT) / 2;
				(*cuMesh.pM2)[idx] = ((*cuDiffEq.psM1_2)[idx] + (*cuMesh.pM2)[idx] + rhs2 * dT) / 2;

				if (*cuDiffEq.prenormalize) {

					cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
					cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
					(*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
					(*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				dmdt = ((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);
				include_in_average = true;
			}
			else {

				cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				(*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
				(*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	//only reduce for dmdt (and mxh) if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_avg(0, 1, &dmdt, *cuDiffEq.pdmdt_av, *cuDiffEq.pavpoints2, include_in_average);
	}
}

__global__ void RunTEuler_Step1_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs2;
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, rhs2);

				//Now estimate magnetization using the second trapezoidal Euler step equation
				(*cuMesh.pM)[idx] = ((*cuDiffEq.psM1)[idx] + (*cuMesh.pM)[idx] + rhs * dT) / 2;
				(*cuMesh.pM2)[idx] = ((*cuDiffEq.psM1_2)[idx] + (*cuMesh.pM2)[idx] + rhs2 * dT) / 2;

				if (*cuDiffEq.prenormalize) {

					cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
					cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
					(*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
					(*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
				}
			}
			else {

				cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				(*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
				(*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
			}
		}
	}
}

//----------------------------------------- DifferentialEquationCUDA Launchers

//TRAPEZOIDAL EULER

void DifferentialEquationAFMCUDA::RunTEuler(int step, bool calculate_mxh, bool calculate_dmdt)
{
	if (step == 0) {

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step0_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step0_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
	}
	else {

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step1_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step1_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
	}
}

#endif
#endif
#endif