#include "DiffEqFMCUDA.h"

#if COMPILECUDA == 1
#ifdef ODE_EVAL_COMPILATION_RKF45
#ifdef MESH_COMPILATION_FERROMAGNETIC

#include "MeshParamsControlCUDA.h"

#include "Reduction.cuh"

//defines evaluation methods kernel launchers

//----------------------------------------- EVALUATIONS : RKF45

__global__ void RunRKF45_Step0_withReductions_Kernel(ManagedDiffEqFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuBReal mxh = 0.0;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			//Save current magnetization for later use
			(*cuDiffEq.psM1)[idx] = (*cuMesh.pM)[idx];

			if (!cuMesh.pM->is_skipcell(idx)) {

				//obtain maximum normalized torque term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				mxh = cu_GetMagnitude((*cuMesh.pM)[idx] ^ (*cuMesh.pHeff)[idx]) / (Mnorm * Mnorm);

				//First evaluate RHS of set equation at the current time step
				(*cuDiffEq.psEval0)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx);

				//Now estimate magnetization using RKF first step
				(*cuMesh.pM)[idx] += (*cuDiffEq.psEval0)[idx] * (2 * dT / 9);
			}
		}
	}

	//only reduce for mxh if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &mxh, *cuDiffEq.pmxh);
	}
}

__global__ void RunRKF45_Step0_Kernel(ManagedDiffEqFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			//Save current magnetization for later use
			(*cuDiffEq.psM1)[idx] = (*cuMesh.pM)[idx];

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				(*cuDiffEq.psEval0)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx);

				//Now estimate magnetization using RKF first step
				(*cuMesh.pM)[idx] += (*cuDiffEq.psEval0)[idx] * (2 * dT / 9);
			}
		}
	}
}

__global__ void RunRKF45_Step1_Kernel(ManagedDiffEqFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval1)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx);

			//Now estimate magnetization using RKF midle step 1
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + ((*cuDiffEq.psEval0)[idx] / 12 + (*cuDiffEq.psEval1)[idx] / 4) * dT;
		}
	}
}

__global__ void RunRKF45_Step2_Kernel(ManagedDiffEqFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval2)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx);

			//Now estimate magnetization using RKF midle step 2
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (69 * (*cuDiffEq.psEval0)[idx] / 128 - 243 * (*cuDiffEq.psEval1)[idx] / 128 + 135 * (*cuDiffEq.psEval2)[idx] / 64) * dT;
		}
	}
}

__global__ void RunRKF45_Step3_Kernel(ManagedDiffEqFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval3)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx);

			//Now estimate magnetization using RKF midle step 3
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (-17 * (*cuDiffEq.psEval0)[idx] / 12 + 27 * (*cuDiffEq.psEval1)[idx] / 4 - 27 * (*cuDiffEq.psEval2)[idx] / 5 + 16 * (*cuDiffEq.psEval3)[idx] / 15) * dT;
		}
	}
}

__global__ void RunRKF45_Step4_Kernel(ManagedDiffEqFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval4)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx);

			//Now estimate magnetization using RKF midle step 4
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (65 * (*cuDiffEq.psEval0)[idx] / 432 - 5 * (*cuDiffEq.psEval1)[idx] / 16 + 13 * (*cuDiffEq.psEval2)[idx] / 16 + 4 * (*cuDiffEq.psEval3)[idx] / 27 + 5 * (*cuDiffEq.psEval4)[idx] / 144) * dT;
		}
	}
}

__global__ void RunRKF45_Step5_withReductions_Kernel(ManagedDiffEqFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuBReal dmdt = 0.0;
	cuBReal lte = 0.0;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx);

				//4th order evaluation
				(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + ((*cuDiffEq.psEval0)[idx] / 9 + 9 * (*cuDiffEq.psEval2)[idx] / 20 + 16 * (*cuDiffEq.psEval3)[idx] / 45 + (*cuDiffEq.psEval4)[idx] / 12) * dT;

				//5th order evaluation
				cuReal3 prediction = (*cuDiffEq.psM1)[idx] + (47 * (*cuDiffEq.psEval0)[idx] / 450 + 12 * (*cuDiffEq.psEval2)[idx] / 25 + 32 * (*cuDiffEq.psEval3)[idx] / 225 + 1 * (*cuDiffEq.psEval4)[idx] / 30 + 6 * rhs / 25) * dT;

				if (*cuDiffEq.prenormalize) {

					cuBReal Ms = *cuMesh.pMs;
					cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms);
					(*cuMesh.pM)[idx].renormalize(Ms);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				dmdt = cu_GetMagnitude((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuMesh.pM)[idx] - prediction) / (*cuMesh.pM)[idx].norm();
			}
			else {

				cuBReal Ms = *cuMesh.pMs;
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms);
				(*cuMesh.pM)[idx].renormalize(Ms);		//re-normalize the skipped cells no matter what - temperature can change
			}
		}
	}

	//only reduce for dmdt if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &dmdt, *cuDiffEq.pdmdt);
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

__global__ void RunRKF45_Step5_Kernel(ManagedDiffEqFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuBReal lte = 0.0;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx);

				//4th order evaluation
				(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + ((*cuDiffEq.psEval0)[idx] / 9 + 9 * (*cuDiffEq.psEval2)[idx] / 20 + 16 * (*cuDiffEq.psEval3)[idx] / 45 + (*cuDiffEq.psEval4)[idx] / 12) * dT;

				//5th order evaluation
				cuReal3 prediction = (*cuDiffEq.psM1)[idx] + (47 * (*cuDiffEq.psEval0)[idx] / 450 + 12 * (*cuDiffEq.psEval2)[idx] / 25 + 32 * (*cuDiffEq.psEval3)[idx] / 225 + 1 * (*cuDiffEq.psEval4)[idx] / 30 + 6 * rhs / 25) * dT;

				if (*cuDiffEq.prenormalize) {

					cuBReal Ms = *cuMesh.pMs;
					cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms);
					(*cuMesh.pM)[idx].renormalize(Ms);
				}

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuMesh.pM)[idx] - prediction) / (*cuMesh.pM)[idx].norm();
			}
			else {

				cuBReal Ms = *cuMesh.pMs;
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms);
				(*cuMesh.pM)[idx].renormalize(Ms);		//re-normalize the skipped cells no matter what - temperature can change
			}
		}
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

//----------------------------------------- DifferentialEquationCUDA Launchers

//RUNGE KUTTA FEHLBERG

void DifferentialEquationFMCUDA::RunRKF45(int step, bool calculate_mxh, bool calculate_dmdt)
{
	switch (step) {

	case 0:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			if (calculate_mxh) {

				RunRKF45_Step0_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
			else {

				RunRKF45_Step0_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;

	case 1:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step1_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 2:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step2_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 3:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step3_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 4:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step4_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 5:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			if (calculate_dmdt) {

				RunRKF45_Step5_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
			else {

				RunRKF45_Step5_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;
	}
}

#endif
#endif
#endif