#include "DiffEqAFMCUDA.h"

#if COMPILECUDA == 1
#ifdef ODE_EVAL_COMPILATION_RKF56
#ifdef MESH_COMPILATION_ANTIFERROMAGNETIC

#include "MeshParamsControlCUDA.h"

#include "Reduction.cuh"

//defines evaluation methods kernel launchers

//----------------------------------------- EVALUATIONS : RKF56

__global__ void RunRKF56_Step0_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuBReal mxh = 0.0;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			//Save current magnetization for later use
			(*cuDiffEq.psM1)[idx] = (*cuMesh.pM)[idx];
			(*cuDiffEq.psM1_2)[idx] = (*cuMesh.pM2)[idx];

			if (!cuMesh.pM->is_skipcell(idx)) {

				//obtain maximum normalized torque term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				mxh = cu_GetMagnitude((*cuMesh.pM)[idx] ^ (*cuMesh.pHeff)[idx]) / (Mnorm * Mnorm);

				//First evaluate RHS of set equation at the current time step
				(*cuDiffEq.psEval0)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval0_2)[idx]);

				//Now estimate magnetization using RKF first step
				(*cuMesh.pM)[idx] += (*cuDiffEq.psEval0)[idx] * (dT / 6);
				(*cuMesh.pM2)[idx] += (*cuDiffEq.psEval0_2)[idx] * (dT / 6);
			}
		}
	}

	//only reduce for mxh if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &mxh, *cuDiffEq.pmxh);
	}
}

__global__ void RunRKF56_Step0_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			//Save current magnetization for later use
			(*cuDiffEq.psM1)[idx] = (*cuMesh.pM)[idx];
			(*cuDiffEq.psM1_2)[idx] = (*cuMesh.pM2)[idx];

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				(*cuDiffEq.psEval0)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval0_2)[idx]);

				//Now estimate magnetization using RKF first step
				(*cuMesh.pM)[idx] += (*cuDiffEq.psEval0)[idx] * (dT / 6);
				(*cuMesh.pM2)[idx] += (*cuDiffEq.psEval0_2)[idx] * (dT / 6);
			}
		}
	}
}

__global__ void RunRKF56_Step1_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval1)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval1_2)[idx]);

			//Now estimate magnetization using RKF midle step 1
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (4 * (*cuDiffEq.psEval0)[idx] + 16 * (*cuDiffEq.psEval1)[idx]) * dT / 75;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (4 * (*cuDiffEq.psEval0_2)[idx] + 16 * (*cuDiffEq.psEval1_2)[idx]) * dT / 75;
		}
	}
}

__global__ void RunRKF56_Step2_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval2)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval2_2)[idx]);

			//Now estimate magnetization using RKF midle step 2
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (5 * (*cuDiffEq.psEval0)[idx] / 6 - 8 * (*cuDiffEq.psEval1)[idx] / 3 + 5 * (*cuDiffEq.psEval2)[idx] / 2) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (5 * (*cuDiffEq.psEval0_2)[idx] / 6 - 8 * (*cuDiffEq.psEval1_2)[idx] / 3 + 5 * (*cuDiffEq.psEval2_2)[idx] / 2) * dT;
		}
	}
}

__global__ void RunRKF56_Step3_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval3)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval3_2)[idx]);

			//Now estimate magnetization using RKF midle step 3
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (-8 * (*cuDiffEq.psEval0)[idx] / 5 + 144 * (*cuDiffEq.psEval1)[idx] / 25 - 4 * (*cuDiffEq.psEval2)[idx] + 16 * (*cuDiffEq.psEval3)[idx] / 25) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (-8 * (*cuDiffEq.psEval0_2)[idx] / 5 + 144 * (*cuDiffEq.psEval1_2)[idx] / 25 - 4 * (*cuDiffEq.psEval2_2)[idx] + 16 * (*cuDiffEq.psEval3_2)[idx] / 25) * dT;
		}
	}
}

__global__ void RunRKF56_Step4_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval4)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval4_2)[idx]);

			//Now estimate magnetization using RKF midle step 4
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (361 * (*cuDiffEq.psEval0)[idx] / 320 - 18 * (*cuDiffEq.psEval1)[idx] / 5 + 407 * (*cuDiffEq.psEval2)[idx] / 128 - 11 * (*cuDiffEq.psEval3)[idx] / 80 + 55 * (*cuDiffEq.psEval4)[idx] / 128) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (361 * (*cuDiffEq.psEval0_2)[idx] / 320 - 18 * (*cuDiffEq.psEval1_2)[idx] / 5 + 407 * (*cuDiffEq.psEval2_2)[idx] / 128 - 11 * (*cuDiffEq.psEval3_2)[idx] / 80 + 55 * (*cuDiffEq.psEval4_2)[idx] / 128) * dT;
		}
	}
}

__global__ void RunRKF56_Step5_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval5)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval5_2)[idx]);

			//Now estimate magnetization using RKF midle step 4
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (-11 * (*cuDiffEq.psEval0)[idx] / 640 + 11 * (*cuDiffEq.psEval2)[idx] / 256 - 11 * (*cuDiffEq.psEval3)[idx] / 160 + 11 * (*cuDiffEq.psEval4)[idx] / 256) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (-11 * (*cuDiffEq.psEval0_2)[idx] / 640 + 11 * (*cuDiffEq.psEval2_2)[idx] / 256 - 11 * (*cuDiffEq.psEval3_2)[idx] / 160 + 11 * (*cuDiffEq.psEval4_2)[idx] / 256) * dT;
		}
	}
}

__global__ void RunRKF56_Step6_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval6)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval6_2)[idx]);

			//Now estimate magnetization using RKF midle step 4
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (93 * (*cuDiffEq.psEval0)[idx] / 640 - 18 * (*cuDiffEq.psEval1)[idx] / 5 + 803 * (*cuDiffEq.psEval2)[idx] / 256 - 11 * (*cuDiffEq.psEval3)[idx] / 160 + 99 * (*cuDiffEq.psEval4)[idx] / 256 + (*cuDiffEq.psEval6)[idx]) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (93 * (*cuDiffEq.psEval0_2)[idx] / 640 - 18 * (*cuDiffEq.psEval1_2)[idx] / 5 + 803 * (*cuDiffEq.psEval2_2)[idx] / 256 - 11 * (*cuDiffEq.psEval3_2)[idx] / 160 + 99 * (*cuDiffEq.psEval4_2)[idx] / 256 + (*cuDiffEq.psEval6_2)[idx]) * dT;
		}
	}
}

__global__ void RunRKF56_Step7_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuBReal dmdt = 0.0;
	cuBReal lte = 0.0;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs2;
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, rhs2);

				//5th order evaluation
				(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (31 * (*cuDiffEq.psEval0)[idx] / 384 + 1125 * (*cuDiffEq.psEval2)[idx] / 2816 + 9 * (*cuDiffEq.psEval3)[idx] / 32 + 125 * (*cuDiffEq.psEval4)[idx] / 768 + 5 * (*cuDiffEq.psEval5)[idx] / 66) * dT;
				(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (31 * (*cuDiffEq.psEval0_2)[idx] / 384 + 1125 * (*cuDiffEq.psEval2_2)[idx] / 2816 + 9 * (*cuDiffEq.psEval3_2)[idx] / 32 + 125 * (*cuDiffEq.psEval4_2)[idx] / 768 + 5 * (*cuDiffEq.psEval5_2)[idx] / 66) * dT;

				//local truncation error from 5th order evaluation and 6th order evaluation
				cuReal3 lte_diff = 5 * ((*cuDiffEq.psEval0)[idx] + (*cuDiffEq.psEval5)[idx] - (*cuDiffEq.psEval6)[idx] - rhs) * dT / 66;

				if (*cuDiffEq.prenormalize) {

					cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
					cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
					(*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
					(*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				dmdt = cu_GetMagnitude((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude(lte_diff) / (*cuMesh.pM)[idx].norm();
			}
			else {

				cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				(*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
				(*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	//only reduce for dmdt if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &dmdt, *cuDiffEq.pdmdt);
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

__global__ void RunRKF56_Step7_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuBReal lte = 0.0;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs2;
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, rhs2);

				//5th order evaluation
				(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (31 * (*cuDiffEq.psEval0)[idx] / 384 + 1125 * (*cuDiffEq.psEval2)[idx] / 2816 + 9 * (*cuDiffEq.psEval3)[idx] / 32 + 125 * (*cuDiffEq.psEval4)[idx] / 768 + 5 * (*cuDiffEq.psEval5)[idx] / 66) * dT;
				(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (31 * (*cuDiffEq.psEval0_2)[idx] / 384 + 1125 * (*cuDiffEq.psEval2_2)[idx] / 2816 + 9 * (*cuDiffEq.psEval3_2)[idx] / 32 + 125 * (*cuDiffEq.psEval4_2)[idx] / 768 + 5 * (*cuDiffEq.psEval5_2)[idx] / 66) * dT;

				//local truncation error from 5th order evaluation and 6th order evaluation
				cuReal3 lte_diff = 5 * ((*cuDiffEq.psEval0)[idx] + (*cuDiffEq.psEval5)[idx] - (*cuDiffEq.psEval6)[idx] - rhs) * dT / 66;

				if (*cuDiffEq.prenormalize) {

					cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
					cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
					(*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
					(*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
				}

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude(lte_diff) / (*cuMesh.pM)[idx].norm();
			}
			else {

				cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				(*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
				(*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

//----------------------------------------- DifferentialEquationCUDA Launchers

//RUNGE KUTTA FEHLBERG 5(6)

void DifferentialEquationAFMCUDA::RunRKF56(int step, bool calculate_mxh, bool calculate_dmdt)
{
	switch (step) {

	case 0:

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF56_Step0_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF56_Step0_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;

	case 1:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF56_Step1_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 2:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF56_Step2_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 3:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF56_Step3_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 4:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF56_Step4_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 5:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF56_Step5_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 6:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF56_Step6_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 7:

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF56_Step7_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF56_Step7_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;
	}
}

#endif
#endif
#endif