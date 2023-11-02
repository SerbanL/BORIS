#include "DiffEqAFMCUDA.h"

#if COMPILECUDA == 1
#ifdef ODE_EVAL_COMPILATION_RKCK
#ifdef MESH_COMPILATION_ANTIFERROMAGNETIC

#include "MeshParamsControlCUDA.h"

#include "Reduction.cuh"

//defines evaluation methods kernel launchers

//----------------------------------------- EVALUATIONS : RKCK45

__global__ void RunRKCK45_Step0_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
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

				//Now estimate magnetization using RKCK first step
				(*cuMesh.pM)[idx] += (*cuDiffEq.psEval0)[idx] * (dT / 5);
				(*cuMesh.pM2)[idx] += (*cuDiffEq.psEval0_2)[idx] * (dT / 5);
			}
		}
	}

	//only reduce for mxh if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &mxh, *cuDiffEq.pmxh);
	}
}

__global__ void RunRKCK45_Step0_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
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

				//Now estimate magnetization using RKCK first step
				(*cuMesh.pM)[idx] += (*cuDiffEq.psEval0)[idx] * (dT / 5);
				(*cuMesh.pM2)[idx] += (*cuDiffEq.psEval0_2)[idx] * (dT / 5);
			}
		}
	}
}

__global__ void RunRKCK45_Step1_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval1)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval1_2)[idx]);

			//Now estimate magnetization using RKCK midle step 1
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (3 * (*cuDiffEq.psEval0)[idx] / 40 + 9 * (*cuDiffEq.psEval1)[idx] / 40) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (3 * (*cuDiffEq.psEval0_2)[idx] / 40 + 9 * (*cuDiffEq.psEval1_2)[idx] / 40) * dT;
		}
	}
}

__global__ void RunRKCK45_Step2_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval2)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval2_2)[idx]);

			//Now estimate magnetization using RKCK midle step 2
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (3 * (*cuDiffEq.psEval0)[idx] / 10 - 9 * (*cuDiffEq.psEval1)[idx] / 10 + 6 * (*cuDiffEq.psEval2)[idx] / 5) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (3 * (*cuDiffEq.psEval0_2)[idx] / 10 - 9 * (*cuDiffEq.psEval1_2)[idx] / 10 + 6 * (*cuDiffEq.psEval2_2)[idx] / 5) * dT;
		}
	}
}

__global__ void RunRKCK45_Step3_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval3)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval3_2)[idx]);

			//Now estimate magnetization using RKCK midle step 3
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (-11 * (*cuDiffEq.psEval0)[idx] / 54 + 5 * (*cuDiffEq.psEval1)[idx] / 2 - 70 * (*cuDiffEq.psEval2)[idx] / 27 + 35 * (*cuDiffEq.psEval3)[idx] / 27) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (-11 * (*cuDiffEq.psEval0_2)[idx] / 54 + 5 * (*cuDiffEq.psEval1_2)[idx] / 2 - 70 * (*cuDiffEq.psEval2_2)[idx] / 27 + 35 * (*cuDiffEq.psEval3_2)[idx] / 27) * dT;
		}
	}
}

__global__ void RunRKCK45_Step4_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval4)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval4_2)[idx]);

			//Now estimate magnetization using RKF midle step 4
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (1631 * (*cuDiffEq.psEval0)[idx] / 55296 + 175 * (*cuDiffEq.psEval1)[idx] / 512 + 575 * (*cuDiffEq.psEval2)[idx] / 13824 + 44275 * (*cuDiffEq.psEval3)[idx] / 110592 + 253 * (*cuDiffEq.psEval4)[idx] / 4096) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (1631 * (*cuDiffEq.psEval0_2)[idx] / 55296 + 175 * (*cuDiffEq.psEval1_2)[idx] / 512 + 575 * (*cuDiffEq.psEval2_2)[idx] / 13824 + 44275 * (*cuDiffEq.psEval3_2)[idx] / 110592 + 253 * (*cuDiffEq.psEval4_2)[idx] / 4096) * dT;
		}
	}
}

__global__ void RunRKCK45_Step5_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
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

				//RKCK45 : 4th order evaluation
				(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (2825 * (*cuDiffEq.psEval0)[idx] / 27648 + 18575 * (*cuDiffEq.psEval2)[idx] / 48384 + 13525 * (*cuDiffEq.psEval3)[idx] / 55296 + 277 * (*cuDiffEq.psEval4)[idx] / 14336 + rhs / 4) * dT;
				(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (2825 * (*cuDiffEq.psEval0_2)[idx] / 27648 + 18575 * (*cuDiffEq.psEval2_2)[idx] / 48384 + 13525 * (*cuDiffEq.psEval3_2)[idx] / 55296 + 277 * (*cuDiffEq.psEval4_2)[idx] / 14336 + rhs2 / 4) * dT;

				//Now calculate 5th order evaluation for adaptive time step
				cuReal3 prediction = (*cuDiffEq.psM1)[idx] + (37 * (*cuDiffEq.psEval0)[idx] / 378 + 250 * (*cuDiffEq.psEval2)[idx] / 621 + 125 * (*cuDiffEq.psEval3)[idx] / 594 + 512 * rhs / 1771) * dT;

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
				lte = cu_GetMagnitude((*cuMesh.pM)[idx] - prediction) / (*cuMesh.pM)[idx].norm();
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

__global__ void RunRKCK45_Step5_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
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

				//RKCK45 : 4th order evaluation
				(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (2825 * (*cuDiffEq.psEval0)[idx] / 27648 + 18575 * (*cuDiffEq.psEval2)[idx] / 48384 + 13525 * (*cuDiffEq.psEval3)[idx] / 55296 + 277 * (*cuDiffEq.psEval4)[idx] / 14336 + rhs / 4) * dT;
				(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (2825 * (*cuDiffEq.psEval0_2)[idx] / 27648 + 18575 * (*cuDiffEq.psEval2_2)[idx] / 48384 + 13525 * (*cuDiffEq.psEval3_2)[idx] / 55296 + 277 * (*cuDiffEq.psEval4_2)[idx] / 14336 + rhs2 / 4) * dT;

				//Now calculate 5th order evaluation for adaptive time step
				cuReal3 prediction = (*cuDiffEq.psM1)[idx] + (37 * (*cuDiffEq.psEval0)[idx] / 378 + 250 * (*cuDiffEq.psEval2)[idx] / 621 + 125 * (*cuDiffEq.psEval3)[idx] / 594 + 512 * rhs / 1771) * dT;

				if (*cuDiffEq.prenormalize) {

					cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
					cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
					(*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
					(*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
				}

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuMesh.pM)[idx] - prediction) / (*cuMesh.pM)[idx].norm();
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

//RUNGE KUTTA CASH-KARP

void DifferentialEquationAFMCUDA::RunRKCK45(int step, bool calculate_mxh, bool calculate_dmdt)
{
	switch (step) {

	case 0:

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKCK45_Step0_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKCK45_Step0_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;

	case 1:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKCK45_Step1_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 2:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKCK45_Step2_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 3:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKCK45_Step3_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 4:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKCK45_Step4_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 5:

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKCK45_Step5_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKCK45_Step5_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;
	}
}

#endif
#endif
#endif