#include "DiffEqAFMCUDA.h"

#if COMPILECUDA == 1
#ifdef ODE_EVAL_COMPILATION_RKDP
#ifdef MESH_COMPILATION_ANTIFERROMAGNETIC

#include "MeshParamsControlCUDA.h"

#include "Reduction.cuh"

//defines evaluation methods kernel launchers

//----------------------------------------- EVALUATIONS : RKDP54

__global__ void RunRKDP54_Step0_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuBReal mxh = 0.0;
	cuBReal lte = 0.0;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			if (!cuMesh.pM->is_skipcell(idx)) {

				//obtain maximum normalized torque term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				mxh = cu_GetMagnitude((*cuMesh.pM)[idx] ^ (*cuMesh.pHeff)[idx]) / (Mnorm * Mnorm);

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs2;
				cuReal3 rhs = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, rhs2);

				//Now calculate 5th order evaluation for adaptive time step -> FSAL property (a full pass required for this to be valid)
				cuReal3 prediction = (*cuDiffEq.psM1)[idx] + (5179 * (*cuDiffEq.psEval0)[idx] / 57600 + 7571 * (*cuDiffEq.psEval2)[idx] / 16695 + 393 * (*cuDiffEq.psEval3)[idx] / 640 - 92097 * (*cuDiffEq.psEval4)[idx] / 339200 + 187 * (*cuDiffEq.psEval5)[idx] / 2100 + rhs / 40) * dT;

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuMesh.pM)[idx] - prediction) / Mnorm;

				//save evaluation for later use
				(*cuDiffEq.psEval0)[idx] = rhs;
				(*cuDiffEq.psEval0_2)[idx] = rhs2;
			}
		}
	}

	//only reduce for mxh if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &mxh, *cuDiffEq.pmxh);
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

__global__ void RunRKDP54_Step0_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
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

				//Now calculate 5th order evaluation for adaptive time step -> FSAL property (a full pass required for this to be valid)
				cuReal3 prediction = (*cuDiffEq.psM1)[idx] + (5179 * (*cuDiffEq.psEval0)[idx] / 57600 + 7571 * (*cuDiffEq.psEval2)[idx] / 16695 + 393 * (*cuDiffEq.psEval3)[idx] / 640 - 92097 * (*cuDiffEq.psEval4)[idx] / 339200 + 187 * (*cuDiffEq.psEval5)[idx] / 2100 + rhs / 40) * dT;

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuMesh.pM)[idx] - prediction) / (*cuMesh.pM)[idx].norm();

				//save evaluation for later use
				(*cuDiffEq.psEval0)[idx] = rhs;
				(*cuDiffEq.psEval0_2)[idx] = rhs2;
			}
		}
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

__global__ void RunRKDP54_Step0_Advance_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			if (!cuMesh.pM->is_skipcell(idx)) {

				//Save current magnetization for later use
				(*cuDiffEq.psM1)[idx] = (*cuMesh.pM)[idx];
				(*cuDiffEq.psM1_2)[idx] = (*cuMesh.pM2)[idx];

				//Now estimate magnetization using RKDP first step
				(*cuMesh.pM)[idx] += (*cuDiffEq.psEval0)[idx] * (dT / 5);
				(*cuMesh.pM2)[idx] += (*cuDiffEq.psEval0_2)[idx] * (dT / 5);
			}
		}
	}
}

__global__ void RunRKDP54_Step1_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval1)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval1_2)[idx]);

			//Now estimate magnetization using RKDP midle step 1
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (3 * (*cuDiffEq.psEval0)[idx] / 40 + 9 * (*cuDiffEq.psEval1)[idx] / 40) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (3 * (*cuDiffEq.psEval0_2)[idx] / 40 + 9 * (*cuDiffEq.psEval1_2)[idx] / 40) * dT;
		}
	}
}

__global__ void RunRKDP54_Step2_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval2)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval2_2)[idx]);

			//Now estimate magnetization using RKDP midle step 2
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (44 * (*cuDiffEq.psEval0)[idx] / 45 - 56 * (*cuDiffEq.psEval1)[idx] / 15 + 32 * (*cuDiffEq.psEval2)[idx] / 9) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (44 * (*cuDiffEq.psEval0_2)[idx] / 45 - 56 * (*cuDiffEq.psEval1_2)[idx] / 15 + 32 * (*cuDiffEq.psEval2_2)[idx] / 9) * dT;
		}
	}
}

__global__ void RunRKDP54_Step3_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval3)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval3_2)[idx]);

			//Now estimate magnetization using RKDP midle step 3
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (19372 * (*cuDiffEq.psEval0)[idx] / 6561 - 25360 * (*cuDiffEq.psEval1)[idx] / 2187 + 64448 * (*cuDiffEq.psEval2)[idx] / 6561 - 212 * (*cuDiffEq.psEval3)[idx] / 729) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (19372 * (*cuDiffEq.psEval0_2)[idx] / 6561 - 25360 * (*cuDiffEq.psEval1_2)[idx] / 2187 + 64448 * (*cuDiffEq.psEval2_2)[idx] / 6561 - 212 * (*cuDiffEq.psEval3_2)[idx] / 729) * dT;
		}
	}
}

__global__ void RunRKDP54_Step4_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx) && !cuMesh.pM->is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			(*cuDiffEq.psEval4)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval4_2)[idx]);

			//Now estimate magnetization using RKDP midle step 4
			(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (9017 * (*cuDiffEq.psEval0)[idx] / 3168 - 355 * (*cuDiffEq.psEval1)[idx] / 33 + 46732 * (*cuDiffEq.psEval2)[idx] / 5247 + 49 * (*cuDiffEq.psEval3)[idx] / 176 - 5103 * (*cuDiffEq.psEval4)[idx] / 18656) * dT;
			(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (9017 * (*cuDiffEq.psEval0_2)[idx] / 3168 - 355 * (*cuDiffEq.psEval1_2)[idx] / 33 + 46732 * (*cuDiffEq.psEval2_2)[idx] / 5247 + 49 * (*cuDiffEq.psEval3_2)[idx] / 176 - 5103 * (*cuDiffEq.psEval4_2)[idx] / 18656) * dT;
		}
	}
}

__global__ void RunRKDP54_Step5_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuBReal dmdt = 0.0;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				(*cuDiffEq.psEval5)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval5_2)[idx]);

				//RKDP54 : 5th order evaluation
				(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (35 * (*cuDiffEq.psEval0)[idx] / 384 + 500 * (*cuDiffEq.psEval2)[idx] / 1113 + 125 * (*cuDiffEq.psEval3)[idx] / 192 - 2187 * (*cuDiffEq.psEval4)[idx] / 6784 + 11 * (*cuDiffEq.psEval5)[idx] / 84) * dT;
				(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (35 * (*cuDiffEq.psEval0_2)[idx] / 384 + 500 * (*cuDiffEq.psEval2_2)[idx] / 1113 + 125 * (*cuDiffEq.psEval3_2)[idx] / 192 - 2187 * (*cuDiffEq.psEval4_2)[idx] / 6784 + 11 * (*cuDiffEq.psEval5_2)[idx] / 84) * dT;

				if (*cuDiffEq.prenormalize) {

					cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
					cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
					(*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
					(*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				dmdt = cu_GetMagnitude((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);
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
}

__global__ void RunRKDP54_Step5_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			if (!cuMesh.pM->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				(*cuDiffEq.psEval5)[idx] = (cuDiffEq.*(cuDiffEq.pODEFunc))(idx, (*cuDiffEq.psEval5_2)[idx]);

				//RKDP54 : 5th order evaluation
				(*cuMesh.pM)[idx] = (*cuDiffEq.psM1)[idx] + (35 * (*cuDiffEq.psEval0)[idx] / 384 + 500 * (*cuDiffEq.psEval2)[idx] / 1113 + 125 * (*cuDiffEq.psEval3)[idx] / 192 - 2187 * (*cuDiffEq.psEval4)[idx] / 6784 + 11 * (*cuDiffEq.psEval5)[idx] / 84) * dT;
				(*cuMesh.pM2)[idx] = (*cuDiffEq.psM1_2)[idx] + (35 * (*cuDiffEq.psEval0_2)[idx] / 384 + 500 * (*cuDiffEq.psEval2_2)[idx] / 1113 + 125 * (*cuDiffEq.psEval3_2)[idx] / 192 - 2187 * (*cuDiffEq.psEval4_2)[idx] / 6784 + 11 * (*cuDiffEq.psEval5_2)[idx] / 84) * dT;

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

//RUNGE KUTTA DORMAND-PRINCE

void DifferentialEquationAFMCUDA::RunRKDP54_Step0_NoAdvance(bool calculate_mxh)
{
	if (calculate_mxh) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKDP54_Step0_withReductions_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> > (cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKDP54_Step0_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> > (cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}
	}
}

void DifferentialEquationAFMCUDA::RunRKDP54(int step, bool calculate_mxh, bool calculate_dmdt)
{
	switch (step) {

	case 0:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKDP54_Step0_Advance_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 1:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKDP54_Step1_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 2:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKDP54_Step2_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 3:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKDP54_Step3_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 4:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKDP54_Step4_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 5:

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKDP54_Step5_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKDP54_Step5_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;
	}
}

#endif
#endif
#endif