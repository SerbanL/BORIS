#include "DiffEqAFMCUDA.h"

#if COMPILECUDA == 1
#ifdef MESH_COMPILATION_ANTIFERROMAGNETIC

#include "MeshParamsControlCUDA.h"

#include "Reduction.cuh"

//defines evaluation methods kernel launchers. LLG equation in-lined for faster evaluation

//----------------------------------------- EVALUATIONS: Euler

__global__ void RunEuler_LLG_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuReal3 mxh = cuReal3();
	cuReal3 dmdt = cuReal3();
	bool include_in_average = false;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//obtain average normalized torque term
				cuBReal Mnorm = M[idx].norm();
				mxh = (M[idx] ^ Heff[idx]) / (Mnorm * Mnorm);
				include_in_average = true;

				//Now estimate magnetization for the next time step
				M[idx] += rhs * dT;
				M2[idx] += rhs2 * dT;

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}
				
				//obtain maximum normalized dmdt term
				dmdt = ((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	//only reduce for dmdt (and mxh) if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_avg(0, 1, &mxh, *cuDiffEq.pmxh_av, *cuDiffEq.pavpoints, include_in_average);
		reduction_avg(0, 1, &dmdt, *cuDiffEq.pdmdt_av, *cuDiffEq.pavpoints2, include_in_average);
	}
}

__global__ void RunEuler_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//Now estimate magnetization for the next time step
				M[idx] += rhs * dT;
				M2[idx] += rhs2 * dT;

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}
}

//----------------------------------------- EVALUATIONS : Trapezoidal Euler

__global__ void RunTEuler_Step0_LLG_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuReal3 mxh = cuReal3();
	bool include_in_average = false;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization for the next step
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//obtain average normalized torque term
				cuBReal Mnorm = M[idx].norm();
				mxh = (M[idx] ^ Heff[idx]) / (Mnorm * Mnorm);
				include_in_average = true;

				//Now estimate magnetization for the next time step
				M[idx] += rhs * dT;
				M2[idx] += rhs2 * dT;
			}
		}
	}

	//only reduce for mxh if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_avg(0, 1, &mxh, *cuDiffEq.pmxh_av, *cuDiffEq.pavpoints, include_in_average);
	}
}

__global__ void RunTEuler_Step0_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization for the next step
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//Now estimate magnetization for the next time step
				M[idx] += rhs * dT;
				M2[idx] += rhs2 * dT;
			}
		}
	}
}

__global__ void RunTEuler_Step1_LLG_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuReal3 dmdt = cuReal3();
	bool include_in_average = false;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//Now estimate magnetization using the second trapezoidal Euler step equation
				M[idx] = ((*cuDiffEq.psM1)[idx] + M[idx] + rhs * dT) / 2;
				M2[idx] = ((*cuDiffEq.psM1_2)[idx] + M2[idx] + rhs2 * dT) / 2;

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				dmdt = ((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);
				include_in_average = true;
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	//only reduce for dmdt if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_avg(0, 1, &dmdt, *cuDiffEq.pdmdt_av, *cuDiffEq.pavpoints2, include_in_average);
	}
}

__global__ void RunTEuler_Step1_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//Now estimate magnetization using the second trapezoidal Euler step equation
				M[idx] = ((*cuDiffEq.psM1)[idx] + M[idx] + rhs * dT) / 2;
				M2[idx] = ((*cuDiffEq.psM1_2)[idx] + M2[idx] + rhs2 * dT) / 2;

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}
}

//----------------------------------------- EVALUATIONS : Adaptive Heun

//Step0 same as for TEuler

__global__ void RunAHeun_Step1_LLG_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuBReal lte = 0.0;
	cuReal3 dmdt = cuReal3();
	bool include_in_average = false;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//First save predicted magnetization for lte calculation
				cuReal3 saveM = (*cuMesh.pM)[idx];

				//Now estimate magnetization using the second trapezoidal Euler step equation
				M[idx] = ((*cuDiffEq.psM1)[idx] + M[idx] + rhs * dT) / 2;
				M2[idx] = ((*cuDiffEq.psM1_2)[idx] + M2[idx] + rhs2 * dT) / 2;

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				dmdt = ((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);
				include_in_average = true;

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuMesh.pM)[idx] - saveM) / (*cuMesh.pM)[idx].norm();
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	//only reduce for dmdt if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_avg(0, 1, &dmdt, *cuDiffEq.pdmdt_av, *cuDiffEq.pavpoints2, include_in_average);
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

__global__ void RunAHeun_Step1_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuBReal lte = 0.0;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//First save predicted magnetization for lte calculation
				cuReal3 saveM = (*cuMesh.pM)[idx];

				//Now estimate magnetization using the second trapezoidal Euler step equation
				M[idx] = ((*cuDiffEq.psM1)[idx] + M[idx] + rhs * dT) / 2;
				M2[idx] = ((*cuDiffEq.psM1_2)[idx] + M2[idx] + rhs2 * dT) / 2;

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuMesh.pM)[idx] - saveM) / (*cuMesh.pM)[idx].norm();
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

//----------------------------------------- EVALUATIONS : RK4

__global__ void RunRK4_Step0_LLG_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuBReal mxh = 0.0;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization for later use
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				(*cuDiffEq.psEval0)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				(*cuDiffEq.psEval0_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//obtain maximum normalized torque term
				cuBReal Mnorm = M[idx].norm();
				mxh = cu_GetMagnitude(M[idx] ^ Heff[idx]) / (Mnorm * Mnorm);

				//Now estimate magnetization using RK4 midle step
				M[idx] += (*cuDiffEq.psEval0)[idx] * (dT / 2);
				M2[idx] += (*cuDiffEq.psEval0_2)[idx] * (dT / 2);
			}
		}
	}

	//only reduce for mxh if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &mxh, *cuDiffEq.pmxh);
	}
}

__global__ void RunRK4_Step0_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization for later use
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);
				
				(*cuDiffEq.psEval0)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				(*cuDiffEq.psEval0_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//Now estimate magnetization using RK4 midle step
				M[idx] += (*cuDiffEq.psEval0)[idx] * (dT / 2);
				M2[idx] += (*cuDiffEq.psEval0_2)[idx] * (dT / 2);
			}
		}
	}
}

__global__ void RunRK4_Step1_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx) && !M.is_skipcell(idx)) {

			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);
			
			(*cuDiffEq.psEval1)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
			(*cuDiffEq.psEval1_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

			//Now estimate magnetization using RK4 midle step
			M[idx] = (*cuDiffEq.psM1)[idx] + (*cuDiffEq.psEval1)[idx] * (dT / 2);
			M2[idx] = (*cuDiffEq.psM1_2)[idx] + (*cuDiffEq.psEval1_2)[idx] * (dT / 2);
		}
	}
}

__global__ void RunRK4_Step2_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx) && !M.is_skipcell(idx)) {

			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);
			
			(*cuDiffEq.psEval2)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
			(*cuDiffEq.psEval2_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

			//Now estimate magnetization using RK4 last step
			M[idx] = (*cuDiffEq.psM1)[idx] + (*cuDiffEq.psEval2)[idx] * dT;
			M2[idx] = (*cuDiffEq.psM1_2)[idx] + (*cuDiffEq.psEval2_2)[idx] * dT;

		}
	}
}

__global__ void RunRK4_Step3_LLG_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuBReal dmdt = 0.0;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//Now estimate magnetization using previous RK4 evaluations
				M[idx] = (*cuDiffEq.psM1)[idx] + ((*cuDiffEq.psEval0)[idx] + 2 * (*cuDiffEq.psEval1)[idx] + 2 * (*cuDiffEq.psEval2)[idx] + rhs) * (dT / 6);
				M2[idx] = (*cuDiffEq.psM1_2)[idx] + ((*cuDiffEq.psEval0_2)[idx] + 2 * (*cuDiffEq.psEval1_2)[idx] + 2 * (*cuDiffEq.psEval2_2)[idx] + rhs2) * (dT / 6);

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				dmdt = cu_GetMagnitude((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	//only reduce for dmdt if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &dmdt, *cuDiffEq.pdmdt);
	}
}

__global__ void RunRK4_Step3_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);
				
				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//Now estimate magnetization using previous RK4 evaluations
				M[idx] = (*cuDiffEq.psM1)[idx] + ((*cuDiffEq.psEval0)[idx] + 2 * (*cuDiffEq.psEval1)[idx] + 2 * (*cuDiffEq.psEval2)[idx] + rhs) * (dT / 6);
				M2[idx] = (*cuDiffEq.psM1_2)[idx] + ((*cuDiffEq.psEval0_2)[idx] + 2 * (*cuDiffEq.psEval1_2)[idx] + 2 * (*cuDiffEq.psEval2_2)[idx] + rhs2) * (dT / 6);

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}
}

//----------------------------------------- EVALUATIONS : ABM

__global__ void RunABM_Predictor_LLG_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuBReal mxh = 0.0;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization for the next step
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//obtain maximum normalized torque term
				cuBReal Mnorm = M[idx].norm();
				mxh = cu_GetMagnitude(M[idx] ^ Heff[idx]) / (Mnorm * Mnorm);

				//ABM predictor : pk+1 = mk + (dt/2) * (3*fk - fk-1)
				if (*cuDiffEq.palternator) {

					M[idx] += dT * (3 * rhs - (*cuDiffEq.psEval0)[idx]) / 2;
					M2[idx] += dT * (3 * rhs2 - (*cuDiffEq.psEval0_2)[idx]) / 2;

					(*cuDiffEq.psEval1)[idx] = rhs;
					(*cuDiffEq.psEval1_2)[idx] = rhs2;
				}
				else {

					M[idx] += dT * (3 * rhs - (*cuDiffEq.psEval1)[idx]) / 2;
					M2[idx] += dT * (3 * rhs2 - (*cuDiffEq.psEval1_2)[idx]) / 2;

					(*cuDiffEq.psEval0)[idx] = rhs;
					(*cuDiffEq.psEval0_2)[idx] = rhs2;
				}
			}
		}
	}

	//only reduce for mxh if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &mxh, *cuDiffEq.pmxh);
	}
}

__global__ void RunABM_Predictor_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization for the next step
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//ABM predictor : pk+1 = mk + (dt/2) * (3*fk - fk-1)
				if (*cuDiffEq.palternator) {

					M[idx] += dT * (3 * rhs - (*cuDiffEq.psEval0)[idx]) / 2;
					M2[idx] += dT * (3 * rhs2 - (*cuDiffEq.psEval0_2)[idx]) / 2;

					(*cuDiffEq.psEval1)[idx] = rhs;
					(*cuDiffEq.psEval1_2)[idx] = rhs2;
				}
				else {

					M[idx] += dT * (3 * rhs - (*cuDiffEq.psEval1)[idx]) / 2;
					M2[idx] += dT * (3 * rhs2 - (*cuDiffEq.psEval1_2)[idx]) / 2;

					(*cuDiffEq.psEval0)[idx] = rhs;
					(*cuDiffEq.psEval0_2)[idx] = rhs2;
				}
			}
		}
	}
}

__global__ void RunABM_Corrector_LLG_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuBReal dmdt = 0.0;
	cuBReal lte = 0.0;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//First save predicted magnetization for lte calculation
				cuReal3 saveM = M[idx];

				//ABM corrector : mk+1 = mk + (dt/2) * (fk+1 + fk)
				if (*cuDiffEq.palternator) {

					M[idx] = (*cuDiffEq.psM1)[idx] + dT * (rhs + (*cuDiffEq.psEval1)[idx]) / 2;
					M2[idx] = (*cuDiffEq.psM1_2)[idx] + dT * (rhs2 + (*cuDiffEq.psEval1_2)[idx]) / 2;
				}
				else {

					M[idx] = (*cuDiffEq.psM1)[idx] + dT * (rhs + (*cuDiffEq.psEval0)[idx]) / 2;
					M2[idx] = (*cuDiffEq.psM1_2)[idx] + dT * (rhs2 + (*cuDiffEq.psEval0_2)[idx]) / 2;
				}

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				dmdt = cu_GetMagnitude((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);		

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude(M[idx] - saveM) / Mnorm;
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	//only reduce for dmdt if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &dmdt, *cuDiffEq.pdmdt);
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

__global__ void RunABM_Corrector_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuBReal lte = 0.0;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//First save predicted magnetization for lte calculation
				cuReal3 saveM = M[idx];

				//ABM corrector : mk+1 = mk + (dt/2) * (fk+1 + fk)
				if (*cuDiffEq.palternator) {

					M[idx] = (*cuDiffEq.psM1)[idx] + dT * (rhs + (*cuDiffEq.psEval1)[idx]) / 2;
					M2[idx] = (*cuDiffEq.psM1_2)[idx] + dT * (rhs2 + (*cuDiffEq.psEval1_2)[idx]) / 2;
				}
				else {

					M[idx] = (*cuDiffEq.psM1)[idx] + dT * (rhs + (*cuDiffEq.psEval0)[idx]) / 2;
					M2[idx] = (*cuDiffEq.psM1_2)[idx] + dT * (rhs2 + (*cuDiffEq.psEval0_2)[idx]) / 2;
				}

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude(M[idx] - saveM) / (*cuMesh.pM)[idx].norm();
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

__global__ void RunABMTEuler_Step0_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization for the next step
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				(*cuDiffEq.psEval0)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				(*cuDiffEq.psEval0_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//Now estimate magnetization for the next time step
				M[idx] += (*cuDiffEq.psEval0)[idx] * dT;
				M2[idx] += (*cuDiffEq.psEval0_2)[idx] * dT;
			}
		}
	}
}

__global__ void RunABMTEuler_Step1_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//Now estimate magnetization using the second trapezoidal Euler step equation
				M[idx] = ((*cuDiffEq.psM1)[idx] + M[idx] + rhs * dT) / 2;
				M2[idx] = ((*cuDiffEq.psM1_2)[idx] + M2[idx] + rhs2 * dT) / 2;

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}
}

//----------------------------------------- EVALUATIONS : RKF45

__global__ void RunRKF45_Step0_LLG_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuBReal mxh = 0.0;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization for later use
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				(*cuDiffEq.psEval0)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				(*cuDiffEq.psEval0_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//obtain maximum normalized torque term
				cuBReal Mnorm = M[idx].norm();
				mxh = cu_GetMagnitude(M[idx] ^ Heff[idx]) / (Mnorm * Mnorm);

				//Now estimate magnetization using RKF first step
				M[idx] += (*cuDiffEq.psEval0)[idx] * (2 * dT / 9);
				M2[idx] += (*cuDiffEq.psEval0_2)[idx] * (2 * dT / 9);
			}
		}
	}

	//only reduce for mxh if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &mxh, *cuDiffEq.pmxh);
	}
}

__global__ void RunRKF45_Step0_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			//Save current magnetization for later use
			(*cuDiffEq.psM1)[idx] = M[idx];
			(*cuDiffEq.psM1_2)[idx] = M2[idx];

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				(*cuDiffEq.psEval0)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				(*cuDiffEq.psEval0_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//Now estimate magnetization using RKF first step
				M[idx] += (*cuDiffEq.psEval0)[idx] * (2 * dT / 9);
				M2[idx] += (*cuDiffEq.psEval0_2)[idx] * (2 * dT / 9);
			}
		}
	}
}

__global__ void RunRKF45_Step1_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx) && !M.is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

			(*cuDiffEq.psEval1)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
			(*cuDiffEq.psEval1_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

			//Now estimate magnetization using RKF midle step 1
			M[idx] = (*cuDiffEq.psM1)[idx] + ((*cuDiffEq.psEval0)[idx] / 12 + (*cuDiffEq.psEval1)[idx] / 4) * dT;
			M2[idx] = (*cuDiffEq.psM1_2)[idx] + ((*cuDiffEq.psEval0_2)[idx] / 12 + (*cuDiffEq.psEval1_2)[idx] / 4) * dT;
		}
	}
}

__global__ void RunRKF45_Step2_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx) && !M.is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

			(*cuDiffEq.psEval2)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
			(*cuDiffEq.psEval2_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

			//Now estimate magnetization using RKF midle step 2
			M[idx] = (*cuDiffEq.psM1)[idx] + (69 * (*cuDiffEq.psEval0)[idx] / 128 - 243 * (*cuDiffEq.psEval1)[idx] / 128 + 135 * (*cuDiffEq.psEval2)[idx] / 64) * dT;
			M2[idx] = (*cuDiffEq.psM1_2)[idx] + (69 * (*cuDiffEq.psEval0_2)[idx] / 128 - 243 * (*cuDiffEq.psEval1_2)[idx] / 128 + 135 * (*cuDiffEq.psEval2_2)[idx] / 64) * dT;
		}
	}
}

__global__ void RunRKF45_Step3_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx) && !M.is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

			(*cuDiffEq.psEval3)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
			(*cuDiffEq.psEval3_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

			//Now estimate magnetization using RKF midle step 3
			M[idx] = (*cuDiffEq.psM1)[idx] + (-17 * (*cuDiffEq.psEval0)[idx] / 12 + 27 * (*cuDiffEq.psEval1)[idx] / 4 - 27 * (*cuDiffEq.psEval2)[idx] / 5 + 16 * (*cuDiffEq.psEval3)[idx] / 15) * dT;
			M2[idx] = (*cuDiffEq.psM1_2)[idx] + (-17 * (*cuDiffEq.psEval0_2)[idx] / 12 + 27 * (*cuDiffEq.psEval1_2)[idx] / 4 - 27 * (*cuDiffEq.psEval2_2)[idx] / 5 + 16 * (*cuDiffEq.psEval3_2)[idx] / 15) * dT;
		}
	}
}

__global__ void RunRKF45_Step4_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx) && !M.is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

			(*cuDiffEq.psEval4)[idx] = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
			(*cuDiffEq.psEval4_2)[idx] = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

			//Now estimate magnetization using RKF midle step 4
			M[idx] = (*cuDiffEq.psM1)[idx] + (65 * (*cuDiffEq.psEval0)[idx] / 432 - 5 * (*cuDiffEq.psEval1)[idx] / 16 + 13 * (*cuDiffEq.psEval2)[idx] / 16 + 4 * (*cuDiffEq.psEval3)[idx] / 27 + 5 * (*cuDiffEq.psEval4)[idx] / 144) * dT;
			M2[idx] = (*cuDiffEq.psM1_2)[idx] + (65 * (*cuDiffEq.psEval0_2)[idx] / 432 - 5 * (*cuDiffEq.psEval1_2)[idx] / 16 + 13 * (*cuDiffEq.psEval2_2)[idx] / 16 + 4 * (*cuDiffEq.psEval3_2)[idx] / 27 + 5 * (*cuDiffEq.psEval4_2)[idx] / 144) * dT;
		}
	}
}

__global__ void RunRKF45_Step5_LLG_withReductions_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuBReal dmdt = 0.0;
	cuBReal lte = 0.0;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//4th order evaluation
				M[idx] = (*cuDiffEq.psM1)[idx] + ((*cuDiffEq.psEval0)[idx] / 9 + 9 * (*cuDiffEq.psEval2)[idx] / 20 + 16 * (*cuDiffEq.psEval3)[idx] / 45 + (*cuDiffEq.psEval4)[idx] / 12) * dT;
				M2[idx] = (*cuDiffEq.psM1_2)[idx] + ((*cuDiffEq.psEval0_2)[idx] / 9 + 9 * (*cuDiffEq.psEval2_2)[idx] / 20 + 16 * (*cuDiffEq.psEval3_2)[idx] / 45 + (*cuDiffEq.psEval4_2)[idx] / 12) * dT;

				//5th order evaluation
				cuReal3 prediction = (*cuDiffEq.psM1)[idx] + (47 * (*cuDiffEq.psEval0)[idx] / 450 + 12 * (*cuDiffEq.psEval2)[idx] / 25 + 32 * (*cuDiffEq.psEval3)[idx] / 225 + 1 * (*cuDiffEq.psEval4)[idx] / 30 + 6 * rhs / 25) * dT;

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuMesh.pM)[idx].norm();
				dmdt = cu_GetMagnitude((*cuMesh.pM)[idx] - (*cuDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * Mnorm * Mnorm);

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude(M[idx] - prediction) / Mnorm;
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	//only reduce for dmdt if grel is not zero (if it's zero this means magnetization dynamics is disabled in this mesh)
	if (cuMesh.pgrel->get0()) {

		reduction_max(0, 1, &dmdt, *cuDiffEq.pdmdt);
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

__global__ void RunRKF45_Step5_LLG_Kernel(ManagedDiffEqAFMCUDA& cuDiffEq, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuDiffEq.pdT;

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
	cuReal2 alpha_AFM = *cuMesh.palpha_AFM;
	cuReal2 grel_AFM = *cuMesh.pgrel_AFM;

	cuBReal lte = 0.0;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			if (!M.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.palpha_AFM, alpha_AFM, *cuMesh.pgrel_AFM, grel_AFM);

				cuReal3 rhs = (-(cuBReal)GAMMA * grel_AFM.i / (1 + alpha_AFM.i * alpha_AFM.i)) * ((M[idx] ^ Heff[idx]) + alpha_AFM.i * ((M[idx] / Ms_AFM.i) ^ (M[idx] ^ Heff[idx])));
				cuReal3 rhs2 = (-(cuBReal)GAMMA * grel_AFM.j / (1 + alpha_AFM.j * alpha_AFM.j)) * ((M2[idx] ^ Heff2[idx]) + alpha_AFM.j * ((M2[idx] / Ms_AFM.j) ^ (M2[idx] ^ Heff2[idx])));

				//4th order evaluation
				M[idx] = (*cuDiffEq.psM1)[idx] + ((*cuDiffEq.psEval0)[idx] / 9 + 9 * (*cuDiffEq.psEval2)[idx] / 20 + 16 * (*cuDiffEq.psEval3)[idx] / 45 + (*cuDiffEq.psEval4)[idx] / 12) * dT;
				M2[idx] = (*cuDiffEq.psM1_2)[idx] + ((*cuDiffEq.psEval0_2)[idx] / 9 + 9 * (*cuDiffEq.psEval2_2)[idx] / 20 + 16 * (*cuDiffEq.psEval3_2)[idx] / 45 + (*cuDiffEq.psEval4_2)[idx] / 12) * dT;

				//5th order evaluation
				cuReal3 prediction = (*cuDiffEq.psM1)[idx] + (47 * (*cuDiffEq.psEval0)[idx] / 450 + 12 * (*cuDiffEq.psEval2)[idx] / 25 + 32 * (*cuDiffEq.psEval3)[idx] / 225 + 1 * (*cuDiffEq.psEval4)[idx] / 30 + 6 * rhs / 25) * dT;

				if (*cuDiffEq.prenormalize) {

					M[idx].renormalize(Ms_AFM.i);
					M2[idx].renormalize(Ms_AFM.j);
				}

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude(M[idx] - prediction) / (*cuMesh.pM)[idx].norm();
			}
			else {

				cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);
				M[idx].renormalize(Ms_AFM.i);
				M2[idx].renormalize(Ms_AFM.j);
			}
		}
	}

	reduction_max(0, 1, &lte, *cuDiffEq.plte);
}

//----------------------------------------- DifferentialEquationCUDA Launchers

//EULER

void DifferentialEquationAFMCUDA::RunEuler_LLG(bool calculate_mxh, bool calculate_dmdt)
{
	if (calculate_mxh || calculate_dmdt) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunEuler_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunEuler_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}
	}
}

//TRAPEZOIDAL EULER

void DifferentialEquationAFMCUDA::RunTEuler_LLG(int step, bool calculate_mxh, bool calculate_dmdt)
{
	if (step == 0) {

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step0_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step0_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
	}
	else {

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step1_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step1_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
	}
}

//ADAPTIVE HEUN

void DifferentialEquationAFMCUDA::RunAHeun_LLG(int step, bool calculate_mxh, bool calculate_dmdt)
{
	if (step == 0) {

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step0_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step0_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
	}
	else {

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunAHeun_Step1_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunAHeun_Step1_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
	}
}

//RUNGE KUTTA 4th order

void DifferentialEquationAFMCUDA::RunRK4_LLG(int step, bool calculate_mxh, bool calculate_dmdt)
{
	switch (step) {

	case 0:

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRK4_Step0_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRK4_Step0_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;

	case 1:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRK4_Step1_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 2:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRK4_Step2_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 3:

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRK4_Step3_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRK4_Step3_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;
	}
}

//Adams-Bashforth-Moulton 2nd order

void DifferentialEquationAFMCUDA::RunABM_LLG(int step, bool calculate_mxh, bool calculate_dmdt)
{
	if (step == 0) {

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Predictor_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Predictor_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
	}
	else {

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Corrector_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Corrector_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
	}
}

//Adams-Bashforth-Moulton 2nd order priming using Trapezoidal Euler

void DifferentialEquationAFMCUDA::RunABMTEuler_LLG(int step)
{
	if (step == 0) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunABMTEuler_Step0_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunABMTEuler_Step1_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}
	}
}

//RUNGE KUTTA FEHLBERG

void DifferentialEquationAFMCUDA::RunRKF45_LLG(int step, bool calculate_mxh, bool calculate_dmdt)
{
	switch (step) {

	case 0:

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF45_Step0_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF45_Step0_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;

	case 1:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step1_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 2:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step2_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 3:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step3_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 4:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step4_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}

		break;

	case 5:

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF45_Step5_LLG_withReductions_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF45_Step5_LLG_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuDiffEq.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
			}
		}

		break;
	}
}

#endif
#endif
