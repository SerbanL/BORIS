#include "Atom_DiffEqCubicCUDA.h"

#if COMPILECUDA == 1
#ifdef MESH_COMPILATION_ATOM_CUBIC

#include "Atom_MeshParamsControlCUDA.h"

#include "Reduction.cuh"

//defines evaluation methods kernel launchers. LLG equation in-lined for faster evaluation

//----------------------------------------- EVALUATIONS: Euler

__global__ void RunEuler_LLG_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuReal3 mxh = cuReal3();
	cuReal3 dmdt = cuReal3();
	bool include_in_average = false;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / M1.h.dim();

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//obtain average normalized torque term
				cuBReal Mnorm = M1[idx].norm();
				mxh = ((*cuaMesh.pM1)[idx] ^ (*cuaMesh.pHeff1)[idx]) / (conversion * Mnorm * Mnorm);
				include_in_average = true;

				//Now estimate moment for the next time step
				M1[idx] += rhs * dT;

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}

				//obtain maximum normalized dmdt term
				dmdt = ((*cuaMesh.pM1)[idx] - (*cuaDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * conversion * Mnorm * Mnorm);
				include_in_average = true;
			}
		}
	}

	if (cuaMesh.pgrel->get0()) {

		reduction_avg(0, 1, &mxh, *cuaDiffEq.pmxh_av, *cuaDiffEq.pavpoints, include_in_average);
		reduction_avg(0, 1, &dmdt, *cuaDiffEq.pdmdt_av, *cuaDiffEq.pavpoints2, include_in_average);
	}
}

__global__ void RunEuler_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//Now estimate moment for the next time step
				M1[idx] += rhs * dT;
				
				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}
			}
		}
	}
}

//----------------------------------------- EVALUATIONS : Trapezoidal Euler

__global__ void RunTEuler_Step0_LLG_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuReal3 mxh = cuReal3();
	bool include_in_average = false;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / M1.h.dim();

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment for the next step
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//obtain average normalized torque term
				cuBReal Mnorm = M1[idx].norm();
				mxh = (M1[idx] ^ Heff1[idx]) / (conversion * Mnorm * Mnorm);
				include_in_average = true;

				//Now estimate moment for the next time step
				M1[idx] += rhs * dT;
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_avg(0, 1, &mxh, *cuaDiffEq.pmxh_av, *cuaDiffEq.pavpoints, include_in_average);
}

__global__ void RunTEuler_Step0_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment for the next step
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//Now estimate moment for the next time step
				M1[idx] += rhs * dT;
			}
		}
	}
}

__global__ void RunTEuler_Step1_LLG_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuReal3 dmdt = cuReal3();
	bool include_in_average = false;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / M1.h.dim();

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//Now estimate moment using the second trapezoidal Euler step equation
				M1[idx] = ((*cuaDiffEq.psM1)[idx] + M1[idx] + rhs * dT) / 2;

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuaMesh.pM1)[idx].norm();
				dmdt = ((*cuaMesh.pM1)[idx] - (*cuaDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * conversion * Mnorm * Mnorm);
				include_in_average = true;
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_avg(0, 1, &dmdt, *cuaDiffEq.pdmdt_av, *cuaDiffEq.pavpoints2, include_in_average);
}

__global__ void RunTEuler_Step1_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//Now estimate moment using the second trapezoidal Euler step equation
				M1[idx] = ((*cuaDiffEq.psM1)[idx] + M1[idx] + rhs * dT) / 2;

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}
			}
		}
	}
}

//----------------------------------------- EVALUATIONS : Adaptive Heun

//Step0 same as for TEuler

__global__ void RunAHeun_Step1_LLG_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuReal3 dmdt = cuReal3();
	bool include_in_average = false;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / M1.h.dim();

	cuBReal lte = 0.0;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//First save predicted moment for lte calculation
				cuReal3 saveM = (*cuaMesh.pM1)[idx];

				//Now estimate moment using the second trapezoidal Euler step equation
				M1[idx] = ((*cuaDiffEq.psM1)[idx] + M1[idx] + rhs * dT) / 2;

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuaMesh.pM1)[idx].norm();
				dmdt = ((*cuaMesh.pM1)[idx] - (*cuaDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * conversion * Mnorm * Mnorm);
				include_in_average = true;

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuaMesh.pM1)[idx] - saveM) / (*cuaMesh.pM1)[idx].norm();
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_avg(0, 1, &dmdt, *cuaDiffEq.pdmdt_av, *cuaDiffEq.pavpoints2, include_in_average);

	reduction_max(0, 1, &lte, *cuaDiffEq.plte);
}

__global__ void RunAHeun_Step1_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuBReal lte = 0.0;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//First save predicted moment for lte calculation
				cuReal3 saveM = (*cuaMesh.pM1)[idx];

				//Now estimate moment using the second trapezoidal Euler step equation
				M1[idx] = ((*cuaDiffEq.psM1)[idx] + M1[idx] + rhs * dT) / 2;

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuaMesh.pM1)[idx] - saveM) / (*cuaMesh.pM1)[idx].norm();
			}
		}
	}

	reduction_max(0, 1, &lte, *cuaDiffEq.plte);
}

//----------------------------------------- EVALUATIONS : RK4

__global__ void RunRK4_Step0_LLG_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuBReal mxh = 0.0;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / M1.h.dim();

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment for later use
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);

				(*cuaDiffEq.psEval0)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//obtain maximum normalized torque term
				cuBReal Mnorm = M1[idx].norm();
				mxh = cu_GetMagnitude(M1[idx] ^ Heff1[idx]) / (conversion * Mnorm * Mnorm);

				//Now estimate moment using RK4 midle step
				M1[idx] += (*cuaDiffEq.psEval0)[idx] * (dT / 2);
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_max(0, 1, &mxh, *cuaDiffEq.pmxh);
}

__global__ void RunRK4_Step0_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment for later use
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				
				(*cuaDiffEq.psEval0)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//Now estimate moment using RK4 midle step
				M1[idx] += (*cuaDiffEq.psEval0)[idx] * (dT / 2);
			}
		}
	}
}

__global__ void RunRK4_Step1_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx) && !M1.is_skipcell(idx)) {

			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
			
			(*cuaDiffEq.psEval1)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

			//Now estimate moment using RK4 midle step
			M1[idx] = (*cuaDiffEq.psM1)[idx] + (*cuaDiffEq.psEval1)[idx] * (dT / 2);
		}
	}
}

__global__ void RunRK4_Step2_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx) && !M1.is_skipcell(idx)) {

			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
			
			(*cuaDiffEq.psEval2)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

			//Now estimate moment using RK4 last step
			M1[idx] = (*cuaDiffEq.psM1)[idx] + (*cuaDiffEq.psEval2)[idx] * dT;
		}
	}
}

__global__ void RunRK4_Step3_LLG_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuBReal dmdt = 0.0;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / M1.h.dim();

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);

				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//Now estimate moment using previous RK4 evaluations
				M1[idx] = (*cuaDiffEq.psM1)[idx] + ((*cuaDiffEq.psEval0)[idx] + 2 * (*cuaDiffEq.psEval1)[idx] + 2 * (*cuaDiffEq.psEval2)[idx] + rhs) * (dT / 6);

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuaMesh.pM1)[idx].norm();
				dmdt = cu_GetMagnitude((*cuaMesh.pM1)[idx] - (*cuaDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * conversion * Mnorm * Mnorm);
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_max(0, 1, &dmdt, *cuaDiffEq.pdmdt);
}

__global__ void RunRK4_Step3_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//Now estimate moment using previous RK4 evaluations
				M1[idx] = (*cuaDiffEq.psM1)[idx] + ((*cuaDiffEq.psEval0)[idx] + 2 * (*cuaDiffEq.psEval1)[idx] + 2 * (*cuaDiffEq.psEval2)[idx] + rhs) * (dT / 6);

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}
			}
		}
	}
}

//----------------------------------------- EVALUATIONS : ABM

__global__ void RunABM_Predictor_LLG_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuBReal mxh = 0.0;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / M1.h.dim();

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment for the next step
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//obtain maximum normalized torque term
				cuBReal Mnorm = M1[idx].norm();
				mxh = cu_GetMagnitude(M1[idx] ^ Heff1[idx]) / (conversion * Mnorm * Mnorm);

				//ABM predictor : pk+1 = mk + (dt/2) * (3*fk - fk-1)
				if (*cuaDiffEq.palternator) {

					M1[idx] += dT * (3 * rhs - (*cuaDiffEq.psEval0)[idx]) / 2;
					(*cuaDiffEq.psEval1)[idx] = rhs;
				}
				else {

					M1[idx] += dT * (3 * rhs - (*cuaDiffEq.psEval1)[idx]) / 2;
					(*cuaDiffEq.psEval0)[idx] = rhs;
				}
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_max(0, 1, &mxh, *cuaDiffEq.pmxh);
}

__global__ void RunABM_Predictor_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment for the next step
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//ABM predictor : pk+1 = mk + (dt/2) * (3*fk - fk-1)
				if (*cuaDiffEq.palternator) {

					M1[idx] += dT * (3 * rhs - (*cuaDiffEq.psEval0)[idx]) / 2;
					(*cuaDiffEq.psEval1)[idx] = rhs;
				}
				else {

					M1[idx] += dT * (3 * rhs - (*cuaDiffEq.psEval1)[idx]) / 2;
					(*cuaDiffEq.psEval0)[idx] = rhs;
				}
			}
		}
	}
}

__global__ void RunABM_Corrector_LLG_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuBReal dmdt = 0.0;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / M1.h.dim();

	cuBReal lte = 0.0;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//First save predicted moment for lte calculation
				cuReal3 saveM = M1[idx];

				//ABM corrector : mk+1 = mk + (dt/2) * (fk+1 + fk)
				if (*cuaDiffEq.palternator) {

					M1[idx] = (*cuaDiffEq.psM1)[idx] + dT * (rhs + (*cuaDiffEq.psEval1)[idx]) / 2;
				}
				else {

					M1[idx] = (*cuaDiffEq.psM1)[idx] + dT * (rhs + (*cuaDiffEq.psEval0)[idx]) / 2;
				}

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuaMesh.pM1)[idx].norm();
				dmdt = cu_GetMagnitude((*cuaMesh.pM1)[idx] - (*cuaDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * conversion * Mnorm * Mnorm);

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude(M1[idx] - saveM) / mu_s;
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_max(0, 1, &dmdt, *cuaDiffEq.pdmdt);
	reduction_max(0, 1, &lte, *cuaDiffEq.plte);
}

__global__ void RunABM_Corrector_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuBReal lte = 0.0;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//First save predicted moment for lte calculation
				cuReal3 saveM = M1[idx];

				//ABM corrector : mk+1 = mk + (dt/2) * (fk+1 + fk)
				if (*cuaDiffEq.palternator) {

					M1[idx] = (*cuaDiffEq.psM1)[idx] + dT * (rhs + (*cuaDiffEq.psEval1)[idx]) / 2;
				}
				else {

					M1[idx] = (*cuaDiffEq.psM1)[idx] + dT * (rhs + (*cuaDiffEq.psEval0)[idx]) / 2;
				}

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude(M1[idx] - saveM) / mu_s;
			}
		}
	}

	reduction_max(0, 1, &lte, *cuaDiffEq.plte);
}

__global__ void RunABMTEuler_Step0_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment for the next step
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				(*cuaDiffEq.psEval0)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//Now estimate moment for the next time step
				M1[idx] += (*cuaDiffEq.psEval0)[idx] * dT;
			}
		}
	}
}

__global__ void RunABMTEuler_Step1_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//Now estimate moment using the second trapezoidal Euler step equation
				M1[idx] = ((*cuaDiffEq.psM1)[idx] + M1[idx] + rhs * dT) / 2;

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}
			}
		}
	}
}

//----------------------------------------- EVALUATIONS : RKF45

__global__ void RunRKF45_Step0_LLG_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuBReal mxh = 0.0;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / M1.h.dim();

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment for later use
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				(*cuaDiffEq.psEval0)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//obtain maximum normalized torque term
				cuBReal Mnorm = M1[idx].norm();
				mxh = cu_GetMagnitude(M1[idx] ^ Heff1[idx]) / (conversion * Mnorm * Mnorm);

				//Now estimate moment using RKF first step
				M1[idx] += (*cuaDiffEq.psEval0)[idx] * (2 * dT / 9);
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_max(0, 1, &mxh, *cuaDiffEq.pmxh);
}

__global__ void RunRKF45_Step0_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			//Save current moment for later use
			(*cuaDiffEq.psM1)[idx] = M1[idx];

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				(*cuaDiffEq.psEval0)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//Now estimate moment using RKF first step
				M1[idx] += (*cuaDiffEq.psEval0)[idx] * (2 * dT / 9);
			}
		}
	}
}

__global__ void RunRKF45_Step1_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx) && !M1.is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
			(*cuaDiffEq.psEval1)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

			//Now estimate moment using RKF midle step 1
			M1[idx] = (*cuaDiffEq.psM1)[idx] + ((*cuaDiffEq.psEval0)[idx] / 12 + (*cuaDiffEq.psEval1)[idx] / 4) * dT;
		}
	}
}

__global__ void RunRKF45_Step2_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx) && !M1.is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
			(*cuaDiffEq.psEval2)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

			//Now estimate moment using RKF midle step 2
			M1[idx] = (*cuaDiffEq.psM1)[idx] + (69 * (*cuaDiffEq.psEval0)[idx] / 128 - 243 * (*cuaDiffEq.psEval1)[idx] / 128 + 135 * (*cuaDiffEq.psEval2)[idx] / 64) * dT;
		}
	}
}

__global__ void RunRKF45_Step3_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx) && !M1.is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
			(*cuaDiffEq.psEval3)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

			//Now estimate moment using RKF midle step 3
			M1[idx] = (*cuaDiffEq.psM1)[idx] + (-17 * (*cuaDiffEq.psEval0)[idx] / 12 + 27 * (*cuaDiffEq.psEval1)[idx] / 4 - 27 * (*cuaDiffEq.psEval2)[idx] / 5 + 16 * (*cuaDiffEq.psEval3)[idx] / 15) * dT;
		}
	}
}

__global__ void RunRKF45_Step4_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx) && !M1.is_skipcell(idx)) {

			//First evaluate RHS of set equation at the current time step
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
			(*cuaDiffEq.psEval4)[idx] = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

			//Now estimate moment using RKF midle step 4
			M1[idx] = (*cuaDiffEq.psM1)[idx] + (65 * (*cuaDiffEq.psEval0)[idx] / 432 - 5 * (*cuaDiffEq.psEval1)[idx] / 16 + 13 * (*cuaDiffEq.psEval2)[idx] / 16 + 4 * (*cuaDiffEq.psEval3)[idx] / 27 + 5 * (*cuaDiffEq.psEval4)[idx] / 144) * dT;
		}
	}
}

__global__ void RunRKF45_Step5_LLG_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuBReal dmdt = 0.0;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / M1.h.dim();

	cuBReal lte = 0.0;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//4th order evaluation
				M1[idx] = (*cuaDiffEq.psM1)[idx] + ((*cuaDiffEq.psEval0)[idx] / 9 + 9 * (*cuaDiffEq.psEval2)[idx] / 20 + 16 * (*cuaDiffEq.psEval3)[idx] / 45 + (*cuaDiffEq.psEval4)[idx] / 12) * dT;

				//5th order evaluation
				cuReal3 prediction = (*cuaDiffEq.psM1)[idx] + (47 * (*cuaDiffEq.psEval0)[idx] / 450 + 12 * (*cuaDiffEq.psEval2)[idx] / 25 + 32 * (*cuaDiffEq.psEval3)[idx] / 225 + 1 * (*cuaDiffEq.psEval4)[idx] / 30 + 6 * rhs / 25) * dT;

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuaMesh.pM1)[idx].norm();
				dmdt = cu_GetMagnitude((*cuaMesh.pM1)[idx] - (*cuaDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * conversion * Mnorm * Mnorm);

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude(M1[idx] - prediction) / mu_s;
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_max(0, 1, &dmdt, *cuaDiffEq.pdmdt);

	reduction_max(0, 1, &lte, *cuaDiffEq.plte);
}

__global__ void RunRKF45_Step5_LLG_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	cuBReal mu_s = *cuaMesh.pmu_s;
	cuBReal alpha = *cuaMesh.palpha;

	cuBReal lte = 0.0;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			if (!M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.palpha, alpha);
				cuReal3 rhs = (-(cuBReal)GAMMA / (1 + alpha * alpha)) * ((M1[idx] ^ Heff1[idx]) + alpha * ((M1[idx] / mu_s) ^ (M1[idx] ^ Heff1[idx])));

				//4th order evaluation
				M1[idx] = (*cuaDiffEq.psM1)[idx] + ((*cuaDiffEq.psEval0)[idx] / 9 + 9 * (*cuaDiffEq.psEval2)[idx] / 20 + 16 * (*cuaDiffEq.psEval3)[idx] / 45 + (*cuaDiffEq.psEval4)[idx] / 12) * dT;

				//5th order evaluation
				cuReal3 prediction = (*cuaDiffEq.psM1)[idx] + (47 * (*cuaDiffEq.psEval0)[idx] / 450 + 12 * (*cuaDiffEq.psEval2)[idx] / 25 + 32 * (*cuaDiffEq.psEval3)[idx] / 225 + 1 * (*cuaDiffEq.psEval4)[idx] / 30 + 6 * rhs / 25) * dT;

				if (*cuaDiffEq.prenormalize) {

					M1[idx].renormalize(mu_s);
				}

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude(M1[idx] - prediction) / mu_s;
			}
		}
	}

	reduction_max(0, 1, &lte, *cuaDiffEq.plte);
}

//----------------------------------------- DifferentialEquationCUDA Launchers

//EULER

void Atom_DifferentialEquationCubicCUDA::RunEuler_LLG(bool calculate_mxh, bool calculate_dmdt)
{
	if (calculate_mxh || calculate_dmdt) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunEuler_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunEuler_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}
	}
}

//TRAPEZOIDAL EULER

void Atom_DifferentialEquationCubicCUDA::RunTEuler_LLG(int step, bool calculate_mxh, bool calculate_dmdt)
{
	if (step == 0) {

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step0_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step0_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
	}
	else {

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step1_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step1_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
	}
}

//ADAPTIVE HEUN

void Atom_DifferentialEquationCubicCUDA::RunAHeun_LLG(int step, bool calculate_mxh, bool calculate_dmdt)
{
	if (step == 0) {

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step0_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunTEuler_Step0_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
	}
	else {

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunAHeun_Step1_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunAHeun_Step1_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
	}
}

//RUNGE KUTTA 4th order

void Atom_DifferentialEquationCubicCUDA::RunRK4_LLG(int step, bool calculate_mxh, bool calculate_dmdt)
{
	switch (step) {

	case 0:

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRK4_Step0_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRK4_Step0_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}

		break;

	case 1:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRK4_Step1_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}

		break;

	case 2:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRK4_Step2_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}

		break;

	case 3:

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRK4_Step3_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRK4_Step3_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}

		break;
	}
}

//Adams-Bashforth-Moulton 2nd order

void Atom_DifferentialEquationCubicCUDA::RunABM_LLG(int step, bool calculate_mxh, bool calculate_dmdt)
{
	if (step == 0) {

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Predictor_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Predictor_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
	}
	else {

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Corrector_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Corrector_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
	}
}

//Adams-Bashforth-Moulton 2nd order priming using Trapezoidal Euler

void Atom_DifferentialEquationCubicCUDA::RunABMTEuler_LLG(int step)
{
	if (step == 0) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunABMTEuler_Step0_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunABMTEuler_Step1_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}
	}
}

//RUNGE KUTTA FEHLBERG

void Atom_DifferentialEquationCubicCUDA::RunRKF45_LLG(int step, bool calculate_mxh, bool calculate_dmdt)
{
	switch (step) {

	case 0:

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF45_Step0_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF45_Step0_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}

		break;

	case 1:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step1_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}

		break;

	case 2:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step2_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}

		break;

	case 3:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step3_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}

		break;

	case 4:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunRKF45_Step4_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}

		break;

	case 5:

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF45_Step5_LLG_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunRKF45_Step5_LLG_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}

		break;
	}
}

#endif
#endif