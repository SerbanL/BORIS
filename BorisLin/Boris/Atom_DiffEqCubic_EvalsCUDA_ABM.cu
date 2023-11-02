#include "Atom_DiffEqCubicCUDA.h"

#if COMPILECUDA == 1
#ifdef ODE_EVAL_COMPILATION_ABM
#ifdef MESH_COMPILATION_ATOM_CUBIC

#include "Atom_MeshParamsControlCUDA.h"

#include "Reduction.cuh"

//defines evaluation methods kernel launchers

//----------------------------------------- EVALUATIONS : ABM

__global__ void RunABM_Predictor_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuBReal mxh = 0.0;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / cuaMesh.pM1->h.dim();

	if (idx < cuaMesh.pM1->linear_size()) {

		if (cuaMesh.pM1->is_not_empty(idx)) {

			//Save current moment for the next step
			(*cuaDiffEq.psM1)[idx] = (*cuaMesh.pM1)[idx];

			if (!cuaMesh.pM1->is_skipcell(idx)) {

				//obtain maximum normalized torque term
				cuBReal Mnorm = (*cuaMesh.pM1)[idx].norm();
				mxh = cu_GetMagnitude((*cuaMesh.pM1)[idx] ^ (*cuaMesh.pHeff1)[idx]) / (conversion * Mnorm * Mnorm);

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs = (cuaDiffEq.*(cuaDiffEq.pODEFunc))(idx);

				//ABM predictor : pk+1 = mk + (dt/2) * (3*fk - fk-1)
				if (*cuaDiffEq.palternator) {

					(*cuaMesh.pM1)[idx] += dT * (3 * rhs - (*cuaDiffEq.psEval0)[idx]) / 2;
					(*cuaDiffEq.psEval1)[idx] = rhs;
				}
				else {

					(*cuaMesh.pM1)[idx] += dT * (3 * rhs - (*cuaDiffEq.psEval1)[idx]) / 2;
					(*cuaDiffEq.psEval0)[idx] = rhs;
				}
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_max(0, 1, &mxh, *cuaDiffEq.pmxh);
}

__global__ void RunABM_Predictor_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	if (idx < cuaMesh.pM1->linear_size()) {

		if (cuaMesh.pM1->is_not_empty(idx)) {

			//Save current moment for the next step
			(*cuaDiffEq.psM1)[idx] = (*cuaMesh.pM1)[idx];

			if (!cuaMesh.pM1->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs = (cuaDiffEq.*(cuaDiffEq.pODEFunc))(idx);

				//ABM predictor : pk+1 = mk + (dt/2) * (3*fk - fk-1)
				if (*cuaDiffEq.palternator) {

					(*cuaMesh.pM1)[idx] += dT * (3 * rhs - (*cuaDiffEq.psEval0)[idx]) / 2;
					(*cuaDiffEq.psEval1)[idx] = rhs;
				}
				else {

					(*cuaMesh.pM1)[idx] += dT * (3 * rhs - (*cuaDiffEq.psEval1)[idx]) / 2;
					(*cuaDiffEq.psEval0)[idx] = rhs;
				}
			}
		}
	}
}

__global__ void RunABM_Corrector_withReductions_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuBReal dmdt = 0.0;
	cuBReal lte = 0.0;

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	cuBReal conversion = (cuBReal)MUB / cuaMesh.pM1->h.dim();

	if (idx < cuaMesh.pM1->linear_size()) {

		if (cuaMesh.pM1->is_not_empty(idx)) {

			if (!cuaMesh.pM1->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs = (cuaDiffEq.*(cuaDiffEq.pODEFunc))(idx);

				//First save predicted moment for lte calculation
				cuReal3 saveM = (*cuaMesh.pM1)[idx];

				//ABM corrector : mk+1 = mk + (dt/2) * (fk+1 + fk)
				if (*cuaDiffEq.palternator) {

					(*cuaMesh.pM1)[idx] = (*cuaDiffEq.psM1)[idx] + dT * (rhs + (*cuaDiffEq.psEval1)[idx]) / 2;
				}
				else {

					(*cuaMesh.pM1)[idx] = (*cuaDiffEq.psM1)[idx] + dT * (rhs + (*cuaDiffEq.psEval0)[idx]) / 2;
				}

				if (*cuaDiffEq.prenormalize) {

					cuBReal mu_s = *cuaMesh.pmu_s;
					cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s);
					(*cuaMesh.pM1)[idx].renormalize(mu_s);
				}

				//obtain maximum normalized dmdt term
				cuBReal Mnorm = (*cuaMesh.pM1)[idx].norm();
				dmdt = cu_GetMagnitude((*cuaMesh.pM1)[idx] - (*cuaDiffEq.psM1)[idx]) / (dT * (cuBReal)GAMMA * conversion * Mnorm * Mnorm);

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuaMesh.pM1)[idx] - saveM) / (*cuaMesh.pM1)[idx].norm();
			}
		}
	}

	if (cuaMesh.pgrel->get0()) reduction_max(0, 1, &dmdt, *cuaDiffEq.pdmdt);
	reduction_max(0, 1, &lte, *cuaDiffEq.plte);
}

__global__ void RunABM_Corrector_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	cuBReal lte = 0.0;

	if (idx < cuaMesh.pM1->linear_size()) {

		if (cuaMesh.pM1->is_not_empty(idx)) {

			if (!cuaMesh.pM1->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs = (cuaDiffEq.*(cuaDiffEq.pODEFunc))(idx);

				//First save predicted moment for lte calculation
				cuReal3 saveM = (*cuaMesh.pM1)[idx];

				//ABM corrector : mk+1 = mk + (dt/2) * (fk+1 + fk)
				if (*cuaDiffEq.palternator) {

					(*cuaMesh.pM1)[idx] = (*cuaDiffEq.psM1)[idx] + dT * (rhs + (*cuaDiffEq.psEval1)[idx]) / 2;
				}
				else {

					(*cuaMesh.pM1)[idx] = (*cuaDiffEq.psM1)[idx] + dT * (rhs + (*cuaDiffEq.psEval0)[idx]) / 2;
				}

				if (*cuaDiffEq.prenormalize) {

					cuBReal mu_s = *cuaMesh.pmu_s;
					cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s);
					(*cuaMesh.pM1)[idx].renormalize(mu_s);
				}

				//local truncation error (between predicted and corrected)
				lte = cu_GetMagnitude((*cuaMesh.pM1)[idx] - saveM) / (*cuaMesh.pM1)[idx].norm();
			}
		}
	}

	reduction_max(0, 1, &lte, *cuaDiffEq.plte);
}

__global__ void RunABMTEuler_Step0_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	if (idx < cuaMesh.pM1->linear_size()) {

		if (cuaMesh.pM1->is_not_empty(idx)) {

			//Save current moment for the next step
			(*cuaDiffEq.psM1)[idx] = (*cuaMesh.pM1)[idx];

			if (!cuaMesh.pM1->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				(*cuaDiffEq.psEval0)[idx] = (cuaDiffEq.*(cuaDiffEq.pODEFunc))(idx);

				//Now estimate moment for the next time step
				(*cuaMesh.pM1)[idx] += (*cuaDiffEq.psEval0)[idx] * dT;
			}
		}
	}
}

__global__ void RunABMTEuler_Step1_Kernel(ManagedAtom_DiffEqCubicCUDA& cuaDiffEq, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal dT = *cuaDiffEq.pdT;

	if (idx < cuaMesh.pM1->linear_size()) {

		if (cuaMesh.pM1->is_not_empty(idx)) {

			if (!cuaMesh.pM1->is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				cuReal3 rhs = (cuaDiffEq.*(cuaDiffEq.pODEFunc))(idx);

				//Now estimate moment using the second trapezoidal Euler step equation
				(*cuaMesh.pM1)[idx] = ((*cuaDiffEq.psM1)[idx] + (*cuaMesh.pM1)[idx] + rhs * dT) / 2;

				if (*cuaDiffEq.prenormalize) {

					cuBReal mu_s = *cuaMesh.pmu_s;
					cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s);
					(*cuaMesh.pM1)[idx].renormalize(mu_s);
				}
			}
		}
	}
}

//----------------------------------------- DifferentialEquationCUDA Launchers

//Adams-Bashforth-Moulton 2nd order

void Atom_DifferentialEquationCubicCUDA::RunABM(int step, bool calculate_mxh, bool calculate_dmdt)
{
	if (step == 0) {

		if (calculate_mxh) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Predictor_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Predictor_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
	}
	else {

		if (calculate_dmdt) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Corrector_withReductions_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				RunABM_Corrector_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
			}
		}
	}
}

//Adams-Bashforth-Moulton 2nd order priming using Trapezoidal Euler

void Atom_DifferentialEquationCubicCUDA::RunABMTEuler(int step)
{
	if (step == 0) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunABMTEuler_Step0_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			RunABMTEuler_Step1_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (cuaDiffEq.get_deviceobject(mGPU), paMeshCUDA->cuaMesh.get_deviceobject(mGPU));
		}
	}
}

#endif
#endif
#endif