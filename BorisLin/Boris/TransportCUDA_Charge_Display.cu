#include "TransportCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "mcuVEC_halo.cuh"

#include "MeshCUDA.h"
#include "SuperMeshCUDA.h"
#include "MeshParamsControlCUDA.h"

//-------------------Display Calculation Methods

//--------------------------------------------------------------- Current Density

__global__ void CalculateFixedCurrentDensity_Charge_Kernel(cuVEC_VC<cuReal3>& Jc, cuVEC_VC<cuReal3>& E, cuVEC_VC<cuBReal>& elC)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Jc.linear_size()) {

		//only calculate current on non-empty cells - empty cells have already been assigned 0 at UpdateConfiguration
		if (elC.is_not_empty(idx)) {

			Jc[idx] = elC[idx] * E[idx];
		}
		else Jc[idx] = cuReal3(0.0);
	}
}

__global__ void CalculateCurrentDensity_Spin_Kernel(cuVEC_VC<cuReal3>& Jc, ManagedMeshCUDA& cuMesh, TransportCUDA_Spin_V_Funcs& poisson_Spin_V, cuVEC_VC<cuReal3>& dM_dt)
{
	cuVEC<cuReal3>& E = *cuMesh.pE;
	cuVEC_VC<cuBReal>& V = *cuMesh.pV;
	cuVEC_VC<cuBReal>& elC = *cuMesh.pelC;
	cuVEC_VC<cuReal3>& S = *cuMesh.pS;
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Jc.linear_size()) {

		//only calculate current on non-empty cells - empty cells have already been assigned 0 at UpdateConfiguration
		if (V.is_not_empty(idx)) {

			bool cppgmr_enabled = cuIsNZ(cuMesh.pbetaD->get0());
			bool cpump_enabled = cuIsNZ(cuMesh.pcpump_eff->get0());
			bool the_enabled = cuIsNZ(cuMesh.pthe_eff->get0());

			if (poisson_Spin_V.stsolve == STSOLVE_NORMALMETAL) {

				//non-magnetic mesh

				if (cuIsZ(cuMesh.piSHA->get0())) {

					//no iSHE contribution.
					Jc[idx] = -elC[idx] * V.grad_diri(idx);
				}
				else {

					cuBReal SHA = *cuMesh.pSHA;
					cuBReal iSHA = *cuMesh.piSHA;
					cuBReal De = *cuMesh.pDe;
					cuMesh.update_parameters_ecoarse(idx, *cuMesh.pSHA, SHA, *cuMesh.piSHA, iSHA, *cuMesh.pDe, De);

					//iSHE enabled, must use non-homogeneous Neumann boundary condition for grad V -> Note homogeneous Neumann boundary conditions apply when calculating S differentials here (due to Jc.n = 0 at boundaries)
					Jc[idx] = -elC[idx] * V.grad_diri_nneu(idx, (iSHA * De / ((cuBReal)MUB_E * elC[idx])) * S.curl_neu(idx));

					//must also add iSHE contribution -> here we must use non-homogeneous Neumann boundary conditions when calculating S differentials
					Jc[idx] += (iSHA * De / (cuBReal)MUB_E) * S.curl_nneu(idx, cu_epsilon3(E[idx]) * (SHA * elC[idx] * (cuBReal)MUB_E / De));
				}
			}
			else {

				//magnetic mesh

				cuReal3 grad_V = V.grad_diri(idx);

				//1. principal term : always present
				Jc[idx] = -elC[idx] * grad_V;

				//additional contributions if enabled
				if (cppgmr_enabled || cpump_enabled || the_enabled) {

					int idx_M = M.position_to_cellidx(S.cellidx_to_position(idx));

					cuReal3 m = cu_normalize(M[idx_M]);
					cuReal33 grad_S = S.grad_neu(idx);		//homogeneous Neumann since SHA = 0 in magnetic meshes

					//2. CPP-GMR contribution
					if (cppgmr_enabled) {

						cuBReal betaD = *cuMesh.pbetaD;
						cuBReal De = *cuMesh.pDe;
						cuMesh.update_parameters_ecoarse(idx, *cuMesh.pbetaD, betaD, *cuMesh.pDe, De);

						Jc[idx] += (grad_S * m) * betaD * De / (cuBReal)MUB_E;
					}

					//3. topological Hall effect contribution
					//4. charge pumping contribution
					if (cpump_enabled || the_enabled) {

						cuBReal P = *cuMesh.pP;
						cuBReal n_density = *cuMesh.pn_density;
						cuMesh.update_parameters_ecoarse(idx, *cuMesh.pP, P, *cuMesh.pn_density, n_density);

						cuReal33 grad_M = M.grad_neu(idx_M);
						cuReal3 dx_m = cu_normalize(grad_M.x, M[idx_M]);
						cuReal3 dy_m = cu_normalize(grad_M.y, M[idx_M]);

						//topological Hall effect contribution
						if (the_enabled) {

							cuBReal Bz_the = (dx_m ^ dy_m) * m;
							Jc[idx] += cuMesh.pthe_eff->get0() * (P * elC[idx] * (cuBReal)HBAR_E / ((cuBReal)ECHARGE * n_density)) * elC[idx] * cuReal3(grad_V.y * Bz_the, -grad_V.x *Bz_the, 0.0);
						}

						//charge pumping contribution
						if (cpump_enabled) {

							cuReal3 dm_dt = cu_normalize(dM_dt[idx_M], M[idx_M]);
							Jc[idx] += cuMesh.pcpump_eff->get0() * (P * elC[idx] * (cuBReal)HBAR_E / 2) * cuReal3((dm_dt ^ dx_m) * m, (dm_dt ^ dy_m) * m, 0.0);
						}
					}
				}
			}
		}
		else Jc[idx] = cuReal3(0);
	}
}

//-------------------Calculation Methods : Charge Current Density

//calculate charge current density over the mesh
mcu_VEC_VC(cuReal3)& TransportCUDA::GetChargeCurrent(void)
{
	if (!PrepareDisplayVEC_VC(pMeshCUDA->h_e)) return displayVEC_VC;

	if (!pSMeshCUDA->DisabledTransportSolver()) {

		if (stsolve == STSOLVE_NONE) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				CalculateFixedCurrentDensity_Charge_Kernel <<< (pMeshCUDA->elC.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(displayVEC_VC.get_deviceobject(mGPU), pMeshCUDA->E.get_deviceobject(mGPU), pMeshCUDA->elC.get_deviceobject(mGPU));
			}
		}
		else {
			
			pMeshCUDA->V.exchange_halos();
			pMeshCUDA->S.exchange_halos();
			pMeshCUDA->M.exchange_halos();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				CalculateCurrentDensity_Spin_Kernel <<< (pMeshCUDA->elC.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(displayVEC_VC.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU), poisson_Spin_V.get_deviceobject(mGPU), dM_dt.get_deviceobject(mGPU));
			}
		}
	}
	else {

		//if transport solver disabled we need to set displayVEC_VC directly from E and elC as Jc = elC * E
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			CalculateFixedCurrentDensity_Charge_Kernel <<< (pMeshCUDA->elC.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(displayVEC_VC.get_deviceobject(mGPU), pMeshCUDA->E.get_deviceobject(mGPU), pMeshCUDA->elC.get_deviceobject(mGPU));
		}
	}

	return displayVEC_VC;
}

#endif

#endif