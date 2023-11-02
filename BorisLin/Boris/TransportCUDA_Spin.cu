#include "TransportCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "mcuVEC_solve.cuh"
#include "mcuVEC_halo.cuh"

#include "MeshCUDA.h"
#include "SuperMeshCUDA.h"
#include "MeshParamsControlCUDA.h"

#include "ManagedDiffEqPolicyFMCUDA.h"

//-------------------Calculation Methods : Iterate Spin-Charge Solver

void TransportCUDA::IterateSpinSolver_Charge_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value, bool use_NNeu)
{
	pMeshCUDA->elC.exchange_halos();
	pMeshCUDA->S.exchange_halos();
	pMeshCUDA->M.exchange_halos();

	if (!use_NNeu || stsolve == STSOLVE_FERROMAGNETIC || stsolve == STSOLVE_NONE) {

		//no iSHE contribution. Note, iSHE is not included in magnetic meshes.
		pMeshCUDA->V.IteratePoisson_SOR(poisson_Spin_V, damping, max_error, max_value);
	}
	else {

		//iSHE enabled, must use non-homogeneous Neumann boundary condition for grad V -> Note homogeneous Neumann boundary conditions apply when calculating S differentials here (due to Jc.n = 0 at boundaries)
		pMeshCUDA->V.IteratePoisson_NNeu_SOR(poisson_Spin_V, damping, max_error, max_value);
	}
}

//------------------- PRIME SPIN-CHARGE SOLVER

__global__ void Get_dM_dt_Kernel(cuVEC_VC<cuReal3>& dM_dt, ManagedDiffEqFMCUDA& cuDiffEq)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < dM_dt.linear_size()) {

		if (dM_dt.is_not_empty(idx)) {

			dM_dt[idx] = cuDiffEq.dMdt(idx);
		}
	}
}

__global__ void PrimeSpinSolver_Charge_Kernel(ManagedMeshCUDA& cuMesh, TransportCUDA_Spin_V_Funcs& poisson_Spin_V, cuVEC_VC<cuReal3>& dM_dt)
{
	cuVEC_VC<cuBReal>& V = *cuMesh.pV;
	cuVEC_VC<cuBReal>& elC = *cuMesh.pelC;
	cuVEC_VC<cuReal3>& S = *cuMesh.pS;
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;

	cuVEC<cuBReal>& delsq_V_fixed = *poisson_Spin_V.pdelsq_V_fixed;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < delsq_V_fixed.linear_size()) {

		delsq_V_fixed[idx] = 0.0;

		//only calculate current on non-empty cells - empty cells have already been assigned 0 at UpdateConfiguration
		if (V.is_not_empty(idx)) {

			bool cppgmr_enabled = cuIsNZ(cuMesh.pbetaD->get0());
			bool cpump_enabled = cuIsNZ(cuMesh.pcpump_eff->get0());

			if (cppgmr_enabled || cpump_enabled) {

				int idx_M = M.position_to_cellidx(V.cellidx_to_position(idx));
				cuReal33 grad_m = cu_normalize(M.grad_neu(idx_M), M[idx_M]);
				cuReal3 m = cu_normalize(M[idx_M]);

				//CPP-GMR contribution
				if (cppgmr_enabled) {

					cuBReal De = *cuMesh.pDe;
					cuBReal betaD = *cuMesh.pbetaD;
					cuMesh.update_parameters_ecoarse(idx, *cuMesh.pDe, De, *cuMesh.pbetaD, betaD);

					cuReal33 grad_S = S.grad_neu(idx);
					cuReal3 delsq_S = S.delsq_neu(idx);
					cuBReal div_grad_S_m = (grad_S.i * grad_m.i) + (grad_S.j * grad_m.j) + (grad_S.k * grad_m.k) + (m * delsq_S);

					delsq_V_fixed[idx] += div_grad_S_m * betaD * De / ((cuBReal)MUB_E * elC[idx]);
				}

				//Charge pumping pre-calculation
				if (cpump_enabled) {

					cuReal33 grad_dm_dt = cu_normalize(dM_dt.grad_neu(idx_M), M[idx_M]);
					cuReal3 dm_dt = cu_normalize(dM_dt[idx_M], M[idx_M]);

					cuBReal P = *cuMesh.pP;
					cuMesh.update_parameters_ecoarse(idx, *cuMesh.pP, P);

					cuReal3 dx_m = grad_m.x;
					cuReal3 dy_m = grad_m.y;
					cuReal3 dxx_m = cu_normalize(M.dxx_neu(idx_M), M[idx_M]);
					cuReal3 dyy_m = cu_normalize(M.dyy_neu(idx_M), M[idx_M]);

					delsq_V_fixed[idx] += (cuMesh.pcpump_eff->get0() * P * (cuBReal)HBAR_E / 2) * ((grad_dm_dt.x ^ dx_m) + (grad_dm_dt.y ^ dy_m) + (dm_dt ^ (dxx_m + dyy_m))) * m;
				}
			}
		}
	}
}

//before iterating the spin solver (charge part) we need to prime it : pre-compute values which do not change as the spin solver relaxes.
void TransportCUDA::PrimeSpinSolver_Charge(void)
{
	//Update dM_dt values if needed
	if (Need_dM_dt_Calculation()) {

		//dM_dt diff ops not needed in Get_dM_dt_Kernel, but will be needed in PrimeSpinSolver_Charge_Kernel
		dM_dt.exchange_halos();

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Get_dM_dt_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(dM_dt.get_deviceobject(mGPU), pMeshCUDA->Get_ManagedDiffEqFMCUDA().get_deviceobject(mGPU));
		}
	}

	//the rest are terms to calculate in delsq_V_fixed
	if (Need_delsq_V_fixed_Precalculation()) {

		pMeshCUDA->S.exchange_halos();
		pMeshCUDA->M.exchange_halos();

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			PrimeSpinSolver_Charge_Kernel <<< (pMeshCUDA->elC.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU), poisson_Spin_V.get_deviceobject(mGPU), dM_dt.get_deviceobject(mGPU));
		}
	}
}

//-------------------Calculation Methods : Iterate Spin-Spin Solver

//solve for spin accumulation using Poisson equation for delsq_S, solved using SOR algorithm
void TransportCUDA::IterateSpinSolver_Spin_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value, bool use_NNeu)
{
	//no halo exchanges needed
	if (!use_NNeu || stsolve == STSOLVE_FERROMAGNETIC) {

		//no SHE contribution. Note, SHE is not included in magnetic meshes.
		pMeshCUDA->S.IteratePoisson_SOR(poisson_Spin_S, damping, max_error, max_value);
	}
	else {

		//SHE enabled, must use non-homogeneous Neumann boundary condition for grad S
		pMeshCUDA->S.IteratePoisson_NNeu_SOR(poisson_Spin_S, damping, max_error, max_value);
	}
}

//------------------- PRIME SPIN-SPIN SOLVER

__global__ void PrimeSpinSolver_Spin_Kernel(ManagedMeshCUDA& cuMesh, TransportCUDA_Spin_S_Funcs& poisson_Spin_S, cuVEC_VC<cuReal3>& dM_dt)
{
	cuVEC_VC<cuBReal>& V = *cuMesh.pV;
	cuVEC_VC<cuBReal>& elC = *cuMesh.pelC;
	cuVEC_VC<cuReal3>& E = *cuMesh.pE;
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;

	cuVEC<cuReal3>& delsq_S_fixed = *poisson_Spin_S.pdelsq_S_fixed;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < delsq_S_fixed.linear_size()) {

		delsq_S_fixed[idx] = 0.0;

		//only calculate current on non-empty cells - empty cells have already been assigned 0 at UpdateConfiguration
		if (V.is_not_empty(idx)) {

			bool cpump_enabled = cuIsNZ(cuMesh.pcpump_eff->get0());
			bool the_enabled = cuIsNZ(cuMesh.pthe_eff->get0());
			bool she_enabled = cuIsNZ(cuMesh.pSHA->get0());

			if (poisson_Spin_S.stsolve == STSOLVE_FERROMAGNETIC) {

				//magnetic mesh

				cuBReal P = *cuMesh.pP;
				cuBReal De = *cuMesh.pDe;
				cuMesh.update_parameters_ecoarse(idx, *cuMesh.pP, P, *cuMesh.pDe, De);

				//term due to drift (non-uniformity of M term, and delsq V contribution - non-uniformity of E term)

				//find grad M and M at the M cell in which the current S cell center is
				int idx_M = M.position_to_cellidx(V.cellidx_to_position(idx));

				cuReal3 m = cu_normalize(M[idx_M]);
				cuReal33 grad_m = cu_normalize(M.grad_neu(idx_M), M[idx_M]);
				cuReal3 E_dot_del_m = grad_m | E[idx];

				//E_dot_del_m term is very important, but Evaluate_SpinSolver_delsqV_RHS term could be neglected in most cases especially if E is uniform.
				delsq_S_fixed[idx] += (P * (cuBReal)MUB_E * elC[idx] / De) * (poisson_Spin_S.pPoisson_Spin_V->Poisson_RHS(idx) * m - E_dot_del_m);

				//charge pumping and topological Hall effect
				if (cpump_enabled || the_enabled) {

					cuReal3 dx_m = grad_m.x;
					cuReal3 dy_m = grad_m.y;
					cuReal3 dxy_m = cu_normalize(M.dxy_neu(idx_M), M[idx_M]);
					cuReal3 dxx_m = cu_normalize(M.dxx_neu(idx_M), M[idx_M]);
					cuReal3 dyy_m = cu_normalize(M.dyy_neu(idx_M), M[idx_M]);

					if (cpump_enabled) {

						cuReal3 dmdt = cu_normalize(dM_dt[idx_M], M[idx_M]);
						cuReal33 grad_dm_dt = cu_normalize(dM_dt.grad_neu(idx_M), M[idx_M]);

						delsq_S_fixed[idx] += cuMesh.pcpump_eff->get0() * (elC[idx] * (cuBReal)HBAR_E * (cuBReal)MUB_E / (2 * De)) * ((grad_dm_dt.x ^ dx_m) + (grad_dm_dt.y ^ dy_m) + (dmdt ^ (dxx_m + dyy_m)));
					}

					if (the_enabled) {

						cuBReal n_density = *cuMesh.pn_density;
						cuMesh.update_parameters_ecoarse(idx, *cuMesh.pn_density, n_density);

						delsq_S_fixed[idx] += cuMesh.pthe_eff->get0() * ((cuBReal)HBAR_E * (cuBReal)MUB_E * elC[idx] * elC[idx] / ((cuBReal)ECHARGE * n_density * De)) * (E[idx].x * ((dxy_m ^ dy_m) + (dx_m ^ dyy_m)) - E[idx].y * ((dxx_m ^ dy_m) + (dx_m ^ dxy_m)));
					}
				}
			}

			//terms occuring only in non-magnetic meshes
			else {

				//1. SHA term (this is negligible in most cases, even if E is non-uniform, but might as well include it) 
				if (she_enabled) {

					cuBReal SHA = *cuMesh.pSHA;
					cuBReal De = *cuMesh.pDe;
					cuMesh.update_parameters_ecoarse(idx, *cuMesh.pSHA, SHA, *cuMesh.pDe, De);

					//Check boundary conditions for this term : should be Dirichlet with 0 Jc value normal to the boundary except for electrodes.
					delsq_S_fixed[idx] += (SHA * elC[idx] * (cuBReal)MUB_E / De) * E.diveps3_sided(idx);
				}
			}
		}
	}
}

//before iterating the spin solver (spin part) we need to prime it : pre-compute values which do not change as the spin solver relaxes.
void TransportCUDA::PrimeSpinSolver_Spin(void)
{
	if (Need_delsq_S_fixed_Precalculation()) {

		pMeshCUDA->V.exchange_halos();
		pMeshCUDA->S.exchange_halos();
		pMeshCUDA->M.exchange_halos();
		pMeshCUDA->elC.exchange_halos();
		pMeshCUDA->E.exchange_halos();
		dM_dt.exchange_halos();

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			PrimeSpinSolver_Spin_Kernel <<< (pMeshCUDA->elC.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU), poisson_Spin_S.get_deviceobject(mGPU), dM_dt.get_deviceobject(mGPU));
		}
	}
}

//--------------------------------------------------------------- Effective field from spin accumulation

__global__ void CalculateSAField_Kernel(ManagedMeshCUDA& cuMesh)
{
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& S = *cuMesh.pS;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < M.linear_size()) {

		if (M.is_not_empty(idx)) {

			cuBReal De = *cuMesh.pDe;
			cuBReal ts_eff = *cuMesh.pts_eff;
			cuBReal grel = *cuMesh.pgrel;
			cuBReal l_ex = *cuMesh.pl_ex;
			cuBReal l_ph = *cuMesh.pl_ph;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pgrel, grel, *cuMesh.pDe, De, *cuMesh.pts_eff, ts_eff, *cuMesh.pl_ex, l_ex, *cuMesh.pl_ph, l_ph);
			
			cuBReal Mnorm = M[idx].norm();
			if (cuIsNZ((cuBReal)grel) && Mnorm > 0.0) {

				cuReal3 Sa = S.weighted_average(M.cellidx_to_position(idx), M.h);

				Heff[idx] += (De * ts_eff / ((cuBReal)GAMMA * grel * Mnorm)) * (Sa / (l_ex * l_ex) + (M[idx] ^ Sa) / (l_ph * l_ph * Mnorm));
			}
		}
	}
}

//Spin accumulation field
void TransportCUDA::CalculateSAField(void)
{
	if (stsolve == STSOLVE_FERROMAGNETIC) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			CalculateSAField_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}
	}
}

//--------------------------------------------------------------- Effective field from interface spin accumulation drop

__global__ void CalculateSAInterfaceField_Kernel(CMBNDInfoCUDA& contact, TransportCUDA_Spin_S_CMBND_Sec& cmbndFuncs_sec, TransportCUDA_Spin_S_CMBND_Pri& cmbndFuncs_pri)
{
	cuVEC<cuReal3>& Heff = *cmbndFuncs_pri.pcuMesh->pHeff;
	cuVEC_VC<cuReal3>& M = *cmbndFuncs_pri.pcuMesh->pM;
	cuVEC_VC<cuReal3>& S_pri = *cmbndFuncs_pri.pcuMesh->pS;
	//access S on first device, which contains origin of entire mcuVEC
	cuVEC_VC<cuReal3>& S0_sec = *cmbndFuncs_sec.ppcuMesh[0]->pS;

	int box_idx = blockIdx.x * blockDim.x + threadIdx.x;

	//interface conductance method with F being the primary mesh : calculate and set spin torque

	//convert the cells box from S mesh to M mesh
	cuINT3 mbox_start = M.cellidx_from_position(S_pri.cellidx_to_position(contact.cells_box.s) + M.rect.s);
	cuINT3 mbox_end = M.cellidx_from_position(S_pri.cellidx_to_position(contact.cells_box.e - cuINT3(1)) + M.rect.s);

	if ((mbox_end.i - mbox_start.i) == 0) mbox_end.i = mbox_start.i + 1;
	if ((mbox_end.j - mbox_start.j) == 0) mbox_end.j = mbox_start.j + 1;
	if ((mbox_end.k - mbox_start.k) == 0) mbox_end.k = mbox_start.k + 1;

	cuINT3 box_sizes = mbox_end - mbox_start;

	if (box_idx < box_sizes.dim()) {

		//the cellsize perpendicular to the contact (in the M mesh)
		cuBReal dh = (cuReal3(contact.cell_shift) & M.h).norm();

		int i = (box_idx % box_sizes.x) + mbox_start.i;
		int j = ((box_idx / box_sizes.x) % box_sizes.y) + mbox_start.j;
		int k = (box_idx / (box_sizes.x * box_sizes.y)) + mbox_start.k;

		//index of magnetic cell 1
		int mcell1_idx = i + j * M.n.x + k * M.n.x*M.n.y;

		if (M.is_empty(mcell1_idx)) return;

		cuBReal grel = *cmbndFuncs_pri.pcuMesh->pgrel;
		cuBReal tsi_eff = *cmbndFuncs_pri.pcuMesh->ptsi_eff;
		cmbndFuncs_pri.pcuMesh->update_parameters_mcoarse(mcell1_idx, *cmbndFuncs_pri.pcuMesh->pgrel, grel, *cmbndFuncs_pri.pcuMesh->ptsi_eff, tsi_eff);

		if (cuIsNZ((cuBReal)grel)) {

			//position at interface relative to primary mesh
			cuReal3 mhshift_primary = contact.hshift_primary.normalized() & M.h;
			cuReal3 relpos_interf = ((cuReal3(i, j, k) + cuReal3(0.5)) & M.h) + mhshift_primary / 2;

			cuReal3 relpos_1 = relpos_interf - contact.hshift_primary / 2;

			//relpos_m1 is relative to entire mcuVEC (sec)
			cuReal3 relpos_m1 = S_pri.rect.s - S0_sec.rect.s + relpos_interf + contact.hshift_secondary / 2;
			//get device and device-relative position on secondary side
			int device = 0;
			cuReal3 devrelpos_m1 = cmbndFuncs_sec.global_relpos_to_device_relpos(relpos_m1, device);
			cuVEC_VC<cuReal3>& S_sec = *cmbndFuncs_sec.ppcuMesh[device]->pS;

			cuReal3 stencil_pri = M.h - cu_mod(mhshift_primary) + cu_mod(contact.hshift_primary);
			cuReal3 stencil_sec = M.h - cu_mod(mhshift_primary) + cu_mod(contact.hshift_secondary);

			//S values
			cuReal3 S_1 = S_pri.weighted_average(relpos_1, stencil_pri);
			cuReal3 S_2 = S_pri.weighted_average(relpos_1 - contact.hshift_primary, stencil_pri);
			cuReal3 S_m1 = S_sec.weighted_average(devrelpos_m1, stencil_sec);
			cuReal3 S_m2 = S_sec.weighted_average(devrelpos_m1 + contact.hshift_secondary, stencil_sec);

			//c values
			cuBReal c_1 = cmbndFuncs_pri.c_func_sec(relpos_1, stencil_pri);
			cuBReal c_2 = cmbndFuncs_pri.c_func_sec(relpos_1 - contact.hshift_primary, stencil_pri);
			cuBReal c_m1 = cmbndFuncs_sec.c_func_sec(relpos_m1, stencil_sec);
			cuBReal c_m2 = cmbndFuncs_sec.c_func_sec(relpos_m1 + contact.hshift_secondary, stencil_sec);

			//Calculate S drop at the interface
			cuReal3 Vs_F = 1.5 * c_1 * S_1 - 0.5 * c_2 * S_2;
			cuReal3 Vs_N = 1.5 * c_m1 * S_m1 - 0.5 * c_m2 * S_m2;
			cuReal3 dVs = Vs_F - Vs_N;

			//Get G values from top contacting mesh
			cuReal2 Gmix;
			if (contact.IsPrimaryTop()) {

				Gmix = *cmbndFuncs_pri.pcuMesh->pGmix;
				cmbndFuncs_pri.pcuMesh->update_parameters_mcoarse(mcell1_idx, *cmbndFuncs_pri.pcuMesh->pGmix, Gmix);
			}
			else {

				Gmix = *cmbndFuncs_sec.ppcuMesh[device]->pGmix;
				cmbndFuncs_sec.ppcuMesh[device]->update_parameters_atposition(relpos_m1, *cmbndFuncs_sec.ppcuMesh[device]->pGmix, Gmix);
			}

			cuBReal Mnorm = M[mcell1_idx].norm();
			if (Mnorm > 0.0) {

				cuBReal gI = (2.0 * (cuBReal)GMUB_2E / dh) * cuReal2(Gmix).j / (-(cuBReal)GAMMA * grel * Mnorm);
				cuBReal gR = (2.0 * (cuBReal)GMUB_2E / dh) * cuReal2(Gmix).i / (-(cuBReal)GAMMA * grel * Mnorm);

				Heff[mcell1_idx] += tsi_eff * (gI * dVs + gR * (M[mcell1_idx] ^ dVs) / Mnorm);
			}
		}
	}
}

//Calculate the field resulting from interface spin accumulation torque for a given contact (in magnetic meshes for NF interfaces with G interface conductance set)
void TransportCUDA::CalculateSAInterfaceField(TransportBaseCUDA* ptrans_sec, mCMBNDInfoCUDA& contactCUDA, bool primary_top)
{
	//the top contacting mesh sets G values
	bool isGInterface_Enabled = ((primary_top && GInterface_Enabled()) || (!primary_top && ptrans_sec->GInterface_Enabled()));
	
	if (isGInterface_Enabled && stsolve == STSOLVE_FERROMAGNETIC && (ptrans_sec->Get_STSolveType() == STSOLVE_NORMALMETAL || ptrans_sec->Get_STSolveType() == STSOLVE_TUNNELING)) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			CalculateSAInterfaceField_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(contactCUDA.get_deviceobject(mGPU), ptrans_sec->spin_S_cmbnd_funcs_sec.get_deviceobject(mGPU), spin_S_cmbnd_funcs_pri.get_deviceobject(mGPU));
		}
	}
}

#endif

#endif