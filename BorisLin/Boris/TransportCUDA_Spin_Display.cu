#include "TransportCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "mcuVEC_halo.cuh"

#include "MeshCUDA.h"
#include "SuperMeshCUDA.h"
#include "MeshParamsControlCUDA.h"

//-------------------Display Calculation Methods

//SPIN CURRENT

__global__ void GetSpinCurrent_Kernel(int component, cuVEC<cuReal3>& displayVEC, ManagedMeshCUDA& cuMesh, TransportCUDA_Spin_S_Funcs& poisson_Spin_S, cuVEC_VC<cuReal3>& dM_dt)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& S = *cuMesh.pS;
	cuVEC_VC<cuReal3>& E = *cuMesh.pE;
	cuVEC_VC<cuBReal>& elC = *cuMesh.pelC;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < S.linear_size()) {

		bool cpump_enabled = cuIsNZ(cuMesh.pcpump_eff->get0());
		bool the_enabled = cuIsNZ(cuMesh.pthe_eff->get0());

		cuReal33 Js = cuReal33();

		if (S.is_not_empty(idx)) {
			
			if (poisson_Spin_S.stsolve == STSOLVE_FERROMAGNETIC) {

				//magnetic mesh terms

				cuBReal P = *cuMesh.pP;
				cuBReal De = *cuMesh.pDe;
				cuMesh.update_parameters_ecoarse(idx, *cuMesh.pP, P, *cuMesh.pDe, De);

				//1. drift
				int idx_M = M.position_to_cellidx(S.cellidx_to_position(idx));

				cuReal3 mval = cu_normalize(M[idx_M]);
				cuReal33 grad_S = S.grad_neu(idx);

				Js = (E[idx] | mval) * (P * elC[idx]) * (-(cuBReal)MUB_E);

				//2. diffusion with homogeneous Neumann boundary condition
				Js -= grad_S * De;

				//3. charge pumping
				//4. topological Hall effect

				if (component != 2 && (cpump_enabled || the_enabled)) {

					cuReal33 grad_m = cu_normalize(M.grad_neu(idx_M), M[idx_M]);

					//topological Hall effect contribution
					if (the_enabled) {

						cuBReal n_density = *cuMesh.pn_density;
						cuMesh.update_parameters_ecoarse(idx, *cuMesh.pn_density, n_density);

						cuReal3 B = (grad_m.x ^ grad_m.y);
						Js += cuMesh.pthe_eff->get0() * ((cuBReal)HBAR_E * (cuBReal)MUB_E * elC[idx] * elC[idx] / ((cuBReal)ECHARGE * n_density)) * cuReal33(-E[idx].y * B, E[idx].x * B, cuReal3());
					}

					//charge pumping contribution
					if (cpump_enabled) {

						//value a1
						cuReal3 dm_dt = cu_normalize(dM_dt[idx_M], M[idx_M]);
						Js += cuMesh.pcpump_eff->get0() * ((cuBReal)HBAR_E * (cuBReal)MUB_E * elC[idx] / 2) * cuReal33(dm_dt ^ grad_m.x, dm_dt ^ grad_m.y, cuReal3());
					}
				}
			}
			else {

				//non-magnetic mesh terms

				cuBReal De = *cuMesh.pDe;
				cuBReal SHA = *cuMesh.pSHA;
				cuMesh.update_parameters_ecoarse(idx, *cuMesh.pDe, De, *cuMesh.pSHA, SHA);

				//1. SHE contribution
				Js = cu_epsilon3(E[idx]) * SHA * elC[idx] * (cuBReal)MUB_E;

				//2. diffusion with non-homogeneous Neumann boundary condition
				Js -= S.grad_nneu(idx, cu_epsilon3(E[idx]) * (SHA * elC[idx] * (cuBReal)MUB_E / De)) * De;
			}
		}

		switch (component) {

		case 0:
			displayVEC[idx] = Js.x;
			break;
		case 1:
			displayVEC[idx] = Js.y;
			break;
		case 2:
			displayVEC[idx] = Js.z;
			break;
		}
	}
}

//SPIN TORQUE

__global__ void GetSpinTorque_Kernel(cuVEC<cuReal3>& displayVEC, ManagedMeshCUDA& cuMesh)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& S = *cuMesh.pS;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < M.linear_size()) {

		if (M.is_empty(idx)) {

			displayVEC[idx] = cuReal3();
			return;
		}

		cuBReal De = *cuMesh.pDe;
		cuBReal ts_eff = *cuMesh.pts_eff;
		cuBReal l_ex = *cuMesh.pl_ex;
		cuBReal l_ph = *cuMesh.pl_ph;
		cuMesh.update_parameters_mcoarse(idx, *cuMesh.pDe, De, *cuMesh.pts_eff, ts_eff, *cuMesh.pl_ex, l_ex, *cuMesh.pl_ph, l_ph);

		cuReal3 Sav = S.weighted_average(M.cellidx_to_position(idx), M.h);
		cuReal3 m = cu_normalize(M[idx]);

		displayVEC[idx] = ts_eff * ((Sav ^ m) * De / (l_ex * l_ex) + (m ^ (Sav ^ m)) * De / (l_ph * l_ph));
	}
}

//SPIN INTERFACE TORQUE

__global__ void CalculateDisplaySAInterfaceTorque_Kernel(
	CMBNDInfoCUDA& contact, 
	TransportCUDA_Spin_S_CMBND_Sec& cmbndFuncs_sec, TransportCUDA_Spin_S_CMBND_Pri& cmbndFuncs_pri, 
	cuVEC<cuReal3>& displayVEC)
{
	cuVEC_VC<cuReal3>& M = *cmbndFuncs_pri.pcuMesh->pM;
	cuVEC_VC<cuReal3>& S_pri = *cmbndFuncs_pri.pcuMesh->pS;
	//access S on first device, which contains origin of entire mcuVEC
	cuVEC_VC<cuReal3>& S0_sec = *cmbndFuncs_sec.ppcuMesh[0]->pS;

	int box_idx = blockIdx.x * blockDim.x + threadIdx.x;

	//interface conductance method with F being the primary mesh : calculate and set spin torque

	//convert the cells box from S mesh to M mesh
	cuINT3 mbox_start = M.cellidx_from_position(S_pri.cellidx_to_position(contact.cells_box.s) + M.rect.s);
	cuINT3 mbox_end = M.cellidx_from_position(S_pri.cellidx_to_position(contact.cells_box.e - cuINT3(1)) + M.rect.s) + cuINT3(1);

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

		cuBReal tsi_eff = *cmbndFuncs_pri.pcuMesh->ptsi_eff;
		cmbndFuncs_pri.pcuMesh->update_parameters_mcoarse(mcell1_idx, *cmbndFuncs_pri.pcuMesh->ptsi_eff, tsi_eff);

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

		cuReal3 stencil_sec = M.h - cu_mod(mhshift_primary) + cu_mod(contact.hshift_secondary);
		cuReal3 stencil_pri = M.h - cu_mod(mhshift_primary) + cu_mod(contact.hshift_primary);

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

			cuBReal gI = (2.0 * (cuBReal)GMUB_2E / dh) * Gmix.j / Mnorm;
			cuBReal gR = (2.0 * (cuBReal)GMUB_2E / dh) * Gmix.i / Mnorm;

			displayVEC[mcell1_idx] += tsi_eff * (gI * (M[mcell1_idx] ^ dVs) + gR * (M[mcell1_idx] ^ (M[mcell1_idx] ^ dVs)) / Mnorm);
		}
	}
}

//Launchers

//return x, y, or z component of spin current (component = 0, 1, or 2)
mcu_VEC(cuReal3)& TransportCUDA::GetSpinCurrent(int component)
{
	if (!PrepareDisplayVEC(pMeshCUDA->h_e)) return displayVEC;

	if (stsolve != STSOLVE_NONE) {

		pMeshCUDA->S.exchange_halos();
		pMeshCUDA->M.exchange_halos();

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			GetSpinCurrent_Kernel <<< (pMeshCUDA->elC.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(component, displayVEC.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU), poisson_Spin_S.get_deviceobject(mGPU), dM_dt.get_deviceobject(mGPU));
		}
	}

	return displayVEC;
}

//return spin torque computed from spin accumulation
mcu_VEC(cuReal3)& TransportCUDA::GetSpinTorque(void)
{
	if (!PrepareDisplayVEC(pMeshCUDA->h)) return displayVEC;
	
	if (stsolve == STSOLVE_FERROMAGNETIC) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			GetSpinTorque_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(displayVEC.get_deviceobject(mGPU), pMeshCUDA->cuMesh.get_deviceobject(mGPU));
		}
	}

	return displayVEC;
}

//Calculate the interface spin accumulation torque for a given contact (in magnetic meshes for NF interfaces with G interface conductance set), accumulating result in displayVEC
void TransportCUDA::CalculateDisplaySAInterfaceTorque(TransportBaseCUDA* ptrans_sec, mCMBNDInfoCUDA& contactCUDA, bool primary_top)
{
	//the top contacting mesh sets G values
	bool isGInterface_Enabled = ((primary_top && GInterface_Enabled()) || (!primary_top && ptrans_sec->GInterface_Enabled()));

	if (isGInterface_Enabled && stsolve == STSOLVE_FERROMAGNETIC && (ptrans_sec->Get_STSolveType() == STSOLVE_NORMALMETAL || ptrans_sec->Get_STSolveType() == STSOLVE_TUNNELING)) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			CalculateDisplaySAInterfaceTorque_Kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(contactCUDA.get_deviceobject(mGPU), ptrans_sec->spin_S_cmbnd_funcs_sec.get_deviceobject(mGPU), spin_S_cmbnd_funcs_pri.get_deviceobject(mGPU), displayVEC.get_deviceobject(mGPU));
		}
	}
}

#endif

#endif