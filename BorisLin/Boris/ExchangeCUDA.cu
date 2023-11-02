#include "ExchangeCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_EXCHANGE

#include "Reduction.cuh"
#include "mcuVEC_halo.cuh"

#include "MeshCUDA.h"
#include "MeshParamsControlCUDA.h"
#include "MeshDefs.h"

//////////////////////////////////////////////////////////////////////// UPDATE FIELD

__global__ void ExchangeCUDA_FM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuReal3 Hexch = cuReal3();

		if (M.is_not_empty(idx)) {

			cuBReal Ms = *cuMesh.pMs;
			cuBReal A = *cuMesh.pA;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms, *cuMesh.pA, A);

			if (*cuMesh.pbase_temperature > 0.0 && *cuMesh.pT_Curie > 0.0) {

				//for finite temperature simulations the magnetization length may have a spatial variation
				//this will not affect the transverse torque (mxH), but will affect the longitudinal term in the sLLB equation (m.H) and cannot be neglected when close to Tc.

				cuReal33 Mg = M.grad_neu(idx);
				cuReal3 dMdx = Mg.x, dMdy = Mg.y, dMdz = Mg.z;

				cuBReal delsq_Msq = 2 * M[idx] * (M.dxx_neu(idx) + M.dyy_neu(idx) + M.dzz_neu(idx)) + 2 * (dMdx * dMdx + dMdy * dMdy + dMdz * dMdz);
				cuBReal Mnorm = M[idx].norm();
				if (cuIsNZ(Mnorm)) Hexch = (2 * A / (MU0*Ms*Ms)) * (M.delsq_neu(idx) - M[idx] * delsq_Msq / (2 * Mnorm*Mnorm));
			}
			else {

				//zero temperature simulations : magnetization length could still vary but will only affect mxH term, so not needed for 0K simulations.
				Hexch = 2 * A * M.delsq_neu(idx) / ((cuBReal)MU0 * Ms * Ms);
			}

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * M[idx] * Hexch / (2 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hexch;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MU0 * (M[idx] * Hexch) / 2;
		}

		Heff[idx] += Hexch;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void ExchangeCUDA_AFM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuReal3 Hexch = cuReal3();
		cuReal3 Hexch2 = cuReal3();

		if (M.is_not_empty(idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuReal2 A_AFM = *cuMesh.pA_AFM;
			cuReal2 Ah = *cuMesh.pAh;
			cuReal2 Anh = *cuMesh.pAnh;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pA_AFM, A_AFM, *cuMesh.pAh, Ah, *cuMesh.pAnh, Anh);

			cuReal3 delsq_M_A = M.delsq_neu(idx);
			cuReal3 delsq_M_B = M2.delsq_neu(idx);

			cuReal2 Mmag = cuReal2(M[idx].norm(), M2[idx].norm());
			if (cuIsNZ(Mmag.i)) Hexch = (2 * A_AFM.i / ((cuBReal)MU0*Ms_AFM.i*Ms_AFM.i)) * delsq_M_A + (-4 * Ah.i * (M[idx] ^ (M[idx] ^ M2[idx])) / (Mmag.i*Mmag.i) + Anh.i * delsq_M_B) / ((cuBReal)MU0*Ms_AFM.i*Ms_AFM.j);
			if (cuIsNZ(Mmag.j)) Hexch2 = (2 * A_AFM.j / ((cuBReal)MU0*Ms_AFM.j*Ms_AFM.j)) * delsq_M_B + (-4 * Ah.j * (M2[idx] ^ (M2[idx] ^ M[idx])) / (Mmag.j*Mmag.j) + Anh.j * delsq_M_A) / ((cuBReal)MU0*Ms_AFM.i*Ms_AFM.j);

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * (M[idx] * Hexch  + M2[idx] * Hexch2) / (4 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hexch;
			if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[idx] = Hexch2;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -MU0 * (M[idx] * Hexch) / 2;
			if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[idx] = -MU0 * (M2[idx] * Hexch2) / 2;
		}

		Heff[idx] += Hexch;
		Heff2[idx] += Hexch2;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Exch_6ngbr_NeuCUDA::UpdateField(void)
{
	if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

		//anti-ferromagnetic mesh

		pMeshCUDA->M.exchange_halos();
		pMeshCUDA->M2.exchange_halos();

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				ExchangeCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				ExchangeCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
	else {

		//ferromagnetic mesh

		pMeshCUDA->M.exchange_halos();
		
		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();
			
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				ExchangeCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				ExchangeCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}

	//if using UVA to compute differential operators then synchronization needed now
	//otherwise a kernel on a device could finish and continue on to diff eq update (which will update M on device), whilst neighboring devices are still accessing these data - data race!
	//if using halo exchanges instead this is not a problem
	mGPU.synchronize_if_uva();

	if (pMeshCUDA->GetMeshExchangeCoupling()) CalculateExchangeCoupling(energy);
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && MONTE_CARLO == 1

__device__ cuBReal ManagedMeshCUDA::Get_EnergyChange_FM_ExchangeCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M = *pM;

	cuBReal Ms = *pMs;
	cuBReal A = *pA;
	update_parameters_mcoarse(spin_index, *pA, A, *pMs, Ms);

	cuReal3 Hexch = (2 * A / ((cuBReal)MU0 * Ms * Ms)) * M.delsq_neu(spin_index);
	cuBReal energy_ = M[spin_index] * Hexch;

	if (Mnew != cuReal3()) {

		//NOTE : here we only need the change in energy due to spin rotation only. Thus the longitudinal part, which is dependent on spin length only, cancels out. Enforce this by making Mnew length same as old one.
		Mnew.renormalize(M[spin_index].norm());

		cuReal3 Mold = M[spin_index];
		M[spin_index] = Mnew;
		Hexch = (2 * A / ((cuBReal)MU0 * Ms * Ms)) * M.delsq_neu(spin_index);
		cuBReal energynew_ = M[spin_index] * Hexch;
		M[spin_index] = Mold;

		//do not divide by 2 as we are not double-counting here
		return -(cuBReal)MU0 * M.h.dim() * (energynew_ - energy_);
	}
	else return -(cuBReal)MU0 * M.h.dim() * energy_;
}

__device__ cuReal2 ManagedMeshCUDA::Get_EnergyChange_AFM_ExchangeCUDA(int spin_index, cuReal3 Mnew_A, cuReal3 Mnew_B)
{
	cuVEC_VC<cuReal3>& M = *pM;
	cuVEC_VC<cuReal3>& M2 = *pM2;

	cuReal2 Ms_AFM = *pMs_AFM;
	cuReal2 A_AFM = *pA_AFM;
	cuReal2 Ah = *pAh;
	cuReal2 Anh = *pAnh;
	update_parameters_mcoarse(spin_index, *pA_AFM, A_AFM, *pMs_AFM, Ms_AFM, *pAh, Ah, *pAnh, Anh);

	auto Get_Energy = [&](void) -> cuReal2 {

		cuReal3 delsq_M_A = M.delsq_neu(spin_index);
		cuReal3 delsq_M_B = M2.delsq_neu(spin_index);

		cuReal3 Hexch = (2 * A_AFM.i / (MU0 * Ms_AFM.i * Ms_AFM.i)) * delsq_M_A + (4 * Ah.i * M2[spin_index] + Anh.i * delsq_M_B) / (MU0 * Ms_AFM.i * Ms_AFM.j);
		cuReal3 Hexch2 = (2 * A_AFM.j / (MU0 * Ms_AFM.j * Ms_AFM.j)) * delsq_M_B + (4 * Ah.j * M[spin_index] + Anh.j * delsq_M_A) / (MU0 * Ms_AFM.i * Ms_AFM.j);

		return cuReal2(M[spin_index] * Hexch, M2[spin_index] * Hexch2);
	};

	cuReal2 energy_ = Get_Energy();

	if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) {

		//NOTE : here we only need the change in energy due to spin rotation only. Thus the longitudinal part, which is dependent on spin length only, cancels out. Enforce this by making Mnew length same as old one.
		Mnew_A.renormalize(M[spin_index].norm());
		Mnew_B.renormalize(M2[spin_index].norm());

		cuReal3 Mold_A = M[spin_index];
		cuReal3 Mold_B = M2[spin_index];

		M[spin_index] = Mnew_A;
		M2[spin_index] = Mnew_B;

		cuReal2 energynew_ = Get_Energy();

		M[spin_index] = Mold_A;
		M2[spin_index] = Mold_B;

		//do not divide by 2 as we are not double-counting here
		return -(cuBReal)MU0 * M.h.dim() * (energynew_ - energy_);
	}
	//If Mnew is null then this method is used to obtain current energy only, not energy change
	else return -(cuBReal)MU0 * M.h.dim() * energy_;
}

#endif