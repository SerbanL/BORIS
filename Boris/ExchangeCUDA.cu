#include "ExchangeCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_EXCHANGE

#include "BorisCUDALib.cuh"

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