#include "AnisotropyCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_ANIUNI

#include "Reduction.cuh"

#include "MeshCUDA.h"
#include "MeshParamsControlCUDA.h"
#include "MeshDefs.h"

__global__ void Anisotropy_UniaxialCUDA_FM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuReal3 Heff_value = cuReal3();

		if (M.is_not_empty(idx)) {

			cuBReal Ms = *cuMesh.pMs;
			cuBReal K1 = *cuMesh.pK1;
			cuBReal K2 = *cuMesh.pK2;
			cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms, *cuMesh.pK1, K1, *cuMesh.pK2, K2, *cuMesh.pmcanis_ea1, mcanis_ea1);

			//calculate m.ea dot product
			cuBReal dotprod = (M[idx] * mcanis_ea1) / Ms;

			//update effective field with the anisotropy field
			Heff_value = (2 / ((cuBReal)MU0 * Ms)) * dotprod * (K1 + 2 * K2 * (1 - dotprod * dotprod)) * mcanis_ea1;

			if (do_reduction) {

				//update energy (E/V) = K1 * sin^2(theta) + K2 * sin^4(theta) = K1 * [ 1 - dotprod*dotprod ] + K2 * [1 - dotprod * dotprod]^2
				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = ((K1 + K2 * (1 - dotprod * dotprod)) * (1 - dotprod * dotprod)) / non_empty_cells;
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = (K1 + K2 * (1 - dotprod * dotprod)) * (1 - dotprod * dotprod);
		}

		Heff[idx] += Heff_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void Anisotropy_UniaxialCUDA_AFM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuReal3 Heff_value = cuReal3();
		cuReal3 Heff2_value = cuReal3();

		if (M.is_not_empty(idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuReal2 K1_AFM = *cuMesh.pK1_AFM;
			cuReal2 K2_AFM = *cuMesh.pK2_AFM;
			cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pK1_AFM, K1_AFM, *cuMesh.pK2_AFM, K2_AFM, *cuMesh.pmcanis_ea1, mcanis_ea1);

			//calculate m.ea dot product
			cuBReal dotprod = (M[idx] * mcanis_ea1) / Ms_AFM.i;
			cuBReal dotprod2 = (M2[idx] * mcanis_ea1) / Ms_AFM.j;

			//update effective field with the anisotropy field
			Heff_value = (2 / ((cuBReal)MU0 * Ms_AFM.i)) * dotprod * (K1_AFM.i + 2 * K2_AFM.i * (1 - dotprod * dotprod)) * mcanis_ea1;
			Heff2_value = (2 / ((cuBReal)MU0 * Ms_AFM.j)) * dotprod2 * (K1_AFM.j + 2 * K2_AFM.j * (1 - dotprod2 * dotprod2)) * mcanis_ea1;

			if (do_reduction) {

				//update energy (E/V) = K1 * sin^2(theta) + K2 * sin^4(theta) = K1 * [ 1 - dotprod*dotprod ] + K2 * [1 - dotprod * dotprod]^2
				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = ((K1_AFM.i + K2_AFM.i * (1 - dotprod * dotprod)) * (1 - dotprod * dotprod) + (K1_AFM.j + K2_AFM.j * (1 - dotprod2 * dotprod2)) * (1 - dotprod2 * dotprod2)) / (2 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[idx] = Heff2_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = (K1_AFM.i + K2_AFM.i * (1 - dotprod * dotprod)) * (1 - dotprod * dotprod);
			if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[idx] = (K1_AFM.j + K2_AFM.j * (1 - dotprod2 * dotprod2)) * (1 - dotprod2 * dotprod2);
		}

		Heff[idx] += Heff_value;
		Heff2[idx] += Heff2_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Anisotropy_UniaxialCUDA::UpdateField(void)
{
	if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

		//anti-ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_UniaxialCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_UniaxialCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
	else {

		//ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_UniaxialCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_UniaxialCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && MONTE_CARLO == 1

//Ferromagnetic
__device__ cuBReal ManagedMeshCUDA::Get_EnergyChange_FM_AnisotropyCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M = *pM;

	cuBReal K1 = *pK1;
	cuBReal K2 = *pK2;
	cuBReal Ms = *pMs;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	update_parameters_mcoarse(spin_index, *pMs, Ms, *pK1, K1, *pK2, K2, *pmcanis_ea1, mcanis_ea1);

	cuBReal dotprod = (M[spin_index] * mcanis_ea1) / Ms;
	cuBReal dpsq = dotprod * dotprod;

	if (Mnew != cuReal3()) {

		//calculate m.ea dot product
		cuBReal dotprod_new = Mnew * mcanis_ea1 / Ms;
		cuBReal dpsq_new = dotprod_new * dotprod_new;

		return M.h.dim() * ((K1 + K2 * (1 - dpsq_new)) * (1 - dpsq_new) - (K1 + K2 * (1 - dpsq)) * (1 - dpsq));
	}
	else return M.h.dim() * (K1 + K2 * (1 - dpsq)) * (1 - dpsq);
}

//Antiferromagnetic
__device__ cuReal2 ManagedMeshCUDA::Get_EnergyChange_AFM_AnisotropyCUDA(int spin_index, cuReal3 Mnew_A, cuReal3 Mnew_B)
{
	cuVEC_VC<cuReal3>& M = *pM;
	cuVEC_VC<cuReal3>& M2 = *pM2;

	cuReal2 Ms_AFM = *pMs_AFM;
	cuReal2 K1_AFM = *pK1_AFM;
	cuReal2 K2_AFM = *pK2_AFM;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	update_parameters_mcoarse(spin_index, *pMs_AFM, Ms_AFM, *pK1_AFM, K1_AFM, *pK2_AFM, K2_AFM, *pmcanis_ea1, mcanis_ea1);

	//calculate m.ea dot product
	cuBReal dotprod = (M[spin_index] * mcanis_ea1) / Ms_AFM.i;
	cuBReal dotprod2 = (M2[spin_index] * mcanis_ea1) / Ms_AFM.j;

	cuBReal energyA = (K1_AFM.i + K2_AFM.i * (1 - dotprod * dotprod)) * (1 - dotprod * dotprod);
	cuBReal energyB = (K1_AFM.j + K2_AFM.j * (1 - dotprod2 * dotprod2)) * (1 - dotprod2 * dotprod2);

	if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) {

		dotprod = (Mnew_A * mcanis_ea1) / Ms_AFM.i;
		dotprod2 = (Mnew_B * mcanis_ea1) / Ms_AFM.j;

		cuBReal energynewA = (K1_AFM.i + K2_AFM.i * (1 - dotprod * dotprod)) * (1 - dotprod * dotprod);
		cuBReal energynewB = (K1_AFM.j + K2_AFM.j * (1 - dotprod2 * dotprod2)) * (1 - dotprod2 * dotprod2);

		return M.h.dim() * cuReal2(energynewA - energyA, energynewB - energyB);
	}
	else return M.h.dim() * cuReal2(energyA, energyB);
}

#endif