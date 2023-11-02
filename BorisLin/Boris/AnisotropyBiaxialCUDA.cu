#include "AnisotropyBiaxialCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_ANIBI

#include "Reduction.cuh"

#include "MeshCUDA.h"
#include "MeshParamsControlCUDA.h"
#include "MeshDefs.h"

__global__ void Anisotropy_BiaxialCUDA_FM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
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
			cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
			cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms, *cuMesh.pK1, K1, *cuMesh.pK2, K2, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

			//calculate m.ea1 dot product (uniaxial contribution)
			cuBReal u1 = (M[idx] * mcanis_ea1) / Ms;

			//calculate m.ea2 and m.ea3 dot products (biaxial contribution)
			cuBReal b1 = (M[idx] * mcanis_ea2) / Ms;
			cuBReal b2 = (M[idx] * mcanis_ea3) / Ms;

			//update effective field with the anisotropy field
			Heff_value = (2 / ((cuBReal)MU0*Ms)) * (K1 * u1 * mcanis_ea1 - K2 * (b1*b2*b2 * mcanis_ea2 + b1*b1*b2 * mcanis_ea3));

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = (K1 * (1 - u1*u1) + K2 * b1*b1*b2*b2) / non_empty_cells;
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = (K1 * (1 - u1*u1) + K2 * b1*b1*b2*b2);
		}

		Heff[idx] += Heff_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void Anisotropy_BiaxialCUDA_AFM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
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
			cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
			cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pK1_AFM, K1_AFM, *cuMesh.pK2_AFM, K2_AFM, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

			//calculate m.ea1 dot product (uniaxial contribution)
			cuBReal u1_A = (M[idx] * mcanis_ea1) / Ms_AFM.i;
			cuBReal u1_B = (M2[idx] * mcanis_ea1) / Ms_AFM.j;

			//calculate m.ea2 and m.ea3 dot products (biaxial contribution)
			cuBReal b1_A = (M[idx] * mcanis_ea2) / Ms_AFM.i;
			cuBReal b1_B = (M2[idx] * mcanis_ea2) / Ms_AFM.j;

			cuBReal b2_A = (M[idx] * mcanis_ea3) / Ms_AFM.i;
			cuBReal b2_B = (M2[idx] * mcanis_ea3) / Ms_AFM.j;

			//update effective field with the anisotropy field
			Heff_value = (2 / ((cuBReal)MU0*Ms_AFM.i)) * (K1_AFM.i * u1_A * mcanis_ea1 - K2_AFM.i * (b1_A*b2_A*b2_A * mcanis_ea2 + b1_A*b1_A*b2_A * mcanis_ea3));
			Heff2_value = (2 / ((cuBReal)MU0*Ms_AFM.j)) * (K1_AFM.j * u1_B * mcanis_ea1 - K2_AFM.j * (b1_B*b2_B*b2_B * mcanis_ea2 + b1_B*b1_B*b2_B * mcanis_ea3));

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = (K1_AFM.i * (1 - u1_A*u1_A) + K2_AFM.i * b1_A*b1_A*b2_A*b2_A + K1_AFM.j * (1 - u1_B*u1_B) + K2_AFM.j * b1_B*b1_B*b2_B*b2_B) / (2 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[idx] = Heff2_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = K1_AFM.i * (1 - u1_A*u1_A) + K2_AFM.i * b1_A*b1_A*b2_A*b2_A;
			if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[idx] = K1_AFM.j * (1 - u1_B*u1_B) + K2_AFM.j * b1_B*b1_B*b2_B*b2_B;
		}

		Heff[idx] += Heff_value;
		Heff2[idx] += Heff2_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Anisotropy_BiaxialCUDA::UpdateField(void)
{
	if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

		//anti-ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_BiaxialCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_BiaxialCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
	else {

		//ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_BiaxialCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_BiaxialCUDA_FM_UpdateField << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
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
__device__ cuBReal ManagedMeshCUDA::Get_EnergyChange_FM_AnisotropyBiaxialCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M = *pM;

	cuBReal K1 = *pK1;
	cuBReal K2 = *pK2;
	cuBReal Ms = *pMs;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	cuReal3 mcanis_ea2 = *pmcanis_ea2;
	cuReal3 mcanis_ea3 = *pmcanis_ea3;
	update_parameters_mcoarse(spin_index, *pMs, Ms, *pK1, K1, *pK2, K2, *pmcanis_ea1, mcanis_ea1, *pmcanis_ea2, mcanis_ea2, *pmcanis_ea3, mcanis_ea3);

	//calculate m.ea1 dot product (uniaxial contribution)
	cuBReal u1 = M[spin_index] * mcanis_ea1 / Ms;

	//calculate m.ea2 and m.ea3 dot products (biaxial contribution)
	cuBReal b1 = M[spin_index] * mcanis_ea2 / Ms;
	cuBReal b2 = M[spin_index] * mcanis_ea3 / Ms;

	if (Mnew != cuReal3()) {

		//calculate m.ea1 dot product (uniaxial contribution)
		cuBReal u1new = Mnew * mcanis_ea1 / Ms;

		//calculate m.ea2 and m.ea3 dot products (biaxial contribution)
		cuBReal b1new = Mnew * mcanis_ea2 / Ms;
		cuBReal b2new = Mnew * mcanis_ea3 / Ms;

		return M.h.dim() * ((K1 * (1 - u1new * u1new) + K2 * b1new * b1new * b2new * b2new) - (K1 * (1 - u1 * u1) + K2 * b1 * b1 * b2 * b2));
	}
	else return M.h.dim() * (K1 * (1 - u1 * u1) + K2 * b1 * b1 * b2 * b2);
}

//Antiferromagnetic
__device__ cuReal2 ManagedMeshCUDA::Get_EnergyChange_AFM_AnisotropyBiaxialCUDA(int spin_index, cuReal3 Mnew_A, cuReal3 Mnew_B)
{
	cuVEC_VC<cuReal3>& M = *pM;
	cuVEC_VC<cuReal3>& M2 = *pM2;

	cuReal2 Ms_AFM = *pMs_AFM;
	cuReal2 K1_AFM = *pK1_AFM;
	cuReal2 K2_AFM = *pK2_AFM;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	cuReal3 mcanis_ea2 = *pmcanis_ea2;
	cuReal3 mcanis_ea3 = *pmcanis_ea3;
	update_parameters_mcoarse(spin_index, *pMs_AFM, Ms_AFM, *pK1_AFM, K1_AFM, *pK2_AFM, K2_AFM, *pmcanis_ea1, mcanis_ea1, *pmcanis_ea2, mcanis_ea2, *pmcanis_ea3, mcanis_ea3);

	//calculate m.ea1 dot product (uniaxial contribution)
	cuBReal u1_A = (M[spin_index] * mcanis_ea1) / Ms_AFM.i;
	cuBReal u1_B = (M2[spin_index] * mcanis_ea1) / Ms_AFM.j;

	//calculate m.ea2 and m.ea3 dot products (biaxial contribution)
	cuBReal b1_A = (M[spin_index] * mcanis_ea2) / Ms_AFM.i;
	cuBReal b1_B = (M2[spin_index] * mcanis_ea2) / Ms_AFM.j;

	cuBReal b2_A = (M[spin_index] * mcanis_ea3) / Ms_AFM.i;
	cuBReal b2_B = (M2[spin_index] * mcanis_ea3) / Ms_AFM.j;

	cuBReal energyA = K1_AFM.i * (1 - u1_A * u1_A) + K2_AFM.i * b1_A * b1_A * b2_A * b2_A;
	cuBReal energyB = K1_AFM.j * (1 - u1_B * u1_B) + K2_AFM.j * b1_B * b1_B * b2_B * b2_B;

	if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) {

		//calculate m.ea1 dot product (uniaxial contribution)
		cuBReal u1_Anew = (Mnew_A * mcanis_ea1) / Ms_AFM.i;
		cuBReal u1_Bnew = (Mnew_B * mcanis_ea1) / Ms_AFM.j;

		//calculate m.ea2 and m.ea3 dot products (biaxial contribution)
		cuBReal b1_Anew = (Mnew_A * mcanis_ea2) / Ms_AFM.i;
		cuBReal b1_Bnew = (Mnew_B * mcanis_ea2) / Ms_AFM.j;

		cuBReal b2_Anew = (Mnew_A * mcanis_ea3) / Ms_AFM.i;
		cuBReal b2_Bnew = (Mnew_B * mcanis_ea3) / Ms_AFM.j;

		cuBReal energyAnew = K1_AFM.i * (1 - u1_Anew * u1_Anew) + K2_AFM.i * b1_Anew * b1_Anew * b2_Anew * b2_Anew;
		cuBReal energyBnew = K1_AFM.j * (1 - u1_Bnew * u1_Bnew) + K2_AFM.j * b1_Bnew * b1_Bnew * b2_Bnew * b2_Bnew;

		return M.h.dim() * cuReal2(energyAnew - energyA, energyBnew - energyB);
	}
	else return M.h.dim() * cuReal2(energyA, energyB);
}

#endif