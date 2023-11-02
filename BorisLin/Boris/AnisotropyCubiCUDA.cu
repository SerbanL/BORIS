#include "AnisotropyCubiCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_ANICUBI

#include "Reduction.cuh"

#include "MeshCUDA.h"
#include "MeshParamsControlCUDA.h"
#include "MeshDefs.h"

__global__ void Anisotropy_CubicCUDA_FM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
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

			//calculate m.ea1, m.ea2 and m.ea3 dot products
			cuBReal d1 = (M[idx] * mcanis_ea1) / Ms;
			cuBReal d2 = (M[idx] * mcanis_ea2) / Ms;
			cuBReal d3 = (M[idx] * mcanis_ea3) / Ms;

			//terms for K1 contribution
			cuBReal a1 = d1 * (d2*d2 + d3 * d3);
			cuBReal a2 = d2 * (d1*d1 + d3 * d3);
			cuBReal a3 = d3 * (d1*d1 + d2 * d2);

			//terms for K2 contribution
			cuBReal d123 = d1 * d2*d3;

			cuBReal b1 = d123 * d2*d3;
			cuBReal b2 = d123 * d1*d3;
			cuBReal b3 = d123 * d1*d2;

			//update effective field with the anisotropy field
			Heff_value = cuReal3(
				(-2 * K1 / ((cuBReal)MU0*Ms)) * (mcanis_ea1.i * a1 + mcanis_ea2.i * a2 + mcanis_ea3.i * a3)
				+ (-2 * K2 / ((cuBReal)MU0*Ms)) * (mcanis_ea1.i * b1 + mcanis_ea2.i * b2 + mcanis_ea3.i * b3),

				(-2 * K1 / ((cuBReal)MU0*Ms)) * (mcanis_ea1.j * a1 + mcanis_ea2.j * a2 + mcanis_ea3.j * a3)
				+ (-2 * K2 / ((cuBReal)MU0*Ms)) * (mcanis_ea1.j * b1 + mcanis_ea2.j * b2 + mcanis_ea3.j * b3),

				(-2 * K1 / ((cuBReal)MU0*Ms)) * (mcanis_ea1.k * a1 + mcanis_ea2.k * a2 + mcanis_ea3.k * a3)
				+ (-2 * K2 / ((cuBReal)MU0*Ms)) * (mcanis_ea1.k * b1 + mcanis_ea2.k * b2 + mcanis_ea3.k * b3)
			);

			if (do_reduction) {

				//update energy (E/V)		
				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = (K1 * (d1*d1*d2*d2 + d1 * d1*d3*d3 + d2 * d2*d3*d3) + K2 * d123*d123) / non_empty_cells;
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = (K1 * (d1*d1*d2*d2 + d1 * d1*d3*d3 + d2 * d2*d3*d3) + K2 * d123*d123);
		}

		Heff[idx] += Heff_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void Anisotropy_CubicCUDA_AFM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
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

			//calculate m.ea1, m.ea2 and m.ea3 dot products
			cuBReal d1 = (M[idx] * mcanis_ea1) / Ms_AFM.i;
			cuBReal d2 = (M[idx] * mcanis_ea2) / Ms_AFM.i;
			cuBReal d3 = (M[idx] * mcanis_ea3) / Ms_AFM.i;

			//terms for K1 contribution
			cuBReal a1 = d1 * (d2*d2 + d3 * d3);
			cuBReal a2 = d2 * (d1*d1 + d3 * d3);
			cuBReal a3 = d3 * (d1*d1 + d2 * d2);

			//terms for K2 contribution
			cuBReal d123 = d1*d2*d3;

			cuBReal b1 = d123 * d2*d3;
			cuBReal b2 = d123 * d1*d3;
			cuBReal b3 = d123 * d1*d2;

			//update effective field with the anisotropy field
			Heff_value = cuReal3(
				(-2 * K1_AFM.i / ((cuBReal)MU0*Ms_AFM.i)) * (mcanis_ea1.i * a1 + mcanis_ea2.i * a2 + mcanis_ea3.i * a3)
				+ (-2 * K2_AFM.i / ((cuBReal)MU0*Ms_AFM.i)) * (mcanis_ea1.i * b1 + mcanis_ea2.i * b2 + mcanis_ea3.i * b3),

				(-2 * K1_AFM.i / ((cuBReal)MU0*Ms_AFM.i)) * (mcanis_ea1.j * a1 + mcanis_ea2.j * a2 + mcanis_ea3.j * a3)
				+ (-2 * K2_AFM.i / ((cuBReal)MU0*Ms_AFM.i)) * (mcanis_ea1.j * b1 + mcanis_ea2.j * b2 + mcanis_ea3.j * b3),

				(-2 * K1_AFM.i / ((cuBReal)MU0*Ms_AFM.i)) * (mcanis_ea1.k * a1 + mcanis_ea2.k * a2 + mcanis_ea3.k * a3)
				+ (-2 * K2_AFM.i / ((cuBReal)MU0*Ms_AFM.i)) * (mcanis_ea1.k * b1 + mcanis_ea2.k * b2 + mcanis_ea3.k * b3)
			);

			//same thing for sub-lattice B

			cuBReal d1B = (M2[idx] * mcanis_ea1) / Ms_AFM.j;
			cuBReal d2B = (M2[idx] * mcanis_ea2) / Ms_AFM.j;
			cuBReal d3B = (M2[idx] * mcanis_ea3) / Ms_AFM.j;

			cuBReal a1B = d1B * (d2B*d2B + d3B*d3B);
			cuBReal a2B = d2B * (d1B*d1B + d3B*d3B);
			cuBReal a3B = d3B * (d1B*d1B + d2B*d2B);

			cuBReal d123B = d1B*d2B*d3B;

			cuBReal b1B = d123B*d2B*d3B;
			cuBReal b2B = d123B*d1B*d3B;
			cuBReal b3B = d123B*d1B*d2B;

			Heff2_value = cuReal3(
				(-2 * K1_AFM.j / ((cuBReal)MU0*Ms_AFM.j)) * (mcanis_ea1.i * a1B + mcanis_ea2.i * a2B + mcanis_ea3.i * a3B)
				+ (-2 * K2_AFM.j / ((cuBReal)MU0*Ms_AFM.j)) * (mcanis_ea1.i * b1B + mcanis_ea2.i * b2B + mcanis_ea3.i * b3B),

				(-2 * K1_AFM.j / ((cuBReal)MU0*Ms_AFM.j)) * (mcanis_ea1.j * a1B + mcanis_ea2.j * a2B + mcanis_ea3.j * a3B)
				+ (-2 * K2_AFM.j / ((cuBReal)MU0*Ms_AFM.j)) * (mcanis_ea1.j * b1B + mcanis_ea2.j * b2B + mcanis_ea3.j * b3B),

				(-2 * K1_AFM.j / ((cuBReal)MU0*Ms_AFM.j)) * (mcanis_ea1.k * a1B + mcanis_ea2.k * a2B + mcanis_ea3.k * a3B)
				+ (-2 * K2_AFM.j / ((cuBReal)MU0*Ms_AFM.j)) * (mcanis_ea1.k * b1B + mcanis_ea2.k * b2B + mcanis_ea3.k * b3B)
			);

			if (do_reduction) {

				//update energy (E/V)		
				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = ((K1_AFM.i * (d1*d1*d2*d2 + d1*d1*d3*d3 + d2*d2*d3*d3) + K2_AFM.i * d123*d123) + (K1_AFM.j * (d1B*d1B*d2B*d2B + d1B*d1B*d3B*d3B + d2B*d2B*d3B*d3B) + K2_AFM.j * d123B*d123B)) / (2*non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[idx] = Heff2_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = K1_AFM.i * (d1*d1*d2*d2 + d1*d1*d3*d3 + d2*d2*d3*d3) + K2_AFM.i * d123*d123;
			if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[idx] = K1_AFM.j * (d1B*d1B*d2B*d2B + d1B*d1B*d3B*d3B + d2B*d2B*d3B*d3B) + K2_AFM.j * d123B*d123B;
		}

		Heff[idx] += Heff_value;
		Heff2[idx] += Heff2_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Anisotropy_CubicCUDA::UpdateField(void)
{
	if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

		//anti-ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_CubicCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_CubicCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
	else {

		//ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_CubicCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_CubicCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
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
__device__ cuBReal ManagedMeshCUDA::Get_EnergyChange_FM_AnisotropyCubiCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M = *pM;

	cuBReal K1 = *pK1;
	cuBReal K2 = *pK2;
	cuBReal Ms = *pMs;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	cuReal3 mcanis_ea2 = *pmcanis_ea2;
	cuReal3 mcanis_ea3 = *pmcanis_ea3;
	update_parameters_mcoarse(spin_index, *pMs, Ms, *pK1, K1, *pK2, K2, *pmcanis_ea1, mcanis_ea1, *pmcanis_ea2, mcanis_ea2, *pmcanis_ea3, mcanis_ea3);

	cuReal3 S = M[spin_index] / Ms;
	cuReal3 S_new = Mnew / Ms;

	//calculate m.ea1, m.ea2 and m.ea3 dot products
	cuBReal d1 = S * mcanis_ea1;
	cuBReal d2 = S * mcanis_ea2;
	cuBReal d3 = S * mcanis_ea3;
	cuBReal d123 = d1 * d2 * d3;

	if (Mnew != cuReal3()) {

		cuBReal d1_new = S_new * mcanis_ea1;
		cuBReal d2_new = S_new * mcanis_ea2;
		cuBReal d3_new = S_new * mcanis_ea3;
		cuBReal d123_new = d1_new * d2_new * d3_new;

		return M.h.dim() * (
			(K1 * (d1_new * d1_new * d2_new * d2_new + d1_new * d1_new * d3_new * d3_new + d2_new * d2_new * d3_new * d3_new) + K2 * d123_new * d123_new)
			- (K1 * (d1 * d1 * d2 * d2 + d1 * d1 * d3 * d3 + d2 * d2 * d3 * d3) + K2 * d123 * d123));
	}
	else return M.h.dim() * (K1 * (d1 * d1 * d2 * d2 + d1 * d1 * d3 * d3 + d2 * d2 * d3 * d3) + K2 * d123 * d123);
}

//Antiferromagnetic
__device__ cuReal2 ManagedMeshCUDA::Get_EnergyChange_AFM_AnisotropyCubiCUDA(int spin_index, cuReal3 Mnew_A, cuReal3 Mnew_B)
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

	//calculate m.ea1, m.ea2 and m.ea3 dot products
	cuBReal d1 = (M[spin_index] * mcanis_ea1) / Ms_AFM.i;
	cuBReal d2 = (M[spin_index] * mcanis_ea2) / Ms_AFM.i;
	cuBReal d3 = (M[spin_index] * mcanis_ea3) / Ms_AFM.i;
	cuBReal d123 = d1 * d2 * d3;

	//same thing for sub-lattice B

	cuBReal d1B = (M2[spin_index] * mcanis_ea1) / Ms_AFM.j;
	cuBReal d2B = (M2[spin_index] * mcanis_ea2) / Ms_AFM.j;
	cuBReal d3B = (M2[spin_index] * mcanis_ea3) / Ms_AFM.j;
	cuBReal d123B = d1B * d2B * d3B;

	cuBReal energyA = K1_AFM.i * (d1 * d1 * d2 * d2 + d1 * d1 * d3 * d3 + d2 * d2 * d3 * d3) + K2_AFM.i * d123 * d123;
	cuBReal energyB = K1_AFM.j * (d1B * d1B * d2B * d2B + d1B * d1B * d3B * d3B + d2B * d2B * d3B * d3B) + K2_AFM.j * d123B * d123B;

	if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) {

		//calculate m.ea1, m.ea2 and m.ea3 dot products
		cuBReal d1new = (M[spin_index] * mcanis_ea1) / Ms_AFM.i;
		cuBReal d2new = (M[spin_index] * mcanis_ea2) / Ms_AFM.i;
		cuBReal d3new = (M[spin_index] * mcanis_ea3) / Ms_AFM.i;
		cuBReal d123new = d1new * d2new * d3new;

		//same thing for sub-lattice B

		cuBReal d1Bnew = (M2[spin_index] * mcanis_ea1) / Ms_AFM.j;
		cuBReal d2Bnew = (M2[spin_index] * mcanis_ea2) / Ms_AFM.j;
		cuBReal d3Bnew = (M2[spin_index] * mcanis_ea3) / Ms_AFM.j;
		cuBReal d123Bnew = d1Bnew * d2Bnew * d3Bnew;

		cuBReal energyAnew = K1_AFM.i * (d1new * d1new * d2new * d2new + d1new * d1new * d3new * d3new + d2new * d2new * d3new * d3new) + K2_AFM.i * d123new * d123new;
		cuBReal energyBnew = K1_AFM.j * (d1Bnew * d1Bnew * d2Bnew * d2Bnew + d1Bnew * d1Bnew * d3Bnew * d3Bnew + d2Bnew * d2Bnew * d3Bnew * d3Bnew) + K2_AFM.j * d123Bnew * d123Bnew;

		return M.h.dim() * cuReal2(energyAnew - energyA, energyBnew - energyB);
	}
	else return M.h.dim() * cuReal2(energyA, energyB);
}

#endif