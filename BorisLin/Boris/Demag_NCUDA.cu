#include "Demag_NCUDA.h"
#include "MeshCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_DEMAG_N

#include "Reduction.cuh"

#include "MeshDefs.h"

__global__ void Demag_NCUDA_FM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	MatPCUDA<cuReal2, cuBReal>& Nxy = *cuMesh.pNxy;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuReal3 Heff_value = cuReal3();

		if (M.is_not_empty(idx)) {

			Heff_value = cuReal3(-cuReal2(Nxy).x * M[idx].x, -cuReal2(Nxy).y * M[idx].y, -(1 - cuReal2(Nxy).x - cuReal2(Nxy).y) * M[idx].z);

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * M[idx] * Heff_value / (2 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MU0 * (M[idx] * Heff_value) / 2;
		}

		Heff[idx] += Heff_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void Demag_NCUDA_AFM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	MatPCUDA<cuReal2, cuBReal>& Nxy = *cuMesh.pNxy;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuReal3 Heff_value = cuReal3();

		if (M.is_not_empty(idx)) {

			cuReal3 Mval = (M[idx] + M2[idx]) / 2;

			Heff_value = cuReal3(-cuReal2(Nxy).x * Mval.x, -cuReal2(Nxy).y * Mval.y, -(1 - cuReal2(Nxy).x - cuReal2(Nxy).y) * Mval.z);

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * Mval * Heff_value / (2 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MU0 * (Mval * Heff_value) / 2;
		}

		Heff[idx] += Heff_value;
		Heff2[idx] += Heff_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Demag_NCUDA::UpdateField(void)
{
	if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Demag_NCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Demag_NCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}

	else if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Demag_NCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Demag_NCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
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
__device__ cuBReal ManagedMeshCUDA::Get_EnergyChange_FM_DemagNCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M = *pM;

	//Nxy shouldn't have a temperature (or spatial) dependence so not using update_parameters_mcoarse here
	cuReal2 Nxy = *pNxy;
	cuBReal Nz = (1 - Nxy.x - Nxy.y);

	if (Mnew != cuReal3()) {

		return ((cuBReal)MU0 / 2) * M.h.dim() * (
			(Mnew * cuReal3(Nxy.x * Mnew.x, Nxy.y * Mnew.y, Nz * Mnew.z))
			- (M[spin_index] * cuReal3(Nxy.x * M[spin_index].x, Nxy.y * M[spin_index].y, Nz * M[spin_index].z)));
	}
	else return ((cuBReal)MU0 / 2) * M.h.dim() * (M[spin_index] * cuReal3(Nxy.x * M[spin_index].x, Nxy.y * M[spin_index].y, Nz * M[spin_index].z));
}

//Antiferromagnetic
__device__ cuReal2 ManagedMeshCUDA::Get_EnergyChange_AFM_DemagNCUDA(int spin_index, cuReal3 Mnew_A, cuReal3 Mnew_B)
{
	cuVEC_VC<cuReal3>& Mvec = *pM;
	cuVEC_VC<cuReal3>& Mvec2 = *pM2;

	//Nxy shouldn't have a temperature (or spatial) dependence so not using update_parameters_mcoarse here
	cuReal2 Nxy = *pNxy;
	cuBReal Nz = (1 - Nxy.x - Nxy.y);

	cuReal3 M = (Mvec[spin_index] + Mvec2[spin_index]) / 2;
	cuReal3 Mnew = (Mnew_A + Mnew_B) / 2;

	cuBReal energy_ = 0.0;

	if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) {

		energy_ = ((cuBReal)MU0 / 2) * Mvec.h.dim() * (
			(Mnew * cuReal3(Nxy.x * Mnew.x, Nxy.y * Mnew.y, Nz * Mnew.z)) -
			(M * cuReal3(Nxy.x * M.x, Nxy.y * M.y, Nz * M.z)));
	}
	else energy_ = ((cuBReal)MU0 / 2) * Mvec.h.dim() * (M * cuReal3(Nxy.x * M.x, Nxy.y * M.y, Nz * M.z));

	return cuReal2(energy_, energy_);
}

#endif