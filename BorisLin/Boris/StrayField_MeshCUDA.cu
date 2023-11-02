#include "StrayField_MeshCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_STRAYFIELD

#include "Reduction.cuh"

#include "MeshCUDA.h"
#include "MeshDefs.h"

//----------------------- Initialization

__global__ void set_StrayField_MeshCUDA_pointers_kernel(
	ManagedMeshCUDA& cuMesh, cuVEC<cuReal3>& strayField)
{
	if (threadIdx.x == 0) cuMesh.pstrayField = &strayField;
}

void StrayField_MeshCUDA::set_StrayField_MeshCUDA_pointers(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		set_StrayField_MeshCUDA_pointers_kernel <<< 1, CUDATHREADS >>>
			(pMeshCUDA->cuMesh.get_deviceobject(mGPU), strayField.get_deviceobject(mGPU));
	}
}

//----------------------- Computation

__global__ void UpdateStrayField_FM_kernel(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, cuVEC<cuReal3>& strayField, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuReal3 Hstray = strayField[idx];

		Heff[idx] += Hstray;

		if (do_reduction) {

			int non_empty_cells = M.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * M[idx] * Hstray / non_empty_cells;
		}

		if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hstray;
		if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MU0 * M[idx] * Hstray;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void UpdateStrayField_AFM_kernel(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, cuVEC<cuReal3>& strayField, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuReal3 Hstray = strayField[idx];

		Heff[idx] += Hstray;
		Heff2[idx] += Hstray;

		if (do_reduction) {

			int non_empty_cells = M.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (M[idx] + M2[idx]) * Hstray / (2 * non_empty_cells);
		}

		if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hstray;
		if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[idx] = Hstray;
		if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -MU0 * M[idx] * Hstray;
		if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[idx] = -MU0 * M2[idx] * Hstray;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

void StrayField_MeshCUDA::UpdateFieldCUDA(void)
{
	if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

		//anti-ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				UpdateStrayField_AFM_kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), strayField.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				UpdateStrayField_AFM_kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), strayField.get_deviceobject(mGPU), false);
			}
		}
	}
	else {

		//ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				UpdateStrayField_FM_kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), strayField.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				UpdateStrayField_FM_kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), strayField.get_deviceobject(mGPU), false);
			}
		}
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && MONTE_CARLO == 1

//Ferromagnetic
__device__ cuBReal ManagedMeshCUDA::Get_EnergyChange_FM_StrayField_MeshCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M = *pM;

	cuReal3 Hstray = cuReal3();

	if (pstrayField && pstrayField->linear_size()) {

		Hstray = (*pstrayField)[spin_index];
	}

	if (Mnew != cuReal3()) return -M.h.dim() * (Mnew - M[spin_index]) * (cuBReal)MU0 * Hstray;
	else return -M.h.dim() * M[spin_index] * (cuBReal)MU0 * Hstray;
}

//Antiferromagnetic
__device__ cuReal2 ManagedMeshCUDA::Get_EnergyChange_AFM_StrayField_MeshCUDA(int spin_index, cuReal3 Mnew_A, cuReal3 Mnew_B)
{
	cuVEC_VC<cuReal3>& M = *pM;
	cuVEC_VC<cuReal3>& M2 = *pM2;

	cuReal3 Hstray = cuReal3();

	if (pstrayField && pstrayField->linear_size()) {

		Hstray = (*pstrayField)[spin_index];
	}

	if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) return -M.h.dim() * cuReal2((Mnew_A - M[spin_index]) * (cuBReal)MU0 * Hstray, (Mnew_B - M2[spin_index]) * (cuBReal)MU0 * Hstray);
	else return -M.h.dim() * cuReal2(M[spin_index] * (cuBReal)MU0 * Hstray, M2[spin_index] * (cuBReal)MU0 * Hstray);
}

#endif