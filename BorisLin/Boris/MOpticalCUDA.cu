#include "MOpticalCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MOPTICAL

#include "Reduction.cuh"

#include "MeshDefs.h"
#include "MeshCUDA.h"
#include "MeshParamsControlCUDA.h"

__global__ void MOpticalCUDA_UpdateField_FM(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuBReal cHmo = *cuMesh.pcHmo;
		cuMesh.update_parameters_mcoarse(idx, *cuMesh.pcHmo, cHmo);

		//magneto-optical field along z direction only : spatial and time dependence set through the usual material parameter mechanism
		Heff[idx] += cuReal3(0, 0, cHmo);

		if (do_reduction) {

			int non_empty_cells = M.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * M[idx] * cuReal3(0, 0, cHmo) / non_empty_cells;
		}

		if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = cuReal3(0, 0, cHmo);
		if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MU0 * M[idx] * cuReal3(0, 0, cHmo);
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void MOpticalCUDA_UpdateField_AFM(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuBReal cHmo = *cuMesh.pcHmo;
		cuMesh.update_parameters_mcoarse(idx, *cuMesh.pcHmo, cHmo);

		//magneto-optical field along z direction only : spatial and time dependence set through the usual material parameter mechanism
		Heff[idx] += cuReal3(0, 0, cHmo);
		Heff2[idx] += cuReal3(0, 0, cHmo);

		if (do_reduction) {

			int non_empty_cells = M.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (M[idx] + M2[idx]) * cuReal3(0, 0, cHmo) / (2 * non_empty_cells);
		}

		if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = cuReal3(0, 0, cHmo);
		if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[idx] = cuReal3(0, 0, cHmo);
		if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -MU0 * M[idx] * cuReal3(0, 0, cHmo);
		if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[idx] = -MU0 * M2[idx] * cuReal3(0, 0, cHmo);
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void MOpticalCUDA::UpdateField(void)
{
	if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				MOpticalCUDA_UpdateField_AFM <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				MOpticalCUDA_UpdateField_AFM <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}

	else {

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				MOpticalCUDA_UpdateField_FM <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				MOpticalCUDA_UpdateField_FM <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
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
__device__ cuBReal ManagedMeshCUDA::Get_EnergyChange_FM_MOpticalCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M = *pM;

	cuBReal cHmo = *pcHmo;
	update_parameters_mcoarse(spin_index, *pcHmo, cHmo);

	if (Mnew != cuReal3()) return -M.h.dim() * (Mnew - M[spin_index]) * (cuBReal)MU0 * cuReal3(0, 0, cHmo);
	else return -M.h.dim() * M[spin_index] * (cuBReal)MU0 * cuReal3(0, 0, cHmo);
}

//Antiferromagnetic
__device__ cuReal2 ManagedMeshCUDA::Get_EnergyChange_AFM_MOpticalCUDA(int spin_index, cuReal3 Mnew_A, cuReal3 Mnew_B)
{
	cuVEC_VC<cuReal3>& M = *pM;
	cuVEC_VC<cuReal3>& M2 = *pM2;

	cuBReal cHmo = *pcHmo;
	update_parameters_mcoarse(spin_index, *pcHmo, cHmo);

	if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) return -MU0 * M.h.dim() * cuReal2((Mnew_A - M[spin_index]) * cuReal3(0, 0, cHmo), (Mnew_B - M2[spin_index]) * cuReal3(0, 0, cHmo));
	else return -MU0 * M.h.dim() * cuReal2(M[spin_index] * cuReal3(0, 0, cHmo), M2[spin_index] * cuReal3(0, 0, cHmo));
}

#endif