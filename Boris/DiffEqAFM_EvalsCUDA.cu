#include "DiffEqAFMCUDA.h"

#if COMPILECUDA == 1
#ifdef MESH_COMPILATION_ANTIFERROMAGNETIC

#include "BorisCUDALib.cuh"

#include "MeshParamsControlCUDA.h"

//-----------------------------------------

__global__ void Restoremagnetization_AFM_kernel(cuVEC_VC<cuReal3>& M, cuVEC<cuReal3>& sM1, cuVEC_VC<cuReal3>& M2, cuVEC<cuReal3>& sM1_2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < M.linear_size()) {

		M[idx] = sM1[idx];
		M2[idx] = sM1_2[idx];
	}
}

//Restore magnetization after a failed step for adaptive time-step methods
void DifferentialEquationAFMCUDA::RestoreMagnetization(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Restoremagnetization_AFM_kernel <<< (sM1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(pMeshCUDA->M.get_deviceobject(mGPU), sM1.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU), sM1_2.get_deviceobject(mGPU));
	}
}

__global__ void SaveMagnetization_AFM_kernel(cuVEC_VC<cuReal3>& M, cuVEC<cuReal3>& sM1, cuVEC_VC<cuReal3>& M2, cuVEC<cuReal3>& sM1_2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < M.linear_size()) {

		sM1[idx] = M[idx];
		sM1_2[idx] = M2[idx];
	}
}

//Save current magnetization in sM VECs (e.g. useful to reset dM / dt calculation)
void DifferentialEquationAFMCUDA::SaveMagnetization(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		SaveMagnetization_AFM_kernel <<< (sM1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(pMeshCUDA->M.get_deviceobject(mGPU), sM1.get_deviceobject(mGPU), pMeshCUDA->M2.get_deviceobject(mGPU), sM1_2.get_deviceobject(mGPU));
	}
}

//-----------------------------------------

__global__ void RenormalizeMagnetization_AFM_kernel(ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM);

			if (Ms_AFM.i) (*cuMesh.pM)[idx].renormalize(Ms_AFM.i);
			if (Ms_AFM.j) (*cuMesh.pM2)[idx].renormalize(Ms_AFM.j);
		}
	}
}

//Restore magnetization after a failed step for adaptive time-step methods
void DifferentialEquationAFMCUDA::RenormalizeMagnetization(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		RenormalizeMagnetization_AFM_kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(pMeshCUDA->cuMesh.get_deviceobject(mGPU));
	}
}

//-----------------------------------------

#endif
#endif