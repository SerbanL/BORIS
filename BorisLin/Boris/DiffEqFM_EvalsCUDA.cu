#include "DiffEqFMCUDA.h"

#if COMPILECUDA == 1
#ifdef MESH_COMPILATION_FERROMAGNETIC

//defines evaluation methods kernel launchers

#include "MeshParamsControlCUDA.h"

//-----------------------------------------

__global__ void RestoreMagnetization_FM_kernel(cuVEC_VC<cuReal3>& M, cuVEC<cuReal3>& sM1)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < M.linear_size()) {

		M[idx] = sM1[idx];
	}
}

//Restore magnetization after a failed step for adaptive time-step methods
void DifferentialEquationFMCUDA::RestoreMagnetization(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		RestoreMagnetization_FM_kernel <<< (sM1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (pMeshCUDA->M.get_deviceobject(mGPU), sM1.get_deviceobject(mGPU));
	}
}

__global__ void SaveMagnetization_FM_kernel(cuVEC_VC<cuReal3>& M, cuVEC<cuReal3>& sM1)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < M.linear_size()) {

		sM1[idx] = M[idx];
	}
}

//Save current magnetization in sM VECs (e.g. useful to reset dM / dt calculation)
void DifferentialEquationFMCUDA::SaveMagnetization(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		SaveMagnetization_FM_kernel <<< (sM1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (pMeshCUDA->M.get_deviceobject(mGPU), sM1.get_deviceobject(mGPU));
	}
}

//-----------------------------------------

__global__ void RenormalizeMagnetization_FM_kernel(ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < cuMesh.pM->linear_size()) {

		if (cuMesh.pM->is_not_empty(idx)) {

			cuBReal Ms = *cuMesh.pMs;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms);

			if (Ms) (*cuMesh.pM)[idx].renormalize(Ms);
		}
	}
}

//Restore magnetization after a failed step for adaptive time-step methods
void DifferentialEquationFMCUDA::RenormalizeMagnetization(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		RenormalizeMagnetization_FM_kernel <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (pMeshCUDA->cuMesh.get_deviceobject(mGPU));
	}

}

//-----------------------------------------

#endif
#endif