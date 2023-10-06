#include "DemagMCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_DEMAG

#include "BorisCUDALib.cuh"
#include "MeshCUDA.h"

//----------------------- Initialization

__global__ void set_DemagCUDA_pointers_kernel(
	ManagedMeshCUDA& cuMesh, cuVEC<cuReal3>& Module_Heff)
{
	if (threadIdx.x == 0) cuMesh.pDemag_Heff = &Module_Heff;
}

void DemagMCUDA::set_DemagCUDA_pointers(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		set_DemagCUDA_pointers_kernel <<< 1, CUDATHREADS >>>
			(pMeshCUDA->cuMesh.get_deviceobject(mGPU), Module_Heff.get_deviceobject(mGPU));
	}
}

#endif

#endif