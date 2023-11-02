#include "SDemagCUDA_Demag.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SDEMAG

#include "MeshCUDA.h"
#include "Atom_MeshCUDA.h"

//----------------------- Initialization

__global__ void set_SDemag_DemagCUDA_pointers_kernel(
	ManagedMeshCUDA& cuMesh, cuVEC<cuReal3>& Module_Heff)
{
	if (threadIdx.x == 0) cuMesh.pDemag_Heff = &Module_Heff;
}

__global__ void set_SDemag_DemagCUDA_pointers_atomistic_kernel(
	ManagedAtom_MeshCUDA& cuaMesh, cuVEC<cuReal3>& Module_Heff)
{
	if (threadIdx.x == 0) cuaMesh.pAtom_Demag_Heff = &Module_Heff;
}

void SDemagCUDA_Demag::set_SDemag_DemagCUDA_pointers(void)
{
	if (pMeshCUDA) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			set_SDemag_DemagCUDA_pointers_kernel <<< 1, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU), Module_Heff.get_deviceobject(mGPU));
		}
	}
	else if (paMeshCUDA) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			set_SDemag_DemagCUDA_pointers_atomistic_kernel <<< 1, CUDATHREADS >>>
				(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), Module_Heff.get_deviceobject(mGPU));
		}
	}
}

//-------------------Getters

__global__ void Add_Energy_Kernel(cuBReal& energy, cuBReal& total_energy)
{
	if (threadIdx.x == 0) total_energy += energy;
}

//add energy in this module to a running total
void SDemagCUDA_Demag::Add_Energy(mcu_val<cuBReal>& total_energy)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Add_Energy_Kernel <<< (1 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(energy(mGPU), total_energy(mGPU));
	}
}

#endif

#endif