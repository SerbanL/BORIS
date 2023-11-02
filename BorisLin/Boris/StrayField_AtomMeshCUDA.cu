#include "StrayField_AtomMeshCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_STRAYFIELD) && ATOMISTIC == 1

#include "Reduction.cuh"

#include "Atom_MeshCUDA.h"
#include "MeshDefs.h"

//----------------------- Initialization

__global__ void set_StrayField_AtomMeshCUDA_pointers_kernel(
	ManagedAtom_MeshCUDA& cuaMesh, cuVEC<cuReal3>& strayField)
{
	if (threadIdx.x == 0) cuaMesh.pstrayField = &strayField;
}

void StrayField_AtomMeshCUDA::set_StrayField_AtomMeshCUDA_pointers(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		set_StrayField_AtomMeshCUDA_pointers_kernel <<< 1, CUDATHREADS >>>
			(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), strayField.get_deviceobject(mGPU));
	}
}

//----------------------- Computation

__global__ void UpdateStrayField_ASC_kernel(ManagedAtom_MeshCUDA& cuaMesh, ManagedModulesCUDA& cuModule, cuVEC<cuReal3>& strayField, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff1.linear_size()) {

		cuReal3 Hstray = strayField[idx];

		Heff1[idx] += Hstray;

		if (do_reduction) {

			int non_empty_cells = M1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MUB_MU0 * M1[idx] * Hstray / (non_empty_cells * M1.h.dim());
		}

		if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hstray;
		if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MUB_MU0 * M1[idx] * Hstray / M1.h.dim();
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

void StrayField_AtomMeshCUDA::UpdateFieldCUDA(void)
{
	if (paMeshCUDA->CurrentTimeStepSolved()) {

		ZeroEnergy();

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			UpdateStrayField_ASC_kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), strayField.get_deviceobject(mGPU), true);
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			UpdateStrayField_ASC_kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), strayField.get_deviceobject(mGPU), false);
		}
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && ATOMISTIC == 1 && MONTE_CARLO == 1

__device__ cuBReal ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_StrayField_AtomMeshCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M1 = *pM1;

	cuReal3 Hstray = cuReal3();

	if (pstrayField && pstrayField->linear_size()) {

		Hstray = (*pstrayField)[spin_index];
	}

	if (Mnew != cuReal3()) return -(cuBReal)MUB_MU0 * (Mnew - M1[spin_index]) * Hstray;
	else return -(cuBReal)MUB_MU0 * M1[spin_index] * Hstray;
}

#endif