#include "Atom_Demag_NCUDA.h"
#include "Atom_MeshCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_DEMAG_N) && ATOMISTIC == 1

#include "Reduction.cuh"

#include "MeshDefs.h"

__global__ void Demag_NCUDA_Cubic_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;
	MatPCUDA<cuReal2, cuBReal>& Nxy = *cuaMesh.pNxy;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff1.linear_size()) {

		//used to convert moment to magnetization in each atomistic unit cell
		cuBReal conversion = (cuBReal)MUB / M1.h.dim();

		cuReal3 Heff_value = cuReal3();

		if (M1.is_not_empty(idx)) {

			Heff_value = cuReal3(-cuReal2(Nxy).x * M1[idx].x, -cuReal2(Nxy).y * M1[idx].y, -(1 - cuReal2(Nxy).x - cuReal2(Nxy).y) * M1[idx].z) * conversion;

			if (do_reduction) {

				int non_empty_cells = M1.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * conversion * M1[idx] * Heff_value / (2 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MUB_MU0 * M1[idx] * Heff_value / (2 * M1.h.dim());
		}

		Heff1[idx] += Heff_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Atom_Demag_NCUDA::UpdateField(void)
{
	if (paMeshCUDA->CurrentTimeStepSolved()) {

		ZeroEnergy();

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Demag_NCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Demag_NCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
		}
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && ATOMISTIC == 1 && MONTE_CARLO == 1

__device__ cuBReal ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_DemagNCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M1 = *pM1;

	//Nxy shouldn't have a temperature (or spatial) dependence so not using update_parameters_mcoarse here
	cuReal2 Nxy = *pNxy;

	cuBReal Nz = (1 - Nxy.x - Nxy.y);

	cuBReal r = (cuBReal)MUB / M1.h.dim();
	cuReal3 S = M1[spin_index];

	if (Mnew != cuReal3()) return ((cuBReal)MUB_MU0 / 2) * r * (Nxy.x * (Mnew.x * Mnew.x - S.x * S.x) + Nxy.y * (Mnew.y * Mnew.y - S.y * S.y) + Nz * (Mnew.z * Mnew.z - S.z * S.z));
	else return ((cuBReal)MUB_MU0 / 2) * r * (Nxy.x * S.x * S.x + Nxy.y * S.y * S.y + Nz * S.z * S.z);
}

#endif