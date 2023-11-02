#include "DemagMCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_DEMAG

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

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && MONTE_CARLO == 1

//Ferromagnetic
__device__ cuBReal ManagedMeshCUDA::Get_EnergyChange_FM_DemagCUDA(int spin_index, cuReal3 Mnew)
{
	if (pDemag_Heff && pDemag_Heff->linear_size()) {

		cuVEC_VC<cuReal3>& M = *pM;

		if (Mnew != cuReal3()) return -(cuBReal)MU0 * M.h.dim() * (*pDemag_Heff)[M.cellidx_to_position(spin_index)] * (Mnew - M[spin_index]);
		else return -(cuBReal)MU0 * M.h.dim() * (*pDemag_Heff)[M.cellidx_to_position(spin_index)] * M[spin_index];
	}
	else return 0.0;
}

//Antiferromagnetic
__device__ cuReal2 ManagedMeshCUDA::Get_EnergyChange_AFM_DemagCUDA(int spin_index, cuReal3 Mnew_A, cuReal3 Mnew_B)
{
	//Module_Heff needs to be calculated (done during a Monte Carlo simulation, where this method would be used)
	if (pDemag_Heff && pDemag_Heff->linear_size()) {

		cuVEC_VC<cuReal3>& Mvec = *pM;
		cuVEC_VC<cuReal3>& Mvec2 = *pM2;

		cuReal3 M = (Mvec[spin_index] + Mvec2[spin_index]) / 2;
		cuReal3 Mnew = (Mnew_A + Mnew_B) / 2;

		cuBReal energy_ = 0.0;

		//do not divide by 2 as we are not double-counting here
		if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) {

			energy_ = -(cuBReal)MU0 * Mvec.h.dim() * (*pDemag_Heff)[Mvec.cellidx_to_position(spin_index)] * (Mnew - M);
		}
		else {

			energy_ = -(cuBReal)MU0 * Mvec.h.dim() * (*pDemag_Heff)[Mvec.cellidx_to_position(spin_index)] * M;
		}

		return cuReal2(energy_, energy_);
	}
	else return cuReal2();
}

#endif