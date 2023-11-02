#include "Atom_AnisotropyBiaxialCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_ANIBI) && ATOMISTIC == 1

#include "Reduction.cuh"

#include "Atom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"
#include "MeshDefs.h"

__global__ void Atom_Anisotropy_BiaxialCUDA_Cubic_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff1.linear_size()) {

		cuReal3 Heff_value = cuReal3();

		if (M1.is_not_empty(idx)) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal K1 = *cuaMesh.pK1;
			cuBReal K2 = *cuaMesh.pK2;
			cuReal3 mcanis_ea1 = *cuaMesh.pmcanis_ea1;
			cuReal3 mcanis_ea2 = *cuaMesh.pmcanis_ea2;
			cuReal3 mcanis_ea3 = *cuaMesh.pmcanis_ea3;
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.pK1, K1, *cuaMesh.pK2, K2, *cuaMesh.pmcanis_ea1, mcanis_ea1, *cuaMesh.pmcanis_ea2, mcanis_ea2, *cuaMesh.pmcanis_ea3, mcanis_ea3);

			//calculate m.ea1 dot product (uniaxial contribution)
			cuBReal u1 = (M1[idx] * mcanis_ea1) / mu_s;

			//calculate m.ea2 and m.ea3 dot products (biaxial contribution)
			cuBReal b1 = (M1[idx] * mcanis_ea2) / mu_s;
			cuBReal b2 = (M1[idx] * mcanis_ea3) / mu_s;

			//update effective field with the anisotropy field
			Heff_value = (2 / ((cuBReal)MUB_MU0*mu_s)) * (K1 * u1 * mcanis_ea1 - K2 * (b1*b2*b2 * mcanis_ea2 + b1*b1*b2 * mcanis_ea3));

			if (do_reduction) {

				//update energy density
				cuBReal non_empty_volume = M1.get_nonempty_cells() * M1.h.dim();
				if (non_empty_volume) energy_ = (K1 * (1 - u1*u1) + K2 * b1*b1*b2*b2) / non_empty_volume;
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = (K1 * (1 - u1*u1) + K2 * b1*b1*b2*b2) / M1.h.dim();
		}

		Heff1[idx] += Heff_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Atom_Anisotropy_BiaxialCUDA::UpdateField(void)
{
	if (paMeshCUDA->GetMeshType() == MESH_ATOM_CUBIC) {

		//atomistic simple-cubic mesh

		if (paMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Atom_Anisotropy_BiaxialCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Atom_Anisotropy_BiaxialCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && ATOMISTIC == 1 && MONTE_CARLO == 1

__device__ cuBReal ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_AnisotropyBiaxialCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M1 = *pM1;

	cuBReal K1 = *pK1;
	cuBReal K2 = *pK2;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	cuReal3 mcanis_ea2 = *pmcanis_ea2;
	cuReal3 mcanis_ea3 = *pmcanis_ea3;
	update_parameters_mcoarse(spin_index, *pK1, K1, *pK2, K2, *pmcanis_ea1, mcanis_ea1, *pmcanis_ea2, mcanis_ea2, *pmcanis_ea3, mcanis_ea3);

	//calculate m.ea1 dot product (uniaxial contribution)
	cuBReal u1 = M1[spin_index].normalized() * mcanis_ea1;

	//calculate m.ea2 and m.ea3 dot products (biaxial contribution)
	cuBReal b1 = M1[spin_index].normalized() * mcanis_ea2;
	cuBReal b2 = M1[spin_index].normalized() * mcanis_ea3;

	if (Mnew != cuReal3()) {

		//calculate m.ea1 dot product (uniaxial contribution)
		cuBReal u1new = Mnew.normalized() * mcanis_ea1;

		//calculate m.ea2 and m.ea3 dot products (biaxial contribution)
		cuBReal b1new = Mnew.normalized() * mcanis_ea2;
		cuBReal b2new = Mnew.normalized() * mcanis_ea3;

		return (K1 * (1 - u1new * u1new) + K2 * b1new * b1new * b2new * b2new) - (K1 * (1 - u1 * u1) + K2 * b1 * b1 * b2 * b2);
	}
	else return K1 * (1 - u1 * u1) + K2 * b1 * b1 * b2 * b2;
}

#endif