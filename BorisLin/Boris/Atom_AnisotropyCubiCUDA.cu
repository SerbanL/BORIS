#include "Atom_AnisotropyCubiCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_ANICUBI) && ATOMISTIC == 1

#include "Reduction.cuh"

#include "Atom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"
#include "MeshDefs.h"

__global__ void Atom_Anisotropy_CubiCUDA_Cubic_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
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

			//calculate m.ea1, m.ea2 and m.ea3 dot products
			cuBReal d1 = (M1[idx] * mcanis_ea1) / mu_s;
			cuBReal d2 = (M1[idx] * mcanis_ea2) / mu_s;
			cuBReal d3 = (M1[idx] * mcanis_ea3) / mu_s;

			//terms for K1 contribution
			cuBReal a1 = d1 * (d2*d2 + d3*d3);
			cuBReal a2 = d2 * (d1*d1 + d3*d3);
			cuBReal a3 = d3 * (d1*d1 + d2*d2);

			//terms for K2 contribution
			cuBReal d123 = d1*d2*d3;

			cuBReal b1 = d123 * d2*d3;
			cuBReal b2 = d123 * d1*d3;
			cuBReal b3 = d123 * d1*d2;

			//update effective field with the anisotropy field
			Heff_value = cuReal3(
				(-2 * K1 / ((cuBReal)MUB_MU0*mu_s)) * (mcanis_ea1.i * a1 + mcanis_ea2.i * a2 + mcanis_ea3.i * a3)
				+ (-2 * K2 / ((cuBReal)MUB_MU0*mu_s)) * (mcanis_ea1.i * b1 + mcanis_ea2.i * b2 + mcanis_ea3.i * b3),
				(-2 * K1 / ((cuBReal)MUB_MU0*mu_s)) * (mcanis_ea1.j * a1 + mcanis_ea2.j * a2 + mcanis_ea3.j * a3)
				+ (-2 * K2 / ((cuBReal)MUB_MU0*mu_s)) * (mcanis_ea1.j * b1 + mcanis_ea2.j * b2 + mcanis_ea3.j * b3),
				(-2 * K1 / ((cuBReal)MUB_MU0*mu_s)) * (mcanis_ea1.k * a1 + mcanis_ea2.k * a2 + mcanis_ea3.k * a3)
				+ (-2 * K2 / ((cuBReal)MUB_MU0*mu_s)) * (mcanis_ea1.k * b1 + mcanis_ea2.k * b2 + mcanis_ea3.k * b3)
			);

			if (do_reduction) {

				//update energy density
				cuBReal non_empty_volume = M1.get_nonempty_cells() * M1.h.dim();
				if (non_empty_volume) energy_ = (K1 * (d1*d1*d2*d2 + d1*d1*d3*d3 + d2*d2*d3*d3) + K2 * d123*d123) / non_empty_volume;
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = (K1 * (d1*d1*d2*d2 + d1*d1*d3*d3 + d2*d2*d3*d3) + K2 * d123*d123) / M1.h.dim();
		}

		Heff1[idx] += Heff_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Atom_Anisotropy_CubiCUDA::UpdateField(void)
{
	if (paMeshCUDA->GetMeshType() == MESH_ATOM_CUBIC) {

		//atomistic simple-cubic mesh

		if (paMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Atom_Anisotropy_CubiCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Atom_Anisotropy_CubiCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && ATOMISTIC == 1 && MONTE_CARLO == 1

__device__ cuBReal ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_AnisotropyCubiCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M1 = *pM1;

	cuBReal K1 = *pK1;
	cuBReal K2 = *pK2;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	cuReal3 mcanis_ea2 = *pmcanis_ea2;
	cuReal3 mcanis_ea3 = *pmcanis_ea3;
	update_parameters_mcoarse(spin_index, *pK1, K1, *pK2, K2, *pmcanis_ea1, mcanis_ea1, *pmcanis_ea2, mcanis_ea2, *pmcanis_ea3, mcanis_ea3);

	cuReal3 S = M1[spin_index].normalized();
	cuReal3 Snew = Mnew.normalized();

	//calculate m.ea1, m.ea2 and m.ea3 dot products
	cuBReal d1 = S * mcanis_ea1;
	cuBReal d2 = S * mcanis_ea2;
	cuBReal d3 = S * mcanis_ea3;

	if (Mnew != cuReal3()) {

		cuBReal d1_new = Snew * mcanis_ea1;
		cuBReal d2_new = Snew * mcanis_ea2;
		cuBReal d3_new = Snew * mcanis_ea3;

		//Hamiltonian contribution as K * (Sx^2*Sy^2 + Sx^2*Sz^2 + Sy^2*Sz^2), where S is the local spin direction (for easy axes coinciding with the xyz system)
		//This is equivalent to the form -K/2 * (Sx^4 + Sy^4 + Sz^4) - energy zero point differs but that's immaterial.
		//Also note the correct signs here for given easy axes (need to be careful, some publications have this wrong).
		return K1 * (d1_new * d1_new * d2_new * d2_new + d1_new * d1_new * d3_new * d3_new + d2_new * d2_new * d3_new * d3_new - d1 * d1 * d2 * d2 - d1 * d1 * d3 * d3 - d2 * d2 * d3 * d3)
			+ K2 * (d1_new * d2_new * d3_new * d1_new * d2_new * d3_new - d1 * d2 * d3 * d1 * d2 * d3);
	}
	else return K1 * (d1 * d1 * d2 * d2 + d1 * d1 * d3 * d3 + d2 * d2 * d3 * d3) + K2 * d1 * d2 * d3 * d1 * d2 * d3;
}

#endif