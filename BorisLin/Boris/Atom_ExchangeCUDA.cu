#include "Atom_ExchangeCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_EXCHANGE) && ATOMISTIC == 1

#include "Reduction.cuh"
#include "mcuVEC_halo.cuh"

#include "Atom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"
#include "MeshDefs.h"

//////////////////////////////////////////////////////////////////////// UPDATE FIELD

__global__ void Atom_ExchangeCUDA_Cubic_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff1.linear_size()) {

		cuReal3 Hexch = cuReal3();

		if (M1.is_not_empty(idx)) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal J = *cuaMesh.pJ;
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.pJ, J);

			//update effective field with the Heisenberg exchange field
			//if (idx % M1.n.x < M1.n.x - 1 && idx % M1.n.x > 0)
			Hexch = (J / (MUB_MU0*mu_s)) * M1.ngbr_dirsum(idx);

			if (do_reduction) {

				//energy E = -mu_s * Bex
				//update energy density
				cuBReal non_empty_volume = M1.get_nonempty_cells() * M1.h.dim();
				if (non_empty_volume) energy_ = -(cuBReal)MUB_MU0 * M1[idx] * Hexch / (2*non_empty_volume);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hexch;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MUB_MU0 * M1[idx] * Hexch / (2 * M1.h.dim());
		}

		Heff1[idx] += Hexch;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Atom_ExchangeCUDA::UpdateField(void)
{
	if (paMeshCUDA->GetMeshType() == MESH_ATOM_CUBIC) {

		//atomistic simple cubic mesh

		paMeshCUDA->M1.exchange_halos();

		if (paMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Atom_ExchangeCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Atom_ExchangeCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}

	//if using UVA to compute differential operators then synchronization needed now
	//otherwise a kernel on a device could finish and continue on to diff eq update (which will update M on device), whilst neighboring devices are still accessing these data - data race!
	//if using halo exchanges instead this is not a problem
	mGPU.synchronize_if_uva();

	if (paMeshCUDA->GetMeshExchangeCoupling()) CalculateExchangeCoupling(energy);
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && ATOMISTIC == 1 && MONTE_CARLO == 1

__device__ cuBReal ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_ExchangeCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M1 = *pM1;

	cuBReal J = *pJ;
	update_parameters_mcoarse(spin_index, *pJ, J);

	if (Mnew != cuReal3()) return -J * ((Mnew.normalized() - M1[spin_index].normalized()) * M1.ngbr_dirsum(spin_index));
	else return -J * (M1[spin_index].normalized() * M1.ngbr_dirsum(spin_index));
}

#endif