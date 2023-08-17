#include "Atom_viDMExchangeCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_VIDMEXCHANGE) && ATOMISTIC == 1

#include "BorisCUDALib.cuh"

#include "Atom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"
#include "MeshDefs.h"
#include "ModulesDefs.h"

//////////////////////////////////////////////////////////////////////// UPDATE FIELD

__global__ void Atom_viDMExchangeCUDA_Cubic_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedModulesCUDA& cuModule, bool do_reduction, bool showtotalenergy)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff1.linear_size()) {

		cuReal3 Hexch_A = cuReal3(), Hexch_D = cuReal3();

		if (M1.is_not_empty(idx)) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal J = *cuaMesh.pJ;
			cuBReal D = *cuaMesh.pD;
			cuReal3 D_dir = *cuaMesh.pD_dir;
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.pJ, J, *cuaMesh.pD, D, *cuaMesh.pD_dir, D_dir);

			//update effective field with the Heisenberg and DMI exchange field
			Hexch_A = J * M1.ngbr_dirsum(idx) / ((cuBReal)MUB_MU0*mu_s);

			cuReal3 hexch_D_x = M1.xanisotropic_ngbr_dirsum(idx);
			cuReal3 hexch_D_y = M1.yanisotropic_ngbr_dirsum(idx);
			cuReal3 hexch_D_z = M1.zanisotropic_ngbr_dirsum(idx);

			Hexch_D = D * (D_dir.x * hexch_D_x + D_dir.y * hexch_D_y + D_dir.z * hexch_D_z) / ((cuBReal)MUB_MU0*mu_s);

			if (do_reduction) {

				//energy E = -mu_s * Bex
				//update energy density
				cuBReal non_empty_volume = M1.get_nonempty_cells() * M1.h.dim();
				if (non_empty_volume) energy_ = -(cuBReal)MUB_MU0 * M1[idx] * (Hexch_A + Hexch_D) / (2 * non_empty_volume);
			}

			//spatial dependence display of effective field and energy density
			if (do_reduction && cuModule.pModule_Heff->linear_size() && cuModule.pModule_energy->linear_size()) {

				if (showtotalenergy) {

					//total : direct and DMI
					(*cuModule.pModule_Heff)[idx] = Hexch_A + Hexch_D;
					(*cuModule.pModule_energy)[idx] = -(cuBReal)MUB_MU0 * M1[idx] * (Hexch_A + Hexch_D) / (2 * M1.h.dim());
				}
				else {

					//just DMI
					(*cuModule.pModule_Heff)[idx] = Hexch_D;
					(*cuModule.pModule_energy)[idx] = -(cuBReal)MUB_MU0 * M1[idx] * Hexch_D / (2 * M1.h.dim());
				}
			}
		}

		Heff1[idx] += (Hexch_A + Hexch_D);
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Atom_viDMExchangeCUDA::UpdateField(void)
{
	if (paMeshCUDA->GetMeshType() == MESH_ATOM_CUBIC) {

		//atomistic simple cubic mesh

		paMeshCUDA->M1.exchange_halos();

		if (paMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Atom_viDMExchangeCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true, (MOD_)paMeshCUDA->Get_Module_Heff_Display() == MOD_EXCHANGE);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Atom_viDMExchangeCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false, (MOD_)paMeshCUDA->Get_Module_Heff_Display() == MOD_EXCHANGE);
			}
		}
	}

	//if using UVA to compute differential operators then synchronization needed now
	//otherwise a kernel on a device could finish and continue on to diff eq update (which will update M on device), whilst neighboring devices are still accessing these data - data race!
	//if using halo exchanges instead this is not a problem
	mGPU.synchronize_if_uva();
}

#endif

#endif