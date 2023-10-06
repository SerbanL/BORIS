#include "Atom_ExchangeCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_EXCHANGE) && ATOMISTIC == 1

#include "BorisCUDALib.cuh"

#include "MeshCUDA.h"
#include "Atom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"
#include "MeshDefs.h"

// both contacting meshes are atomistic
__global__ void CalculateExchangeCoupling_Atom_kernel(
	ManagedAtom_MeshCUDA& mesh_sec, ManagedAtom_MeshCUDA& mesh_pri,
	CMBNDInfoCUDA& contact,
	cuBReal& energy, bool do_reduction)
{
	int box_idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	cuINT3 box_sizes = contact.cells_box.size();

	cuVEC_VC<cuReal3>& M1_pri = *mesh_pri.pM1;
	cuVEC<cuReal3>& Heff1_pri = *mesh_pri.pHeff1;
	mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1_sec = mesh_sec.pM1->mcuvec();

	if (box_idx < box_sizes.dim()) {

		int i = (box_idx % box_sizes.x) + contact.cells_box.s.i;
		int j = ((box_idx / box_sizes.x) % box_sizes.y) + contact.cells_box.s.j;
		int k = (box_idx / (box_sizes.x * box_sizes.y)) + contact.cells_box.s.k;

		cuBReal hRsq = contact.hshift_primary.norm();
		hRsq *= hRsq;

		int cell1_idx = i + j * M1_pri.n.x + k * M1_pri.n.x * M1_pri.n.y;

		if (M1_pri.is_not_empty(cell1_idx) && M1_pri.is_cmbnd(cell1_idx)) {

			//calculate second primary cell index
			int cell2_idx = (i + contact.cell_shift.i) + (j + contact.cell_shift.j) * M1_pri.n.x + (k + contact.cell_shift.k) * M1_pri.n.x * M1_pri.n.y;

			//relative position of cell -1 in secondary mesh
			cuReal3 relpos_m1 = M1_pri.rect.s - M1_sec.rect.s + ((cuReal3(i, j, k) + cuReal3(0.5)) & M1_pri.h) + (contact.hshift_primary + contact.hshift_secondary) / 2;

			//stencil is used for weighted_average to obtain values in the secondary mesh : has size equal to primary cellsize area on interface with thickness set by secondary cellsize thickness
			cuReal3 stencil = M1_pri.h - cu_mod(contact.hshift_primary) + cu_mod(contact.hshift_secondary);

			cuBReal mu_s = *mesh_pri.pmu_s;
			cuBReal J = *mesh_pri.pJ;
			mesh_pri.update_parameters_mcoarse(cell1_idx, *mesh_pri.pmu_s, mu_s, *mesh_pri.pJ, J);

			cuReal3 Hexch;

			//direction values at cells -1, 2
			cuReal3 m_2;
			if (cell2_idx < M1_pri.n.dim() && M1_pri.is_not_empty(cell2_idx)) m_2 = M1_pri[cell2_idx].normalized();

			cuReal3 m_m1 = M1_sec.weighted_average(relpos_m1, stencil);
			if (m_m1 != cuReal3()) m_m1 = m_m1.normalized();

			Hexch = (J / (MUB_MU0 * mu_s)) * (m_2 + m_m1);

			Heff1_pri[cell1_idx] += Hexch;

			if (do_reduction) {

				cuBReal non_empty_volume = M1_pri.get_nonempty_cells() * M1_pri.h.dim();
				if (non_empty_volume) energy_ = -(cuBReal)MUB_MU0 * M1_pri[cell1_idx] * Hexch / (2 * non_empty_volume);
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

// FM to Atomistic
__global__ void CalculateExchangeCoupling_FM_to_Atom_kernel(
	ManagedMeshCUDA& mesh_sec, ManagedAtom_MeshCUDA& mesh_pri,
	CMBNDInfoCUDA& contact,
	cuBReal& energy, bool do_reduction)
{
	int box_idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	cuINT3 box_sizes = contact.cells_box.size();

	cuVEC_VC<cuReal3>& M1_pri = *mesh_pri.pM1;
	cuVEC<cuReal3>& Heff1_pri = *mesh_pri.pHeff1;
	mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_sec = mesh_sec.pM->mcuvec();

	if (box_idx < box_sizes.dim()) {

		int i = (box_idx % box_sizes.x) + contact.cells_box.s.i;
		int j = ((box_idx / box_sizes.x) % box_sizes.y) + contact.cells_box.s.j;
		int k = (box_idx / (box_sizes.x * box_sizes.y)) + contact.cells_box.s.k;

		cuBReal hRsq = contact.hshift_primary.norm();
		hRsq *= hRsq;

		int cell1_idx = i + j * M1_pri.n.x + k * M1_pri.n.x * M1_pri.n.y;

		if (M1_pri.is_not_empty(cell1_idx) && M1_pri.is_cmbnd(cell1_idx)) {

			//calculate second primary cell index
			int cell2_idx = (i + contact.cell_shift.i) + (j + contact.cell_shift.j) * M1_pri.n.x + (k + contact.cell_shift.k) * M1_pri.n.x * M1_pri.n.y;

			//relative position of cell -1 in secondary mesh
			cuReal3 relpos_m1 = M1_pri.rect.s - M_sec.rect.s + ((cuReal3(i, j, k) + cuReal3(0.5)) & M1_pri.h) + (contact.hshift_primary + contact.hshift_secondary) / 2;

			cuBReal mu_s = *mesh_pri.pmu_s;
			cuBReal J = *mesh_pri.pJ;
			mesh_pri.update_parameters_mcoarse(cell1_idx, *mesh_pri.pmu_s, mu_s, *mesh_pri.pJ, J);

			cuReal3 Hexch;

			//direction values at cells -1, 2
			cuReal3 m_2;
			if (cell2_idx < M1_pri.n.dim() && M1_pri.is_not_empty(cell2_idx)) m_2 = M1_pri[cell2_idx].normalized();

			cuReal3 m_m1;
			if (!M_sec.is_empty(relpos_m1)) m_m1 = M_sec[relpos_m1].normalized();

			Hexch = (J / (MUB_MU0 * mu_s)) * (m_2 + m_m1);
			Heff1_pri[cell1_idx] += Hexch;

			if (do_reduction) {

				cuBReal non_empty_volume = M1_pri.get_nonempty_cells() * M1_pri.h.dim();
				if (non_empty_volume) energy_ = -(cuBReal)MUB_MU0 * M1_pri[cell1_idx] * Hexch / (2 * non_empty_volume);
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

//----------------------- CalculateExchangeCoupling LAUNCHER

//calculate exchange field at coupled cells in this mesh; accumulate energy density contribution in energy
void Atom_ExchangeCUDA::CalculateExchangeCoupling(mcu_val<cuBReal>& energy)
{
	for (int contact_idx = 0; contact_idx < CMBNDcontacts.size(); contact_idx++) {

		size_t size = CMBNDcontacts[contact_idx].cells_box.size().dim();

		//the contacting meshes indexes : secondary mesh index is the one in contact with this one (the primary)
		int idx_sec = CMBNDcontacts[contact_idx].mesh_idx.i;
		int idx_pri = CMBNDcontacts[contact_idx].mesh_idx.j;

		if (pContactingMeshes[idx_pri]->is_atomistic() && pContactingMeshes[idx_sec]->is_atomistic()) {

			//both meshes are atomistic

			if (pMeshBaseCUDA->CurrentTimeStepSolved()) {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					size_t size = CMBNDcontactsCUDA[contact_idx].contact_size(mGPU);
					if (!size) continue;

					CalculateExchangeCoupling_Atom_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
						(pContactingManagedAtomMeshes[idx_sec]->get_deviceobject(mGPU), pContactingManagedAtomMeshes[idx_pri]->get_deviceobject(mGPU), 
						CMBNDcontactsCUDA[contact_idx].get_deviceobject(mGPU), energy(mGPU), true);
				}
			}
			else {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					size_t size = CMBNDcontactsCUDA[contact_idx].contact_size(mGPU);
					if (!size) continue;

					CalculateExchangeCoupling_Atom_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
						(pContactingManagedAtomMeshes[idx_sec]->get_deviceobject(mGPU), pContactingManagedAtomMeshes[idx_pri]->get_deviceobject(mGPU), 
						CMBNDcontactsCUDA[contact_idx].get_deviceobject(mGPU), energy(mGPU), false);
				}
			}
		}

		else if (pContactingMeshes[idx_pri]->is_atomistic() && pContactingMeshes[idx_sec]->GetMeshType() == MESH_FERROMAGNETIC) {

			//atomistic to micromagnetic mesh coupling (here the secondary mesh will be ferromagnetic, as the primary must be atomistic)

			//FM to atomistic

			if (pMeshBaseCUDA->CurrentTimeStepSolved()) {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					size_t size = CMBNDcontactsCUDA[contact_idx].contact_size(mGPU);
					if (!size) continue;

					CalculateExchangeCoupling_FM_to_Atom_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
						(pContactingManagedMeshes[idx_sec]->get_deviceobject(mGPU), pContactingManagedAtomMeshes[idx_pri]->get_deviceobject(mGPU), 
						CMBNDcontactsCUDA[contact_idx].get_deviceobject(mGPU), energy(mGPU), true);
				}
			}
			else {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					size_t size = CMBNDcontactsCUDA[contact_idx].contact_size(mGPU);
					if (!size) continue;

					CalculateExchangeCoupling_FM_to_Atom_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
						(pContactingManagedMeshes[idx_sec]->get_deviceobject(mGPU), pContactingManagedAtomMeshes[idx_pri]->get_deviceobject(mGPU), 
						CMBNDcontactsCUDA[contact_idx].get_deviceobject(mGPU), energy(mGPU), false);
				}
			}
		}
	}
}

#endif

#endif