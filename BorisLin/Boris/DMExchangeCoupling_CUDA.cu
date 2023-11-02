#include "DMExchangeCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_DMEXCHANGE

#include "Reduction.cuh"
#include "mcuVEC_Managed_avg.cuh"
#include "cuVEC_VC_mcuVEC.cuh"

#include "MeshCUDA.h"
#include "Atom_MeshCUDA.h"
#include "MeshParamsControlCUDA.h"
#include "MeshDefs.h"

// both contacting meshes are ferromagnetic
__global__ void CalculateDMExchangeCoupling_FM_kernel(
	ManagedMeshCUDA& mesh_sec, ManagedMeshCUDA& mesh_pri,
	CMBNDInfoCUDA& contact,
	cuBReal& energy, bool do_reduction)
{
	int box_idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	cuINT3 box_sizes = contact.cells_box.size();

	cuVEC_VC<cuReal3>& M_pri = *mesh_pri.pM;
	cuVEC<cuReal3>& Heff_pri = *mesh_pri.pHeff;
	mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_sec = mesh_sec.pM->mcuvec();

	if (box_idx < box_sizes.dim()) {

		int i = (box_idx % box_sizes.x) + contact.cells_box.s.i;
		int j = ((box_idx / box_sizes.x) % box_sizes.y) + contact.cells_box.s.j;
		int k = (box_idx / (box_sizes.x * box_sizes.y)) + contact.cells_box.s.k;

		cuReal3 hshift_primary = contact.hshift_primary;
		cuBReal hRsq = hshift_primary.norm();
		hRsq *= hRsq;

		int cell1_idx = i + j * M_pri.n.x + k * M_pri.n.x * M_pri.n.y;

		if (M_pri.is_not_empty(cell1_idx) && M_pri.is_cmbnd(cell1_idx)) {

			//calculate second primary cell index
			int cell2_idx = (i + contact.cell_shift.i) + (j + contact.cell_shift.j) * M_pri.n.x + (k + contact.cell_shift.k) * M_pri.n.x * M_pri.n.y;

			//relative position of cell -1 in secondary mesh
			cuReal3 relpos_m1 = M_pri.rect.s - M_sec.rect.s + ((cuReal3(i, j, k) + cuReal3(0.5)) & M_pri.h) + (contact.hshift_primary + contact.hshift_secondary) / 2;

			//stencil is used for weighted_average to obtain values in the secondary mesh : has size equal to primary cellsize area on interface with thickness set by secondary cellsize thickness
			cuReal3 stencil = M_pri.h - cu_mod(contact.hshift_primary) + cu_mod(contact.hshift_secondary);

			cuBReal Ms = *mesh_pri.pMs;
			cuBReal A = *mesh_pri.pA;
			cuBReal D = *mesh_pri.pD;
			mesh_pri.update_parameters_mcoarse(cell1_idx, *mesh_pri.pA, A, *mesh_pri.pD, D, *mesh_pri.pMs, Ms);

			cuReal3 Hexch;

			//values at cells -1, 1
			cuReal3 M_1 = M_pri[cell1_idx];
			cuReal3 M_m1 = M_sec.weighted_average(relpos_m1, stencil);

			if (cell2_idx < M_pri.n.dim() && M_pri.is_not_empty(cell2_idx)) {

				//cell2_idx is valid and M is not empty there
				cuReal3 M_2 = M_pri[cell2_idx];

				//set effective field value contribution at cell 1 : direct exchange coupling
				Hexch = (2 * A / (MU0 * Ms * Ms)) * (M_2 + M_m1 - 2 * M_1) / hRsq;

				//add DMI contributions at CMBND cells, correcting for the sided differentials already applied here
				//the contributions are different depending on the CMBND coupling direction
				if (cuIsNZ(hshift_primary.x)) {

					//along x
					Hexch += (-2 * D / (MU0 * Ms * Ms)) * cuReal3(0, -M_2.z - M_m1.z + 2 * M_1.z, M_2.y - M_m1.y + 2 * M_1.y) / (2 * hshift_primary.x);
				}
				else if (cuIsNZ(hshift_primary.y)) {

					//along y
					Hexch += (-2 * D / (MU0 * Ms * Ms)) * cuReal3(M_2.z + M_m1.z - 2 * M_1.z, 0, -M_2.x - M_m1.x + 2 * M_1.x) / (2 * hshift_primary.y);
				}
				else if (cuIsNZ(hshift_primary.z)) {

					//along z
					Hexch += (-2 * D / (MU0 * Ms * Ms)) * cuReal3(-M_2.y - M_m1.y + 2 * M_1.y, M_2.x + M_m1.x - 2 * M_1.x, 0) / (2 * hshift_primary.z);
				}
			}
			else {

				//set effective field value contribution at cell 1 : direct exchange coupling
				Hexch = (2 * A / (MU0 * Ms * Ms)) * (M_m1 - M_1) / hRsq;
			}

			Heff_pri[cell1_idx] += Hexch;

			if (do_reduction) {

				int non_empty_cells = M_pri.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * M_1 * Hexch / (2 * non_empty_cells);
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

// both contacting meshes are antiferromagnetic
__global__ void CalculateDMExchangeCoupling_AFM_kernel(
	ManagedMeshCUDA& mesh_sec, ManagedMeshCUDA& mesh_pri,
	CMBNDInfoCUDA& contact,
	cuBReal& energy, bool do_reduction)
{
	int box_idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	cuINT3 box_sizes = contact.cells_box.size();

	cuVEC_VC<cuReal3>& M_pri = *mesh_pri.pM;
	cuVEC_VC<cuReal3>& M2_pri = *mesh_pri.pM2;

	cuVEC<cuReal3>& Heff_pri = *mesh_pri.pHeff;
	cuVEC<cuReal3>& Heff2_pri = *mesh_pri.pHeff2;

	mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_sec = mesh_sec.pM->mcuvec();
	mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2_sec = mesh_sec.pM2->mcuvec();

	if (box_idx < box_sizes.dim()) {

		int i = (box_idx % box_sizes.x) + contact.cells_box.s.i;
		int j = ((box_idx / box_sizes.x) % box_sizes.y) + contact.cells_box.s.j;
		int k = (box_idx / (box_sizes.x * box_sizes.y)) + contact.cells_box.s.k;

		cuReal3 hshift_primary = contact.hshift_primary;
		cuBReal hRsq = hshift_primary.norm();
		hRsq *= hRsq;

		int cell1_idx = i + j * M_pri.n.x + k * M_pri.n.x * M_pri.n.y;

		if (M_pri.is_not_empty(cell1_idx) && M_pri.is_cmbnd(cell1_idx)) {

			//calculate second primary cell index
			int cell2_idx = (i + contact.cell_shift.i) + (j + contact.cell_shift.j) * M_pri.n.x + (k + contact.cell_shift.k) * M_pri.n.x * M_pri.n.y;

			//relative position of cell -1 in secondary mesh
			cuReal3 relpos_m1 = M_pri.rect.s - M_sec.rect.s + ((cuReal3(i, j, k) + cuReal3(0.5)) & M_pri.h) + (contact.hshift_primary + contact.hshift_secondary) / 2;

			//stencil is used for weighted_average to obtain values in the secondary mesh : has size equal to primary cellsize area on interface with thickness set by secondary cellsize thickness
			cuReal3 stencil = M_pri.h - cu_mod(contact.hshift_primary) + cu_mod(contact.hshift_secondary);

			cuReal2 Ms_AFM = *mesh_pri.pMs_AFM;
			cuReal2 A_AFM = *mesh_pri.pA_AFM;
			cuReal2 D_AFM = *mesh_pri.pD_AFM;
			cuReal2 Anh = *mesh_pri.pAnh;
			mesh_pri.update_parameters_mcoarse(cell1_idx, *mesh_pri.pA_AFM, A_AFM, *mesh_pri.pD_AFM, D_AFM, *mesh_pri.pMs_AFM, Ms_AFM, *mesh_pri.pAnh, Anh);

			cuReal3 Hexch, Hexch_B;

			//values at cells -1, 1
			cuReal3 M_1 = M_pri[cell1_idx];
			cuReal3 M_m1 = M_sec.weighted_average(relpos_m1, stencil);

			cuReal3 M_1_B = M2_pri[cell1_idx];
			cuReal3 M_m1_B = M2_sec.weighted_average(relpos_m1, stencil);

			if (cell2_idx < M_pri.n.dim() && M_pri.is_not_empty(cell2_idx)) {

				//cell2_idx is valid and M is not empty there
				cuReal3 M_2 = M_pri[cell2_idx];
				cuReal3 M_2_B = M2_pri[cell2_idx];

				cuReal3 delsq_M_A = (M_2 + M_m1 - 2 * M_1) / hRsq;
				cuReal3 delsq_M_B = (M_2_B + M_m1_B - 2 * M_1_B) / hRsq;

				//set effective field value contribution at cell 1 : direct exchange coupling
				Hexch = (2 * A_AFM.i / ((cuBReal)MU0 * Ms_AFM.i * Ms_AFM.i)) * delsq_M_A + (Anh.i / ((cuBReal)MU0 * Ms_AFM.i * Ms_AFM.j)) * delsq_M_B;
				Hexch_B = (2 * A_AFM.j / ((cuBReal)MU0 * Ms_AFM.j * Ms_AFM.j)) * delsq_M_B + (Anh.j / ((cuBReal)MU0 * Ms_AFM.i * Ms_AFM.j)) * delsq_M_A;

				//add DMI contributions at CMBND cells, correcting for the sided differentials already applied here
				//the contributions are different depending on the CMBND coupling direction
				if (cuIsNZ(hshift_primary.x)) {

					//along x
					Hexch += (-2 * D_AFM.i / (MU0 * Ms_AFM.i * Ms_AFM.i)) * cuReal3(0, -M_2.z - M_m1.z + 2 * M_1.z, M_2.y - M_m1.y + 2 * M_1.y) / (2 * hshift_primary.x);
					Hexch_B += (-2 * D_AFM.j / (MU0 * Ms_AFM.j * Ms_AFM.j)) * cuReal3(0, -M_2_B.z - M_m1_B.z + 2 * M_1_B.z, M_2_B.y - M_m1_B.y + 2 * M_1_B.y) / (2 * hshift_primary.x);
				}
				else if (cuIsNZ(hshift_primary.y)) {

					//along y
					Hexch += (-2 * D_AFM.i / (MU0 * Ms_AFM.i * Ms_AFM.i)) * cuReal3(M_2.z + M_m1.z - 2 * M_1.z, 0, -M_2.x - M_m1.x + 2 * M_1.x) / (2 * hshift_primary.y);
					Hexch_B += (-2 * D_AFM.j / (MU0 * Ms_AFM.j * Ms_AFM.j)) * cuReal3(M_2_B.z + M_m1_B.z - 2 * M_1_B.z, 0, -M_2_B.x - M_m1_B.x + 2 * M_1_B.x) / (2 * hshift_primary.y);
				}
				else if (cuIsNZ(hshift_primary.z)) {

					//along z
					Hexch += (-2 * D_AFM.i / (MU0 * Ms_AFM.i * Ms_AFM.i)) * cuReal3(-M_2.y - M_m1.y + 2 * M_1.y, M_2.x + M_m1.x - 2 * M_1.x, 0) / (2 * hshift_primary.z);
					Hexch_B += (-2 * D_AFM.j / (MU0 * Ms_AFM.j * Ms_AFM.j)) * cuReal3(-M_2_B.y - M_m1_B.y + 2 * M_1_B.y, M_2_B.x + M_m1_B.x - 2 * M_1_B.x, 0) / (2 * hshift_primary.z);
				}
			}
			else {

				cuReal3 delsq_M_A = (M_m1 - M_1) / hRsq;
				cuReal3 delsq_M_B = (M_m1_B - M_1_B) / hRsq;

				//set effective field value contribution at cell 1 : direct exchange coupling
				Hexch = (2 * A_AFM.i / ((cuBReal)MU0 * Ms_AFM.i * Ms_AFM.i)) * delsq_M_A + (Anh.i / ((cuBReal)MU0 * Ms_AFM.i * Ms_AFM.j)) * delsq_M_B;
				Hexch_B = (2 * A_AFM.j / ((cuBReal)MU0 * Ms_AFM.j * Ms_AFM.j)) * delsq_M_B + (Anh.j / ((cuBReal)MU0 * Ms_AFM.i * Ms_AFM.j)) * delsq_M_A;
			}

			Heff_pri[cell1_idx] += Hexch;
			Heff2_pri[cell1_idx] += Hexch_B;

			if (do_reduction) {

				int non_empty_cells = M_pri.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * (M_1 * Hexch + M_1_B * Hexch_B) / (4 * non_empty_cells);
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

// Atomistic to FM coupling
__global__ void CalculateDMExchangeCoupling_Atom_to_FM_kernel(
	ManagedAtom_MeshCUDA& mesh_sec, ManagedMeshCUDA& mesh_pri,
	CMBNDInfoCUDA& contact,
	cuBReal& energy, bool do_reduction)
{
	int box_idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	cuINT3 box_sizes = contact.cells_box.size();

	cuVEC_VC<cuReal3>& M_pri = *mesh_pri.pM;
	cuVEC<cuReal3>& Heff_pri = *mesh_pri.pHeff;
	mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1_sec = mesh_sec.pM1->mcuvec();

	if (box_idx < box_sizes.dim()) {

		int i = (box_idx % box_sizes.x) + contact.cells_box.s.i;
		int j = ((box_idx / box_sizes.x) % box_sizes.y) + contact.cells_box.s.j;
		int k = (box_idx / (box_sizes.x * box_sizes.y)) + contact.cells_box.s.k;

		cuReal3 hshift_primary = contact.hshift_primary;
		cuBReal hRsq = hshift_primary.norm();
		hRsq *= hRsq;

		int cell1_idx = i + j * M_pri.n.x + k * M_pri.n.x * M_pri.n.y;

		if (M_pri.is_not_empty(cell1_idx) && M_pri.is_cmbnd(cell1_idx)) {

			//calculate second primary cell index
			int cell2_idx = (i + contact.cell_shift.i) + (j + contact.cell_shift.j) * M_pri.n.x + (k + contact.cell_shift.k) * M_pri.n.x * M_pri.n.y;

			//relative position of cell -1 in secondary mesh
			cuReal3 relpos_m1 = M_pri.rect.s - M1_sec.rect.s + ((cuReal3(i, j, k) + cuReal3(0.5)) & M_pri.h) + (contact.hshift_primary + contact.hshift_secondary) / 2;

			//stencil is used for weighted_average to obtain values in the secondary mesh : has size equal to primary cellsize area on interface with thickness set by secondary cellsize thickness
			cuReal3 stencil = M_pri.h - cu_mod(contact.hshift_primary) + cu_mod(contact.hshift_secondary);

			cuBReal Ms = *mesh_pri.pMs;
			cuBReal A = *mesh_pri.pA;
			cuBReal D = *mesh_pri.pD;
			mesh_pri.update_parameters_mcoarse(cell1_idx, *mesh_pri.pA, A, *mesh_pri.pD, D, *mesh_pri.pMs, Ms);

			cuReal3 Hexch;

			//values at cells -1, 1
			cuReal3 M_1 = M_pri[cell1_idx];
			cuReal3 M_m1 = M1_sec.average(cuRect(relpos_m1 - stencil / 2, relpos_m1 + stencil / 2)) * (cuBReal)MUB / M1_sec.h.dim();

			if (cell2_idx < M_pri.n.dim() && M_pri.is_not_empty(cell2_idx)) {

				//cell2_idx is valid and M is not empty there
				cuReal3 M_2 = M_pri[cell2_idx];

				//set effective field value contribution at cell 1 : direct exchange coupling
				if (M_m1 != cuReal3()) Hexch = (2 * A / (MU0 * Ms * Ms)) * (M_2 + M_m1 - 2 * M_1) / hRsq;
				else Hexch = (2 * A / (MU0 * Ms * Ms)) * (M_2  - M_1) / hRsq;

				//add DMI contributions at CMBND cells, correcting for the sided differentials already applied here
				//the contributions are different depending on the CMBND coupling direction
				if (cuIsNZ(hshift_primary.x)) {

					//along x
					Hexch += (-2 * D / (MU0 * Ms * Ms)) * cuReal3(0, -M_2.z - M_m1.z + 2 * M_1.z, M_2.y - M_m1.y + 2 * M_1.y) / (2 * hshift_primary.x);
				}
				else if (cuIsNZ(hshift_primary.y)) {

					//along y
					Hexch += (-2 * D / (MU0 * Ms * Ms)) * cuReal3(M_2.z + M_m1.z - 2 * M_1.z, 0, -M_2.x - M_m1.x + 2 * M_1.x) / (2 * hshift_primary.y);
				}
				else if (cuIsNZ(hshift_primary.z)) {

					//along z
					Hexch += (-2 * D / (MU0 * Ms * Ms)) * cuReal3(-M_2.y - M_m1.y + 2 * M_1.y, M_2.x + M_m1.x - 2 * M_1.x, 0) / (2 * hshift_primary.z);
				}
			}
			else {

				//set effective field value contribution at cell 1 : direct exchange coupling
				if (M_m1 != cuReal3()) Hexch = (2 * A / (MU0 * Ms * Ms)) * (M_m1 - M_1) / hRsq;
			}

			Heff_pri[cell1_idx] += Hexch;

			if (do_reduction) {

				int non_empty_cells = M_pri.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * M_1 * Hexch / (2 * non_empty_cells);
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

//----------------------- CalculateExchangeCoupling LAUNCHER

//calculate exchange field at coupled cells in this mesh; accumulate energy density contribution in energy
void DMExchangeCUDA::CalculateExchangeCoupling(mcu_val<cuBReal>& energy)
{
	for (int contact_idx = 0; contact_idx < CMBNDcontacts.size(); contact_idx++) {

		size_t size = CMBNDcontacts[contact_idx].cells_box.size().dim();

		//the contacting meshes indexes : secondary mesh index is the one in contact with this one (the primary)
		int idx_sec = CMBNDcontacts[contact_idx].mesh_idx.i;
		int idx_pri = CMBNDcontacts[contact_idx].mesh_idx.j;

		if (pContactingMeshes[idx_pri]->GetMeshType() == MESH_ANTIFERROMAGNETIC && pContactingMeshes[idx_sec]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

			// both contacting meshes are antiferromagnetic

			if (pMeshCUDA->CurrentTimeStepSolved()) {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					size_t size = CMBNDcontactsCUDA[contact_idx].contact_size(mGPU);
					if (!size) continue;

					CalculateDMExchangeCoupling_AFM_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
						(pContactingManagedMeshes[idx_sec]->get_deviceobject(mGPU), pContactingManagedMeshes[idx_pri]->get_deviceobject(mGPU),
						CMBNDcontactsCUDA[contact_idx].get_deviceobject(mGPU), energy(mGPU), true);
				}
			}
			else {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					size_t size = CMBNDcontactsCUDA[contact_idx].contact_size(mGPU);
					if (!size) continue;

					CalculateDMExchangeCoupling_AFM_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
						(pContactingManagedMeshes[idx_sec]->get_deviceobject(mGPU), pContactingManagedMeshes[idx_pri]->get_deviceobject(mGPU),
						CMBNDcontactsCUDA[contact_idx].get_deviceobject(mGPU), energy(mGPU), false);
				}
			}
		}
		else if (pContactingMeshes[idx_pri]->GetMeshType() == MESH_FERROMAGNETIC && pContactingMeshes[idx_sec]->GetMeshType() == MESH_FERROMAGNETIC) {

			//both meshes are ferromagnetic

			if (pMeshCUDA->CurrentTimeStepSolved()) {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					size_t size = CMBNDcontactsCUDA[contact_idx].contact_size(mGPU);
					if (!size) continue;

					CalculateDMExchangeCoupling_FM_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
						(pContactingManagedMeshes[idx_sec]->get_deviceobject(mGPU), pContactingManagedMeshes[idx_pri]->get_deviceobject(mGPU),
						CMBNDcontactsCUDA[contact_idx].get_deviceobject(mGPU), energy(mGPU), true);
				}
			}
			else {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					size_t size = CMBNDcontactsCUDA[contact_idx].contact_size(mGPU);
					if (!size) continue;

					CalculateDMExchangeCoupling_FM_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
						(pContactingManagedMeshes[idx_sec]->get_deviceobject(mGPU), pContactingManagedMeshes[idx_pri]->get_deviceobject(mGPU),
						CMBNDcontactsCUDA[contact_idx].get_deviceobject(mGPU), energy(mGPU), false);
				}
			}
		}

		else if (pContactingMeshes[idx_pri]->GetMeshType() == MESH_FERROMAGNETIC && pContactingMeshes[idx_sec]->is_atomistic()) {

			//micromagnetic to atomistic mesh coupling (here the secondary mesh will be atomistic, as the primary must be magnetic)

			//atomistic to FM

			if (pMeshBaseCUDA->CurrentTimeStepSolved()) {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					size_t size = CMBNDcontactsCUDA[contact_idx].contact_size(mGPU);
					if (!size) continue;

					CalculateDMExchangeCoupling_Atom_to_FM_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
						(pContactingManagedAtomMeshes[idx_sec]->get_deviceobject(mGPU), pContactingManagedMeshes[idx_pri]->get_deviceobject(mGPU), 
						CMBNDcontactsCUDA[contact_idx].get_deviceobject(mGPU), energy(mGPU), true);
				}
			}
			else {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					size_t size = CMBNDcontactsCUDA[contact_idx].contact_size(mGPU);
					if (!size) continue;

					CalculateDMExchangeCoupling_Atom_to_FM_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
						(pContactingManagedAtomMeshes[idx_sec]->get_deviceobject(mGPU), pContactingManagedMeshes[idx_pri]->get_deviceobject(mGPU), 
						CMBNDcontactsCUDA[contact_idx].get_deviceobject(mGPU), energy(mGPU), false);
				}
			}
		}
	}

	//synchronization needed now
	//otherwise a kernel on a device could finish and continue on to diff eq update (which will update M on device), whilst neighboring devices are still accessing these data - data race!
	mGPU.synchronize_if_multi();
}

#endif

#endif