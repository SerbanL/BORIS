#include "Atom_SurfExchangeCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_SURFEXCHANGE) && ATOMISTIC == 1

#include "Reduction.cuh"
#include "cuVEC_VC_mcuVEC.cuh"

#include "Atom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"

#include "ManagedMeshCUDA.h"
#include "MeshParamsControlCUDA.h"

#include "MeshDefs.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------ SURFACE COUPLING Z STACKING

//Top mesh is atomistic
__global__ void SurfExchangeCUDA_Top_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedAtom_MeshCUDA** ppaMesh_Top, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M1.n;
	cuReal3 h = M1.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x + (n.z - 1) * n.x*n.y;

		//skip empty cells
		if (M1.is_not_empty(cell_idx)) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuaMesh.update_parameters_mcoarse(cell_idx, *cuaMesh.pmu_s, mu_s);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1_Top = ppaMesh_Top[mesh_idx]->pM1->mcuvec();

				//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M1.rect.s.x - M1_Top.rect.s.x,
					(j + 0.5) * h.y + M1.rect.s.y - M1_Top.rect.s.y,
					M1_Top.h.z / 2);

				//can't couple to an empty cell
				if (!M1_Top.rect.contains(cell_rel_pos + M1_Top.rect.s) || M1_Top.is_empty(cell_rel_pos)) continue;

				cuBReal Js = *(ppaMesh_Top[mesh_idx]->pJs);
				ppaMesh_Top[mesh_idx]->update_parameters_atposition(cell_rel_pos, *(ppaMesh_Top[mesh_idx]->pJs), Js);

				//get magnetization value in top mesh cell to couple with
				cuReal3 m_j = M1_Top[cell_rel_pos].normalized();
				cuReal3 m_i = M1[cell_idx] / mu_s;

				cuBReal dot_prod = m_i * m_j;

				cuReal3 Hsurfexch = m_j * Js / ((cuBReal)MUB_MU0 * mu_s);

				if (do_reduction) {

					energy_ = -Js * dot_prod / (M1.get_nonempty_cells() * h.dim());
				}

				Heff1[cell_idx] += Hsurfexch;

				//NOTE : we must add into the module display VECs, since there could be 2 contributions for some cells (top and bottom). This is why we had to zero the VECs before calling this kernel.
				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy_ * M1.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//Top mesh is ferromagnetic
__global__ void SurfExchangeCUDA_TopFM_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedMeshCUDA** ppMesh_Top, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M1.n;
	cuReal3 h = M1.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x + (n.z - 1) * n.x*n.y;

		//skip empty cells
		if (M1.is_not_empty(cell_idx)) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal Js = *cuaMesh.pJs;
			cuaMesh.update_parameters_mcoarse(cell_idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.pJs, Js);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M = ppMesh_Top[mesh_idx]->pM->mcuvec();

				cuRect tmeshRect = M.rect;

				//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M1.rect.s.x - tmeshRect.s.x,
					(j + 0.5) * h.y + M1.rect.s.y - tmeshRect.s.y,
					M.h.z / 2);

				//can't couple to an empty cell
				if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s) || M.is_empty(cell_rel_pos)) continue;

				//get magnetization value in top mesh cell to couple with
				cuReal3 m_j = M[cell_rel_pos].normalized();
				cuReal3 m_i = M1[cell_idx] / mu_s;

				cuBReal dot_prod = m_i * m_j;

				cuReal3 Hsurfexch = m_j * Js / (MUB_MU0 * mu_s);

				if (do_reduction) {

					energy_ = -Js * dot_prod / (M1.get_nonempty_cells() * h.dim());
				}

				Heff1[cell_idx] += Hsurfexch;

				//NOTE : we must add into the module display VECs, since there could be 2 contributions for some cells (top and bottom). This is why we had to zero the VECs before calling this kernel.
				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy_ * M1.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//Top mesh is antiferromagnetic
__global__ void SurfExchangeCUDA_TopAFM_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedMeshCUDA** ppMesh_Top, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M1.n;
	cuReal3 h = M1.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x + (n.z - 1) * n.x*n.y;

		//skip empty cells
		if (M1.is_not_empty(cell_idx)) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal Js = *cuaMesh.pJs;
			cuBReal Js2 = *cuaMesh.pJs2;
			cuaMesh.update_parameters_mcoarse(cell_idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.pJs, Js, *cuaMesh.pJs2, Js2);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M = ppMesh_Top[mesh_idx]->pM->mcuvec();
				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2 = ppMesh_Top[mesh_idx]->pM2->mcuvec();

				cuRect tmeshRect = M.rect;

				//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M1.rect.s.x - tmeshRect.s.x,
					(j + 0.5) * h.y + M1.rect.s.y - tmeshRect.s.y,
					M.h.z / 2);

				//can't couple to an empty cell
				if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s) || M.is_empty(cell_rel_pos)) continue;

				//get magnetization value in top mesh cell to couple with
				cuReal3 m_j1 = M[cell_rel_pos].normalized();
				cuReal3 m_j2 = M2[cell_rel_pos].normalized();
				cuReal3 m_i = M1[cell_idx] / mu_s;

				cuBReal dot_prod1 = m_i * m_j1;
				cuBReal dot_prod2 = m_i * m_j2;

				cuReal3 Hsurfexch = m_j1 * Js / (MUB_MU0 * mu_s);
				Hsurfexch += m_j2 * Js2 / (MUB_MU0 * mu_s);

				if (do_reduction) {

					energy_ = -Js * dot_prod1 / (M1.get_nonempty_cells() * h.dim());
					energy_ += -Js2 * dot_prod2 / (M1.get_nonempty_cells() * h.dim());
				}

				Heff1[cell_idx] += Hsurfexch;

				//NOTE : we must add into the module display VECs, since there could be 2 contributions for some cells (top and bottom). This is why we had to zero the VECs before calling this kernel.
				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy_ * M1.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//Bottom mesh is atomistic
__global__ void SurfExchangeCUDA_Bot_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedAtom_MeshCUDA** ppaMesh_Bot, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M1.n;
	cuReal3 h = M1.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x;

		//skip empty cells
		if (M1.is_not_empty(cell_idx)) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal Js = *cuaMesh.pJs;
			cuaMesh.update_parameters_mcoarse(cell_idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.pJs, Js);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1_Bot = ppaMesh_Bot[mesh_idx]->pM1->mcuvec();

				//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M1.rect.s.x - M1_Bot.rect.s.x,
					(j + 0.5) * h.y + M1.rect.s.y - M1_Bot.rect.s.y,
					M1_Bot.rect.height() - M1_Bot.h.z / 2);

				//can't couple to an empty cell
				if (!M1_Bot.rect.contains(cell_rel_pos + M1_Bot.rect.s) || M1_Bot.is_empty(cell_rel_pos)) continue;

				//yes, then get value of magnetization used in coupling with current cell at cell_idx
				cuReal3 m_j = M1_Bot[cell_rel_pos].normalized();
				cuReal3 m_i = M1[cell_idx] / mu_s;

				cuBReal dot_prod = m_i * m_j;

				cuReal3 Hsurfexch = m_j * Js / ((cuBReal)MUB_MU0 * mu_s);

				if (do_reduction) {

					energy_ = -Js * dot_prod / (M1.get_nonempty_cells()* h.dim());
				}

				Heff1[cell_idx] += Hsurfexch;

				//NOTE : we must add into the module display VECs, since there could be 2 contributions for some cells (top and bottom). This is why we had to zero the VECs before calling this kernel.
				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy_ * M1.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//Bottom mesh is ferromagnetic
__global__ void SurfExchangeCUDA_BotFM_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedMeshCUDA** ppMesh_Bot, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M1.n;
	cuReal3 h = M1.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x;

		//skip empty cells
		if (M1.is_not_empty(cell_idx)) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal Js = *cuaMesh.pJs;
			cuaMesh.update_parameters_mcoarse(cell_idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.pJs, Js);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M = ppMesh_Bot[mesh_idx]->pM->mcuvec();

				cuRect bmeshRect = M.rect;

				//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M1.rect.s.x - bmeshRect.s.x,
					(j + 0.5) * h.y + M1.rect.s.y - bmeshRect.s.y,
					bmeshRect.height() - M.h.z / 2);

				//can't couple to an empty cell
				if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s) || M.is_empty(cell_rel_pos)) continue;

				//get magnetization value in top mesh cell to couple with
				cuReal3 m_j = M[cell_rel_pos].normalized();
				cuReal3 m_i = M1[cell_idx] / mu_s;

				cuBReal dot_prod = m_i * m_j;

				cuReal3 Hsurfexch = m_j * Js / (MUB_MU0 * mu_s);

				if (do_reduction) {

					energy_ = -Js * dot_prod / (M1.get_nonempty_cells()* h.dim());
				}

				Heff1[cell_idx] += Hsurfexch;

				//NOTE : we must add into the module display VECs, since there could be 2 contributions for some cells (top and bottom). This is why we had to zero the VECs before calling this kernel.
				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy_ * M1.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//Bottom mesh is antiferromagnetic
__global__ void SurfExchangeCUDA_BotAFM_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedMeshCUDA** ppMesh_Bot, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M1.n;
	cuReal3 h = M1.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x;

		//skip empty cells
		if (M1.is_not_empty(cell_idx)) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal Js = *cuaMesh.pJs;
			cuBReal Js2 = *cuaMesh.pJs2;
			cuaMesh.update_parameters_mcoarse(cell_idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.pJs, Js, *cuaMesh.pJs2, Js2);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M = ppMesh_Bot[mesh_idx]->pM->mcuvec();
				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2 = ppMesh_Bot[mesh_idx]->pM2->mcuvec();

				cuRect bmeshRect = M.rect;

				//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M1.rect.s.x - bmeshRect.s.x,
					(j + 0.5) * h.y + M1.rect.s.y - bmeshRect.s.y,
					bmeshRect.height() - M.h.z / 2);

				//can't couple to an empty cell
				if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s) || M.is_empty(cell_rel_pos)) continue;

				//get magnetization value in top mesh cell to couple with
				cuReal3 m_j1 = M[cell_rel_pos].normalized();
				cuReal3 m_j2 = M2[cell_rel_pos].normalized();
				cuReal3 m_i = M1[cell_idx] / mu_s;

				cuBReal dot_prod1 = m_i * m_j1;
				cuBReal dot_prod2 = m_i * m_j2;

				cuReal3 Hsurfexch = m_j1 * Js / (MUB_MU0 * mu_s);
				Hsurfexch += m_j2 * Js2 / (MUB_MU0 * mu_s);

				if (do_reduction) {

					energy_ = -Js * dot_prod1 / (M1.get_nonempty_cells()* h.dim());
					energy_ += -Js2 * dot_prod2 / (M1.get_nonempty_cells()* h.dim());
				}

				Heff1[cell_idx] += Hsurfexch;

				//NOTE : we must add into the module display VECs, since there could be 2 contributions for some cells (top and bottom). This is why we had to zero the VECs before calling this kernel.
				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy_ * M1.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

__global__ void SurfExchangeAtomCUDA_Bulk_UpdateField(
	ManagedAtom_MeshCUDA& cuaMesh,
	cuVEC<cuINT3>& bulk_coupling_mask,
	ManagedAtom_MeshCUDA** ppaMesh_Bulk, size_t coupled_ameshes,
	ManagedMeshCUDA** ppMeshFM_Bulk, size_t coupledFM_meshes,
	ManagedMeshCUDA** ppMeshAFM_Bulk, size_t coupledAFM_meshes,
	ManagedModulesCUDA& cuModule, bool do_reduction)
{
	//------------------ Coupling functions

	auto calculate_atom_coupling = [](
		cuVEC_VC<cuReal3>& M1, int cell_idx,
		ManagedAtom_MeshCUDA& aMeshCoupled, cuReal3 cell_rel_pos,
		cuBReal mu_s, cuBReal Js,
		cuReal3& Hsurfexch, cuBReal& cell_energy, bool do_reduction) -> void
	{
		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1_Bulk = aMeshCoupled.pM1->mcuvec();

		//get magnetization value in top mesh cell to couple with
		cuReal3 m_j = cu_normalize(M1_Bulk[cell_rel_pos]);
		cuReal3 m_i = M1[cell_idx] / mu_s;

		cuBReal dot_prod = m_i * m_j;

		Hsurfexch += m_j * Js / ((cuBReal)MUB_MU0 * mu_s);
		if (do_reduction) cell_energy += -Js * dot_prod / (M1.h.dim() * M1.get_nonempty_cells());
	};

	auto calculate_mm_FM_coupling = [](
		cuVEC_VC<cuReal3>& M1, int cell_idx,
		ManagedMeshCUDA& MeshCoupled, cuReal3 cell_rel_pos,
		cuBReal mu_s, cuBReal Js,
		cuReal3& Hsurfexch, cuBReal& cell_energy, bool do_reduction) -> void
	{
		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bulk = MeshCoupled.pM->mcuvec();

		//get magnetization value in top mesh cell to couple with
		cuReal3 m_j = cu_normalize(M_Bulk[cell_rel_pos]);
		cuReal3 m_i = M1[cell_idx] / mu_s;

		cuBReal dot_prod = m_i * m_j;

		Hsurfexch += m_j * Js / ((cuBReal)MUB_MU0 * mu_s);
		if (do_reduction) cell_energy += -Js * dot_prod / (M1.h.dim() * M1.get_nonempty_cells());
	};

	auto calculate_mm_AFM_coupling = [](
		cuVEC_VC<cuReal3>& M1, int cell_idx,
		ManagedMeshCUDA& MeshCoupled, cuReal3 cell_rel_pos,
		cuBReal mu_s, cuBReal Js, cuBReal Js2,
		cuReal3& Hsurfexch, cuBReal& cell_energy, bool do_reduction) -> void
	{
		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bulk = MeshCoupled.pM->mcuvec();
		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2_Bulk = MeshCoupled.pM2->mcuvec();

		//get magnetization value in top mesh cell to couple with
		cuReal3 m_j1 = cu_normalize(M_Bulk[cell_rel_pos]);
		cuReal3 m_j2 = cu_normalize(M2_Bulk[cell_rel_pos]);
		cuReal3 m_i = M1[cell_idx] / mu_s;

		cuBReal dot_prod1 = m_i * m_j1;
		cuBReal dot_prod2 = m_i * m_j2;

		Hsurfexch += m_j1 * Js / ((cuBReal)MUB_MU0 * mu_s);
		Hsurfexch += m_j2 * Js2 / ((cuBReal)MUB_MU0 * mu_s);
		
		if (do_reduction) {
			cell_energy += -Js * dot_prod1 / (M1.h.dim() * M1.get_nonempty_cells());
			cell_energy += -Js2 * dot_prod2 / (M1.h.dim() * M1.get_nonempty_cells());
		}
	};

	//------------------

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M1.n;
	cuReal3 h = M1.h;

	cuBReal energy_ = 0.0;

	if (idx < n.dim()) {

		int i = idx % n.x;
		int j = (idx / n.x) % n.y;
		int k = idx / (n.x * n.y);
		int idx = i + j * n.x + k * n.x * n.y;

		//skip empty cells
		if (M1.is_not_empty(idx) && bulk_coupling_mask[idx] != cuINT3()) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal Js = *cuaMesh.pJs;
			cuBReal Js2 = *cuaMesh.pJs2;
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.pJs, Js, *cuaMesh.pJs2, Js2);

			//surface cell which needs to be exchange coupled
			cuReal3 Hsurfexch = cuReal3();
			int num_couplings = 0;

			cuReal3 abs_pos = M1.cellidx_to_position(idx) + M1.rect.s;

			//+x coupling direction
			if (bulk_coupling_mask[idx].x & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[idx].x & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < coupled_ameshes) {
					calculate_atom_coupling(
						M1, idx,
						*ppaMesh_Bulk[mesh_idx], abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3((h.x + ppaMesh_Bulk[mesh_idx]->pM1->h.x) / 2, 0, 0),
						mu_s, Js,
						Hsurfexch, energy_, do_reduction);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < coupled_ameshes + coupledFM_meshes) {
						calculate_mm_FM_coupling(
							M1, idx,
							*ppMeshFM_Bulk[mesh_idx - coupled_ameshes],
							abs_pos - ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->origin + cuReal3((h.x + ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->h.x) / 2, 0, 0),
							mu_s, Js,
							Hsurfexch, energy_, do_reduction);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							M1, idx,
							*ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes],
							abs_pos - ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->origin + cuReal3((h.x + ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->h.x) / 2, 0, 0),
							mu_s, Js, Js2,
							Hsurfexch, energy_, do_reduction);
					}
				}
			}

			//-x coupling direction
			if (bulk_coupling_mask[idx].x & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[idx].x >> 16) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < coupled_ameshes) {
					calculate_atom_coupling(
						M1, idx,
						*ppaMesh_Bulk[mesh_idx], abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3(-(h.x + ppaMesh_Bulk[mesh_idx]->pM1->h.x) / 2, 0, 0),
						mu_s, Js,
						Hsurfexch, energy_, do_reduction);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < coupled_ameshes + coupledFM_meshes) {
						calculate_mm_FM_coupling(
							M1, idx,
							*ppMeshFM_Bulk[mesh_idx - coupled_ameshes],
							abs_pos - ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->origin + cuReal3(-(h.x + ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->h.x) / 2, 0, 0),
							mu_s, Js,
							Hsurfexch, energy_, do_reduction);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							M1, idx,
							*ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes],
							abs_pos - ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->origin + cuReal3(-(h.x + ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->h.x) / 2, 0, 0),
							mu_s, Js, Js2,
							Hsurfexch, energy_, do_reduction);
					}
				}
			}

			//+y coupling direction
			if (bulk_coupling_mask[idx].y & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[idx].y & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < coupled_ameshes) {
					calculate_atom_coupling(
						M1, idx,
						*ppaMesh_Bulk[mesh_idx], abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3(0, (h.y + ppaMesh_Bulk[mesh_idx]->pM1->h.y) / 2, 0),
						mu_s, Js,
						Hsurfexch, energy_, do_reduction);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < coupled_ameshes + coupledFM_meshes) {
						calculate_mm_FM_coupling(
							M1, idx,
							*ppMeshFM_Bulk[mesh_idx - coupled_ameshes],
							abs_pos - ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->origin + cuReal3(0, (h.y + ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->h.y) / 2, 0),
							mu_s, Js,
							Hsurfexch, energy_, do_reduction);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							M1, idx,
							*ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes],
							abs_pos - ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->origin + cuReal3(0, (h.y + ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->h.y) / 2, 0),
							mu_s, Js, Js2,
							Hsurfexch, energy_, do_reduction);
					}
				}
			}

			//-y coupling direction
			if (bulk_coupling_mask[idx].y & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[idx].y >> 16) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < coupled_ameshes) {
					calculate_atom_coupling(
						M1, idx,
						*ppaMesh_Bulk[mesh_idx], abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3(0, -(h.y + ppaMesh_Bulk[mesh_idx]->pM1->h.y) / 2, 0),
						mu_s, Js,
						Hsurfexch, energy_, do_reduction);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < coupled_ameshes + coupledFM_meshes) {
						calculate_mm_FM_coupling(
							M1, idx,
							*ppMeshFM_Bulk[mesh_idx - coupled_ameshes],
							abs_pos - ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->origin + cuReal3(0, -(h.y + ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->h.y) / 2, 0),
							mu_s, Js,
							Hsurfexch, energy_, do_reduction);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							M1, idx,
							*ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes],
							abs_pos - ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->origin + cuReal3(0, -(h.y + ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->h.y) / 2, 0),
							mu_s, Js, Js2,
							Hsurfexch, energy_, do_reduction);
					}
				}
			}

			//+z coupling direction
			if (bulk_coupling_mask[idx].z & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[idx].z & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < coupled_ameshes) {
					calculate_atom_coupling(
						M1, idx,
						*ppaMesh_Bulk[mesh_idx], abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3(0, 0, (h.z + ppaMesh_Bulk[mesh_idx]->pM1->h.z) / 2),
						mu_s, Js,
						Hsurfexch, energy_, do_reduction);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < coupled_ameshes + coupledFM_meshes) {
						calculate_mm_FM_coupling(
							M1, idx,
							*ppMeshFM_Bulk[mesh_idx - coupled_ameshes],
							abs_pos - ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->origin + cuReal3(0, 0, (h.z + ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->h.z) / 2),
							mu_s, Js,
							Hsurfexch, energy_, do_reduction);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							M1, idx,
							*ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes],
							abs_pos - ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->origin + cuReal3(0, 0, (h.z + ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->h.z) / 2),
							mu_s, Js, Js2,
							Hsurfexch, energy_, do_reduction);
					}
				}
			}

			//-z coupling direction
			if (bulk_coupling_mask[idx].z & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[idx].z >> 16) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < coupled_ameshes) {
					calculate_atom_coupling(
						M1, idx,
						*ppaMesh_Bulk[mesh_idx], abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3(0, 0, -(h.z + ppaMesh_Bulk[mesh_idx]->pM1->h.z) / 2),
						mu_s, Js,
						Hsurfexch, energy_, do_reduction);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < coupled_ameshes + coupledFM_meshes) {
						calculate_mm_FM_coupling(
							M1, idx,
							*ppMeshFM_Bulk[mesh_idx - coupled_ameshes],
							abs_pos - ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->origin + cuReal3(0, 0, -(h.z + ppMeshFM_Bulk[mesh_idx - coupled_ameshes]->pM->h.z) / 2),
							mu_s, Js,
							Hsurfexch, energy_, do_reduction);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							M1, idx,
							*ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes],
							abs_pos - ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->origin + cuReal3(0, 0, -(h.z + ppMeshAFM_Bulk[mesh_idx - coupled_ameshes - coupledFM_meshes]->pM->h.z) / 2),
							mu_s, Js, Js2,
							Hsurfexch, energy_, do_reduction);
					}
				}
			}

			if (num_couplings) {

				//need average if cell receives multiple coupling contributions
				Hsurfexch /= num_couplings;
				energy_ /= num_couplings;
			}

			Heff1[idx] += Hsurfexch;

			//NOTE : we must add into the module display VECs, since there could be 2 contributions for some cells (top and bottom). This is why we had to zero the VECs before calling this kernel.
			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] += Hsurfexch;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] += energy_ * M1.get_nonempty_cells();

		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Atom_SurfExchangeCUDA::UpdateField(void)
{
	if (paMeshCUDA->CurrentTimeStepSolved()) {

		ZeroEnergy();
		ZeroModuleVECs();

		//------------------ SURFACE COUPLING Z STACKING

		//Top - Atomistic
		if (paMesh_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_Top_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), paMesh_Top.get_array(mGPU), paMesh_Top.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//Top - Ferromagnetic
		if (pMeshFM_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_TopFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), pMeshFM_Top.get_array(mGPU), pMeshFM_Top.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//Top - AntiFerromagnetic
		if (pMeshAFM_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_TopAFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), pMeshAFM_Top.get_array(mGPU), pMeshAFM_Top.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//Bottom - Atomistic
		if (paMesh_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_Bot_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), paMesh_Bot.get_array(mGPU), paMesh_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//Bottom - Ferromagnetic
		if (pMeshFM_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_BotFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), pMeshFM_Bot.get_array(mGPU), pMeshFM_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//Bottom - AntiFerromagnetic
		if (pMeshAFM_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_BotAFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), pMeshAFM_Bot.get_array(mGPU), pMeshAFM_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

		if (paMesh_Bulk.size() + pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				SurfExchangeAtomCUDA_Bulk_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU),
						bulk_coupling_mask.get_deviceobject(mGPU),
						paMesh_Bulk.get_array(mGPU), paMesh_Bulk.size(mGPU),
						pMeshFM_Bulk.get_array(mGPU), pMeshFM_Bulk.size(mGPU),
						pMeshAFM_Bulk.get_array(mGPU), pMeshAFM_Bulk.size(mGPU),
						cuModule.get_deviceobject(mGPU), true);
			}
		}
	}
	else {

		//------------------ SURFACE COUPLING Z STACKING

		//Top - Atomistic
		if (paMesh_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_Top_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), paMesh_Top.get_array(mGPU), paMesh_Top.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//Top - Ferromagnetic
		if (pMeshFM_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_TopFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), pMeshFM_Top.get_array(mGPU), pMeshFM_Top.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//Top - AntiFerromagnetic
		if (pMeshAFM_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_TopAFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), pMeshAFM_Top.get_array(mGPU), pMeshAFM_Top.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//Bottom - Atomistic
		if (paMesh_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_Bot_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), paMesh_Bot.get_array(mGPU), paMesh_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//Bottom - Ferromagnetic
		if (pMeshFM_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_BotFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), pMeshFM_Bot.get_array(mGPU), pMeshFM_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//Bottom - AntiFerromagnetic
		if (pMeshAFM_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = paMeshCUDA->M1.device_n(mGPU);
				SurfExchangeCUDA_BotAFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), pMeshAFM_Bot.get_array(mGPU), pMeshAFM_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

		if (paMesh_Bulk.size() + pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				SurfExchangeAtomCUDA_Bulk_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU),
						bulk_coupling_mask.get_deviceobject(mGPU),
						paMesh_Bulk.get_array(mGPU), paMesh_Bulk.size(mGPU),
						pMeshFM_Bulk.get_array(mGPU), pMeshFM_Bulk.size(mGPU),
						pMeshAFM_Bulk.get_array(mGPU), pMeshAFM_Bulk.size(mGPU),
						cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
}

//----------------------- Initialization

//Top mesh is ferromagnetic
__global__ void set_Atom_SurfExchangeCUDA_pointers_kernel(
	ManagedAtom_MeshCUDA& cuaMesh,
	ManagedAtom_MeshCUDA** ppaMesh_Bot, size_t coupled_bot_meshes,
	ManagedAtom_MeshCUDA** ppaMesh_Top, size_t coupled_top_meshes,
	ManagedMeshCUDA** ppMeshFM_Bot, size_t coupledFM_bot_meshes,
	ManagedMeshCUDA** ppMeshFM_Top, size_t coupledFM_top_meshes,
	ManagedMeshCUDA** ppMeshAFM_Bot, size_t coupledAFM_bot_meshes,
	ManagedMeshCUDA** ppMeshAFM_Top, size_t coupledAFM_top_meshes,
	ManagedAtom_MeshCUDA** ppaMesh_Bulk, size_t paMesh_Bulk_size,
	ManagedMeshCUDA** ppMeshFM_Bulk, size_t pMeshFM_Bulk_size,
	ManagedMeshCUDA** ppMeshAFM_Bulk, size_t pMeshAFM_Bulk_size,
	cuVEC<cuINT3>& bulk_coupling_mask)
{
	if (threadIdx.x == 0) cuaMesh.ppaMesh_Bot = ppaMesh_Bot;
	if (threadIdx.x == 1) cuaMesh.paMesh_Bot_size = coupled_bot_meshes;
	if (threadIdx.x == 2) cuaMesh.ppaMesh_Top = ppaMesh_Top;
	if (threadIdx.x == 3) cuaMesh.paMesh_Top_size = coupled_top_meshes;

	if (threadIdx.x == 4) cuaMesh.ppMeshFM_Bot = ppMeshFM_Bot;
	if (threadIdx.x == 5) cuaMesh.pMeshFM_Bot_size = coupledFM_bot_meshes;
	if (threadIdx.x == 6) cuaMesh.ppMeshFM_Top = ppMeshFM_Top;
	if (threadIdx.x == 7) cuaMesh.pMeshFM_Top_size = coupledFM_top_meshes;

	if (threadIdx.x == 8) cuaMesh.ppMeshAFM_Bot = ppMeshAFM_Bot;
	if (threadIdx.x == 9) cuaMesh.pMeshAFM_Bot_size = coupledAFM_bot_meshes;
	if (threadIdx.x == 10) cuaMesh.ppMeshAFM_Top = ppMeshAFM_Top;
	if (threadIdx.x == 11) cuaMesh.pMeshAFM_Top_size = coupledAFM_top_meshes;

	if (threadIdx.x == 12) cuaMesh.ppaMesh_Bulk = ppaMesh_Bulk;
	if (threadIdx.x == 13) cuaMesh.paMesh_Bulk_size = paMesh_Bulk_size;
	if (threadIdx.x == 14) cuaMesh.ppMeshFM_Bulk = ppMeshFM_Bulk;
	if (threadIdx.x == 15) cuaMesh.pMeshFM_Bulk_size = pMeshFM_Bulk_size;
	if (threadIdx.x == 16) cuaMesh.ppMeshAFM_Bulk = ppMeshAFM_Bulk;
	if (threadIdx.x == 17) cuaMesh.pMeshAFM_Bulk_size = pMeshAFM_Bulk_size;
	if (threadIdx.x == 18) cuaMesh.pbulk_coupling_mask = &bulk_coupling_mask;
}

//Called by Atom_SurfExchangeCUDA module
void Atom_SurfExchangeCUDA::set_Atom_SurfExchangeCUDA_pointers(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		set_Atom_SurfExchangeCUDA_pointers_kernel <<< 1, CUDATHREADS >>>
			(paMeshCUDA->cuaMesh.get_deviceobject(mGPU),
			paMesh_Bot.get_array(mGPU), paMesh_Bot.size(mGPU), paMesh_Top.get_array(mGPU), paMesh_Top.size(mGPU),
			pMeshFM_Bot.get_array(mGPU), pMeshFM_Bot.size(mGPU), pMeshFM_Top.get_array(mGPU), pMeshFM_Top.size(mGPU),
			pMeshAFM_Bot.get_array(mGPU), pMeshAFM_Bot.size(mGPU), pMeshAFM_Top.get_array(mGPU), pMeshAFM_Top.size(mGPU),
			paMesh_Bulk.get_array(mGPU), paMesh_Bulk.size(mGPU),
			pMeshFM_Bulk.get_array(mGPU), pMeshFM_Bulk.size(mGPU),
			pMeshAFM_Bulk.get_array(mGPU), pMeshAFM_Bulk.size(mGPU),
			bulk_coupling_mask.get_deviceobject(mGPU));
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && ATOMISTIC == 1 && MONTE_CARLO == 1

__device__ cuBReal ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_SurfExchangeCUDA(int spin_index, cuReal3 Mnew)
{
	cuBReal energy_new = 0, energy_old = 0;

	cuVEC_VC<cuReal3>& M1 = *pM1;

	cuSZ3 n = M1.n;
	cuReal3 h = M1.h;

	//------------------ Coupling functions

	auto calculate_atom_coupling = [](
		cuVEC_VC<cuReal3>& M1, cuReal3 Mnew, int spin_index,
		ManagedAtom_MeshCUDA& aMeshCoupled, cuReal3& cell_rel_pos,
		cuBReal Js,
		cuBReal& energy_old, cuBReal& energy_new) -> void
	{
		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1_Bulk = aMeshCoupled.pM1->mcuvec();

		//get magnetization value in top mesh cell to couple with
		cuReal3 m_j = cu_normalize(M1_Bulk[cell_rel_pos]);
		cuReal3 m_i = cu_normalize(M1[spin_index]);

		cuBReal dot_prod = m_i * m_j;
		energy_old += -Js * dot_prod;

		if (Mnew != cuReal3()) {

			cuReal3 mnew_i = cu_normalize(Mnew);
			cuBReal dot_prod_new = mnew_i * m_j;
			energy_new += -Js * dot_prod_new;
		}
	};

	auto calculate_mm_FM_coupling = [](
		cuVEC_VC<cuReal3>& M1, cuReal3 Mnew, int spin_index,
		ManagedMeshCUDA& MeshCoupled, cuReal3& cell_rel_pos,
		cuBReal Js,
		cuBReal& energy_old, cuBReal& energy_new) -> void
	{
		//Surface exchange field from a ferromagnetic mesh (RKKY)

		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bulk = MeshCoupled.pM->mcuvec();

		//get magnetization value in top mesh cell to couple with
		cuReal3 m_j = cu_normalize(M_Bulk[cell_rel_pos]);
		cuReal3 m_i = cu_normalize(M1[spin_index]);

		cuBReal dot_prod = m_i * m_j;
		energy_old += -Js * dot_prod;

		if (Mnew != cuReal3()) {

			cuReal3 mnew_i = cu_normalize(Mnew);
			cuBReal dot_prod_new = mnew_i * m_j;
			energy_new += -Js * dot_prod_new;
		}
	};

	auto calculate_mm_AFM_coupling = [](
		cuVEC_VC<cuReal3>& M1, cuReal3 Mnew, int spin_index,
		ManagedMeshCUDA& MeshCoupled, cuReal3& cell_rel_pos,
		cuBReal Js, cuBReal Js2,
		cuBReal& energy_old, cuBReal& energy_new) -> void
	{
		//Surface exchange field from an antiferromagnetic mesh (exchange bias)

		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bulk = MeshCoupled.pM->mcuvec();
		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2_Bulk = MeshCoupled.pM2->mcuvec();

		//get magnetization value in top mesh cell to couple with
		cuReal3 m_j1 = cu_normalize(M_Bulk[cell_rel_pos]);
		cuReal3 m_j2 = cu_normalize(M2_Bulk[cell_rel_pos]);
		cuReal3 m_i = cu_normalize(M1[spin_index]);

		cuBReal dot_prod1 = m_i * m_j1;
		cuBReal dot_prod2 = m_i * m_j2;
		energy_old += -Js * dot_prod1;
		energy_old += -Js2 * dot_prod2;

		if (Mnew != cuReal3()) {

			cuReal3 mnew_i = cu_normalize(Mnew);
			cuBReal dot_prod_new1 = mnew_i * m_j1;
			cuBReal dot_prod_new2 = mnew_i * m_j2;
			energy_new += -Js * dot_prod_new1;
			energy_new += -Js2 * dot_prod_new2;
		}
	};

	//------------------ SURFACE COUPLING Z STACKING

	//if spin is on top surface then look at paMesh_Top
	if (spin_index / (n.x * n.y) == n.z - 1 && (paMesh_Top_size + pMeshFM_Top_size)) {

		if (!M1.is_empty(spin_index)) {

			int i = spin_index % n.x;
			int j = (spin_index / n.x) % n.y;
			bool cell_coupled = false;

			//check all meshes for coupling
			//1. coupling from other atomistic meshes
			for (int mesh_idx = 0; mesh_idx < paMesh_Top_size; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1_Top = ppaMesh_Top[mesh_idx]->pM1->mcuvec();

				//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M1.rect.s.x - M1_Top.rect.s.x,
					(j + 0.5) * h.y + M1.rect.s.y - M1_Top.rect.s.y,
					M1_Top.h.z / 2);

				//can't couple to an empty cell
				if (!M1_Top.rect.contains(cell_rel_pos + M1_Top.rect.s) || M1_Top.is_empty(cell_rel_pos)) continue;

				cuBReal Js = *(ppaMesh_Top[mesh_idx]->pJs);
				ppaMesh_Top[mesh_idx]->update_parameters_atposition(cell_rel_pos, *(ppaMesh_Top[mesh_idx]->pJs), Js);

				calculate_atom_coupling(
					M1, Mnew, spin_index,
					*ppaMesh_Top[mesh_idx], cell_rel_pos,
					Js,
					energy_old, energy_new);

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				cell_coupled = true;
				break;
			}

			if (!cell_coupled) {

				//2. coupling from micromagnetic meshes - ferromagnetic
				for (int mesh_idx = 0; mesh_idx < pMeshFM_Top_size; mesh_idx++) {

					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M = ppMeshFM_Top[mesh_idx]->pM->mcuvec();

					cuRect tmeshRect = M.rect;

					//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
					cuReal3 cell_rel_pos = cuReal3(
						(i + 0.5) * h.x + M1.rect.s.x - tmeshRect.s.x,
						(j + 0.5) * h.y + M1.rect.s.y - tmeshRect.s.y,
						M.h.z / 2);

					//can't couple to an empty cell
					if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s) || M.is_empty(cell_rel_pos)) continue;

					cuBReal Js = *pJs;
					update_parameters_mcoarse(spin_index, *pJs, Js);

					calculate_mm_FM_coupling(
						M1, Mnew, spin_index,
						*ppMeshFM_Top[mesh_idx], cell_rel_pos,
						Js,
						energy_old, energy_new);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					cell_coupled = true;
					break;
				}
			}

			if (!cell_coupled) {

				//2. coupling from micromagnetic meshes - antiferromagnetic
				for (int mesh_idx = 0; mesh_idx < pMeshAFM_Top_size; mesh_idx++) {

					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M = ppMeshAFM_Top[mesh_idx]->pM->mcuvec();
					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2 = ppMeshAFM_Top[mesh_idx]->pM2->mcuvec();

					cuRect tmeshRect = M.rect;

					//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
					cuReal3 cell_rel_pos = cuReal3(
						(i + 0.5) * h.x + M1.rect.s.x - tmeshRect.s.x,
						(j + 0.5) * h.y + M1.rect.s.y - tmeshRect.s.y,
						M.h.z / 2);

					//can't couple to an empty cell
					if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s) || M.is_empty(cell_rel_pos)) continue;

					cuBReal Js = *pJs;
					cuBReal Js2 = *pJs2;
					update_parameters_mcoarse(spin_index, *pJs, Js, *pJs2, Js2);

					calculate_mm_AFM_coupling(
						M1, Mnew, spin_index,
						*ppMeshAFM_Top[mesh_idx], cell_rel_pos,
						Js, Js2,
						energy_old, energy_new);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					break;
				}
			}
		}
	}

	//if spin is on bottom surface then look at paMesh_Top
	if (spin_index / (n.x * n.y) == 0 && (paMesh_Bot_size + pMeshFM_Bot_size)) {

		if (!M1.is_empty(spin_index)) {

			int i = spin_index % n.x;
			int j = (spin_index / n.x) % n.y;
			bool cell_coupled = false;

			cuBReal Js = *pJs;
			update_parameters_mcoarse(spin_index, *pJs, Js);

			//check all meshes for coupling
			//1. coupling from other atomistic meshes
			for (int mesh_idx = 0; mesh_idx < paMesh_Bot_size; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1_Bot = ppaMesh_Bot[mesh_idx]->pM1->mcuvec();

				//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M1.rect.s.x - M1_Bot.rect.s.x,
					(j + 0.5) * h.y + M1.rect.s.y - M1_Bot.rect.s.y,
					M1_Bot.rect.e.z - M1_Bot.rect.s.z - M1_Bot.h.z / 2);

				//can't couple to an empty cell
				if (!M1_Bot.rect.contains(cell_rel_pos + M1_Bot.rect.s) || M1_Bot.is_empty(cell_rel_pos)) continue;

				calculate_atom_coupling(
					M1, Mnew, spin_index,
					*ppaMesh_Bot[mesh_idx], cell_rel_pos,
					Js,
					energy_old, energy_new);

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				cell_coupled = true;
				break;
			}

			if (!cell_coupled) {

				//2. coupling from micromagnetic meshes - ferromagnetic
				for (int mesh_idx = 0; mesh_idx < pMeshFM_Bot_size; mesh_idx++) {

					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M = ppMeshFM_Bot[mesh_idx]->pM->mcuvec();

					cuRect bmeshRect = M.rect;

					//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
					cuReal3 cell_rel_pos = cuReal3(
						(i + 0.5) * h.x + M1.rect.s.x - bmeshRect.s.x,
						(j + 0.5) * h.y + M1.rect.s.y - bmeshRect.s.y,
						bmeshRect.height() - M.h.z / 2);

					//can't couple to an empty cell
					if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s) || M.is_empty(cell_rel_pos)) continue;

					calculate_mm_FM_coupling(
						M1, Mnew, spin_index,
						*ppMeshFM_Bot[mesh_idx], cell_rel_pos,
						Js,
						energy_old, energy_new);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					cell_coupled = true;
					break;
				}
			}

			if (!cell_coupled) {

				//2. coupling from micromagnetic meshes - antiferromagnetic
				for (int mesh_idx = 0; mesh_idx < pMeshAFM_Bot_size; mesh_idx++) {

					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M = ppMeshAFM_Bot[mesh_idx]->pM->mcuvec();
					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2 = ppMeshAFM_Bot[mesh_idx]->pM2->mcuvec();

					cuRect bmeshRect = M.rect;

					//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
					cuReal3 cell_rel_pos = cuReal3(
						(i + 0.5) * h.x + M1.rect.s.x - bmeshRect.s.x,
						(j + 0.5) * h.y + M1.rect.s.y - bmeshRect.s.y,
						bmeshRect.height() - M.h.z / 2);

					//can't couple to an empty cell
					if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s) || M.is_empty(cell_rel_pos)) continue;

					cuBReal Js2 = *pJs2;
					update_parameters_mcoarse(spin_index, *pJs2, Js2);

					calculate_mm_AFM_coupling(
						M1, Mnew, spin_index,
						*ppMeshAFM_Bot[mesh_idx], cell_rel_pos,
						Js, Js2,
						energy_old, energy_new);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					break;
				}
			}
		}
	}

	//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

	if (paMesh_Bulk_size + pMeshFM_Bulk_size + pMeshAFM_Bulk_size) {

		if (M1.is_not_empty(spin_index) && (*pbulk_coupling_mask)[spin_index] != cuINT3()) {

			cuBReal energy_bulk_new = 0, energy_bulk_old = 0;

			cuBReal Js = *pJs;
			cuBReal Js2 = *pJs2;
			update_parameters_mcoarse(spin_index, *pJs, Js, *pJs2, Js2);

			int num_couplings = 0;

			cuReal3 abs_pos = M1.cellidx_to_position(spin_index) + M1.rect.s;

			cuReal3 cell_rel_pos;
			int mesh_idx = -1;

			for (int nidx = 0; nidx < 6; nidx++) {

				//+x coupling direction
				if (nidx == 0 && (*pbulk_coupling_mask)[spin_index].x & 0x0000ffff) {

					mesh_idx = ((*pbulk_coupling_mask)[spin_index].x & 0x0000ffff) - 1;
					if (mesh_idx < paMesh_Bulk_size) cell_rel_pos = abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3((M1.h.x + ppaMesh_Bulk[mesh_idx]->pM1->h.x) / 2, 0, 0);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->origin + cuReal3((M1.h.x + ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->h.x) / 2, 0, 0);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size + pMeshAFM_Bulk_size)
						cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->origin + cuReal3((M1.h.x + ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->h.x) / 2, 0, 0);
				}

				//-x coupling direction
				else if (nidx == 1 && (*pbulk_coupling_mask)[spin_index].x & 0xffff0000) {

					mesh_idx = mesh_idx = ((*pbulk_coupling_mask)[spin_index].x >> 16) - 1;
					if (mesh_idx < paMesh_Bulk_size) cell_rel_pos = abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3(-(M1.h.x + ppaMesh_Bulk[mesh_idx]->pM1->h.x) / 2, 0, 0);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->origin + cuReal3(-(M1.h.x + ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->h.x) / 2, 0, 0);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size + pMeshAFM_Bulk_size)
						cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->origin + cuReal3(-(M1.h.x + ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->h.x) / 2, 0, 0);
				}

				//+y coupling direction
				else if (nidx == 2 && (*pbulk_coupling_mask)[spin_index].y & 0x0000ffff) {

					mesh_idx = ((*pbulk_coupling_mask)[spin_index].y & 0x0000ffff) - 1;
					if (mesh_idx < paMesh_Bulk_size) cell_rel_pos = abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3(0, (M1.h.y + ppaMesh_Bulk[mesh_idx]->pM1->h.y) / 2, 0);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->origin + cuReal3(0, (M1.h.y + ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->h.y) / 2, 0);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size + pMeshAFM_Bulk_size)
						cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->origin + cuReal3(0, (M1.h.y + ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->h.y) / 2, 0);
				}

				//-y coupling direction
				else if (nidx == 3 && (*pbulk_coupling_mask)[spin_index].y & 0xffff0000) {

					mesh_idx = ((*pbulk_coupling_mask)[spin_index].y >> 16) - 1;
					if (mesh_idx < paMesh_Bulk_size) cell_rel_pos = abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3(0, -(M1.h.y + ppaMesh_Bulk[mesh_idx]->pM1->h.y) / 2, 0);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->origin + cuReal3(0, -(M1.h.y + ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->h.y) / 2, 0);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size + pMeshAFM_Bulk_size)
						cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->origin + cuReal3(0, -(M1.h.y + ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->h.y) / 2, 0);
				}

				//+z coupling direction
				else if (nidx == 4 && (*pbulk_coupling_mask)[spin_index].z & 0x0000ffff) {

					mesh_idx = ((*pbulk_coupling_mask)[spin_index].z & 0x0000ffff) - 1;
					if (mesh_idx < paMesh_Bulk_size) cell_rel_pos = abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3(0, 0, (M1.h.z + ppaMesh_Bulk[mesh_idx]->pM1->h.z) / 2);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->origin + cuReal3(0, 0, (M1.h.z + ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->h.z) / 2);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size + pMeshAFM_Bulk_size)
						cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->origin + cuReal3(0, 0, (M1.h.z + ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->h.z) / 2);
				}

				//-z coupling direction
				else if (nidx == 5 && (*pbulk_coupling_mask)[spin_index].z & 0xffff0000) {

					mesh_idx = ((*pbulk_coupling_mask)[spin_index].z >> 16) - 1;
					if (mesh_idx < paMesh_Bulk_size) cell_rel_pos = abs_pos - ppaMesh_Bulk[mesh_idx]->pM1->origin + cuReal3(0, 0, -(M1.h.z + ppaMesh_Bulk[mesh_idx]->pM1->h.z) / 2);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->origin + cuReal3(0, 0, -(M1.h.z + ppMeshFM_Bulk[mesh_idx - paMesh_Bulk_size]->pM->h.z) / 2);
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size + pMeshAFM_Bulk_size)
						cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->origin + cuReal3(0, 0, -(M1.h.z + ppMeshAFM_Bulk[mesh_idx - paMesh_Bulk_size - pMeshFM_Bulk_size]->pM->h.z) / 2);
				}

				if (mesh_idx >= 0) {

					num_couplings++;

					//coupling for atomistic mesh
					if (mesh_idx < paMesh_Bulk_size) {
						calculate_atom_coupling(
							M1, Mnew, spin_index,
							*ppaMesh_Bulk[mesh_idx], cell_rel_pos,
							Js,
							energy_old, energy_new);
					}
					//coupling for micromagnetic FM mesh
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size) {
						calculate_mm_FM_coupling(
							M1, Mnew, spin_index,
							*ppMeshFM_Bulk[mesh_idx], cell_rel_pos,
							Js,
							energy_old, energy_new);
					}
					//coupling for micromagnetic AFM mesh
					else if (mesh_idx < paMesh_Bulk_size + pMeshFM_Bulk_size + pMeshAFM_Bulk_size) {
						calculate_mm_AFM_coupling(
							M1, Mnew, spin_index,
							*ppMeshAFM_Bulk[mesh_idx], cell_rel_pos,
							Js, Js2,
							energy_old, energy_new);
					}
				}
				mesh_idx = -1;
			}

			if (num_couplings) {

				energy_old += energy_bulk_old / num_couplings;
				energy_new += energy_bulk_new / num_couplings;
			}
		}
	}

	//------------------

	if (Mnew != cuReal3()) return energy_new - energy_old;
	else return energy_old;
}


#endif