#include "SurfExchangeCUDA_AFM.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SURFEXCHANGE

#include "Reduction.cuh"
#include "cuVEC_VC_mcuVEC.cuh"

#include "Mesh_AntiFerromagneticCUDA.h"
#include "MeshParamsControlCUDA.h"

#include "ManagedAtom_MeshCUDA.h"

#include "MeshDefs.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------ SURFACE COUPLING Z STACKING

//Top mesh is ferromagnetic
__global__ void SurfExchangeCUDA_AFM_TopFM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedMeshCUDA** ppMesh_Top, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M.n;
	cuReal3 h = M.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x + (n.z - 1) * n.x*n.y;

		//skip empty cells
		if (M.is_not_empty(cell_idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuMesh.update_parameters_mcoarse(cell_idx, *cuMesh.pMs_AFM, Ms_AFM);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Top = ppMesh_Top[mesh_idx]->pM->mcuvec();

				//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M.rect.s.x - M_Top.rect.s.x,
					(j + 0.5) * h.y + M.rect.s.y - M_Top.rect.s.y,
					M_Top.h.z / 2);

				//can't couple to an empty cell
				if (!M_Top.rect.contains(cell_rel_pos + M_Top.rect.s) || M_Top.is_empty(cell_rel_pos)) continue;

				cuBReal J1 = *(ppMesh_Top[mesh_idx]->pJ1);
				cuBReal J2 = *(ppMesh_Top[mesh_idx]->pJ2);
				ppMesh_Top[mesh_idx]->update_parameters_atposition(cell_rel_pos, *(ppMesh_Top[mesh_idx]->pJ1), J1, *(ppMesh_Top[mesh_idx]->pJ2), J2);

				//get magnetization value in top mesh cell to couple with
				cuReal3 m_j = cu_normalize(M_Top[cell_rel_pos]);
				cuReal3 m_i1 = cu_normalize(M[cell_idx]);
				cuReal3 m_i2 = cu_normalize(M2[cell_idx]);

				//total surface exchange field in coupling cells, including bilinear and biquadratic terms
				cuReal3 Hsurfexch1 = (m_j / ((cuBReal)MU0 * Ms_AFM.i * h.z)) * J1;
				cuReal3 Hsurfexch2 = (m_j / ((cuBReal)MU0 * Ms_AFM.j * h.z)) * J2;

				cuBReal energy1_ = 0.0, energy2_ = 0.0;

				if (do_reduction) {

					energy1_ = (-J1 * (m_i1 * m_j)) / (h.z * M.get_nonempty_cells());
					energy2_ = (-J2 * (m_i2 * m_j)) / (h.z * M.get_nonempty_cells());
					energy_ = (energy1_ + energy2_) / 2;
				}

				Heff[cell_idx] += Hsurfexch1;
				Heff2[cell_idx] += Hsurfexch2;

				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch1;
				if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[cell_idx] += Hsurfexch2;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy1_ * M.get_nonempty_cells();
				if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[cell_idx] += energy2_ * M.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//Top mesh is antiferromagnetic
__global__ void SurfExchangeCUDA_AFM_TopAFM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedMeshCUDA** ppMesh_Top, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M.n;
	cuReal3 h = M.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x + (n.z - 1) * n.x*n.y;

		//skip empty cells
		if (M.is_not_empty(cell_idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuMesh.update_parameters_mcoarse(cell_idx, *cuMesh.pMs_AFM, Ms_AFM);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Top = ppMesh_Top[mesh_idx]->pM->mcuvec();
				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2_Top = ppMesh_Top[mesh_idx]->pM2->mcuvec();

				//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M.rect.s.x - M_Top.rect.s.x,
					(j + 0.5) * h.y + M.rect.s.y - M_Top.rect.s.y,
					M_Top.h.z / 2);

				//can't couple to an empty cell
				if (!M_Top.rect.contains(cell_rel_pos + M_Top.rect.s) || M_Top.is_empty(cell_rel_pos)) continue;

				cuBReal J1 = *(ppMesh_Top[mesh_idx]->pJ1);
				cuBReal J2 = *(ppMesh_Top[mesh_idx]->pJ2);
				ppMesh_Top[mesh_idx]->update_parameters_atposition(cell_rel_pos, *(ppMesh_Top[mesh_idx]->pJ1), J1, *(ppMesh_Top[mesh_idx]->pJ2), J2);

				//get magnetization value in top mesh cell to couple with
				cuReal3 m_j1 = cu_normalize(M_Top[cell_rel_pos]);
				cuReal3 m_j2 = cu_normalize(M2_Top[cell_rel_pos]);
				cuReal3 m_i1 = cu_normalize(M[cell_idx]);
				cuReal3 m_i2 = cu_normalize(M2[cell_idx]);

				//total surface exchange field in coupling cells, including bilinear and biquadratic terms
				cuReal3 Hsurfexch1 = (m_j1 / ((cuBReal)MU0 * Ms_AFM.i * h.z)) * J1;
				cuReal3 Hsurfexch2 = (m_j2 / ((cuBReal)MU0 * Ms_AFM.j * h.z)) * J2;

				cuBReal energy1_ = 0.0, energy2_ = 0.0;

				if (do_reduction) {

					energy1_ = (-J1 * (m_i1 * m_j1)) / (h.z * M.get_nonempty_cells());
					energy2_ = (-J2 * (m_i2 * m_j2)) / (h.z * M.get_nonempty_cells());
					energy_ = (energy1_ + energy2_) / 2;
				}

				Heff[cell_idx] += Hsurfexch1;
				Heff2[cell_idx] += Hsurfexch2;

				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch1;
				if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[cell_idx] += Hsurfexch2;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy1_ * M.get_nonempty_cells();
				if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[cell_idx] += energy2_ * M.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//Top mesh is atomistic
__global__ void SurfExchangeCUDA_AFM_TopAtom_UpdateField(ManagedMeshCUDA& cuMesh, ManagedAtom_MeshCUDA** ppMesh_Top, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M.n;
	cuReal3 h = M.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x + (n.z - 1) * n.x*n.y;

		//skip empty cells
		if (M.is_not_empty(cell_idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuBReal J1 = *cuMesh.pJ1;
			cuBReal J2 = *cuMesh.pJ2;
			cuMesh.update_parameters_mcoarse(cell_idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pJ1, J1, *cuMesh.pJ2, J2);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1 = ppMesh_Top[mesh_idx]->pM1->mcuvec();

				//coupling rectangle in atomistic mesh in absolute coordinates
				cuRect rect_c = cuRect(
					cuReal3(i * h.x, j * h.y, M.rect.e.z),
					cuReal3((i + 1) * h.x, (j + 1) * h.y, M1.h.z + M.rect.e.z));
				rect_c += cuReal3(M.rect.s.x, M.rect.s.y, 0.0);

				//cells box in atomistic mesh
				cuBox acells = M1.box_from_rect_min(rect_c);

				cuReal3 m_j = cuReal3();
				for (int ai = acells.s.i; ai < acells.e.i; ai++) {
					for (int aj = acells.s.j; aj < acells.e.j; aj++) {

						cuReal3 rel_pos = cuReal3((ai + 0.5) * M1.h.x, (aj + 0.5) * M1.h.y, M1.h.z / 2);

						if (M1.is_empty(rel_pos)) continue;

						m_j += M1[rel_pos];
					}
				}

				m_j = cu_normalize(m_j);
				cuReal3 m_i1 = cu_normalize(M[cell_idx]);
				cuReal3 m_i2 = cu_normalize(M2[cell_idx]);

				//total surface exchange field in coupling cells, including bilinear and biquadratic terms
				cuReal3 Hsurfexch1 = (m_j / ((cuBReal)MU0 * Ms_AFM.i * h.z)) * J1;
				cuReal3 Hsurfexch2 = (m_j / ((cuBReal)MU0 * Ms_AFM.j * h.z)) * J2;

				cuBReal energy1_ = 0.0, energy2_ = 0.0;

				if (do_reduction) {

					energy1_ = (-J1 * (m_i1 * m_j)) / (h.z * M.get_nonempty_cells());
					energy2_ = (-J2 * (m_i2 * m_j)) / (h.z * M.get_nonempty_cells());
					energy_ = (energy1_ + energy2_) / 2;
				}

				Heff[cell_idx] += Hsurfexch1;
				Heff2[cell_idx] += Hsurfexch2;

				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch1;
				if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[cell_idx] += Hsurfexch2;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy1_ * M.get_nonempty_cells();
				if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[cell_idx] += energy2_ * M.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//Bottom mesh is ferromagnetic
__global__ void SurfExchangeCUDA_AFM_BotFM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedMeshCUDA** ppMesh_Bot, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M.n;
	cuReal3 h = M.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x;

		//skip empty cells
		if (M.is_not_empty(cell_idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuBReal J1 = *cuMesh.pJ1;
			cuBReal J2 = *cuMesh.pJ2;
			cuMesh.update_parameters_mcoarse(cell_idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pJ1, J1, *cuMesh.pJ2, J2);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bot = ppMesh_Bot[mesh_idx]->pM->mcuvec();

				//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M.rect.s.x - M_Bot.rect.s.x,
					(j + 0.5) * h.y + M.rect.s.y - M_Bot.rect.s.y,
					M_Bot.rect.e.z - M_Bot.rect.s.z - M_Bot.h.z / 2);

				//can't couple to an empty cell
				if (!M_Bot.rect.contains(cell_rel_pos + M_Bot.rect.s) || M_Bot.is_empty(cell_rel_pos)) continue;

				//get value of magnetization used in coupling with current cell at cell_idx
				cuReal3 m_j = cu_normalize(M_Bot[cell_rel_pos]);
				cuReal3 m_i1 = cu_normalize(M[cell_idx]);
				cuReal3 m_i2 = cu_normalize(M2[cell_idx]);

				//total surface exchange field in coupling cells, including bilinear and biquadratic terms
				cuReal3 Hsurfexch = (m_j / ((cuBReal)MU0 * Ms_AFM.i * h.z)) * J1;
				cuReal3 Hsurfexch2 = (m_j / ((cuBReal)MU0 * Ms_AFM.j * h.z)) * J2;

				cuBReal energy1_ = 0.0, energy2_ = 0.0;

				if (do_reduction) {

					energy1_ = (-J1 * (m_i1 * m_j)) / (h.z * M.get_nonempty_cells());
					energy2_ = (-J2 * (m_i2 * m_j)) / (h.z * M.get_nonempty_cells());
					energy_ = (energy1_ + energy2_) / 2;
				}

				Heff[cell_idx] += Hsurfexch;
				Heff2[cell_idx] += Hsurfexch2;

				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch;
				if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[cell_idx] += Hsurfexch2;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy1_ * M.get_nonempty_cells();
				if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[cell_idx] += energy2_ * M.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//Bottom mesh is antiferromagnetic
__global__ void SurfExchangeCUDA_AFM_BotAFM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedMeshCUDA** ppMesh_Bot, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M.n;
	cuReal3 h = M.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x;

		//skip empty cells
		if (M.is_not_empty(cell_idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuBReal J1 = *cuMesh.pJ1;
			cuBReal J2 = *cuMesh.pJ2;
			cuMesh.update_parameters_mcoarse(cell_idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pJ1, J1, *cuMesh.pJ2, J2);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bot = ppMesh_Bot[mesh_idx]->pM->mcuvec();
				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2_Bot = ppMesh_Bot[mesh_idx]->pM2->mcuvec();

				//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M.rect.s.x - M_Bot.rect.s.x,
					(j + 0.5) * h.y + M.rect.s.y - M_Bot.rect.s.y,
					M_Bot.rect.e.z - M_Bot.rect.s.z - M_Bot.h.z / 2);

				//can't couple to an empty cell
				if (!M_Bot.rect.contains(cell_rel_pos + M_Bot.rect.s) || M_Bot.is_empty(cell_rel_pos)) continue;

				//yes, then get value of magnetization used in coupling with current cell at cell_idx
				cuReal3 m_j1 = cu_normalize(M_Bot[cell_rel_pos]);
				cuReal3 m_j2 = cu_normalize(M2_Bot[cell_rel_pos]);
				cuReal3 m_i1 = cu_normalize(M[cell_idx]);
				cuReal3 m_i2 = cu_normalize(M2[cell_idx]);

				//total surface exchange field in coupling cells, including bilinear and biquadratic terms
				cuReal3 Hsurfexch = (m_j1 / ((cuBReal)MU0 * Ms_AFM.i * h.z)) * J1;
				cuReal3 Hsurfexch2 = (m_j2 / ((cuBReal)MU0 * Ms_AFM.j * h.z)) * J2;

				cuBReal energy1_ = 0.0, energy2_ = 0.0;

				if (do_reduction) {

					energy1_ = (-J1 * (m_i1 * m_j1)) / (h.z * M.get_nonempty_cells());
					energy2_ = (-J2 * (m_i2 * m_j2)) / (h.z * M.get_nonempty_cells());
					energy_ = (energy1_ + energy2_) / 2;
				}

				Heff[cell_idx] += Hsurfexch;
				Heff2[cell_idx] += Hsurfexch2;

				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch;
				if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[cell_idx] += Hsurfexch2;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy1_ * M.get_nonempty_cells();
				if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[cell_idx] += energy2_ * M.get_nonempty_cells();

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				break;
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//Bottom mesh is atomistic
__global__ void SurfExchangeCUDA_AFM_BotAtom_UpdateField(ManagedMeshCUDA& cuMesh, ManagedAtom_MeshCUDA** ppMesh_Bot, size_t coupled_meshes, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M.n;
	cuReal3 h = M.h;

	cuBReal energy_ = 0.0;

	if (idx < n.x * n.y) {

		int i = idx % n.x;
		int j = idx / n.x;
		int cell_idx = i + j * n.x;

		//skip empty cells
		if (M.is_not_empty(cell_idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuBReal J1 = *cuMesh.pJ1;
			cuBReal J2 = *cuMesh.pJ2;
			cuMesh.update_parameters_mcoarse(cell_idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pJ1, J1, *cuMesh.pJ2, J2);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < coupled_meshes; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1 = ppMesh_Bot[mesh_idx]->pM1->mcuvec();

				//coupling rectangle in atomistic mesh in absolute coordinates
				cuRect rect_c = cuRect(
					cuReal3(i * h.x, j * h.y, M1.rect.e.z - M1.h.z),
					cuReal3((i + 1) * h.x, (j + 1) * h.y, M1.rect.e.z));
				rect_c += cuReal3(M.rect.s.x, M.rect.s.y, 0.0);

				//cells box in atomistic mesh
				cuBox acells = M1.box_from_rect_min(rect_c);

				cuReal3 m_j = cuReal3();
				for (int ai = acells.s.i; ai < acells.e.i; ai++) {
					for (int aj = acells.s.j; aj < acells.e.j; aj++) {

						cuReal3 rel_pos = cuReal3((ai + 0.5) * M1.h.x, (aj + 0.5) * M1.h.y, M1.rect.e.z - M1.h.z / 2);

						if (M1.is_empty(rel_pos)) continue;

						m_j += M1[rel_pos];
					}
				}

				m_j = cu_normalize(m_j);
				cuReal3 m_i1 = cu_normalize(M[cell_idx]);
				cuReal3 m_i2 = cu_normalize(M2[cell_idx]);

				//total surface exchange field in coupling cells, including bilinear and biquadratic terms
				cuReal3 Hsurfexch1 = (m_j / ((cuBReal)MU0 * Ms_AFM.i * h.z)) * J1;
				cuReal3 Hsurfexch2 = (m_j / ((cuBReal)MU0 * Ms_AFM.j * h.z)) * J2;

				cuBReal energy1_ = 0.0, energy2_ = 0.0;

				if (do_reduction) {

					energy1_ = (-J1 * (m_i1 * m_j)) / (h.z * M.get_nonempty_cells());
					energy2_ = (-J2 * (m_i2 * m_j)) / (h.z * M.get_nonempty_cells());
					energy_ = (energy1_ + energy2_) / 2;
				}

				Heff[cell_idx] += Hsurfexch1;
				Heff2[cell_idx] += Hsurfexch2;

				if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[cell_idx] += Hsurfexch1;
				if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[cell_idx] += Hsurfexch2;
				if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[cell_idx] += energy1_ * M.get_nonempty_cells();
				if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[cell_idx] += energy2_ * M.get_nonempty_cells();

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

__global__ void SurfExchangeAFMCUDA_Bulk_UpdateField(
	ManagedMeshCUDA& cuMesh,
	cuVEC<cuINT3>& bulk_coupling_mask,
	ManagedMeshCUDA** ppMeshFM_Bulk, size_t coupledFM_meshes,
	ManagedMeshCUDA** ppMeshAFM_Bulk, size_t coupledAFM_meshes,
	ManagedAtom_MeshCUDA** ppaMesh_Bulk, size_t coupled_ameshes,
	ManagedModulesCUDA& cuModule, bool do_reduction)
{
	//------------------ Coupling functions

	auto calculate_mm_FM_coupling = [](
		cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, int cell_idx,
		ManagedMeshCUDA& MeshCoupled, cuReal3 cell_rel_pos,
		cuReal2 Ms_AFM, cuBReal J1, cuBReal J2, cuBReal cell_size,
		cuReal3& Hsurfexch1, cuBReal& cell_energy1, 
		cuReal3& Hsurfexch2, cuBReal& cell_energy2,
		bool do_reduction) -> void
	{
		//NOTE : no need to check here if pMeshCoupled->M contains cell_rel_pos, or if cell is not empty there.
		//This check is done before calling this function for z stacking, or when initializing bulk_coupling_mask for bulk coupling.

		//Surface exchange field from a ferromagnetic mesh

		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bulk = MeshCoupled.pM->mcuvec();

		//get magnetization value in mesh cell to couple with
		cuReal3 m_j = cu_normalize(M_Bulk[cell_rel_pos]);
		cuReal3 m_i1 = cu_normalize(M[cell_idx]);
		cuReal3 m_i2 = cu_normalize(M2[cell_idx]);

		//total surface exchange field in coupling cells
		Hsurfexch1 += (m_j / ((cuBReal)MU0 * Ms_AFM.i * cell_size)) * J1;
		Hsurfexch2 += (m_j / ((cuBReal)MU0 * Ms_AFM.j * cell_size)) * J2;

		if (do_reduction) {

			cell_energy1 += (-J1 * (m_i1 * m_j)) / (cell_size * M.get_nonempty_cells());
			cell_energy2 += (-J2 * (m_i2 * m_j)) / (cell_size * M.get_nonempty_cells());
		}
	};

	auto calculate_mm_AFM_coupling = [](
		cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, int cell_idx,
		ManagedMeshCUDA& MeshCoupled, cuReal3 cell_rel_pos,
		cuReal2 Ms_AFM, cuBReal J1, cuBReal J2, cuBReal cell_size,
		cuReal3& Hsurfexch1, cuBReal& cell_energy1, 
		cuReal3& Hsurfexch2, cuBReal& cell_energy2,
		bool do_reduction) -> void
	{
		//NOTE : no need to check here if pMeshCoupled->M contains cell_rel_pos, or if cell is not empty there.
		//This check is done before calling this function for z stacking, or when initializing bulk_coupling_mask for bulk coupling.

		//Surface exchange field from an antiferromagnetic mesh

		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bulk = MeshCoupled.pM->mcuvec();
		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2_Bulk = MeshCoupled.pM2->mcuvec();

		//get magnetization values in mesh cell to couple with
		cuReal3 m_j1 = cu_normalize(M_Bulk[cell_rel_pos]);
		cuReal3 m_j2 = cu_normalize(M2_Bulk[cell_rel_pos]);
		cuReal3 m_i1 = cu_normalize(M[cell_idx]);
		cuReal3 m_i2 = cu_normalize(M2[cell_idx]);

		//total surface exchange field in coupling cells
		Hsurfexch1 += (m_j1 / ((cuBReal)MU0 * Ms_AFM.i * cell_size)) * J1;
		Hsurfexch2 += (m_j2 / ((cuBReal)MU0 * Ms_AFM.j * cell_size)) * J2;

		if (do_reduction) {

			cell_energy1 += (-J1 * (m_i1 * m_j1)) / (cell_size * M.get_nonempty_cells());
			cell_energy2 += (-J2 * (m_i2 * m_j2)) / (cell_size * M.get_nonempty_cells());
		}
	};

	auto calculate_atom_coupling = [](
		cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, int cell_idx,
		ManagedAtom_MeshCUDA& aMeshCoupled, cuRect& rect_c,
		cuReal2 Ms_AFM, cuBReal J1, cuBReal J2, cuBReal cell_size,
		cuReal3& Hsurfexch1, cuBReal& cell_energy1, 
		cuReal3& Hsurfexch2, cuBReal& cell_energy2,
		bool do_reduction) -> void
	{
		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1 = aMeshCoupled.pM1->mcuvec();

		//cells box in atomistic mesh. NOTE : acells is capped to mesh dimensions, so we are guaranteed proper indexes inside the mesh.
		cuBox acells = M1.box_from_rect_min(rect_c);

		cuReal3 m_j = cuReal3();
		for (int ai = acells.s.i; ai < acells.e.i; ai++) {
			for (int aj = acells.s.j; aj < acells.e.j; aj++) {
				for (int ak = acells.s.k; ak < acells.e.k; ak++) {

					cuReal3 rel_pos = cuReal3((ai + 0.5) * M1.h.x, (aj + 0.5) * M1.h.y, (ak + 0.5) * M1.h.z);

					if (M1.is_empty(rel_pos)) continue;

					m_j += M1[rel_pos];
				}
			}
		}

		m_j = cu_normalize(m_j);
		cuReal3 m_i1 = cu_normalize(M[cell_idx]);
		cuReal3 m_i2 = cu_normalize(M2[cell_idx]);

		//total surface exchange field in coupling cells
		Hsurfexch1 += (m_j / ((cuBReal)MU0 * Ms_AFM.i * cell_size)) * J1;
		Hsurfexch2 += (m_j / ((cuBReal)MU0 * Ms_AFM.j * cell_size)) * J2;

		if (do_reduction) {

			cell_energy1 += (-J1 * (m_i1 * m_j)) / (cell_size * M.get_nonempty_cells());
			cell_energy2 += (-J2 * (m_i2 * m_j)) / (cell_size * M.get_nonempty_cells());
		}
	};

	//------------------

	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = M.n;
	cuReal3 h = M.h;

	cuBReal energy_ = 0.0;

	if (idx < n.dim()) {

		int i = idx % n.x;
		int j = (idx / n.x) % n.y;
		int k = idx / (n.x * n.y);
		int idx = i + j * n.x + k * n.x * n.y;

		//skip empty cells
		if (M.is_not_empty(idx) && bulk_coupling_mask[idx] != cuINT3()) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuBReal J1 = *cuMesh.pJ1;
			cuBReal J2 = *cuMesh.pJ2;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pJ1, J1, *cuMesh.pJ2, J2);

			//surface cell which needs to be exchange coupled
			cuReal3 Hsurfexch1 = cuReal3(), Hsurfexch2 = cuReal3();
			cuBReal cell_energy1 = 0.0, cell_energy2 = 0.0;
			int num_couplings = 0;

			cuReal3 abs_pos = M.cellidx_to_position(idx) + M.rect.s;

			//+x coupling direction
			if (bulk_coupling_mask[idx].x & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[idx].x & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for micromagnetic FM mesh
				if (mesh_idx < coupledFM_meshes) {
					calculate_mm_FM_coupling(
						M, M2, idx,
						*ppMeshFM_Bulk[mesh_idx], abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3((M.h.x + ppMeshFM_Bulk[mesh_idx]->pM->h.x) / 2, 0, 0),
						Ms_AFM, J1, J2, M.h.x,
						Hsurfexch1, cell_energy1, 
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for micromagnetic AFM mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes) {
					calculate_mm_AFM_coupling(
						M, M2, idx,
						*ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes], abs_pos - ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->origin + cuReal3((M.h.x + ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->h.x) / 2, 0, 0),
						Ms_AFM, J1, J2, M.h.x,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for atomistic mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes + coupled_ameshes) {

					//coupling rectangle in atomistic mesh in absolute coordinates
					cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes]->pM1;
					cuRect rect_c = cuRect(
						cuReal3(abs_pos.x + M.h.x / 2, abs_pos.y - M.h.y / 2, abs_pos.z - M.h.z / 2),
						cuReal3(abs_pos.x + M.h.x / 2 + M1.h.x, abs_pos.y + M.h.y / 2, abs_pos.z + M.h.z / 2));

					calculate_atom_coupling(
						M, M2, idx,
						*ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes], rect_c,
						Ms_AFM, J1, J2, M.h.x,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
			}

			//-x coupling direction
			if (bulk_coupling_mask[idx].x & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[idx].x >> 16) - 1;
				num_couplings++;

				//coupling for micromagnetic FM mesh
				if (mesh_idx < coupledFM_meshes) {
					calculate_mm_FM_coupling(
						M, M2, idx,
						*ppMeshFM_Bulk[mesh_idx], abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3(-(M.h.x + ppMeshFM_Bulk[mesh_idx]->pM->h.x) / 2, 0, 0),
						Ms_AFM, J1, J2, M.h.x,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for micromagnetic AFM mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes) {
					calculate_mm_AFM_coupling(
						M, M2, idx,
						*ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes], abs_pos - ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->origin + cuReal3(-(M.h.x + ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->h.x) / 2, 0, 0),
						Ms_AFM, J1, J2, M.h.x,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for atomistic mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes + coupled_ameshes) {

					//coupling rectangle in atomistic mesh in absolute coordinates
					cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes]->pM1;
					cuRect rect_c = cuRect(
						cuReal3(abs_pos.x - M.h.x / 2 - M1.h.x, abs_pos.y - M.h.y / 2, abs_pos.z - M.h.z / 2),
						cuReal3(abs_pos.x - M.h.x / 2, abs_pos.y + M.h.y / 2, abs_pos.z + M.h.z / 2));

					calculate_atom_coupling(
						M, M2, idx,
						*ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes], rect_c,
						Ms_AFM, J1, J2, M.h.x,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
			}

			//+y coupling direction
			if (bulk_coupling_mask[idx].y & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[idx].y & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for micromagnetic FM mesh
				if (mesh_idx < coupledFM_meshes) {
					calculate_mm_FM_coupling(
						M, M2, idx,
						*ppMeshFM_Bulk[mesh_idx], abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3(0, (M.h.y + ppMeshFM_Bulk[mesh_idx]->pM->h.y) / 2, 0),
						Ms_AFM, J1, J2, M.h.y,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for micromagnetic AFM mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes) {
					calculate_mm_AFM_coupling(
						M, M2, idx,
						*ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes], abs_pos - ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->origin + cuReal3(0, (M.h.y + ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->h.y) / 2, 0),
						Ms_AFM, J1, J2, M.h.y,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for atomistic mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes + coupled_ameshes) {

					//coupling rectangle in atomistic mesh in absolute coordinates
					cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes]->pM1;
					cuRect rect_c = cuRect(
						cuReal3(abs_pos.x - M.h.x / 2, abs_pos.y + M.h.y / 2, abs_pos.z - M.h.z / 2),
						cuReal3(abs_pos.x + M.h.x / 2, abs_pos.y + M.h.y / 2 + M1.h.y, abs_pos.z + M.h.z / 2));

					calculate_atom_coupling(
						M, M2, idx,
						*ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes], rect_c,
						Ms_AFM, J1, J2, M.h.y,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
			}

			//-y coupling direction
			if (bulk_coupling_mask[idx].y & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[idx].y >> 16) - 1;
				num_couplings++;

				//coupling for micromagnetic FM mesh
				if (mesh_idx < coupledFM_meshes) {
					calculate_mm_FM_coupling(
						M, M2, idx,
						*ppMeshFM_Bulk[mesh_idx], abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3(0, -(M.h.y + ppMeshFM_Bulk[mesh_idx]->pM->h.y) / 2, 0),
						Ms_AFM, J1, J2, M.h.y,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for micromagnetic AFM mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes) {
					calculate_mm_AFM_coupling(
						M, M2, idx,
						*ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes], abs_pos - ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->origin + cuReal3(0, -(M.h.y + ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->h.y) / 2, 0),
						Ms_AFM, J1, J2, M.h.y,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for atomistic mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes + coupled_ameshes) {

					//coupling rectangle in atomistic mesh in absolute coordinates
					cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes]->pM1;
					cuRect rect_c = cuRect(
						cuReal3(abs_pos.x - M.h.x / 2, abs_pos.y - M.h.y / 2 - M1.h.y, abs_pos.z - M.h.z / 2),
						cuReal3(abs_pos.x + M.h.x / 2, abs_pos.y - M.h.y / 2, abs_pos.z + M.h.z / 2));

					calculate_atom_coupling(
						M, M2, idx,
						*ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes], rect_c,
						Ms_AFM, J1, J2, M.h.y,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
			}

			//+z coupling direction
			if (bulk_coupling_mask[idx].z & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[idx].z & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for micromagnetic FM mesh
				if (mesh_idx < coupledFM_meshes) {
					calculate_mm_FM_coupling(
						M, M2, idx,
						*ppMeshFM_Bulk[mesh_idx], abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3(0, 0, (M.h.z + ppMeshFM_Bulk[mesh_idx]->pM->h.z) / 2),
						Ms_AFM, J1, J2, M.h.z,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for micromagnetic AFM mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes) {
					calculate_mm_AFM_coupling(
						M, M2, idx,
						*ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes], abs_pos - ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->origin + cuReal3(0, 0, (M.h.z + ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->h.z) / 2),
						Ms_AFM, J1, J2, M.h.z,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for atomistic mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes + coupled_ameshes) {

					//coupling rectangle in atomistic mesh in absolute coordinates
					cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes]->pM1;
					cuRect rect_c = cuRect(
						cuReal3(abs_pos.x - M.h.x / 2, abs_pos.y - M.h.y / 2, abs_pos.z + M.h.z / 2),
						cuReal3(abs_pos.x + M.h.x / 2, abs_pos.y + M.h.y / 2, abs_pos.z + M.h.z / 2 + M1.h.z));

					calculate_atom_coupling(
						M, M2, idx,
						*ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes], rect_c,
						Ms_AFM, J1, J2, M.h.z,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
			}

			//-z coupling direction
			if (bulk_coupling_mask[idx].z & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[idx].z >> 16) - 1;
				num_couplings++;

				//coupling for micromagnetic FM mesh
				if (mesh_idx < coupledFM_meshes) {
					calculate_mm_FM_coupling(
						M, M2, idx,
						*ppMeshFM_Bulk[mesh_idx], abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3(0, 0, -(M.h.z + ppMeshFM_Bulk[mesh_idx]->pM->h.z) / 2),
						Ms_AFM, J1, J2, M.h.z,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for micromagnetic AFM mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes) {
					calculate_mm_AFM_coupling(
						M, M2, idx,
						*ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes], abs_pos - ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->origin + cuReal3(0, 0, -(M.h.z + ppMeshAFM_Bulk[mesh_idx - coupledFM_meshes]->pM->h.z) / 2),
						Ms_AFM, J1, J2, M.h.z,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
				//coupling for atomistic mesh
				else if (mesh_idx < coupledFM_meshes + coupledAFM_meshes + coupled_ameshes) {

					//coupling rectangle in atomistic mesh in absolute coordinates
					cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes]->pM1;
					cuRect rect_c = cuRect(
						cuReal3(abs_pos.x - M.h.x / 2, abs_pos.y - M.h.y / 2, abs_pos.z - M.h.z / 2 - M1.h.z),
						cuReal3(abs_pos.x + M.h.x / 2, abs_pos.y + M.h.y / 2, abs_pos.z - M.h.z / 2));

					calculate_atom_coupling(
						M, M2, idx,
						*ppaMesh_Bulk[mesh_idx - coupledFM_meshes - coupledAFM_meshes], rect_c,
						Ms_AFM, J1, J2, M.h.z,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2,
						do_reduction);
				}
			}

			if (num_couplings) {

				//need average if cell receives multiple coupling contributions
				Hsurfexch1 /= num_couplings;
				Hsurfexch2 /= num_couplings;
				cell_energy1 /= num_couplings;
				cell_energy2 /= num_couplings;
				energy_ = (cell_energy1 + cell_energy2) / 2;
			}

			Heff[idx] += Hsurfexch1;
			Heff2[idx] += Hsurfexch2;

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] += Hsurfexch1;
			if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[idx] += Hsurfexch2;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] += cell_energy1 * M.get_nonempty_cells();
			if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[idx] += cell_energy2 * M.get_nonempty_cells();

		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void SurfExchangeCUDA_AFM::UpdateField(void)
{
	if (pMeshCUDA->CurrentTimeStepSolved()) {

		ZeroEnergy();
		ZeroModuleVECs();

		//------------------ SURFACE COUPLING Z STACKING

		//Coupling from ferromagnetic meshes

		//Top
		if (pMeshFM_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_TopFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshFM_Top.get_array(mGPU), pMeshFM_Top.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//Bottom
		if (pMeshFM_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_BotFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshFM_Bot.get_array(mGPU), pMeshFM_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//Coupling from antiferromagnetic meshes

		//Top
		if (pMeshAFM_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_TopAFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshAFM_Top.get_array(mGPU), pMeshAFM_Top.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//Bottom
		if (pMeshAFM_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_BotAFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshAFM_Bot.get_array(mGPU), pMeshAFM_Bot.size(), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//Coupling from atomistic meshes

		//Top
		if (pMeshAtom_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_TopAtom_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshAtom_Top.get_array(mGPU), pMeshAtom_Top.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//Bottom
		if (pMeshAtom_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_BotAtom_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshAtom_Bot.get_array(mGPU), pMeshAtom_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}

		//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

		if (pMeshFM_Bulk.size() + pMeshAFM_Bulk.size() + paMesh_Bulk.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				SurfExchangeAFMCUDA_Bulk_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU),
						bulk_coupling_mask.get_deviceobject(mGPU),
						pMeshFM_Bulk.get_array(mGPU), pMeshFM_Bulk.size(mGPU),
						pMeshAFM_Bulk.get_array(mGPU), pMeshAFM_Bulk.size(mGPU),
						paMesh_Bulk.get_array(mGPU), paMesh_Bulk.size(mGPU),
						cuModule.get_deviceobject(mGPU), true);
			}
		}
	}
	else {

		//------------------ SURFACE COUPLING Z STACKING

		//Coupling from ferromagnetic meshes

		//Top
		if (pMeshFM_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_TopFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshFM_Top.get_array(mGPU), pMeshFM_Top.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//Bottom
		if (pMeshFM_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_BotFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshFM_Bot.get_array(mGPU), pMeshFM_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//Coupling from antiferromagnetic meshes

		//Top
		if (pMeshAFM_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_TopAFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshAFM_Top.get_array(mGPU), pMeshAFM_Top.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//Bottom
		if (pMeshAFM_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_BotAFM_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshAFM_Bot.get_array(mGPU), pMeshAFM_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//Coupling from atomistic meshes

		//Top
		if (pMeshAtom_Top.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_TopAtom_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshAtom_Top.get_array(mGPU), pMeshAtom_Top.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//Bottom
		if (pMeshAtom_Bot.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				cuSZ3 dn = pMeshCUDA->M.device_n(mGPU);
				SurfExchangeCUDA_AFM_BotAtom_UpdateField <<< (dn.x * dn.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), pMeshAtom_Bot.get_array(mGPU), pMeshAtom_Bot.size(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}

		//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

		if (pMeshFM_Bulk.size() + pMeshAFM_Bulk.size() + paMesh_Bulk.size()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				SurfExchangeAFMCUDA_Bulk_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU),
						bulk_coupling_mask.get_deviceobject(mGPU),
						pMeshFM_Bulk.get_array(mGPU), pMeshFM_Bulk.size(mGPU),
						pMeshAFM_Bulk.get_array(mGPU), pMeshAFM_Bulk.size(mGPU),
						paMesh_Bulk.get_array(mGPU), paMesh_Bulk.size(mGPU),
						cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
}

//----------------------- Initialization

//Current mesh is anti-ferromagnetic
__global__ void set_SurfExchangeCUDA_AFM_pointers_kernel(
	ManagedMeshCUDA& cuMesh,
	ManagedMeshCUDA** ppMeshFM_Bot, size_t coupledFM_bot_meshes,
	ManagedMeshCUDA** ppMeshFM_Top, size_t coupledFM_top_meshes,
	ManagedMeshCUDA** ppMeshAFM_Bot, size_t coupledAFM_bot_meshes,
	ManagedMeshCUDA** ppMeshAFM_Top, size_t coupledAFM_top_meshes,
	ManagedAtom_MeshCUDA** ppMeshAtom_Bot, size_t coupledAtom_bot_meshes,
	ManagedAtom_MeshCUDA** ppMeshAtom_Top, size_t coupledAtom_top_meshes,
	ManagedMeshCUDA** ppMeshFM_Bulk, size_t pMeshFM_Bulk_size,
	ManagedMeshCUDA** ppMeshAFM_Bulk, size_t pMeshAFM_Bulk_size,
	ManagedAtom_MeshCUDA** ppaMesh_Bulk, size_t paMesh_Bulk_size,
	cuVEC<cuINT3>& bulk_coupling_mask)
{
	if (threadIdx.x == 0) cuMesh.ppMeshFM_Bot = ppMeshFM_Bot;
	if (threadIdx.x == 1) cuMesh.pMeshFM_Bot_size = coupledFM_bot_meshes;
	if (threadIdx.x == 2) cuMesh.ppMeshFM_Top = ppMeshFM_Top;
	if (threadIdx.x == 3) cuMesh.pMeshFM_Top_size = coupledFM_top_meshes;

	if (threadIdx.x == 4) cuMesh.ppMeshAFM_Bot = ppMeshAFM_Bot;
	if (threadIdx.x == 5) cuMesh.pMeshAFM_Bot_size = coupledAFM_bot_meshes;
	if (threadIdx.x == 6) cuMesh.ppMeshAFM_Top = ppMeshAFM_Top;
	if (threadIdx.x == 7) cuMesh.pMeshAFM_Top_size = coupledAFM_top_meshes;

	if (threadIdx.x == 8) cuMesh.ppMeshAtom_Bot = ppMeshAtom_Bot;
	if (threadIdx.x == 9) cuMesh.pMeshAtom_Bot_size = coupledAtom_bot_meshes;
	if (threadIdx.x == 10) cuMesh.ppMeshAtom_Top = ppMeshAtom_Top;
	if (threadIdx.x == 11) cuMesh.pMeshAtom_Top_size = coupledAtom_top_meshes;
	
	if (threadIdx.x == 12) cuMesh.ppMeshFM_Bulk = ppMeshFM_Bulk;
	if (threadIdx.x == 13) cuMesh.pMeshFM_Bulk_size = pMeshFM_Bulk_size;
	if (threadIdx.x == 14) cuMesh.ppMeshAFM_Bulk = ppMeshAFM_Bulk;
	if (threadIdx.x == 15) cuMesh.pMeshAFM_Bulk_size = pMeshAFM_Bulk_size;
	if (threadIdx.x == 16) cuMesh.ppaMesh_Bulk = ppaMesh_Bulk;
	if (threadIdx.x == 17) cuMesh.paMesh_Bulk_size = paMesh_Bulk_size;
	if (threadIdx.x == 18) cuMesh.pbulk_coupling_mask = &bulk_coupling_mask;
}

//Called by SurfExchangeCUDA module
void SurfExchangeCUDA_AFM::set_SurfExchangeCUDA_AFM_pointers(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		set_SurfExchangeCUDA_AFM_pointers_kernel <<< 1, CUDATHREADS >>>
			(pMeshCUDA->cuMesh.get_deviceobject(mGPU),
			pMeshFM_Bot.get_array(mGPU), pMeshFM_Bot.size(mGPU), pMeshFM_Top.get_array(mGPU), pMeshFM_Top.size(mGPU),
			pMeshAFM_Bot.get_array(mGPU), pMeshAFM_Bot.size(mGPU), pMeshAFM_Top.get_array(mGPU), pMeshAFM_Top.size(mGPU),
			pMeshAtom_Bot.get_array(mGPU), pMeshAtom_Bot.size(mGPU), pMeshAtom_Top.get_array(mGPU), pMeshAtom_Top.size(mGPU),
			pMeshFM_Bulk.get_array(mGPU), pMeshFM_Bulk.size(mGPU),
			pMeshAFM_Bulk.get_array(mGPU), pMeshAFM_Bulk.size(mGPU),
			paMesh_Bulk.get_array(mGPU), paMesh_Bulk.size(mGPU),
			bulk_coupling_mask.get_deviceobject(mGPU));
	}
}

#endif

#endif

#if COMPILECUDA == 1 && MONTE_CARLO == 1

//Ferromagnetic

//Antiferromagnetic
__device__ cuReal2 ManagedMeshCUDA::Get_EnergyChange_AFM_SurfExchangeCUDA(int spin_index, cuReal3 Mnew_A, cuReal3 Mnew_B)
{
	cuReal2 energy_new = cuReal2(), energy_old = cuReal2();

	cuVEC_VC<cuReal3>& M = *pM;
	cuVEC_VC<cuReal3>& M2 = *pM2;

	cuSZ3 n = M.n;
	cuReal3 h = M.h;

	//------------------ Coupling functions

	auto calculate_mm_FM_coupling = [](
		cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3 Mnew_A, cuReal3 Mnew_B, int spin_index,
		ManagedMeshCUDA& MeshCoupled, cuReal3& cell_rel_pos,
		cuBReal J1, cuBReal J2, cuBReal cell_size,
		cuReal2& energy_old, cuReal2& energy_new) -> void
	{
		//Surface exchange field from a ferromagnetic mesh

		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bulk = MeshCoupled.pM->mcuvec();

		cuReal3 m_j = cu_normalize(M_Bulk[cell_rel_pos]);
		cuReal3 m_i1 = cu_normalize(M[spin_index]);
		cuReal3 m_i2 = cu_normalize(M2[spin_index]);

		energy_old.i += (-J1 * (m_i1 * m_j)) / cell_size;
		energy_old.j += (-J2 * (m_i2 * m_j)) / cell_size;

		if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) {

			cuReal3 mnew_i1 = cu_normalize(Mnew_A);
			cuReal3 mnew_i2 = cu_normalize(Mnew_B);

			energy_new.i += (-J1 * (mnew_i1 * m_j)) / cell_size;
			energy_new.j += (-J2 * (mnew_i2 * m_j)) / cell_size;
		}
	};

	auto calculate_mm_AFM_coupling = [](
		cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3 Mnew_A, cuReal3 Mnew_B, int spin_index,
		ManagedMeshCUDA& MeshCoupled, cuReal3& cell_rel_pos,
		cuBReal J1, cuBReal J2, cuBReal cell_size,
		cuReal2& energy_old, cuReal2& energy_new) -> void
	{
		//Surface exchange field from an antiferromagnetic mesh (exchange bias)

		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bulk = MeshCoupled.pM->mcuvec();
		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2_Bulk = MeshCoupled.pM2->mcuvec();

		cuReal3 m_j1 = cu_normalize(M_Bulk[cell_rel_pos]);
		cuReal3 m_j2 = cu_normalize(M2_Bulk[cell_rel_pos]);
		cuReal3 m_i1 = cu_normalize(M[spin_index]);
		cuReal3 m_i2 = cu_normalize(M2[spin_index]);

		energy_old.i += (-J1 * (m_i1 * m_j1)) / cell_size;
		energy_old.j += (-J2 * (m_i2 * m_j2)) / cell_size;

		if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) {

			cuReal3 mnew_i1 = cu_normalize(Mnew_A);
			cuReal3 mnew_i2 = cu_normalize(Mnew_B);

			energy_new.i += (-J1 * (mnew_i1 * m_j1)) / cell_size;
			energy_new.j += (-J2 * (mnew_i2 * m_j2)) / cell_size;
		}
	};

	auto calculate_atom_coupling = [](
		cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3 Mnew_A, cuReal3 Mnew_B, int spin_index,
		ManagedAtom_MeshCUDA& aMeshCoupled, cuRect& rect_c,
		cuBReal J1, cuBReal J2, cuBReal cell_size,
		cuReal2& energy_old, cuReal2& energy_new) -> void
	{
		mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1 = aMeshCoupled.pM1->mcuvec();

		//cells box in atomistic mesh
		cuBox acells = M1.box_from_rect_min(rect_c);

		cuReal3 m_j = cuReal3();
		for (int ai = acells.s.i; ai < acells.e.i; ai++) {
			for (int aj = acells.s.j; aj < acells.e.j; aj++) {
				for (int ak = acells.s.k; ak < acells.e.k; ak++) {

					cuReal3 rel_pos = cuReal3((ai + 0.5) * M1.h.x, (aj + 0.5) * M1.h.y, (ak + 0.5) * M1.h.z);

					if (M1.is_empty(rel_pos)) continue;

					m_j += M1[rel_pos];
				}
			}
		}

		m_j = cu_normalize(m_j);
		cuReal3 m_i1 = cu_normalize(M[spin_index]);
		cuReal3 m_i2 = cu_normalize(M2[spin_index]);

		energy_old.i += (-J1 * (m_i1 * m_j)) / cell_size;
		energy_old.j += (-J2 * (m_i2 * m_j)) / cell_size;

		if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) {

			cuReal3 mnew_i1 = cu_normalize(Mnew_A);
			cuReal3 mnew_i2 = cu_normalize(Mnew_B);

			energy_new.i += (-J1 * (mnew_i1 * m_j)) / cell_size;
			energy_new.j += (-J2 * (mnew_i2 * m_j)) / cell_size;
		}
	};

	//------------------ SURFACE COUPLING Z STACKING

	//if spin is on top surface then look at paMesh_Top
	if (spin_index / (n.x * n.y) == n.z - 1 && (pMeshFM_Top_size + pMeshAFM_Top_size + pMeshAtom_Top_size > 0)) {

		if (!M.is_empty(spin_index)) {

			int i = spin_index % n.x;
			int j = (spin_index / n.x) % n.y;

			bool cell_coupled = false;

			//check all meshes for coupling : FM meshes first
			for (int mesh_idx = 0; mesh_idx < (int)pMeshFM_Top_size; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Top = ppMeshFM_Top[mesh_idx]->pM->mcuvec();

				//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M.rect.s.x - M_Top.rect.s.x,
					(j + 0.5) * h.y + M.rect.s.y - M_Top.rect.s.y,
					M_Top.h.z / 2);

				//can't couple to an empty cell
				if (!M_Top.rect.contains(cell_rel_pos + M_Top.rect.s) || M_Top.is_empty(cell_rel_pos)) continue;

				//Surface exchange field from a ferromagnetic mesh

				//Top mesh sets J1 and J2 values
				cuBReal J1 = *(ppMeshFM_Top[mesh_idx]->pJ1);
				cuBReal J2 = *(ppMeshFM_Top[mesh_idx]->pJ2);
				ppMeshFM_Top[mesh_idx]->update_parameters_atposition(cell_rel_pos, *(ppMeshFM_Top[mesh_idx]->pJ1), J1, *(ppMeshFM_Top[mesh_idx]->pJ2), J2);

				calculate_mm_FM_coupling(
					M, M2, Mnew_A, Mnew_B, spin_index,
					*ppMeshFM_Top[mesh_idx], cell_rel_pos,
					J1, J2, h.z,
					energy_old, energy_new);

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				cell_coupled = true;
				break;
			}

			if (!cell_coupled) {

				//next AFM meshes
				for (int mesh_idx = 0; mesh_idx < (int)pMeshAFM_Top_size; mesh_idx++) {

					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Top = ppMeshAFM_Top[mesh_idx]->pM->mcuvec();
					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2_Top = ppMeshAFM_Top[mesh_idx]->pM2->mcuvec();

					//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
					cuReal3 cell_rel_pos = cuReal3(
						(i + 0.5) * h.x + M.rect.s.x - M_Top.rect.s.x,
						(j + 0.5) * h.y + M.rect.s.y - M_Top.rect.s.y,
						M_Top.h.z / 2);

					//can't couple to an empty cell
					if (!M_Top.rect.contains(cell_rel_pos + M_Top.rect.s) || M_Top.is_empty(cell_rel_pos)) continue;

					//Surface exchange field from an antiferromagnetic mesh (exchange bias)

					//Top mesh sets J1 and J2 values
					cuBReal J1 = *(ppMeshAFM_Top[mesh_idx]->pJ1);
					cuBReal J2 = *(ppMeshAFM_Top[mesh_idx]->pJ2);
					ppMeshAFM_Top[mesh_idx]->update_parameters_atposition(cell_rel_pos, *(ppMeshAFM_Top[mesh_idx]->pJ1), J1, *(ppMeshAFM_Top[mesh_idx]->pJ2), J2);

					calculate_mm_AFM_coupling(
						M, M2, Mnew_A, Mnew_B, spin_index,
						*ppMeshAFM_Top[mesh_idx], cell_rel_pos,
						J1, J2, h.z,
						energy_old, energy_new);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					cell_coupled = true;
					break;
				}
			}

			if (!cell_coupled) {

				//next atomistic meshes
				for (int mesh_idx = 0; mesh_idx < (int)pMeshAtom_Top_size; mesh_idx++) {

					//coupling rectangle in atomistic mesh in absolute coordinates
					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1 = ppMeshAtom_Top[mesh_idx]->pM1->mcuvec();
					cuRect rect_c = cuRect(
						cuReal3(i * h.x, j * h.y, M.rect.e.z),
						cuReal3((i + 1) * h.x, (j + 1) * h.y, M1.h.z + M.rect.e.z));
					rect_c += cuReal3(M.rect.s.x, M.rect.s.y, 0.0);

					//current mesh sets coupling in micromagnetic to atomistic meshes coupling
					cuBReal J1 = *pJ1;
					cuBReal J2 = *pJ2;
					update_parameters_mcoarse(spin_index, *pJ1, J1, *pJ2, J2);

					calculate_atom_coupling(
						M, M2, Mnew_A, Mnew_B, spin_index,
						*ppMeshAtom_Top[mesh_idx], rect_c,
						J1, J2, h.z,
						energy_old, energy_new);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					break;
				}
			}
		}
	}

	if (spin_index / (n.x * n.y) == 0 && (pMeshFM_Bot_size + pMeshAFM_Bot_size + pMeshAtom_Bot_size > 0)) {

		//surface exchange coupling at the bottom

		if (!M.is_empty(spin_index)) {

			int i = spin_index % n.x;
			int j = (spin_index / n.x) % n.y;

			cuBReal J1 = *pJ1;
			cuBReal J2 = *pJ2;
			update_parameters_mcoarse(spin_index, *pJ1, J1, *pJ2, J2);

			bool cell_coupled = false;

			//check all meshes for coupling : FM meshes first
			for (int mesh_idx = 0; mesh_idx < (int)pMeshFM_Bot_size; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bot = ppMeshFM_Bot[mesh_idx]->pM->mcuvec();

				//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h.x + M.rect.s.x - M_Bot.rect.s.x,
					(j + 0.5) * h.y + M.rect.s.y - M_Bot.rect.s.y,
					M_Bot.rect.e.z - M_Bot.rect.s.z - M_Bot.h.z / 2);

				//can't couple to an empty cell
				if (!M_Bot.rect.contains(cell_rel_pos + M_Bot.rect.s) || M_Bot.is_empty(cell_rel_pos)) continue;

				//Surface exchange field from a ferromagnetic mesh

				calculate_mm_FM_coupling(
					M, M2, Mnew_A, Mnew_B, spin_index,
					*ppMeshFM_Bot[mesh_idx], cell_rel_pos,
					J1, J2, h.z,
					energy_old, energy_new);

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				cell_coupled = true;
				break;
			}

			if (!cell_coupled) {

				//next AFM meshes
				for (int mesh_idx = 0; mesh_idx < (int)pMeshAFM_Bot_size; mesh_idx++) {

					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bot = ppMeshAFM_Bot[mesh_idx]->pM->mcuvec();
					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M2_Bot = ppMeshAFM_Bot[mesh_idx]->pM2->mcuvec();

					//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
					cuReal3 cell_rel_pos = cuReal3(
						(i + 0.5) * h.x + M.rect.s.x - M_Bot.rect.s.x,
						(j + 0.5) * h.y + M.rect.s.y - M_Bot.rect.s.y,
						M_Bot.rect.e.z - M_Bot.rect.s.z - M_Bot.h.z / 2);

					//can't couple to an empty cell
					if (!M_Bot.rect.contains(cell_rel_pos + M_Bot.rect.s) || M_Bot.is_empty(cell_rel_pos)) continue;

					//Surface exchange field from an antiferromagnetic mesh (exchange bias)

					calculate_mm_AFM_coupling(
						M, M2, Mnew_A, Mnew_B, spin_index,
						*ppMeshAFM_Bot[mesh_idx], cell_rel_pos,
						J1, J2, h.z,
						energy_old, energy_new);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					cell_coupled = true;
					break;
				}
			}

			if (!cell_coupled) {

				//next atomistic meshes
				for (int mesh_idx = 0; mesh_idx < (int)pMeshAtom_Bot_size; mesh_idx++) {

					//coupling rectangle in atomistic mesh in absolute coordinates
					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1 = ppMeshAtom_Bot[mesh_idx]->pM1->mcuvec();
					cuRect rect_c = cuRect(
						cuReal3(i * h.x, j * h.y, M1.rect.e.z - M1.h.z),
						cuReal3((i + 1) * h.x, (j + 1) * h.y, M1.rect.e.z));
					rect_c += cuReal3(M.rect.s.x, M.rect.s.y, 0.0);

					calculate_atom_coupling(
						M, M2, Mnew_A, Mnew_B, spin_index,
						*ppMeshAtom_Bot[mesh_idx], rect_c,
						J1, J2, h.z,
						energy_old, energy_new);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					break;
				}
			}
		}
	}

	//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

	if (pMeshFM_Bulk_size + pMeshAFM_Bulk_size + paMesh_Bulk_size) {

		if (M.is_not_empty(spin_index) && (*pbulk_coupling_mask)[spin_index] != cuINT3()) {

			cuReal2 energy_bulk_new = 0, energy_bulk_old = 0;

			cuBReal J1 = *pJ1;
			cuBReal J2 = *pJ2;
			update_parameters_mcoarse(spin_index, *pJ1, J1, *pJ2, J2);

			int num_couplings = 0;

			cuReal3 abs_pos = M.cellidx_to_position(spin_index) + M.rect.s;

			cuReal3 cell_rel_pos;
			int mesh_idx = -1;
			cuBReal cell_size = 0.0;
			cuRect rect_c;

			for (int nidx = 0; nidx < 6; nidx++) {

				//+x coupling direction
				if (nidx == 0 && (*pbulk_coupling_mask)[spin_index].x & 0x0000ffff) {

					mesh_idx = ((*pbulk_coupling_mask)[spin_index].x & 0x0000ffff) - 1;
					cell_size = M.h.x;
					if (mesh_idx < pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3((M.h.x + ppMeshFM_Bulk[mesh_idx]->pM->h.x) / 2, 0, 0);
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->origin + cuReal3((M.h.x + ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->h.x) / 2, 0, 0);
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size + paMesh_Bulk_size) {

						cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - pMeshFM_Bulk_size - pMeshAFM_Bulk_size]->pM1;
						rect_c = cuRect(
							cuReal3(abs_pos.x + M.h.x / 2, abs_pos.y - M.h.y / 2, abs_pos.z - M.h.z / 2),
							cuReal3(abs_pos.x + M.h.x / 2 + M1.h.x, abs_pos.y + M.h.y / 2, abs_pos.z + M.h.z / 2));
					}
				}

				//-x coupling direction
				else if (nidx == 1 && (*pbulk_coupling_mask)[spin_index].x & 0xffff0000) {

					mesh_idx = mesh_idx = ((*pbulk_coupling_mask)[spin_index].x >> 16) - 1;
					cell_size = M.h.x;
					//coupling for micromagnetic FM mesh
					if (mesh_idx < pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3(-(M.h.x + ppMeshFM_Bulk[mesh_idx]->pM->h.x) / 2, 0, 0);
					//coupling for micromagnetic AFM mesh
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->origin + cuReal3(-(M.h.x + ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->h.x) / 2, 0, 0);
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size + paMesh_Bulk_size) {

						cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - pMeshFM_Bulk_size - pMeshAFM_Bulk_size]->pM1;
						rect_c = cuRect(
							cuReal3(abs_pos.x - M.h.x / 2 - M1.h.x, abs_pos.y - M.h.y / 2, abs_pos.z - M.h.z / 2),
							cuReal3(abs_pos.x - M.h.x / 2, abs_pos.y + M.h.y / 2, abs_pos.z + M.h.z / 2));
					}
				}

				//+y coupling direction
				else if (nidx == 2 && (*pbulk_coupling_mask)[spin_index].y & 0x0000ffff) {

					mesh_idx = ((*pbulk_coupling_mask)[spin_index].y & 0x0000ffff) - 1;
					cell_size = M.h.y;
					//coupling for micromagnetic mesh
					if (mesh_idx < pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3(0, (M.h.y + ppMeshFM_Bulk[mesh_idx]->pM->h.y) / 2, 0);
					//coupling for micromagnetic AFM mesh
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->origin + cuReal3(0, (M.h.y + ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->h.y) / 2, 0);
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size + paMesh_Bulk_size) {

						cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - pMeshFM_Bulk_size - pMeshAFM_Bulk_size]->pM1;
						rect_c = cuRect(
							cuReal3(abs_pos.x - M.h.x / 2, abs_pos.y + M.h.y / 2, abs_pos.z - M.h.z / 2),
							cuReal3(abs_pos.x + M.h.x / 2, abs_pos.y + M.h.y / 2 + M1.h.y, abs_pos.z + M.h.z / 2));
					}
				}

				//-y coupling direction
				else if (nidx == 3 && (*pbulk_coupling_mask)[spin_index].y & 0xffff0000) {

					mesh_idx = ((*pbulk_coupling_mask)[spin_index].y >> 16) - 1;
					cell_size = M.h.y;
					//coupling for micromagnetic mesh
					if (mesh_idx < pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3(0, -(M.h.y + ppMeshFM_Bulk[mesh_idx]->pM->h.y) / 2, 0);
					//coupling for micromagnetic AFM mesh
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->origin + cuReal3(0, -(M.h.y + ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->h.y) / 2, 0);
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size + paMesh_Bulk_size) {

						cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - pMeshFM_Bulk_size - pMeshAFM_Bulk_size]->pM1;
						rect_c = cuRect(
							cuReal3(abs_pos.x - M.h.x / 2, abs_pos.y - M.h.y / 2 - M1.h.y, abs_pos.z - M.h.z / 2),
							cuReal3(abs_pos.x + M.h.x / 2, abs_pos.y - M.h.y / 2, abs_pos.z + M.h.z / 2));
					}
				}

				//+z coupling direction
				else if (nidx == 4 && (*pbulk_coupling_mask)[spin_index].z & 0x0000ffff) {

					mesh_idx = ((*pbulk_coupling_mask)[spin_index].z & 0x0000ffff) - 1;
					cell_size = M.h.z;
					//coupling for micromagnetic mesh
					if (mesh_idx < pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3(0, 0, (M.h.z + ppMeshFM_Bulk[mesh_idx]->pM->h.z) / 2);
					//coupling for micromagnetic AFM mesh
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->origin + cuReal3(0, 0, (M.h.z + ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->h.z) / 2);
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size + paMesh_Bulk_size) {

						cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - pMeshFM_Bulk_size - pMeshAFM_Bulk_size]->pM1;
						rect_c = cuRect(
							cuReal3(abs_pos.x - M.h.x / 2, abs_pos.y - M.h.y / 2, abs_pos.z + M.h.z / 2),
							cuReal3(abs_pos.x + M.h.x / 2, abs_pos.y + M.h.y / 2, abs_pos.z + M.h.z / 2 + M1.h.z));
					}
				}

				//-z coupling direction
				else if (nidx == 5 && (*pbulk_coupling_mask)[spin_index].z & 0xffff0000) {

					mesh_idx = ((*pbulk_coupling_mask)[spin_index].z >> 16) - 1;
					cell_size = M.h.z;
					//coupling for micromagnetic mesh
					if (mesh_idx < pMeshFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshFM_Bulk[mesh_idx]->pM->origin + cuReal3(0, 0, -(M.h.z + ppMeshFM_Bulk[mesh_idx]->pM->h.z) / 2);
					//coupling for micromagnetic AFM mesh
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size) cell_rel_pos = abs_pos - ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->origin + cuReal3(0, 0, -(M.h.z + ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size]->pM->h.z) / 2);
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size + paMesh_Bulk_size) {

						cuVEC_VC<cuReal3>& M1 = *ppaMesh_Bulk[mesh_idx - pMeshFM_Bulk_size - pMeshAFM_Bulk_size]->pM1;
						rect_c = cuRect(
							cuReal3(abs_pos.x - M.h.x / 2, abs_pos.y - M.h.y / 2, abs_pos.z - M.h.z / 2 - M1.h.z),
							cuReal3(abs_pos.x + M.h.x / 2, abs_pos.y + M.h.y / 2, abs_pos.z - M.h.z / 2));
					}
				}

				if (mesh_idx >= 0) {

					num_couplings++;

					//coupling for micromagnetic FM mesh
					if (mesh_idx < pMeshFM_Bulk_size) {

						calculate_mm_FM_coupling(
							M, M2, Mnew_A, Mnew_B, spin_index,
							*ppMeshFM_Bulk[mesh_idx], cell_rel_pos,
							J1, J2, cell_size,
							energy_bulk_old, energy_bulk_new);
					}
					//coupling for micromagnetic AFM mesh
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size) {
						calculate_mm_AFM_coupling(
							M, M2, Mnew_A, Mnew_B, spin_index,
							*ppMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk_size], cell_rel_pos,
							J1, J2, cell_size,
							energy_bulk_old, energy_bulk_new);
					}
					//coupling for atomistic mesh
					else if (mesh_idx < pMeshFM_Bulk_size + pMeshAFM_Bulk_size + paMesh_Bulk_size) {
						calculate_atom_coupling(
							M, M2, Mnew_A, Mnew_B, spin_index,
							*ppaMesh_Bulk[mesh_idx - pMeshFM_Bulk_size - pMeshAFM_Bulk_size], rect_c,
							J1, J2, cell_size,
							energy_bulk_old, energy_bulk_new);
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

	if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) return M.h.dim() * (energy_new - energy_old);
	else return M.h.dim() * energy_old;
}

#endif