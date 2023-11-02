#include "stdafx.h"
#include "SurfExchange_AFM.h"

#ifdef MODULE_COMPILATION_SURFEXCHANGE

#include "SuperMesh.h"

#include "Mesh_AntiFerromagnetic.h"
#include "MeshParamsControl.h"

#include "Atom_Mesh.h"

/////////////////////////////////////////////////////////////////
//SurfExchange_AFM
//

SurfExchange_AFM::SurfExchange_AFM(Mesh *pMesh_) :
	Modules(),
	ProgramStateNames(this, {}, {})
{
	pMesh = pMesh_;

	//-------------------------- Is CUDA currently enabled?

	//If cuda is enabled we also need to make the cuda module version
	if (pMesh->cudaEnabled) {

		if (!error_on_create) error_on_create = SwitchCUDAState(true);
	}
}

SurfExchange_AFM::~SurfExchange_AFM()
{
}

BError SurfExchange_AFM::Initialize(void)
{
	BError error(CLASS_STR(SurfExchange_AFM));

	//Need to identify all magnetic meshes participating in surface exchange coupling with this module:
	//1. Must be ferromagnetic or anti-ferromagnetic and have the SurfExchange module set -> this results in surface exchange coupling (or equivalently exchange bias - same formula, except in this case you should have J2 set to zero)
	//2. Must overlap in the x-y plane only with the mesh holding this module (either top or bottom) but not along z (i.e. mesh rectangles must not intersect)
	//3. No other magnetic meshes can be sandwiched in between - there could be other types of non-magnetic meshes in between of course (e.g. insulator, conductive layers etc).

	if (!initialized) {

		SuperMesh* pSMesh = pMesh->pSMesh;

		pMesh_Bot.clear();
		pMesh_Top.clear();
		paMesh_Bot.clear();
		paMesh_Top.clear();

		pMeshFM_Bulk.clear();
		pMeshAFM_Bulk.clear();
		paMesh_Bulk.clear();

		//---

		//lambda used to check condition 3
		auto check_candidate = [&](Rect xy_intersection, double z1, double z2) -> bool {

			//check all meshes to find a magnetic mesh with SurfExchange modules set, which intersects in the xy plane with xy_intersection, and has z coordinates between z1 and z2.
			//if found then current candidate not good
			for (int idx = 0; idx < pSMesh->size(); idx++) {

				//consider all meshes in turn - condition 1 first
				if ((*pSMesh)[idx]->MComputation_Enabled() && (*pSMesh)[idx]->IsModuleSet(MOD_SURFEXCHANGE)) {

					//get xy_meshrect (with z coordinates set to zero)
					Rect xy_meshrect = (*pSMesh)[idx]->GetMeshRect();
					xy_meshrect.s.z = 0; xy_meshrect.e.z = 0;

					if (xy_meshrect.intersects(xy_intersection)) {

						//intersection found. are the z coordinates in range also?
						if (IsGE((*pSMesh)[idx]->GetMeshRect().s.z, z1) && IsSE((*pSMesh)[idx]->GetMeshRect().e.z, z2)) {

							//new candidate found - note, new candidate cannot be the mesh being checked or the current candidate, so this is guranteed to be a better candidate
							return false;
						}
					}
				}
			}

			//no new candidate found - current candidate has been validated as the best one of its type (with given intersection)
			return true;
		};

		//---

		Rect meshRect = pMesh->GetMeshRect();

		Rect xy_meshRect = meshRect;
		xy_meshRect.s.z = 0; xy_meshRect.e.z = 0;

		for (int idx = 0; idx < pSMesh->size(); idx++) {

			//skip this mesh
			if ((*pSMesh)[idx]->get_id() == pMesh->get_id()) continue;

			//consider all meshes in turn - condition 1 first
			if ((*pSMesh)[idx]->MComputation_Enabled() && (*pSMesh)[idx]->IsModuleSet(MOD_SURFEXCHANGE)) {

				Rect candidate_meshRect = (*pSMesh)[idx]->GetMeshRect();

				//------------------ SURFACE COUPLING Z STACKING

				//candidate mesh at the top
				if (IsGE(candidate_meshRect.s.z, meshRect.e.z)) {

					double z1 = meshRect.e.z;				//start z
					double z2 = candidate_meshRect.s.z;		//end z
					candidate_meshRect.s.z = 0; candidate_meshRect.e.z = 0;		//leave just the xy plane rect

					if (candidate_meshRect.intersects(xy_meshRect)) {

						//passes condition 2 - identified a candidate mesh at idx index. Does it pass condition 3?
						if (check_candidate(candidate_meshRect.get_intersection(xy_meshRect), z1, z2)) {

							if (!(*pSMesh)[idx]->is_atomistic()) pMesh_Top.push_back(dynamic_cast<Mesh*>((*pSMesh)[idx]));
							else paMesh_Top.push_back(dynamic_cast<Atom_Mesh*>((*pSMesh)[idx]));
						}
					}
				}

				//candidate mesh at the botttom
				else if (IsSE(candidate_meshRect.e.z, meshRect.s.z)) {

					double z1 = candidate_meshRect.e.z;		//start z
					double z2 = meshRect.s.z;				//end z
					candidate_meshRect.s.z = 0; candidate_meshRect.e.z = 0;		//leave just the xy plane rect

					if (candidate_meshRect.intersects(xy_meshRect)) {

						//passes condition 2 - identified a candidate mesh at idx index. Does it pass condition 3?
						if (check_candidate(candidate_meshRect.get_intersection(xy_meshRect), z1, z2)) {

							if (!(*pSMesh)[idx]->is_atomistic()) pMesh_Bot.push_back(dynamic_cast<Mesh*>((*pSMesh)[idx]));
							else paMesh_Bot.push_back(dynamic_cast<Atom_Mesh*>((*pSMesh)[idx]));
						}
					}
				}

				//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

				else if (meshRect.intersects(candidate_meshRect)) {

					if (!(*pSMesh)[idx]->is_atomistic()) {

						Mesh* pMesh_Bulk = dynamic_cast<Mesh*>((*pSMesh)[idx]);
						if (pMesh_Bulk->GetMeshType() == MESH_FERROMAGNETIC) pMeshFM_Bulk.push_back(pMesh_Bulk);
						else if (pMesh_Bulk->GetMeshType() == MESH_ANTIFERROMAGNETIC) pMeshAFM_Bulk.push_back(pMesh_Bulk);
					}
					else paMesh_Bulk.push_back(dynamic_cast<Atom_Mesh*>((*pSMesh)[idx]));
				}
			}
		}

		//calculate bulk_coupling_mask if needed
		if (pMeshFM_Bulk.size() + pMeshAFM_Bulk.size() + paMesh_Bulk.size()) {

			if (!bulk_coupling_mask.assign(pMesh->h, pMesh->meshRect, INT3())) error(BERROR_OUTOFMEMORY_CRIT);

			if (!error) {

				//for absolute position abs_pos search for a mesh coupled in the bulk which contains this position and has non-empty cell there
				//store in bulk_coupling_mask at idx if found
				//direction: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z
				auto set_bulk_coupling = [&](DBL3 abs_pos, VEC<INT3>& bulk_coupling_mask, int idx, int direction) -> void {

					int mesh_index = -1;

					//search micromagnetic meshes - FM
					for (int midx = 0; midx < pMeshFM_Bulk.size(); midx++) {

						if (pMeshFM_Bulk[midx]->M.rect.contains(abs_pos)) {
							if (pMeshFM_Bulk[midx]->M.is_not_empty(abs_pos - pMeshFM_Bulk[midx]->M.rect.s)) {

								//found mesh
								mesh_index = midx;
								break;
							}
						}
					}

					//search micromagnetic meshes - AFM
					if (mesh_index == -1) {

						for (int midx = 0; midx < pMeshAFM_Bulk.size(); midx++) {

							if (pMeshAFM_Bulk[midx]->M.rect.contains(abs_pos)) {
								if (pMeshAFM_Bulk[midx]->M.is_not_empty(abs_pos - pMeshAFM_Bulk[midx]->M.rect.s)) {

									//found mesh
									mesh_index = midx + pMeshFM_Bulk.size();
									break;
								}
							}
						}
					}

					//if not found in micromagnetic meshes, search atomistic meshes
					if (mesh_index == -1) {

						for (int midx = 0; midx < paMesh_Bulk.size(); midx++) {

							if (paMesh_Bulk[midx]->M1.rect.contains(abs_pos)) {
								if (paMesh_Bulk[midx]->M1.is_not_empty(abs_pos - paMesh_Bulk[midx]->M1.rect.s)) {

									//found mesh (must add number of micromagnetic meshes for total compound index)
									mesh_index = midx + pMeshFM_Bulk.size() + pMeshAFM_Bulk.size();
									break;
								}
							}
						}
					}

					//if found coupling set mesh index + 1 in correct position for given coupling direction
					if (mesh_index >= 0) {

						switch (direction)
						{
							//+x (first 2 bytes)
						case 0:
							bulk_coupling_mask[idx].x += (mesh_index + 1);
							break;
							//-x (last 2 bytes)
						case 1:
							bulk_coupling_mask[idx].x |= (mesh_index + 1) << 16;
							break;
							//+y (first 2 bytes)
						case 2:
							bulk_coupling_mask[idx].y += (mesh_index + 1);
							break;
							//-y (last 2 bytes)
						case 3:
							bulk_coupling_mask[idx].y |= (mesh_index + 1) << 16;
							break;
							//+z (first 2 bytes)
						case 4:
							bulk_coupling_mask[idx].z += (mesh_index + 1);
							break;
							//-z (last 2 bytes)
						case 5:
							bulk_coupling_mask[idx].z |= (mesh_index + 1) << 16;
							break;
						}
					}
				};

#pragma omp parallel for
				for (int idx = 0; idx < pMesh->M.linear_size(); idx++) {

					if (pMesh->M.is_not_empty(idx)) {

						std::vector<int> neighbors(6);
						//order is +x, -x, +y, -y, +z, -z
						pMesh->M.get_neighbors(idx, neighbors);

						DBL3 abs_pos = pMesh->M.cellidx_to_position(idx) + pMesh->M.rect.s;

						//empty +x, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[0] == -1) set_bulk_coupling(abs_pos + DBL3(pMesh->M.h.x, 0, 0), bulk_coupling_mask, idx, 0);
						//empty -x, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[1] == -1) set_bulk_coupling(abs_pos + DBL3(-pMesh->M.h.x, 0, 0), bulk_coupling_mask, idx, 1);
						//empty +y, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[2] == -1) set_bulk_coupling(abs_pos + DBL3(0, pMesh->M.h.y, 0), bulk_coupling_mask, idx, 2);
						//empty -y, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[3] == -1) set_bulk_coupling(abs_pos + DBL3(0, -pMesh->M.h.y, 0), bulk_coupling_mask, idx, 3);
						//empty +z, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[4] == -1) set_bulk_coupling(abs_pos + DBL3(0, 0, pMesh->M.h.z), bulk_coupling_mask, idx, 4);
						//empty -z, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[5] == -1) set_bulk_coupling(abs_pos + DBL3(0, 0, -pMesh->M.h.z), bulk_coupling_mask, idx, 5);
					}
				}
			}
		}
		else bulk_coupling_mask.clear();

		initialized = true;
	}
	
	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		pMesh->h, pMesh->meshRect,
		(MOD_)pMesh->Get_Module_Heff_Display() == MOD_SURFEXCHANGE || pMesh->IsOutputDataSet_withRect(DATA_E_SURFEXCH) || pMesh->IsOutputDataSet(DATA_T_SURFEXCH),
		(MOD_)pMesh->Get_Module_Energy_Display() == MOD_SURFEXCHANGE || pMesh->IsOutputDataSet_withRect(DATA_E_SURFEXCH),
		true);
	if (error) initialized = false;

	return error;
}

BError SurfExchange_AFM::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SurfExchange_AFM));

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHCHANGE, UPDATECONFIG_MESHADDED, UPDATECONFIG_MESHDELETED, UPDATECONFIG_MESHSHAPECHANGE, UPDATECONFIG_MODULEADDED, UPDATECONFIG_MODULEDELETED)) {

		Uninitialize();
	}

	//------------------------ CUDA UpdateConfiguration if set

#if COMPILECUDA == 1
	if (pModuleCUDA) {

		if (!error) error = pModuleCUDA->UpdateConfiguration(cfgMessage);
	}
#endif

	return error;
}

BError SurfExchange_AFM::MakeCUDAModule(void)
{
	BError error(CLASS_STR(SurfExchange_AFM));

#if COMPILECUDA == 1

	if (pMesh->pMeshCUDA) {

		//Note : it is posible pMeshCUDA has not been allocated yet, but this module has been created whilst cuda is switched on. This will happen when a new mesh is being made which adds this module by default.
		//In this case, after the mesh has been fully made, it will call SwitchCUDAState on the mesh, which in turn will call this SwitchCUDAState method; then pMeshCUDA will not be nullptr and we can make the cuda module version
		pModuleCUDA = new SurfExchangeCUDA_AFM(pMesh->pMeshCUDA, this);
		error = pModuleCUDA->Error_On_Create();
	}

#endif

	return error;
}

double SurfExchange_AFM::UpdateField(void)
{
	double energy = 0;

	SZ3 n = pMesh->n;

	//zero module display VECs if needed, since contributions must be added into them to account for possiblility of 2 contributions (top and bottom)
	ZeroModuleVECs();

	//------------------ Coupling functions

	auto calculate_mm_coupling = [](
		VEC_VC<DBL3>& M, VEC_VC<DBL3>& M2, int cell_idx,
		Mesh* pMeshCoupled, DBL3 cell_rel_pos,
		DBL2 Ms_AFM, double J1, double J2, double cell_size,
		DBL3& Hsurfexch1, double& cell_energy1,
		DBL3& Hsurfexch2, double& cell_energy2) -> void
	{
		//NOTE : no need to check here if pMeshCoupled->M contains cell_rel_pos, or if cell is not empty there.
		//This check is done before calling this function for z stacking, or when initializing bulk_coupling_mask for bulk coupling.

		if (pMeshCoupled->GetMeshType() == MESH_FERROMAGNETIC) {

			//Surface exchange field from a ferromagnetic mesh

			//get magnetization value in mesh cell to couple with
			DBL3 m_j = normalize(pMeshCoupled->M[cell_rel_pos]);
			DBL3 m_i1 = normalize(M[cell_idx]);
			DBL3 m_i2 = normalize(M2[cell_idx]);

			//total surface exchange field in coupling cells
			Hsurfexch1 += (m_j / (MU0 * Ms_AFM.i * cell_size)) * J1;
			Hsurfexch2 += (m_j / (MU0 * Ms_AFM.j * cell_size)) * J2;
			cell_energy1 += (-J1 * (m_i1 * m_j)) / cell_size;
			cell_energy2 += (-J2 * (m_i2 * m_j)) / cell_size;
		}
		else if (pMeshCoupled->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

			//Surface exchange field from an antiferromagnetic mesh

			//get magnetization value in mesh cell to couple with
			DBL3 m_j1 = normalize(pMeshCoupled->M[cell_rel_pos]);
			DBL3 m_j2 = normalize(pMeshCoupled->M2[cell_rel_pos]);
			DBL3 m_i1 = normalize(M[cell_idx]);
			DBL3 m_i2 = normalize(M2[cell_idx]);

			//total surface exchange field in coupling cells
			Hsurfexch1 += (m_j1 / (MU0 * Ms_AFM.i * cell_size)) * J1;
			Hsurfexch2 += (m_j2 / (MU0 * Ms_AFM.j * cell_size)) * J2;
			cell_energy1 += (-J1 * (m_i1 * m_j1)) / cell_size;
			cell_energy2 += (-J2 * (m_i2 * m_j2)) / cell_size;
		}
	};

	auto calculate_atom_coupling = [](
		VEC_VC<DBL3>& M, VEC_VC<DBL3>& M2, int cell_idx,
		Atom_Mesh* paMeshCoupled, Rect& rect_c,
		DBL2 Ms_AFM, double J1, double J2, double cell_size,
		DBL3& Hsurfexch1, double& cell_energy1,
		DBL3& Hsurfexch2, double& cell_energy2) -> void
	{
		VEC_VC<DBL3>& M1 = paMeshCoupled->M1;

		//cells box in atomistic mesh. NOTE : acells is capped to mesh dimensions, so we are guaranteed proper indexes inside the mesh.
		Box acells = M1.box_from_rect_min(rect_c);

		DBL3 m_j = DBL3();
		for (int ai = acells.s.i; ai < acells.e.i; ai++) {
			for (int aj = acells.s.j; aj < acells.e.j; aj++) {
				for (int ak = acells.s.k; ak < acells.e.k; ak++) {

					int acell_idx = ai + aj * M1.n.x + ak * M1.n.x * M1.n.y;

					if (M1.is_empty(acell_idx)) continue;

					m_j += M1[acell_idx];
				}
			}
		}

		m_j = normalize(m_j);
		DBL3 m_i1 = normalize(M[cell_idx]);
		DBL3 m_i2 = normalize(M2[cell_idx]);

		//total surface exchange field in coupling cells
		Hsurfexch1 += (m_j / (MU0 * Ms_AFM.i * cell_size)) * J1;
		Hsurfexch2 += (m_j / (MU0 * Ms_AFM.j * cell_size)) * J2;
		cell_energy1 += (-J1 * (m_i1 * m_j)) / cell_size;
		cell_energy2 += (-J2 * (m_i2 * m_j)) / cell_size;
	};

	//------------------ SURFACE COUPLING Z STACKING

	if (pMesh_Top.size()) {

		//surface exchange coupling at the top
#pragma omp parallel for reduction(+:energy)
		for (int j = 0; j < n.y; j++) {
			for (int i = 0; i < n.x; i++) {

				int cell_idx = i + j * n.x + (n.z - 1) * n.x*n.y;

				//empty cell here ... next
				if (pMesh->M.is_empty(cell_idx)) continue;

				DBL2 Ms_AFM = pMesh->Ms_AFM;
				pMesh->update_parameters_mcoarse(cell_idx, pMesh->Ms_AFM, Ms_AFM);

				DBL3 Hsurfexch1 = DBL3(), Hsurfexch2 = DBL3();
				double cell_energy1 = 0.0, cell_energy2 = 0.0;
				bool cell_coupled = false;

				//check all meshes for coupling
				//1 : coupling into this micromagnetic mesh from other micromagnetic meshes (FM or AFM)
				for (int mesh_idx = 0; mesh_idx < (int)pMesh_Top.size(); mesh_idx++) {

					if (!check_cell_coupling(pMesh->M, pMesh_Top[mesh_idx]->M,
						(i + 0.5) * pMesh->h.x, (j + 0.5) * pMesh->h.y, pMesh_Top[mesh_idx]->h.z / 2)) continue;

					DBL3 cell_rel_pos = DBL3(
						(i + 0.5) * pMesh->h.x + pMesh->M.rect.s.x - pMesh_Top[mesh_idx]->M.rect.s.x,
						(j + 0.5) * pMesh->h.y + pMesh->M.rect.s.y - pMesh_Top[mesh_idx]->M.rect.s.y,
						pMesh_Top[mesh_idx]->h.z / 2);

					//Top mesh sets J1 and J2 values
					double J1 = pMesh_Top[mesh_idx]->J1;
					double J2 = pMesh_Top[mesh_idx]->J2;
					pMesh_Top[mesh_idx]->update_parameters_atposition(cell_rel_pos, pMesh_Top[mesh_idx]->J1, J1, pMesh_Top[mesh_idx]->J2, J2);

					calculate_mm_coupling(
						pMesh->M, pMesh->M2, cell_idx,
						pMesh_Top[mesh_idx], cell_rel_pos,
						Ms_AFM, J1, J2, pMesh->M.h.z,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					cell_coupled = true;
					break;
				}

				if (!cell_coupled) {

					//2 : coupling into this micromagnetic mesh from atomistic meshes
					for (int mesh_idx = 0; mesh_idx < (int)paMesh_Top.size(); mesh_idx++) {

						VEC_VC<DBL3>& M1 = paMesh_Top[mesh_idx]->M1;

						//coupling rectangle in atomistic mesh in absolute coordinates
						Rect rect_c = Rect(
							DBL3(i * pMesh->h.x, j * pMesh->h.y, pMesh->M.rect.e.z),
							DBL3((i + 1) * pMesh->h.x, (j + 1) * pMesh->h.y, M1.h.z + pMesh->M.rect.e.z));
						rect_c += DBL3(pMesh->M.rect.s.x, pMesh->M.rect.s.y, 0.0);

						double J1 = pMesh->J1;
						double J2 = pMesh->J2;
						pMesh->update_parameters_mcoarse(cell_idx, pMesh->J1, J1, pMesh->J2, J2);

						calculate_atom_coupling(
							pMesh->M, pMesh->M2, cell_idx,
							paMesh_Top[mesh_idx], rect_c,
							Ms_AFM, J1, J2, pMesh->M.h.z,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);

						//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
						break;
					}
				}

				pMesh->Heff[cell_idx] += Hsurfexch1;
				pMesh->Heff2[cell_idx] += Hsurfexch2;

				if (Module_Heff.linear_size()) Module_Heff[cell_idx] += Hsurfexch1;
				if (Module_Heff2.linear_size()) Module_Heff2[cell_idx] += Hsurfexch2;
				if (Module_energy.linear_size()) Module_energy[cell_idx] += cell_energy1;
				if (Module_energy2.linear_size()) Module_energy2[cell_idx] += cell_energy2;

				energy += (cell_energy1 + cell_energy2) / 2;
			}
		}
	}

	if (pMesh_Bot.size()) {

		//surface exchange coupling at the bottom
#pragma omp parallel for reduction(+:energy)
		for (int j = 0; j < n.y; j++) {
			for (int i = 0; i < n.x; i++) {

				int cell_idx = i + j * n.x;

				//empty cell here ... next
				if (pMesh->M.is_empty(cell_idx)) continue;

				DBL2 Ms_AFM = pMesh->Ms_AFM;
				double J1 = pMesh->J1;
				double J2 = pMesh->J2;
				pMesh->update_parameters_mcoarse(cell_idx, pMesh->Ms_AFM, Ms_AFM, pMesh->J1, J1, pMesh->J2, J2);

				DBL3 Hsurfexch1 = DBL3(), Hsurfexch2 = DBL3();
				double cell_energy1 = 0.0, cell_energy2 = 0.0;
				bool cell_coupled = false;

				//check all meshes for coupling
				for (int mesh_idx = 0; mesh_idx < (int)pMesh_Bot.size(); mesh_idx++) {

					if (!check_cell_coupling(pMesh->M, pMesh_Bot[mesh_idx]->M,
						(i + 0.5) * pMesh->h.x, (j + 0.5) * pMesh->h.y, pMesh_Bot[mesh_idx]->meshRect.height() - pMesh_Bot[mesh_idx]->h.z / 2)) continue;

					DBL3 cell_rel_pos = DBL3(
						(i + 0.5) * pMesh->h.x + pMesh->M.rect.s.x - pMesh_Bot[mesh_idx]->M.rect.s.x,
						(j + 0.5) * pMesh->h.y + pMesh->M.rect.s.y - pMesh_Bot[mesh_idx]->M.rect.s.y,
						pMesh_Bot[mesh_idx]->meshRect.height() - pMesh_Bot[mesh_idx]->h.z / 2);

					calculate_mm_coupling(
						pMesh->M, pMesh->M2, cell_idx,
						pMesh_Bot[mesh_idx], cell_rel_pos,
						Ms_AFM, J1, J2, pMesh->M.h.z,
						Hsurfexch1, cell_energy1,
						Hsurfexch2, cell_energy2);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					cell_coupled = true;
					break;
				}

				if (!cell_coupled) {

					//2 : coupling into this micromagnetic mesh from atomistic meshes
					for (int mesh_idx = 0; mesh_idx < (int)paMesh_Bot.size(); mesh_idx++) {

						VEC_VC<DBL3>& M1 = paMesh_Bot[mesh_idx]->M1;

						//coupling rectangle in atomistic mesh in absolute coordinates
						Rect rect_c = Rect(
							DBL3(i * pMesh->h.x, j * pMesh->h.y, M1.rect.e.z - M1.h.z),
							DBL3((i + 1) * pMesh->h.x, (j + 1) * pMesh->h.y, M1.rect.e.z));
						rect_c += DBL3(pMesh->M.rect.s.x, pMesh->M.rect.s.y, 0.0);

						calculate_atom_coupling(
							pMesh->M, pMesh->M2, cell_idx,
							paMesh_Top[mesh_idx], rect_c,
							Ms_AFM, J1, J2, pMesh->M.h.z,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);

						//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
						break;
					}
				}

				pMesh->Heff[cell_idx] += Hsurfexch1;
				pMesh->Heff2[cell_idx] += Hsurfexch2;

				if (Module_Heff.linear_size()) Module_Heff[cell_idx] += Hsurfexch1;
				if (Module_Heff2.linear_size()) Module_Heff2[cell_idx] += Hsurfexch2;
				if (Module_energy.linear_size()) Module_energy[cell_idx] += cell_energy1;
				if (Module_energy2.linear_size()) Module_energy2[cell_idx] += cell_energy2;

				energy += (cell_energy1 + cell_energy2) / 2;
			}
		}
	}

	//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

	if (pMeshFM_Bulk.size() + pMeshAFM_Bulk.size() + paMesh_Bulk.size()) {

#pragma omp parallel for
		for (int idx = 0; idx < pMesh->M.linear_size(); idx++) {

			if (pMesh->M.is_not_empty(idx) && bulk_coupling_mask[idx] != INT3()) {

				DBL2 Ms_AFM = pMesh->Ms_AFM;
				double J1 = pMesh->J1;
				double J2 = pMesh->J2;
				pMesh->update_parameters_mcoarse(idx, pMesh->Ms_AFM, Ms_AFM, pMesh->J1, J1, pMesh->J2, J2);

				//surface cell which needs to be exchange coupled
				DBL3 Hsurfexch1 = DBL3(), Hsurfexch2 = DBL3();
				double cell_energy1 = 0.0, cell_energy2 = 0.0;
				int num_couplings = 0;

				DBL3 abs_pos = pMesh->M.cellidx_to_position(idx) + pMesh->M.rect.s;

				//+x coupling direction
				if (bulk_coupling_mask[idx].x & 0x0000ffff) {

					int mesh_idx = (bulk_coupling_mask[idx].x & 0x0000ffff) - 1;
					num_couplings++;

					//coupling for micromagnetic mesh
					if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
						Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
						calculate_mm_coupling(
							pMesh->M, pMesh->M2, idx,
							pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3((pMesh->h.x + pMesh_Bulk->h.x) / 2, 0, 0),
							Ms_AFM, J1, J2, pMesh->M.h.x,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
					//coupling for atomistic mesh
					else {

						//coupling rectangle in atomistic mesh in absolute coordinates
						VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
						Rect rect_c = Rect(
							DBL3(abs_pos.x + pMesh->M.h.x / 2, abs_pos.y - pMesh->M.h.y / 2, abs_pos.z - pMesh->M.h.z / 2),
							DBL3(abs_pos.x + pMesh->M.h.x / 2 + M1.h.x, abs_pos.y + pMesh->M.h.y / 2, abs_pos.z + pMesh->M.h.z / 2));

						calculate_atom_coupling(
							pMesh->M, pMesh->M2, idx,
							paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
							Ms_AFM, J1, J2, pMesh->M.h.x,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
				}

				//-x coupling direction
				if (bulk_coupling_mask[idx].x & 0xffff0000) {

					int mesh_idx = (bulk_coupling_mask[idx].x >> 16) - 1;
					num_couplings++;

					//coupling for micromagnetic mesh
					if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
						Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
						calculate_mm_coupling(
							pMesh->M, pMesh->M2, idx,
							pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3(-(pMesh->h.x + pMesh_Bulk->h.x) / 2, 0, 0),
							Ms_AFM, J1, J2, pMesh->M.h.x,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
					//coupling for atomistic mesh
					else {

						//coupling rectangle in atomistic mesh in absolute coordinates
						VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
						Rect rect_c = Rect(
							DBL3(abs_pos.x - pMesh->M.h.x / 2 - M1.h.x, abs_pos.y - pMesh->M.h.y / 2, abs_pos.z - pMesh->M.h.z / 2),
							DBL3(abs_pos.x - pMesh->M.h.x / 2, abs_pos.y + pMesh->M.h.y / 2, abs_pos.z + pMesh->M.h.z / 2));

						calculate_atom_coupling(
							pMesh->M, pMesh->M2, idx,
							paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
							Ms_AFM, J1, J2, pMesh->M.h.x,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
				}

				//+y coupling direction
				if (bulk_coupling_mask[idx].y & 0x0000ffff) {

					int mesh_idx = (bulk_coupling_mask[idx].y & 0x0000ffff) - 1;
					num_couplings++;

					//coupling for micromagnetic mesh
					if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
						Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
						calculate_mm_coupling(
							pMesh->M, pMesh->M2, idx,
							pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3(0, (pMesh->h.y + pMesh_Bulk->h.y) / 2, 0),
							Ms_AFM, J1, J2, pMesh->M.h.y,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
					//coupling for atomistic mesh
					else {

						//coupling rectangle in atomistic mesh in absolute coordinates
						VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
						Rect rect_c = Rect(
							DBL3(abs_pos.x - pMesh->M.h.x / 2, abs_pos.y + pMesh->M.h.y / 2, abs_pos.z - pMesh->M.h.z / 2),
							DBL3(abs_pos.x + pMesh->M.h.x / 2, abs_pos.y + pMesh->M.h.y / 2 + M1.h.y, abs_pos.z + pMesh->M.h.z / 2));

						calculate_atom_coupling(
							pMesh->M, pMesh->M2, idx,
							paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
							Ms_AFM, J1, J2, pMesh->M.h.y,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
				}

				//-y coupling direction
				if (bulk_coupling_mask[idx].y & 0xffff0000) {

					int mesh_idx = (bulk_coupling_mask[idx].y >> 16) - 1;
					num_couplings++;

					//coupling for micromagnetic mesh
					if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
						Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
						calculate_mm_coupling(
							pMesh->M, pMesh->M2, idx,
							pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3(0, -(pMesh->h.y + pMesh_Bulk->h.y) / 2, 0),
							Ms_AFM, J1, J2, pMesh->M.h.y,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
					//coupling for atomistic mesh
					else {

						//coupling rectangle in atomistic mesh in absolute coordinates
						VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
						Rect rect_c = Rect(
							DBL3(abs_pos.x - pMesh->M.h.x / 2, abs_pos.y - pMesh->M.h.y / 2 - M1.h.y, abs_pos.z - pMesh->M.h.z / 2),
							DBL3(abs_pos.x + pMesh->M.h.x / 2, abs_pos.y - pMesh->M.h.y / 2, abs_pos.z + pMesh->M.h.z / 2));

						calculate_atom_coupling(
							pMesh->M, pMesh->M2, idx,
							paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
							Ms_AFM, J1, J2, pMesh->M.h.y,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
				}

				//+z coupling direction
				if (bulk_coupling_mask[idx].z & 0x0000ffff) {

					int mesh_idx = (bulk_coupling_mask[idx].z & 0x0000ffff) - 1;
					num_couplings++;

					//coupling for micromagnetic mesh
					if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
						Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
						calculate_mm_coupling(
							pMesh->M, pMesh->M2, idx,
							pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3(0, 0, (pMesh->h.z + pMesh_Bulk->h.z) / 2),
							Ms_AFM, J1, J2, pMesh->M.h.z,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
					//coupling for atomistic mesh
					else {

						//coupling rectangle in atomistic mesh in absolute coordinates
						VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
						Rect rect_c = Rect(
							DBL3(abs_pos.x - pMesh->M.h.x / 2, abs_pos.y - pMesh->M.h.y / 2, abs_pos.z + pMesh->M.h.z / 2),
							DBL3(abs_pos.x + pMesh->M.h.x / 2, abs_pos.y + pMesh->M.h.y / 2, abs_pos.z + pMesh->M.h.z / 2 + M1.h.z));

						calculate_atom_coupling(
							pMesh->M, pMesh->M2, idx,
							paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
							Ms_AFM, J1, J2, pMesh->M.h.z,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
				}

				//-z coupling direction
				if (bulk_coupling_mask[idx].z & 0xffff0000) {

					int mesh_idx = (bulk_coupling_mask[idx].z >> 16) - 1;
					num_couplings++;

					//coupling for micromagnetic mesh
					if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
						Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
						calculate_mm_coupling(
							pMesh->M, pMesh->M2, idx,
							pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3(0, 0, -(pMesh->h.z + pMesh_Bulk->h.z) / 2),
							Ms_AFM, J1, J2, pMesh->M.h.z,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
					//coupling for atomistic mesh
					else {

						//coupling rectangle in atomistic mesh in absolute coordinates
						VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
						Rect rect_c = Rect(
							DBL3(abs_pos.x - pMesh->M.h.x / 2, abs_pos.y - pMesh->M.h.y / 2, abs_pos.z - pMesh->M.h.z / 2 - M1.h.z),
							DBL3(abs_pos.x + pMesh->M.h.x / 2, abs_pos.y + pMesh->M.h.y / 2, abs_pos.z - pMesh->M.h.z / 2));

						calculate_atom_coupling(
							pMesh->M, pMesh->M2, idx,
							paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
							Ms_AFM, J1, J2, pMesh->M.h.z,
							Hsurfexch1, cell_energy1,
							Hsurfexch2, cell_energy2);
					}
				}

				if (num_couplings) {

					//need average if cell receives multiple coupling contributions
					pMesh->Heff[idx] += Hsurfexch1;
					pMesh->Heff2[idx] += Hsurfexch2;

					if (Module_Heff.linear_size()) Module_Heff[idx] += Hsurfexch1;
					if (Module_Heff2.linear_size()) Module_Heff2[idx] += Hsurfexch2;
					if (Module_energy.linear_size()) Module_energy[idx] += cell_energy1;
					if (Module_energy2.linear_size()) Module_energy2[idx] += cell_energy2;

					energy += (cell_energy1 + cell_energy2) / 2;
				}
			}
		}
	}

	//------------------

	energy /= pMesh->M.get_nonempty_cells();
	this->energy = energy;

	return this->energy;
}

//-------------------Energy methods

//AFM
DBL2 SurfExchange_AFM::Get_EnergyChange(int spin_index, DBL3 Mnew_A, DBL3 Mnew_B)
{
	DBL2 energy_new = DBL2(), energy_old = DBL2();

	SZ3 n = pMesh->n;

	//------------------ Coupling functions

	auto calculate_mm_coupling = [](
		VEC_VC<DBL3>& M, VEC_VC<DBL3>& M2, DBL3 Mnew_A, DBL3 Mnew_B, int spin_index,
		Mesh* pMeshCoupled, DBL3 cell_rel_pos,
		double J1, double J2, double cell_size,
		DBL2& energy_old, DBL2& energy_new) -> void
	{
		if (pMeshCoupled->GetMeshType() == MESH_FERROMAGNETIC) {

			//Surface exchange field from a ferromagnetic mesh

			DBL3 m_j = normalize(pMeshCoupled->M[cell_rel_pos]);
			DBL3 m_i1 = normalize(M[spin_index]);
			DBL3 m_i2 = normalize(M2[spin_index]);

			energy_old.i = (-J1 * (m_i1 * m_j)) / cell_size;
			energy_old.j = (-J2 * (m_i2 * m_j)) / cell_size;

			if (Mnew_A != DBL3() && Mnew_B != DBL3()) {

				DBL3 mnew_i1 = normalize(Mnew_A);
				DBL3 mnew_i2 = normalize(Mnew_B);

				energy_new.i = (-J1 * (mnew_i1 * m_j)) / cell_size;
				energy_new.j = (-J2 * (mnew_i2 * m_j)) / cell_size;
			}
		}
		else if (pMeshCoupled->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

			//Surface exchange field from an antiferromagnetic mesh

			DBL3 m_j1 = normalize(pMeshCoupled->M[cell_rel_pos]);
			DBL3 m_j2 = normalize(pMeshCoupled->M2[cell_rel_pos]);
			DBL3 m_i1 = normalize(M[spin_index]);
			DBL3 m_i2 = normalize(M2[spin_index]);

			energy_old.i = (-J1 * (m_i1 * m_j1)) / cell_size;
			energy_old.j = (-J2 * (m_i2 * m_j2)) / cell_size;

			if (Mnew_A != DBL3() && Mnew_B != DBL3()) {

				DBL3 mnew_i1 = normalize(Mnew_A);
				DBL3 mnew_i2 = normalize(Mnew_B);

				energy_new.i = (-J1 * (mnew_i1 * m_j1)) / cell_size;
				energy_new.j = (-J2 * (mnew_i2 * m_j2)) / cell_size;
			}
		}
	};

	auto calculate_atom_coupling = [](
		VEC_VC<DBL3>& M, VEC_VC<DBL3>& M2, DBL3 Mnew_A, DBL3 Mnew_B, int spin_index,
		Atom_Mesh* paMeshCoupled, Rect& rect_c,
		double J1, double J2, double cell_size,
		DBL2& energy_old, DBL2& energy_new) -> void
	{
		VEC_VC<DBL3>& M1 = paMeshCoupled->M1;

		//cells box in atomistic mesh
		Box acells = M1.box_from_rect_min(rect_c);

		DBL3 m_j = DBL3();
		for (int ai = acells.s.i; ai < acells.e.i; ai++) {
			for (int aj = acells.s.j; aj < acells.e.j; aj++) {
				for (int ak = acells.s.k; ak < acells.e.k; ak++) {

					int acell_idx = ai + aj * M1.n.x + ak * M1.n.x * M1.n.y;

					if (M1.is_empty(acell_idx)) continue;

					m_j += M1[acell_idx];
				}
			}
		}

		m_j = normalize(m_j);
		DBL3 m_i1 = normalize(M[spin_index]);
		DBL3 m_i2 = normalize(M2[spin_index]);

		energy_old.i += (-J1 * (m_i1 * m_j)) / cell_size;
		energy_old.j += (-J2 * (m_i2 * m_j)) / cell_size;

		if (Mnew_A != DBL3() && Mnew_B != DBL3()) {

			DBL3 mnew_i1 = normalize(Mnew_A);
			DBL3 mnew_i2 = normalize(Mnew_B);

			energy_new.i += (-J1 * (mnew_i1 * m_j)) / cell_size;
			energy_new.j += (-J2 * (mnew_i2 * m_j)) / cell_size;
		}
	};

	//------------------ SURFACE COUPLING Z STACKING

	//if spin is on top surface then look at paMesh_Top
	if (spin_index / (n.x * n.y) == n.z - 1 && pMesh_Top.size()) {

		if (!pMesh->M.is_empty(spin_index)) {

			int i = spin_index % n.x;
			int j = (spin_index / n.x) % n.y;

			bool cell_coupled = false;

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < (int)pMesh_Top.size(); mesh_idx++) {

				if (!check_cell_coupling(pMesh->M, pMesh_Top[mesh_idx]->M,
					(i + 0.5) * pMesh->h.x, (j + 0.5) * pMesh->h.y, pMesh_Top[mesh_idx]->h.z / 2)) continue;

				DBL3 cell_rel_pos = DBL3(
					(i + 0.5) * pMesh->h.x + pMesh->M.rect.s.x - pMesh_Top[mesh_idx]->M.rect.s.x,
					(j + 0.5) * pMesh->h.y + pMesh->M.rect.s.y - pMesh_Top[mesh_idx]->M.rect.s.y,
					pMesh_Top[mesh_idx]->h.z / 2);

				//Top mesh sets J1 and J2 values
				double J1 = pMesh_Top[mesh_idx]->J1;
				double J2 = pMesh_Top[mesh_idx]->J2;
				pMesh_Top[mesh_idx]->update_parameters_atposition(cell_rel_pos, pMesh_Top[mesh_idx]->J1, J1, pMesh_Top[mesh_idx]->J2, J2);

				calculate_mm_coupling(
					pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
					pMesh_Top[mesh_idx], cell_rel_pos,
					J1, J2, pMesh->h.z,
					energy_old, energy_new);

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				cell_coupled = true;
				break;
			}

			if (!cell_coupled) {

				//2 : coupling into this micromagnetic mesh from atomistic meshes
				for (int mesh_idx = 0; mesh_idx < (int)paMesh_Top.size(); mesh_idx++) {

					VEC_VC<DBL3>& M1 = paMesh_Top[mesh_idx]->M1;

					//coupling rectangle in atomistic mesh in absolute coordinates
					Rect rect_c = Rect(
						DBL3(i * pMesh->h.x, j * pMesh->h.y, pMesh->M.rect.e.z),
						DBL3((i + 1) * pMesh->h.x, (j + 1) * pMesh->h.y, M1.h.z + pMesh->M.rect.e.z));
					rect_c += DBL3(pMesh->M.rect.s.x, pMesh->M.rect.s.y, 0.0);

					//current mesh sets coupling in micromagnetic to atomistic meshes coupling
					double J1 = pMesh->J1;
					double J2 = pMesh->J2;
					pMesh->update_parameters_mcoarse(spin_index, pMesh->J1, J1, pMesh->J2, J2);

					calculate_atom_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						paMesh_Top[mesh_idx], rect_c,
						J1, J2, pMesh->h.z,
						energy_old, energy_new);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					break;
				}
			}
		}
	}

	if (spin_index / (n.x * n.y) == 0 && pMesh_Bot.size()) {

		//surface exchange coupling at the bottom
		
		if (!pMesh->M.is_empty(spin_index)) {

			int i = spin_index % n.x;
			int j = (spin_index / n.x) % n.y;

			double J1 = pMesh->J1;
			double J2 = pMesh->J2;
			pMesh->update_parameters_mcoarse(spin_index, pMesh->J1, J1, pMesh->J2, J2);

			bool cell_coupled = false;

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < (int)pMesh_Bot.size(); mesh_idx++) {

				if (!check_cell_coupling(pMesh->M, pMesh_Bot[mesh_idx]->M,
					(i + 0.5) * pMesh->h.x, (j + 0.5) * pMesh->h.y, pMesh_Bot[mesh_idx]->meshRect.height() - pMesh_Bot[mesh_idx]->h.z / 2)) continue;

				DBL3 cell_rel_pos = DBL3(
					(i + 0.5) * pMesh->h.x + pMesh->M.rect.s.x - pMesh_Bot[mesh_idx]->M.rect.s.x,
					(j + 0.5) * pMesh->h.y + pMesh->M.rect.s.y - pMesh_Bot[mesh_idx]->M.rect.s.y,
					pMesh_Bot[mesh_idx]->meshRect.height() - pMesh_Bot[mesh_idx]->h.z / 2);

				calculate_mm_coupling(
					pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
					pMesh_Bot[mesh_idx], cell_rel_pos,
					J1, J2, pMesh->h.z,
					energy_old, energy_new);

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				cell_coupled = true;
				break;
			}

			if (!cell_coupled) {

				//2 : coupling into this micromagnetic mesh from atomistic meshes
				for (int mesh_idx = 0; mesh_idx < (int)paMesh_Bot.size(); mesh_idx++) {

					VEC_VC<DBL3>& M1 = paMesh_Bot[mesh_idx]->M1;

					//coupling rectangle in atomistic mesh in absolute coordinates
					Rect rect_c = Rect(
						DBL3(i * pMesh->h.x, j * pMesh->h.y, M1.rect.e.z - M1.h.z),
						DBL3((i + 1) * pMesh->h.x, (j + 1) * pMesh->h.y, M1.rect.e.z));
					rect_c += DBL3(pMesh->M.rect.s.x, pMesh->M.rect.s.y, 0.0);

					calculate_atom_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						paMesh_Bot[mesh_idx], rect_c,
						J1, J2, pMesh->h.z,
						energy_old, energy_new);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					break;
				}
			}
		}
	}

	//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

	if (pMeshFM_Bulk.size() + pMeshAFM_Bulk.size() + paMesh_Bulk.size()) {

		if (pMesh->M.is_not_empty(spin_index) && bulk_coupling_mask[spin_index] != INT3()) {

			DBL2 energy_bulk_new = 0, energy_bulk_old = 0;

			double J1 = pMesh->J1;
			double J2 = pMesh->J2;
			pMesh->update_parameters_mcoarse(spin_index, pMesh->J1, J1, pMesh->J2, J2);

			int num_couplings = 0;

			DBL3 abs_pos = pMesh->M.cellidx_to_position(spin_index) + pMesh->M.rect.s;

			//+x coupling direction
			if (bulk_coupling_mask[spin_index].x & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[spin_index].x & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for micromagnetic mesh
				if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
					Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
					calculate_mm_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3((pMesh->h.x + pMesh_Bulk->h.x) / 2, 0, 0),
						J1, J2, pMesh->h.z,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for atomistic mesh
				else {

					//coupling rectangle in atomistic mesh in absolute coordinates
					VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
					Rect rect_c = Rect(
						DBL3(abs_pos.x + pMesh->M.h.x / 2, abs_pos.y - pMesh->M.h.y / 2, abs_pos.z - pMesh->M.h.z / 2),
						DBL3(abs_pos.x + pMesh->M.h.x / 2 + M1.h.x, abs_pos.y + pMesh->M.h.y / 2, abs_pos.z + pMesh->M.h.z / 2));

					calculate_atom_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
						J1, J2, pMesh->h.x,
						energy_bulk_old, energy_bulk_new);
				}
			}

			//-x coupling direction
			if (bulk_coupling_mask[spin_index].x & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[spin_index].x >> 16) - 1;
				num_couplings++;

				//coupling for micromagnetic mesh
				if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
					Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
					calculate_mm_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3(-(pMesh->h.x + pMesh_Bulk->h.x) / 2, 0, 0),
						J1, J2, pMesh->h.z,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for atomistic mesh
				else {

					//coupling rectangle in atomistic mesh in absolute coordinates
					VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
					Rect rect_c = Rect(
						DBL3(abs_pos.x - pMesh->M.h.x / 2 - M1.h.x, abs_pos.y - pMesh->M.h.y / 2, abs_pos.z - pMesh->M.h.z / 2),
						DBL3(abs_pos.x - pMesh->M.h.x / 2, abs_pos.y + pMesh->M.h.y / 2, abs_pos.z + pMesh->M.h.z / 2));

					calculate_atom_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
						J1, J2, pMesh->h.x,
						energy_bulk_old, energy_bulk_new);
				}
			}

			//+y coupling direction
			if (bulk_coupling_mask[spin_index].y & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[spin_index].y & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for micromagnetic mesh
				if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
					Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
					calculate_mm_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3(0, (pMesh->h.y + pMesh_Bulk->h.y) / 2, 0),
						J1, J2, pMesh->h.z,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for atomistic mesh
				else {

					//coupling rectangle in atomistic mesh in absolute coordinates
					VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
					Rect rect_c = Rect(
						DBL3(abs_pos.x - pMesh->M.h.x / 2, abs_pos.y + pMesh->M.h.y / 2, abs_pos.z - pMesh->M.h.z / 2),
						DBL3(abs_pos.x + pMesh->M.h.x / 2, abs_pos.y + pMesh->M.h.y / 2 + M1.h.y, abs_pos.z + pMesh->M.h.z / 2));

					calculate_atom_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
						J1, J2, pMesh->h.y,
						energy_bulk_old, energy_bulk_new);
				}
			}

			//-y coupling direction
			if (bulk_coupling_mask[spin_index].y & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[spin_index].y >> 16) - 1;
				num_couplings++;

				//coupling for micromagnetic mesh
				if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
					Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
					calculate_mm_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3(0, -(pMesh->h.y + pMesh_Bulk->h.y) / 2, 0),
						J1, J2, pMesh->h.z,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for atomistic mesh
				else {

					//coupling rectangle in atomistic mesh in absolute coordinates
					VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
					Rect rect_c = Rect(
						DBL3(abs_pos.x - pMesh->M.h.x / 2, abs_pos.y - pMesh->M.h.y / 2 - M1.h.y, abs_pos.z - pMesh->M.h.z / 2),
						DBL3(abs_pos.x + pMesh->M.h.x / 2, abs_pos.y - pMesh->M.h.y / 2, abs_pos.z + pMesh->M.h.z / 2));

					calculate_atom_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
						J1, J2, pMesh->h.y,
						energy_bulk_old, energy_bulk_new);
				}
			}

			//+z coupling direction
			if (bulk_coupling_mask[spin_index].z & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[spin_index].z & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for micromagnetic mesh
				if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
					Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
					calculate_mm_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3(0, 0, (pMesh->h.z + pMesh_Bulk->h.z) / 2),
						J1, J2, pMesh->h.z,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for atomistic mesh
				else {

					//coupling rectangle in atomistic mesh in absolute coordinates
					VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
					Rect rect_c = Rect(
						DBL3(abs_pos.x - pMesh->M.h.x / 2, abs_pos.y - pMesh->M.h.y / 2, abs_pos.z + pMesh->M.h.z / 2),
						DBL3(abs_pos.x + pMesh->M.h.x / 2, abs_pos.y + pMesh->M.h.y / 2, abs_pos.z + pMesh->M.h.z / 2 + M1.h.z));

					calculate_atom_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
						J1, J2, pMesh->h.z,
						energy_bulk_old, energy_bulk_new);
				}
			}

			//-z coupling direction
			if (bulk_coupling_mask[spin_index].z & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[spin_index].z >> 16) - 1;
				num_couplings++;

				//coupling for micromagnetic mesh
				if (mesh_idx < pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {
					Mesh*& pMesh_Bulk = (mesh_idx < pMeshFM_Bulk.size() ? pMeshFM_Bulk[mesh_idx] : pMeshAFM_Bulk[mesh_idx - pMeshFM_Bulk.size()]);
					calculate_mm_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						pMesh_Bulk, abs_pos - pMesh_Bulk->meshRect.s + DBL3(0, 0, -(pMesh->h.z + pMesh_Bulk->h.z) / 2),
						J1, J2, pMesh->h.z,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for atomistic mesh
				else {

					//coupling rectangle in atomistic mesh in absolute coordinates
					VEC_VC<DBL3>& M1 = paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()]->M1;
					Rect rect_c = Rect(
						DBL3(abs_pos.x - pMesh->M.h.x / 2, abs_pos.y - pMesh->M.h.y / 2, abs_pos.z - pMesh->M.h.z / 2 - M1.h.z),
						DBL3(abs_pos.x + pMesh->M.h.x / 2, abs_pos.y + pMesh->M.h.y / 2, abs_pos.z - pMesh->M.h.z / 2));

					calculate_atom_coupling(
						pMesh->M, pMesh->M2, Mnew_A, Mnew_B, spin_index,
						paMesh_Bulk[mesh_idx - pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()], rect_c,
						J1, J2, pMesh->h.z,
						energy_bulk_old, energy_bulk_new);
				}
			}

			if (num_couplings) {

				energy_old += energy_bulk_old / num_couplings;
				energy_new += energy_bulk_new / num_couplings;
			}
		}
	}

	//------------------

	if (Mnew_A != DBL3() && Mnew_B != DBL3()) return pMesh->h.dim() * (energy_new - energy_old);
	else return pMesh->h.dim() * energy_old;
}

//-------------------Torque methods

DBL3 SurfExchange_AFM::GetTorque(Rect& avRect)
{
#if COMPILECUDA == 1
	if (pModuleCUDA) return pModuleCUDA->GetTorque(avRect);
#endif

	return CalculateTorque(pMesh->M, avRect);
}

#endif