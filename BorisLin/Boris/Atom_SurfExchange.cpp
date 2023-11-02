#include "stdafx.h"
#include "Atom_SurfExchange.h"

#if defined(MODULE_COMPILATION_SURFEXCHANGE) && ATOMISTIC == 1

#include "SuperMesh.h"
#include "Atom_Mesh.h"
#include "Atom_MeshParamsControl.h"
#include "Mesh.h"
#include "MeshParamsControl.h"

/////////////////////////////////////////////////////////////////
//SurfExchange
//

//this mesh : mesh in which we set coupling field
//coupling mesh : mesh from which we obtain coupling direction

// Atom to Atom : top mesh along z sets coupling constants
// e.g.
// Hs = m * Js / mu0 mu_s
//
// Here m is direction from coupling mesh, Js (J) is coupling constant, mu_s is magnetic moment in this mesh

// FM to Atom : atomistic mesh sets coupling constant irrespective of z order
// e.g.
// Hs = m * Js / mu0 mu_s
//
// Here m is direction from coupling mesh, Js (J) is coupling constant, mu_s is magnetic moment in this mesh
//
// AFM to Atom : 
// e.g.
// 

Atom_SurfExchange::Atom_SurfExchange(Atom_Mesh *paMesh_) :
	Modules(),
	ProgramStateNames(this, {}, {})
{
	paMesh = paMesh_;

	//-------------------------- Is CUDA currently enabled?

	//If cuda is enabled we also need to make the cuda module version
	if (paMesh->cudaEnabled) {

		if (!error_on_create) error_on_create = SwitchCUDAState(true);
	}
}

Atom_SurfExchange::~Atom_SurfExchange()
{
}

BError Atom_SurfExchange::Initialize(void)
{
	BError error(CLASS_STR(Atom_SurfExchange));

	//Need to identify all magnetic meshes participating in surface exchange coupling with this module:
	//1. Must be atomistic and have the SurfExchange module set -> this results in surface exchange coupling
	//2. Must overlap in the x-y plane only with the mesh holding this module (either top or bottom) but not along z (i.e. mesh rectangles must not intersect)
	//3. No other magnetic meshes can be sandwiched in between - there could be other types of non-magnetic meshes in between of course (e.g. insulator, conductive layers etc).

	if (!initialized) {

		SuperMesh* pSMesh = paMesh->pSMesh;

		paMesh_Bot.clear();
		paMesh_Top.clear();
		pMesh_Bot.clear();
		pMesh_Top.clear();

		paMesh_Bulk.clear();
		pMeshFM_Bulk.clear();
		pMeshAFM_Bulk.clear();

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

		Rect meshRect = paMesh->GetMeshRect();

		Rect xy_meshRect = meshRect;
		xy_meshRect.s.z = 0; xy_meshRect.e.z = 0;

		for (int idx = 0; idx < pSMesh->size(); idx++) {

			//skip this mesh
			if ((*pSMesh)[idx]->get_id() == paMesh->get_id()) continue;

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

							if ((*pSMesh)[idx]->is_atomistic()) paMesh_Top.push_back(dynamic_cast<Atom_Mesh*>((*pSMesh)[idx]));
							else pMesh_Top.push_back(dynamic_cast<Mesh*>((*pSMesh)[idx]));
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

							if ((*pSMesh)[idx]->is_atomistic()) paMesh_Bot.push_back(dynamic_cast<Atom_Mesh*>((*pSMesh)[idx]));
							else pMesh_Bot.push_back(dynamic_cast<Mesh*>((*pSMesh)[idx]));
						}
					}
				}

				//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

				else if (meshRect.intersects(candidate_meshRect)) {

					if ((*pSMesh)[idx]->is_atomistic()) paMesh_Bulk.push_back(dynamic_cast<Atom_Mesh*>((*pSMesh)[idx]));
					else {

						Mesh* pMesh_Bulk = dynamic_cast<Mesh*>((*pSMesh)[idx]);
						if (pMesh_Bulk->GetMeshType() == MESH_FERROMAGNETIC) pMeshFM_Bulk.push_back(pMesh_Bulk);
						else if (pMesh_Bulk->GetMeshType() == MESH_ANTIFERROMAGNETIC) pMeshAFM_Bulk.push_back(pMesh_Bulk);
					}
				}
			}
		}

		//calculate bulk_coupling_mask if needed
		if (paMesh_Bulk.size() + pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {

			if (!bulk_coupling_mask.assign(paMesh->h, paMesh->meshRect, INT3())) error(BERROR_OUTOFMEMORY_CRIT);

			if (!error) {

				//for absolute position abs_pos search for a mesh coupled in the bulk which contains this position and has non-empty cell there
				//store in bulk_coupling_mask at idx if found
				//direction: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z
				auto set_bulk_coupling = [&](DBL3 abs_pos, VEC<INT3>& bulk_coupling_mask, int idx, int direction) -> void {

					int mesh_index = -1;

					//search atomistic meshes
					for (int midx = 0; midx < paMesh_Bulk.size(); midx++) {

						if (paMesh_Bulk[midx]->M1.rect.contains(abs_pos)) {
							if (paMesh_Bulk[midx]->M1.is_not_empty(abs_pos - paMesh_Bulk[midx]->M1.rect.s)) {

								//found mesh
								mesh_index = midx;
								break;
							}
						}
					}

					//if not found in atomistic meshes, search micromagnetic meshes - FM
					if (mesh_index == -1) {

						for (int midx = 0; midx < pMeshFM_Bulk.size(); midx++) {

							if (pMeshFM_Bulk[midx]->M.rect.contains(abs_pos)) {
								if (pMeshFM_Bulk[midx]->M.is_not_empty(abs_pos - pMeshFM_Bulk[midx]->M.rect.s)) {

									//found mesh (must add number of micromagnetic meshes for total compound index)
									mesh_index = midx + paMesh_Bulk.size();
									break;
								}
							}
						}
					}

					//if not found in atomistic meshes, search micromagnetic meshes - AFM
					if (mesh_index == -1) {

						for (int midx = 0; midx < pMeshAFM_Bulk.size(); midx++) {

							if (pMeshAFM_Bulk[midx]->M.rect.contains(abs_pos)) {
								if (pMeshAFM_Bulk[midx]->M.is_not_empty(abs_pos - pMeshAFM_Bulk[midx]->M.rect.s)) {

									//found mesh (must add number of micromagnetic meshes for total compound index)
									mesh_index = midx + paMesh_Bulk.size() + pMeshFM_Bulk.size();
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
				for (int idx = 0; idx < paMesh->M1.linear_size(); idx++) {

					if (paMesh->M1.is_not_empty(idx)) {

						std::vector<int> neighbors(6);
						//order is +x, -x, +y, -y, +z, -z
						paMesh->M1.get_neighbors(idx, neighbors);

						DBL3 abs_pos = paMesh->M1.cellidx_to_position(idx) + paMesh->M1.rect.s;

						//empty +x, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[0] == -1) set_bulk_coupling(abs_pos + DBL3(paMesh->M1.h.x, 0, 0), bulk_coupling_mask, idx, 0);
						//empty -x, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[1] == -1) set_bulk_coupling(abs_pos + DBL3(-paMesh->M1.h.x, 0, 0), bulk_coupling_mask, idx, 1);
						//empty +y, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[2] == -1) set_bulk_coupling(abs_pos + DBL3(0, paMesh->M1.h.y, 0), bulk_coupling_mask, idx, 2);
						//empty -y, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[3] == -1) set_bulk_coupling(abs_pos + DBL3(0, -paMesh->M1.h.y, 0), bulk_coupling_mask, idx, 3);
						//empty +z, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[4] == -1) set_bulk_coupling(abs_pos + DBL3(0, 0, paMesh->M1.h.z), bulk_coupling_mask, idx, 4);
						//empty -z, so idx is a surface cell - can we couple it to a cell in another mesh?
						if (neighbors[5] == -1) set_bulk_coupling(abs_pos + DBL3(0, 0, -paMesh->M1.h.z), bulk_coupling_mask, idx, 5);
					}
				}
			}
		}
		else bulk_coupling_mask.clear();

		initialized = true;
	}

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		paMesh->h, paMesh->meshRect,
		(MOD_)paMesh->Get_Module_Heff_Display() == MOD_SURFEXCHANGE || paMesh->IsOutputDataSet_withRect(DATA_E_SURFEXCH) || paMesh->IsOutputDataSet(DATA_T_SURFEXCH),
		(MOD_)paMesh->Get_Module_Energy_Display() == MOD_SURFEXCHANGE || paMesh->IsOutputDataSet_withRect(DATA_E_SURFEXCH));
	if (error) initialized = false;

	return error;
}

BError Atom_SurfExchange::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Atom_SurfExchange));

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

BError Atom_SurfExchange::MakeCUDAModule(void)
{
	BError error(CLASS_STR(Atom_SurfExchange));

#if COMPILECUDA == 1

	if (paMesh->paMeshCUDA) {

		//Note : it is posible pMeshCUDA has not been allocated yet, but this module has been created whilst cuda is switched on. This will happen when a new mesh is being made which adds this module by default.
		//In this case, after the mesh has been fully made, it will call SwitchCUDAState on the mesh, which in turn will call this SwitchCUDAState method; then pMeshCUDA will not be nullptr and we can make the cuda module version
		pModuleCUDA = new Atom_SurfExchangeCUDA(paMesh->paMeshCUDA, this);
		error = pModuleCUDA->Error_On_Create();
	}

#endif

	return error;
}

double Atom_SurfExchange::UpdateField(void)
{
	double energy = 0;
	
	SZ3 n = paMesh->n;

	//zero module display VECs if needed, since contributions must be added into them to account for possiblility of 2 contributions (top and bottom)
	ZeroModuleVECs();

	//------------------ Coupling functions

	auto calculate_atom_coupling = [](
		VEC_VC<DBL3>& M1, int cell_idx,
		Atom_Mesh* paMeshCoupled, DBL3 cell_rel_pos,
		double mu_s, double Js,
		DBL3& Hsurfexch, double& cell_energy) -> void
	{
		//get magnetization value in top mesh cell to couple with
		DBL3 m_j = normalize(paMeshCoupled->M1[cell_rel_pos]);
		DBL3 m_i = M1[cell_idx] / mu_s;

		double dot_prod = m_i * m_j;

		Hsurfexch += m_j * Js / (MUB_MU0 * mu_s);
		cell_energy += -Js * dot_prod / M1.h.dim();
	};

	auto calculate_mm_FM_coupling = [](
		VEC_VC<DBL3>& M1, int cell_idx,
		Mesh* pMeshCoupled, DBL3 cell_rel_pos,
		double mu_s, double Js,
		DBL3& Hsurfexch, double& cell_energy) -> void
	{
		//get magnetization value in top mesh cell to couple with
		DBL3 m_j = normalize(pMeshCoupled->M[cell_rel_pos]);
		DBL3 m_i = M1[cell_idx] / mu_s;

		double dot_prod = m_i * m_j;

		Hsurfexch += m_j * Js / (MUB_MU0 * mu_s);
		cell_energy += -Js * dot_prod / M1.h.dim();
	};

	auto calculate_mm_AFM_coupling = [](
		VEC_VC<DBL3>& M1, int cell_idx,
		Mesh* pMeshCoupled, DBL3 cell_rel_pos,
		double mu_s, double Js, double Js2,
		DBL3& Hsurfexch, double& cell_energy) -> void
	{
		//get magnetization value in top mesh cell to couple with
		DBL3 m_j1 = normalize(pMeshCoupled->M[cell_rel_pos]);
		DBL3 m_j2 = normalize(pMeshCoupled->M2[cell_rel_pos]);
		DBL3 m_i = M1[cell_idx] / mu_s;

		double dot_prod1 = m_i * m_j1;
		double dot_prod2 = m_i * m_j2;

		Hsurfexch += m_j1 * Js / (MUB_MU0 * mu_s);
		Hsurfexch += m_j2 * Js2 / (MUB_MU0 * mu_s);
		cell_energy += -Js * dot_prod1 / M1.h.dim();
		cell_energy += -Js2 * dot_prod2 / M1.h.dim();
	};

	//------------------ SURFACE COUPLING Z STACKING

	if (paMesh_Top.size() || pMesh_Top.size()) {

		//surface exchange coupling at the top
#pragma omp parallel for reduction(+:energy)
		for (int j = 0; j < n.y; j++) {
			for (int i = 0; i < n.x; i++) {

				int cell_idx = i + j * n.x + (n.z - 1) * n.x*n.y;

				//empty cell here ... next
				if (paMesh->M1.is_empty(cell_idx)) continue;

				double mu_s = paMesh->mu_s;
				paMesh->update_parameters_mcoarse(cell_idx, paMesh->mu_s, mu_s);

				//effective field and energy for this cell
				DBL3 Hsurfexch = DBL3();
				double cell_energy = 0.0;
				bool cell_coupled = false;

				//check all meshes for coupling
				//1. coupling from other atomistic meshes
				for (int mesh_idx = 0; mesh_idx < (int)paMesh_Top.size(); mesh_idx++) {

					Rect tmeshRect = paMesh_Top[mesh_idx]->GetMeshRect();

					//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
					DBL3 cell_rel_pos = DBL3(
						(i + 0.5) * paMesh->h.x + paMesh->meshRect.s.x - tmeshRect.s.x,
						(j + 0.5) * paMesh->h.y + paMesh->meshRect.s.y - tmeshRect.s.y,
						paMesh_Top[mesh_idx]->h.z / 2);

					//can't couple to an empty cell
					if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s) || paMesh_Top[mesh_idx]->M1.is_empty(cell_rel_pos)) continue;

					//Top mesh sets Js
					double Js = paMesh_Top[mesh_idx]->Js;
					paMesh_Top[mesh_idx]->update_parameters_atposition(cell_rel_pos, paMesh_Top[mesh_idx]->Js, Js);

					calculate_atom_coupling(
						paMesh->M1, cell_idx,
						paMesh_Top[mesh_idx], cell_rel_pos,
						mu_s, Js,
						Hsurfexch, cell_energy);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					cell_coupled = true;
					break;
				}

				if (!cell_coupled) {

					//2. coupling from micromagnetic meshes
					for (int mesh_idx = 0; mesh_idx < (int)pMesh_Top.size(); mesh_idx++) {

						Rect tmeshRect = pMesh_Top[mesh_idx]->GetMeshRect();

						//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
						DBL3 cell_rel_pos = DBL3(
							(i + 0.5) * paMesh->h.x + paMesh->meshRect.s.x - tmeshRect.s.x,
							(j + 0.5) * paMesh->h.y + paMesh->meshRect.s.y - tmeshRect.s.y,
							pMesh_Top[mesh_idx]->h.z / 2);

						//can't couple to an empty cell
						if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s) || pMesh_Top[mesh_idx]->M.is_empty(cell_rel_pos)) continue;

						if (pMesh_Top[mesh_idx]->GetMeshType() == MESH_FERROMAGNETIC) {

							//atomistic mesh sets coupling
							double Js = paMesh->Js;
							paMesh->update_parameters_mcoarse(cell_idx, paMesh->Js, Js);

							calculate_mm_FM_coupling(
								paMesh->M1, cell_idx,
								pMesh_Top[mesh_idx], cell_rel_pos,
								mu_s, Js,
								Hsurfexch, cell_energy);
						}
						else if (pMesh_Top[mesh_idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

							//atomistic mesh sets coupling
							double Js = paMesh->Js;
							double Js2 = paMesh->Js2;
							paMesh->update_parameters_mcoarse(cell_idx, paMesh->Js, Js, paMesh->Js2, Js2);

							calculate_mm_AFM_coupling(
								paMesh->M1, cell_idx,
								pMesh_Top[mesh_idx], cell_rel_pos,
								mu_s, Js, Js2,
								Hsurfexch, cell_energy);
						}

						//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
						break;
					}
				}

				paMesh->Heff1[cell_idx] += Hsurfexch;
				energy += cell_energy;

				if (Module_Heff.linear_size()) Module_Heff[cell_idx] += Hsurfexch;
				if (Module_energy.linear_size()) Module_energy[cell_idx] += cell_energy;
			}
		}
	}

	if (paMesh_Bot.size() || pMesh_Bot.size()) {

		//surface exchange coupling at the bottom
#pragma omp parallel for reduction(+:energy)
		for (int j = 0; j < n.y; j++) {
			for (int i = 0; i < n.x; i++) {

				int cell_idx = i + j * n.x;

				//empty cell here ... next
				if (paMesh->M1.is_empty(cell_idx)) continue;

				double mu_s = paMesh->mu_s;
				double Js = paMesh->Js;
				paMesh->update_parameters_mcoarse(cell_idx, paMesh->mu_s, mu_s, paMesh->Js, Js);

				//effective field and energy for this cell
				DBL3 Hsurfexch = DBL3();
				double cell_energy = 0.0;
				bool cell_coupled = false;

				//check all meshes for coupling
				//1. coupling from other atomistic meshes
				for (int mesh_idx = 0; mesh_idx < (int)paMesh_Bot.size(); mesh_idx++) {

					Rect bmeshRect = paMesh_Bot[mesh_idx]->GetMeshRect();

					//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
					DBL3 cell_rel_pos = DBL3(
						(i + 0.5) * paMesh->h.x + paMesh->meshRect.s.x - bmeshRect.s.x,
						(j + 0.5) * paMesh->h.y + paMesh->meshRect.s.y - bmeshRect.s.y,
						bmeshRect.height() - (paMesh_Bot[mesh_idx]->h.z / 2));

					//can't couple to an empty cell
					if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s) || paMesh_Bot[mesh_idx]->M1.is_empty(cell_rel_pos)) continue;

					calculate_atom_coupling(
						paMesh->M1, cell_idx,
						paMesh_Bot[mesh_idx], cell_rel_pos,
						mu_s, Js,
						Hsurfexch, cell_energy);

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					cell_coupled = true;
					break;
				}

				if (!cell_coupled) {

					//2. coupling from micromagnetic meshes
					for (int mesh_idx = 0; mesh_idx < (int)pMesh_Bot.size(); mesh_idx++) {

						Rect bmeshRect = pMesh_Bot[mesh_idx]->GetMeshRect();

						//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
						DBL3 cell_rel_pos = DBL3(
							(i + 0.5) * paMesh->h.x + paMesh->meshRect.s.x - bmeshRect.s.x,
							(j + 0.5) * paMesh->h.y + paMesh->meshRect.s.y - bmeshRect.s.y,
							bmeshRect.height() - pMesh_Bot[mesh_idx]->h.z / 2);

						//can't couple to an empty cell
						if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s) || pMesh_Bot[mesh_idx]->M.is_empty(cell_rel_pos)) continue;
						
						if (pMesh_Bot[mesh_idx]->GetMeshType() == MESH_FERROMAGNETIC) {

							//atomistic mesh sets coupling
							double Js = paMesh->Js;
							paMesh->update_parameters_mcoarse(cell_idx, paMesh->Js, Js);

							calculate_mm_FM_coupling(
								paMesh->M1, cell_idx,
								pMesh_Bot[mesh_idx], cell_rel_pos,
								mu_s, Js,
								Hsurfexch, cell_energy);
						}
						else if (pMesh_Bot[mesh_idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

							//atomistic mesh sets coupling
							double Js = paMesh->Js;
							double Js2 = paMesh->Js2;
							paMesh->update_parameters_mcoarse(cell_idx, paMesh->Js, Js, paMesh->Js2, Js2);

							calculate_mm_AFM_coupling(
								paMesh->M1, cell_idx,
								pMesh_Bot[mesh_idx], cell_rel_pos,
								mu_s, Js, Js2,
								Hsurfexch, cell_energy);
						}

						//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
						break;
					}
				}

				paMesh->Heff1[cell_idx] += Hsurfexch;
				energy += cell_energy;

				if (Module_Heff.linear_size()) Module_Heff[cell_idx] = Hsurfexch;
				if (Module_energy.linear_size()) Module_energy[cell_idx] = cell_energy;
			}
		}
	}

	//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

	if (paMesh_Bulk.size() + pMeshFM_Bulk.size() + pMeshAFM_Bulk.size()) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->M1.linear_size(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && bulk_coupling_mask[idx] != INT3()) {

				double mu_s = paMesh->mu_s;
				double Js = paMesh->Js;
				double Js2 = paMesh->Js2;
				paMesh->update_parameters_mcoarse(idx, paMesh->mu_s, mu_s, paMesh->Js, Js, paMesh->Js2, Js2);

				//surface cell which needs to be exchange coupled
				DBL3 Hsurfexch = DBL3();
				double cell_energy = 0.0;
				int num_couplings = 0;

				DBL3 abs_pos = paMesh->M1.cellidx_to_position(idx) + paMesh->M1.rect.s;

				//+x coupling direction
				if (bulk_coupling_mask[idx].x & 0x0000ffff) {

					int mesh_idx = (bulk_coupling_mask[idx].x & 0x0000ffff) - 1;
					num_couplings++;

					//coupling for atomistic mesh
					if (mesh_idx < paMesh_Bulk.size()) {		
						calculate_atom_coupling(
							paMesh->M1, idx,
							paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3((paMesh->h.x + paMesh_Bulk[mesh_idx]->h.x) / 2, 0, 0),
							mu_s, Js,
							Hsurfexch, cell_energy);
					}
					//coupling for micromagnetic mesh
					else {

						//FM
						if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
							calculate_mm_FM_coupling(
								paMesh->M1, idx,
								pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()], 
								abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3((paMesh->h.x + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.x) / 2, 0, 0),
								mu_s, Js,
								Hsurfexch, cell_energy);
						}
						//AFM
						else {
							calculate_mm_AFM_coupling(
								paMesh->M1, idx,
								pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()], 
								abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3((paMesh->h.x + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.x) / 2, 0, 0),
								mu_s, Js, Js2,
								Hsurfexch, cell_energy);
						}
					}
				}

				//-x coupling direction
				if (bulk_coupling_mask[idx].x & 0xffff0000) {

					int mesh_idx = (bulk_coupling_mask[idx].x >> 16) - 1;
					num_couplings++;

					//coupling for atomistic mesh
					if (mesh_idx < paMesh_Bulk.size()) {
						calculate_atom_coupling(
							paMesh->M1, idx,
							paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3(-(paMesh->h.x + paMesh_Bulk[mesh_idx]->h.x) / 2, 0, 0),
							mu_s, Js,
							Hsurfexch, cell_energy);
					}
					//coupling for micromagnetic mesh
					else {

						//FM
						if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
							calculate_mm_FM_coupling(
								paMesh->M1, idx,
								pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
								abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3(-(paMesh->h.x + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.x) / 2, 0, 0),
								mu_s, Js,
								Hsurfexch, cell_energy);
						}
						//AFM
						else {
							calculate_mm_AFM_coupling(
								paMesh->M1, idx,
								pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
								abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3(-(paMesh->h.x + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.x) / 2, 0, 0),
								mu_s, Js, Js2,
								Hsurfexch, cell_energy);
						}
					}
				}

				//+y coupling direction
				if (bulk_coupling_mask[idx].y & 0x0000ffff) {

					int mesh_idx = (bulk_coupling_mask[idx].y & 0x0000ffff) - 1;
					num_couplings++;

					//coupling for atomistic mesh
					if (mesh_idx < paMesh_Bulk.size()) {
						calculate_atom_coupling(
							paMesh->M1, idx,
							paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3(0, (paMesh->h.y + paMesh_Bulk[mesh_idx]->h.y) / 2, 0),
							mu_s, Js,
							Hsurfexch, cell_energy);
					}
					//coupling for micromagnetic mesh
					else {

						//FM
						if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
							calculate_mm_FM_coupling(
								paMesh->M1, idx,
								pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
								abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3(0, (paMesh->h.y + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.y) / 2, 0),
								mu_s, Js,
								Hsurfexch, cell_energy);
						}
						//AFM
						else {
							calculate_mm_AFM_coupling(
								paMesh->M1, idx,
								pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
								abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3(0, (paMesh->h.y + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.y) / 2, 0),
								mu_s, Js, Js2,
								Hsurfexch, cell_energy);
						}
					}
				}

				//-y coupling direction
				if (bulk_coupling_mask[idx].y & 0xffff0000) {

					int mesh_idx = (bulk_coupling_mask[idx].y >> 16) - 1;
					num_couplings++;

					//coupling for atomistic mesh
					if (mesh_idx < paMesh_Bulk.size()) {
						calculate_atom_coupling(
							paMesh->M1, idx,
							paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3(0, -(paMesh->h.y + paMesh_Bulk[mesh_idx]->h.y) / 2, 0),
							mu_s, Js,
							Hsurfexch, cell_energy);
					}
					//coupling for micromagnetic mesh
					else {

						//FM
						if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
							calculate_mm_FM_coupling(
								paMesh->M1, idx,
								pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
								abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3(0, -(paMesh->h.y + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.y) / 2, 0),
								mu_s, Js,
								Hsurfexch, cell_energy);
						}
						//AFM
						else {
							calculate_mm_AFM_coupling(
								paMesh->M1, idx,
								pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
								abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3(0, -(paMesh->h.y + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.y) / 2, 0),
								mu_s, Js, Js2,
								Hsurfexch, cell_energy);
						}
					}
				}

				//+z coupling direction
				if (bulk_coupling_mask[idx].z & 0x0000ffff) {

					int mesh_idx = (bulk_coupling_mask[idx].z & 0x0000ffff) - 1;
					num_couplings++;

					//coupling for atomistic mesh
					if (mesh_idx < paMesh_Bulk.size()) {
						calculate_atom_coupling(
							paMesh->M1, idx,
							paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3(0,  0, (paMesh->h.z + paMesh_Bulk[mesh_idx]->h.z) / 2),
							mu_s, Js,
							Hsurfexch, cell_energy);
					}
					//coupling for micromagnetic mesh
					else {

						//FM
						if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
							calculate_mm_FM_coupling(
								paMesh->M1, idx,
								pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
								abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3(0, 0, (paMesh->h.z + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.z) / 2),
								mu_s, Js,
								Hsurfexch, cell_energy);
						}
						//AFM
						else {
							calculate_mm_AFM_coupling(
								paMesh->M1, idx,
								pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
								abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3(0, 0, (paMesh->h.z + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.z) / 2),
								mu_s, Js, Js2,
								Hsurfexch, cell_energy);
						}
					}
				}

				//-z coupling direction
				if (bulk_coupling_mask[idx].z & 0xffff0000) {

					int mesh_idx = (bulk_coupling_mask[idx].z >> 16) - 1;
					num_couplings++;

					//coupling for atomistic mesh
					if (mesh_idx < paMesh_Bulk.size()) {
						calculate_atom_coupling(
							paMesh->M1, idx,
							paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3(0, 0, -(paMesh->h.z + paMesh_Bulk[mesh_idx]->h.z) / 2),
							mu_s, Js,
							Hsurfexch, cell_energy);
					}
					//coupling for micromagnetic mesh
					else {

						//FM
						if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
							calculate_mm_FM_coupling(
								paMesh->M1, idx,
								pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
								abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3(0, 0, -(paMesh->h.z + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.z) / 2),
								mu_s, Js,
								Hsurfexch, cell_energy);
						}
						//AFM
						else {
							calculate_mm_AFM_coupling(
								paMesh->M1, idx,
								pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
								abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3(0, 0, -(paMesh->h.z + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.z) / 2),
								mu_s, Js, Js2,
								Hsurfexch, cell_energy);
						}
					}
				}

				if (num_couplings) {

					//need average if cell receives multiple coupling contributions
					Hsurfexch /= num_couplings;
					cell_energy /= num_couplings;

					paMesh->Heff1[idx] += Hsurfexch;
					energy += cell_energy;

					if (Module_Heff.linear_size()) Module_Heff[idx] = Hsurfexch;
					if (Module_energy.linear_size()) Module_energy[idx] = cell_energy;
				}
			}
		}
	}

	//------------------

	energy /= paMesh->M1.get_nonempty_cells();
	this->energy = energy;

	return this->energy;
}

//-------------------Torque methods

DBL3 Atom_SurfExchange::GetTorque(Rect& avRect)
{
#if COMPILECUDA == 1
	if (pModuleCUDA) return pModuleCUDA->GetTorque(avRect);
#endif

	return CalculateTorque(paMesh->M1, avRect);
}

//-------------------Energy methods

//For simple cubic mesh spin_index coincides with index in M1
double Atom_SurfExchange::Get_EnergyChange(int spin_index, DBL3 Mnew)
{
	double energy_new = 0, energy_old = 0;

	SZ3 n = paMesh->n;

	//------------------ Coupling functions

	auto calculate_atom_coupling = [](
		VEC_VC<DBL3>& M1, DBL3 Mnew, int spin_index,
		Atom_Mesh* paMeshCoupled, DBL3 cell_rel_pos,
		double Js,
		double& energy_old, double& energy_new) -> void
	{
		//get magnetization value in top mesh cell to couple with
		DBL3 m_j = normalize(paMeshCoupled->M1[cell_rel_pos]);
		DBL3 m_i = normalize(M1[spin_index]);
		double dot_prod = m_i * m_j;
		energy_old = -Js * dot_prod;

		if (Mnew != DBL3()) {

			DBL3 mnew_i = normalize(Mnew);
			double dot_prod_new = mnew_i * m_j;
			energy_new = -Js * dot_prod_new;
		}
	};

	auto calculate_mm_FM_coupling = [](
		VEC_VC<DBL3>& M1, DBL3 Mnew, int spin_index,
		Mesh* pMeshCoupled, DBL3 cell_rel_pos,
		double Js,
		double& energy_old, double& energy_new) -> void
	{
		//get magnetization value in top mesh cell to couple with
		DBL3 m_j = normalize(pMeshCoupled->M[cell_rel_pos]);
		DBL3 m_i = normalize(M1[spin_index]);

		double dot_prod = m_i * m_j;
		energy_old = -Js * dot_prod;

		if (Mnew != DBL3()) {

			DBL3 mnew_i = normalize(Mnew);
			double dot_prod_new = mnew_i * m_j;
			energy_new = -Js * dot_prod_new;
		}
	};

	auto calculate_mm_AFM_coupling = [](
		VEC_VC<DBL3>& M1, DBL3 Mnew, int spin_index,
		Mesh* pMeshCoupled, DBL3 cell_rel_pos,
		double Js, double Js2,
		double& energy_old, double& energy_new) -> void
	{
		//get magnetization value in top mesh cell to couple with
		DBL3 m_j1 = normalize(pMeshCoupled->M[cell_rel_pos]);
		DBL3 m_j2 = normalize(pMeshCoupled->M2[cell_rel_pos]);
		DBL3 m_i = normalize(M1[spin_index]);

		double dot_prod1 = m_i * m_j1;
		double dot_prod2 = m_i * m_j2;
		energy_old = -Js * dot_prod1;
		energy_old += -Js2 * dot_prod2;

		if (Mnew != DBL3()) {

			DBL3 mnew_i = normalize(Mnew);
			double dot_prod_new1 = mnew_i * m_j1;
			double dot_prod_new2 = mnew_i * m_j2;
			energy_new = -Js * dot_prod_new1;
			energy_new += -Js2 * dot_prod_new2;
		}
	};

	//------------------ SURFACE COUPLING Z STACKING

	//if spin is on top surface then look at paMesh_Top
	if (spin_index / (n.x * n.y) == n.z - 1 && (paMesh_Top.size() || pMesh_Top.size())) {

		if (!paMesh->M1.is_empty(spin_index)) {

			int i = spin_index % n.x;
			int j = (spin_index / n.x) % n.y;
			bool cell_coupled = false;

			//check all meshes for coupling
			//1. coupling from other atomistic meshes
			for (int mesh_idx = 0; mesh_idx < (int)paMesh_Top.size(); mesh_idx++) {

				Rect tmeshRect = paMesh_Top[mesh_idx]->GetMeshRect();

				//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
				DBL3 cell_rel_pos = DBL3(
					(i + 0.5) * paMesh->h.x + paMesh->meshRect.s.x - tmeshRect.s.x,
					(j + 0.5) * paMesh->h.y + paMesh->meshRect.s.y - tmeshRect.s.y,
					paMesh_Top[mesh_idx]->h.z / 2);

				//can't couple to an empty cell
				if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s) || paMesh_Top[mesh_idx]->M1.is_empty(cell_rel_pos)) continue;

				//top mesh sets Js
				double Js = paMesh_Top[mesh_idx]->Js;
				paMesh_Top[mesh_idx]->update_parameters_atposition(cell_rel_pos, paMesh_Top[mesh_idx]->Js, Js);

				calculate_atom_coupling(
					paMesh->M1, Mnew, spin_index,
					paMesh_Top[mesh_idx], cell_rel_pos,
					Js,
					energy_old, energy_new);

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				cell_coupled = true;
				break;
			}

			if (!cell_coupled) {

				//2. coupling from micromagnetic meshes
				for (int mesh_idx = 0; mesh_idx < (int)pMesh_Top.size(); mesh_idx++) {

					Rect tmeshRect = pMesh_Top[mesh_idx]->GetMeshRect();

					//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
					DBL3 cell_rel_pos = DBL3(
						(i + 0.5) * paMesh->h.x + paMesh->meshRect.s.x - tmeshRect.s.x,
						(j + 0.5) * paMesh->h.y + paMesh->meshRect.s.y - tmeshRect.s.y,
						pMesh_Top[mesh_idx]->h.z / 2);

					//can't couple to an empty cell
					if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s) || pMesh_Top[mesh_idx]->M.is_empty(cell_rel_pos)) continue;

					if (pMesh_Top[mesh_idx]->GetMeshType() == MESH_FERROMAGNETIC) {

						//atomistic mesh sets coupling
						double Js = paMesh->Js;
						paMesh->update_parameters_mcoarse(spin_index, paMesh->Js, Js);

						calculate_mm_FM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMesh_Top[mesh_idx], cell_rel_pos,
							Js,
							energy_old, energy_new);
					}
					else if (pMesh_Top[mesh_idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						//atomistic mesh sets coupling
						double Js = paMesh->Js;
						double Js2 = paMesh->Js2;
						paMesh->update_parameters_mcoarse(spin_index, paMesh->Js, Js, paMesh->Js2, Js2);

						calculate_mm_AFM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMesh_Top[mesh_idx], cell_rel_pos,
							Js, Js2,
							energy_old, energy_new);
					}

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					break;
				}
			}
		}
	}

	//if spin is on bottom surface then look at paMesh_Top
	if (spin_index / (n.x * n.y) == 0 && (paMesh_Bot.size() || pMesh_Bot.size())) {

		if (!paMesh->M1.is_empty(spin_index)) {

			int i = spin_index % n.x;
			int j = (spin_index / n.x) % n.y;
			bool cell_coupled = false;

			double Js = paMesh->Js;
			paMesh->update_parameters_mcoarse(spin_index, paMesh->Js, Js);

			//check all meshes for coupling
			for (int mesh_idx = 0; mesh_idx < (int)paMesh_Bot.size(); mesh_idx++) {

				Rect bmeshRect = paMesh_Bot[mesh_idx]->GetMeshRect();

				//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
				DBL3 cell_rel_pos = DBL3(
					(i + 0.5) * paMesh->h.x + paMesh->meshRect.s.x - bmeshRect.s.x,
					(j + 0.5) * paMesh->h.y + paMesh->meshRect.s.y - bmeshRect.s.y,
					paMesh_Bot[mesh_idx]->meshRect.e.z - paMesh_Bot[mesh_idx]->meshRect.s.z - (paMesh_Bot[mesh_idx]->h.z / 2));

				//can't couple to an empty cell
				if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s) || paMesh_Bot[mesh_idx]->M1.is_empty(cell_rel_pos)) continue;

				calculate_atom_coupling(
					paMesh->M1, Mnew, spin_index,
					paMesh_Bot[mesh_idx], cell_rel_pos,
					Js,
					energy_old, energy_new);

				//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
				cell_coupled = true;
				break;
			}

			if (!cell_coupled) {

				//2. coupling from micromagnetic meshes
				for (int mesh_idx = 0; mesh_idx < (int)pMesh_Bot.size(); mesh_idx++) {

					Rect bmeshRect = pMesh_Bot[mesh_idx]->GetMeshRect();

					//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
					DBL3 cell_rel_pos = DBL3(
						(i + 0.5) * paMesh->h.x + paMesh->meshRect.s.x - bmeshRect.s.x,
						(j + 0.5) * paMesh->h.y + paMesh->meshRect.s.y - bmeshRect.s.y,
						bmeshRect.height() - pMesh_Bot[mesh_idx]->h.z / 2);

					//can't couple to an empty cell
					if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s) || pMesh_Bot[mesh_idx]->M.is_empty(cell_rel_pos)) continue;

					if (pMesh_Bot[mesh_idx]->GetMeshType() == MESH_FERROMAGNETIC) {

						calculate_mm_FM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMesh_Bot[mesh_idx], cell_rel_pos,
							Js,
							energy_old, energy_new);
					}
					else if (pMesh_Bot[mesh_idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						double Js2 = paMesh->Js2;
						paMesh->update_parameters_mcoarse(spin_index, paMesh->Js2, Js2);

						calculate_mm_AFM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMesh_Bot[mesh_idx], cell_rel_pos,
							Js, Js2,
							energy_old, energy_new);
					}

					//for each cell, either it's not coupled to any other mesh cell (so we never get here), or else it's coupled to exactly one cell on this surface (thus can stop looping over meshes now)
					break;
				}
			}
		}
	}

	//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

	if (pMeshFM_Bulk.size() + pMeshAFM_Bulk.size() + paMesh_Bulk.size()) {

		if (paMesh->M1.is_not_empty(spin_index) && bulk_coupling_mask[spin_index] != INT3()) {

			double energy_bulk_new = 0, energy_bulk_old = 0;

			double Js = paMesh->Js;
			double Js2 = paMesh->Js2;
			paMesh->update_parameters_mcoarse(spin_index, paMesh->Js, Js, paMesh->Js2, Js2);

			int num_couplings = 0;

			DBL3 abs_pos = paMesh->M1.cellidx_to_position(spin_index) + paMesh->M1.rect.s;

			//+x coupling direction
			if (bulk_coupling_mask[spin_index].x & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[spin_index].x & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < paMesh_Bulk.size()) {
					calculate_atom_coupling(
						paMesh->M1, Mnew, spin_index,
						paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3((paMesh->h.x + paMesh_Bulk[mesh_idx]->h.x) / 2, 0, 0),
						Js,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
						calculate_mm_FM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
							abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3((paMesh->h.x + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.x) / 2, 0, 0),
							Js,
							energy_bulk_old, energy_bulk_new);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
							abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3((paMesh->h.x + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.x) / 2, 0, 0),
							Js, Js2,
							energy_bulk_old, energy_bulk_new);
					}
				}
			}

			//-x coupling direction
			if (bulk_coupling_mask[spin_index].x & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[spin_index].x >> 16) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < paMesh_Bulk.size()) {
					calculate_atom_coupling(
						paMesh->M1, Mnew, spin_index,
						paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3(-(paMesh->h.x + paMesh_Bulk[mesh_idx]->h.x) / 2, 0, 0),
						Js,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
						calculate_mm_FM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
							abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3(-(paMesh->h.x + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.x) / 2, 0, 0),
							Js,
							energy_bulk_old, energy_bulk_new);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
							abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3(-(paMesh->h.x + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.x) / 2, 0, 0),
							Js, Js2,
							energy_bulk_old, energy_bulk_new);
					}
				}
			}

			//+y coupling direction
			if (bulk_coupling_mask[spin_index].y & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[spin_index].y & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < paMesh_Bulk.size()) {
					calculate_atom_coupling(
						paMesh->M1, Mnew, spin_index,
						paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3(0, (paMesh->h.y + paMesh_Bulk[mesh_idx]->h.y) / 2, 0),
						Js,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
						calculate_mm_FM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
							abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3(0, (paMesh->h.y + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.y) / 2, 0),
							Js,
							energy_bulk_old, energy_bulk_new);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
							abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3(0, (paMesh->h.y + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.y) / 2, 0),
							Js, Js2,
							energy_bulk_old, energy_bulk_new);
					}
				}
			}

			//-y coupling direction
			if (bulk_coupling_mask[spin_index].y & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[spin_index].y >> 16) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < paMesh_Bulk.size()) {
					calculate_atom_coupling(
						paMesh->M1, Mnew, spin_index,
						paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3(0, -(paMesh->h.y + paMesh_Bulk[mesh_idx]->h.y) / 2, 0),
						Js,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
						calculate_mm_FM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
							abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3(0, -(paMesh->h.y + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.y) / 2, 0),
							Js,
							energy_bulk_old, energy_bulk_new);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
							abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3(0, -(paMesh->h.y + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.y) / 2, 0),
							Js, Js2,
							energy_bulk_old, energy_bulk_new);
					}
				}
			}

			//+z coupling direction
			if (bulk_coupling_mask[spin_index].z & 0x0000ffff) {

				int mesh_idx = (bulk_coupling_mask[spin_index].z & 0x0000ffff) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < paMesh_Bulk.size()) {
					calculate_atom_coupling(
						paMesh->M1, Mnew, spin_index,
						paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3(0, 0, (paMesh->h.z + paMesh_Bulk[mesh_idx]->h.z) / 2),
						Js,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
						calculate_mm_FM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
							abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3(0, 0, (paMesh->h.z + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.z) / 2),
							Js,
							energy_bulk_old, energy_bulk_new);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
							abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3(0, 0, (paMesh->h.z + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.z) / 2),
							Js, Js2,
							energy_bulk_old, energy_bulk_new);
					}
				}
			}

			//-z coupling direction
			if (bulk_coupling_mask[spin_index].z & 0xffff0000) {

				int mesh_idx = (bulk_coupling_mask[spin_index].z >> 16) - 1;
				num_couplings++;

				//coupling for atomistic mesh
				if (mesh_idx < paMesh_Bulk.size()) {
					calculate_atom_coupling(
						paMesh->M1, Mnew, spin_index,
						paMesh_Bulk[mesh_idx], abs_pos - paMesh_Bulk[mesh_idx]->meshRect.s + DBL3(0, 0, -(paMesh->h.z + paMesh_Bulk[mesh_idx]->h.z) / 2),
						Js,
						energy_bulk_old, energy_bulk_new);
				}
				//coupling for micromagnetic mesh
				else {

					//FM
					if (mesh_idx < paMesh_Bulk.size() + pMeshFM_Bulk.size()) {
						calculate_mm_FM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()],
							abs_pos - pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->meshRect.s + DBL3(0, 0, -(paMesh->h.z + pMeshFM_Bulk[mesh_idx - paMesh_Bulk.size()]->h.z) / 2),
							Js,
							energy_bulk_old, energy_bulk_new);
					}
					//AFM
					else {
						calculate_mm_AFM_coupling(
							paMesh->M1, Mnew, spin_index,
							pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()],
							abs_pos - pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->meshRect.s + DBL3(0, 0, -(paMesh->h.z + pMeshAFM_Bulk[mesh_idx - paMesh_Bulk.size() - pMeshFM_Bulk.size()]->h.z) / 2),
							Js, Js2,
							energy_bulk_old, energy_bulk_new);
					}
				}
			}

			if (num_couplings) {

				energy_old += energy_bulk_old / num_couplings;
				energy_new += energy_bulk_new / num_couplings;
			}
		}
	}

	//------------------

	if (Mnew != DBL3()) return energy_new - energy_old;
	else return energy_old;
}

#endif