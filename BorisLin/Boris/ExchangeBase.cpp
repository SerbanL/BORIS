#include "stdafx.h"
#include "ExchangeBase.h"
#include "SuperMesh.h"
#include "Mesh.h"
#include "Atom_Mesh.h"
#include "MeshParamsControl.h"

ExchangeBase::ExchangeBase(MeshBase *pMeshBase_)
{
	pMeshBase = pMeshBase_;
}

BError ExchangeBase::Initialize(void)
{
	BError error(CLASS_STR(ExchangeBase));

	//clear everything then rebuild
	pM.clear();
	pMeshes.clear();
	CMBNDcontacts.clear();

	if (pMeshBase->GetMeshExchangeCoupling()) {

		//ExchangeBase given friend access to Mesh
		SuperMesh* pSMesh = pMeshBase->pSMesh;

		//now build pM
		for (int idx = 0; idx < pSMesh->size(); idx++) {

			//use MComputation_Enabled check, not magnetization_Enabled check as we only want to couple to other (anti)ferromagnetic meshes, not dipole meshes.
			//additionally only couple like meshes, e.g. FM to FM, AFM to AFM, but not FM to AFM - for the latter the coupling should be through the exchange bias mechanism (surface exchange coupling).
			//here we only make a list of all possible meshes which could be exchange coupled; the actual coupling is done, where applicable, in the respective exchange modules.
			if ((*pSMesh)[idx]->MComputation_Enabled() && (*pSMesh)[idx]->GetMeshExchangeCoupling()) {

				pMeshes.push_back((*pSMesh)[idx]);

				if (!(*pSMesh)[idx]->is_atomistic()) {

					pM.push_back(&dynamic_cast<Mesh*>((*pSMesh)[idx])->M);
				}
				else {

					pM.push_back(&dynamic_cast<Atom_Mesh*>((*pSMesh)[idx])->M1);
				}
			}
		}

		//set cmbnd flags
		for (int idx = 0; idx < pMeshes.size(); idx++) {

			if (pMeshes[idx] == pMeshBase) {

				//set CMBND flags, even for 1 cell thickness in cmbnd direction
				CMBNDcontacts = pM[idx]->set_cmbnd_flags(idx, pM, false);
				break;
			}
		}
	}

	return error;
}

//calculate exchange field at coupled cells in this mesh; accumulate energy density contribution in energy
void ExchangeBase::CalculateExchangeCoupling(
	double& energy, 
	std::function<double(int, int, DBL3, DBL3, DBL3, MeshBase&, MeshBase&)> calculate_coupling)
{
	for (int contact_idx = 0; contact_idx < CMBNDcontacts.size(); contact_idx++) {

		//the contacting meshes indexes : secondary mesh index is the one in contact with this one (the primary)
		int idx_sec = CMBNDcontacts[contact_idx].mesh_idx.i;
		int idx_pri = CMBNDcontacts[contact_idx].mesh_idx.j;

		//secondary and primary ferromagnetic meshes
		MeshBase& Mesh_pri = *pMeshes[idx_pri];
		MeshBase& Mesh_sec = *pMeshes[idx_sec];

		SZ3 n = Mesh_pri.n;
		DBL3 h = Mesh_pri.h;

		VEC_VC<DBL3>& Mpri = Mesh_pri.is_atomistic() ? reinterpret_cast<Atom_Mesh*>(&Mesh_pri)->M1 : reinterpret_cast<Mesh*>(&Mesh_pri)->M;
		VEC_VC<DBL3>& Msec = Mesh_sec.is_atomistic() ? reinterpret_cast<Atom_Mesh*>(&Mesh_sec)->M1 : reinterpret_cast<Mesh*>(&Mesh_sec)->M;

		//cellsize perpendicular to the interface, pointing towards it from the primary side (normal direction).
		//thus if primary is on the right hR is negative
		DBL3 hshift_primary = CMBNDcontacts[contact_idx].hshift_primary;

		INT3 box_sizes = CMBNDcontacts[contact_idx].cells_box.size();

		double energy_ = 0.0;

		//primary cells in this contact
#pragma omp parallel for reduction (+:energy_)
		for (int box_idx = 0; box_idx < box_sizes.dim(); box_idx++) {

			int i = (box_idx % box_sizes.x) + CMBNDcontacts[contact_idx].cells_box.s.i;
			int j = ((box_idx / box_sizes.x) % box_sizes.y) + CMBNDcontacts[contact_idx].cells_box.s.j;
			int k = (box_idx / (box_sizes.x * box_sizes.y)) + CMBNDcontacts[contact_idx].cells_box.s.k;

			int cell1_idx = i + j * n.x + k * n.x * n.y;

			if (Mpri.is_empty(cell1_idx) || Mpri.is_not_cmbnd(cell1_idx)) continue;

			//calculate second primary cell index
			int cell2_idx = (i + CMBNDcontacts[contact_idx].cell_shift.i) + (j + CMBNDcontacts[contact_idx].cell_shift.j) * n.x + (k + CMBNDcontacts[contact_idx].cell_shift.k) * n.x * n.y;

			//relative position of cell -1 in secondary mesh
			DBL3 relpos_m1 = Mpri.rect.s - Msec.rect.s + ((DBL3(i, j, k) + DBL3(0.5)) & h) + (CMBNDcontacts[contact_idx].hshift_primary + CMBNDcontacts[contact_idx].hshift_secondary) / 2;

			//stencil is used for weighted_average to obtain values in the secondary mesh : has size equal to primary cellsize area on interface with thickness set by secondary cellsize thickness
			DBL3 stencil = h - mod(CMBNDcontacts[contact_idx].hshift_primary) + mod(CMBNDcontacts[contact_idx].hshift_secondary);

			energy_ += calculate_coupling(cell1_idx, cell2_idx, relpos_m1, stencil, hshift_primary, Mesh_pri, Mesh_sec);
		}

		energy += energy_;
	}
}

