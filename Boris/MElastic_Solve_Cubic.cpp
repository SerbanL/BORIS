#include "stdafx.h"
#include "MElastic.h"

#ifdef MODULE_COMPILATION_MELASTIC

#include "Mesh_Ferromagnetic.h"
#include "MeshParamsControl.h"

#include "SuperMesh.h"

#include "MElastic_Boundaries.h"

//-----MElastic Field

double MElastic::Calculate_MElastic_Field_Cubic(void)
{
	double energy = 0;

	//disabled by setting magnetoelastic coefficient to zero (also disabled in non-magnetic meshes)
	if (melastic_field_disabled) return energy;

	//Energy density formula:

	//Let e1, e2, e3 be the orthogonal cubic axes, e.g. x, y, z in the simplest case

	//diagonal terms in stress tensor :
	//let Sd = (exx, eyy, ezz) be the diagonal terms.

	//off-diagonal terms in stress tensor (remember it is symmetric so we only have 3 of these):
	//let Sod = (eyz, exz, exy)

	//Then:

	//energy density from diagonal terms:
	//emel_d = B1 * [ (m.e1)^2*(Sd.e1) + (m.e2)^2*(Sd.e2) + (m.e3)^2*(Sd.e3) ]

	//energy density from off-diagonal terms (but remember this is zero for uniform strains):
	//emel_od = 2 * B2 * [ (m.e1)*(m.e2)*(Sod.e3) + (m.e1)*(m.e3)*(Sod.e2) + (m.e2)*(m.e3)*(Sod.e1) ]

	//Obtain Hmel as usual : Hmel = -1/mu0Ms * de/dm, where exchange stiffness-related terms are ignored here.

	if (pMesh->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

#pragma omp parallel for reduction(+:energy)
		for (int idx = 0; idx < pMesh->n.dim(); idx++) {

			if (pMesh->M.is_not_empty(idx)) {

				DBL2 Ms_AFM = pMesh->Ms_AFM;
				DBL3 mcanis_ea1 = pMesh->mcanis_ea1;
				DBL3 mcanis_ea2 = pMesh->mcanis_ea2;
				DBL3 mcanis_ea3 = pMesh->mcanis_ea3;
				DBL2 MEc = pMesh->MEc;
				pMesh->update_parameters_mcoarse(idx, pMesh->Ms_AFM, Ms_AFM, pMesh->MEc, MEc, pMesh->mcanis_ea1, mcanis_ea1, pMesh->mcanis_ea2, mcanis_ea2, pMesh->mcanis_ea3, mcanis_ea3);

				DBL3 position = pMesh->M.cellidx_to_position(idx);
				//xx, yy, zz
				DBL3 Sd = pMesh->strain_diag[position];
				//yz, xz, xy
				DBL3 Sod = pMesh->strain_odiag[position];

				//normalised magnetization
				//Magneto-elastic term here applicable for a cubic crystal. We use the mcanis_ea1 and mcanis_ea2 axes to fix the cubic lattice orientation, thus rotate the m, Sd and Sod vectors.

				DBL3 mA = DBL3(pMesh->M[idx] * mcanis_ea1, pMesh->M[idx] * mcanis_ea2, pMesh->M[idx] * mcanis_ea3) / Ms_AFM.i;
				DBL3 mB = DBL3(pMesh->M2[idx] * mcanis_ea1, pMesh->M2[idx] * mcanis_ea2, pMesh->M2[idx] * mcanis_ea3) / Ms_AFM.j;

				Sd = DBL3(Sd * mcanis_ea1, Sd * mcanis_ea2, Sd * mcanis_ea3);
				Sod = DBL3(Sod * mcanis_ea1, Sod * mcanis_ea2, Sod * mcanis_ea3);

				DBL3 Hmel_1_A = (-2.0 * MEc.i / (MU0 * Ms_AFM.i)) * DBL3(
					mA.x*Sd.x*mcanis_ea1.x + mA.y*Sd.y*mcanis_ea2.x + mA.z*Sd.z*mcanis_ea3.x,
					mA.x*Sd.x*mcanis_ea1.y + mA.y*Sd.y*mcanis_ea2.y + mA.z*Sd.z*mcanis_ea3.y,
					mA.x*Sd.x*mcanis_ea1.z + mA.y*Sd.y*mcanis_ea2.z + mA.z*Sd.z*mcanis_ea3.z);

				DBL3 Hmel_2_A = (-2.0 * MEc.j / (MU0 * Ms_AFM.i)) * DBL3(
					Sod.z * (mcanis_ea1.x*mA.y + mcanis_ea2.x*mA.x) + Sod.y * (mcanis_ea1.x*mA.z + mcanis_ea3.x*mA.x) + Sod.x * (mcanis_ea2.x*mA.z + mcanis_ea3.x*mA.y),
					Sod.z * (mcanis_ea1.y*mA.y + mcanis_ea2.y*mA.x) + Sod.y * (mcanis_ea1.y*mA.z + mcanis_ea3.y*mA.x) + Sod.x * (mcanis_ea2.y*mA.z + mcanis_ea3.y*mA.y),
					Sod.z * (mcanis_ea1.z*mA.y + mcanis_ea2.z*mA.x) + Sod.y * (mcanis_ea1.z*mA.z + mcanis_ea3.z*mA.x) + Sod.x * (mcanis_ea2.z*mA.z + mcanis_ea3.z*mA.y));

				DBL3 Hmel_1_B = (-2.0 * MEc.i / (MU0 * Ms_AFM.j)) * DBL3(
					mB.x*Sd.x*mcanis_ea1.x + mB.y*Sd.y*mcanis_ea2.x + mB.z*Sd.z*mcanis_ea3.x,
					mB.x*Sd.x*mcanis_ea1.y + mB.y*Sd.y*mcanis_ea2.y + mB.z*Sd.z*mcanis_ea3.y,
					mB.x*Sd.x*mcanis_ea1.z + mB.y*Sd.y*mcanis_ea2.z + mB.z*Sd.z*mcanis_ea3.z);

				DBL3 Hmel_2_B = (-2.0 * MEc.j / (MU0 * Ms_AFM.j)) * DBL3(
					Sod.z * (mcanis_ea1.x*mB.y + mcanis_ea2.x*mB.x) + Sod.y * (mcanis_ea1.x*mB.z + mcanis_ea3.x*mB.x) + Sod.x * (mcanis_ea2.x*mB.z + mcanis_ea3.x*mB.y),
					Sod.z * (mcanis_ea1.y*mB.y + mcanis_ea2.y*mB.x) + Sod.y * (mcanis_ea1.y*mB.z + mcanis_ea3.y*mB.x) + Sod.x * (mcanis_ea2.y*mB.z + mcanis_ea3.y*mB.y),
					Sod.z * (mcanis_ea1.z*mB.y + mcanis_ea2.z*mB.x) + Sod.y * (mcanis_ea1.z*mB.z + mcanis_ea3.z*mB.x) + Sod.x * (mcanis_ea2.z*mB.z + mcanis_ea3.z*mB.y));

				pMesh->Heff[idx] += Hmel_1_A + Hmel_2_A;
				pMesh->Heff2[idx] += Hmel_1_B + Hmel_2_B;

				energy += pMesh->M[idx] * (Hmel_1_A + Hmel_2_A) / 2 + pMesh->M2[idx] * (Hmel_1_B + Hmel_2_B) / 2;

				if (Module_Heff.linear_size()) Module_Heff[idx] = Hmel_1_A + Hmel_2_A;
				if (Module_Heff2.linear_size()) Module_Heff2[idx] = Hmel_1_B + Hmel_2_B;
				if (Module_energy.linear_size()) Module_energy[idx] = -MU0 * pMesh->M[idx] * (Hmel_1_A + Hmel_2_A) / 2;
				if (Module_energy2.linear_size()) Module_energy2[idx] = -MU0 * pMesh->M2[idx] * (Hmel_1_B + Hmel_2_B) / 2;
			}
		}
	}

	else if (pMesh->GetMeshType() == MESH_FERROMAGNETIC) {

#pragma omp parallel for reduction(+:energy)
		for (int idx = 0; idx < pMesh->n.dim(); idx++) {

			if (pMesh->M.is_not_empty(idx)) {

				double Ms = pMesh->Ms;
				DBL3 mcanis_ea1 = pMesh->mcanis_ea1;
				DBL3 mcanis_ea2 = pMesh->mcanis_ea2;
				DBL3 mcanis_ea3 = pMesh->mcanis_ea3;
				DBL2 MEc = pMesh->MEc;
				pMesh->update_parameters_mcoarse(idx, pMesh->Ms, Ms, pMesh->MEc, MEc, pMesh->mcanis_ea1, mcanis_ea1, pMesh->mcanis_ea2, mcanis_ea2, pMesh->mcanis_ea3, mcanis_ea3);

				DBL3 position = pMesh->M.cellidx_to_position(idx);
				//xx, yy, zz
				DBL3 Sd = pMesh->strain_diag[position];
				//yz, xz, xy
				DBL3 Sod = pMesh->strain_odiag[position];

				//normalised magnetization
				//Magneto-elastic term here applicable for a cubic crystal. We use the mcanis_ea1 and mcanis_ea2 axes to fix the cubic lattice orientation, thus rotate the m, Sd and Sod vectors.

				DBL3 m = DBL3(pMesh->M[idx] * mcanis_ea1, pMesh->M[idx] * mcanis_ea2, pMesh->M[idx] * mcanis_ea3) / Ms;
				Sd = DBL3(Sd * mcanis_ea1, Sd * mcanis_ea2, Sd * mcanis_ea3);
				Sod = DBL3(Sod * mcanis_ea1, Sod * mcanis_ea2, Sod * mcanis_ea3);

				DBL3 Hmel_1 = (-2.0 * MEc.i / (MU0 * Ms)) * DBL3(
					m.x*Sd.x*mcanis_ea1.x + m.y*Sd.y*mcanis_ea2.x + m.z*Sd.z*mcanis_ea3.x,
					m.x*Sd.x*mcanis_ea1.y + m.y*Sd.y*mcanis_ea2.y + m.z*Sd.z*mcanis_ea3.y,
					m.x*Sd.x*mcanis_ea1.z + m.y*Sd.y*mcanis_ea2.z + m.z*Sd.z*mcanis_ea3.z);

				DBL3 Hmel_2 = (-2.0 * MEc.j / (MU0 * Ms)) * DBL3(
					Sod.z * (mcanis_ea1.x*m.y + mcanis_ea2.x*m.x) + Sod.y * (mcanis_ea1.x*m.z + mcanis_ea3.x*m.x) + Sod.x * (mcanis_ea2.x*m.z + mcanis_ea3.x*m.y),
					Sod.z * (mcanis_ea1.y*m.y + mcanis_ea2.y*m.x) + Sod.y * (mcanis_ea1.y*m.z + mcanis_ea3.y*m.x) + Sod.x * (mcanis_ea2.y*m.z + mcanis_ea3.y*m.y),
					Sod.z * (mcanis_ea1.z*m.y + mcanis_ea2.z*m.x) + Sod.y * (mcanis_ea1.z*m.z + mcanis_ea3.z*m.x) + Sod.x * (mcanis_ea2.z*m.z + mcanis_ea3.z*m.y));

				pMesh->Heff[idx] += Hmel_1 + Hmel_2;

				energy += pMesh->M[idx] * (Hmel_1 + Hmel_2);

				if (Module_Heff.linear_size()) Module_Heff[idx] = Hmel_1 + Hmel_2;
				if (Module_energy.linear_size()) Module_energy[idx] = -MU0 * pMesh->M[idx] * (Hmel_1 + Hmel_2) / 2;
			}
		}
	}

	if (pMesh->M.get_nonempty_cells()) energy *= -MU0 / (2 * pMesh->M.get_nonempty_cells());
	else energy = 0;

	this->energy = energy;

	return this->energy;
}

//----------------------------------------------- Computational Helpers

//-----Stress

void MElastic::Iterate_Elastic_Solver_Stress_Cubic(double dT)
{
	DBL3 h_m = pMesh->u_disp.h;
	SZ3 n_m = pMesh->u_disp.n;

	double time = pSMesh->GetStageTime();

	//1b. Update stress

	for (int k = 0; k < n_m.k + 1; k++) {
#pragma omp parallel for
		for (int j = 0; j < n_m.j + 1; j++) {
			for (int i = 0; i < n_m.i + 1; i++) {

				//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
				INT3 ijk_u = INT3(i < n_m.i ? i : n_m.i - 1, j < n_m.j ? j : n_m.j - 1, k < n_m.k ? k : n_m.k - 1);
				int idx_u = ijk_u.i + ijk_u.j * n_m.x + ijk_u.k * n_m.x * n_m.y;

				DBL3 cC = pMesh->cC;
				pMesh->update_parameters_scoarse(idx_u, pMesh->cC, cC);
				double cr = cC.j / cC.i;

				INT3 ijk = INT3(i, j, k);

				//needed for magnetostriction (time derivatives of stress due to magnetostriction)
				DBL3 dsdd_dt_ms = DBL3();
				double dsxy_dt_ms = 0.0, dsxz_dt_ms = 0.0, dsyz_dt_ms = 0.0;
				if (magnetostriction_enabled) {

					int idx_M = pMesh->M.position_to_cellidx(pMesh->u_disp.cellidx_to_position(idx_u));

					if (pMesh->M.is_not_empty(idx_M)) {

						if (pMesh->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

							DBL2 Ms_AFM = pMesh->Ms_AFM;
							DBL3 mcanis_ea1 = pMesh->mcanis_ea1;
							DBL3 mcanis_ea2 = pMesh->mcanis_ea2;
							DBL3 mcanis_ea3 = pMesh->mcanis_ea3;
							DBL2 mMEc = pMesh->mMEc;
							pMesh->update_parameters_mcoarse(idx_M, pMesh->Ms_AFM, Ms_AFM, pMesh->mMEc, mMEc, pMesh->mcanis_ea1, mcanis_ea1, pMesh->mcanis_ea2, mcanis_ea2, pMesh->mcanis_ea3, mcanis_ea3);

							DBL3 mA = DBL3(pMesh->M[idx_M] * mcanis_ea1, pMesh->M[idx_M] * mcanis_ea2, pMesh->M[idx_M] * mcanis_ea3) / Ms_AFM.i;
							DBL3 mB = DBL3(pMesh->M2[idx_M] * mcanis_ea1, pMesh->M2[idx_M] * mcanis_ea2, pMesh->M2[idx_M] * mcanis_ea3) / Ms_AFM.j;
							DBL3 dM_dtA = pMesh->dMdt(idx_M);
							DBL3 dm_dtA = DBL3(dM_dtA * mcanis_ea1, dM_dtA * mcanis_ea2, dM_dtA * mcanis_ea3) / Ms_AFM.i;
							DBL3 dM_dtB = pMesh->dMdt2(idx_M);
							DBL3 dm_dtB = DBL3(dM_dtB * mcanis_ea1, dM_dtB * mcanis_ea2, dM_dtB * mcanis_ea3) / Ms_AFM.j;

							dsdd_dt_ms = mMEc.i * DBL3(
								(mA.x*dm_dtA.x + mB.x*dm_dtB.x)*mcanis_ea1.x + (mA.y*dm_dtA.y + mB.y*dm_dtB.y)*mcanis_ea2.x + (mA.z*dm_dtA.z + mB.z*dm_dtB.z)*mcanis_ea3.x,
								(mA.x*dm_dtA.x + mB.x*dm_dtB.x)*mcanis_ea1.y + (mA.y*dm_dtA.y + mB.y*dm_dtB.y)*mcanis_ea2.y + (mA.z*dm_dtA.z + mB.z*dm_dtB.z)*mcanis_ea3.y,
								(mA.x*dm_dtA.x + mB.x*dm_dtB.x)*mcanis_ea1.z + (mA.y*dm_dtA.y + mB.y*dm_dtB.y)*mcanis_ea2.z + (mA.z*dm_dtA.z + mB.z*dm_dtB.z)*mcanis_ea3.z);

							dsxy_dt_ms = (mMEc.j / 2) * ((mA.x*dm_dtA.y + mA.y*dm_dtA.x + mB.x*dm_dtB.y + mB.y*dm_dtB.x)*mcanis_ea3.z + (mA.x*dm_dtA.z + mA.z*dm_dtA.x + mB.x*dm_dtB.z + mB.z*dm_dtB.x)*mcanis_ea2.z + (mA.y*dm_dtA.z + mA.z*dm_dtA.y + mB.y*dm_dtB.z + mB.z*dm_dtB.y)*mcanis_ea1.z);
							dsxz_dt_ms = (mMEc.j / 2) * ((mA.x*dm_dtA.y + mA.y*dm_dtA.x + mB.x*dm_dtB.y + mB.y*dm_dtB.x)*mcanis_ea3.y + (mA.x*dm_dtA.z + mA.z*dm_dtA.x + mB.x*dm_dtB.z + mB.z*dm_dtB.x)*mcanis_ea2.y + (mA.y*dm_dtA.z + mA.z*dm_dtA.y + mB.y*dm_dtB.z + mB.z*dm_dtB.y)*mcanis_ea1.y);
							dsyz_dt_ms = (mMEc.j / 2) * ((mA.x*dm_dtA.y + mA.y*dm_dtA.x + mB.x*dm_dtB.y + mB.y*dm_dtB.x)*mcanis_ea3.x + (mA.x*dm_dtA.z + mA.z*dm_dtA.x + mB.x*dm_dtB.z + mB.z*dm_dtB.x)*mcanis_ea2.x + (mA.y*dm_dtA.z + mA.z*dm_dtA.y + mB.y*dm_dtB.z + mB.z*dm_dtB.y)*mcanis_ea1.x);
						}
						else if (pMesh->GetMeshType() == MESH_FERROMAGNETIC) {

							double Ms = pMesh->Ms;
							DBL3 mcanis_ea1 = pMesh->mcanis_ea1;
							DBL3 mcanis_ea2 = pMesh->mcanis_ea2;
							DBL3 mcanis_ea3 = pMesh->mcanis_ea3;
							DBL2 mMEc = pMesh->mMEc;
							pMesh->update_parameters_mcoarse(idx_M, pMesh->Ms, Ms, pMesh->mMEc, mMEc, pMesh->mcanis_ea1, mcanis_ea1, pMesh->mcanis_ea2, mcanis_ea2, pMesh->mcanis_ea3, mcanis_ea3);

							DBL3 m = DBL3(pMesh->M[idx_M] * mcanis_ea1, pMesh->M[idx_M] * mcanis_ea2, pMesh->M[idx_M] * mcanis_ea3) / Ms;
							DBL3 dM_dt = pMesh->dMdt(idx_M);
							DBL3 dm_dt = DBL3(dM_dt * mcanis_ea1, dM_dt * mcanis_ea2, dM_dt * mcanis_ea3) / Ms;

							dsdd_dt_ms = 2 * mMEc.i * DBL3(
								m.x*dm_dt.x*mcanis_ea1.x + m.y*dm_dt.y*mcanis_ea2.x + m.z*dm_dt.z*mcanis_ea3.x,
								m.x*dm_dt.x*mcanis_ea1.y + m.y*dm_dt.y*mcanis_ea2.y + m.z*dm_dt.z*mcanis_ea3.y,
								m.x*dm_dt.x*mcanis_ea1.z + m.y*dm_dt.y*mcanis_ea2.z + m.z*dm_dt.z*mcanis_ea3.z);

							dsxy_dt_ms = mMEc.j * ((m.x*dm_dt.y + m.y*dm_dt.x)*mcanis_ea3.z + (m.x*dm_dt.z + m.z*dm_dt.x)*mcanis_ea2.z + (m.y*dm_dt.z + m.z*dm_dt.y)*mcanis_ea1.z);
							dsxz_dt_ms = mMEc.j * ((m.x*dm_dt.y + m.y*dm_dt.x)*mcanis_ea3.y + (m.x*dm_dt.z + m.z*dm_dt.x)*mcanis_ea2.y + (m.y*dm_dt.z + m.z*dm_dt.y)*mcanis_ea1.y);
							dsyz_dt_ms = mMEc.j * ((m.x*dm_dt.y + m.y*dm_dt.x)*mcanis_ea3.x + (m.x*dm_dt.z + m.z*dm_dt.x)*mcanis_ea2.x + (m.y*dm_dt.z + m.z*dm_dt.y)*mcanis_ea1.x);
						}
					}
				}

				//needed for thermoelasticity (includes time derivative of temperature)
				double dsdd_dt_te = 0.0;
				if (thermoelasticity_enabled) {

					int idx_T = pMesh->Temp.position_to_cellidx(pMesh->u_disp.cellidx_to_position(idx_u));

					if (pMesh->Temp.is_not_empty(idx_T)) {

						double thalpha = pMesh->thalpha;
						pMesh->update_parameters_scoarse(idx_u, pMesh->thalpha, thalpha);

						double Temperature = 0.0;
						//for 2TM we need to use the lattice temperature
						if (pMesh->Temp_l.linear_size()) Temperature = pMesh->Temp_l[idx_T];
						else Temperature = pMesh->Temp[idx_T];

						dsdd_dt_te = (cC.i + 2 * cC.j) * thalpha * (Temperature - Temp_previous[idx_T]) / pSMEl->magnetic_dT;
					}
				}

				double adT = dsdd_dt_te / cC.i;
				DBL3 dms = dsdd_dt_ms / cC.i;

				//external forces on different faces (keep track separately in case an edge cell is excited simultaneously by 2 or more external forces
				//also only need the component normal to the surface, at the vertex
				double Fext_xface = 0.0, Fext_yface = 0.0, Fext_zface = 0.0;
				//time derivatives of forces on the different faces, divided by c11
				double dFx = 0.0, dFy = 0.0, dFz = 0.0;

				//is there an external force? If so, get it, otherwise it will be zero
				if (
					((i == 0 || i == n_m.i) && pMesh->strain_diag.is_dirichlet_x(idx_u)) ||
					((j == 0 || j == n_m.j) && pMesh->strain_diag.is_dirichlet_y(idx_u)) ||
					((k == 0 || k == n_m.k) && pMesh->strain_diag.is_dirichlet_z(idx_u))) {

					//search through all available surfaces to get external force
					for (int sidx = 0; sidx < external_stress_surfaces.size(); sidx++) {

						int orientation = external_stress_surfaces[sidx].contains(ijk_u);
						if (orientation) {

							switch (abs(orientation)) {

								//x face
							case 1:
								Fext_xface = external_stress_surfaces[sidx].get_ext_force_vertices(ijk, time);
								break;

								//y face
							case 2:
								Fext_yface = external_stress_surfaces[sidx].get_ext_force_vertices(ijk, time);
								break;

								//z face
							case 3:
								Fext_zface = external_stress_surfaces[sidx].get_ext_force_vertices(ijk, time);
								break;
							};
						}
					}
				}

				//update sxx, syy, szz
				int niend = (i < n_m.i);
				int njend = (j < n_m.j);
				int nkend = (k < n_m.k);

				//check if required edges are present
				bool xedge_u =
					i < n_m.i &&
					(pMesh->u_disp.is_not_empty(idx_u) ||
					(k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y)) ||
						(j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x)) ||
						(j > 0 && k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x)));

				bool xedge_l =
					i > 0 &&
					(pMesh->u_disp.is_not_empty(idx_u - niend) ||
					(k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend)) ||
						(j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x - niend)) ||
						(j > 0 && k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x - niend)));

				bool yedge_u =
					j < n_m.j &&
					(pMesh->u_disp.is_not_empty(idx_u) ||
					(k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y)) ||
						(i > 0 && pMesh->u_disp.is_not_empty(idx_u - niend)) ||
						(i > 0 && k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend)));

				bool yedge_l =
					j > 0 &&
					(pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x) ||
					(k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x)) ||
						(i > 0 && pMesh->u_disp.is_not_empty(idx_u - niend - njend * n_m.x)) ||
						(i > 0 && k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend - njend * n_m.x)));

				bool zedge_u =
					k < n_m.k &&
					(pMesh->u_disp.is_not_empty(idx_u) ||
					(j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x)) ||
						(i > 0 && pMesh->u_disp.is_not_empty(idx_u - niend)) ||
						(i > 0 && j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x - niend)));

				bool zedge_l =
					k > 0 &&
					(pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y) ||
					(j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x - nkend * n_m.x*n_m.y)) ||
						(i > 0 && pMesh->u_disp.is_not_empty(idx_u - niend - nkend * n_m.x*n_m.y)) ||
						(i > 0 && j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x - niend - nkend * n_m.x*n_m.y)));

				//check for fixed faces at ends
				bool xfixed_l = (i == 0 && pMesh->u_disp.is_dirichlet_px(idx_u));
				bool xfixed_u = (i == n_m.i && pMesh->u_disp.is_dirichlet_nx(idx_u));

				bool yfixed_l = (j == 0 && pMesh->u_disp.is_dirichlet_py(idx_u));
				bool yfixed_u = (j == n_m.j && pMesh->u_disp.is_dirichlet_ny(idx_u));

				bool zfixed_l = (k == 0 && pMesh->u_disp.is_dirichlet_pz(idx_u));
				bool zfixed_u = (k == n_m.k && pMesh->u_disp.is_dirichlet_nz(idx_u));

				double dvx_dx = 0.0;

				//interior
				if (xedge_u && xedge_l) dvx_dx = (vx[ijk] - vx[INT3(ijk.i - 1, ijk.j, ijk.k)]) / h_m.x;
				//fixed face : Dirichlet value of zero for velocity derivative
				else if (xedge_l && xfixed_u) {

					dvx_dx = -vx[INT3(ijk.i - 1, ijk.j, ijk.k)] / (h_m.x / 2);
				}
				else if (xedge_u && xfixed_l) {

					dvx_dx = vx[ijk] / (h_m.x / 2);
				}
				//free face
				else {

					//both side derivatives
					if (yedge_l && yedge_u && zedge_l && zedge_u) {

						dvx_dx = -cr * ((vy[ijk] - vy[INT3(ijk.i, ijk.j - 1, ijk.k)]) / h_m.y + (vz[ijk] - vz[INT3(ijk.i, ijk.j, ijk.k - 1)]) / h_m.z) + adT - dms.x + dFx;
					}
					//only z derivative
					else if (zedge_l && zedge_u) {

						//dvx = (dFx - cr*dFy - dmsx + cr*dmsy) / (1 - cr^2) + (adT - cr*dvz) / (1 + cr)
						dvx_dx = (adT - cr * (vz[ijk] - vz[INT3(ijk.i, ijk.j, ijk.k - 1)]) / h_m.z) / (1 + cr) + (-dms.x + cr * dms.y + dFx - cr * dFy) / (1 - cr * cr);
					}
					//only y derivative
					else if (yedge_l && yedge_u) {

						//dvx = (dFx - cr*dFz - dmsx + cr*dmsz) / (1 - cr^2) + (adT - cr*dvy) / (1 + cr)
						dvx_dx = (adT - cr * (vy[ijk] - vy[INT3(ijk.i, ijk.j - 1, ijk.k)]) / h_m.y) / (1 + cr) + (-dms.x + cr * dms.z + dFx - cr * dFz) / (1 - cr * cr);
					}
					//no side derivatives : corner point. In this case all diagonal stress components set from external conditions, so derivatives not needed (set zero)
					else dvx_dx = 0.0;
				}

				double dvy_dy = 0.0;

				//interior
				if (yedge_u && yedge_l) dvy_dy = (vy[ijk] - vy[INT3(ijk.i, ijk.j - 1, ijk.k)]) / h_m.y;
				//fixed face : Dirichlet value of zero for velocity derivative
				else if (yedge_l && yfixed_u) {

					dvy_dy = -vy[INT3(ijk.i, ijk.j - 1, ijk.k)] / (h_m.y / 2);
				}
				else if (yedge_u && yfixed_l) {

					dvy_dy = vy[ijk] / (h_m.y / 2);
				}
				//free face
				else {

					//both side derivatives
					if (xedge_l && xedge_u && zedge_l && zedge_u) {

						dvy_dy = -cr * ((vx[ijk] - vx[INT3(ijk.i - 1, ijk.j, ijk.k)]) / h_m.x + (vz[ijk] - vz[INT3(ijk.i, ijk.j, ijk.k - 1)]) / h_m.z) + adT - dms.y + dFy;
					}
					//only z derivative
					else if (zedge_l && zedge_u) {

						//dvy = (dFy - cr*dFx - dmsy + cr*dmsx) / (1 - cr^2) + (adT - cr*dvz) / (1 + cr)
						dvy_dy = (adT - cr * (vz[ijk] - vz[INT3(ijk.i, ijk.j, ijk.k - 1)]) / h_m.z) / (1 + cr) + (-dms.y + cr * dms.x + dFy - cr * dFx) / (1 - cr * cr);
					}
					//only x derivative
					else if (xedge_l && xedge_u) {

						//dvy = (dFy - cr*dFz - dmsy + cr*dmsz) / (1 - cr^2) + (adT - cr*dvx) / (1 + cr)
						dvy_dy = (adT - cr * (vx[ijk] - vx[INT3(ijk.i - 1, ijk.j, ijk.k)]) / h_m.x) / (1 + cr) + (-dms.y + cr * dms.z + dFy - cr * dFz) / (1 - cr * cr);
					}
					//no side derivatives : corner point. In this case all diagonal stress components set from external conditions, so derivatives not needed (set zero)
					else dvy_dy = 0.0;
				}

				double dvz_dz = 0.0;

				//interior
				if (zedge_u && zedge_l) dvz_dz = (vz[ijk] - vz[INT3(ijk.i, ijk.j, ijk.k - 1)]) / h_m.z;
				//fixed face : Dirichlet value of zero for velocity derivative
				else if (zedge_l && zfixed_u) {

					dvz_dz = -vz[INT3(ijk.i, ijk.j, ijk.k - 1)] / (h_m.z / 2);
				}
				//fixed face : Dirichlet value of zero for velocity derivative
				else if (zedge_u && zfixed_l) {

					dvz_dz = vz[ijk] / (h_m.z / 2);
				}
				//free face
				else {

					//both side derivatives
					if (xedge_l && xedge_u && yedge_l && yedge_u) {

						dvz_dz = -cr * ((vx[ijk] - vx[INT3(ijk.i - 1, ijk.j, ijk.k)]) / h_m.x + (vy[ijk] - vy[INT3(ijk.i, ijk.j - 1, ijk.k)]) / h_m.y) + adT - dms.z + dFz;
					}
					//only y derivative
					else if (yedge_l && yedge_u) {

						//dvz = (dFz - cr*dFx - dmsz + cr*dmsx) / (1 - cr^2) + (adT - cr*dvy) / (1 + cr)
						dvz_dz = (adT - cr * (vy[ijk] - vy[INT3(ijk.i, ijk.j - 1, ijk.k)]) / h_m.y) / (1 + cr) + (-dms.z + cr * dms.x + dFz - cr * dFx) / (1 - cr * cr);
					}
					//only x derivative
					else if (xedge_l && xedge_u) {

						//dvz = (dFz - cr*dFy - dmsz + cr*dmsy) / (1 - cr^2) + (adT - cr*dvx) / (1 + cr)
						dvz_dz = (adT - cr * (vx[ijk] - vx[INT3(ijk.i - 1, ijk.j, ijk.k)]) / h_m.x) / (1 + cr) + (-dms.z + cr * dms.y + dFz - cr * dFy) / (1 - cr * cr);
					}
					//no side derivatives : corner point. In this case all diagonal stress components set from external conditions, so derivatives not needed (set zero)
					else dvz_dz = 0.0;
				}

				//update sdd if not empty
				if ((xedge_u || xedge_l) && (yedge_u || yedge_l) && (zedge_u || zedge_l)) {

					if ((!xedge_u && !xfixed_u) || (!xedge_l && !xfixed_l)) sdd[ijk].x = Fext_xface;
					else sdd[ijk].x += dT * (cC.i * dvx_dx + cC.j * (dvy_dy + dvz_dz) + dsdd_dt_ms.x - dsdd_dt_te);

					if ((!yedge_u && !yfixed_u) || (!yedge_l && !yfixed_l)) sdd[ijk].y = Fext_yface;
					else sdd[ijk].y += dT * (cC.i * dvy_dy + cC.j * (dvx_dx + dvz_dz) + dsdd_dt_ms.y - dsdd_dt_te);

					if ((!zedge_u && !zfixed_u) || (!zedge_l && !zfixed_l)) sdd[ijk].z = Fext_zface;
					else sdd[ijk].z += dT * (cC.i * dvz_dz + cC.j * (dvx_dx + dvy_dy) + dsdd_dt_ms.z - dsdd_dt_te);
				}
				else sdd[ijk] = DBL3();

				//update sxy
				if (i < n_m.i && j < n_m.j) {

					bool zface =
						(pMesh->u_disp.is_not_empty(idx_u) ||
						(k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y)));

					if (zface) {

						double dvx_dy = (vx[INT3(i, j + 1, k)] - vx[ijk]) / h_m.y;
						double dvy_dx = (vy[INT3(i + 1, j, k)] - vy[ijk]) / h_m.x;

						sxy[ijk] += dT * (cC.k * (dvx_dy + dvy_dx) + dsxy_dt_ms);
					}
					else sxy[ijk] = 0.0;
				}

				//update sxz
				if (i < n_m.i && k < n_m.k) {

					bool yface =
						(pMesh->u_disp.is_not_empty(idx_u) ||
						(j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x)));

					if (yface) {

						double dvx_dz = (vx[INT3(i, j, k + 1)] - vx[ijk]) / h_m.z;
						double dvz_dx = (vz[INT3(i + 1, j, k)] - vz[ijk]) / h_m.x;

						sxz[ijk] += dT * (cC.k * (dvx_dz + dvz_dx) + dsxz_dt_ms);
					}
					else sxz[ijk] = 0.0;
				}

				//update syz
				if (j < n_m.j && k < n_m.k) {

					bool xface =
						(pMesh->u_disp.is_not_empty(idx_u) ||
						(i > 0 && pMesh->u_disp.is_not_empty(idx_u - niend)));

					if (xface) {

						double dvy_dz = (vy[INT3(i, j, k + 1)] - vy[ijk]) / h_m.z;
						double dvz_dy = (vz[INT3(i, j + 1, k)] - vz[ijk]) / h_m.y;

						syz[ijk] += dT * (cC.k * (dvy_dz + dvz_dy) + dsyz_dt_ms);
					}
					else syz[ijk] = 0.0;
				}

				//update mechanical displacement using velocity (remember u is cell-centred)
				if (i < n_m.i && j < n_m.j && k < n_m.k) {

					if (pMesh->u_disp.is_not_empty(idx_u)) {

						//find velocity values cell-centred
						double vx_cc = (vx[ijk] + vx[ijk + INT3(0, 1, 0)] + vx[ijk + INT3(0, 0, 1)] + vx[ijk + INT3(0, 1, 1)]) / 4;
						double vy_cc = (vy[ijk] + vy[ijk + INT3(1, 0, 0)] + vy[ijk + INT3(0, 0, 1)] + vy[ijk + INT3(1, 0, 1)]) / 4;
						double vz_cc = (vz[ijk] + vz[ijk + INT3(1, 0, 0)] + vz[ijk + INT3(0, 1, 0)] + vz[ijk + INT3(1, 1, 0)]) / 4;

						pMesh->u_disp[idx_u] += dT * DBL3(vx_cc, vy_cc, vz_cc);
					}
					else pMesh->u_disp[idx_u] = DBL3();
				}
			}
		}
	}
}

//-----Initial Stress

//if thermoelasticity or magnetostriction is enabled, then initial stress must be set correctly
void MElastic::Set_Initial_Stress_Cubic(void)
{
	if (!magnetostriction_enabled && !thermoelasticity_enabled) {

		sdd.set(DBL3());
		sxy.set(0.0); sxz.set(0.0); syz.set(0.0);
	}
	else {

		//reset for dT / dt computation
		if (thermoelasticity_enabled) {

			if (malloc_vector(Temp_previous, pMesh->n_t.dim())) Save_Current_Temperature();

			//refresh ambient temperature here
			T_ambient = pMesh->CallModuleMethod(&HeatBase::GetAmbientTemperature);
		}

		//reset for dm / dt computation
		if (magnetostriction_enabled) pMesh->SaveMagnetization();

		DBL3 h_m = pMesh->u_disp.h;
		SZ3 n_m = pMesh->u_disp.n;

		for (int k = 0; k < n_m.k + 1; k++) {
#pragma omp parallel for
			for (int j = 0; j < n_m.j + 1; j++) {
				for (int i = 0; i < n_m.i + 1; i++) {

					//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
					INT3 ijk_u = INT3(i < n_m.i ? i : n_m.i - 1, j < n_m.j ? j : n_m.j - 1, k < n_m.k ? k : n_m.k - 1);
					int idx_u = ijk_u.i + ijk_u.j * n_m.x + ijk_u.k * n_m.x * n_m.y;

					INT3 ijk = INT3(i, j, k);

					int niend = (i < n_m.i);
					int njend = (j < n_m.j);
					int nkend = (k < n_m.k);

					//check if required edges are present
					bool xedge_u =
						i < n_m.i &&
						(pMesh->u_disp.is_not_empty(idx_u) ||
						(k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y)) ||
							(j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x)) ||
							(j > 0 && k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x)));

					bool xedge_l =
						i > 0 &&
						(pMesh->u_disp.is_not_empty(idx_u - niend) ||
						(k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend)) ||
							(j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x - niend)) ||
							(j > 0 && k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x - niend)));

					bool yedge_u =
						j < n_m.j &&
						(pMesh->u_disp.is_not_empty(idx_u) ||
						(k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y)) ||
							(i > 0 && pMesh->u_disp.is_not_empty(idx_u - niend)) ||
							(i > 0 && k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend)));

					bool yedge_l =
						j > 0 &&
						(pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x) ||
						(k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x)) ||
							(i > 0 && pMesh->u_disp.is_not_empty(idx_u - niend - njend * n_m.x)) ||
							(i > 0 && k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend - njend * n_m.x)));

					bool zedge_u =
						k < n_m.k &&
						(pMesh->u_disp.is_not_empty(idx_u) ||
						(j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x)) ||
							(i > 0 && pMesh->u_disp.is_not_empty(idx_u - niend)) ||
							(i > 0 && j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x - niend)));

					bool zedge_l =
						k > 0 &&
						(pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y) ||
						(j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x - nkend * n_m.x*n_m.y)) ||
							(i > 0 && pMesh->u_disp.is_not_empty(idx_u - niend - nkend * n_m.x*n_m.y)) ||
							(i > 0 && j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x - niend - nkend * n_m.x*n_m.y)));

					DBL3 Stress_MS_dd = DBL3();
					double Stress_MS_xy = 0.0, Stress_MS_xz = 0.0, Stress_MS_yz = 0.0;
					if (magnetostriction_enabled) {

						int idx_M = pMesh->M.position_to_cellidx(pMesh->u_disp.cellidx_to_position(idx_u));

						if (pMesh->M.is_not_empty(idx_M)) {

							if (pMesh->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

								DBL2 Ms_AFM = pMesh->Ms_AFM;
								DBL3 mcanis_ea1 = pMesh->mcanis_ea1;
								DBL3 mcanis_ea2 = pMesh->mcanis_ea2;
								DBL3 mcanis_ea3 = pMesh->mcanis_ea3;
								DBL2 mMEc = pMesh->mMEc;
								pMesh->update_parameters_mcoarse(idx_M, pMesh->Ms_AFM, Ms_AFM, pMesh->mMEc, mMEc, pMesh->mcanis_ea1, mcanis_ea1, pMesh->mcanis_ea2, mcanis_ea2, pMesh->mcanis_ea3, mcanis_ea3);

								DBL3 mA = DBL3(pMesh->M[idx_M] * mcanis_ea1, pMesh->M[idx_M] * mcanis_ea2, pMesh->M[idx_M] * mcanis_ea3) / Ms_AFM.i;
								DBL3 mB = DBL3(pMesh->M2[idx_M] * mcanis_ea1, pMesh->M2[idx_M] * mcanis_ea2, pMesh->M2[idx_M] * mcanis_ea3) / Ms_AFM.j;

								Stress_MS_dd = (mMEc.i / 2) * DBL3(
									(mA.x*mA.x + mB.x*mB.x)*mcanis_ea1.x + (mA.y*mA.y + mB.y*mB.y)*mcanis_ea2.x + (mA.z*mA.z + mB.z*mB.z)*mcanis_ea3.x,
									(mA.x*mA.x + mB.x*mB.x)*mcanis_ea1.y + (mA.y*mA.y + mB.y*mB.y)*mcanis_ea2.y + (mA.z*mA.z + mB.z*mB.z)*mcanis_ea3.y,
									(mA.x*mA.x + mB.x*mB.x)*mcanis_ea1.z + (mA.y*mA.y + mB.y*mB.y)*mcanis_ea2.z + (mA.z*mA.z + mB.z*mB.z)*mcanis_ea3.z);

								Stress_MS_xy = (mMEc.j / 2) * ((mA.x*mA.y + mB.x*mB.y)*mcanis_ea3.z + (mA.x*mA.z + mB.x*mB.z)*mcanis_ea2.z + (mA.y*mA.z + mB.y*mB.z)*mcanis_ea1.z);
								Stress_MS_xz = (mMEc.j / 2) * ((mA.x*mA.y + mB.x*mB.y)*mcanis_ea3.y + (mA.x*mA.z + mB.x*mB.z)*mcanis_ea2.y + (mA.y*mA.z + mB.y*mB.z)*mcanis_ea1.y);
								Stress_MS_yz = (mMEc.j / 2) * ((mA.x*mA.y + mB.x*mB.y)*mcanis_ea3.x + (mA.x*mA.z + mB.x*mB.z)*mcanis_ea2.x + (mA.y*mA.z + mB.y*mB.z)*mcanis_ea1.x);
							}
							else if (pMesh->GetMeshType() == MESH_FERROMAGNETIC) {

								double Ms = pMesh->Ms;
								DBL3 mcanis_ea1 = pMesh->mcanis_ea1;
								DBL3 mcanis_ea2 = pMesh->mcanis_ea2;
								DBL3 mcanis_ea3 = pMesh->mcanis_ea3;
								DBL2 mMEc = pMesh->mMEc;
								pMesh->update_parameters_mcoarse(idx_M, pMesh->Ms, Ms, pMesh->mMEc, mMEc, pMesh->mcanis_ea1, mcanis_ea1, pMesh->mcanis_ea2, mcanis_ea2, pMesh->mcanis_ea3, mcanis_ea3);

								DBL3 m = DBL3(pMesh->M[idx_M] * mcanis_ea1, pMesh->M[idx_M] * mcanis_ea2, pMesh->M[idx_M] * mcanis_ea3) / Ms;

								Stress_MS_dd = mMEc.i * DBL3(
									m.x*m.x*mcanis_ea1.x + m.y*m.y*mcanis_ea2.x + m.z*m.z*mcanis_ea3.x,
									m.x*m.x*mcanis_ea1.y + m.y*m.y*mcanis_ea2.y + m.z*m.z*mcanis_ea3.y,
									m.x*m.x*mcanis_ea1.z + m.y*m.y*mcanis_ea2.z + m.z*m.z*mcanis_ea3.z);

								Stress_MS_xy = mMEc.j * (m.x*m.y*mcanis_ea3.z + m.x*m.z*mcanis_ea2.z + m.y*m.z*mcanis_ea1.z);
								Stress_MS_xz = mMEc.j * (m.x*m.y*mcanis_ea3.y + m.x*m.z*mcanis_ea2.y + m.y*m.z*mcanis_ea1.y);
								Stress_MS_yz = mMEc.j * (m.x*m.y*mcanis_ea3.x + m.x*m.z*mcanis_ea2.x + m.y*m.z*mcanis_ea1.x);
							}
						}
					}

					double Stress_Temp = 0.0;
					if (thermoelasticity_enabled) {

						int idx_T = pMesh->Temp.position_to_cellidx(pMesh->u_disp.cellidx_to_position(idx_u));

						if (pMesh->Temp.is_not_empty(idx_T)) {

							double thalpha = pMesh->thalpha;
							DBL3 cC = pMesh->cC;
							pMesh->update_parameters_scoarse(idx_u, pMesh->cC, cC, pMesh->thalpha, thalpha);

							double Temperature = 0.0;
							//for 2TM we need to use the lattice temperature
							if (pMesh->Temp_l.linear_size()) Temperature = pMesh->Temp_l[idx_T];
							else Temperature = pMesh->Temp[idx_T];

							Stress_Temp = (cC.i + 2 * cC.j) * thalpha * (Temperature - T_ambient);
						}
					}

					//update sdd
					if ((xedge_u || xedge_l) && (yedge_u || yedge_l) && (zedge_u || zedge_l)) {

						sdd[ijk].x = -Stress_Temp + Stress_MS_dd.x;
						sdd[ijk].y = -Stress_Temp + Stress_MS_dd.y;
						sdd[ijk].z = -Stress_Temp + Stress_MS_dd.z;
					}
					else sdd[ijk] = DBL3();

					//update sxy
					if (i < n_m.i && j < n_m.j) {

						bool zface =
							(pMesh->u_disp.is_not_empty(idx_u) ||
							(k > 0 && pMesh->u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y)));

						if (zface) {

							sxy[ijk] = Stress_MS_xy;
						}
						else sxy[ijk] = 0.0;
					}

					//update sxz
					if (i < n_m.i && k < n_m.k) {

						bool yface =
							(pMesh->u_disp.is_not_empty(idx_u) ||
							(j > 0 && pMesh->u_disp.is_not_empty(idx_u - njend * n_m.x)));

						if (yface) {

							sxz[ijk] = Stress_MS_xz;
						}
						else sxz[ijk] = 0.0;
					}

					//update syz
					if (j < n_m.j && k < n_m.k) {

						bool xface =
							(pMesh->u_disp.is_not_empty(idx_u) ||
							(i > 0 && pMesh->u_disp.is_not_empty(idx_u - niend)));

						if (xface) {

							syz[ijk] = Stress_MS_yz;
						}
						else syz[ijk] = 0.0;
					}
				}
			}
		}
	}
}

#endif

