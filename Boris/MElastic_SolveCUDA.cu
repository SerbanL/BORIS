#include "MElasticCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MELASTIC

#include "BorisCUDALib.cuh"

#include "ManagedDiffEqFMCUDA.h"
#include "MeshParamsControlCUDA.h"

#include "MElastic_BoundariesCUDA.h"

//----------------------- Iterate_Elastic_Solver KERNELS

__global__ void Iterate_Elastic_Solver_Velocity1_Kernel(
	ManagedMeshCUDA& cuMesh,
	MElastic_BoundaryCUDA* external_stress_surfaces, size_t num_surfaces,
	cuVEC<cuBReal>& vx, cuVEC<cuBReal>& vy, cuVEC<cuBReal>& vz,
	cuVEC<cuReal3>& sdd,
	cuVEC<cuBReal>& sxy, cuVEC<cuBReal>& sxz, cuVEC<cuBReal>& syz,
	cuBReal time, cuBReal dT)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;

	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	//kernel launch with size (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1) 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = idx % (n_m.i + 1);
	int j = (idx / (n_m.i + 1)) % (n_m.j + 1);
	int k = idx / ((n_m.i + 1)*(n_m.j + 1));

	if (idx < (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1)) {

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < n_m.i ? i : n_m.i - 1, j < n_m.j ? j : n_m.j - 1, k < n_m.k ? k : n_m.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * n_m.x + ijk_u.k * n_m.x * n_m.y;

		cuBReal density = *cuMesh.pdensity;
		cuBReal mdamping = *cuMesh.pmdamping;
		cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pdensity, density, *cuMesh.pmdamping, mdamping);

		cuINT3 ijk = cuINT3(i, j, k);

		//external forces on different faces (keep track separately in case an edge cell is excited simultaneously by 2 or more external forces
		cuReal3 Fext_xface = cuReal3(), Fext_yface = cuReal3(), Fext_zface = cuReal3();
		
		//is there an external force? If so, get it, otherwise it will be zero
		if (
			((i == 0 || i == n_m.i) && strain_diag.is_dirichlet_x(idx_u)) ||
			((j == 0 || j == n_m.j) && strain_diag.is_dirichlet_y(idx_u)) ||
			((k == 0 || k == n_m.k) && strain_diag.is_dirichlet_z(idx_u))) {

			//search through all available surfaces to get external force
			for (int sidx = 0; sidx < num_surfaces; sidx++) {

				int orientation = external_stress_surfaces[sidx].contains(ijk_u);
				if (orientation) {

					switch (abs(orientation)) {

						//x face
					case 1:
						Fext_xface = external_stress_surfaces[sidx].get_ext_force_edges(ijk, time);
						break;

						//y face
					case 2:
						Fext_yface = external_stress_surfaces[sidx].get_ext_force_edges(ijk, time);
						break;

						//z face
					case 3:
						Fext_zface = external_stress_surfaces[sidx].get_ext_force_edges(ijk, time);
						break;
					};
				}
			}
		}
		
		//update vx
		if (i < n_m.i) {

			//set zero at fixed faces (for vx only y and z faces are applicable)
			if (((j == 0 || j == n_m.j) && u_disp.is_dirichlet_y(idx_u)) || ((k == 0 || k == n_m.k) && u_disp.is_dirichlet_z(idx_u))) {

				vx[ijk] = 0.0;
			}
			else {

				int njend = (j < n_m.j);
				int nkend = (k < n_m.k);

				//check for required axis normal faces being present
				bool zface_u =
					j < n_m.j &&
					(u_disp.is_not_empty(idx_u) ||
					(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y)));

				bool zface_l =
					j > 0 &&
					(u_disp.is_not_empty(idx_u - njend * n_m.x) ||
					(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x)));

				bool yface_u =
					k < n_m.k &&
					(u_disp.is_not_empty(idx_u) ||
					(j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x)));

				bool yface_l =
					k > 0 &&
					(u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y) ||
					(j > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x)));

				//at least one face is required, otherwise velocity must be zero
				if (zface_u || zface_l || yface_u || yface_l) {

					cuBReal dsxx_dx = 0.0, dsxy_dy = 0.0, dsxz_dz = 0.0;

					//always interior
					dsxx_dx = (sdd[cuINT3(i + 1, j, k)].x - sdd[ijk].x) / h_m.x;

					//interior
					if (zface_u && zface_l) dsxy_dy = (sxy[ijk] - sxy[cuINT3(i, j - 1, k)]) / h_m.y;
					else if (zface_l) dsxy_dy = (Fext_yface.x - sxy[cuINT3(i, j - 1, k)]) / (h_m.y / 2);
					else if (zface_u) dsxy_dy = (sxy[ijk] - Fext_yface.x) / (h_m.y / 2);

					//interior
					if (yface_u && yface_l) dsxz_dz = (sxz[ijk] - sxz[cuINT3(i, j, k - 1)]) / h_m.z;
					else if (yface_l) dsxz_dz = (Fext_zface.x - sxz[cuINT3(i, j, k - 1)]) / (h_m.z / 2);
					else if (yface_u) dsxz_dz = (sxz[ijk] - Fext_zface.x) / (h_m.z / 2);

					vx[ijk] += dT * (dsxx_dx + dsxy_dy + dsxz_dz - mdamping * vx[ijk]) / density;
				}
				else vx[ijk] = 0.0;
			}
		}

		//update vy
		if (j < n_m.j) {

			//set zero at fixed faces (for vy only x and z faces are applicable)
			if (((i == 0 || i == n_m.i) && u_disp.is_dirichlet_x(idx_u)) || ((k == 0 || k == n_m.k) && u_disp.is_dirichlet_z(idx_u))) {

				vy[ijk] = 0.0;
			}
			else {

				int niend = (i < n_m.i);
				int nkend = (k < n_m.k);

				//check for required axis normal faces being present
				bool zface_u =
					i < n_m.i &&
					(u_disp.is_not_empty(idx_u) ||
					(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y)));

				bool zface_l =
					i > 0 &&
					(u_disp.is_not_empty(idx_u - niend) ||
					(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend)));

				bool xface_u =
					k < n_m.k &&
					(u_disp.is_not_empty(idx_u) ||
					(i > 0 && u_disp.is_not_empty(idx_u - niend)));

				bool xface_l =
					k > 0 &&
					(u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y) ||
					(i > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend)));

				//at least one face is required, otherwise velocity must be zero
				if (zface_u || zface_l || xface_u || xface_l) {

					cuBReal dsxy_dx = 0.0, dsyy_dy = 0.0, dsyz_dz = 0.0;

					//always interior
					dsyy_dy = (sdd[cuINT3(i, j + 1, k)].y - sdd[ijk].y) / h_m.y;

					//interior
					if (zface_u && zface_l) dsxy_dx = (sxy[ijk] - sxy[cuINT3(i - 1, j, k)]) / h_m.x;
					else if (zface_l) dsxy_dx = (Fext_xface.y - sxy[cuINT3(i - 1, j, k)]) / (h_m.x / 2);
					else if (zface_u) dsxy_dx = (sxy[ijk] - Fext_xface.y) / (h_m.x / 2);

					//interior
					if (xface_u && xface_l) dsyz_dz = (syz[ijk] - syz[cuINT3(i, j, k - 1)]) / h_m.z;
					else if (xface_l) dsyz_dz = (Fext_zface.y - syz[cuINT3(i, j, k - 1)]) / (h_m.z / 2);
					else if (xface_u) dsyz_dz = (syz[ijk] - Fext_zface.y) / (h_m.z / 2);

					vy[ijk] += dT * (dsxy_dx + dsyy_dy + dsyz_dz - mdamping * vy[ijk]) / density;
				}
				else vy[ijk] = 0.0;
			}
		}

		//update vz
		if (k < n_m.k) {

			//set zero at fixed faces (for vz only x and y faces are applicable)
			if (((i == 0 || i == n_m.i) && u_disp.is_dirichlet_x(idx_u)) || ((j == 0 || j == n_m.j) && u_disp.is_dirichlet_y(idx_u))) {

				vz[ijk] = 0.0;
			}
			else {

				int niend = (i < n_m.i);
				int njend = (j < n_m.j);

				//check for required axis normal faces being present
				bool yface_u =
					i < n_m.i &&
					(u_disp.is_not_empty(idx_u) ||
					(j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x)));

				bool yface_l =
					i > 0 &&
					(u_disp.is_not_empty(idx_u - niend) ||
					(j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - niend)));

				bool xface_u =
					j < n_m.j &&
					(u_disp.is_not_empty(idx_u) ||
					(i > 0 && u_disp.is_not_empty(idx_u - niend)));

				bool xface_l =
					j > 0 &&
					(u_disp.is_not_empty(idx_u - njend * n_m.x) ||
					(i > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - niend)));

				//at least one face is required, otherwise velocity must be zero
				if (yface_u || yface_l || xface_u || xface_l) {

					cuBReal dsxz_dx = 0.0, dsyz_dy = 0.0, dszz_dz = 0.0;

					//always interior
					dszz_dz = (sdd[cuINT3(i, j, k + 1)].z - sdd[ijk].z) / h_m.z;

					//interior
					if (yface_u && yface_l) dsxz_dx = (sxz[ijk] - sxz[cuINT3(i - 1, j, k)]) / h_m.x;
					else if (yface_l) dsxz_dx = (Fext_xface.z - sxz[cuINT3(i - 1, j, k)]) / (h_m.x / 2);
					else if (yface_u) dsxz_dx = (sxz[ijk] - Fext_xface.z) / (h_m.x / 2);

					//interior
					if (xface_u && xface_l) dsyz_dy = (syz[ijk] - syz[cuINT3(i, j - 1, k)]) / h_m.y;
					else if (xface_l) dsyz_dy = (Fext_yface.z - syz[cuINT3(i, j - 1, k)]) / (h_m.y / 2);
					else if (xface_u) dsyz_dy = (syz[ijk] - Fext_yface.z) / (h_m.y / 2);

					vz[ijk] += dT * (dsxz_dx + dsyz_dy + dszz_dz - mdamping * vz[ijk]) / density;
				}
				else vz[ijk] = 0.0;
			}
		}

		//IMPORTANT: don't update mechanical displacement now. Do it in the stress update routine instead.
		//The reason is velocity components on CMBND will be incorrect here, but these will be set from continuity condition correctly after all meshes have updated, and before stress is calculated.
		//Thus mechnical displacement computed here will be incorrect, but when computed in stress update routine it will be correct
	}
}

//----------------------- Iterate_Elastic_Solver LAUNCHERS

//update velocity for dT time increment (also updating displacement)
void MElasticCUDA::Iterate_Elastic_Solver_Velocity1(double dT)
{
	size_t size = (pMeshCUDA->n_m.i + 1) * (pMeshCUDA->n_m.j + 1) * (pMeshCUDA->n_m.k + 1);

	//1a. Update velocity
	Iterate_Elastic_Solver_Velocity1_Kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(pMeshCUDA->cuMesh, external_stress_surfaces_arr, external_stress_surfaces.size(),
		vx, vy, vz, sdd, sxy, sxz, syz,
		pMeshCUDA->GetStageTime(), dT);
}

__global__ void Save_Current_Temperature_Kernel(cuBReal* Temp_previous, cuVEC_VC<cuBReal>& Temp, cuVEC_VC<cuBReal>& Temp_l)
{
	//kernel launch with size n_m.i * n_m.j * n_m.k 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Temp.linear_size()) {

		//2TM
		if (Temp_l.linear_size()) Temp_previous[idx] = Temp_l[idx];
		//1TM
		else Temp_previous[idx] = Temp[idx];
	}
}

//if thermoelasticity is enabled then save current temperature values in Temp_previous (called after elastic solver fully incremented by magnetic_dT)
void MElasticCUDA::Save_Current_Temperature(void)
{
	if (thermoelasticity_enabled) {

		Save_Current_Temperature_Kernel <<< (pMeshCUDA->n_t.dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Temp_previous, pMeshCUDA->Temp, pMeshCUDA->Temp_l);
	}
}

//---------------------------------------------- CMBND routines (verlocity) KERNELS

__global__ void make_velocity_continuous_x_Kernel(
	ManagedMeshCUDA& cuMesh, ManagedMeshCUDA& cuMesh_sec,
	CMBNDInfoCUDA& contact,
	cuVEC<cuBReal>& vx, cuVEC<cuBReal>& vy, cuVEC<cuBReal>& vz,
	cuVEC<cuBReal>& vx_sec, cuVEC<cuBReal>& vy_sec, cuVEC<cuBReal>& vz_sec)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
	cuVEC_VC<cuReal3>& u_disp_sec = *cuMesh_sec.pu_disp;

	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	const cuBox& cb = contact.cells_box;

	//kernel launch with size (cb.e.k + 1 - cb.s.k) * (cb.e.j + 1 - cb.s.j)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int ydim = cb.e.j + 1 - cb.s.j;
	int zdim = cb.e.k + 1 - cb.s.k;

	if (idx < ydim * zdim) {

		int i = (contact.IsPrimaryTop() ? cb.s.i : cb.e.i);
		int j = (idx % ydim) + cb.s.j;
		int k = (idx / ydim) + cb.s.k;

		cuBReal spacing = (h_m.x + u_disp_sec.h.x);
		cuBReal wpri = 1.0 - h_m.x / spacing;
		cuBReal wsec = 1.0 - u_disp_sec.h.x / spacing;

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < u_disp.n.i ? i : u_disp.n.i - 1, j < u_disp.n.j ? j : u_disp.n.j - 1, k < u_disp.n.k ? k : u_disp.n.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * u_disp.n.x + ijk_u.k * u_disp.n.x * u_disp.n.y;

		//absolute position of interface vertex
		cuReal3 abs_pos = (u_disp.h & ijk) + u_disp.rect.s;

		int niend = (i < n_m.i);
		int njend = (j < n_m.j);
		int nkend = (k < n_m.k);

		//check for required faces being present (used for vx and vy components)
		bool xface_u = u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u);
		bool xface_l_vz = j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x) && u_disp.is_cmbnd(idx_u - njend * n_m.x);
		bool xface_l_vy = k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y) && u_disp.is_cmbnd(idx_u - nkend * n_m.x * n_m.y);

		//check for required edge being present (used for vy component)
		bool xedge_u = xface_u || xface_l_vy || xface_l_vz || (j > 0 && k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - njend * n_m.x) && u_disp.is_cmbnd(idx_u - nkend * n_m.x * n_m.y - njend * n_m.x));

		if (contact.IsPrimaryTop()) {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3(-u_disp_sec.h.x / 2, (j == cb.e.j ? -1 : +1) * u_disp.h.y / 2, (k == cb.e.k ? -1 : +1) * u_disp.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);
			int idx_u_sec = ijk_sec.i + ijk_sec.j * u_disp_sec.n.x + ijk_sec.k * u_disp_sec.n.x * u_disp_sec.n.y;

			//vy
			if (j < cb.e.j) {

				//value at interface : interpolate between primary and secondary
				if (xface_u || xface_l_vy) vy[cuINT3(0, j, k)] = vy[cuINT3(1, j, k)] * wpri + vy_sec[ijk_sec + cuINT3(0, 0, k == cb.e.k)] * wsec;
			}

			//vz
			if (k < cb.e.k) {

				//value at interface : interpolate between primary and secondary
				if (xface_u || xface_l_vz) vz[cuINT3(0, j, k)] = vz[cuINT3(1, j, k)] * wpri + vz_sec[ijk_sec + cuINT3(0, j == cb.e.j, 0)] * wsec;
			}

			//set vx continuity obtained from sxx continuity
			if (xedge_u) {

				cuReal3 cC_p = *cuMesh.pcC;
				cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pcC, cC_p);
				cuReal3 cC_s = *cuMesh_sec.pcC;
				cuMesh_sec.update_parameters_scoarse(idx_u_sec, *cuMesh_sec.pcC, cC_s);
				cuBReal rdu = (cC_p.i / cC_s.i) * (u_disp_sec.h.x / h_m.x);

				//simplest case just interpolate values either side (see derivation in notes)
				//in theory there are further contributions here due to 1) side derivatives, 2) thermoelastic contribution, 3) magnetostriction contribution
				//these contributions will be zero if there's no change in cC, alphaT, B1 coefficients across the interface since they are proportional to respective coefficient differences (see notes)
				//however even if there's a material mismatch at the interface, these contributions are still virtually zero since scaled by (cellsize / c11). complicated to include them and no real extra accuracy.
				vx[cuINT3(0, j, k)] = (vx[cuINT3(1, j, k)] * (1 + 3 * rdu) + 2 * vx_sec[ijk_sec + cuINT3(-1, 0, 0)]) / (3 * (1 + rdu));
			}
		}
		else {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3(u_disp_sec.h.x / 2, (j == cb.e.j ? -1 : +1) * u_disp.h.y / 2, (k == cb.e.k ? -1 : +1) * u_disp.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);
			int idx_u_sec = ijk_sec.i + ijk_sec.j * u_disp_sec.n.x + ijk_sec.k * u_disp_sec.n.x * u_disp_sec.n.y;

			//vy
			if (j < cb.e.j) {

				if (xface_u || xface_l_vy) vy[cuINT3(vy.n.x - 1, j, k)] = vy[cuINT3(vy.n.x - 2, j, k)] * wpri + vy_sec[ijk_sec + cuINT3(1, 0, k == cb.e.k)] * wsec;
			}

			//vz
			if (k < cb.e.k) {

				if (xface_u || xface_l_vz) vz[cuINT3(vz.n.x - 1, j, k)] = vz[cuINT3(vz.n.x - 2, j, k)] * wpri + vz_sec[ijk_sec + cuINT3(1, j == cb.e.j, 0)] * wsec;
			}

			//set vx continuity obtained from sxx continuity
			if (xedge_u) {

				cuReal3 cC_p = *cuMesh.pcC;
				cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pcC, cC_p);
				cuReal3 cC_s = *cuMesh_sec.pcC;
				cuMesh_sec.update_parameters_scoarse(idx_u_sec, *cuMesh_sec.pcC, cC_s);
				cuBReal rdu = (cC_p.i / cC_s.i) * (u_disp_sec.h.x / h_m.x);

				//simplest case just interpolate values either side (see derivation in notes)
				//in theory there are further contributions here due to 1) side derivatives, 2) thermoelastic contribution, 3) magnetostriction contribution
				//these contributions will be zero if there's no change in cC, alphaT, B1 coefficients across the interface since they are proportional to respective coefficient differences (see notes)
				//however even if there's a material mismatch at the interface, these contributions are still virtually zero since scaled by (cellsize / c11). complicated to include them and no real extra accuracy.
				vx[cuINT3(vx.n.x - 1, j, k)] = (vx[cuINT3(vx.n.x - 2, j, k)] * (1 + 3 * rdu) + 2 * vx_sec[ijk_sec + cuINT3(1, 0, 0)]) / (3 * (1 + rdu));
			}
		}
	}
}

__global__ void make_velocity_continuous_y_Kernel(
	ManagedMeshCUDA& cuMesh, ManagedMeshCUDA& cuMesh_sec,
	CMBNDInfoCUDA& contact,
	cuVEC<cuBReal>& vx, cuVEC<cuBReal>& vy, cuVEC<cuBReal>& vz,
	cuVEC<cuBReal>& vx_sec, cuVEC<cuBReal>& vy_sec, cuVEC<cuBReal>& vz_sec)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
	cuVEC_VC<cuReal3>& u_disp_sec = *cuMesh_sec.pu_disp;

	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	const cuBox& cb = contact.cells_box;

	//kernel launch with size (cb.e.k + 1 - cb.s.k) * (cb.e.i + 1 - cb.s.i)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int xdim = cb.e.i + 1 - cb.s.i;
	int zdim = cb.e.k + 1 - cb.s.k;

	if (idx < xdim * zdim) {

		int i = (idx % xdim) + cb.s.i;
		int j = (contact.IsPrimaryTop() ? cb.s.j : cb.e.j);
		int k = (idx / xdim) + cb.s.k;

		cuBReal spacing = (h_m.y + u_disp_sec.h.y);
		cuBReal wpri = 1.0 - h_m.y / spacing;
		cuBReal wsec = 1.0 - u_disp_sec.h.y / spacing;

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < u_disp.n.i ? i : u_disp.n.i - 1, j < u_disp.n.j ? j : u_disp.n.j - 1, k < u_disp.n.k ? k : u_disp.n.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * u_disp.n.x + ijk_u.k * u_disp.n.x * u_disp.n.y;

		//absolute position of interface vertex
		cuReal3 abs_pos = (u_disp.h & ijk) + u_disp.rect.s;

		int niend = (i < n_m.i);
		int njend = (j < n_m.j);
		int nkend = (k < n_m.k);

		//check for required faces being present (used for vx and vy components)
		bool yface_u = u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u);
		bool yface_l_vz = i > 0 && u_disp.is_not_empty(idx_u - niend) && u_disp.is_cmbnd(idx_u - niend);
		bool yface_l_vx = k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y) && u_disp.is_cmbnd(idx_u - nkend * n_m.x * n_m.y);

		//check for required edge being present (used for vy component)
		bool yedge_u = yface_u || yface_l_vx || yface_l_vz || (i > 0 && k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - niend) && u_disp.is_cmbnd(idx_u - nkend * n_m.x * n_m.y - niend));

		if (contact.IsPrimaryTop()) {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3((i == cb.e.i ? -1 : +1) * u_disp.h.x / 2, -u_disp_sec.h.y / 2, (k == cb.e.k ? -1 : +1) * u_disp.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);
			int idx_u_sec = ijk_sec.i + ijk_sec.j * u_disp_sec.n.x + ijk_sec.k * u_disp_sec.n.x * u_disp_sec.n.y;

			//vx
			if (i < cb.e.i) {

				if (yface_u || yface_l_vx) vx[cuINT3(i, 0, k)] = vx[cuINT3(i, 1, k)] * wpri + vx_sec[ijk_sec + cuINT3(0, 0, k == cb.e.k)] * wsec;
			}

			//vz
			if (k < cb.e.k) {

				if (yface_u || yface_l_vz) vz[cuINT3(i, 0, k)] = vz[cuINT3(i, 1, k)] * wpri + vz_sec[ijk_sec + cuINT3(i == cb.e.i, 0, 0)] * wsec;
			}

			//set vy continuity obtained from syy continuity
			if (yedge_u) {

				cuReal3 cC_p = *cuMesh.pcC;
				cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pcC, cC_p);
				cuReal3 cC_s = *cuMesh_sec.pcC;
				cuMesh_sec.update_parameters_scoarse(idx_u_sec, *cuMesh_sec.pcC, cC_s);
				cuBReal rdu = (cC_p.i / cC_s.i) * (u_disp_sec.h.y / h_m.y);

				//simplest case just interpolate values either side (see derivation in notes)
				//in theory there are further contributions here due to 1) side derivatives, 2) thermoelastic contribution, 3) magnetostriction contribution
				//these contributions will be zero if there's no change in cC, alphaT, B1 coefficients across the interface since they are proportional to respective coefficient differences (see notes)
				//however even if there's a material mismatch at the interface, these contributions are still virtually zero since scaled by (cellsize / c11). complicated to include them and no real extra accuracy.
				vy[cuINT3(i, 0, k)] = (vy[cuINT3(i, 1, k)] * (1 + 3 * rdu) + 2 * vy_sec[ijk_sec + cuINT3(0, -1, 0)]) / (3 * (1 + rdu));
			}
		}
		else {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3((i == cb.e.i ? -1 : +1) * u_disp.h.x / 2, u_disp_sec.h.y / 2, (k == cb.e.k ? -1 : +1) * u_disp.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);
			int idx_u_sec = ijk_sec.i + ijk_sec.j * u_disp_sec.n.x + ijk_sec.k * u_disp_sec.n.x * u_disp_sec.n.y;

			//vx
			if (i < cb.e.i) {

				vx[cuINT3(i, vx.n.y - 1, k)] = vx[cuINT3(i, vx.n.y - 2, k)] * wpri + vx_sec[ijk_sec + cuINT3(0, 1, k == cb.e.k)] * wsec;
			}

			//vz
			if (k < cb.e.k) {

				vz[cuINT3(i, vz.n.y - 1, k)] = vz[cuINT3(i, vz.n.y - 1, k)] * wpri + vz_sec[ijk_sec + cuINT3(i == cb.e.i, 1, 0)] * wsec;
			}

			//set vy continuity obtained from syy continuity
			if (yedge_u) {

				cuReal3 cC_p = *cuMesh.pcC;
				cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pcC, cC_p);
				cuReal3 cC_s = *cuMesh_sec.pcC;
				cuMesh_sec.update_parameters_scoarse(idx_u_sec, *cuMesh_sec.pcC, cC_s);
				cuBReal rdu = (cC_p.i / cC_s.i) * (u_disp_sec.h.y / h_m.y);

				//simplest case just interpolate values either side (see derivation in notes)
				//in theory there are further contributions here due to 1) side derivatives, 2) thermoelastic contribution, 3) magnetostriction contribution
				//these contributions will be zero if there's no change in cC, alphaT, B1 coefficients across the interface since they are proportional to respective coefficient differences (see notes)
				//however even if there's a material mismatch at the interface, these contributions are still virtually zero since scaled by (cellsize / c11). complicated to include them and no real extra accuracy.
				vy[cuINT3(i, vy.n.y - 1, k)] = (vy[cuINT3(i, vy.n.y - 2, k)] * (1 + 3 * rdu) + 2 * vy_sec[ijk_sec + cuINT3(0, 1, 0)]) / (3 * (1 + rdu));
			}
		}
	}
}

__global__ void make_velocity_continuous_z_Kernel(
	ManagedMeshCUDA& cuMesh, ManagedMeshCUDA& cuMesh_sec,
	CMBNDInfoCUDA& contact,
	cuVEC<cuBReal>& vx, cuVEC<cuBReal>& vy, cuVEC<cuBReal>& vz,
	cuVEC<cuBReal>& vx_sec, cuVEC<cuBReal>& vy_sec, cuVEC<cuBReal>& vz_sec)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
	cuVEC_VC<cuReal3>& u_disp_sec = *cuMesh_sec.pu_disp;

	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	const cuBox& cb = contact.cells_box;

	//kernel launch with size (cb.e.i + 1 - cb.s.i) * (cb.e.j + 1 - cb.s.j)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int xdim = cb.e.i + 1 - cb.s.i;
	int ydim = cb.e.j + 1 - cb.s.j;

	if (idx < xdim * ydim) {

		int i = (idx % xdim) + cb.s.i;
		int j = (idx / xdim) + cb.s.j;
		int k = (contact.IsPrimaryTop() ? cb.s.k : cb.e.k);

		cuBReal spacing = (h_m.z + u_disp_sec.h.z);
		cuBReal wpri = 1.0 - h_m.z / spacing;
		cuBReal wsec = 1.0 - u_disp_sec.h.z / spacing;

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < u_disp.n.i ? i : u_disp.n.i - 1, j < u_disp.n.j ? j : u_disp.n.j - 1, k < u_disp.n.k ? k : u_disp.n.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * u_disp.n.x + ijk_u.k * u_disp.n.x * u_disp.n.y;

		//absolute position of interface vertex
		cuReal3 abs_pos = (u_disp.h & ijk) + u_disp.rect.s;

		int niend = (i < n_m.i);
		int njend = (j < n_m.j);
		int nkend = (k < n_m.k);

		//check for required faces being present (used for vx and vy components)
		bool zface_u = u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u);
		bool zface_l_vy = i > 0 && u_disp.is_not_empty(idx_u - niend) && u_disp.is_cmbnd(idx_u - niend);
		bool zface_l_vx = j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x) && u_disp.is_cmbnd(idx_u - njend * n_m.x);

		//check for required edge being present (used for vz component)
		bool zedge_u = zface_u || zface_l_vx || zface_l_vy || (i > 0 && j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - niend) && u_disp.is_cmbnd(idx_u - njend * n_m.x - niend));

		if (contact.IsPrimaryTop()) {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3((i == cb.e.i ? -1 : +1) * u_disp.h.x / 2, (j == cb.e.j ? -1 : +1) * u_disp.h.y / 2, -u_disp_sec.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);
			int idx_u_sec = ijk_sec.i + ijk_sec.j * u_disp_sec.n.x + ijk_sec.k * u_disp_sec.n.x * u_disp_sec.n.y;

			//vx
			if (i < cb.e.i) {

				if (zface_u || zface_l_vx) vx[cuINT3(i, j, 0)] = vx[cuINT3(i, j, 1)] * wpri + vx_sec[ijk_sec + cuINT3(0, j == cb.e.j, 0)] * wsec;
			}

			//vy
			if (j < cb.e.j) {

				if (zface_u || zface_l_vy) vy[cuINT3(i, j, 0)] = vy[cuINT3(i, j, 1)] * wpri + vy_sec[ijk_sec + cuINT3(i == cb.e.i, 0, 0)] * wsec;
			}

			//set vz continuity obtained from szz continuity
			if (zedge_u) {

				cuReal3 cC_p = *cuMesh.pcC;
				cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pcC, cC_p);
				cuReal3 cC_s = *cuMesh_sec.pcC;
				cuMesh_sec.update_parameters_scoarse(idx_u_sec, *cuMesh_sec.pcC, cC_s);
				cuBReal rdu = (cC_p.i / cC_s.i) * (u_disp_sec.h.z / h_m.z);

				//simplest case just interpolate values either side (see derivation in notes)
				//in theory there are further contributions here due to 1) side derivatives, 2) thermoelastic contribution, 3) magnetostriction contribution
				//these contributions will be zero if there's no change in cC, alphaT, B1 coefficients across the interface since they are proportional to respective coefficient differences (see notes)
				//however even if there's a material mismatch at the interface, these contributions are still virtually zero since scaled by (cellsize / c11). complicated to include them and no real extra accuracy.
				vz[cuINT3(i, j, 0)] = (vz[cuINT3(i, j, 1)] * (1 + 3 * rdu) + 2 * vz_sec[ijk_sec + cuINT3(0, 0, -1)]) / (3 * (1 + rdu));
			}
		}
		else {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3((i == cb.e.i ? -1 : +1) * u_disp.h.x / 2, (j == cb.e.j ? -1 : +1) * u_disp.h.y / 2, u_disp_sec.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);
			int idx_u_sec = ijk_sec.i + ijk_sec.j * u_disp_sec.n.x + ijk_sec.k * u_disp_sec.n.x * u_disp_sec.n.y;

			//vx
			if (i < cb.e.i) {

				if (zface_u || zface_l_vx) vx[cuINT3(i, j, vx.n.z - 1)] = vx[cuINT3(i, j, vx.n.z - 2)] * wpri + vx_sec[ijk_sec + cuINT3(0, j == cb.e.j, 1)] * wsec;
			}

			//vy
			if (j < cb.e.j) {

				if (zface_u || zface_l_vy) vy[cuINT3(i, j, vy.n.z - 1)] = vy[cuINT3(i, j, vy.n.z - 2)] * wpri + vy_sec[ijk_sec + cuINT3(i == cb.e.i, 0, 1)] * wsec;
			}

			//set vz continuity obtained from szz continuity
			if (zedge_u) {

				cuReal3 cC_p = *cuMesh.pcC;
				cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pcC, cC_p);
				cuReal3 cC_s = *cuMesh_sec.pcC;
				cuMesh_sec.update_parameters_scoarse(idx_u_sec, *cuMesh_sec.pcC, cC_s);
				cuBReal rdu = (cC_p.i / cC_s.i) * (u_disp_sec.h.z / h_m.z);

				//simplest case just interpolate values either side (see derivation in notes)
				//in theory there are further contributions here due to 1) side derivatives, 2) thermoelastic contribution, 3) magnetostriction contribution
				//these contributions will be zero if there's no change in cC, alphaT, B1 coefficients across the interface since they are proportional to respective coefficient differences (see notes)
				//however even if there's a material mismatch at the interface, these contributions are still virtually zero since scaled by (cellsize / c11). complicated to include them and no real extra accuracy.
				vz[cuINT3(i, j, vz.n.z - 1)] = (vz[cuINT3(i, j, vz.n.z - 2)] * (1 + 3 * rdu) + 2 * vz_sec[ijk_sec + cuINT3(0, 0, 1)]) / (3 * (1 + rdu));
			}
		}
	}
}

//---------------------------------------------- CMBND routines (stress) LAUNCHER

void MElasticCUDA::make_velocity_continuous(
	cuSZ3 box_dims, int axis,
	cu_obj<CMBNDInfoCUDA>& contact,
	MElasticCUDA* pMElastic_sec)
{
	//+/-x normal face
	if (axis == 1) {

		make_velocity_continuous_x_Kernel <<< ((box_dims.j + 1)*(box_dims.k + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh, pMElastic_sec->pMeshCUDA->cuMesh,
			contact,
			vx, vy, vz,
			pMElastic_sec->vx, pMElastic_sec->vy, pMElastic_sec->vz);
	}

	//+/-y normal face
	else if (axis == 2) {

		make_velocity_continuous_y_Kernel <<< ((box_dims.i + 1)*(box_dims.k + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh, pMElastic_sec->pMeshCUDA->cuMesh, 
			contact,
			vx, vy, vz,
			pMElastic_sec->vx, pMElastic_sec->vy, pMElastic_sec->vz);
	}

	//+/-z normal face
	else if (axis == 3) {

		make_velocity_continuous_z_Kernel <<< ((box_dims.i + 1)*(box_dims.j + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh, pMElastic_sec->pMeshCUDA->cuMesh, 
			contact,
			vx, vy, vz,
			pMElastic_sec->vx, pMElastic_sec->vy, pMElastic_sec->vz);
	}
}

//---------------------------------------------- CMBND routines (stress) KERNELS

__global__ void make_stress_continuous_x_Kernel(
	ManagedMeshCUDA& cuMesh,
	CMBNDInfoCUDA& contact,
	cuVEC<cuReal3>& sdd, cuVEC<cuBReal>& sxy, cuVEC<cuBReal>& sxz, cuVEC<cuBReal>& syz,
	cuVEC<cuReal3>& sdd_sec, cuVEC<cuBReal>& sxy_sec, cuVEC<cuBReal>& sxz_sec, cuVEC<cuBReal>& syz_sec,
	cuVEC_VC<cuReal3>& u_disp_sec)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
	
	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	const cuBox& cb = contact.cells_box;

	//kernel launch with size (cb.e.k + 1 - cb.s.k) * (cb.e.j + 1 - cb.s.j)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int ydim = cb.e.j + 1 - cb.s.j;
	int zdim = cb.e.k + 1 - cb.s.k;

	if (idx < ydim * zdim) {

		int i = (contact.IsPrimaryTop() ? cb.s.i : cb.e.i);
		int j = (idx % ydim) + cb.s.j;
		int k = (idx / ydim) + cb.s.k;

		cuBReal spacing = (h_m.x + u_disp_sec.h.x);
		cuBReal wpri = 1.0 - h_m.x / spacing;
		cuBReal wsec = 1.0 - u_disp_sec.h.x / spacing;

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < u_disp.n.i ? i : u_disp.n.i - 1, j < u_disp.n.j ? j : u_disp.n.j - 1, k < u_disp.n.k ? k : u_disp.n.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * u_disp.n.x + ijk_u.k * u_disp.n.x * u_disp.n.y;

		//absolute position of interface vertex
		cuReal3 abs_pos = (u_disp.h & ijk) + u_disp.rect.s;

		int niend = (ijk.i < n_m.i);
		int njend = (ijk.j < n_m.j);
		int nkend = (ijk.k < n_m.k);

		//check if required edges are present
		bool yedge_u =
			ijk.j < n_m.j &&
			((u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u)) ||
			(ijk.k > 0 && (u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y) && u_disp.is_cmbnd(idx_u - nkend * n_m.x * n_m.y))));

		bool yedge_l =
			ijk.j > 0 &&
			((u_disp.is_not_empty(idx_u - njend * n_m.x) && u_disp.is_cmbnd(idx_u - njend * n_m.x)) ||
			(ijk.k > 0 && (u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - njend * n_m.x) && u_disp.is_cmbnd(idx_u - nkend * n_m.x * n_m.y - njend * n_m.x))));

		bool zedge_u =
			ijk.k < n_m.k &&
			((u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u)) ||
			(ijk.j > 0 && (u_disp.is_not_empty(idx_u - njend * n_m.x) && u_disp.is_cmbnd(idx_u - njend * n_m.x))));

		bool zedge_l =
			ijk.k > 0 &&
			((u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y) && u_disp.is_cmbnd(idx_u - nkend * n_m.x * n_m.y)) ||
			(ijk.j > 0 && (u_disp.is_not_empty(idx_u - njend * n_m.x - nkend * n_m.x * n_m.y) && u_disp.is_cmbnd(idx_u - njend * n_m.x - nkend * n_m.x * n_m.y))));

		if (contact.IsPrimaryTop()) {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3(-u_disp_sec.h.x / 2, (j == cb.e.j ? -1 : +1) * u_disp.h.y / 2, (k == cb.e.k ? -1 : +1) * u_disp.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);

			if ((yedge_u || yedge_l) && (zedge_u || zedge_l))
				sdd[cuINT3(0, j, k)] = sdd[cuINT3(1, j, k)] * wpri + sdd_sec[ijk_sec + cuINT3(0, j == cb.e.j, k == cb.e.k)] * wsec;

			//syz
			if (j < cb.e.j && k < cb.e.k && u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u))
				syz[cuINT3(0, j, k)] = syz[cuINT3(1, j, k)] * wpri + syz_sec[ijk_sec] * wsec;
		}
		else {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3(u_disp_sec.h.x / 2, (j == cb.e.j ? -1 : +1) * u_disp.h.y / 2, (k == cb.e.k ? -1 : +1) * u_disp.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);

			if ((yedge_u || yedge_l) && (zedge_u || zedge_l))
				sdd[cuINT3(sdd.n.x - 1, j, k)] = sdd[cuINT3(sdd.n.x - 2, j, k)] * wpri + sdd_sec[ijk_sec + cuINT3(1, j == cb.e.j, k == cb.e.k)] * wsec;

			//syz
			if (j < cb.e.j && k < cb.e.k && u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u))
				syz[cuINT3(syz.n.x - 1, j, k)] = syz[cuINT3(syz.n.x - 2, j, k)] * wpri + syz_sec[ijk_sec + cuINT3(1, 0, 0)] * wsec;
		}
	}
}

__global__ void make_stress_continuous_y_Kernel(
	ManagedMeshCUDA& cuMesh,
	CMBNDInfoCUDA& contact,
	cuVEC<cuReal3>& sdd, cuVEC<cuBReal>& sxy, cuVEC<cuBReal>& sxz, cuVEC<cuBReal>& syz,
	cuVEC<cuReal3>& sdd_sec, cuVEC<cuBReal>& sxy_sec, cuVEC<cuBReal>& sxz_sec, cuVEC<cuBReal>& syz_sec,
	cuVEC_VC<cuReal3>& u_disp_sec)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;

	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	const cuBox& cb = contact.cells_box;

	//kernel launch with size (cb.e.k + 1 - cb.s.k) * (cb.e.i + 1 - cb.s.i)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int xdim = cb.e.i + 1 - cb.s.i;
	int zdim = cb.e.k + 1 - cb.s.k;

	if (idx < xdim * zdim) {

		int i = (idx % xdim) + cb.s.i;
		int j = (contact.IsPrimaryTop() ? cb.s.j : cb.e.j);
		int k = (idx / xdim) + cb.s.k;

		cuBReal spacing = (h_m.y + u_disp_sec.h.y);
		cuBReal wpri = 1.0 - h_m.y / spacing;
		cuBReal wsec = 1.0 - u_disp_sec.h.y / spacing;

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < u_disp.n.i ? i : u_disp.n.i - 1, j < u_disp.n.j ? j : u_disp.n.j - 1, k < u_disp.n.k ? k : u_disp.n.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * u_disp.n.x + ijk_u.k * u_disp.n.x * u_disp.n.y;

		//absolute position of interface vertex
		cuReal3 abs_pos = (u_disp.h & ijk) + u_disp.rect.s;

		int niend = (ijk.i < n_m.i);
		int njend = (ijk.j < n_m.j);
		int nkend = (ijk.k < n_m.k);

		//check if required edges are present
		bool xedge_u =
			ijk.i < n_m.i &&
			((u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u)) ||
			(ijk.k > 0 && (u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y) && u_disp.is_cmbnd(idx_u - nkend * n_m.x * n_m.y))));

		bool xedge_l =
			ijk.i > 0 &&
			((u_disp.is_not_empty(idx_u - niend) && u_disp.is_cmbnd(idx_u - niend)) ||
			(ijk.k > 0 && (u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - niend) && u_disp.is_cmbnd(idx_u - nkend * n_m.x * n_m.y - niend))));

		bool zedge_u =
			ijk.k < n_m.k &&
			((u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u)) ||
			(ijk.i > 0 && (u_disp.is_not_empty(idx_u - niend) && u_disp.is_cmbnd(idx_u - niend))));

		bool zedge_l =
			ijk.k > 0 &&
			((u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y) && u_disp.is_cmbnd(idx_u - nkend * n_m.x * n_m.y)) ||
			(ijk.i > 0 && (u_disp.is_not_empty(idx_u - niend - nkend * n_m.x * n_m.y) && u_disp.is_cmbnd(idx_u - niend - nkend * n_m.x * n_m.y))));

		if (contact.IsPrimaryTop()) {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3((i == cb.e.i ? -1 : +1) * u_disp.h.x / 2, -u_disp_sec.h.y / 2, (k == cb.e.k ? -1 : +1) * u_disp.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);

			//sxx, syy, szz
			if ((xedge_u || xedge_l) && (zedge_u || zedge_l))
				sdd[cuINT3(i, 0, k)] = sdd[cuINT3(i, 1, k)] * wpri + sdd_sec[ijk_sec + cuINT3(i == cb.e.i, 0, k == cb.e.k)] * wsec;

			//sxz
			if (i < cb.e.i && k < cb.e.k && u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u))
				sxz[cuINT3(i, 0, k)] = sxz[cuINT3(i, 1, k)] * wpri + sxz_sec[ijk_sec] * wsec;
		}
		else {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3((i == cb.e.i ? -1 : +1) * u_disp.h.x / 2, u_disp_sec.h.y / 2, (k == cb.e.k ? -1 : +1) * u_disp.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);

			//sxx, syy, szz
			if ((xedge_u || xedge_l) && (zedge_u || zedge_l))
				sdd[cuINT3(i, sdd.n.y - 1, k)] = sdd[cuINT3(i, sdd.n.y - 2, k)] * wpri + sdd_sec[ijk_sec + cuINT3(i == cb.e.i, 1, k == cb.e.k)] * wsec;

			//sxz
			if (i < cb.e.j && k < cb.e.k && u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u))
				sxz[cuINT3(i, sxz.n.y - 1, k)] = sxz[cuINT3(i, sxz.n.y - 2, k)] * wpri + sxz_sec[ijk_sec + cuINT3(0, 1, 0)] * wsec;
		}
	}
}

__global__ void make_stress_continuous_z_Kernel(
	ManagedMeshCUDA& cuMesh,
	CMBNDInfoCUDA& contact,
	cuVEC<cuReal3>& sdd, cuVEC<cuBReal>& sxy, cuVEC<cuBReal>& sxz, cuVEC<cuBReal>& syz,
	cuVEC<cuReal3>& sdd_sec, cuVEC<cuBReal>& sxy_sec, cuVEC<cuBReal>& sxz_sec, cuVEC<cuBReal>& syz_sec,
	cuVEC_VC<cuReal3>& u_disp_sec)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;

	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	const cuBox& cb = contact.cells_box;

	//kernel launch with size (cb.e.i + 1 - cb.s.i) * (cb.e.j + 1 - cb.s.j)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int xdim = cb.e.i + 1 - cb.s.i;
	int ydim = cb.e.j + 1 - cb.s.j;

	if (idx < xdim * ydim) {

		int i = (idx % xdim) + cb.s.i;
		int j = (idx / xdim) + cb.s.j;
		int k = (contact.IsPrimaryTop() ? cb.s.k : cb.e.k);

		cuBReal spacing = (h_m.z + u_disp_sec.h.z);
		cuBReal wpri = 1.0 - h_m.z / spacing;
		cuBReal wsec = 1.0 - u_disp_sec.h.z / spacing;

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < u_disp.n.i ? i : u_disp.n.i - 1, j < u_disp.n.j ? j : u_disp.n.j - 1, k < u_disp.n.k ? k : u_disp.n.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * u_disp.n.x + ijk_u.k * u_disp.n.x * u_disp.n.y;

		//absolute position of interface vertex
		cuReal3 abs_pos = (u_disp.h & ijk) + u_disp.rect.s;

		int niend = (ijk.i < n_m.i);
		int njend = (ijk.j < n_m.j);
		int nkend = (ijk.k < n_m.k);

		//check if required edges are present
		bool xedge_u =
			ijk.i < n_m.i &&
			((u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u)) ||
			(ijk.j > 0 && (u_disp.is_not_empty(idx_u - njend * n_m.x) && u_disp.is_cmbnd(idx_u - njend * n_m.x))));

		bool xedge_l =
			ijk.i > 0 &&
			((u_disp.is_not_empty(idx_u - niend) && u_disp.is_cmbnd(idx_u - niend)) ||
			(ijk.j > 0 && (u_disp.is_not_empty(idx_u - njend * n_m.x - niend) && u_disp.is_cmbnd(idx_u - njend * n_m.x - niend))));

		bool yedge_u =
			ijk.j < n_m.j &&
			((u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u)) ||
			(ijk.i > 0 && (u_disp.is_not_empty(idx_u - niend) && u_disp.is_cmbnd(idx_u - niend))));

		bool yedge_l =
			ijk.j > 0 &&
			((u_disp.is_not_empty(idx_u - njend * n_m.x) && u_disp.is_cmbnd(idx_u - njend * n_m.x)) ||
			(ijk.i > 0 && (u_disp.is_not_empty(idx_u - niend - njend * n_m.x) && u_disp.is_cmbnd(idx_u - niend - njend * n_m.x))));

		if (contact.IsPrimaryTop()) {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3((i == cb.e.i ? -1 : +1) * u_disp.h.x / 2, (j == cb.e.j ? -1 : +1) * u_disp.h.y / 2, -u_disp_sec.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);

			//sxx, syy, szz
			if ((xedge_u || xedge_l) && (yedge_u || yedge_l))
				sdd[cuINT3(i, j, 0)] = sdd[cuINT3(i, j, 1)] * wpri + sdd_sec[ijk_sec + cuINT3(i == cb.e.i, j == cb.e.j, 0)] * wsec;
	
			//sxy
			if (i < cb.e.i && j < cb.e.j && u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u)) 
				sxy[cuINT3(i, j, 0)] = sxy[cuINT3(i, j, 1)] * wpri + sxy_sec[ijk_sec] * wsec;
		}
		else {

			//absolute position in secondary, half a cellsize into it, and middle of primary cell face
			cuReal3 abs_pos_sec = abs_pos + cuReal3((i == cb.e.i ? -1 : +1) * u_disp.h.x / 2, (j == cb.e.j ? -1 : +1) * u_disp.h.y / 2, u_disp_sec.h.z / 2);
			//index of secondary cell just next to boundary
			cuINT3 ijk_sec = u_disp_sec.cellidx_from_position(abs_pos_sec);

			//sxx, syy, szz
			if ((xedge_u || xedge_l) && (yedge_u || yedge_l))
				sdd[cuINT3(i, j, sdd.n.z - 1)] = sdd[cuINT3(i, j, sdd.n.z - 2)] * wpri + sdd_sec[ijk_sec + cuINT3(i == cb.e.i, j == cb.e.j, 1)] * wsec;

			//sxy
			if (i < cb.e.i && j < cb.e.j && u_disp.is_not_empty(idx_u) && u_disp.is_cmbnd(idx_u)) 
				sxy[cuINT3(i, j, sxy.n.z - 1)] = sxy[cuINT3(i, j, sxy.n.z - 2)] * wpri + sxy_sec[ijk_sec + cuINT3(0, 0, 1)] * wsec;
				
		}
	}
}

//---------------------------------------------- CMBND routines (stress) LAUNCHERS

void MElasticCUDA::make_stress_continuous(
	cuSZ3 box_dims, int axis,
	cu_obj<CMBNDInfoCUDA>& contact,
	cu_obj<cuVEC<cuReal3>>& sdd_sec, cu_obj<cuVEC<cuBReal>>& sxy_sec, cu_obj<cuVEC<cuBReal>>& sxz_sec, cu_obj<cuVEC<cuBReal>>& syz_sec,
	cu_obj<cuVEC_VC<cuReal3>>& u_disp_sec)
{
	//+/-x normal face
	if (axis == 1) {

		make_stress_continuous_x_Kernel <<< ((box_dims.j + 1)*(box_dims.k + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh, contact, 
			sdd, sxy, sxz, syz,
			sdd_sec, sxy_sec, sxz_sec, syz_sec,
			u_disp_sec);
	}

	//+/-y normal face
	else if (axis == 2) {

		make_stress_continuous_y_Kernel <<< ((box_dims.i + 1)*(box_dims.k + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh, contact,
			sdd, sxy, sxz, syz,
			sdd_sec, sxy_sec, sxz_sec, syz_sec,
			u_disp_sec);
	}

	//+/-z normal face
	else if (axis == 3) {

		make_stress_continuous_z_Kernel <<< ((box_dims.i + 1)*(box_dims.j + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh, contact,
			sdd, sxy, sxz, syz,
			sdd_sec, sxy_sec, sxz_sec, syz_sec,
			u_disp_sec);
	}
}

#endif

#endif
