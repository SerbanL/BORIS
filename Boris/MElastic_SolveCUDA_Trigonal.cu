#include "MElasticCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MELASTIC

#include "BorisCUDALib.cuh"

#include "MeshDefs.h"

#include "ManagedDiffEqFMCUDA.h"
#include "ManagedDiffEqAFMCUDA.h"
#include "MeshParamsControlCUDA.h"

#include "MElastic_BoundariesCUDA.h"

//----------------------- Calculate_MElastic_Field KERNELS

__global__ void MElasticCUDA_Trigonal_UpdateField_FM(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;

	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;
	cuVEC_VC<cuReal3>& strain_odiag = *cuMesh.pstrain_odiag;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		if (M.is_not_empty(idx)) {

			cuBReal Ms = *cuMesh.pMs;
			cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
			cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
			cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
			cuReal2 MEc = *cuMesh.pMEc;
			cuReal2 MEc2 = *cuMesh.pMEc2;
			cuReal2 MEc3 = *cuMesh.pMEc3;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms, *cuMesh.pMEc, MEc, *cuMesh.pMEc2, MEc2, *cuMesh.pMEc3, MEc3, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);
			
			cuBReal& B21 = MEc.i;
			cuBReal& B22 = MEc.j;
			cuBReal& B3 = MEc2.i;
			cuBReal& B4 = MEc2.j;
			cuBReal& B14 = MEc3.i;
			cuBReal& B34 = MEc3.j;
			
			cuReal3 position = M.cellidx_to_position(idx);
			//xx, yy, zz
			cuReal3 Sd = strain_diag[position];
			//yz, xz, xy
			cuReal3 Sod = strain_odiag[position];

			//normalised magnetization
			//Magneto-elastic term here applicable for a cubic crystal. We use the mcanis_ea1 and mcanis_ea2 axes to fix the cubic lattice orientation, thus rotate the m, Sd and Sod vectors.

			cuReal3 m = cuReal3(M[idx] * mcanis_ea1, M[idx] * mcanis_ea2, M[idx] * mcanis_ea3) / Ms;
			Sd = cuReal3(Sd * mcanis_ea1, Sd * mcanis_ea2, Sd * mcanis_ea3);
			Sod = cuReal3(Sod * mcanis_ea1, Sod * mcanis_ea2, Sod * mcanis_ea3);

			cuReal3 Hmel = cuReal3();
			Hmel += 2 * B21*(Sd.x + Sd.y)*m.z*mcanis_ea3 + 2 * B22*Sd.z*m.z*mcanis_ea3;
			Hmel += B3 * (Sd.x - Sd.y) * (m.x*mcanis_ea1 - m.y*mcanis_ea2) + 2 * B3 * Sod.z * (m.x*mcanis_ea2 + m.y*mcanis_ea1);
			Hmel += 2 * B4*Sod.y*(m.x*mcanis_ea3 + m.z*mcanis_ea1) + 2 * B4*Sod.x * (m.y*mcanis_ea3 + m.z*mcanis_ea2);
			Hmel += 2 * B14*Sod.x*(m.x*mcanis_ea1 - m.y*mcanis_ea2) + 2 * B14*Sod.y*(m.x*mcanis_ea2 + m.y*mcanis_ea1);
			Hmel += B34 * (Sd.x - Sd.y) * (m.y*mcanis_ea3 + m.z*mcanis_ea2) + 2 * B34*Sod.z*(m.x*mcanis_ea3 + m.z*mcanis_ea1);
			Hmel *= -1 / ((cuBReal)MU0 * Ms);

			Heff[idx] += Hmel;

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * M[idx] * Hmel / (2 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hmel;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MU0 * M[idx] * Hmel / 2;
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void MElasticCUDA_Trigonal_UpdateField_AFM(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;

	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;
	cuVEC_VC<cuReal3>& strain_odiag = *cuMesh.pstrain_odiag;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		if (M.is_not_empty(idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
			cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
			cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
			cuReal2 MEc = *cuMesh.pMEc;
			cuReal2 MEc2 = *cuMesh.pMEc2;
			cuReal2 MEc3 = *cuMesh.pMEc3;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pMEc, MEc, *cuMesh.pMEc2, MEc2, *cuMesh.pMEc3, MEc3, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

			cuBReal& B21 = MEc.i;
			cuBReal& B22 = MEc.j;
			cuBReal& B3 = MEc2.i;
			cuBReal& B4 = MEc2.j;
			cuBReal& B14 = MEc3.i;
			cuBReal& B34 = MEc3.j;

			cuReal3 position = M.cellidx_to_position(idx);
			//xx, yy, zz
			cuReal3 Sd = strain_diag[position];
			//yz, xz, xy
			cuReal3 Sod = strain_odiag[position];

			//normalised magnetization
			//Magneto-elastic term here applicable for a cubic crystal. We use the mcanis_ea1 and mcanis_ea2 axes to fix the cubic lattice orientation, thus rotate the m, Sd and Sod vectors.

			cuReal3 mA = cuReal3(M[idx] * mcanis_ea1, M[idx] * mcanis_ea2, M[idx] * mcanis_ea3) / Ms_AFM.i;
			cuReal3 mB = cuReal3(M2[idx] * mcanis_ea1, M2[idx] * mcanis_ea2, M2[idx] * mcanis_ea3) / Ms_AFM.j;

			Sd = cuReal3(Sd * mcanis_ea1, Sd * mcanis_ea2, Sd * mcanis_ea3);
			Sod = cuReal3(Sod * mcanis_ea1, Sod * mcanis_ea2, Sod * mcanis_ea3);

			cuReal3 Hmel_A = cuReal3();
			Hmel_A += 2 * B21*(Sd.x + Sd.y)*mA.z*mcanis_ea3 + 2 * B22*Sd.z*mA.z*mcanis_ea3;
			Hmel_A += B3 * (Sd.x - Sd.y) * (mA.x*mcanis_ea1 - mA.y*mcanis_ea2) + 2 * B3 * Sod.z * (mA.x*mcanis_ea2 + mA.y*mcanis_ea1);
			Hmel_A += 2 * B4*Sod.y*(mA.x*mcanis_ea3 + mA.z*mcanis_ea1) + 2 * B4*Sod.x * (mA.y*mcanis_ea3 + mA.z*mcanis_ea2);
			Hmel_A += 2 * B14*Sod.x*(mA.x*mcanis_ea1 - mA.y*mcanis_ea2) + 2 * B14*Sod.y*(mA.x*mcanis_ea2 + mA.y*mcanis_ea1);
			Hmel_A += B34 * (Sd.x - Sd.y) * (mA.y*mcanis_ea3 + mA.z*mcanis_ea2) + 2 * B34*Sod.z*(mA.x*mcanis_ea3 + mA.z*mcanis_ea1);
			Hmel_A *= -1 / ((cuBReal)MU0 * Ms_AFM.i);

			cuReal3 Hmel_B = cuReal3();
			Hmel_B += 2 * B21*(Sd.x + Sd.y)*mB.z*mcanis_ea3 + 2 * B22*Sd.z*mB.z*mcanis_ea3;
			Hmel_B += B3 * (Sd.x - Sd.y) * (mB.x*mcanis_ea1 - mB.y*mcanis_ea2) + 2 * B3 * Sod.z * (mB.x*mcanis_ea2 + mB.y*mcanis_ea1);
			Hmel_B += 2 * B4*Sod.y*(mB.x*mcanis_ea3 + mB.z*mcanis_ea1) + 2 * B4*Sod.x * (mB.y*mcanis_ea3 + mB.z*mcanis_ea2);
			Hmel_B += 2 * B14*Sod.x*(mB.x*mcanis_ea1 - mB.y*mcanis_ea2) + 2 * B14*Sod.y*(mB.x*mcanis_ea2 + mB.y*mcanis_ea1);
			Hmel_B += B34 * (Sd.x - Sd.y) * (mB.y*mcanis_ea3 + mB.z*mcanis_ea2) + 2 * B34*Sod.z*(mB.x*mcanis_ea3 + mB.z*mcanis_ea1);
			Hmel_B *= -1 / ((cuBReal)MU0 * Ms_AFM.j);

			Heff[idx] += Hmel_A;
			Heff2[idx] += Hmel_B;

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * (M[idx] * Hmel_A + M2[idx] * Hmel_B) / (2 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hmel_A;
			if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[idx] = Hmel_B;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MU0 * M[idx] * Hmel_A / 2;
			if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[idx] = -(cuBReal)MU0 * M2[idx] * Hmel_B / 2;
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- Calculate_MElastic_Field LAUNCHER

//compute magnetoelastic effective field to use in magnetization equation.
void MElasticCUDA::Calculate_MElastic_Field_Trigonal(void)
{
	//disabled by setting magnetoelastic coefficient to zero (also disabled in non-magnetic meshes)
	if (melastic_field_disabled) return;

	ZeroEnergy();

	if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

		//anti-ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			MElasticCUDA_Trigonal_UpdateField_AFM <<< (pMeshCUDA->n.dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (pMeshCUDA->cuMesh, cuModule, true);
		}
		else {

			MElasticCUDA_Trigonal_UpdateField_AFM <<< (pMeshCUDA->n.dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (pMeshCUDA->cuMesh, cuModule, false);
		}
	}
	else if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

		//ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			MElasticCUDA_Trigonal_UpdateField_FM <<< (pMeshCUDA->n.dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (pMeshCUDA->cuMesh, cuModule, true);
		}
		else {

			MElasticCUDA_Trigonal_UpdateField_FM <<< (pMeshCUDA->n.dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (pMeshCUDA->cuMesh, cuModule, false);
		}
	}
}

//----------------------------------------------- Computational Helpers

//----- Velocity

__global__ void Iterate_Elastic_Solver_Velocity2_Kernel(
	ManagedMeshCUDA& cuMesh,
	MElastic_BoundaryCUDA* external_stress_surfaces, size_t num_surfaces,
	cuVEC<cuBReal>& vx2, cuVEC<cuBReal>& vy2, cuVEC<cuBReal>& vz2,
	cuVEC<cuReal3>& sdd2,
	cuVEC<cuBReal>& sxy2, cuVEC<cuBReal>& sxz2, cuVEC<cuBReal>& syz2,
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
	int k = idx / ((n_m.i + 1) * (n_m.j + 1));

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

		//update vx2
		if (i < n_m.i && j < n_m.j && k < n_m.k) {

			if (u_disp.is_not_empty(idx_u)) {

				cuBReal dsxx2_dx = (sdd2[cuINT3(i + 1, j, k)].x - sdd2[ijk].x) / h_m.x;
				cuBReal dsxy2_dy = (sxy2[cuINT3(i, j + 1, k)] - sxy2[ijk]) / h_m.y;
				cuBReal dsxz2_dz = (sxz2[cuINT3(i, j, k + 1)] - sxz2[ijk]) / h_m.z;

				vx2[ijk] += dT * (dsxx2_dx + dsxy2_dy + dsxz2_dz - mdamping * vx2[ijk]) / density;
			}
			else vx2[ijk] = 0.0;
		}

		//update vy2
		if (k < n_m.k) {

			//set zero at fixed faces (for vy2 only x and y faces are applicable)
			if (((i == 0 || i == n_m.i) && u_disp.is_dirichlet_x(idx_u)) || ((j == 0 || j == n_m.j) && u_disp.is_dirichlet_y(idx_u))) {

				vy2[ijk] = 0.0;
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

					cuBReal dsxy2_dx = 0.0, dsyy2_dy = 0.0, dsyz2_dz = 0.0;

					//always interior
					dsyz2_dz = (syz2[cuINT3(i, j, k + 1)] - syz2[ijk]) / h_m.z;

					//interior
					if (yface_u && yface_l) dsxy2_dx = (sxy2[ijk] - sxy2[cuINT3(i - 1, j, k)]) / h_m.x;
					else if (yface_l) dsxy2_dx = (Fext_xface.y - sxy2[cuINT3(i - 1, j, k)]) / (h_m.x / 2);
					else if (yface_u) dsxy2_dx = (sxy2[ijk] - Fext_xface.y) / (h_m.x / 2);

					//interior
					if (xface_u && xface_l) dsyy2_dy = (sdd2[ijk].y - sdd2[cuINT3(i, j - 1, k)].y) / h_m.y;
					else if (xface_l) dsyy2_dy = (Fext_yface.y - sdd2[cuINT3(i, j - 1, k)].y) / (h_m.y / 2);
					else if (xface_u) dsyy2_dy = (sdd2[ijk].y - Fext_yface.y) / (h_m.y / 2);

					vy2[ijk] += dT * (dsxy2_dx + dsyy2_dy + dsyz2_dz - mdamping * vy2[ijk]) / density;
				}
				else vy2[ijk] = 0.0;
			}
		}

		//update vz2
		if (j < n_m.j) {

			//set zero at fixed faces (for vz2 only x and z faces are applicable)
			if (((i == 0 || i == n_m.i) && u_disp.is_dirichlet_x(idx_u)) || ((k == 0 || k == n_m.k) && u_disp.is_dirichlet_z(idx_u))) {

				vz2[ijk] = 0.0;
			}
			else {

				int niend = (i < n_m.i);
				int nkend = (k < n_m.k);

				//check for required axis normal faces being present
				bool zface_u =
					i < n_m.i &&
					(u_disp.is_not_empty(idx_u) ||
						(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y)));

				bool zface_l =
					i > 0 &&
					(u_disp.is_not_empty(idx_u - niend) ||
						(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - niend)));

				bool xface_u =
					k < n_m.k &&
					(u_disp.is_not_empty(idx_u) ||
						(i > 0 && u_disp.is_not_empty(idx_u - niend)));

				bool xface_l =
					k > 0 &&
					(u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y) ||
						(i > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - niend)));

				//at least one face is required, otherwise velocity must be zero
				if (zface_u || zface_l || xface_u || xface_l) {

					cuBReal dsxz2_dx = 0.0, dsyz2_dy = 0.0, dszz2_dz = 0.0;

					//always interior
					dsyz2_dy = (syz2[cuINT3(i, j + 1, k)] - syz2[ijk]) / h_m.y;

					//interior
					if (zface_u && zface_l) dsxz2_dx = (sxz2[ijk] - sxz2[cuINT3(i - 1, j, k)]) / h_m.x;
					else if (zface_l) dsxz2_dx = (Fext_xface.z - sxz2[cuINT3(i - 1, j, k)]) / (h_m.x / 2);
					else if (zface_u) dsxz2_dx = (sxz2[ijk] - Fext_xface.z) / (h_m.x / 2);

					//interior
					if (xface_u && xface_l) dszz2_dz = (sdd2[ijk].z - sdd2[cuINT3(i, j, k - 1)].z) / h_m.z;
					else if (xface_l) dszz2_dz = (Fext_zface.z - sdd2[cuINT3(i, j, k - 1)].z) / (h_m.z / 2);
					else if (xface_u) dszz2_dz = (sdd2[ijk].z - Fext_zface.z) / (h_m.z / 2);

					vz2[ijk] += dT * (dsxz2_dx + dsyz2_dy + dszz2_dz - mdamping * vz2[ijk]) / density;
				}
				else vz2[ijk] = 0.0;
			}
		}
	}
}

//----------------------- Iterate_Elastic_Solver LAUNCHERS

//update velocity for dT time increment (also updating displacement)
void MElasticCUDA::Iterate_Elastic_Solver_Velocity2(double dT)
{
	size_t size = (pMeshCUDA->n_m.i + 1) * (pMeshCUDA->n_m.j + 1) * (pMeshCUDA->n_m.k + 1);

	//1a. Update velocity
	Iterate_Elastic_Solver_Velocity2_Kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(pMeshCUDA->cuMesh, external_stress_surfaces_arr, external_stress_surfaces.size(),
		vx2, vy2, vz2, sdd2, sxy2, sxz2, syz2,
		pMeshCUDA->GetStageTime(), dT);
}

//----------------------- Iterate_Elastic_Solver KERNELS

__device__ void Iterate_Elastic_Solver_Stress_Trigonal_CUDA(
	cuINT3 ijk, cuINT3 ijk_u, int idx_u,
	ManagedMeshCUDA& cuMesh,
	MElastic_BoundaryCUDA* external_stress_surfaces, size_t num_surfaces,
	cuVEC<cuBReal>& vx, cuVEC<cuBReal>& vy, cuVEC<cuBReal>& vz,
	cuVEC<cuReal3>& sdd, cuVEC<cuBReal>& sxy, cuVEC<cuBReal>& sxz, cuVEC<cuBReal>& syz,
	cuVEC<cuBReal>& vx2, cuVEC<cuBReal>& vy2, cuVEC<cuBReal>& vz2,
	cuVEC<cuReal3>& sdd2, cuVEC<cuBReal>& sxy2, cuVEC<cuBReal>& sxz2, cuVEC<cuBReal>& syz2,
	cuBReal time, cuBReal dT,
	bool thermoelasticity_enabled,
	cuBReal* Temp_previous, cuBReal magnetic_dT,
	cuReal3 dsdd_dt_ms, cuReal3 dsod_dt_ms)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;

	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	cuReal3 cC = *cuMesh.pcC;
	cuReal3 cC3 = *cuMesh.pcC3;
	cuReal3 cCs = *cuMesh.pcCs;
	cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pcC, cC, *cuMesh.pcC3, cC3, *cuMesh.pcCs, cCs);

	//get named coefficients for clarity to avoid typing errors
	cuBReal& c11 = cC.i;
	cuBReal& c12 = cC.j;
	cuBReal& c44 = cC.k;
	cuBReal& c33 = cC3.i;
	cuBReal& c13 = cC3.j;
	cuBReal& c66 = cC3.k;
	cuBReal& c14 = cCs.i;

	cuBReal r12_11 = c12 / c11;
	cuBReal r13_11 = c13 / c11;
	cuBReal r14_11 = c14 / c11;
	cuBReal r13_33 = c13 / c33;
	cuBReal r14_44 = c14 / c44;
	cuBReal r14_66 = c14 / c66;
	cuBReal r11_44 = c11 / c44;
	cuBReal r12_44 = c12 / c44;
	cuBReal r13_44 = c13 / c44;

	int& i = ijk.i;
	int& j = ijk.j;
	int& k = ijk.k;

	///////////// THERMOELASTICITY CONTRIBUTION

	//needed for thermoelasticity (includes time derivative of temperature)
	cuBReal dsxx_yy_dt_te = 0.0;
	cuBReal dszz_dt_te = 0.0;
	if (thermoelasticity_enabled) {

		cuVEC_VC<cuBReal>& Temp = *cuMesh.pTemp;
		cuVEC_VC<cuBReal>& Temp_l = *cuMesh.pTemp_l;

		int idx_T = Temp.position_to_cellidx(u_disp.cellidx_to_position(idx_u));

		if (Temp.is_not_empty(idx_T)) {

			cuBReal thalpha = *cuMesh.pthalpha;
			cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pthalpha, thalpha);

			cuBReal Temperature = 0.0;
			//for 2TM we need to use the lattice temperature
			if (Temp_l.linear_size()) Temperature = Temp_l[idx_T];
			else Temperature = Temp[idx_T];

			dsxx_yy_dt_te = (c11 + c12 + c13) * thalpha * (Temperature - Temp_previous[idx_T]) / magnetic_dT;
			dszz_dt_te = (c33 + 2 * c13) * thalpha * (Temperature - Temp_previous[idx_T]) / magnetic_dT;
		}
	}

	///////////// EXTERNAL FORCES

	//external forces on different faces (keep track separately in case an edge cell is excited simultaneously by 2 or more external forces
	cuReal3 Fext_xface = cuReal3(), Fext_yface = cuReal3(), Fext_zface = cuReal3();

	//time derivatives of forces on the different faces
	//first index is face normal, second index is force component. e.g. dFyz is for y-face, force z component.
	cuBReal dFxx = -dsdd_dt_ms.x + dsxx_yy_dt_te;
	cuBReal dFyy = -dsdd_dt_ms.y + dsxx_yy_dt_te;
	cuBReal dFzz = -dsdd_dt_ms.z + dszz_dt_te;
	cuBReal dFyz = -dsod_dt_ms.x;
	cuBReal dFzx = -dsod_dt_ms.y;
	cuBReal dFyx = -dsod_dt_ms.z;

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
		(u_disp.is_not_empty(idx_u) ||
			(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y)) ||
			(j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x)) ||
			(j > 0 && k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - njend * n_m.x)));

	bool xedge_l =
		i > 0 &&
		(u_disp.is_not_empty(idx_u - niend) ||
			(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - niend)) ||
			(j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - niend)) ||
			(j > 0 && k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - njend * n_m.x - niend)));

	bool yedge_u =
		j < n_m.j &&
		(u_disp.is_not_empty(idx_u) ||
			(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y)) ||
			(i > 0 && u_disp.is_not_empty(idx_u - niend)) ||
			(i > 0 && k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - niend)));

	bool yedge_l =
		j > 0 &&
		(u_disp.is_not_empty(idx_u - njend * n_m.x) ||
			(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - njend * n_m.x)) ||
			(i > 0 && u_disp.is_not_empty(idx_u - niend - njend * n_m.x)) ||
			(i > 0 && k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y - niend - njend * n_m.x)));

	bool zedge_u =
		k < n_m.k &&
		(u_disp.is_not_empty(idx_u) ||
			(j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x)) ||
			(i > 0 && u_disp.is_not_empty(idx_u - niend)) ||
			(i > 0 && j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - niend)));

	bool zedge_l =
		k > 0 &&
		(u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y) ||
			(j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - nkend * n_m.x * n_m.y)) ||
			(i > 0 && u_disp.is_not_empty(idx_u - niend - nkend * n_m.x * n_m.y)) ||
			(i > 0 && j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - niend - nkend * n_m.x * n_m.y)));

	//check for fixed faces at ends
	bool xfixed_l = (i == 0 && u_disp.is_dirichlet_px(idx_u));
	bool xfixed_u = (i == n_m.i && u_disp.is_dirichlet_nx(idx_u));

	bool yfixed_l = (j == 0 && u_disp.is_dirichlet_py(idx_u));
	bool yfixed_u = (j == n_m.j && u_disp.is_dirichlet_ny(idx_u));

	bool zfixed_l = (k == 0 && u_disp.is_dirichlet_pz(idx_u));
	bool zfixed_u = (k == n_m.k && u_disp.is_dirichlet_nz(idx_u));

	//dvx/dx at vertex
	cuBReal dvx_dx = 0.0;

	//interior
	if (xedge_u && xedge_l) dvx_dx = (vx[ijk] - vx[cuINT3(i - 1, j, k)]) / h_m.x;
	//fixed face : Dirichlet value of zero for velocity derivative
	else if (xedge_l && xfixed_u) {

		dvx_dx = -vx[cuINT3(i - 1, j, k)] / (h_m.x / 2);
	}
	else if (xedge_u && xfixed_l) {

		dvx_dx = vx[ijk] / (h_m.x / 2);
	}
	//free x face
	else {

		//both side derivatives
		if (yedge_l && yedge_u && zedge_l && zedge_u) {

			//VERIFIED
			dvx_dx = (dFxx / c11 - r12_11 * (vy[ijk] - vy[cuINT3(i, j - 1, k)]) / h_m.y - r13_11 * (vz[ijk] - vz[cuINT3(i, j, k - 1)]) / h_m.z - r14_11 * ((vz2[ijk] - vz2[cuINT3(i, j - 1, k)]) / h_m.y + (vy2[ijk] - vy2[cuINT3(i, j, k - 1)]) / h_m.z));
		}
		//only z derivative
		else if (zedge_l && zedge_u) {

			//VERIFIED
			cuBReal rdiv = 1 - r12_11 * r12_11 - 4 * r14_11 * r14_44 * (1 + r12_11);
			dvx_dx = ((dFxx / c11) * (1 - 2 * r14_11 * r14_44) - (dFyy / c11) * (r12_11 + 2 * r14_11 * r14_44) - (dFyz / c44) * r14_11 * (1 + r12_11) - ((vz[ijk] - vz[cuINT3(i, j, k - 1)]) / h_m.z) * r13_11 * (1 - r12_11 - 4 * r14_11 * r14_44)) / rdiv;
		}
		//only y derivative
		else if (yedge_l && yedge_u) {

			//VERIFIED
			dvx_dx = ((dFxx / c11) - (dFzz / c33) * r13_11 - (dFyz / c44) * r14_11 - ((vy[ijk] - vy[cuINT3(i, j - 1, k)]) / h_m.y) * (r12_11 - r13_11 * r13_33 + 2 * r14_11 * r14_44)) / (1 - r13_11 * r13_33 - 2 * r14_11 * r14_44);
		}
		//no side derivatives : corner point. In this case all diagonal stress components set from external conditions, so derivatives not needed (set zero)
		else dvx_dx = 0.0;
	}

	//dvy/dy and dvz2_dy at vertex
	cuBReal dvy_dy = 0.0;
	cuBReal dvz2_dy = 0.0;

	//interior
	if (yedge_u && yedge_l) {

		dvy_dy = (vy[ijk] - vy[cuINT3(i, j - 1, k)]) / h_m.y;
		dvz2_dy = (vz2[ijk] - vz2[cuINT3(i, j - 1, k)]) / h_m.y;
	}
	//fixed face : Dirichlet value of zero for velocity derivative
	else if (yedge_l && yfixed_u) {

		dvy_dy = -vy[cuINT3(i, j - 1, k)] / (h_m.y / 2);
		dvz2_dy = -vz2[cuINT3(i, j - 1, k)] / (h_m.y / 2);
	}
	else if (yedge_u && yfixed_l) {

		dvy_dy = vy[ijk] / (h_m.y / 2);
		dvz2_dy = vz2[ijk] / (h_m.y / 2);
	}
	//free face
	else {

		//z derivative present (if x derivative not present, then dvx_dx is calculated above when only z derivative present, so no need to consider this again here)
		if (zedge_l && zedge_u) {

			//VERIFIED
			dvy_dy = ((dFyy / c11) + r14_11 * (dFyz / c44) - dvx_dx * (r12_11 + 2 * r14_11 * r14_44) - r13_11 * (vz[ijk] - vz[cuINT3(i, j, k - 1)]) / h_m.z) / (1 - 2 * r14_11 * r14_44);
			//VERIFIED
			//now that we have dvy_dy. this can be used (together with dvx_dx) in formula for dvz2_dy, even if x derivative not present
			dvz2_dy = (dFyz / c44) - 2 * r14_44 * (dvx_dx - dvy_dy) - (vy2[ijk] - vy2[cuINT3(i, j, k - 1)]) / h_m.z;
		}
		//only x derivative
		else if (xedge_l && xedge_u) {

			//VERIFIED
			dvy_dy = ((dFyy / c11) - r13_11 * (dFzz / c33) + r14_11 * (dFyz / c44) - dvx_dx * (r12_11 - r13_11 * r13_33 + 2 * r14_11 * r14_44)) / (1 - r13_11 * r13_33 - 2 * r14_11 * r14_44);

			//VERIFIED
			//need dvz2_dy also. In this case it's not possible to obtain it separately, but we only need (dvz2_dy + dvy2_dz), which is possible to obtain
			//this is : (dvz2_dy + dvy2_dz) = ((dFyy / c11) * 2 * r14_44 + (dFyz / c44) * (1 - r13_11 * r13_33) - (dFzz / c33) * 2 * r13_11*r14_44 - dvx_dx * 2 * r14_44 * (1 + r12_11 - 2 * r13_11*r13_33)) / (1 - r13_11*r13_33 - 2*r14_11*r14_44);
			//thus here set dvz2_dy as this value, and later when dvy2_dz is calculated below, set dvy2_dz to zero if only x derivative available
			dvz2_dy = ((dFyy / c11) * 2 * r14_44 + (dFyz / c44) * (1 - r13_11 * r13_33) - (dFzz / c33) * 2 * r13_11 * r14_44 - dvx_dx * 2 * r14_44 * (1 + r12_11 - 2 * r13_11 * r13_33)) / (1 - r13_11 * r13_33 - 2 * r14_11 * r14_44);

			//there are exceptions to this : if fixed z surface then dvy2_dz will be calculated using Dirichlet boundary condition, so use it here also
			if (zedge_u && zedge_l) {

				dvz2_dy -= (vy2[ijk] - vy2[cuINT3(i, j, k - 1)]) / h_m.z;
			}
			else if (zedge_l && zfixed_u) {

				dvz2_dy -= -vy2[cuINT3(i, j, k - 1)] / (h_m.z / 2);
			}
		}
		//no side derivatives : corner point. In this case all diagonal stress components set from external conditions, so derivatives not needed (set zero)
		//similarly dvz2_dy set to zero since this is also used to obtain sig_yz, but at corner point this is set from external conditions
		else {

			dvy_dy = 0.0;
			dvz2_dy = 0.0;
		}
	}

	//dvz/dz and dyz2_dz at vertex
	cuBReal dvz_dz = 0.0;
	cuBReal dvy2_dz = 0.0;

	//interior
	if (zedge_u && zedge_l) {

		dvz_dz = (vz[ijk] - vz[cuINT3(i, j, k - 1)]) / h_m.z;
		dvy2_dz = (vy2[ijk] - vy2[cuINT3(i, j, k - 1)]) / h_m.z;
	}
	//fixed face : Dirichlet value of zero for velocity derivative
	else if (zedge_l && zfixed_u) {

		dvz_dz = -vz[cuINT3(i, j, k - 1)] / (h_m.z / 2);
		dvy2_dz = -vy2[cuINT3(i, j, k - 1)] / (h_m.z / 2);
	}
	//fixed face : Dirichlet value of zero for velocity derivative
	else if (zedge_u && zfixed_l) {

		dvz_dz = vz[ijk] / (h_m.z / 2);
		dvy2_dz = vy2[ijk] / (h_m.z / 2);
	}
	//free face
	else {

		//VERIFIED
		//don't need to check if x and y derivatives present, as these are calculated above when obtaining dvy/dy, dvx/dx and dvz/dy
		//if this is a corner point then derivative values won't matter as values at corner points set from external conditions
		dvz_dz = (dFzz / c33 - r13_33 * (dvx_dx + dvy_dy));

		//VERIFIED
		//y derivative present
		if (yedge_l && yedge_u) dvy2_dz = (dFyz / c44 - 2 * r14_44 * (dvx_dx - dvy_dy)) - dvz2_dy;
		//if no y derivative present, then either only x is present (in which case keep dvy2_dz as zero - see comments for dvz2_dy calculated above when only x derivative is present), 
		//or else this is a corner point and dvy2_dz is not required, so still zero.
	}

	//update sdd and syz2 if not empty
	if ((xedge_u || xedge_l) && (yedge_u || yedge_l) && (zedge_u || zedge_l)) {

		//update sdd
		if ((!xedge_u && !xfixed_u) || (!xedge_l && !xfixed_l)) sdd[ijk].x = Fext_xface.x;
		else sdd[ijk].x += dT * (c11 * dvx_dx + c12 * dvy_dy + c13 * dvz_dz + c14 * (dvz2_dy + dvy2_dz) - dsxx_yy_dt_te + dsdd_dt_ms.x);

		bool yfree = (!yedge_u && !yfixed_u) || (!yedge_l && !yfixed_l);
		bool zfree = (!zedge_u && !zfixed_u) || (!zedge_l && !zfixed_l);

		if (yfree) {

			sdd[ijk].y = Fext_yface.y;
			syz2[ijk] = Fext_yface.z;
		}
		else sdd[ijk].y += dT * (c11 * dvy_dy + c12 * dvx_dx + c13 * dvz_dz - c14 * (dvz2_dy + dvy2_dz) - dsxx_yy_dt_te + dsdd_dt_ms.y);

		if (zfree) {

			sdd[ijk].z = Fext_zface.z;
			syz2[ijk] = Fext_zface.y;
		}
		else sdd[ijk].z += dT * (c33 * dvz_dz + c13 * (dvx_dx + dvy_dy) - dszz_dt_te + dsdd_dt_ms.z);

		//update syz2 if we don't have free y or z faces (otherwise it is set from external conditions above)
		if (!yfree && !zfree) syz2[ijk] += dT * (c44 * (dvz2_dy + dvy2_dz) + 2 * c14 * (dvx_dx - dvy_dy) + dsod_dt_ms.x);
	}
	else {

		sdd[ijk] = cuReal3();
		syz2[ijk] = 0.0;
	}

	//update sxy and sxz2
	if (i < n_m.i && j < n_m.j) {

		bool zface =
			(u_disp.is_not_empty(idx_u) ||
				(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y)));

		//both cells (distinct) present either side of the z face
		bool zstencil = k < n_m.z && u_disp.is_not_empty(idx_u) && k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y);

		if (zface) {

			cuBReal dvx_dy = (vx[cuINT3(i, j + 1, k)] - vx[ijk]) / h_m.y;
			cuBReal dvy_dx = (vy[cuINT3(i + 1, j, k)] - vy[ijk]) / h_m.x;

			cuBReal dvz2_dx = (vz2[cuINT3(i + 1, j, k)] - vz2[ijk]) / h_m.x;
			cuBReal dvx2_dz = 0.0;

			//interior
			if (zstencil) dvx2_dz = (vx2[ijk] - vx2[cuINT3(i, j, k - 1)]) / h_m.z;
			else {

				//fixed surfaces use Dirichlet
				if (zfixed_l) dvx2_dz = vx2[ijk] / (h_m.z / 2);
				else if (zfixed_u) dvx2_dz = -vx2[cuINT3(i, j, k - 1)] / (h_m.z / 2);
				//free surfaces
				else {

					//VERIFIED
					dvx2_dz = (dFzx / c44 - r14_44 * (dvy_dx + dvx_dy)) - dvz2_dx;
				}
			}

			sxy[ijk] += dT * (c66 * (dvx_dy + dvy_dx) + c14 * (dvz2_dx + dvx2_dz) + dsod_dt_ms.z);

			if (zstencil || zfixed_l || zfixed_u) sxz2[ijk] += dT * (c44 * (dvz2_dx + dvx2_dz) + c14 * (dvy_dx + dvx_dy) + dsod_dt_ms.y);
			//for free surface (z normal), set sxz2 directly from external condition
			else sxz2[ijk] = Fext_zface.x;
		}
		else {

			sxy[ijk] = 0.0;
			sxz2[ijk] = 0.0;
		}
	}

	//update sxz and sxy2
	if (i < n_m.i && k < n_m.k) {

		bool yface =
			(u_disp.is_not_empty(idx_u) ||
				(j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x)));

		//both cells (distinct) present either side of the y face
		bool ystencil = j < n_m.y && u_disp.is_not_empty(idx_u) && j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x);

		if (yface) {

			cuBReal dvx_dz = (vx[cuINT3(i, j, k + 1)] - vx[ijk]) / h_m.z;
			cuBReal dvz_dx = (vz[cuINT3(i + 1, j, k)] - vz[ijk]) / h_m.x;

			cuBReal dvy2_dx = (vy2[cuINT3(i + 1, j, k)] - vy2[ijk]) / h_m.x;
			cuBReal dvx2_dy = 0.0;

			//interior
			if (ystencil) dvx2_dy = (vx2[ijk] - vx2[cuINT3(i, j - 1, k)]) / h_m.y;
			else {

				//fixed surfaces use Dirichlet
				if (yfixed_l) dvx2_dy = vx2[ijk] / (h_m.y / 2);
				else if (yfixed_u) dvx2_dy = -vx2[cuINT3(i, j - 1, k)] / (h_m.y / 2);
				//free surfaces
				else {

					//VERIFIED
					dvx2_dy = (dFyx / c66 - r14_66 * (dvz_dx + dvx_dz)) - dvy2_dx;
				}
			}

			sxz[ijk] += dT * (c44 * (dvx_dz + dvz_dx) + c14 * (dvy2_dx + dvx2_dy) + dsod_dt_ms.y);

			if (ystencil || yfixed_l || yfixed_u)  sxy2[ijk] += dT * (c66 * (dvy2_dx + dvx2_dy) + c14 * (dvz_dx + dvx_dz) + dsod_dt_ms.z);
			//for free surface (y normal), set sxy2 directly from external condition
			else sxy2[ijk] = Fext_yface.x;
		}
		else {

			sxz[ijk] = 0.0;
			sxy2[ijk] = 0.0;
		}
	}

	//update syz and sdd2
	if (j < n_m.j && k < n_m.k) {

		bool xface =
			(u_disp.is_not_empty(idx_u) ||
				(i > 0 && u_disp.is_not_empty(idx_u - niend)));

		//both cells (distinct) present either side of the x face
		bool xstencil = i < n_m.x && u_disp.is_not_empty(idx_u) && i > 0 && u_disp.is_not_empty(idx_u - niend);

		if (xface) {

			cuBReal dvy_dz = (vy[cuINT3(i, j, k + 1)] - vy[ijk]) / h_m.z;
			cuBReal dvz_dy = (vz[cuINT3(i, j + 1, k)] - vz[ijk]) / h_m.y;
			cuBReal dvz2_dz = (vz2[cuINT3(i, j, k + 1)] - vz2[ijk]) / h_m.z;

			cuBReal dvy2_dy = (vy2[cuINT3(i, j + 1, k)] - vy2[ijk]) / h_m.y;
			cuBReal dvx2_dx = 0.0;

			//interior
			if (xstencil) dvx2_dx = (vx2[ijk] - vx2[cuINT3(i - 1, j, k)]) / h_m.x;
			else {

				//fixed surfaces use Dirichlet
				if (xfixed_l) dvx2_dx = vx2[ijk] / (h_m.x / 2);
				else if (xfixed_u) dvx2_dx = -vx2[cuINT3(i - 1, j, k)] / (h_m.x / 2);
				//free surfaces
				else {

					//VERIFIED
					dvx2_dx = (dFxx / c11 - r12_11 * dvy2_dy - r13_11 * dvz2_dz - r14_11 * (dvz_dy + dvy_dz));
				}
			}

			syz[ijk] += dT * (c44 * (dvy_dz + dvz_dy) + 2 * c14 * (dvx2_dx - dvy2_dy) + dsod_dt_ms.x);

			if (xstencil || xfixed_l || xfixed_u) sdd2[ijk].x += dT * (c11 * dvx2_dx + c12 * dvy2_dy + c13 * dvz2_dz + c14 * (dvz_dy + dvy_dz) - dsxx_yy_dt_te + dsdd_dt_ms.x);
			//for free surface (x normal), set sdd2.x directly from external condition
			else sdd2[ijk].x = Fext_xface.x;

			//the y and z components cannot be set from external conditions here
			sdd2[ijk].y += dT * (c11 * dvy2_dy + c12 * dvx2_dx + c13 * dvz2_dz - c14 * (dvz_dy + dvy_dz) - dsxx_yy_dt_te + dsdd_dt_ms.y);
			sdd2[ijk].z += dT * (c33 * dvz2_dz + c13 * (dvx2_dx + dvy2_dy) - dszz_dt_te + dsdd_dt_ms.z);
		}
		else {

			syz[ijk] = 0.0;
			sdd2[ijk] = cuReal3();
		}
	}

	///////////// MECHANICAL DISPLACEMENT

	//update mechanical displacement using velocity (remember u is cell-centred)
	if (i < n_m.i && j < n_m.j && k < n_m.k) {

		if (u_disp.is_not_empty(idx_u)) {

			//find velocity values cell-centred
			cuBReal vx_cc = (vx[ijk] + vx[ijk + cuINT3(0, 1, 0)] + vx[ijk + cuINT3(0, 0, 1)] + vx[ijk + cuINT3(0, 1, 1)]) / 4;
			cuBReal vy_cc = (vy[ijk] + vy[ijk + cuINT3(1, 0, 0)] + vy[ijk + cuINT3(0, 0, 1)] + vy[ijk + cuINT3(1, 0, 1)]) / 4;
			cuBReal vz_cc = (vz[ijk] + vz[ijk + cuINT3(1, 0, 0)] + vz[ijk + cuINT3(0, 1, 0)] + vz[ijk + cuINT3(1, 1, 0)]) / 4;

			u_disp[idx_u] += dT * cuReal3(vx_cc, vy_cc, vz_cc);
		}
		else u_disp[idx_u] = cuReal3();
	}
}

__global__ void Iterate_Elastic_Solver_Stress_FM_Trigonal_Kernel(
	ManagedMeshCUDA& cuMesh,
	MElastic_BoundaryCUDA* external_stress_surfaces, size_t num_surfaces,
	cuVEC<cuBReal>& vx, cuVEC<cuBReal>& vy, cuVEC<cuBReal>& vz,
	cuVEC<cuReal3>& sdd, cuVEC<cuBReal>& sxy, cuVEC<cuBReal>& sxz, cuVEC<cuBReal>& syz,
	cuVEC<cuBReal>& vx2, cuVEC<cuBReal>& vy2, cuVEC<cuBReal>& vz2,
	cuVEC<cuReal3>& sdd2, cuVEC<cuBReal>& sxy2, cuVEC<cuBReal>& sxz2, cuVEC<cuBReal>& syz2,
	cuBReal time, cuBReal dT,
	bool magnetostriction_enabled, bool thermoelasticity_enabled,
	cuBReal* Temp_previous, cuBReal magnetic_dT,
	ManagedDiffEqFMCUDA& cuDiffEq_FM)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;

	cuSZ3& n_m = u_disp.n;

	//kernel launch with size (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1) 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = idx % (n_m.i + 1);
	int j = (idx / (n_m.i + 1)) % (n_m.j + 1);
	int k = idx / ((n_m.i + 1)*(n_m.j + 1));

	if (idx < (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1)) {

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < n_m.i ? i : n_m.i - 1, j < n_m.j ? j : n_m.j - 1, k < n_m.k ? k : n_m.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * n_m.x + ijk_u.k * n_m.x * n_m.y;

		//needed for magnetostriction (time derivatives of stress due to magnetostriction)
		//xx, yy, zz
		cuReal3 dsdd_dt_ms = cuReal3();
		//yz, xz, xy
		cuReal3 dsod_dt_ms = cuReal3();
		if (magnetostriction_enabled) {

			cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
			cuVEC_VC<cuReal3>& M = *cuMesh.pM;

			int idx_M = M.position_to_cellidx(u_disp.cellidx_to_position(idx_u));

			if (M.is_not_empty(idx_M)) {

				cuBReal Ms = *cuMesh.pMs;
				cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
				cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
				cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
				cuReal2 mMEc = *cuMesh.pmMEc;
				cuReal2 mMEc2 = *cuMesh.pmMEc2;
				cuReal2 mMEc3 = *cuMesh.pmMEc3;
				cuMesh.update_parameters_mcoarse(idx_M, *cuMesh.pMs, Ms, *cuMesh.pmMEc, mMEc, *cuMesh.pmMEc2, mMEc2, *cuMesh.pmMEc3, mMEc3, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

				cuBReal& B21 = mMEc.i;
				cuBReal& B22 = mMEc.j;
				cuBReal& B3 = mMEc2.i;
				cuBReal& B4 = mMEc2.j;
				cuBReal& B14 = mMEc3.i;
				cuBReal& B34 = mMEc3.j;

				cuReal3 m = cuReal3(M[idx_M] * mcanis_ea1, M[idx_M] * mcanis_ea2, M[idx_M] * mcanis_ea3) / Ms;
				cuReal3 dM_dt = (M[idx_M] - (*cuDiffEq_FM.psM1)[idx_M]) / magnetic_dT;
				cuReal3 dm_dt = cuReal3(dM_dt * mcanis_ea1, dM_dt * mcanis_ea2, dM_dt * mcanis_ea3) / Ms;

				dsdd_dt_ms += 2 * B21 * m.z*dm_dt.z * (mcanis_ea1 + mcanis_ea2) + 2 * B22 * m.z*dm_dt.z * mcanis_ea3;
				dsdd_dt_ms += B3 * (m.x*dm_dt.x - m.y*dm_dt.y)*(mcanis_ea1 - mcanis_ea2);
				dsdd_dt_ms += B34 * (m.y*dm_dt.z + dm_dt.y*m.z)*(mcanis_ea1 - mcanis_ea2);

				dsod_dt_ms += B3 * (m.x*dm_dt.y + dm_dt.x*m.y)*mcanis_ea3;
				dsod_dt_ms += B4 * (m.x*dm_dt.z + dm_dt.x*m.z)*mcanis_ea2 + B4 * (m.y*dm_dt.z + dm_dt.y*m.z)*mcanis_ea1;
				dsod_dt_ms += B14 * (m.x*dm_dt.x - m.y*dm_dt.y) * mcanis_ea1 + B14 * (m.x*dm_dt.y + dm_dt.x*m.y)*mcanis_ea2;
				dsod_dt_ms += B34 * (m.x*dm_dt.z + dm_dt.x*m.z) * mcanis_ea3;
			}
		}

		//now solve the main part, with the possible addition of magnetostriction contribution
		Iterate_Elastic_Solver_Stress_Trigonal_CUDA(
			ijk, ijk_u, idx_u,
			cuMesh,
			external_stress_surfaces, num_surfaces,
			vx, vy, vz,
			sdd, sxy, sxz, syz,
			vx2, vy2, vz2,
			sdd2, sxy2, sxz2, syz2,
			time, dT,
			thermoelasticity_enabled,
			Temp_previous, magnetic_dT,
			dsdd_dt_ms, dsod_dt_ms);
	}
}

__global__ void Iterate_Elastic_Solver_Stress_AFM_Trigonal_Kernel(
	ManagedMeshCUDA& cuMesh,
	MElastic_BoundaryCUDA* external_stress_surfaces, size_t num_surfaces,
	cuVEC<cuBReal>& vx, cuVEC<cuBReal>& vy, cuVEC<cuBReal>& vz,
	cuVEC<cuReal3>& sdd, cuVEC<cuBReal>& sxy, cuVEC<cuBReal>& sxz, cuVEC<cuBReal>& syz,
	cuVEC<cuBReal>& vx2, cuVEC<cuBReal>& vy2, cuVEC<cuBReal>& vz2,
	cuVEC<cuReal3>& sdd2, cuVEC<cuBReal>& sxy2, cuVEC<cuBReal>& sxz2, cuVEC<cuBReal>& syz2,
	cuBReal time, cuBReal dT,
	bool magnetostriction_enabled, bool thermoelasticity_enabled,
	cuBReal* Temp_previous, cuBReal magnetic_dT,
	ManagedDiffEqAFMCUDA& cuDiffEq_AFM)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;

	cuSZ3& n_m = u_disp.n;

	//kernel launch with size (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1) 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = idx % (n_m.i + 1);
	int j = (idx / (n_m.i + 1)) % (n_m.j + 1);
	int k = idx / ((n_m.i + 1)*(n_m.j + 1));

	if (idx < (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1)) {

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < n_m.i ? i : n_m.i - 1, j < n_m.j ? j : n_m.j - 1, k < n_m.k ? k : n_m.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * n_m.x + ijk_u.k * n_m.x * n_m.y;

		//needed for magnetostriction (time derivatives of stress due to magnetostriction)
		//xx, yy, zz
		cuReal3 dsdd_dt_ms = cuReal3();
		//yz, xz, xy
		cuReal3 dsod_dt_ms = cuReal3();
		if (magnetostriction_enabled) {

			cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
			cuVEC_VC<cuReal3>& M = *cuMesh.pM;
			cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;

			int idx_M = M.position_to_cellidx(u_disp.cellidx_to_position(idx_u));

			if (M.is_not_empty(idx_M)) {

				cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
				cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
				cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
				cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
				cuReal2 mMEc = *cuMesh.pmMEc;
				cuReal2 mMEc2 = *cuMesh.pmMEc2;
				cuReal2 mMEc3 = *cuMesh.pmMEc3;
				cuMesh.update_parameters_mcoarse(idx_M, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pmMEc, mMEc, *cuMesh.pmMEc2, mMEc2, *cuMesh.pmMEc3, mMEc3, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

				cuBReal& B21 = mMEc.i;
				cuBReal& B22 = mMEc.j;
				cuBReal& B3 = mMEc2.i;
				cuBReal& B4 = mMEc2.j;
				cuBReal& B14 = mMEc3.i;
				cuBReal& B34 = mMEc3.j;

				cuReal3 mA = cuReal3(M[idx_M] * mcanis_ea1, M[idx_M] * mcanis_ea2, M[idx_M] * mcanis_ea3) / Ms_AFM.i;
				cuReal3 mB = cuReal3(M2[idx_M] * mcanis_ea1, M2[idx_M] * mcanis_ea2, M2[idx_M] * mcanis_ea3) / Ms_AFM.j;
				cuReal3 dM_dtA = (M[idx_M] - (*cuDiffEq_AFM.psM1)[idx_M]) / magnetic_dT;
				cuReal3 dm_dtA = cuReal3(dM_dtA * mcanis_ea1, dM_dtA * mcanis_ea2, dM_dtA * mcanis_ea3) / Ms_AFM.i;
				cuReal3 dM_dtB = (M2[idx_M] - (*cuDiffEq_AFM.psM1_2)[idx_M]) / magnetic_dT;
				cuReal3 dm_dtB = cuReal3(dM_dtB * mcanis_ea1, dM_dtB * mcanis_ea2, dM_dtB * mcanis_ea3) / Ms_AFM.j;

				dsdd_dt_ms += B21 * (mA.z*dm_dtA.z + mB.z*dm_dtB.z) * (mcanis_ea1 + mcanis_ea2) + B22 * (mA.z*dm_dtA.z + mB.z*dm_dtB.z) * mcanis_ea3;
				dsdd_dt_ms += 0.5 * B3 * (mA.x*dm_dtA.x + mB.x*dm_dtB.x - mA.y*dm_dtA.y - mB.y*dm_dtB.y)*(mcanis_ea1 - mcanis_ea2);
				dsdd_dt_ms += 0.5 * B34 * (mA.y*dm_dtA.z + dm_dtA.y*mA.z + mB.y*dm_dtB.z + dm_dtB.y*mB.z)*(mcanis_ea1 - mcanis_ea2);

				dsod_dt_ms += B3 * (mA.x*dm_dtA.y + mB.x*dm_dtB.y)*mcanis_ea3;
				dsod_dt_ms += 0.5 * B4 * (mA.x*dm_dtA.z + dm_dtA.x*mA.z + mB.x*dm_dtB.z + dm_dtB.x*mB.z)*mcanis_ea2 + 0.5 * B4 * (mA.y*dm_dtA.z + dm_dtA.y*mA.z + mB.y*dm_dtB.z + dm_dtB.y*mB.z)*mcanis_ea1;
				dsod_dt_ms += 0.5 * B14 * (mA.x*dm_dtA.x + mB.x*dm_dtB.x - mA.y*dm_dtA.y - mB.y*dm_dtB.y) * mcanis_ea1 + 0.5 * B14 * (mA.x*dm_dtA.y + dm_dtA.x*mA.y + mB.x*dm_dtB.y + dm_dtB.x*mB.y)*mcanis_ea2;
				dsod_dt_ms += 0.5 * B34 * (mA.x*dm_dtA.z + dm_dtA.x*mA.z + mB.x*dm_dtB.z + dm_dtB.x*mB.z) * mcanis_ea3;
			}
		}

		//now solve the main part, with the possible addition of magnetostriction contribution
		Iterate_Elastic_Solver_Stress_Trigonal_CUDA(
			ijk, ijk_u, idx_u,
			cuMesh,
			external_stress_surfaces, num_surfaces,
			vx, vy, vz,
			sdd, sxy, sxz, syz,
			vx2, vy2, vz2,
			sdd2, sxy2, sxz2, syz2,
			time, dT,
			thermoelasticity_enabled,
			Temp_previous, magnetic_dT,
			dsdd_dt_ms, dsod_dt_ms);
	}
}

__global__ void Iterate_Elastic_Solver_Stress_NoMS_Trigonal_Kernel(
	ManagedMeshCUDA& cuMesh,
	MElastic_BoundaryCUDA* external_stress_surfaces, size_t num_surfaces,
	cuVEC<cuBReal>& vx, cuVEC<cuBReal>& vy, cuVEC<cuBReal>& vz,
	cuVEC<cuReal3>& sdd, cuVEC<cuBReal>& sxy, cuVEC<cuBReal>& sxz, cuVEC<cuBReal>& syz,
	cuVEC<cuBReal>& vx2, cuVEC<cuBReal>& vy2, cuVEC<cuBReal>& vz2,
	cuVEC<cuReal3>& sdd2, cuVEC<cuBReal>& sxy2, cuVEC<cuBReal>& sxz2, cuVEC<cuBReal>& syz2,
	cuBReal time, cuBReal dT,
	bool thermoelasticity_enabled,
	cuBReal* Temp_previous, cuBReal magnetic_dT)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;

	cuSZ3& n_m = u_disp.n;

	//kernel launch with size (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1) 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i = idx % (n_m.i + 1);
	int j = (idx / (n_m.i + 1)) % (n_m.j + 1);
	int k = idx / ((n_m.i + 1)*(n_m.j + 1));

	if (idx < (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1)) {

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < n_m.i ? i : n_m.i - 1, j < n_m.j ? j : n_m.j - 1, k < n_m.k ? k : n_m.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * n_m.x + ijk_u.k * n_m.x * n_m.y;

		//now solve the main part without magnetostriction
		Iterate_Elastic_Solver_Stress_Trigonal_CUDA(
			ijk, ijk_u, idx_u,
			cuMesh,
			external_stress_surfaces, num_surfaces,
			vx, vy, vz,
			sdd, sxy, sxz, syz,
			vx2, vy2, vz2,
			sdd2, sxy2, sxz2, syz2,
			time, dT,
			thermoelasticity_enabled,
			Temp_previous, magnetic_dT,
			cuReal3(), cuReal3());
	}
}

//----------------------- Iterate_Elastic_Solver LAUNCHERS

//update stress for dT time increment
void MElasticCUDA::Iterate_Elastic_Solver_Stress_Trigonal(double dT, double magnetic_dT)
{
	size_t size = (pMeshCUDA->n_m.i + 1) * (pMeshCUDA->n_m.j + 1) * (pMeshCUDA->n_m.k + 1);

	//1b. Update stress
	if (magnetostriction_enabled) {

		if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

			Iterate_Elastic_Solver_Stress_AFM_Trigonal_Kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh, external_stress_surfaces_arr, external_stress_surfaces.size(),
					vx, vy, vz, sdd, sxy, sxz, syz,
					vx2, vy2, vz2, sdd2, sxy2, sxz2, syz2,
					pMeshCUDA->GetStageTime(), dT,
					magnetostriction_enabled, thermoelasticity_enabled,
					Temp_previous, magnetic_dT,
					reinterpret_cast<AFMeshCUDA*>(pMeshCUDA)->Get_ManagedDiffEqCUDA());
		}
		else if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

			Iterate_Elastic_Solver_Stress_FM_Trigonal_Kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh, external_stress_surfaces_arr, external_stress_surfaces.size(),
					vx, vy, vz, sdd, sxy, sxz, syz,
					vx2, vy2, vz2, sdd2, sxy2, sxz2, syz2,
					pMeshCUDA->GetStageTime(), dT,
					magnetostriction_enabled, thermoelasticity_enabled,
					Temp_previous, magnetic_dT,
					reinterpret_cast<FMeshCUDA*>(pMeshCUDA)->Get_ManagedDiffEqCUDA());
		}
	}
	else {

		Iterate_Elastic_Solver_Stress_NoMS_Trigonal_Kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh, external_stress_surfaces_arr, external_stress_surfaces.size(),
				vx, vy, vz, sdd, sxy, sxz, syz,
				vx2, vy2, vz2, sdd2, sxy2, sxz2, syz2,
				pMeshCUDA->GetStageTime(), dT,
				thermoelasticity_enabled,
				Temp_previous, magnetic_dT);
	}
}

//---------------------------------------------- Initial Conditions Launchers and Kernels

__global__ void Set_Initial_Stress_Trigonal_Kernel(
	ManagedMeshCUDA& cuMesh,
	cuVEC<cuReal3>& sdd, cuVEC<cuReal3>& sdd2,
	cuVEC<cuBReal>& sxy, cuVEC<cuBReal>& sxz, cuVEC<cuBReal>& syz,
	cuVEC<cuBReal>& sxy2, cuVEC<cuBReal>& sxz2, cuVEC<cuBReal>& syz2,
	bool magnetostriction_enabled, bool thermoelasticity_enabled, cuBReal& T_ambient)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;
	cuVEC_VC<cuReal3>& strain_odiag = *cuMesh.pstrain_odiag;

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

		cuINT3 ijk = cuINT3(i, j, k);

		//update sxx, syy, szz
		int niend = (ijk.i < n_m.i);
		int njend = (ijk.j < n_m.j);
		int nkend = (ijk.k < n_m.k);

		//check if required edges are present
		bool xedge_u =
			ijk.i < n_m.i &&
			(u_disp.is_not_empty(idx_u) ||
			(ijk.k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y)) ||
				(ijk.j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x)) ||
				(ijk.j > 0 && ijk.k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x)));

		bool xedge_l =
			ijk.i > 0 &&
			(u_disp.is_not_empty(idx_u - niend) ||
			(ijk.k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend)) ||
				(ijk.j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - niend)) ||
				(ijk.j > 0 && ijk.k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x - niend)));

		bool yedge_u =
			ijk.j < n_m.j &&
			(u_disp.is_not_empty(idx_u) ||
			(ijk.k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y)) ||
				(ijk.i > 0 && u_disp.is_not_empty(idx_u - niend)) ||
				(ijk.i > 0 && ijk.k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend)));

		bool yedge_l =
			ijk.j > 0 &&
			(u_disp.is_not_empty(idx_u - njend * n_m.x) ||
			(ijk.k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - njend * n_m.x)) ||
				(ijk.i > 0 && u_disp.is_not_empty(idx_u - niend - njend * n_m.x)) ||
				(ijk.i > 0 && ijk.k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y - niend - njend * n_m.x)));

		bool zedge_u =
			ijk.k < n_m.k &&
			(u_disp.is_not_empty(idx_u) ||
			(ijk.j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x)) ||
				(ijk.i > 0 && u_disp.is_not_empty(idx_u - niend)) ||
				(ijk.i > 0 && ijk.j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - niend)));

		bool zedge_l =
			ijk.k > 0 &&
			(u_disp.is_not_empty(idx_u - nkend * n_m.x*n_m.y) ||
			(ijk.j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - nkend * n_m.x*n_m.y)) ||
				(ijk.i > 0 && u_disp.is_not_empty(idx_u - niend - nkend * n_m.x*n_m.y)) ||
				(ijk.i > 0 && ijk.j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x - niend - nkend * n_m.x*n_m.y)));

		//xx, yy, zz
		cuReal3 Stress_MS_dd = cuReal3();
		//yz, xz, xy
		cuReal3 Stress_MS_od = cuReal3();
		if (magnetostriction_enabled) {

			cuVEC_VC<cuReal3>& M = *cuMesh.pM;
			cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;

			int idx_M = M.position_to_cellidx(u_disp.cellidx_to_position(idx_u));

			if (M.is_not_empty(idx_M)) {

				//MESH_ANTIFERROMAGNETIC
				if (M2.linear_size()) {

					cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
					cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
					cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
					cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
					cuReal2 mMEc = *cuMesh.pmMEc;
					cuReal2 mMEc2 = *cuMesh.pmMEc2;
					cuReal2 mMEc3 = *cuMesh.pmMEc3;
					cuMesh.update_parameters_mcoarse(idx_M, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pmMEc, mMEc, *cuMesh.pmMEc2, mMEc2, *cuMesh.pmMEc3, mMEc3, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

					cuBReal& B21 = mMEc.i;
					cuBReal& B22 = mMEc.j;
					cuBReal& B3 = mMEc2.i;
					cuBReal& B4 = mMEc2.j;
					cuBReal& B14 = mMEc3.i;
					cuBReal& B34 = mMEc3.j;

					cuReal3 mA = cuReal3(M[idx_M] * mcanis_ea1, M[idx_M] * mcanis_ea2, M[idx_M] * mcanis_ea3) / Ms_AFM.i;
					cuReal3 mB = cuReal3(M2[idx_M] * mcanis_ea1, M2[idx_M] * mcanis_ea2, M2[idx_M] * mcanis_ea3) / Ms_AFM.j;

					Stress_MS_dd += 0.5 * B21 * (mA.z*mA.z + mB.z*mB.z) * (mcanis_ea1 + mcanis_ea2) + 0.5 * B22 * (mA.z*mA.z + mB.z*mB.z) * mcanis_ea3;
					Stress_MS_dd += 0.25 * B3 * (mA.x*mA.x + mB.x*mB.x - mA.y*mA.y - mB.y*mB.y)*(mcanis_ea1 - mcanis_ea2);
					Stress_MS_dd += 0.5 * B34 * (mA.y*mA.z + mB.y*mB.z)*(mcanis_ea1 - mcanis_ea2);

					Stress_MS_od += 0.5 * B3 * (mA.x*mA.y + mB.x*mB.y)*mcanis_ea3;
					Stress_MS_od += 0.5 * B4 * (mA.x*mA.z + mB.x*mB.z)*mcanis_ea2 + 0.5 * B4 * (mA.y*mA.z + mB.y*mB.z)*mcanis_ea1;
					Stress_MS_od += 0.25 * B14 * (mA.x*mA.x + mB.x*mB.x - mA.y*mA.y - mB.y*mB.y) * mcanis_ea1 + 0.5 * B14 * (mA.x*mA.y + mB.x*mB.y)*mcanis_ea2;
					Stress_MS_od += 0.5 * B34 * (mA.x*mA.z + mB.x*mB.z) * mcanis_ea3;
				}
				//MESH_FERROMAGNETIC
				else {

					cuBReal Ms = *cuMesh.pMs;
					cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
					cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
					cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
					cuReal2 mMEc = *cuMesh.pmMEc;
					cuReal2 mMEc2 = *cuMesh.pmMEc2;
					cuReal2 mMEc3 = *cuMesh.pmMEc3;
					cuMesh.update_parameters_mcoarse(idx_M, *cuMesh.pMs, Ms, *cuMesh.pmMEc, mMEc, *cuMesh.pmMEc2, mMEc2, *cuMesh.pmMEc3, mMEc3, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

					cuBReal& B21 = mMEc.i;
					cuBReal& B22 = mMEc.j;
					cuBReal& B3 = mMEc2.i;
					cuBReal& B4 = mMEc2.j;
					cuBReal& B14 = mMEc3.i;
					cuBReal& B34 = mMEc3.j;

					cuReal3 m = cuReal3(M[idx_M] * mcanis_ea1, M[idx_M] * mcanis_ea2, M[idx_M] * mcanis_ea3) / Ms;

					Stress_MS_dd += B21 * m.z*m.z * (mcanis_ea1 + mcanis_ea2) + B22 * m.z*m.z * mcanis_ea3;
					Stress_MS_dd += 0.5 * B3 * (m.x*m.x - m.y*m.y)*(mcanis_ea1 - mcanis_ea2);
					Stress_MS_dd += B34 * m.y*m.z*(mcanis_ea1 - mcanis_ea2);

					Stress_MS_od += B3 * m.x*m.y*mcanis_ea3;
					Stress_MS_od += B4 * m.x*m.z*mcanis_ea2 + B4 * m.y*m.z*mcanis_ea1;
					Stress_MS_od += 0.5 * B14 * (m.x*m.x - m.y*m.y) * mcanis_ea1 + B14 * m.x*m.y*mcanis_ea2;
					Stress_MS_od += B34 * m.x*m.z * mcanis_ea3;
				}
			}
		}

		cuBReal Stress_Temp_xx_yy = 0.0;
		cuBReal Stress_Temp_zz = 0.0;
		if (thermoelasticity_enabled) {

			cuVEC_VC<cuBReal>& Temp = *cuMesh.pTemp;
			cuVEC_VC<cuBReal>& Temp_l = *cuMesh.pTemp_l;

			int idx_T = Temp.position_to_cellidx(u_disp.cellidx_to_position(idx_u));

			if (Temp.is_not_empty(idx_T)) {

				cuBReal thalpha = *cuMesh.pthalpha;
				cuReal3 cC = *cuMesh.pcC;
				cuReal3 cC3 = *cuMesh.pcC3;
				cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pcC, cC, *cuMesh.pcC3, cC3, *cuMesh.pthalpha, thalpha);

				cuBReal Temperature = 0.0;
				//for 2TM we need to use the lattice temperature
				if (Temp_l.linear_size()) Temperature = Temp_l[idx_T];
				else Temperature = Temp[idx_T];

				Stress_Temp_xx_yy = (cC.i + cC.j + cC3.j) * thalpha * (Temperature - T_ambient);
				Stress_Temp_zz = (cC3.i + 2 * cC3.j) * thalpha * (Temperature - T_ambient);
			}
		}

		//update sdd if not empty
		if ((xedge_u || xedge_l) && (yedge_u || yedge_l) && (zedge_u || zedge_l)) {

			sdd[ijk].x = -Stress_Temp_xx_yy + Stress_MS_dd.x;
			sdd[ijk].y = -Stress_Temp_xx_yy + Stress_MS_dd.y;
			sdd[ijk].z = -Stress_Temp_zz + Stress_MS_dd.z;
		}
		else sdd[ijk] = cuReal3();

		//update sdd and syz2
		if ((xedge_u || xedge_l) && (yedge_u || yedge_l) && (zedge_u || zedge_l)) {

			sdd[ijk].x = -Stress_Temp_xx_yy + Stress_MS_dd.x;
			sdd[ijk].y = -Stress_Temp_xx_yy + Stress_MS_dd.y;
			sdd[ijk].z = -Stress_Temp_zz + Stress_MS_dd.z;

			syz2[ijk] = Stress_MS_od.x;
		}
		else {

			sdd[ijk] = cuReal3();
			syz2[ijk] = 0.0;
		}

		//update sxy and sxz2
		if (i < n_m.i && j < n_m.j) {

			bool zface =
				(u_disp.is_not_empty(idx_u) ||
					(k > 0 && u_disp.is_not_empty(idx_u - nkend * n_m.x * n_m.y)));

			if (zface) {

				sxy[ijk] = Stress_MS_od.z;
				sxz2[ijk] = Stress_MS_od.y;
			}
			else {

				sxy[ijk] = 0.0;
				sxz2[ijk] = 0.0;
			}
		}

		//update sxz and sxy2
		if (i < n_m.i && k < n_m.k) {

			bool yface =
				(u_disp.is_not_empty(idx_u) ||
					(j > 0 && u_disp.is_not_empty(idx_u - njend * n_m.x)));

			if (yface) {

				sxz[ijk] = Stress_MS_od.y;
				sxy2[ijk] = Stress_MS_od.z;
			}
			else {

				sxz[ijk] = 0.0;
				sxy2[ijk] = 0.0;
			}
		}

		//update syz and sdd2
		if (j < n_m.j && k < n_m.k) {

			bool xface =
				(u_disp.is_not_empty(idx_u) ||
					(i > 0 && u_disp.is_not_empty(idx_u - niend)));

			if (xface) {

				syz[ijk] = Stress_MS_od.x;
				sdd2[ijk].x = -Stress_Temp_xx_yy + Stress_MS_dd.x;
				sdd2[ijk].x = -Stress_Temp_xx_yy + Stress_MS_dd.y;
				sdd2[ijk].x = -Stress_Temp_zz + Stress_MS_dd.z;
			}
			else {

				syz[ijk] = 0.0;
				sdd2[ijk] = cuReal3();
			}
		}
	}
}

//if thermoelasticity or magnetostriction is enabled, then initial stress must be set correctly
void MElasticCUDA::Set_Initial_Stress_Trigonal(void)
{
	if (!magnetostriction_enabled && !thermoelasticity_enabled) {

		sdd()->set(cuReal3());
		sdd2()->set(cuReal3());
		sxy()->set(0.0); sxz()->set(0.0); syz()->set(0.0);
		sxy2()->set(0.0); sxz2()->set(0.0); syz2()->set(0.0);
	}
	else {

		//reset for dT / dt computation
		if (thermoelasticity_enabled) {

			if (Temp_previous.resize(pMeshCUDA->n_t.dim())) Save_Current_Temperature();
		}

		//reset for dm / dt computation
		if (magnetostriction_enabled) pMeshCUDA->SaveMagnetization();

		size_t size = (pMeshCUDA->n_m.i + 1) * (pMeshCUDA->n_m.j + 1) * (pMeshCUDA->n_m.k + 1);

		Set_Initial_Stress_Trigonal_Kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh, sdd, sdd2, sxy, sxz, syz, sxy2, sxz2, syz2, magnetostriction_enabled, thermoelasticity_enabled, T_ambient);
	}
}

#endif

#endif
