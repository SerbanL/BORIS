#include "MElasticCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MELASTIC

#include "BorisCUDALib.cuh"

#include "MeshDefs.h"

#include "ManagedDiffEqPolicyFMCUDA.h"
#include "ManagedDiffEqPolicyAFMCUDA.h"
#include "MeshParamsControlCUDA.h"

#include "MElastic_PolicyBoundariesCUDA.h"

//----------------------- Calculate_MElastic_Field KERNELS

__global__ void MElasticCUDA_Cubic_UpdateField_FM(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
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
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms, *cuMesh.pMEc, MEc, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

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

			cuReal3 Hmel_1 = (-2.0 * MEc.i / (MU0 * Ms)) * cuReal3(
				m.x*Sd.x*mcanis_ea1.x + m.y*Sd.y*mcanis_ea2.x + m.z*Sd.z*mcanis_ea3.x,
				m.x*Sd.x*mcanis_ea1.y + m.y*Sd.y*mcanis_ea2.y + m.z*Sd.z*mcanis_ea3.y,
				m.x*Sd.x*mcanis_ea1.z + m.y*Sd.y*mcanis_ea2.z + m.z*Sd.z*mcanis_ea3.z);

			cuReal3 Hmel_2 = (-2.0 * MEc.j / (MU0 * Ms)) * cuReal3(
				Sod.z * (mcanis_ea1.x*m.y + mcanis_ea2.x*m.x) + Sod.y * (mcanis_ea1.x*m.z + mcanis_ea3.x*m.x) + Sod.x * (mcanis_ea2.x*m.z + mcanis_ea3.x*m.y),
				Sod.z * (mcanis_ea1.y*m.y + mcanis_ea2.y*m.x) + Sod.y * (mcanis_ea1.y*m.z + mcanis_ea3.y*m.x) + Sod.x * (mcanis_ea2.y*m.z + mcanis_ea3.y*m.y),
				Sod.z * (mcanis_ea1.z*m.y + mcanis_ea2.z*m.x) + Sod.y * (mcanis_ea1.z*m.z + mcanis_ea3.z*m.x) + Sod.x * (mcanis_ea2.z*m.z + mcanis_ea3.z*m.y));

			Heff[idx] += Hmel_1 + Hmel_2;

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * M[idx] * (Hmel_1 + Hmel_2) / (2 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hmel_1 + Hmel_2;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MU0 * M[idx] * (Hmel_1 + Hmel_2) / 2;
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void MElasticCUDA_Cubic_UpdateField_AFM(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
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
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pMEc, MEc, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

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

			cuReal3 Hmel_1_A = (-2.0 * MEc.i / (MU0 * Ms_AFM.i)) * cuReal3(
				mA.x*Sd.x*mcanis_ea1.x + mA.y*Sd.y*mcanis_ea2.x + mA.z*Sd.z*mcanis_ea3.x,
				mA.x*Sd.x*mcanis_ea1.y + mA.y*Sd.y*mcanis_ea2.y + mA.z*Sd.z*mcanis_ea3.y,
				mA.x*Sd.x*mcanis_ea1.z + mA.y*Sd.y*mcanis_ea2.z + mA.z*Sd.z*mcanis_ea3.z);

			cuReal3 Hmel_2_A = (-2.0 * MEc.j / (MU0 * Ms_AFM.i)) * cuReal3(
				Sod.z * (mcanis_ea1.x*mA.y + mcanis_ea2.x*mA.x) + Sod.y * (mcanis_ea1.x*mA.z + mcanis_ea3.x*mA.x) + Sod.x * (mcanis_ea2.x*mA.z + mcanis_ea3.x*mA.y),
				Sod.z * (mcanis_ea1.y*mA.y + mcanis_ea2.y*mA.x) + Sod.y * (mcanis_ea1.y*mA.z + mcanis_ea3.y*mA.x) + Sod.x * (mcanis_ea2.y*mA.z + mcanis_ea3.y*mA.y),
				Sod.z * (mcanis_ea1.z*mA.y + mcanis_ea2.z*mA.x) + Sod.y * (mcanis_ea1.z*mA.z + mcanis_ea3.z*mA.x) + Sod.x * (mcanis_ea2.z*mA.z + mcanis_ea3.z*mA.y));

			cuReal3 Hmel_1_B = (-2.0 * MEc.i / (MU0 * Ms_AFM.j)) * cuReal3(
				mB.x*Sd.x*mcanis_ea1.x + mB.y*Sd.y*mcanis_ea2.x + mB.z*Sd.z*mcanis_ea3.x,
				mB.x*Sd.x*mcanis_ea1.y + mB.y*Sd.y*mcanis_ea2.y + mB.z*Sd.z*mcanis_ea3.y,
				mB.x*Sd.x*mcanis_ea1.z + mB.y*Sd.y*mcanis_ea2.z + mB.z*Sd.z*mcanis_ea3.z);

			cuReal3 Hmel_2_B = (-2.0 * MEc.j / (MU0 * Ms_AFM.j)) * cuReal3(
				Sod.z * (mcanis_ea1.x*mB.y + mcanis_ea2.x*mB.x) + Sod.y * (mcanis_ea1.x*mB.z + mcanis_ea3.x*mB.x) + Sod.x * (mcanis_ea2.x*mB.z + mcanis_ea3.x*mB.y),
				Sod.z * (mcanis_ea1.y*mB.y + mcanis_ea2.y*mB.x) + Sod.y * (mcanis_ea1.y*mB.z + mcanis_ea3.y*mB.x) + Sod.x * (mcanis_ea2.y*mB.z + mcanis_ea3.y*mB.y),
				Sod.z * (mcanis_ea1.z*mB.y + mcanis_ea2.z*mB.x) + Sod.y * (mcanis_ea1.z*mB.z + mcanis_ea3.z*mB.x) + Sod.x * (mcanis_ea2.z*mB.z + mcanis_ea3.z*mB.y));

			Heff[idx] += Hmel_1_A + Hmel_2_A;
			Heff2[idx] += Hmel_1_B + Hmel_2_B;

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = -(cuBReal)MU0 * (M[idx] * (Hmel_1_A + Hmel_2_A) + M2[idx] * (Hmel_1_B + Hmel_2_B)) / (2 * non_empty_cells);
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hmel_1_A + Hmel_2_A;
			if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[idx] = Hmel_1_B + Hmel_2_B;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MU0 * M[idx] * (Hmel_1_A + Hmel_2_A) / 2;
			if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[idx] = -(cuBReal)MU0 * M2[idx] * (Hmel_1_B + Hmel_2_B) / 2;
		}
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- Calculate_MElastic_Field LAUNCHER

//compute magnetoelastic effective field to use in magnetization equation.
void MElasticCUDA::Calculate_MElastic_Field_Cubic(void)
{
	//disabled by setting magnetoelastic coefficient to zero (also disabled in non-magnetic meshes)
	if (melastic_field_disabled) return;

	ZeroEnergy();

	if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

		//anti-ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				MElasticCUDA_Cubic_UpdateField_AFM <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				MElasticCUDA_Cubic_UpdateField_AFM <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
	else if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

		//ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				MElasticCUDA_Cubic_UpdateField_FM <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				MElasticCUDA_Cubic_UpdateField_FM <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
}

//----------------------- Iterate_Elastic_Solver KERNELS

__device__ void Iterate_Elastic_Solver_Stress_Cubic_CUDA(
	cuINT3 ijk, cuINT3 ijk_u, int idx_u,
	ManagedMeshCUDA& cuMesh,
	MElastic_BoundaryCUDA* external_stress_surfaces, size_t num_surfaces,
	cuVEC_VC<cuBReal>& vx, cuVEC_VC<cuBReal>& vy, cuVEC_VC<cuBReal>& vz,
	cuVEC_VC<cuReal3>& sdd, cuVEC_VC<cuBReal>& sxy, cuVEC_VC<cuBReal>& sxz, cuVEC_VC<cuBReal>& syz,
	cuBReal time, cuBReal dT,
	bool thermoelasticity_enabled,
	cuVEC<cuBReal>& Temp_previous, cuBReal magnetic_dT,
	cuReal3 dsdd_dt_ms, cuBReal dsxy_dt_ms, cuBReal dsxz_dt_ms, cuBReal dsyz_dt_ms)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;

	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	cuReal3 cC = *cuMesh.pcC;
	cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pcC, cC);
	cuBReal cr = cC.j / cC.i;

	//needed for thermoelasticity (includes time derivative of temperature)
	cuBReal dsdd_dt_te = 0.0;
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

			dsdd_dt_te = (cC.i + 2 * cC.j) * thalpha * (Temperature - Temp_previous[idx_T]) / magnetic_dT;
		}
	}

	cuBReal adT = dsdd_dt_te / cC.i;
	cuReal3 dms = dsdd_dt_ms / cC.i;

	//external forces on different faces (keep track separately in case an edge cell is excited simultaneously by 2 or more external forces
	cuBReal Fext_xface = 0.0, Fext_yface = 0.0, Fext_zface = 0.0;
	//time derivatives of forces on the different faces, divided by c11
	cuBReal dFx = 0.0, dFy = 0.0, dFz = 0.0;

	//is there an external force? If so, get it, otherwise it will be zero
	if (
		((ijk.i == 0 || ijk.i == n_m.i) && strain_diag.is_dirichlet_x(idx_u)) ||
		((ijk.j == 0 || ijk.j == n_m.j) && strain_diag.is_dirichlet_y(idx_u)) ||
		((ijk.k == 0 || ijk.k == n_m.k) && strain_diag.is_dirichlet_z(idx_u))) {

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
	int niend = (ijk.i < n_m.i);
	int njend = (ijk.j < n_m.j);
	int nkend = (ijk.k < n_m.k);

	//check if required edges are present
	bool xedge_u = ijk.i < n_m.i && u_disp.is_edge_x(idx_u);
	bool xedge_l = 
		(ijk.i > 0 && u_disp.is_edge_x(idx_u - niend)) ||
		(ijk.i == 0 &&
		(u_disp.is_halo_nx(idx_u) ||
		(ijk.k > 0 && u_disp.is_halo_nx(idx_u - nkend * n_m.x * n_m.y)) ||
		(ijk.j > 0 && u_disp.is_halo_nx(idx_u - njend * n_m.x)) ||
		(ijk.j > 0 && ijk.k > 0 && u_disp.is_halo_nx(idx_u - nkend * n_m.x * n_m.y - njend * n_m.x))));
	bool yedge_u = ijk.j < n_m.j && u_disp.is_edge_y(idx_u);
	bool yedge_l = ijk.j > 0 && u_disp.is_edge_y(idx_u - njend * n_m.x);
	bool zedge_u = ijk.k < n_m.k && u_disp.is_edge_z(idx_u);
	bool zedge_l = ijk.k > 0 && u_disp.is_edge_z(idx_u - nkend * n_m.x * n_m.y);

	//check for fixed faces at ends
	bool xfixed_l = (ijk.i == 0 && u_disp.is_dirichlet_px(idx_u));
	bool xfixed_u = (ijk.i == n_m.i && u_disp.is_dirichlet_nx(idx_u));
	bool yfixed_l = (ijk.j == 0 && u_disp.is_dirichlet_py(idx_u));
	bool yfixed_u = (ijk.j == n_m.j && u_disp.is_dirichlet_ny(idx_u));
	bool zfixed_l = (ijk.k == 0 && u_disp.is_dirichlet_pz(idx_u));
	bool zfixed_u = (ijk.k == n_m.k && u_disp.is_dirichlet_nz(idx_u));

	cuBReal dvx_dx = 0.0;

	//interior
	if (xedge_u && xedge_l) dvx_dx = (vx[ijk] - vx(cuINT3(ijk.i - 1, ijk.j, ijk.k))) / h_m.x;
	//fixed face : Dirichlet value of zero for velocity derivative
	else if (xedge_l && xfixed_u) {

		dvx_dx = -vx(cuINT3(ijk.i - 1, ijk.j, ijk.k)) / (h_m.x / 2);
	}
	else if (xedge_u && xfixed_l) {

		dvx_dx = vx[ijk] / (h_m.x / 2);
	}
	//free face
	else {

		//both side derivatives
		if (yedge_l && yedge_u && zedge_l && zedge_u) {

			dvx_dx = -cr * ((vy[ijk] - vy(cuINT3(ijk.i, ijk.j - 1, ijk.k))) / h_m.y + (vz[ijk] - vz(cuINT3(ijk.i, ijk.j, ijk.k - 1))) / h_m.z) + adT - dms.x + dFx;
		}
		//only z derivative
		else if (zedge_l && zedge_u) {

			//dvx = (dFx - cr*dFy - dmsx + cr*dmsy) / (1 - cr^2) + (adT - cr*dvz) / (1 + cr)
			dvx_dx = (adT - cr * (vz[ijk] - vz(cuINT3(ijk.i, ijk.j, ijk.k - 1))) / h_m.z) / (1 + cr) + (-dms.x + cr * dms.y + dFx - cr * dFy) / (1 - cr * cr);
		}
		//only y derivative
		else if (yedge_l && yedge_u) {

			//dvx = (dFx - cr*dFz - dmsx + cr*dmsz) / (1 - cr^2) + (adT - cr*dvy) / (1 + cr)
			dvx_dx = (adT - cr * (vy[ijk] - vy(cuINT3(ijk.i, ijk.j - 1, ijk.k))) / h_m.y) / (1 + cr) + (-dms.x + cr * dms.z + dFx - cr * dFz) / (1 - cr * cr);
		}
		//no side derivatives : corner point. In this case all diagonal stress components set from external conditions, so derivatives not needed (set zero)
		else dvx_dx = 0.0;
	}

	cuBReal dvy_dy = 0.0;

	//interior
	if (yedge_u && yedge_l) dvy_dy = (vy[ijk] - vy(cuINT3(ijk.i, ijk.j - 1, ijk.k))) / h_m.y;
	//fixed face : Dirichlet value of zero for velocity derivative
	else if (yedge_l && yfixed_u) {

		dvy_dy = -vy(cuINT3(ijk.i, ijk.j - 1, ijk.k)) / (h_m.y / 2);
	}
	else if (yedge_u && yfixed_l) {

		dvy_dy = vy[ijk] / (h_m.y / 2);
	}
	//free face
	else {

		//both side derivatives
		if (xedge_l && xedge_u && zedge_l && zedge_u) {

			dvy_dy = -cr * ((vx[ijk] - vx(cuINT3(ijk.i - 1, ijk.j, ijk.k))) / h_m.x + (vz[ijk] - vz(cuINT3(ijk.i, ijk.j, ijk.k - 1))) / h_m.z) + adT - dms.y + dFy;
		}
		//only z derivative
		else if (zedge_l && zedge_u) {

			//dvy = (dFy - cr*dFx - dmsy + cr*dmsx) / (1 - cr^2) + (adT - cr*dvz) / (1 + cr)
			dvy_dy = (adT - cr * (vz[ijk] - vz(cuINT3(ijk.i, ijk.j, ijk.k - 1))) / h_m.z) / (1 + cr) + (-dms.y + cr * dms.x + dFy - cr * dFx) / (1 - cr * cr);
		}
		//only x derivative
		else if (xedge_l && xedge_u) {

			//dvy = (dFy - cr*dFz - dmsy + cr*dmsz) / (1 - cr^2) + (adT - cr*dvx) / (1 + cr)
			dvy_dy = (adT - cr * (vx[ijk] - vx(cuINT3(ijk.i - 1, ijk.j, ijk.k))) / h_m.x) / (1 + cr) + (-dms.y + cr * dms.z + dFy - cr * dFz) / (1 - cr * cr);
		}
		//no side derivatives : corner point. In this case all diagonal stress components set from external conditions, so derivatives not needed (set zero)
		else dvy_dy = 0.0;
	}

	cuBReal dvz_dz = 0.0;

	//interior
	if (zedge_u && zedge_l) dvz_dz = (vz[ijk] - vz(cuINT3(ijk.i, ijk.j, ijk.k - 1))) / h_m.z;
	//fixed face : Dirichlet value of zero for velocity derivative
	else if (zedge_l && zfixed_u) {

		dvz_dz = -vz(cuINT3(ijk.i, ijk.j, ijk.k - 1)) / (h_m.z / 2);
	}
	//fixed face : Dirichlet value of zero for velocity derivative
	else if (zedge_u && zfixed_l) {

		dvz_dz = vz[ijk] / (h_m.z / 2);
	}
	//free face
	else {

		//both side derivatives
		if (xedge_l && xedge_u && yedge_l && yedge_u) {

			dvz_dz = -cr * ((vx[ijk] - vx(cuINT3(ijk.i - 1, ijk.j, ijk.k))) / h_m.x + (vy[ijk] - vy(cuINT3(ijk.i, ijk.j - 1, ijk.k))) / h_m.y) + adT - dms.z + dFz;
		}
		//only y derivative
		else if (yedge_l && yedge_u) {

			//dvz = (dFz - cr*dFx - dmsz + cr*dmsx) / (1 - cr^2) + (adT - cr*dvy) / (1 + cr)
			dvz_dz = (adT - cr * (vy[ijk] - vy(cuINT3(ijk.i, ijk.j - 1, ijk.k))) / h_m.y) / (1 + cr) + (-dms.z + cr * dms.x + dFz - cr * dFx) / (1 - cr * cr);
		}
		//only x derivative
		else if (xedge_l && xedge_u) {

			//dvz = (dFz - cr*dFy - dmsz + cr*dmsy) / (1 - cr^2) + (adT - cr*dvx) / (1 + cr)
			dvz_dz = (adT - cr * (vx[ijk] - vx(cuINT3(ijk.i - 1, ijk.j, ijk.k))) / h_m.x) / (1 + cr) + (-dms.z + cr * dms.y + dFz - cr * dFy) / (1 - cr * cr);
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
	else sdd[ijk] = cuReal3();

	//update sxy
	if (ijk.i < n_m.i && ijk.j < n_m.j) {

		bool zface = u_disp.is_face_z(idx_u);

		if (zface) {

			cuBReal dvx_dy = (vx(cuINT3(ijk.i, ijk.j + 1, ijk.k)) - vx[ijk]) / h_m.y;
			cuBReal dvy_dx = (vy(cuINT3(ijk.i + 1, ijk.j, ijk.k)) - vy[ijk]) / h_m.x;

			sxy[ijk] += dT * (cC.k * (dvx_dy + dvy_dx) + dsxy_dt_ms);
		}
		else sxy[ijk] = 0.0;
	}

	//update sxz
	if (ijk.i < n_m.i && ijk.k < n_m.k) {

		bool yface = u_disp.is_face_y(idx_u);

		if (yface) {

			cuBReal dvx_dz = (vx(cuINT3(ijk.i, ijk.j, ijk.k + 1)) - vx[ijk]) / h_m.z;
			cuBReal dvz_dx = (vz(cuINT3(ijk.i + 1, ijk.j, ijk.k)) - vz[ijk]) / h_m.x;

			sxz[ijk] += dT * (cC.k * (dvx_dz + dvz_dx) + dsxz_dt_ms);
		}
		else sxz[ijk] = 0.0;
	}

	//update syz
	if (ijk.j < n_m.j && ijk.k < n_m.k) {

		bool xface = u_disp.is_face_x(idx_u);

		if (xface) {

			cuBReal dvy_dz = (vy(cuINT3(ijk.i, ijk.j, ijk.k + 1)) - vy[ijk]) / h_m.z;
			cuBReal dvz_dy = (vz(cuINT3(ijk.i, ijk.j + 1, ijk.k)) - vz[ijk]) / h_m.y;

			syz[ijk] += dT * (cC.k * (dvy_dz + dvz_dy) + dsyz_dt_ms);
		}
		else syz[ijk] = 0.0;
	}

	//update mechanical displacement using velocity (remember u is cell-centred)
	if (ijk.i < n_m.i && ijk.j < n_m.j && ijk.k < n_m.k) {

		if (u_disp.is_not_empty(idx_u)) {

			//find velocity values cell-centred
			cuBReal vx_cc = (vx[ijk] + vx(ijk + cuINT3(0, 1, 0)) + vx(ijk + cuINT3(0, 0, 1)) + vx(ijk + cuINT3(0, 1, 1))) / 4;
			cuBReal vy_cc = (vy[ijk] + vy(ijk + cuINT3(1, 0, 0)) + vy(ijk + cuINT3(0, 0, 1)) + vy(ijk + cuINT3(1, 0, 1))) / 4;
			cuBReal vz_cc = (vz[ijk] + vz(ijk + cuINT3(1, 0, 0)) + vz(ijk + cuINT3(0, 1, 0)) + vz(ijk + cuINT3(1, 1, 0))) / 4;

			u_disp[idx_u] += dT * cuReal3(vx_cc, vy_cc, vz_cc);
		}
		else u_disp[idx_u] = cuReal3();
	}
}

__global__ void Iterate_Elastic_Solver_Stress_FM_Cubic_Kernel(
	ManagedMeshCUDA& cuMesh,
	MElastic_BoundaryCUDA* external_stress_surfaces, size_t num_surfaces,
	cuVEC_VC<cuBReal>& vx, cuVEC_VC<cuBReal>& vy, cuVEC_VC<cuBReal>& vz,
	cuVEC_VC<cuReal3>& sdd, cuVEC_VC<cuBReal>& sxy, cuVEC_VC<cuBReal>& sxz, cuVEC_VC<cuBReal>& syz,
	cuBReal time, cuBReal dT,
	bool magnetostriction_enabled, bool thermoelasticity_enabled,
	cuVEC<cuBReal>& Temp_previous, cuBReal magnetic_dT,
	ManagedDiffEqFMCUDA& cuDiffEq_FM)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;

	cuSZ3& n_m = u_disp.n;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//kernel launched with size sdd.device_size(mGPU). For a single GPU this has (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1) cells.
	//for multiple GPUs only last one has an extra cell along partition dimension, the other ones have same dimension as u_disp along partition dimension
	//this means the ijk index can always be used for reading and writing, but with +/-1 along partition direction need to use the () operator to read values
	int i = idx % sdd.n.i;
	int j = (idx / sdd.n.i) % sdd.n.j;
	int k = idx / (sdd.n.i * sdd.n.j);

	if (idx < sdd.n.dim()) {

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < n_m.i ? i : n_m.i - 1, j < n_m.j ? j : n_m.j - 1, k < n_m.k ? k : n_m.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * n_m.x + ijk_u.k * n_m.x * n_m.y;

		//needed for magnetostriction (time derivatives of stress due to magnetostriction)
		cuReal3 dsdd_dt_ms = cuReal3();
		cuBReal dsxy_dt_ms = 0.0, dsxz_dt_ms = 0.0, dsyz_dt_ms = 0.0;
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
				cuMesh.update_parameters_mcoarse(idx_M, *cuMesh.pMs, Ms, *cuMesh.pmMEc, mMEc, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

				cuReal3 m = cuReal3(M[idx_M] * mcanis_ea1, M[idx_M] * mcanis_ea2, M[idx_M] * mcanis_ea3) / Ms;
				cuReal3 dM_dt = (M[idx_M] - (*cuDiffEq_FM.psM1)[idx_M]) / magnetic_dT;
				cuReal3 dm_dt = cuReal3(dM_dt * mcanis_ea1, dM_dt * mcanis_ea2, dM_dt * mcanis_ea3) / Ms;

				dsdd_dt_ms = 2 * mMEc.i * cuReal3(
					m.x*dm_dt.x*mcanis_ea1.x + m.y*dm_dt.y*mcanis_ea2.x + m.z*dm_dt.z*mcanis_ea3.x,
					m.x*dm_dt.x*mcanis_ea1.y + m.y*dm_dt.y*mcanis_ea2.y + m.z*dm_dt.z*mcanis_ea3.y,
					m.x*dm_dt.x*mcanis_ea1.z + m.y*dm_dt.y*mcanis_ea2.z + m.z*dm_dt.z*mcanis_ea3.z);

				dsxy_dt_ms = mMEc.j * ((m.x*dm_dt.y + m.y*dm_dt.x)*mcanis_ea3.z + (m.x*dm_dt.z + m.z*dm_dt.x)*mcanis_ea2.z + (m.y*dm_dt.z + m.z*dm_dt.y)*mcanis_ea1.z);
				dsxz_dt_ms = mMEc.j * ((m.x*dm_dt.y + m.y*dm_dt.x)*mcanis_ea3.y + (m.x*dm_dt.z + m.z*dm_dt.x)*mcanis_ea2.y + (m.y*dm_dt.z + m.z*dm_dt.y)*mcanis_ea1.y);
				dsyz_dt_ms = mMEc.j * ((m.x*dm_dt.y + m.y*dm_dt.x)*mcanis_ea3.x + (m.x*dm_dt.z + m.z*dm_dt.x)*mcanis_ea2.x + (m.y*dm_dt.z + m.z*dm_dt.y)*mcanis_ea1.x);
			}
		}

		//now solve the main part, with the possible addition of magnetostriction contribution
		Iterate_Elastic_Solver_Stress_Cubic_CUDA(
			ijk, ijk_u, idx_u,
			cuMesh,
			external_stress_surfaces, num_surfaces,
			vx, vy, vz,
			sdd, sxy, sxz, syz,
			time, dT,
			thermoelasticity_enabled,
			Temp_previous, magnetic_dT,
			dsdd_dt_ms, dsxy_dt_ms, dsxz_dt_ms, dsyz_dt_ms);
	}
}

__global__ void Iterate_Elastic_Solver_Stress_AFM_Cubic_Kernel(
	ManagedMeshCUDA& cuMesh,
	MElastic_BoundaryCUDA* external_stress_surfaces, size_t num_surfaces,
	cuVEC_VC<cuBReal>& vx, cuVEC_VC<cuBReal>& vy, cuVEC_VC<cuBReal>& vz,
	cuVEC_VC<cuReal3>& sdd, cuVEC_VC<cuBReal>& sxy, cuVEC_VC<cuBReal>& sxz, cuVEC_VC<cuBReal>& syz,
	cuBReal time, cuBReal dT,
	bool magnetostriction_enabled, bool thermoelasticity_enabled,
	cuVEC<cuBReal>& Temp_previous, cuBReal magnetic_dT,
	ManagedDiffEqAFMCUDA& cuDiffEq_AFM)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;

	cuSZ3& n_m = u_disp.n;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//kernel launched with size sdd.device_size(mGPU). For a single GPU this has (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1) cells.
	//for multiple GPUs only last one has an extra cell along partition dimension, the other ones have same dimension as u_disp along partition dimension
	//this means the ijk index can always be used for reading and writing, but with +/-1 along partition direction need to use the () operator to read values
	int i = idx % sdd.n.i;
	int j = (idx / sdd.n.i) % sdd.n.j;
	int k = idx / (sdd.n.i * sdd.n.j);

	if (idx < sdd.n.dim()) {

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < n_m.i ? i : n_m.i - 1, j < n_m.j ? j : n_m.j - 1, k < n_m.k ? k : n_m.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * n_m.x + ijk_u.k * n_m.x * n_m.y;

		//needed for magnetostriction (time derivatives of stress due to magnetostriction)
		cuReal3 dsdd_dt_ms = cuReal3();
		cuBReal dsxy_dt_ms = 0.0, dsxz_dt_ms = 0.0, dsyz_dt_ms = 0.0;
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
				cuMesh.update_parameters_mcoarse(idx_M, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pmMEc, mMEc, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

				cuReal3 mA = cuReal3(M[idx_M] * mcanis_ea1, M[idx_M] * mcanis_ea2, M[idx_M] * mcanis_ea3) / Ms_AFM.i;
				cuReal3 mB = cuReal3(M2[idx_M] * mcanis_ea1, M2[idx_M] * mcanis_ea2, M2[idx_M] * mcanis_ea3) / Ms_AFM.j;
				cuReal3 dM_dtA = (M[idx_M] - (*cuDiffEq_AFM.psM1)[idx_M]) / magnetic_dT;
				cuReal3 dm_dtA = cuReal3(dM_dtA * mcanis_ea1, dM_dtA * mcanis_ea2, dM_dtA * mcanis_ea3) / Ms_AFM.i;
				cuReal3 dM_dtB = (M2[idx_M] - (*cuDiffEq_AFM.psM1_2)[idx_M]) / magnetic_dT;
				cuReal3 dm_dtB = cuReal3(dM_dtB * mcanis_ea1, dM_dtB * mcanis_ea2, dM_dtB * mcanis_ea3) / Ms_AFM.j;

				dsdd_dt_ms = mMEc.i * cuReal3(
					(mA.x*dm_dtA.x + mB.x*dm_dtB.x)*mcanis_ea1.x + (mA.y*dm_dtA.y + mB.y*dm_dtB.y)*mcanis_ea2.x + (mA.z*dm_dtA.z + mB.z*dm_dtB.z)*mcanis_ea3.x,
					(mA.x*dm_dtA.x + mB.x*dm_dtB.x)*mcanis_ea1.y + (mA.y*dm_dtA.y + mB.y*dm_dtB.y)*mcanis_ea2.y + (mA.z*dm_dtA.z + mB.z*dm_dtB.z)*mcanis_ea3.y,
					(mA.x*dm_dtA.x + mB.x*dm_dtB.x)*mcanis_ea1.z + (mA.y*dm_dtA.y + mB.y*dm_dtB.y)*mcanis_ea2.z + (mA.z*dm_dtA.z + mB.z*dm_dtB.z)*mcanis_ea3.z);

				dsxy_dt_ms = (mMEc.j / 2) * ((mA.x*dm_dtA.y + mA.y*dm_dtA.x + mB.x*dm_dtB.y + mB.y*dm_dtB.x)*mcanis_ea3.z + (mA.x*dm_dtA.z + mA.z*dm_dtA.x + mB.x*dm_dtB.z + mB.z*dm_dtB.x)*mcanis_ea2.z + (mA.y*dm_dtA.z + mA.z*dm_dtA.y + mB.y*dm_dtB.z + mB.z*dm_dtB.y)*mcanis_ea1.z);
				dsxz_dt_ms = (mMEc.j / 2) * ((mA.x*dm_dtA.y + mA.y*dm_dtA.x + mB.x*dm_dtB.y + mB.y*dm_dtB.x)*mcanis_ea3.y + (mA.x*dm_dtA.z + mA.z*dm_dtA.x + mB.x*dm_dtB.z + mB.z*dm_dtB.x)*mcanis_ea2.y + (mA.y*dm_dtA.z + mA.z*dm_dtA.y + mB.y*dm_dtB.z + mB.z*dm_dtB.y)*mcanis_ea1.y);
				dsyz_dt_ms = (mMEc.j / 2) * ((mA.x*dm_dtA.y + mA.y*dm_dtA.x + mB.x*dm_dtB.y + mB.y*dm_dtB.x)*mcanis_ea3.x + (mA.x*dm_dtA.z + mA.z*dm_dtA.x + mB.x*dm_dtB.z + mB.z*dm_dtB.x)*mcanis_ea2.x + (mA.y*dm_dtA.z + mA.z*dm_dtA.y + mB.y*dm_dtB.z + mB.z*dm_dtB.y)*mcanis_ea1.x);
			}
		}

		//now solve the main part, with the possible addition of magnetostriction contribution
		Iterate_Elastic_Solver_Stress_Cubic_CUDA(
			ijk, ijk_u, idx_u,
			cuMesh,
			external_stress_surfaces, num_surfaces,
			vx, vy, vz,
			sdd, sxy, sxz, syz,
			time, dT,
			thermoelasticity_enabled,
			Temp_previous, magnetic_dT,
			dsdd_dt_ms, dsxy_dt_ms, dsxz_dt_ms, dsyz_dt_ms);
	}
}

__global__ void Iterate_Elastic_Solver_Stress_NoMS_Cubic_Kernel(
	ManagedMeshCUDA& cuMesh,
	MElastic_BoundaryCUDA* external_stress_surfaces, size_t num_surfaces,
	cuVEC_VC<cuBReal>& vx, cuVEC_VC<cuBReal>& vy, cuVEC_VC<cuBReal>& vz,
	cuVEC_VC<cuReal3>& sdd, cuVEC_VC<cuBReal>& sxy, cuVEC_VC<cuBReal>& sxz, cuVEC_VC<cuBReal>& syz,
	cuBReal time, cuBReal dT,
	bool thermoelasticity_enabled,
	cuVEC<cuBReal>& Temp_previous, cuBReal magnetic_dT)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;

	cuSZ3& n_m = u_disp.n;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//kernel launched with size sdd.device_size(mGPU). For a single GPU this has (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1) cells.
	//for multiple GPUs only last one has an extra cell along partition dimension, the other ones have same dimension as u_disp along partition dimension
	//this means the ijk index can always be used for reading and writing, but with +/-1 along partition direction need to use the () operator to read values
	int i = idx % sdd.n.i;
	int j = (idx / sdd.n.i) % sdd.n.j;
	int k = idx / (sdd.n.i * sdd.n.j);

	if (idx < sdd.n.dim()) {

		cuINT3 ijk = cuINT3(i, j, k);

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < n_m.i ? i : n_m.i - 1, j < n_m.j ? j : n_m.j - 1, k < n_m.k ? k : n_m.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * n_m.x + ijk_u.k * n_m.x * n_m.y;

		//now solve the main part without magnetostriction
		Iterate_Elastic_Solver_Stress_Cubic_CUDA(
			ijk, ijk_u, idx_u,
			cuMesh,
			external_stress_surfaces, num_surfaces,
			vx, vy, vz,
			sdd, sxy, sxz, syz,
			time, dT,
			thermoelasticity_enabled,
			Temp_previous, magnetic_dT,
			cuReal3(), 0.0, 0.0, 0.0);
	}
}

//----------------------- Iterate_Elastic_Solver LAUNCHERS

//update stress for dT time increment
void MElasticCUDA::Iterate_Elastic_Solver_Stress_Cubic(double dT, double magnetic_dT)
{
	//use sdd device dimensions, since this has total size (pMeshCUDA->n_m.i + 1) * (pMeshCUDA->n_m.j + 1) * (pMeshCUDA->n_m.k + 1)

	vx.exchange_halos();
	vy.exchange_halos();
	vz.exchange_halos();

	//1b. Update stress
	if (magnetostriction_enabled) {

		if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Iterate_Elastic_Solver_Stress_AFM_Cubic_Kernel <<< (sdd.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), 
					external_stress_surfaces_arr(mGPU), external_stress_surfaces.size(),
					vx.get_deviceobject(mGPU), vy.get_deviceobject(mGPU), vz.get_deviceobject(mGPU), 
					sdd.get_deviceobject(mGPU), sxy.get_deviceobject(mGPU), sxz.get_deviceobject(mGPU), syz.get_deviceobject(mGPU),
					pMeshCUDA->GetStageTime(), dT,
					magnetostriction_enabled, thermoelasticity_enabled,
					Temp_previous.get_deviceobject(mGPU), magnetic_dT,
					reinterpret_cast<AFMeshCUDA*>(pMeshCUDA)->Get_ManagedDiffEqCUDA().get_deviceobject(mGPU));
			}
		}
		else if (pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Iterate_Elastic_Solver_Stress_FM_Cubic_Kernel <<< (sdd.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), 
					external_stress_surfaces_arr(mGPU), external_stress_surfaces.size(),
					vx.get_deviceobject(mGPU), vy.get_deviceobject(mGPU), vz.get_deviceobject(mGPU),
					sdd.get_deviceobject(mGPU), sxy.get_deviceobject(mGPU), sxz.get_deviceobject(mGPU), syz.get_deviceobject(mGPU),
					pMeshCUDA->GetStageTime(), dT,
					magnetostriction_enabled, thermoelasticity_enabled,
					Temp_previous.get_deviceobject(mGPU), magnetic_dT,
					reinterpret_cast<FMeshCUDA*>(pMeshCUDA)->Get_ManagedDiffEqCUDA().get_deviceobject(mGPU));
			}
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Iterate_Elastic_Solver_Stress_NoMS_Cubic_Kernel <<< (sdd.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU), 
				external_stress_surfaces_arr(mGPU), external_stress_surfaces.size(),
				vx.get_deviceobject(mGPU), vy.get_deviceobject(mGPU), vz.get_deviceobject(mGPU),
				sdd.get_deviceobject(mGPU), sxy.get_deviceobject(mGPU), sxz.get_deviceobject(mGPU), syz.get_deviceobject(mGPU),
				pMeshCUDA->GetStageTime(), dT,
				thermoelasticity_enabled,
				Temp_previous.get_deviceobject(mGPU), magnetic_dT);
		}
	}
}

//---------------------------------------------- Initial Conditions Launchers and Kernels

__global__ void Set_Initial_Stress_Cubic_Kernel(
	ManagedMeshCUDA& cuMesh,
	cuVEC_VC<cuReal3>& sdd,
	cuVEC_VC<cuBReal>& sxy, cuVEC_VC<cuBReal>& sxz, cuVEC_VC<cuBReal>& syz,
	bool magnetostriction_enabled, bool thermoelasticity_enabled, cuBReal& T_ambient)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;
	cuVEC_VC<cuReal3>& strain_odiag = *cuMesh.pstrain_odiag;

	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//kernel launched with size sdd.device_size(mGPU). For a single GPU this has (n_m.i + 1) * (n_m.j + 1) * (n_m.k + 1) cells.
	//for multiple GPUs only last one has an extra cell along partition dimension, the other ones have same dimension as u_disp along partition dimension
	//this means the ijk index can always be used for reading and writing, but when adding +1 along partition direction need to use the () operator to read values
	int i = idx % sdd.n.i;
	int j = (idx / sdd.n.i) % sdd.n.j;
	int k = idx / (sdd.n.i * sdd.n.j);

	if (idx < sdd.n.dim()) {

		//convert vertex index to cell-center index by capping maximum index size (use this to index u_disp)
		cuINT3 ijk_u = cuINT3(i < n_m.i ? i : n_m.i - 1, j < n_m.j ? j : n_m.j - 1, k < n_m.k ? k : n_m.k - 1);
		int idx_u = ijk_u.i + ijk_u.j * n_m.x + ijk_u.k * n_m.x * n_m.y;

		cuINT3 ijk = cuINT3(i, j, k);

		//update sxx, syy, szz
		int niend = (ijk.i < n_m.i);
		int njend = (ijk.j < n_m.j);
		int nkend = (ijk.k < n_m.k);

		//check if required edges are present
		bool xedge_u = ijk.i < n_m.i && u_disp.is_edge_x(idx_u);
		bool xedge_l =
			(ijk.i > 0 && u_disp.is_edge_x(idx_u - niend)) ||
			(ijk.i == 0 &&
			(u_disp.is_halo_nx(idx_u) ||
			(ijk.k > 0 && u_disp.is_halo_nx(idx_u - nkend * n_m.x * n_m.y)) ||
			(ijk.j > 0 && u_disp.is_halo_nx(idx_u - njend * n_m.x)) ||
			(ijk.j > 0 && ijk.k > 0 && u_disp.is_halo_nx(idx_u - nkend * n_m.x * n_m.y - njend * n_m.x))));
		bool yedge_u = ijk.j < n_m.j && u_disp.is_edge_y(idx_u);
		bool yedge_l = ijk.j > 0 && u_disp.is_edge_y(idx_u - njend * n_m.x);
		bool zedge_u = ijk.k < n_m.k && u_disp.is_edge_z(idx_u);
		bool zedge_l = ijk.k > 0 && u_disp.is_edge_z(idx_u - nkend * n_m.x * n_m.y);

		//check for fixed faces at ends
		bool xfixed_l = (ijk.i == 0 && u_disp.is_dirichlet_px(idx_u));
		bool xfixed_u = (ijk.i == n_m.i && u_disp.is_dirichlet_nx(idx_u));
		bool yfixed_l = (ijk.j == 0 && u_disp.is_dirichlet_py(idx_u));
		bool yfixed_u = (ijk.j == n_m.j && u_disp.is_dirichlet_ny(idx_u));
		bool zfixed_l = (ijk.k == 0 && u_disp.is_dirichlet_pz(idx_u));
		bool zfixed_u = (ijk.k == n_m.k && u_disp.is_dirichlet_nz(idx_u));

		cuReal3 Stress_MS_dd = cuReal3();
		cuBReal Stress_MS_xy = 0.0, Stress_MS_xz = 0.0, Stress_MS_yz = 0.0;
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
					cuMesh.update_parameters_mcoarse(idx_M, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pmMEc, mMEc, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

					cuReal3 mA = cuReal3(M[idx_M] * mcanis_ea1, M[idx_M] * mcanis_ea2, M[idx_M] * mcanis_ea3) / Ms_AFM.i;
					cuReal3 mB = cuReal3(M2[idx_M] * mcanis_ea1, M2[idx_M] * mcanis_ea2, M2[idx_M] * mcanis_ea3) / Ms_AFM.j;

					Stress_MS_dd = (mMEc.i / 2) * cuReal3(
						(mA.x*mA.x + mB.x*mB.x)*mcanis_ea1.x + (mA.y*mA.y + mB.y*mB.y)*mcanis_ea2.x + (mA.z*mA.z + mB.z*mB.z)*mcanis_ea3.x,
						(mA.x*mA.x + mB.x*mB.x)*mcanis_ea1.y + (mA.y*mA.y + mB.y*mB.y)*mcanis_ea2.y + (mA.z*mA.z + mB.z*mB.z)*mcanis_ea3.y,
						(mA.x*mA.x + mB.x*mB.x)*mcanis_ea1.z + (mA.y*mA.y + mB.y*mB.y)*mcanis_ea2.z + (mA.z*mA.z + mB.z*mB.z)*mcanis_ea3.z);

					Stress_MS_xy = (mMEc.j / 2) * ((mA.x*mA.y + mB.x*mB.y)*mcanis_ea3.z + (mA.x*mA.z + mB.x*mB.z)*mcanis_ea2.z + (mA.y*mA.z + mB.y*mB.z)*mcanis_ea1.z);
					Stress_MS_xz = (mMEc.j / 2) * ((mA.x*mA.y + mB.x*mB.y)*mcanis_ea3.y + (mA.x*mA.z + mB.x*mB.z)*mcanis_ea2.y + (mA.y*mA.z + mB.y*mB.z)*mcanis_ea1.y);
					Stress_MS_yz = (mMEc.j / 2) * ((mA.x*mA.y + mB.x*mB.y)*mcanis_ea3.x + (mA.x*mA.z + mB.x*mB.z)*mcanis_ea2.x + (mA.y*mA.z + mB.y*mB.z)*mcanis_ea1.x);
				}
				//MESH_FERROMAGNETIC
				else {

					cuBReal Ms = *cuMesh.pMs;
					cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
					cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
					cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
					cuReal2 mMEc = *cuMesh.pmMEc;
					cuMesh.update_parameters_mcoarse(idx_M, *cuMesh.pMs, Ms, *cuMesh.pmMEc, mMEc, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

					cuReal3 m = cuReal3(M[idx_M] * mcanis_ea1, M[idx_M] * mcanis_ea2, M[idx_M] * mcanis_ea3) / Ms;

					Stress_MS_dd = mMEc.i * cuReal3(
						m.x*m.x*mcanis_ea1.x + m.y*m.y*mcanis_ea2.x + m.z*m.z*mcanis_ea3.x,
						m.x*m.x*mcanis_ea1.y + m.y*m.y*mcanis_ea2.y + m.z*m.z*mcanis_ea3.y,
						m.x*m.x*mcanis_ea1.z + m.y*m.y*mcanis_ea2.z + m.z*m.z*mcanis_ea3.z);

					Stress_MS_xy = mMEc.j * (m.x*m.y*mcanis_ea3.z + m.x*m.z*mcanis_ea2.z + m.y*m.z*mcanis_ea1.z);
					Stress_MS_xz = mMEc.j * (m.x*m.y*mcanis_ea3.y + m.x*m.z*mcanis_ea2.y + m.y*m.z*mcanis_ea1.y);
					Stress_MS_yz = mMEc.j * (m.x*m.y*mcanis_ea3.x + m.x*m.z*mcanis_ea2.x + m.y*m.z*mcanis_ea1.x);
				}
			}
		}

		cuBReal Stress_Temp = 0.0;
		if (thermoelasticity_enabled) {

			cuVEC_VC<cuBReal>& Temp = *cuMesh.pTemp;
			cuVEC_VC<cuBReal>& Temp_l = *cuMesh.pTemp_l;

			int idx_T = Temp.position_to_cellidx(u_disp.cellidx_to_position(idx_u));

			if (Temp.is_not_empty(idx_T)) {

				cuBReal thalpha = *cuMesh.pthalpha;
				cuReal3 cC = *cuMesh.pcC;
				cuMesh.update_parameters_scoarse(idx_u, *cuMesh.pcC, cC, *cuMesh.pthalpha, thalpha);

				cuBReal Temperature = 0.0;
				//for 2TM we need to use the lattice temperature
				if (Temp_l.linear_size()) Temperature = Temp_l[idx_T];
				else Temperature = Temp[idx_T];

				Stress_Temp = (cC.i + 2 * cC.j) * thalpha * (Temperature - T_ambient);
			}
		}

		//update sdd if not empty
		if ((xedge_u || xedge_l) && (yedge_u || yedge_l) && (zedge_u || zedge_l)) {

			sdd[ijk].x = -Stress_Temp + Stress_MS_dd.x;
			sdd[ijk].y = -Stress_Temp + Stress_MS_dd.y;
			sdd[ijk].z = -Stress_Temp + Stress_MS_dd.z;
		}
		else sdd[ijk] = cuReal3();

		//update sxy
		if (ijk.i < n_m.i && ijk.j < n_m.j) {

			bool zface = u_disp.is_face_z(idx_u);

			if (zface) {

				sxy[ijk] = Stress_MS_xy;
			}
			else sxy[ijk] = 0.0;
		}

		//update sxz
		if (ijk.i < n_m.i && ijk.k < n_m.k) {

			bool yface = u_disp.is_face_y(idx_u);

			if (yface) {

				sxz[ijk] = Stress_MS_xz;
			}
			else sxz[ijk] = 0.0;
		}

		//update syz
		if (ijk.j < n_m.j && ijk.k < n_m.k) {

			bool xface = u_disp.is_face_x(idx_u);

			if (xface) {

				syz[ijk] = Stress_MS_yz;
			}
			else syz[ijk] = 0.0;
		}
	}
}

//if thermoelasticity or magnetostriction is enabled, then initial stress must be set correctly
void MElasticCUDA::Set_Initial_Stress_Cubic(void)
{
	if (!magnetostriction_enabled && !thermoelasticity_enabled) {

		sdd.set(cuReal3());
		sxy.set(0.0); sxz.set(0.0); syz.set(0.0);
	}
	else {

		//reset for dT / dt computation
		if (thermoelasticity_enabled) {

			if (Temp_previous.resize(pMeshCUDA->n_t.dim())) Save_Current_Temperature();
		}

		//reset for dm / dt computation
		if (magnetostriction_enabled) pMeshCUDA->SaveMagnetization();

		//use sdd device dimensions, since this has total size (pMeshCUDA->n_m.i + 1) * (pMeshCUDA->n_m.j + 1) * (pMeshCUDA->n_m.k + 1)

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Set_Initial_Stress_Cubic_Kernel <<< (sdd.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU), 
				sdd.get_deviceobject(mGPU), sxy.get_deviceobject(mGPU), sxz.get_deviceobject(mGPU), syz.get_deviceobject(mGPU), 
				magnetostriction_enabled, thermoelasticity_enabled, T_ambient(mGPU));
		}
	}
}

#endif

#endif
