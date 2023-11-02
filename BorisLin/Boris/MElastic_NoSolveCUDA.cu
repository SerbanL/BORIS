#include "MElasticCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MELASTIC

#include "MeshCUDA.h"

__global__ void Set_Strain_From_Formula_Sd_Sod_Kernel(
	ManagedMeshCUDA& cuMesh,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sd_equation_x,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sd_equation_y,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sd_equation_z,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sod_equation_x,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sod_equation_y,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sod_equation_z,
	cuBReal time)
{
	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;
	cuVEC_VC<cuReal3>& strain_odiag = *cuMesh.pstrain_odiag;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < strain_diag.linear_size()) {

		if (strain_diag.is_not_empty(idx)) {

			cuReal3 relpos = strain_diag.cellidx_to_position(idx);
			strain_diag[idx] = cuReal3(
				Sd_equation_x.evaluate(relpos.x, relpos.y, relpos.z, time),
				Sd_equation_y.evaluate(relpos.x, relpos.y, relpos.z, time),
				Sd_equation_z.evaluate(relpos.x, relpos.y, relpos.z, time));

			strain_odiag[idx] = cuReal3(
				Sod_equation_x.evaluate(relpos.x, relpos.y, relpos.z, time),
				Sod_equation_y.evaluate(relpos.x, relpos.y, relpos.z, time),
				Sod_equation_z.evaluate(relpos.x, relpos.y, relpos.z, time));
		}
	}
}

__global__ void Set_Strain_From_Formula_Sd_Kernel(
	ManagedMeshCUDA& cuMesh,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sd_equation_x,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sd_equation_y,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sd_equation_z,
	cuBReal time)
{
	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;
	cuVEC_VC<cuReal3>& strain_odiag = *cuMesh.pstrain_odiag;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < strain_diag.linear_size()) {

		if (strain_diag.is_not_empty(idx)) {

			cuReal3 relpos = strain_diag.cellidx_to_position(idx);
			strain_diag[idx] = cuReal3(
				Sd_equation_x.evaluate(relpos.x, relpos.y, relpos.z, time),
				Sd_equation_y.evaluate(relpos.x, relpos.y, relpos.z, time),
				Sd_equation_z.evaluate(relpos.x, relpos.y, relpos.z, time));

			strain_odiag[idx] = cuReal3();
		}
	}
}

__global__ void Set_Strain_From_Formula_Sod_Kernel(
	ManagedMeshCUDA& cuMesh,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sod_equation_x,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sod_equation_y,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sod_equation_z,
	cuBReal time)
{
	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;
	cuVEC_VC<cuReal3>& strain_odiag = *cuMesh.pstrain_odiag;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < strain_diag.linear_size()) {

		if (strain_diag.is_not_empty(idx)) {

			cuReal3 relpos = strain_diag.cellidx_to_position(idx);
			strain_diag[idx] = cuReal3();

			strain_odiag[idx] = cuReal3(
				Sod_equation_x.evaluate(relpos.x, relpos.y, relpos.z, time),
				Sod_equation_y.evaluate(relpos.x, relpos.y, relpos.z, time),
				Sod_equation_z.evaluate(relpos.x, relpos.y, relpos.z, time));
		}
	}
}

//----------------------------------------------- Auxiliary

//Run-time auxiliary to set strain directly from user supplied text formulas
void MElasticCUDA::Set_Strain_From_Formula(void)
{
	if (Sd_equation.is_set() && Sod_equation.is_set()) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Set_Strain_From_Formula_Sd_Sod_Kernel <<< (pMeshCUDA->u_disp.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU),
				Sd_equation.get_x(mGPU), Sd_equation.get_y(mGPU), Sd_equation.get_z(mGPU),
				Sod_equation.get_x(mGPU), Sod_equation.get_y(mGPU), Sod_equation.get_z(mGPU),
				pMeshCUDA->GetStageTime());
		}
	}
	else if (Sd_equation.is_set()) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Set_Strain_From_Formula_Sd_Kernel <<< (pMeshCUDA->u_disp.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU),
				Sd_equation.get_x(mGPU), Sd_equation.get_y(mGPU), Sd_equation.get_z(mGPU),
				pMeshCUDA->GetStageTime());
		}
	}
	else if (Sod_equation.is_set()) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Set_Strain_From_Formula_Sod_Kernel <<< (pMeshCUDA->u_disp.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU),
				Sod_equation.get_x(mGPU), Sod_equation.get_y(mGPU), Sod_equation.get_z(mGPU),
				pMeshCUDA->GetStageTime());
		}
	}
}

//----------------------- UpdateField LAUNCHER

void MElasticCUDA::UpdateField(void)
{
	if (Sd_equation.is_set() || Sod_equation.is_set()) {

		//strain specified using a formula
		Set_Strain_From_Formula();
	}
}

//----------------------------------------------- Computational Helpers

__global__ void Calculate_Strain_Kernel(
	ManagedMeshCUDA& cuMesh,
	cuVEC_VC<cuReal3>& sdd,
	cuVEC_VC<cuBReal>& sxy, cuVEC_VC<cuBReal>& sxz, cuVEC_VC<cuBReal>& syz)
{
	cuVEC_VC<cuReal3>& u_disp = *cuMesh.pu_disp;
	cuVEC_VC<cuReal3>& strain_diag = *cuMesh.pstrain_diag;
	cuVEC_VC<cuReal3>& strain_odiag = *cuMesh.pstrain_odiag;

	cuReal3& h_m = u_disp.h;
	cuSZ3& n_m = u_disp.n;

	//kernel launch with size n_m.i * n_m.j * n_m.k 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < u_disp.linear_size()) {

		if (u_disp.is_not_empty(idx)) {

			//get all 9 first-order differentials of u
			cuReal33 grad_u = u_disp.grad_sided(idx);

			//diagonal components
			strain_diag[idx] = cuReal3(grad_u.x.x, grad_u.y.y, grad_u.z.z);

			//off-diagonal components (yz, xz, xy)
			strain_odiag[idx] = 0.5 * cuReal3(grad_u.y.z + grad_u.z.y, grad_u.x.z + grad_u.z.x, grad_u.x.y + grad_u.y.x);
		}
		else {

			strain_diag[idx] = cuReal3();
			strain_odiag[idx] = cuReal3();
		}
	}
}

void MElasticCUDA::Calculate_Strain(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Calculate_Strain_Kernel <<< (pMeshCUDA->u_disp.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh.get_deviceobject(mGPU), sdd.get_deviceobject(mGPU), sxy.get_deviceobject(mGPU), sxz.get_deviceobject(mGPU), syz.get_deviceobject(mGPU));
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && MONTE_CARLO == 1

#include "MeshParamsControlCUDA.h"

//Ferromagnetic
__device__ cuBReal ManagedMeshCUDA::Get_EnergyChange_FM_MElasticCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M = *pM;
	cuVEC_VC<cuReal3>& strain_diag = *pstrain_diag;
	cuVEC_VC<cuReal3>& strain_odiag = *pstrain_odiag;

	cuBReal Ms = *pMs;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	cuReal3 mcanis_ea2 = *pmcanis_ea2;
	cuReal3 mcanis_ea3 = *pmcanis_ea3;
	cuReal2 MEc = *pMEc;
	update_parameters_mcoarse(spin_index, *pMs, Ms, *pMEc, MEc, *pmcanis_ea1, mcanis_ea1, *pmcanis_ea2, mcanis_ea2, *pmcanis_ea3, mcanis_ea3);

	cuReal3 position = M.cellidx_to_position(spin_index);
	//xx, yy, zz
	cuReal3 Sd = strain_diag[position];
	//yz, xz, xy
	cuReal3 Sod = strain_odiag[position];

	//normalised magnetization
	//Magneto-elastic term here applicable for a cubic crystal. We use the mcanis_ea1 and mcanis_ea2 axes to fix the cubic lattice orientation, thus rotate the m, Sd and Sod vectors.

	Sd = cuReal3(Sd * mcanis_ea1, Sd * mcanis_ea2, Sd * mcanis_ea3);
	Sod = cuReal3(Sod * mcanis_ea1, Sod * mcanis_ea2, Sod * mcanis_ea3);

	auto Get_Energy = [&](cuReal3 M) -> cuBReal
	{
		cuReal3 m = cuReal3(M * mcanis_ea1, M * mcanis_ea2, M * mcanis_ea3) / Ms;

		cuReal3 Hmel_1 = (-2.0 * MEc.i / ((cuBReal)MU0 * Ms)) * cuReal3(
			m.x * Sd.x * mcanis_ea1.x + m.y * Sd.y * mcanis_ea2.x + m.z * Sd.z * mcanis_ea3.x,
			m.x * Sd.x * mcanis_ea1.y + m.y * Sd.y * mcanis_ea2.y + m.z * Sd.z * mcanis_ea3.y,
			m.x * Sd.x * mcanis_ea1.z + m.y * Sd.y * mcanis_ea2.z + m.z * Sd.z * mcanis_ea3.z);

		cuReal3 Hmel_2 = (-2.0 * MEc.j / ((cuBReal)MU0 * Ms)) * cuReal3(
			Sod.z * (mcanis_ea1.x * m.y + mcanis_ea2.x * m.x) + Sod.y * (mcanis_ea1.x * m.z + mcanis_ea3.x * m.x) + Sod.x * (mcanis_ea2.x * m.z + mcanis_ea3.x * m.y),
			Sod.z * (mcanis_ea1.y * m.y + mcanis_ea2.y * m.x) + Sod.y * (mcanis_ea1.y * m.z + mcanis_ea3.y * m.x) + Sod.x * (mcanis_ea2.y * m.z + mcanis_ea3.y * m.y),
			Sod.z * (mcanis_ea1.z * m.y + mcanis_ea2.z * m.x) + Sod.y * (mcanis_ea1.z * m.z + mcanis_ea3.z * m.x) + Sod.x * (mcanis_ea2.z * m.z + mcanis_ea3.z * m.y));

		return -(cuBReal)MU0 * M * (Hmel_1 + Hmel_2) / 2;
	};

	if (Mnew != cuReal3()) return M.h.dim() * (Get_Energy(Mnew) - Get_Energy(M[spin_index]));
	else return M.h.dim() * Get_Energy(M[spin_index]);
}

//Antiferromagnetic
//N/A

#endif