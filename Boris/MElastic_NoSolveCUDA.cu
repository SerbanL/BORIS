#include "MElasticCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MELASTIC

#include "BorisCUDALib.cuh"

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

		Set_Strain_From_Formula_Sd_Sod_Kernel <<< (pMeshCUDA->n_m.dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh,
			Sd_equation.get_x(), Sd_equation.get_y(), Sd_equation.get_z(),
			Sod_equation.get_x(), Sod_equation.get_y(), Sod_equation.get_z(),
			pMeshCUDA->GetStageTime());
	}
	else if (Sd_equation.is_set()) {

		Set_Strain_From_Formula_Sd_Kernel <<< (pMeshCUDA->n_m.dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh,
			Sd_equation.get_x(), Sd_equation.get_y(), Sd_equation.get_z(),
			pMeshCUDA->GetStageTime());
	}
	else if (Sod_equation.is_set()) {

		Set_Strain_From_Formula_Sod_Kernel <<< (pMeshCUDA->n_m.dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->cuMesh,
			Sod_equation.get_x(), Sod_equation.get_y(), Sod_equation.get_z(),
			pMeshCUDA->GetStageTime());
	}
}

//----------------------- UpdateField LAUNCHER

void MElasticCUDA::UpdateField(void)
{
	if (Sd_equation.is_set_vector() || Sod_equation.is_set_vector()) {

		//strain specified using a formula
		Set_Strain_From_Formula();
	}
}

//----------------------------------------------- Computational Helpers

__global__ void Calculate_Strain_Kernel(
	ManagedMeshCUDA& cuMesh,
	cuVEC<cuReal3>& sdd,
	cuVEC<cuBReal>& sxy, cuVEC<cuBReal>& sxz, cuVEC<cuBReal>& syz)
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
	Calculate_Strain_Kernel <<< (pMeshCUDA->n_m.dim() + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(pMeshCUDA->cuMesh, sdd, sxy, sxz, syz);
}

#endif

#endif