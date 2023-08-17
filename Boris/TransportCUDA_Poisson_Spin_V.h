#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "BorisCUDALib.h"

#include "ErrorHandler.h"

#include "ManagedMeshCUDA.h"
#include "ManagedAtom_MeshCUDA.h"

#include "MeshParamsControlCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"

#include "Transport_Defs.h"

class MeshCUDA;
class Atom_MeshCUDA;
class TransportBaseCUDA;

//This is held as a mcu_obj managed class in TransportCUDA modules
//It provides methods and access to mesh data for use in cuVEC_VC methods.
//The methods have fixed names, e.g. Poisson_RHS is used by Poisson solvers to evaluate the r.h.s. of the Poisson equation
//This particular class is used for charge transport within the spin current solver.
class TransportCUDA_Spin_V_Funcs {

public:

	//spin transport solver type (see Transport_Defs.h) : copy of stsolve in TransportCUDA, but on the gpu so we can use it in device code
	int stsolve;

	//micromagnetic version : managed mesh for access to all required mesh VECs and material parameters (if nullptr then not used)
	ManagedMeshCUDA* pcuMesh;

	//atomistic version : managed mesh for access to all required mesh VECs and material parameters (if nullptr then not used)
	ManagedAtom_MeshCUDA* pcuaMesh;

	//dM_dt VEC when we need to do vector calculus operations on it
	//points to cuVEC in TransportBaseCUDA
	cuVEC_VC<cuReal3>* pdM_dt;

	//for Poisson equations for V some values are fixed during relaxation, so pre-calculate them and store here to re-use.
	//points to cuVEC in TransportCUDA or Atom_TransportCUDA
	cuVEC<cuBReal>* pdelsq_V_fixed;

public:

	////////////////////////////////////////////////////
	//
	// CTOR/DTOR

	__host__ void construct_cu_obj(void) 
	{
		nullgpuptr(pcuMesh);
		nullgpuptr(pcuaMesh);
		nullgpuptr(pdM_dt);
		nullgpuptr(pdelsq_V_fixed);
	}

	__host__ void destruct_cu_obj(void) {}

	////////////////////////////////////////////////////
	//
	// Configuration

	//for modules held in micromagnetic meshes
	BError set_pointers(MeshCUDA* pMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int device_idx);
	//for modules held in atomistic meshes
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int device_idx);

	__host__ void set_stsolve(int stsolve_) { set_gpu_value(stsolve, stsolve_); }

	////////////////////////////////////////////////////
	//
	// Runtime

	//this evaluates the Poisson RHS when solving the Poisson equation on V (in the context of full spin solver)
	__device__ cuBReal Poisson_RHS(int idx)
	{
		cuVEC_VC<cuBReal>& V = (pcuMesh ? *pcuMesh->pV : *pcuaMesh->pV);
		cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);
		cuVEC_VC<cuReal3>& S = (pcuMesh ? *pcuMesh->pS : *pcuaMesh->pS);
		cuVEC_VC<cuReal3>& M = (pcuMesh ? *pcuMesh->pM : *pcuaMesh->pM1);

		if (stsolve == STSOLVE_TUNNELING) return -(V.grad_diri(idx) * elC.grad_sided(idx)) / elC[idx];

		cuBReal iSHA = (pcuMesh ? *pcuMesh->piSHA : 0.0);
		cuBReal De = (pcuMesh ? *pcuMesh->pDe : *pcuaMesh->pDe);

		//The Poisson solver calls this method to evaluate the RHS of this equation
		cuBReal value = 0.0;

		if (stsolve == STSOLVE_NORMALMETAL || stsolve == STSOLVE_NONE) {

			//non-magnetic mesh

			if (cuIsZ(iSHA) || stsolve == STSOLVE_NONE) {

				//1. no iSHE contribution.
				value = -(V.grad_diri(idx) * elC.grad_sided(idx)) / elC[idx];
			}
			else {

				//pcuMesh only, not pcuaMesh, but best to check
				if (pcuMesh) {

					pcuMesh->update_parameters_ecoarse(idx, *pcuMesh->pDe, De, *pcuMesh->piSHA, iSHA);

					//1. iSHE enabled, must use non-homogeneous Neumann boundary condition for grad V -> Note homogeneous Neumann boundary conditions apply when calculating S differentials here (due to Jc.n = 0 at boundaries)
					value = -(V.grad_diri_nneu(idx, (iSHA * De / ((cuBReal)MUB_E * elC[idx])) * S.curl_neu(idx)) * elC.grad_sided(idx)) / elC[idx];
				}
			}
		}
		else {

			//magnetic mesh

			//homogeneous Neumann boundary condition applies to V in magnetic meshes
			cuReal3 grad_V = V.grad_diri(idx);

			//1. principal term : always present
			value = -(grad_V * elC.grad_sided(idx)) / elC[idx];

			cuBReal the_eff = (pcuMesh ? *pcuMesh->pthe_eff : *pcuaMesh->pthe_eff);

			//2. topological Hall effect contribution
			if (cuIsNZ(the_eff)) {

				cuBReal the_eff0 = (pcuMesh ? pcuMesh->pthe_eff->get0() : pcuaMesh->pthe_eff->get0());
				cuBReal P = (pcuMesh ? *pcuMesh->pP : *pcuaMesh->pP);
				cuBReal n_density = (pcuMesh ? *pcuMesh->pn_density : *pcuaMesh->pn_density);

				if (pcuMesh) pcuMesh->update_parameters_ecoarse(idx, *pcuMesh->pP, P, *pcuMesh->pn_density, n_density);
				else pcuaMesh->update_parameters_ecoarse(idx, *pcuaMesh->pP, P, *pcuaMesh->pn_density, n_density);

				int idx_M = M.position_to_cellidx(V.cellidx_to_position(idx));
				cuReal3 m = cu_normalize(M[idx_M]);

				cuReal33 grad_m = cu_normalize(M.grad_neu(idx_M), M[idx_M]);
				cuReal3 dx_m = grad_m.x;
				cuReal3 dy_m = grad_m.y;
				cuReal3 dxy_m = cu_normalize(M.dxy_neu(idx_M), M[idx_M]);
				cuReal3 dxx_m = cu_normalize(M.dxx_neu(idx_M), M[idx_M]);
				cuReal3 dyy_m = cu_normalize(M.dyy_neu(idx_M), M[idx_M]);

				cuReal3 B_the = cuReal3(
					((dxy_m ^ dy_m) + (dx_m ^ dyy_m)) * m,
					-1.0 * ((dxx_m ^ dy_m) + (dx_m ^ dxy_m)) * m,
					0.0);

				value -= (the_eff0 * P * elC[idx] * (cuBReal)HBAR_E / ((cuBReal)ECHARGE * n_density)) * (grad_V * B_the);
			}
		}

		//additional fixed contributions if needed (e.g. CPP-GMR and charge pumping)
		if (pdelsq_V_fixed->linear_size()) value += (*pdelsq_V_fixed)[idx];

		return value;
	}

	//boundary differential of V for non-homogeneous Neumann boundary conditions
	__device__ cuVAL3<cuBReal> bdiff(int idx)
	{
		if (stsolve == STSOLVE_FERROMAGNETIC || stsolve == STSOLVE_FERROMAGNETIC_ATOM) return cuReal3();

		//pcuMesh only, but best to check
		if (pcuMesh) {

			cuVEC_VC<cuBReal>& elC = *pcuMesh->pelC;
			cuVEC_VC<cuReal3>& S = *pcuMesh->pS;

			cuBReal De = *pcuMesh->pDe;
			cuBReal iSHA = *pcuMesh->piSHA;
			pcuMesh->update_parameters_ecoarse(idx, *pcuMesh->piSHA, iSHA, *pcuMesh->pDe, De);

			return (iSHA * De / ((cuBReal)MUB_E * elC[idx])) * S.curl_neu(idx);
		}
		else return cuVAL3<cuBReal>();
	}
};

#endif

#endif
