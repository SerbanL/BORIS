#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "BorisCUDALib.h"

#include "ErrorHandler.h"

#include "ManagedMeshCUDA.h"
#include "MeshParamsControlCUDA.h"

#include "ManagedAtom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"

#include "TransportCUDA_Poisson_Spin_V.h"

#include "Transport_Defs.h"

class MeshCUDA;
class Atom_MeshCUDA;
class TransportBaseCUDA;

//This is held as a cu_obj managed class in TransportCUDA modules
//It provides methods and access to mesh data for use in cuVEC_VC methods.
//The methods have fixed names, e.g. Poisson_RHS is used by Poisson solvers to evaluate the r.h.s. of the Poisson equation
//This particular class is used for spin transport within the spin current solver.
class TransportCUDA_Spin_S_Funcs {

public:

	//spin transport solver type (see Transport_Defs.h) : copy of stsolve in TransportCUDA, but on the gpu so we can use it in device code
	int stsolve;

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedMeshCUDA* pcuMesh;

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedAtom_MeshCUDA* pcuaMesh;

	//need this (held in TransportCUDA) to access delsq V (Poisson_RHS) calculation
	TransportCUDA_Spin_V_Funcs* pPoisson_Spin_V;

	//for Poisson equations for S some values are fixed during relaxation, so pre-calculate them and store here to re-use.
	////points to cuVEC in TransportCUDA
	cuVEC<cuReal3>* pdelsq_S_fixed;

public:

	////////////////////////////////////////////////////
	//
	// CTOR/DTOR

	__host__ void construct_cu_obj(void) 
	{
		nullgpuptr(pcuMesh);
		nullgpuptr(pcuaMesh);
		nullgpuptr(pPoisson_Spin_V);
		nullgpuptr(pdelsq_S_fixed);
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

	//this evaluates the Poisson RHS when solving the Poisson equation on S
	__device__ cuReal3 Poisson_RHS(int idx)
	{
		cuReal3 delsq_S_RHS;

		cuVEC_VC<cuBReal>& V = (pcuMesh ? *pcuMesh->pV : *pcuaMesh->pV);
		cuVEC_VC<cuReal3>& S = (pcuMesh ? *pcuMesh->pS : *pcuaMesh->pS);
		cuVEC_VC<cuReal3>& M = (pcuMesh ? *pcuMesh->pM : *pcuaMesh->pM1);

		cuBReal l_sf = (pcuMesh ? *pcuMesh->pl_sf : *pcuaMesh->pl_sf);
		if (pcuMesh) pcuMesh->update_parameters_ecoarse(idx, *pcuMesh->pl_sf, l_sf);
		else pcuaMesh->update_parameters_ecoarse(idx, *pcuaMesh->pl_sf, l_sf);

		if (stsolve == STSOLVE_TUNNELING && pcuMesh) {

			cuBReal elecCond = *pcuMesh->pelecCond;
			pcuMesh->update_parameters_ecoarse(idx, *pcuMesh->pelecCond, elecCond);

			//Use elecCond to mark out metallic pinholes.
			//tunelling : l_sf tends to infinity
			if (elecCond > 0.0) delsq_S_RHS = (S[idx] / (l_sf * l_sf));
			return delsq_S_RHS;
		}

		//Contributions which apply equally in ferromagnetic and non-ferromagnetic meshes
		//longitudinal S decay term
		delsq_S_RHS = (S[idx] / (l_sf * l_sf));

		//Terms occuring only in magnetic meshes
		if (stsolve == STSOLVE_FERROMAGNETIC || stsolve == STSOLVE_FERROMAGNETIC_ATOM) {

			cuBReal l_ex = (pcuMesh ? *pcuMesh->pl_ex : *pcuaMesh->pl_ex);
			cuBReal l_ph = (pcuMesh ? *pcuMesh->pl_ph : *pcuaMesh->pl_ph);
			if (pcuMesh)pcuMesh->update_parameters_ecoarse(idx, *pcuMesh->pl_ex, l_ex, *pcuMesh->pl_ph, l_ph);
			else pcuaMesh->update_parameters_ecoarse(idx, *pcuaMesh->pl_ex, l_ex, *pcuaMesh->pl_ph, l_ph);

			int idx_M = M.position_to_cellidx(V.cellidx_to_position(idx));
			cuReal3 m = cu_normalize(M[idx_M]);

			//transverse S decay terms
			delsq_S_RHS += ((S[idx] ^ m) / (l_ex * l_ex) + (m ^ (S[idx] ^ m)) / (l_ph * l_ph));
		}

		//additional fixed contributions if needed
		if (pdelsq_S_fixed->linear_size()) delsq_S_RHS += (*pdelsq_S_fixed)[idx];

		return delsq_S_RHS;
	}

	//boundary differential of S for non-homogeneous Neumann boundary conditions
	__device__ cuVAL3<cuReal3> bdiff(int idx)
	{
		if (stsolve == STSOLVE_FERROMAGNETIC || stsolve == STSOLVE_FERROMAGNETIC_ATOM || stsolve == STSOLVE_TUNNELING) return cuReal33();

		//only pcuMesh here, but best to check
		if (pcuMesh) {

			cuVEC_VC<cuReal3>& E = *pcuMesh->pE;
			cuVEC_VC<cuBReal>& elC = *pcuMesh->pelC;

			cuBReal De = *pcuMesh->pDe;
			cuBReal SHA = *pcuMesh->pSHA;
			pcuMesh->update_parameters_ecoarse(idx, *pcuMesh->pSHA, SHA, *pcuMesh->pDe, De);

			return cu_epsilon3(E[idx]) * (SHA * elC[idx] * (cuBReal)MUB_E / De);
		}
		else return cuVAL3<cuReal3>();
	}
};

#endif

#endif