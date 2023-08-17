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

//This is held as a mcu_obj managed class in TransportCUDA modules
//It provides methods and access to mesh data for use in cuVEC_VC methods.
//The methods have fixed names, e.g. Poisson_RHS is used by Poisson solvers to evaluate the r.h.s. of the Poisson equation
//This particular class is used for charge transport only.
class TransportCUDA_V_Funcs {

public:

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedMeshCUDA* pcuMesh;

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedAtom_MeshCUDA* pcuaMesh;

	//flags set
	bool is_thermoelectric_mesh;

public:

	////////////////////////////////////////////////////
	//
	// CTOR/DTOR

	__host__ void construct_cu_obj(void) 
	{
		nullgpuptr(pcuMesh);
		nullgpuptr(pcuaMesh);
	}

	__host__ void destruct_cu_obj(void) {}

	////////////////////////////////////////////////////
	//
	// Configuration

	//for modules held in micromagnetic meshes
	BError set_pointers(MeshCUDA* pMeshCUDA, int device_idx);
	//for modules held in atomistic meshes
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA, int device_idx);

	__host__ void set_thermoelectric_mesh_flag(bool status) { set_gpu_value(is_thermoelectric_mesh, status); }

	////////////////////////////////////////////////////
	//
	// Runtime

	//this evaluates the Poisson RHS when solving the Poisson equation on V
	__device__ cuBReal Poisson_RHS(int idx)
	{
		cuVEC_VC<cuBReal>& V = (pcuMesh ? *pcuMesh->pV : *pcuaMesh->pV);
		cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);

		if (!is_thermoelectric_mesh) {

			//no thermoelectric effect
			return -(V.grad_diri(idx) * elC.grad_sided(idx)) / elC[idx];
		}
		else {

			//include thermoelectric effect with Seebeck coefficient : delsq V = -S delsq T, obtained from div J = 0, where J = -sigma(grad V + S * grad T).
			//here we ignore gradients in sigma and S

			cuBReal Sc = (pcuMesh ? *pcuMesh->pSc : *pcuaMesh->pSc);
			cuBReal thermCond = (pcuMesh ? *pcuMesh->pthermCond : *pcuaMesh->pthermCond);
			if (pcuMesh) pcuMesh->update_parameters_ecoarse(idx, *pcuMesh->pSc, Sc, *pcuMesh->pthermCond, thermCond);
			else pcuaMesh->update_parameters_ecoarse(idx, *pcuaMesh->pSc, Sc, *pcuaMesh->pthermCond, thermCond);

			cuVEC_VC<cuBReal>& Temp = (pcuMesh ? *pcuMesh->pTemp : *pcuaMesh->pTemp);

			//corresponding index in Temp
			int idx_temp = Temp.position_to_cellidx(V.cellidx_to_position(idx));

			return -Sc * Temp.delsq_robin(idx_temp, thermCond);
		}
	}

	//boundary differential of V for non-homogeneous Neumann boundary conditions (when thermoelectric effect is used)
	__device__ cuReal3 bdiff(int idx)
	{
		//Gradient of V normal to boundary is -S * grad T normal to boundary.
		//grad T normal to boundary obtained using Robin boundary condition as:

		//-K * grad T . n = heat flux normal to boundary = alpha(Tb - Ta), where alpha is Robin value, Ta ambient temperature and Tb temperature at boundary, K thermal conductivity
		//Thus grad V . n = S*alpha*(Tb-Ta) / K

		cuVEC_VC<cuBReal>& V = (pcuMesh ? *pcuMesh->pV : *pcuaMesh->pV);
		cuVEC_VC<cuBReal>& Temp = (pcuMesh ? *pcuMesh->pTemp : *pcuaMesh->pTemp);

		cuReal3 bdiff = cuReal3();
		cuReal3 shift = V.get_shift_to_emptycell(idx);

		if (!shift.IsNull()) {

			//corresponding index in Temp
			int idx_temp = Temp.position_to_cellidx(V.cellidx_to_position(idx));

			if (Temp.is_cmbnd(idx_temp)) {

				//at composite media boundaries (for Temp) cannot use Robin boundary for heat flux
				//instead we can approximate it using a sided differential in the cell just next to the boundary
				cuBReal Sc = (pcuMesh ? *pcuMesh->pSc : *pcuaMesh->pSc);
				if (pcuMesh) pcuMesh->update_parameters_ecoarse(idx, *pcuMesh->pSc, Sc);
				else pcuaMesh->update_parameters_ecoarse(idx, *pcuaMesh->pSc, Sc);

				bdiff = Sc * Temp.grad_sided(idx_temp);
			}
			else {

				//boundary, not a cmbnd. Use grad V . n = S*alpha*(Tb-Ta) / K
				cuBReal Sc = (pcuMesh ? *pcuMesh->pSc : *pcuaMesh->pSc);
				cuBReal thermCond = (pcuMesh ? *pcuMesh->pthermCond : *pcuaMesh->pthermCond);
				if (pcuMesh) pcuMesh->update_parameters_ecoarse(idx, *pcuMesh->pSc, Sc, *pcuMesh->pthermCond, thermCond);
				else pcuaMesh->update_parameters_ecoarse(idx, *pcuaMesh->pSc, Sc, *pcuaMesh->pthermCond, thermCond);

				bdiff = Temp.get_robin_value(V.cellidx_to_position(idx), shift) * Sc / thermCond;
				//at negative boundary inverse sign since heat flux normal also has opposite sign
				if (shift.x < 0) bdiff.x *= -1;
				if (shift.y < 0) bdiff.y *= -1;
				if (shift.z < 0) bdiff.z *= -1;
			}
		}

		return bdiff;
	}
};

#endif

#endif
