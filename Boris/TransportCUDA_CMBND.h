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

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//This is held as a mcu_obj managed class in TransportCUDA modules with an associated policy class
//It provides methods and access to mesh data for use in cuVEC_VC cmbnd methods
//The a_func, b_func and diff2_func methods are used to set CMBND conditions based on the continuity of a quantity and a flux.
//If V is the potential, then the flux is the function f(V) = a_func + b_func * V', where the V' differential direction is perpendicular to the interface.
//TransportCUDA_CMBND_Pri is used for access on the primary side, where we are setting cmbnd values on the primary side for a given device.
//TransportCUDA_CMBND_Pri will be set for the same device.
class TransportCUDA_CMBND_Pri {

private:

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedMeshCUDA* pcuMesh;

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedAtom_MeshCUDA* pcuaMesh;

	//flags set
	bool is_thermoelectric_mesh;
	bool is_open_potential;

public:

	////////////////////////////////////////////////////
	//
	// CTOR/DTOR

	__host__ void construct_cu_obj(void) 
	{
		nullgpuptr(pcuMesh);
		nullgpuptr(pcuaMesh);
		set_gpu_value(is_thermoelectric_mesh, false);
		set_gpu_value(is_open_potential, false);
	}

	__host__ void destruct_cu_obj(void) {}

	////////////////////////////////////////////////////
	//
	// Configuration

	//not needed here, but needed in TransportCUDA_CMBND_Sec, and must work with same policy class
	void set_number_of_devices_mesh(int num_devices_cpu) {}
	void set_number_of_devices_amesh(int num_devices_cpu) {}

	BError set_pointers(MeshCUDA* pMeshCUDA, int device_idx);
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA, int device_idx);

	__host__ void set_thermoelectric_mesh_flag(bool status) { set_gpu_value(is_thermoelectric_mesh, status); }
	__host__ void set_open_potential_flag(bool status) { set_gpu_value(is_open_potential, status); }

	////////////////////////////////////////////////////
	//
	// Runtime

	//Charge transport only : V

	//For V only : V and Jc are continuous; Jc = -sigma * grad V = a + b * grad V -> a = 0 and b = -sigma taken at the interface
	//With thermoelectric effect this becomes:
	//1. No net current (e.g. no electrodes attached):
	//Jc = -sigma * grad V - sigma * Sc * grad T = a + b * grad V -> a = -sigma * Sc * grad T and b = -sigma taken at the interface
	//grad T normal to interface is given by Robin boundary condition : grad T . n = -alpha(Tb-Ta)/K
	//2. Net current generated (open potential condition):
	//Jc = sigma * Sc * grad T, so a = sigma * Sc * grad T, b = 0
	__device__ cuBReal a_func_pri(int cell1_idx, int cell2_idx, cuReal3 shift)
	{
		if (!is_thermoelectric_mesh) return 0.0;
		else {

			//include thermoelectric effect

			cuVEC_VC<cuBReal>& V = (pcuMesh ? *pcuMesh->pV : *pcuaMesh->pV);
			cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);
			cuVEC_VC<cuBReal>& Temp = (pcuMesh ? *pcuMesh->pTemp : *pcuaMesh->pTemp);

			cuBReal Sc = (pcuMesh ? *pcuMesh->pSc : *pcuaMesh->pSc);
			if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pSc, Sc);
			else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pSc, Sc);

			//normalized shift in normal direction to boundary: use * operator (dot product) with nshift to eliminate differentials orthogonal to the shift axis
			//do not use mod here as we need the temperature gradient to point in normal direction to boundary
			cuReal3 nshift = cu_normalize(shift);

			//corresponding index in Temp
			int idx_temp1 = Temp.position_to_cellidx(V.cellidx_to_position(cell1_idx));
			int idx_temp2 = Temp.position_to_cellidx(V.cellidx_to_position(cell2_idx));

			cuBReal T_grad1 = Temp.grad_sided(idx_temp1) * nshift;
			cuBReal T_grad2 = Temp.grad_sided(idx_temp2) * nshift;
			cuBReal T_grad = 1.5 * T_grad1 - 0.5 * T_grad2;

			//shift is from cell1 to cell2 so no need for minus sign adjustment
			return Sc * elC[cell1_idx] * T_grad;
		}
	}

	//For V only : V and Jc are continuous; Jc = -sigma * grad V = a + b * grad V -> a = 0 and b = -sigma taken at the interface	
	__device__ cuBReal b_func_pri(int cell1_idx, int cell2_idx)
	{
		if (is_thermoelectric_mesh && is_open_potential) return 0.0;
		else {

			cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);
			return -(1.5 * elC[cell1_idx] - 0.5 * elC[cell2_idx]);
		}
	}

	//second order differential of V at cells either side of the boundary; delsq V = -grad V * grad elC / elC
	__device__ cuBReal diff2_pri(int cell1_idx, cuReal3 shift)
	{
		if (!is_thermoelectric_mesh) {

			//normalized, positive shift: use * operator (dot product) with nshift to eliminate differentials orthogonal to the shift axis
			cuReal3 nshift = cu_mod(cu_normalize(shift));

			cuVEC_VC<cuBReal>& V = (pcuMesh ? *pcuMesh->pV : *pcuaMesh->pV);
			cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);

			//no thermoelectric effect
			return -((V.grad_diri(cell1_idx) * nshift) * (elC.grad_sided(cell1_idx) * nshift)) / elC[cell1_idx];
		}
		else {

			//include thermoelectric effect with Seebeck coefficient

			cuVEC_VC<cuBReal>& V = (pcuMesh ? *pcuMesh->pV : *pcuaMesh->pV);
			cuVEC_VC<cuBReal>& Temp = (pcuMesh ? *pcuMesh->pTemp : *pcuaMesh->pTemp);

			cuBReal Sc = (pcuMesh ? *pcuMesh->pSc : *pcuaMesh->pSc);
			if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pSc, Sc);
			else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pSc, Sc);

			//corresponding index in Temp
			int idx_temp = Temp.position_to_cellidx(V.cellidx_to_position(cell1_idx));

			return -Sc * Temp.delsq_neu(idx_temp);
		}
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//This is held as a mcu_obj managed class in TransportCUDA modules with an associated policy class
//It provides methods and access to mesh data for use in cuVEC_VC cmbnd methods
//The a_func, b_func and diff2_func methods are used to set CMBND conditions based on the continuity of a quantity and a flux.
//If V is the potential, then the flux is the function f(V) = a_func + b_func * V', where the V' differential direction is perpendicular to the interface.
//TransportCUDA_CMBND_Sec is used for access on the secondary side, where we are setting cmbnd values on the primary side for a given device.
//However, on the secondary side we may need access from a different device. TransportCUDA_CMBND_Sec allows access to data on any device using UVA.
//Thus TransportCUDA_CMBND_Sec differs from TransportCUDA_CMBND_Pri to allow access to all devices, and identify which device we need depending on the relpos_m1 passed in (which is relative to the entire cuVEC)
class TransportCUDA_CMBND_Sec {

private:

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedMeshCUDA** ppcuMesh;

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedAtom_MeshCUDA** ppcuaMesh;

	//flags set
	bool is_thermoelectric_mesh;
	bool is_open_potential;

	//number of devices available (i.e. size of above arrays)
	int num_devices;

	//the device in which this HeatCUDA_CMBND_Sec is held
	int curr_device;

public:

	////////////////////////////////////////////////////
	//
	// CTOR/DTOR

	__host__ void construct_cu_obj(void);
	__host__ void destruct_cu_obj(void);

	////////////////////////////////////////////////////
	//
	// Configuration

	void set_number_of_devices_mesh(int num_devices_cpu);
	void set_number_of_devices_amesh(int num_devices_cpu);

	BError set_pointers(MeshCUDA* pMeshCUDA, int device_idx);
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA, int device_idx);

	__host__ void set_thermoelectric_mesh_flag(bool status) { set_gpu_value(is_thermoelectric_mesh, status); }
	__host__ void set_open_potential_flag(bool status) { set_gpu_value(is_open_potential, status); }

	////////////////////////////////////////////////////
	//
	// Runtime - Auxiliary

	//relpos_m1 is relative to entire cuVEC, and will be contained on a certain device. From this return a coordinate relative to containing device, also updating device if needed.
	//int device is passed as an initial hint (0)
	__device__ cuReal3 global_relpos_to_device_relpos(const cuReal3& relpos_m1, int& device)
	{
		cuVEC_VC<cuBReal>& V_d = (ppcuMesh ? *ppcuMesh[device]->pV : *ppcuaMesh[device]->pV);

		//if you add rect.s of first device you get an absolute coordinate
		cuVEC_VC<cuBReal>& V_0 = (ppcuMesh ? *ppcuMesh[0]->pV : *ppcuaMesh[0]->pV);
		cuReal3 abspos_m1 = relpos_m1 + V_0.rect.s;

		//check hint device first
		if (V_d.rect.contains(abspos_m1)) {

			//return relative to hint device
			return abspos_m1 - V_d.rect.s;
		}

		//not in hint device. find which device contains it
		for (int device_idx = 0; device_idx < num_devices; device_idx++) {

			//skip hint device since already checked
			if (device_idx == device) continue;

			cuVEC_VC<cuBReal>& V = (ppcuMesh ? *ppcuMesh[device_idx]->pV : *ppcuaMesh[device_idx]->pV);

			if (V.rect.contains(abspos_m1)) {

				//update hint device
				device = device_idx;

				//return relative to containing device
				return abspos_m1 - V.rect.s;
			}
		}

		return cuReal3();
	}

	////////////////////////////////////////////////////
	//
	// Runtime

	//Charge transport only : V

	//For V only : V and Jc are continuous; Jc = -sigma * grad V = a + b * grad V -> a = 0 and b = -sigma taken at the interface
	//With thermoelectric effect this becomes:
	//1. No net current (e.g. no electrodes attached):
	//Jc = -sigma * grad V - sigma * Sc * grad T = a + b * grad V -> a = -sigma * Sc * grad T and b = -sigma taken at the interface
	//grad T normal to interface is given by Robin boundary condition : grad T . n = -alpha(Tb-Ta)/K
	//2. Net current generated (open potential condition):
	//Jc = sigma * Sc * grad T, so a = sigma * Sc * grad T, b = 0
	__device__ cuBReal a_func_sec(cuReal3 relpos_m1, cuReal3 shift, cuReal3 stencil)
	{
		if (!is_thermoelectric_mesh) return 0.0;
		else {

			int device = 0;
			cuReal3 devrelpos_m1 = global_relpos_to_device_relpos(relpos_m1, device);

			//include thermoelectric effect

			cuVEC_VC<cuBReal>& V = (ppcuMesh ? *ppcuMesh[device]->pV : *ppcuaMesh[device]->pV);
			cuVEC_VC<cuBReal>& elC = (ppcuMesh ? *ppcuMesh[device]->pelC : *ppcuaMesh[device]->pelC);
			cuVEC_VC<cuBReal>& Temp = (ppcuMesh ? *ppcuMesh[device]->pTemp : *ppcuaMesh[device]->pTemp);

			int cellm1_idx = V.position_to_cellidx(devrelpos_m1);

			//corresponding index in Temp
			int idx_temp1 = Temp.position_to_cellidx(devrelpos_m1);
			int idx_temp2 = Temp.position_to_cellidx(devrelpos_m1 + shift);

			cuBReal Sc = (ppcuMesh ? *ppcuMesh[curr_device]->pSc : *ppcuaMesh[curr_device]->pSc);
			if (ppcuMesh) ppcuMesh[device]->update_parameters_ecoarse(cellm1_idx, *ppcuMesh[device]->pSc, Sc);
			else ppcuaMesh[device]->update_parameters_ecoarse(cellm1_idx, *ppcuaMesh[device]->pSc, Sc);

			//normalized shift in normal direction to boundary: use * operator (dot product) with nshift to eliminate differentials orthogonal to the shift axis
			//do not use mod here as we need the temperature gradient to point in normal direction to boundary
			cuReal3 nshift = cu_normalize(shift);

			cuBReal T_grad1 = Temp.grad_sided(idx_temp1) * nshift;
			cuBReal T_grad2 = Temp.grad_sided(idx_temp2) * nshift;
			cuBReal T_grad = 1.5 * T_grad1 - 0.5 * T_grad2;

			//shift is from m1 to m2 cell, so use minus sign here if open potential mode
			if (is_open_potential) return -Sc * elC[cellm1_idx] * T_grad;
			else return Sc * elC[cellm1_idx] * T_grad;
		}
	}

	//For V only : V and Jc are continuous; Jc = -sigma * grad V = a + b * grad V -> a = 0 and b = -sigma taken at the interface	
	__device__ cuBReal b_func_sec(cuReal3 relpos_m1, cuReal3 shift, cuReal3 stencil)
	{
		if (is_thermoelectric_mesh && is_open_potential) return 0.0;
		else {

			int device = 0;
			cuReal3 devrelpos_m1 = global_relpos_to_device_relpos(relpos_m1, device);

			cuVEC_VC<cuBReal>& elC = (ppcuMesh ? *ppcuMesh[device]->pelC : *ppcuaMesh[device]->pelC);
			return -(1.5 * elC.weighted_average(devrelpos_m1, stencil) - 0.5 * elC.weighted_average(devrelpos_m1 + shift, stencil));
		}
	}

	//second order differential of V at cells either side of the boundary; delsq V = -grad V * grad elC / elC
	__device__ cuBReal diff2_pri(int cell1_idx, cuReal3 shift, int device)
	{
		if (!is_thermoelectric_mesh) {

			//normalized, positive shift: use * operator (dot product) with nshift to eliminate differentials orthogonal to the shift axis
			cuReal3 nshift = cu_mod(cu_normalize(shift));

			cuVEC_VC<cuBReal>& V = (ppcuMesh ? *ppcuMesh[device]->pV : *ppcuaMesh[device]->pV);
			cuVEC_VC<cuBReal>& elC = (ppcuMesh ? *ppcuMesh[device]->pelC : *ppcuaMesh[device]->pelC);

			//no thermoelectric effect
			return -((V.grad_diri(cell1_idx) * nshift) * (elC.grad_sided(cell1_idx) * nshift)) / elC[cell1_idx];
		}
		else {

			//include thermoelectric effect with Seebeck coefficient

			cuVEC_VC<cuBReal>& V = (ppcuMesh ? *ppcuMesh[device]->pV : *ppcuaMesh[device]->pV);
			cuVEC_VC<cuBReal>& Temp = (ppcuMesh ? *ppcuMesh[device]->pTemp : *ppcuaMesh[device]->pTemp);

			cuBReal Sc = (ppcuMesh ? *ppcuMesh[curr_device]->pSc : *ppcuaMesh[curr_device]->pSc);
			if (ppcuMesh) ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pSc, Sc);
			else ppcuaMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuaMesh[device]->pSc, Sc);

			//corresponding index in Temp
			int idx_temp = Temp.position_to_cellidx(V.cellidx_to_position(cell1_idx));

			return -Sc * Temp.delsq_neu(idx_temp);
		}
	}

	//second order differential of V at cells either side of the boundary; delsq V = -grad V * grad elC / elC
	__device__ cuBReal diff2_sec(cuReal3 relpos_m1, cuReal3 stencil, cuReal3 shift)
	{
		int device = 0;
		cuReal3 devrelpos_m1 = global_relpos_to_device_relpos(relpos_m1, device);

		cuVEC_VC<cuBReal>& V = (ppcuMesh ? *ppcuMesh[device]->pV : *ppcuaMesh[device]->pV);

		int cellm1_idx = V.position_to_cellidx(devrelpos_m1);
		return diff2_pri(cellm1_idx, shift, device);
	}
};

#endif

#endif