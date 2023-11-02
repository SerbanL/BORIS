#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_HEAT

#include "BorisCUDALib.h"

#include "ErrorHandler.h"

#include "ManagedMeshCUDA.h"
#include "ManagedAtom_MeshCUDA.h"

#include "MeshParamsControlCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"

class MeshCUDA;
class Atom_MeshCUDA;

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//This is held as a mcu_obj managed class in HeatCUDA modules with an associated policy class
//It provides methods and access to mesh data for use in cuVEC_VC cmbnd methods
//The a_func, b_func and diff2_func methods are used to set CMBND conditions based on the continuity of a quantity and a flux.
//If V is the potential, then the flux is the function f(V) = a_func + b_func * V', where the V' differential direction is perpendicular to the interface.
//HeatCUDA_CMBND_Pri is used for access on the primary side, where we are setting cmbnd values on the primary side for a given device.
//HeatCUDA_CMBND_Pri will be set for the same device.
class HeatCUDA_CMBND_Pri {

private:

	ManagedMeshCUDA* pcuMesh;
	ManagedAtom_MeshCUDA* pcuaMesh;

	//Q equation and time
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>* pQ_equation;

public:

	////////////////////////////////////////////////////
	//
	// CTOR/DTOR

	__host__ void construct_cu_obj(void) 
	{
		nullgpuptr(pcuMesh);
		nullgpuptr(pcuaMesh);
		nullgpuptr(pQ_equation);
	}

	__host__ void destruct_cu_obj(void) {}

	////////////////////////////////////////////////////
	//
	// Configuration

	//not needed here, but needed in HeatCUDA_CMBND_Sec, and must work with same policy class
	void set_number_of_devices(int num_devices_cpu) {}

	//used in micromagnetic meshes
	BError set_pointers(MeshCUDA* pMeshCUDA, int idx_device);

	//used in atomistic meshes
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA, int idx_device);

	//set pQ_equation as needed
	BError set_Q_equation(mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Q_equation, int idx_device);

	////////////////////////////////////////////////////
	//
	// Runtime

	//heat flux, f(T) = -K * grad T = a + b * grad T -> a = 0, b = -K
	__device__ cuBReal a_func_pri(int cell1_idx, int cell2_idx, cuReal3 shift)
	{
		return 0.0;
	}

	//heat flux, f(T) = -K * grad T = a + b * grad T -> a = 0, b = -K
	__device__ cuBReal b_func_pri(int cell1_idx, int cell2_idx)
	{
		//micromagnetic
		if (pcuMesh) {

			cuBReal thermCond = *pcuMesh->pthermCond;
			pcuMesh->update_parameters_tcoarse(cell1_idx, *pcuMesh->pthermCond, thermCond);

			return -1.0 * thermCond;
		}
		//atomistic
		else if (pcuaMesh) {

			cuBReal thermCond = *pcuaMesh->pthermCond;
			pcuaMesh->update_parameters_tcoarse(cell1_idx, *pcuaMesh->pthermCond, thermCond);

			return -1.0 * thermCond;
		}
		else return 0.0;
	}

	//second order differential of T at cells either side of the boundary; delsq T = -Jc^2 / K * elC - Q / K - many-temperature model coupling terms / K
	__device__ cuBReal diff2_pri(int cell1_idx, cuReal3 shift)
	{
		//micromagnetic
		if (pcuMesh) {

			cuVEC_VC<cuBReal>& Temp = *pcuMesh->pTemp;
			cuVEC_VC<cuBReal>& Temp_l = *pcuMesh->pTemp_l;

			cuVEC_VC<cuReal3>& E = *pcuMesh->pE;
			cuVEC_VC<cuBReal>& elC = *pcuMesh->pelC;

			cuBReal thermCond = *pcuMesh->pthermCond;

			if (E.linear_size() || cuIsNZ(pcuMesh->pQ->get0()) || pQ_equation) {

				pcuMesh->update_parameters_tcoarse(cell1_idx, *pcuMesh->pthermCond, thermCond);
			}
			else return 0.0;

			cuBReal value = 0.0;

			//Joule heating
			if (E.linear_size()) {

				cuBReal joule_eff = *pcuMesh->pjoule_eff;
				pcuMesh->update_parameters_tcoarse(cell1_idx, *pcuMesh->pjoule_eff, joule_eff);

				if (cuIsNZ(joule_eff)) {

					int idx1_E = E.position_to_cellidx(Temp.cellidx_to_position(cell1_idx));
					value = -joule_eff * (elC[idx1_E] * E[idx1_E] * E[idx1_E]) / thermCond;
				}
			}
			//heat source contribution if set
			if (!pQ_equation) {
				if (cuIsNZ(pcuMesh->pQ->get0())) {

					cuBReal Q = *pcuMesh->pQ;
					pcuMesh->update_parameters_tcoarse(cell1_idx, *pcuMesh->pQ, Q);
					value -= Q / thermCond;
				}
			}
			else {

				cuReal3 relpos = Temp.get_crelpos_from_relpos(Temp.cellidx_to_position(cell1_idx));
				cuBReal Q = pQ_equation->evaluate(relpos.x, relpos.y, relpos.z, *pcuMesh->pcuDiffEq->pstagetime);
				value -= Q / thermCond;
			}
			
			if (Temp_l.linear_size()) {

				cuBReal G_el = *pcuMesh->pG_e;
				pcuMesh->update_parameters_tcoarse(cell1_idx, *pcuMesh->pG_e, G_el);

				value += G_el * (Temp[cell1_idx] - Temp_l[cell1_idx]) / thermCond;
			}

			return value;
		}
		//atomistic
		else if (pcuaMesh) {

			cuVEC_VC<cuBReal>& Temp = *pcuaMesh->pTemp;
			cuVEC_VC<cuBReal>& Temp_l = *pcuaMesh->pTemp_l;

			cuVEC_VC<cuReal3>& E = *pcuaMesh->pE;
			cuVEC_VC<cuBReal>& elC = *pcuaMesh->pelC;

			cuBReal thermCond = *pcuaMesh->pthermCond;

			if (E.linear_size() || cuIsNZ(pcuaMesh->pQ->get0()) || pQ_equation) {

				pcuaMesh->update_parameters_tcoarse(cell1_idx, *pcuaMesh->pthermCond, thermCond);
			}
			else return 0.0;

			cuBReal value = 0.0;

			//Joule heating
			if (E.linear_size()) {

				cuBReal joule_eff = *pcuaMesh->pjoule_eff;
				pcuaMesh->update_parameters_tcoarse(cell1_idx, *pcuaMesh->pjoule_eff, joule_eff);

				if (cuIsNZ(joule_eff)) {

					int idx1_E = E.position_to_cellidx(Temp.cellidx_to_position(cell1_idx));
					value = -joule_eff * (elC[idx1_E] * E[idx1_E] * E[idx1_E]) / thermCond;
				}
			}

			//heat source contribution if set
			if (!pQ_equation) {
				if (cuIsNZ(pcuaMesh->pQ->get0())) {

					cuBReal Q = *pcuaMesh->pQ;
					pcuaMesh->update_parameters_tcoarse(cell1_idx, *pcuaMesh->pQ, Q);
					value -= Q / thermCond;
				}
			}
			else {

				cuReal3 relpos = Temp.get_crelpos_from_relpos(Temp.cellidx_to_position(cell1_idx));
				cuBReal Q = pQ_equation->evaluate(relpos.x, relpos.y, relpos.z, *pcuaMesh->pcuaDiffEq->pstagetime);
				value -= Q / thermCond;
			}

			if (Temp_l.linear_size()) {

				cuBReal G_el = *pcuaMesh->pG_e;
				pcuaMesh->update_parameters_tcoarse(cell1_idx, *pcuaMesh->pG_e, G_el);

				value += G_el * (Temp[cell1_idx] - Temp_l[cell1_idx]) / thermCond;
			}

			return value;
		}
		else return 0.0;
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//This is held as a mcu_obj managed class in HeatCUDA modules with an associated policy class
//It provides methods and access to mesh data for use in cuVEC_VC cmbnd methods
//The a_func, b_func and diff2_func methods are used to set CMBND conditions based on the continuity of a quantity and a flux.
//If V is the potential, then the flux is the function f(V) = a_func + b_func * V', where the V' differential direction is perpendicular to the interface.
//HeatCUDA_CMBND_Sec is used for access on the secondary side, where we are setting cmbnd values on the primary side for a given device.
//However, on the secondary side we may need access from a different device. HeatCUDA_CMBND_Sec allows access to data on any device using UVA.
//Thus HeatCUDA_CMBND_Sec differs from HeatCUDA_CMBND_Pri to allow access to all devices, and identify which device we need depending on the relpos_m1 passed in (which is relative to the entire cuVEC)
class HeatCUDA_CMBND_Sec {

private:

	ManagedMeshCUDA** ppcuMesh;
	ManagedAtom_MeshCUDA** ppcuaMesh;

	//Q equation and time
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>* pQ_equation;

	//number of devices available (i.e. size of above arrays)
	int num_devices;

	//the device in which this HeatCUDA_CMBND_Sec is held
	int curr_device;

public:

	////////////////////////////////////////////////////
	//
	// CTOR/DTOR

	__host__ void construct_cu_obj(void)
	{
		nullgpuptr(ppcuMesh);
		nullgpuptr(ppcuaMesh);
		nullgpuptr(pQ_equation);

		set_gpu_value(num_devices, (int)0);
		set_gpu_value(curr_device, (int)0);
	}

	__host__ void destruct_cu_obj(void) 
	{
		if (get_gpu_value(num_devices)) {

			gpu_free_managed(ppcuMesh);
			nullgpuptr(ppcuMesh);

			gpu_free_managed(ppcuaMesh);
			nullgpuptr(ppcuaMesh);

			nullgpuptr(pQ_equation);
		}
	}

	////////////////////////////////////////////////////
	//
	// Configuration

	void set_number_of_devices(int num_devices_cpu)
	{
		if (!get_gpu_value(num_devices)) {

			gpu_alloc_managed(ppcuMesh, num_devices_cpu);

			gpu_alloc_managed(ppcuaMesh, num_devices_cpu);

			set_gpu_value(num_devices, num_devices_cpu);

			//null all entries
			for (int idx = 0; idx < num_devices_cpu; idx++) {

				ManagedMeshCUDA* cu_pointer_handle_pcuMesh = nullptr;
				cpu_to_gpu_managed(ppcuMesh, &cu_pointer_handle_pcuMesh, 1, idx);

				ManagedAtom_MeshCUDA* cu_pointer_handle_pcuaMesh = nullptr;
				cpu_to_gpu_managed(ppcuaMesh, &cu_pointer_handle_pcuaMesh, 1, idx);
			}
		}
	}

	//used in micromagnetic meshes
	BError set_pointers(MeshCUDA* pMeshCUDA, int idx_device);

	//used in atomistic meshes
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA, int idx_device);

	//set pQ_equation as needed
	BError set_Q_equation(mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Q_equation, int idx_device);

	////////////////////////////////////////////////////
	//
	// Runtime - Auxiliary

	//relpos_m1 is relative to entire cuVEC, and will be contained on a certain device. From this return a coordinate relative to containing device, also updating device if needed.
	//int device is passed as an initial hint (0)
	__device__ cuReal3 global_relpos_to_device_relpos(const cuReal3& relpos_m1, int& device)
	{	
		cuVEC_VC<cuBReal>& Temp_d = (ppcuMesh[device] ? *ppcuMesh[device]->pTemp : *ppcuaMesh[device]->pTemp);
		
		//if you add rect.s of first device you get an absolute coordinate
		cuVEC_VC<cuBReal>& Temp_0 = (ppcuMesh[0] ? *ppcuMesh[0]->pTemp : *ppcuaMesh[0]->pTemp);
		cuReal3 abspos_m1 = relpos_m1 + Temp_0.rect.s;

		//check hint device first
		if (Temp_d.rect.contains(abspos_m1)) {

			//return relative to hint device
			return abspos_m1 - Temp_d.rect.s;
		}

		//not in hint device. find which device contains it
		for (int device_idx = 0; device_idx < num_devices; device_idx++) {

			//skip hint device since already checked
			if (device_idx == device) continue;

			cuVEC_VC<cuBReal>& Temp = (ppcuMesh[device_idx] ? *ppcuMesh[device_idx]->pTemp : *ppcuaMesh[device_idx]->pTemp);

			if (Temp.rect.contains(abspos_m1)) {

				//update hint device
				device = device_idx;

				//return relative to containing device
				return abspos_m1 - Temp.rect.s;
			}
		}

		return cuReal3();
	}

	////////////////////////////////////////////////////
	//
	// Runtime

	//heat flux, f(T) = -K * grad T = a + b * grad T -> a = 0, b = -K
	__device__ cuBReal a_func_sec(cuReal3 relpos_m1, cuReal3 shift, cuReal3 stencil)
	{
		return 0.0;
	}

	//heat flux, f(T) = -K * grad T = a + b * grad T -> a = 0, b = -K
	__device__ cuBReal b_func_sec(cuReal3 relpos_m1, cuReal3 shift, cuReal3 stencil)
	{
		int device = 0;
		global_relpos_to_device_relpos(relpos_m1, device);

		//micromagnetic
		if (ppcuMesh[curr_device]) {

			cuBReal thermCond = *ppcuMesh[curr_device]->pthermCond;
			//if an equation dependence is set then must use current device (result is correct, but in theory should work for any device; it doesn't not sure why yet)
			if (ppcuMesh[curr_device]->pthermCond->is_dep_eq()) ppcuMesh[curr_device]->update_parameters_atposition(relpos_m1, *ppcuMesh[curr_device]->pthermCond, thermCond);
			else ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pthermCond, thermCond);

			return -1.0 * thermCond;
		}
		//atomistic
		else if (ppcuaMesh[curr_device]) {

			cuBReal thermCond = *ppcuaMesh[curr_device]->pthermCond;
			if (ppcuaMesh[curr_device]->pthermCond->is_dep_eq()) ppcuaMesh[curr_device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[curr_device]->pthermCond, thermCond);
			else ppcuaMesh[device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[device]->pthermCond, thermCond);

			return -1.0 * thermCond;
		}
		else return 0.0;
	}

	//second order differential of T at cells either side of the boundary; delsq T = -Jc^2 / K * elC - Q / K - many-temperature model coupling terms / K
	__device__ cuBReal diff2_sec(cuReal3 relpos_m1, cuReal3 stencil, cuReal3 shift)
	{
		int device = 0;
		cuReal3 devrelpos_m1 = global_relpos_to_device_relpos(relpos_m1, device);

		//micromagnetic
		if (ppcuMesh[curr_device]) {

			cuVEC_VC<cuBReal>& Temp = *ppcuMesh[device]->pTemp;
			cuVEC_VC<cuBReal>& Temp_l = *ppcuMesh[device]->pTemp_l;

			cuVEC_VC<cuReal3>& E = *ppcuMesh[device]->pE;
			cuVEC_VC<cuBReal>& elC = *ppcuMesh[device]->pelC;

			cuBReal thermCond = *ppcuMesh[curr_device]->pthermCond;

			if (E.linear_size() || cuIsNZ(ppcuMesh[curr_device]->pQ->get0()) || pQ_equation) {

				if (ppcuMesh[curr_device]->pthermCond->is_dep_eq()) ppcuMesh[curr_device]->update_parameters_atposition(relpos_m1, *ppcuMesh[curr_device]->pthermCond, thermCond);
				else ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pthermCond, thermCond);
			}
			else return 0.0;

			cuBReal value = 0.0;

			//Joule heating
			if (E.linear_size()) {

				cuBReal joule_eff = *ppcuMesh[curr_device]->pjoule_eff;
				if (ppcuMesh[curr_device]->pjoule_eff->is_dep_eq()) ppcuMesh[curr_device]->update_parameters_atposition(relpos_m1, *ppcuMesh[curr_device]->pjoule_eff, joule_eff);
				else ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pjoule_eff, joule_eff);

				if (cuIsNZ(joule_eff)) {

					int idx1_E = E.position_to_cellidx(devrelpos_m1);
					value = -joule_eff * (elC[idx1_E] * E[idx1_E] * E[idx1_E]) / thermCond;
				}
			}
			
			//heat source contribution if set
			if (!pQ_equation) {
				
				if (cuIsNZ(ppcuMesh[curr_device]->pQ->get0())) {

					cuBReal Q = *ppcuMesh[curr_device]->pQ;
					if (ppcuMesh[curr_device]->pQ->is_dep_eq()) ppcuMesh[curr_device]->update_parameters_atposition(relpos_m1, *ppcuMesh[curr_device]->pQ, Q);
					else ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pQ, Q);

					value -= Q / thermCond;
				}
			}
			else {

				cuBReal Q = pQ_equation->evaluate(relpos_m1.x, relpos_m1.y, relpos_m1.z, *ppcuMesh[curr_device]->pcuDiffEq->pstagetime);
				value -= Q / thermCond;
			}
			
			if (Temp_l.linear_size()) {

				cuBReal G_el = *ppcuMesh[curr_device]->pG_e;
				if (ppcuMesh[curr_device]->pG_e->is_dep_eq()) ppcuMesh[curr_device]->update_parameters_atposition(relpos_m1, *ppcuMesh[curr_device]->pG_e, G_el);
				else ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pG_e, G_el);

				value += G_el * (Temp.weighted_average(devrelpos_m1, stencil) - Temp_l.weighted_average(devrelpos_m1, stencil)) / thermCond;
			}

			return value;
		}
		//atomistic
		else if (ppcuaMesh[curr_device]) {

			cuVEC_VC<cuBReal>& Temp = *ppcuaMesh[device]->pTemp;
			cuVEC_VC<cuBReal>& Temp_l = *ppcuaMesh[device]->pTemp_l;

			cuVEC_VC<cuReal3>& E = *ppcuaMesh[device]->pE;
			cuVEC_VC<cuBReal>& elC = *ppcuaMesh[device]->pelC;

			cuBReal thermCond = *ppcuaMesh[curr_device]->pthermCond;

			if (E.linear_size() || cuIsNZ(ppcuaMesh[curr_device]->pQ->get0()) || pQ_equation) {

				if (ppcuaMesh[curr_device]->pthermCond->is_dep_eq()) ppcuaMesh[curr_device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[curr_device]->pthermCond, thermCond);
				else ppcuaMesh[device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[device]->pthermCond, thermCond);
			}
			else return 0.0;

			cuBReal value = 0.0;

			//Joule heating
			if (E.linear_size()) {

				cuBReal joule_eff = *ppcuaMesh[curr_device]->pjoule_eff;
				if (ppcuaMesh[curr_device]->pjoule_eff->is_dep_eq()) ppcuaMesh[curr_device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[curr_device]->pjoule_eff, joule_eff);
				else ppcuaMesh[device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[device]->pjoule_eff, joule_eff);

				if (cuIsNZ(joule_eff)) {

					int idx1_E = E.position_to_cellidx(devrelpos_m1);
					value = -joule_eff * (elC[idx1_E] * E[idx1_E] * E[idx1_E]) / thermCond;
				}
			}

			//heat source contribution if set
			if (!pQ_equation) {
				if (cuIsNZ(ppcuaMesh[curr_device]->pQ->get0())) {

					cuBReal Q = *ppcuaMesh[curr_device]->pQ;
					if (ppcuaMesh[curr_device]->pQ->is_dep_eq()) ppcuaMesh[curr_device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[curr_device]->pQ, Q);
					else ppcuaMesh[device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[device]->pQ, Q);

					value -= Q / thermCond;
				}
			}
			else {

				cuBReal Q = pQ_equation->evaluate(relpos_m1.x, relpos_m1.y, relpos_m1.z, *ppcuaMesh[curr_device]->pcuaDiffEq->pstagetime);
				value -= Q / thermCond;
			}

			if (Temp_l.linear_size()) {

				cuBReal G_el = *ppcuaMesh[curr_device]->pG_e;
				if (ppcuaMesh[curr_device]->pG_e->is_dep_eq()) ppcuaMesh[curr_device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[curr_device]->pG_e, G_el);
				else ppcuaMesh[device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[device]->pG_e, G_el);

				value += G_el * (Temp.weighted_average(devrelpos_m1, stencil) - Temp_l.weighted_average(devrelpos_m1, stencil)) / thermCond;
			}

			return value;
		}
		else return 0.0;
	}
};

#endif

#endif

