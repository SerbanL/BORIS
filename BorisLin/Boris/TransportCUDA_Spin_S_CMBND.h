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

#include "TransportCUDA_Spin_V_CMBND.h"

#include "Transport_Defs.h"

class MeshCUDA;
class Atom_MeshCUDA;
class TransportBaseCUDA;

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//This is held as a mcu_obj managed class in TransportCUDA modules
//It provides methods and access to mesh data for use in cuVEC_VC methods.
//The a_func, b_func and diff2_func methods are used to set CMBND conditions based on the continuity of a quantity and a flux.
//If V is the potential, then the flux is the function f(V) = a_func + b_func * V', where the V' differential direction is perpendicular to the interface.
//This particular class is used for spin transport within the spin current solver.
//TransportCUDA_Spin_S_CMBND_Pri is used for access on the primary side, where we are setting cmbnd values on the primary side for a given device.
//TransportCUDA_Spin_S_CMBND_Pri will be set for the same device.
class TransportCUDA_Spin_S_CMBND_Pri {

public:

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedMeshCUDA* pcuMesh;

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedAtom_MeshCUDA* pcuaMesh;

	//need this (held in TransportBaseCUDA) to access diff2_pri method
	TransportCUDA_Spin_V_CMBND_Pri* pspin_V_cmbnd_funcs_pri;

	//dM_dt VEC (charge pumping and spin pumping)
	//points to cuVEC in TransportBaseCUDA
	cuVEC_VC<cuReal3>* pdM_dt;

	//spin transport solver type (see Transport_Defs.h) : copy of stsolve in TransportCUDA, but on the gpu so we can use it in device code
	int stsolve;

public:

	////////////////////////////////////////////////////
	//
	// CTOR/DTOR

	__host__ void construct_cu_obj(void)
	{
		nullgpuptr(pcuMesh);
		nullgpuptr(pcuaMesh);
		nullgpuptr(pspin_V_cmbnd_funcs_pri);
		nullgpuptr(pdM_dt);
	}

	__host__ void destruct_cu_obj(void) {}

	////////////////////////////////////////////////////
	//
	// Configuration

	//not needed here, but needed in TransportCUDA_Spin_V_CMBND_Sec, and must work with same policy class
	void set_number_of_devices_mesh(int num_devices_cpu) {}
	void set_number_of_devices_amesh(int num_devices_cpu) {}

	//for modules held in micromagnetic meshes
	BError set_pointers(MeshCUDA* pMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int device_idx);
	//for modules held in atomistic meshes
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int device_idx);

	__host__ void set_stsolve(int stsolve_) { set_gpu_value(stsolve, stsolve_); }

	////////////////////////////////////////////////////
	//
	// Runtime

	//Functions used for calculating CMBND values

	//CMBND for S
	//flux = a + b S' at the interface, b = -De, a = -(muB*P*sigma/e) * E ox m + (SHA*sigma*muB/e)*epsE + charge pumping + topological Hall effect	
	__device__ cuReal3 a_func_pri(int cell1_idx, int cell2_idx, cuReal3 shift)
	{
		if (stsolve == STSOLVE_TUNNELING) return cuReal3();

		//need to find value at boundary so use interpolation

		cuVEC_VC<cuReal3>& E = (pcuMesh ? *pcuMesh->pE : *pcuaMesh->pE);
		cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);
		cuVEC_VC<cuReal3>& S = (pcuMesh ? *pcuMesh->pS : *pcuaMesh->pS);
		cuVEC_VC<cuReal3>& M = (pcuMesh ? *pcuMesh->pM : *pcuaMesh->pM1);

		//unit vector perpendicular to interface (pointing from secondary to primary mesh)
		cuReal3 u = shift.normalized() * -1;

		//values on secondary side
		if (stsolve == STSOLVE_FERROMAGNETIC || stsolve == STSOLVE_FERROMAGNETIC_ATOM) {

			cuBReal De = (pcuMesh ? *pcuMesh->pDe : *pcuaMesh->pDe);
			cuBReal P = (pcuMesh ? *pcuMesh->pP : *pcuaMesh->pP);
			if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pDe, De, *pcuMesh->pP, P);
			else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pDe, De, *pcuaMesh->pP, P);

			int idx_M1 = M.position_to_cellidx(S.cellidx_to_position(cell1_idx));
			int idx_M2 = M.position_to_cellidx(S.cellidx_to_position(cell2_idx));

			//a1 value
			cuReal3 m1 = cu_normalize(M[idx_M1]);
			cuReal3 E1 = E[cell1_idx];
			cuBReal sigma_1 = elC[cell1_idx];

			cuReal3 a1 = -(cuBReal)MUB_E * ((E1 | m1) | u) * (P * sigma_1);

			//a2 value
			cuReal3 m2 = cu_normalize(M[idx_M2]);
			cuReal3 E2 = E[cell2_idx];
			cuBReal sigma_2 = elC[cell2_idx];

			cuReal3 a2 = -(cuBReal)MUB_E * ((E2 | m2) | u) * (P * sigma_2);

			cuBReal cpump_eff0 = (pcuMesh ? pcuMesh->pcpump_eff->get0() : pcuaMesh->pcpump_eff->get0());
			bool cpump_enabled = cuIsNZ(cpump_eff0);

			cuBReal the_eff0 = (pcuMesh ? pcuMesh->pthe_eff->get0() : pcuaMesh->pthe_eff->get0());
			bool the_enabled = cuIsNZ(the_eff0);

			if (cuIsZ(shift.z) && (cpump_enabled || the_enabled)) {

				cuReal33 grad_m1 = cu_normalize(M.grad_neu(idx_M1), M[idx_M1]);
				cuReal33 grad_m2 = cu_normalize(M.grad_neu(idx_M2), M[idx_M2]);

				//topological Hall effect contribution
				if (the_enabled) {

					cuBReal n_density = (pcuMesh ? *pcuMesh->pn_density : *pcuaMesh->pn_density);
					if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pn_density, n_density);
					else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pn_density, n_density);

					cuReal3 B1 = (grad_m1.x ^ grad_m1.y);
					a1 += the_eff0 * ((cuBReal)HBAR_E * (cuBReal)MUB_E * sigma_1 * sigma_1 / ((cuBReal)ECHARGE * n_density)) * ((-E1.y * B1 * u.x) + (E1.x * B1 * u.y));

					cuReal3 B2 = (grad_m2.x ^ grad_m2.y);
					a2 += the_eff0 * ((cuBReal)HBAR_E * (cuBReal)MUB_E * sigma_2 * sigma_2 / ((cuBReal)ECHARGE * n_density)) * ((-E2.y * B2 * u.x) + (E2.x * B2 * u.y));
				}

				//charge pumping contribution
				if (cpump_enabled) {

					//value a1
					cuReal3 dm_dt_1 = cu_normalize((*pdM_dt)[idx_M1], M[idx_M1]);
					a1 += cpump_eff0 * ((cuBReal)HBAR_E * (cuBReal)MUB_E * sigma_1 / 2) * (((dm_dt_1 ^ grad_m1.x) * u.x) + ((dm_dt_1 ^ grad_m1.y) * u.y));

					cuReal3 dm_dt_2 = cu_normalize((*pdM_dt)[idx_M2], M[idx_M2]);
					a2 += cpump_eff0 * ((cuBReal)HBAR_E * (cuBReal)MUB_E * sigma_2 / 2) * (((dm_dt_2 ^ grad_m2.x) * u.x) + ((dm_dt_2 ^ grad_m2.y) * u.y));
				}
			}

			return (1.5 * a1 - 0.5 * a2);
		}
		else {

			//non-magnetic mesh. pcuMesh only, but best to check
			if (pcuMesh) {

				cuBReal SHA = *pcuMesh->pSHA;
				pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pSHA, SHA);

				//non-magnetic mesh
				cuReal3 a1 = (cu_epsilon3(E[cell1_idx]) | u) * SHA * elC[cell1_idx] * (cuBReal)MUB_E;
				cuReal3 a2 = (cu_epsilon3(E[cell2_idx]) | u) * SHA * elC[cell2_idx] * (cuBReal)MUB_E;

				return (1.5 * a1 - 0.5 * a2);
			}
			else return cuReal3();
		}
	}

	//For V only : V and Jc are continuous; Jc = -sigma * grad V = a + b * grad V -> a = 0 and b = -sigma taken at the interface	
	__device__ cuBReal b_func_pri(int cell1_idx, int cell2_idx)
	{
		cuBReal De = (pcuMesh ? *pcuMesh->pDe : *pcuaMesh->pDe);
		cuBReal elecCond = (pcuMesh ? *pcuMesh->pelecCond : *pcuaMesh->pelecCond);
		if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pDe, De, *pcuMesh->pelecCond, elecCond);
		else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pDe, De, *pcuaMesh->pelecCond, elecCond);

		//metallic conduction : use De
		if (elecCond > 0.0) return -De;
		//tunelling : use 1
		else return -1;
	}

	//second order differential of S along the shift axis
	//this is simply Evaluate_SpinSolver_delsqS_RHS from which we subtract second order differentials orthogonal to the shift axis
	__device__ cuReal3 diff2_pri(int cell1_idx, cuReal3 shift)
	{
		cuReal3 delsq_S_RHS;

		cuVEC_VC<cuBReal>& V = (pcuMesh ? *pcuMesh->pV : *pcuaMesh->pV);
		cuVEC_VC<cuReal3>& S = (pcuMesh ? *pcuMesh->pS : *pcuaMesh->pS);
		cuVEC_VC<cuReal3>& M = (pcuMesh ? *pcuMesh->pM : *pcuaMesh->pM1);
		cuVEC_VC<cuReal3>& E = (pcuMesh ? *pcuMesh->pE : *pcuaMesh->pE);
		cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);

		cuBReal l_sf = (pcuMesh ? *pcuMesh->pl_sf : *pcuaMesh->pl_sf);
		if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pl_sf, l_sf);
		else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pl_sf, l_sf);

		if (stsolve == STSOLVE_TUNNELING && pcuMesh) {

			cuBReal elecCond = *pcuMesh->pelecCond;
			pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pelecCond, elecCond);

			//Use elecCond to mark out metallic pinholes.
			//tunelling : l_sf tends to infinity
			if (elecCond > 0.0) delsq_S_RHS = (S[cell1_idx] / (l_sf * l_sf));
			return delsq_S_RHS;
		}

		//Contributions which apply equally in ferromagnetic and non-ferromagnetic meshes
		//longitudinal S decay term
		delsq_S_RHS = (S[cell1_idx] / (l_sf * l_sf));

		//Terms occuring only in magnetic meshes
		if (stsolve == STSOLVE_FERROMAGNETIC || stsolve == STSOLVE_FERROMAGNETIC_ATOM) {

			cuBReal l_ex = (pcuMesh ? *pcuMesh->pl_ex : *pcuaMesh->pl_ex);
			cuBReal l_ph = (pcuMesh ? *pcuMesh->pl_ph : *pcuaMesh->pl_ph);
			if (pcuMesh)pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pl_ex, l_ex, *pcuMesh->pl_ph, l_ph);
			else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pl_ex, l_ex, *pcuaMesh->pl_ph, l_ph);

			int idx_M = M.position_to_cellidx(V.cellidx_to_position(cell1_idx));
			cuReal3 m = cu_normalize(M[idx_M]);

			//transverse S decay terms
			delsq_S_RHS += ((S[cell1_idx] ^ m) / (l_ex * l_ex) + (m ^ (S[cell1_idx] ^ m)) / (l_ph * l_ph));
		}

		//only calculate current on non-empty cells - empty cells have already been assigned 0 at UpdateConfiguration
		if (V.is_not_empty(cell1_idx)) {

			bool cpump_enabled = cuIsNZ(pcuMesh ? pcuMesh->pcpump_eff->get0() : pcuaMesh->pcpump_eff->get0());
			bool the_enabled = cuIsNZ(pcuMesh ? pcuMesh->pthe_eff->get0() : pcuaMesh->pthe_eff->get0());
			bool she_enabled = cuIsNZ(pcuMesh ? pcuMesh->pSHA->get0() : pcuaMesh->pSHA->get0());

			if (stsolve == STSOLVE_FERROMAGNETIC) {

				//magnetic mesh

				cuBReal P = (pcuMesh ? *pcuMesh->pP : *pcuaMesh->pP);
				cuBReal De = (pcuMesh ? *pcuMesh->pDe : *pcuaMesh->pDe);
				if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pP, P, *pcuMesh->pDe, De);
				else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pP, P, *pcuaMesh->pDe, De);

				//term due to drift (non-uniformity of M term, and delsq V contribution - non-uniformity of E term)

				//find grad M and M at the M cell in which the current S cell center is
				int idx_M = M.position_to_cellidx(V.cellidx_to_position(cell1_idx));

				cuReal3 m = cu_normalize(M[idx_M]);
				cuReal33 grad_m = cu_normalize(M.grad_neu(idx_M), M[idx_M]);
				cuReal3 E_dot_del_m = grad_m | E[cell1_idx];

				//E_dot_del_m term is very important, but Evaluate_SpinSolver_delsqV_RHS term could be neglected in most cases especially if E is uniform.
				delsq_S_RHS += (P * (cuBReal)MUB_E * elC[cell1_idx] / De) * (pspin_V_cmbnd_funcs_pri->diff2_pri(cell1_idx, shift) * m - E_dot_del_m);

				//charge pumping and topological Hall effect
				if (cpump_enabled || the_enabled) {

					cuReal3 dx_m = grad_m.x;
					cuReal3 dy_m = grad_m.y;
					cuReal3 dxy_m = cu_normalize(M.dxy_neu(idx_M), M[idx_M]);
					cuReal3 dxx_m = cu_normalize(M.dxx_neu(idx_M), M[idx_M]);
					cuReal3 dyy_m = cu_normalize(M.dyy_neu(idx_M), M[idx_M]);

					if (cpump_enabled) {

						cuReal3 dmdt = cu_normalize((*pdM_dt)[idx_M], M[idx_M]);
						cuReal33 grad_dm_dt = cu_normalize((*pdM_dt).grad_neu(idx_M), M[idx_M]);

						cuBReal cpump_eff = (pcuMesh ? pcuMesh->pcpump_eff->get0() : pcuaMesh->pcpump_eff->get0());
						delsq_S_RHS += cpump_eff * (elC[cell1_idx] * (cuBReal)HBAR_E * (cuBReal)MUB_E / (2 * De)) * ((grad_dm_dt.x ^ dx_m) + (grad_dm_dt.y ^ dy_m) + (dmdt ^ (dxx_m + dyy_m)));
					}

					if (the_enabled) {

						cuBReal n_density = (pcuMesh ? *pcuMesh->pn_density : *pcuaMesh->pn_density);
						if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pn_density, n_density);
						else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pn_density, n_density);

						cuBReal the_eff = (pcuMesh ? pcuMesh->pthe_eff->get0() : pcuaMesh->pthe_eff->get0());
						delsq_S_RHS += the_eff * ((cuBReal)HBAR_E * (cuBReal)MUB_E * elC[cell1_idx] * elC[cell1_idx] / ((cuBReal)ECHARGE * n_density * De)) * (E[cell1_idx].x * ((dxy_m ^ dy_m) + (dx_m ^ dyy_m)) - E[cell1_idx].y * ((dxx_m ^ dy_m) + (dx_m ^ dxy_m)));
					}
				}
			}

			//terms occuring only in non-magnetic meshes
			else {

				//1. SHA term (this is negligible in most cases, even if E is non-uniform, but might as well include it) 
				if (she_enabled) {

					cuBReal SHA = (pcuMesh ? *pcuMesh->pSHA : *pcuaMesh->pSHA);
					cuBReal De = (pcuMesh ? *pcuMesh->pDe : *pcuaMesh->pDe);
					if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pSHA, SHA, *pcuMesh->pDe, De);
					else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pSHA, SHA, *pcuaMesh->pDe, De);

					//Check boundary conditions for this term : should be Dirichlet with 0 Jc value normal to the boundary except for electrodes.
					delsq_S_RHS += (SHA * elC[cell1_idx] * (cuBReal)MUB_E / De) * E.diveps3_sided(cell1_idx);
				}
			}
		}

		return delsq_S_RHS;
	}

	//multiply spin accumulation by these to obtain spin potential, i.e. Vs = (De / elC) * (e/muB) * S, evaluated at the boundary
	__device__ cuBReal c_func_pri(int cell_idx)
	{
		cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);

		cuBReal De = (pcuMesh ? *pcuMesh->pDe : *pcuaMesh->pDe);
		cuBReal elecCond = (pcuMesh ? *pcuMesh->pelecCond : *pcuaMesh->pelecCond);
		if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell_idx, *pcuMesh->pDe, De, *pcuMesh->pelecCond, elecCond);
		else pcuaMesh->update_parameters_ecoarse(cell_idx, *pcuaMesh->pDe, De, *pcuaMesh->pelecCond, elecCond);

		//metallic conduction : use De
		if (elecCond > 0.0) return De / (elC[cell_idx] * (cuBReal)MUB_E);
		//tunelling : use 1
		else return 1.0 / (elC[cell_idx] * (cuBReal)MUB_E);
	}

	//multiply spin accumulation by these to obtain spin potential, i.e. Vs = (De / elC) * (e/muB) * S, evaluated at the boundary
	__device__ cuBReal c_func_sec(cuReal3 relpos, cuReal3 stencil)
	{
		cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);

		cuBReal De = (pcuMesh ? *pcuMesh->pDe : *pcuaMesh->pDe);
		cuBReal elecCond = (pcuMesh ? *pcuMesh->pelecCond : *pcuaMesh->pelecCond);
		if (pcuMesh) pcuMesh->update_parameters_atposition(relpos, *pcuMesh->pDe, De, *pcuMesh->pelecCond, elecCond);
		else pcuaMesh->update_parameters_atposition(relpos, *pcuaMesh->pDe, De, *pcuaMesh->pelecCond, elecCond);

		//metallic conduction : use De
		if (elecCond > 0.0) return De / (elC.weighted_average(relpos, stencil) * (cuBReal)MUB_E);
		//tunelling : use 1
		else return 1.0 / (elC.weighted_average(relpos, stencil) * (cuBReal)MUB_E);
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//This is held as a mcu_obj managed class in TransportCUDA modules
//It provides methods and access to mesh data for use in cuVEC_VC methods.
//The a_func, b_func and diff2_func methods are used to set CMBND conditions based on the continuity of a quantity and a flux.
//If V is the potential, then the flux is the function f(V) = a_func + b_func * V', where the V' differential direction is perpendicular to the interface.
//This particular class is used for spin transport within the spin current solver.
//TransportCUDA_Spin_S_CMBND_Sec is used for access on the secondary side, where we are setting cmbnd values on the primary side for a given device.
//However, on the secondary side we may need access from a different device. TransportCUDA_Spin_S_CMBND_Sec allows access to data on any device using UVA.
//Thus TransportCUDA_Spin_S_CMBND_Sec differs from TransportCUDA_Spin_S_CMBND_Sec to allow access to all devices, and identify which device we need depending on the relpos_m1 passed in (which is relative to the entire cuVEC)
class TransportCUDA_Spin_S_CMBND_Sec {

public:

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedMeshCUDA** ppcuMesh;

	//managed mesh for access to all required mesh VECs and material parameters
	ManagedAtom_MeshCUDA** ppcuaMesh;

	//need this (held in TransportBaseCUDA) to access diff2_pri method
	TransportCUDA_Spin_V_CMBND_Pri** ppspin_V_cmbnd_funcs_pri;

	//dM_dt VEC (charge pumping and spin pumping)
	//points to cuVEC in TransportBaseCUDA
	cuVEC_VC<cuReal3>** ppdM_dt;

	//spin transport solver type (see Transport_Defs.h) : copy of stsolve in TransportCUDA, but on the gpu so we can use it in device code
	int stsolve;

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

	//for modules held in micromagnetic meshes
	BError set_pointers(MeshCUDA* pMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int device_idx);
	//for modules held in atomistic meshes
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int device_idx);

	__host__ void set_stsolve(int stsolve_) { set_gpu_value(stsolve, stsolve_); }

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

	//Functions used for calculating CMBND values

	//CMBND for S

	//flux = a + b S' at the interface, b = -De, a = -(muB*P*sigma/e) * E ox m + (SHA*sigma*muB/e)*epsE + charge pumping + topological Hall effect
	__device__ cuReal3 a_func_sec(cuReal3 relpos_m1, cuReal3 shift, cuReal3 stencil)
	{
		if (stsolve == STSOLVE_TUNNELING) return cuReal3();

		int device = 0;
		cuReal3 devrelpos_m1 = global_relpos_to_device_relpos(relpos_m1, device);

		//need to find value at boundary so use interpolation

		cuVEC_VC<cuReal3>& E = (ppcuMesh ? *ppcuMesh[device]->pE : *ppcuaMesh[device]->pE);
		cuVEC_VC<cuBReal>& elC = (ppcuMesh ? *ppcuMesh[device]->pelC : *ppcuaMesh[device]->pelC);
		cuVEC_VC<cuReal3>& S = (ppcuMesh ? *ppcuMesh[device]->pS : *ppcuaMesh[device]->pS);
		cuVEC_VC<cuReal3>& M = (ppcuMesh ? *ppcuMesh[device]->pM : *ppcuaMesh[device]->pM1);

		//unit vector perpendicular to interface (pointing from secondary to primary mesh)
		cuReal3 u = shift.normalized() * -1;

		//values on secondary side
		if (stsolve == STSOLVE_FERROMAGNETIC || stsolve == STSOLVE_FERROMAGNETIC_ATOM) {

			cuBReal De = (ppcuMesh ? *ppcuMesh[device]->pDe : *ppcuaMesh[device]->pDe);
			cuBReal P = (ppcuMesh ? *ppcuMesh[device]->pP : *ppcuaMesh[device]->pP);
			if (ppcuMesh) ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pDe, De, *ppcuMesh[device]->pP, P);
			else ppcuaMesh[device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[device]->pDe, De, *ppcuaMesh[device]->pP, P);

			//a1 value
			cuReal3 m1 = cu_normalize(M.weighted_average(devrelpos_m1, stencil));
			cuReal3 E1 = E.weighted_average(devrelpos_m1, stencil);
			cuBReal sigma_1 = elC.weighted_average(devrelpos_m1, stencil);

			cuReal3 a1 = -(cuBReal)MUB_E * ((E1 | m1) | u) * (P * sigma_1);

			//a2 value
			cuReal3 m2 = cu_normalize(M.weighted_average(devrelpos_m1 + shift, stencil));
			cuReal3 E2 = E.weighted_average(devrelpos_m1 + shift, stencil);
			cuBReal sigma_2 = elC.weighted_average(devrelpos_m1 + shift, stencil);

			cuReal3 a2 = -(cuBReal)MUB_E * ((E2 | m2) | u) * (P * sigma_2);

			cuBReal cpump_eff0 = (ppcuMesh ? ppcuMesh[device]->pcpump_eff->get0() : ppcuaMesh[device]->pcpump_eff->get0());
			bool cpump_enabled = cuIsNZ(cpump_eff0);

			cuBReal the_eff0 = (ppcuMesh ? ppcuMesh[device]->pthe_eff->get0() : ppcuaMesh[device]->pthe_eff->get0());
			bool the_enabled = cuIsNZ(the_eff0);
			if (cuIsZ(shift.z) && (cpump_enabled || the_enabled)) {

				int idx_M1 = M.position_to_cellidx(devrelpos_m1);
				int idx_M2 = M.position_to_cellidx(devrelpos_m1 + shift);

				cuReal33 grad_m1 = cu_normalize(M.grad_neu(idx_M1), M[idx_M1]);
				cuReal33 grad_m2 = cu_normalize(M.grad_neu(idx_M2), M[idx_M2]);

				//topological Hall effect contribution
				if (the_enabled) {

					cuBReal n_density = (ppcuMesh ? *ppcuMesh[device]->pn_density : *ppcuaMesh[device]->pn_density);
					if (ppcuMesh) ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pn_density, n_density);
					else ppcuaMesh[device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[device]->pn_density, n_density);

					cuReal3 B1 = (grad_m1.x ^ grad_m1.y);
					a1 += the_eff0 * ((cuBReal)HBAR_E * (cuBReal)MUB_E * sigma_1 * sigma_1 / ((cuBReal)ECHARGE * n_density)) * ((-E1.y * B1 * u.x) + (E1.x * B1 * u.y));

					cuReal3 B2 = (grad_m2.x ^ grad_m2.y);
					a2 += the_eff0 * ((cuBReal)HBAR_E * (cuBReal)MUB_E * sigma_2 * sigma_2 / ((cuBReal)ECHARGE * n_density)) * ((-E2.y * B2 * u.x) + (E2.x * B2 * u.y));
				}

				//charge pumping contribution
				if (cpump_enabled) {

					//value a1
					cuReal3 dm_dt_1 = cu_normalize(ppdM_dt[device]->weighted_average(devrelpos_m1, stencil), M[idx_M1]);
					a1 += cpump_eff0 * ((cuBReal)HBAR_E * (cuBReal)MUB_E * sigma_1 / 2) * (((dm_dt_1 ^ grad_m1.x) * u.x) + ((dm_dt_1 ^ grad_m1.y) * u.y));

					cuReal3 dm_dt_2 = cu_normalize(ppdM_dt[device]->weighted_average(devrelpos_m1 + shift, stencil), M[idx_M2]);
					a2 += cpump_eff0 * ((cuBReal)HBAR_E * (cuBReal)MUB_E * sigma_2 / 2) * (((dm_dt_2 ^ grad_m2.x) * u.x) + ((dm_dt_2 ^ grad_m2.y) * u.y));
				}
			}

			return (1.5 * a1 - 0.5 * a2);
		}
		else {

			//non-magnetic mesh. ppcuMesh[device] only, but best to check
			if (ppcuMesh) {

				cuBReal SHA = *ppcuMesh[device]->pSHA;
				ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pSHA, SHA);

				//non-magnetic mesh
				cuReal3 a1 = (cu_epsilon3(E.weighted_average(devrelpos_m1, stencil)) | u) * SHA * elC.weighted_average(devrelpos_m1, stencil) * (cuBReal)MUB_E;
				cuReal3 a2 = (cu_epsilon3(E.weighted_average(devrelpos_m1 + shift, stencil)) | u) * SHA * elC.weighted_average(devrelpos_m1 + shift, stencil) * (cuBReal)MUB_E;

				return (1.5 * a1 - 0.5 * a2);
			}
			else return cuReal3();
		}
	}

	//For V only : V and Jc are continuous; Jc = -sigma * grad V = a + b * grad V -> a = 0 and b = -sigma taken at the interface	
	__device__ cuBReal b_func_sec(cuReal3 relpos_m1, cuReal3 shift, cuReal3 stencil)
	{
		int device = 0;
		global_relpos_to_device_relpos(relpos_m1, device);

		cuBReal De = (ppcuMesh ? *ppcuMesh[device]->pDe : *ppcuaMesh[device]->pDe);
		cuBReal elecCond = (ppcuMesh ? *ppcuMesh[device]->pelecCond : *ppcuaMesh[device]->pelecCond);
		if (ppcuMesh) ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pDe, De, *ppcuMesh[device]->pelecCond, elecCond);
		else ppcuaMesh[device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[device]->pDe, De, *ppcuaMesh[device]->pelecCond, elecCond);

		//metallic conduction : use De
		if (elecCond > 0.0) return -De;
		//tunelling : use 1
		else return -1;
	}

	//second order differential of S along the shift axis
	//this is simply Evaluate_SpinSolver_delsqS_RHS from which we subtract second order differentials orthogonal to the shift axis
	__device__ cuReal3 diff2_pri(int cell1_idx, cuReal3 shift, int device)
	{
		cuReal3 delsq_S_RHS;

		cuVEC_VC<cuBReal>& V = (ppcuMesh ? *ppcuMesh[device]->pV : *ppcuaMesh[device]->pV);
		cuVEC_VC<cuReal3>& S = (ppcuMesh ? *ppcuMesh[device]->pS : *ppcuaMesh[device]->pS);
		cuVEC_VC<cuReal3>& M = (ppcuMesh ? *ppcuMesh[device]->pM : *ppcuaMesh[device]->pM1);
		cuVEC_VC<cuReal3>& E = (ppcuMesh ? *ppcuMesh[device]->pE : *ppcuaMesh[device]->pE);
		cuVEC_VC<cuBReal>& elC = (ppcuMesh ? *ppcuMesh[device]->pelC : *ppcuaMesh[device]->pelC);

		cuBReal l_sf = (ppcuMesh ? *ppcuMesh[device]->pl_sf : *ppcuaMesh[device]->pl_sf);
		if (ppcuMesh) ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pl_sf, l_sf);
		else ppcuaMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuaMesh[device]->pl_sf, l_sf);

		if (stsolve == STSOLVE_TUNNELING && ppcuMesh[device]) {

			cuBReal elecCond = *ppcuMesh[device]->pelecCond;
			ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pelecCond, elecCond);

			//Use elecCond to mark out metallic pinholes.
			//tunelling : l_sf tends to infinity
			if (elecCond > 0.0) delsq_S_RHS = (S[cell1_idx] / (l_sf * l_sf));
			return delsq_S_RHS;
		}

		//Contributions which apply equally in ferromagnetic and non-ferromagnetic meshes
		//longitudinal S decay term
		delsq_S_RHS = (S[cell1_idx] / (l_sf * l_sf));

		//Terms occuring only in magnetic meshes
		if (stsolve == STSOLVE_FERROMAGNETIC || stsolve == STSOLVE_FERROMAGNETIC_ATOM) {

			cuBReal l_ex = (ppcuMesh ? *ppcuMesh[device]->pl_ex : *ppcuaMesh[device]->pl_ex);
			cuBReal l_ph = (ppcuMesh ? *ppcuMesh[device]->pl_ph : *ppcuaMesh[device]->pl_ph);
			if (ppcuMesh)ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pl_ex, l_ex, *ppcuMesh[device]->pl_ph, l_ph);
			else ppcuaMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuaMesh[device]->pl_ex, l_ex, *ppcuaMesh[device]->pl_ph, l_ph);

			int idx_M = M.position_to_cellidx(V.cellidx_to_position(cell1_idx));
			cuReal3 m = cu_normalize(M[idx_M]);

			//transverse S decay terms
			delsq_S_RHS += ((S[cell1_idx] ^ m) / (l_ex * l_ex) + (m ^ (S[cell1_idx] ^ m)) / (l_ph * l_ph));
		}

		//only calculate current on non-empty cells - empty cells have already been assigned 0 at UpdateConfiguration
		if (V.is_not_empty(cell1_idx)) {

			bool cpump_enabled = cuIsNZ(ppcuMesh ? ppcuMesh[device]->pcpump_eff->get0() : ppcuaMesh[device]->pcpump_eff->get0());
			bool the_enabled = cuIsNZ(ppcuMesh ? ppcuMesh[device]->pthe_eff->get0() : ppcuaMesh[device]->pthe_eff->get0());
			bool she_enabled = cuIsNZ(ppcuMesh ? ppcuMesh[device]->pSHA->get0() : ppcuaMesh[device]->pSHA->get0());

			if (stsolve == STSOLVE_FERROMAGNETIC) {

				//magnetic mesh

				cuBReal P = (ppcuMesh ? *ppcuMesh[device]->pP : *ppcuaMesh[device]->pP);
				cuBReal De = (ppcuMesh ? *ppcuMesh[device]->pDe : *ppcuaMesh[device]->pDe);
				if (ppcuMesh) ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pP, P, *ppcuMesh[device]->pDe, De);
				else ppcuaMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuaMesh[device]->pP, P, *ppcuaMesh[device]->pDe, De);

				//term due to drift (non-uniformity of M term, and delsq V contribution - non-uniformity of E term)

				//find grad M and M at the M cell in which the current S cell center is
				int idx_M = M.position_to_cellidx(V.cellidx_to_position(cell1_idx));

				cuReal3 m = cu_normalize(M[idx_M]);
				cuReal33 grad_m = cu_normalize(M.grad_neu(idx_M), M[idx_M]);
				cuReal3 E_dot_del_m = grad_m | E[cell1_idx];

				//E_dot_del_m term is very important, but Evaluate_SpinSolver_delsqV_RHS term could be neglected in most cases especially if E is uniform.
				delsq_S_RHS += (P * (cuBReal)MUB_E * elC[cell1_idx] / De) * (ppspin_V_cmbnd_funcs_pri[device]->diff2_pri(cell1_idx, shift) * m - E_dot_del_m);

				//charge pumping and topological Hall effect
				if (cpump_enabled || the_enabled) {

					cuReal3 dx_m = grad_m.x;
					cuReal3 dy_m = grad_m.y;
					cuReal3 dxy_m = cu_normalize(M.dxy_neu(idx_M), M[idx_M]);
					cuReal3 dxx_m = cu_normalize(M.dxx_neu(idx_M), M[idx_M]);
					cuReal3 dyy_m = cu_normalize(M.dyy_neu(idx_M), M[idx_M]);

					if (cpump_enabled) {

						cuReal3 dmdt = cu_normalize((*ppdM_dt[device])[idx_M], M[idx_M]);
						cuReal33 grad_dm_dt = cu_normalize((*ppdM_dt[device]).grad_neu(idx_M), M[idx_M]);

						cuBReal cpump_eff = (ppcuMesh ? ppcuMesh[device]->pcpump_eff->get0() : ppcuaMesh[device]->pcpump_eff->get0());
						delsq_S_RHS += cpump_eff * (elC[cell1_idx] * (cuBReal)HBAR_E * (cuBReal)MUB_E / (2 * De)) * ((grad_dm_dt.x ^ dx_m) + (grad_dm_dt.y ^ dy_m) + (dmdt ^ (dxx_m + dyy_m)));
					}

					if (the_enabled) {

						cuBReal n_density = (ppcuMesh ? *ppcuMesh[device]->pn_density : *ppcuaMesh[device]->pn_density);
						if (ppcuMesh) ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pn_density, n_density);
						else ppcuaMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuaMesh[device]->pn_density, n_density);

						cuBReal the_eff = (ppcuMesh ? ppcuMesh[device]->pthe_eff->get0() : ppcuaMesh[device]->pthe_eff->get0());
						delsq_S_RHS += the_eff * ((cuBReal)HBAR_E * (cuBReal)MUB_E * elC[cell1_idx] * elC[cell1_idx] / ((cuBReal)ECHARGE * n_density * De)) * (E[cell1_idx].x * ((dxy_m ^ dy_m) + (dx_m ^ dyy_m)) - E[cell1_idx].y * ((dxx_m ^ dy_m) + (dx_m ^ dxy_m)));
					}
				}
			}

			//terms occuring only in non-magnetic meshes
			else {

				//1. SHA term (this is negligible in most cases, even if E is non-uniform, but might as well include it) 
				if (she_enabled) {

					cuBReal SHA = (ppcuMesh ? *ppcuMesh[device]->pSHA : *ppcuaMesh[device]->pSHA);
					cuBReal De = (ppcuMesh ? *ppcuMesh[device]->pDe : *ppcuaMesh[device]->pDe);
					if (ppcuMesh) ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pSHA, SHA, *ppcuMesh[device]->pDe, De);
					else ppcuaMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuaMesh[device]->pSHA, SHA, *ppcuaMesh[device]->pDe, De);

					//Check boundary conditions for this term : should be Dirichlet with 0 Jc value normal to the boundary except for electrodes.
					delsq_S_RHS += (SHA * elC[cell1_idx] * (cuBReal)MUB_E / De) * E.diveps3_sided(cell1_idx);
				}
			}
		}

		return delsq_S_RHS;
	}

	//second order differential of S along the shift axis
	//this is simply Evaluate_SpinSolver_delsqS_RHS from which we subtract second order differentials orthogonal to the shift axis
	__device__ cuReal3 diff2_sec(cuReal3 relpos_m1, cuReal3 stencil, cuReal3 shift)
	{
		int device = 0;
		cuReal3 devrelpos_m1 = global_relpos_to_device_relpos(relpos_m1, device);

		cuVEC_VC<cuReal3>& S = (ppcuMesh ? *ppcuMesh[device]->pS : *ppcuaMesh[device]->pS);
		int cellm1_idx = S.position_to_cellidx(devrelpos_m1);
		return diff2_pri(cellm1_idx, shift, device);
	}

	//multiply spin accumulation by these to obtain spin potential, i.e. Vs = (De / elC) * (e/muB) * S, evaluated at the boundary
	__device__ cuBReal c_func_sec(cuReal3 relpos, cuReal3 stencil)
	{
		int device = 0;
		cuReal3 devrelpos_m1 = global_relpos_to_device_relpos(relpos, device);

		cuVEC_VC<cuBReal>& elC = (ppcuMesh ? *ppcuMesh[device]->pelC : *ppcuaMesh[device]->pelC);

		cuBReal De = (ppcuMesh ? *ppcuMesh[device]->pDe : *ppcuaMesh[device]->pDe);
		cuBReal elecCond = (ppcuMesh ? *ppcuMesh[device]->pelecCond : *ppcuaMesh[device]->pelecCond);
		if (ppcuMesh) ppcuMesh[device]->update_parameters_atposition(relpos, *ppcuMesh[device]->pDe, De, *ppcuMesh[device]->pelecCond, elecCond);
		else ppcuaMesh[device]->update_parameters_atposition(relpos, *ppcuaMesh[device]->pDe, De, *ppcuaMesh[device]->pelecCond, elecCond);

		//metallic conduction : use De
		if (elecCond > 0.0) return De / (elC.weighted_average(devrelpos_m1, stencil) * (cuBReal)MUB_E);
		//tunelling : use 1
		else return 1.0 / (elC.weighted_average(devrelpos_m1, stencil) * (cuBReal)MUB_E);
	}
};

#endif

#endif