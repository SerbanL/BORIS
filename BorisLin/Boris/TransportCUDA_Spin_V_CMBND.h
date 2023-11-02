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

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//This is held as a mcu_obj managed class in TransportCUDA modules
//It provides methods and access to mesh data for use in cuVEC_VC methods.
//The a_func, b_func and diff2_func methods are used to set CMBND conditions based on the continuity of a quantity and a flux.
//If V is the potential, then the flux is the function f(V) = a_func + b_func * V', where the V' differential direction is perpendicular to the interface.
//This particular class is used for charge transport within the spin current solver.
//TransportCUDA_Spin_V_CMBND_Pri is used for access on the primary side, where we are setting cmbnd values on the primary side for a given device.
//TransportCUDA_Spin_V_CMBND_Pri will be set for the same device.
class TransportCUDA_Spin_V_CMBND_Pri {

public:

	//micromagnetic version : managed mesh for access to all required mesh VECs and material parameters (if nullptr then not used)
	ManagedMeshCUDA* pcuMesh;

	//atomistic version : managed mesh for access to all required mesh VECs and material parameters (if nullptr then not used)
	ManagedAtom_MeshCUDA* pcuaMesh;

	//dM_dt VEC when we need to do vector calculus operations on it
	//points to cuVEC in TransportCUDA or Atom_TransportCUDA
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

	//CMBND for V
	//flux = a + b V' at the interface, b = -sigma, a = betaD * (De*e/muB) * (grad S)m + (SHA*De*e/muB) * curl S + charge pumping + topological Hall effect
	//Note, the topological Hall effect term includes E, thus gradients in V, but we can include these in the a term for 2 reasons:
	//1. these CMBND functions especially with the topological Hall effect enabled is used for interfaces along z direction normally, and here Ez is zero 
	//(for such interfaces both charge pumping and topological Hall effect have zero contribution to z direction charge current)
	//2. even if the interface is along x or y we can still use the previously calculated E field, and the solution will converge to the same value (but might take more iterations).
	__device__ cuBReal a_func_pri(int cell1_idx, int cell2_idx, cuReal3 shift)
	{
		if (stsolve == STSOLVE_TUNNELING) return 0.0;

		cuVEC_VC<cuReal3>& S = (pcuMesh ? *pcuMesh->pS : *pcuaMesh->pS);
		cuVEC_VC<cuBReal>& V = (pcuMesh ? *pcuMesh->pV : *pcuaMesh->pV);
		cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);
		cuVEC_VC<cuReal3>& M = (pcuMesh ? *pcuMesh->pM : *pcuaMesh->pM1);

		cuBReal a = 0.0;

		cuReal3 u = shift.normalized() * -1;

		bool cppgmr_enabled = (pcuMesh ? cuIsNZ(pcuMesh->pbetaD->get0()) : cuIsNZ(pcuaMesh->pbetaD->get0()));

		cuBReal cpump_eff0 = (pcuMesh ? pcuMesh->pcpump_eff->get0() : pcuaMesh->pcpump_eff->get0());
		bool cpump_enabled = cuIsNZ(cpump_eff0) && cuIsZ(shift.z);

		cuBReal the_eff0 = (pcuMesh ? pcuMesh->pthe_eff->get0() : pcuaMesh->pthe_eff->get0());
		bool the_enabled = cuIsNZ(the_eff0) && cuIsZ(shift.z);

		if ((stsolve == STSOLVE_FERROMAGNETIC || stsolve == STSOLVE_FERROMAGNETIC_ATOM) && (cppgmr_enabled || cpump_enabled || the_enabled)) {

			//magnetic mesh

			int idx_M1 = M.position_to_cellidx(V.cellidx_to_position(cell1_idx));
			int idx_M2 = M.position_to_cellidx(V.cellidx_to_position(cell2_idx));

			cuReal3 m1 = cu_normalize(M[idx_M1]);
			cuReal3 m2 = cu_normalize(M[idx_M2]);

			//1. CPP-GMR contribution
			if (cppgmr_enabled) {

				cuBReal betaD = (pcuMesh ? *pcuMesh->pbetaD : *pcuaMesh->pbetaD);
				cuBReal De = (pcuMesh ? *pcuMesh->pDe : *pcuaMesh->pDe);

				if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pbetaD, betaD, *pcuMesh->pDe, De);
				else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pbetaD, betaD, *pcuaMesh->pDe, De);

				//value a1
				cuReal33 grad_S1 = S.grad_neu(cell1_idx);

				cuBReal a1 = ((grad_S1 * m1) * betaD * De / (cuBReal)MUB_E) * u;

				//value a2
				cuReal33 grad_S2 = S.grad_neu(cell2_idx);

				cuBReal a2 = ((grad_S2 * m2) * betaD * De / (cuBReal)MUB_E) * u;

				//final interpolated a value
				a += (1.5 * a1 - 0.5 * a2);
			}

			//2. Charge pumping
			//3. Topological Hall effect
			if (cpump_enabled || the_enabled) {

				cuBReal P = (pcuMesh ? *pcuMesh->pP : *pcuaMesh->pP);
				cuBReal n_density = (pcuMesh ? *pcuMesh->pn_density : *pcuaMesh->pn_density);
				if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pP, P, *pcuMesh->pn_density, n_density);
				else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pP, P, *pcuaMesh->pn_density, n_density);

				cuReal33 grad_m1 = cu_normalize(M.grad_neu(idx_M1), M[idx_M1]);
				cuReal33 grad_m2 = cu_normalize(M.grad_neu(idx_M2), M[idx_M2]);

				//do not read off the E field directly as it's only calculated after the spin solver (charge part) has relaxed
				cuReal3 E1 = -1.0 * V.grad_diri(cell1_idx);
				cuReal3 E2 = -1.0 * V.grad_diri(cell2_idx);

				cuBReal sigma_1 = elC[cell1_idx];
				cuBReal sigma_2 = elC[cell2_idx];

				//topological Hall effect contribution
				if (the_enabled) {

					//value a1
					cuBReal Bz_the_1 = (grad_m1.x ^ grad_m1.y) * m1;
					cuBReal a1 = the_eff0 * (-P * sigma_1 * sigma_1 * (cuBReal)HBAR_E / ((cuBReal)ECHARGE * n_density)) * cuReal3(E1.y * Bz_the_1, -E1.x * Bz_the_1, 0.0) * u;

					//value a2
					cuBReal Bz_the_2 = (grad_m2.x ^ grad_m2.y) * m2;
					cuBReal a2 = the_eff0 * (-P * sigma_2 * sigma_2 * (cuBReal)HBAR_E / ((cuBReal)ECHARGE * n_density)) * cuReal3(E1.y * Bz_the_1, -E1.x * Bz_the_1, 0.0) * u;

					//final interpolated a value
					a += (1.5 * a1 - 0.5 * a2);
				}

				//charge pumping contribution
				if (cpump_enabled) {

					//value a1
					cuReal3 dm_dt_1 = cu_normalize((*pdM_dt)[idx_M1], M[idx_M1]);
					cuBReal a1 = cpump_eff0 * (P * sigma_1 * (cuBReal)HBAR_E / 2) * cuReal3((dm_dt_1 ^ grad_m1.x) * m1, (dm_dt_1 ^ grad_m1.y) * m1, 0.0) * u;

					//value a2
					cuReal3 dm_dt_2 = cu_normalize((*pdM_dt)[idx_M2], M[idx_M2]);
					cuBReal a2 = cpump_eff0 * (P * sigma_2 * (cuBReal)HBAR_E / 2) * cuReal3((dm_dt_2 ^ grad_m2.x) * m2, (dm_dt_2 ^ grad_m2.y) * m2, 0.0) * u;

					//final interpolated a value
					a += (1.5 * a1 - 0.5 * a2);
				}
			}
		}
		else {

			//non-magnetic mesh. Only pcuMesh, but best to check
			if (pcuMesh) {

				//1. ISHE contribution
				if (cuIsNZ(pcuMesh->piSHA->get0()) && stsolve == STSOLVE_NORMALMETAL) {

					cuBReal iSHA = *pcuMesh->piSHA;
					cuBReal SHA = *pcuMesh->pSHA;
					cuBReal De = *pcuMesh->pDe;
					pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->piSHA, iSHA, *pcuMesh->pSHA, SHA, *pcuMesh->pDe, De);

					//need to find value at boundary so use interpolation

					//value a1

					//do not read off the E field directly as it's only calculated after the spin solver (charge part) has relaxed
					cuReal3 E1 = -1.0 * V.grad_diri_nneu(cell1_idx, (iSHA * De / ((cuBReal)MUB_E * elC[cell1_idx])) * S.curl_neu(cell1_idx));
					cuBReal a1 = (iSHA * De / (cuBReal)MUB_E) * S.curl_nneu(cell1_idx, cu_epsilon3(E1) * (SHA * elC[cell1_idx] * (cuBReal)MUB_E / De)) * u;

					//value a2
					cuReal3 E2 = -1.0 * V.grad_diri_nneu(cell2_idx, (iSHA * De / ((cuBReal)MUB_E * elC[cell2_idx])) * S.curl_neu(cell2_idx));
					cuBReal a2 = (iSHA * De / (cuBReal)MUB_E) * S.curl_nneu(cell2_idx, cu_epsilon3(E2) * (SHA * elC[cell2_idx] * (cuBReal)MUB_E / De)) * u;

					//final interpolated a value
					a += (1.5 * a1 - 0.5 * a2);
				}
			}
		}

		return a;
	}

	__device__ cuBReal b_func_pri(int cell1_idx, int cell2_idx)
	{
		cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);
		return (-1.5 * elC[cell1_idx] + 0.5 * elC[cell2_idx]);
	}

	//second order differential of V along the shift axis
	//this is simply Evaluate_SpinSolver_delsqV_RHS from which we subtract second order differentials orthogonal to the shift axis
	__device__ cuBReal diff2_pri(int cell1_idx, cuReal3 shift)
	{
		//normalized, positive shift: use * operator (dot product) with nshift to eliminate differentials orthogonal to the shift axis
		cuReal3 nshift = cu_mod(cu_normalize(shift));

		cuVEC_VC<cuBReal>& V = (pcuMesh ? *pcuMesh->pV : *pcuaMesh->pV);
		cuVEC_VC<cuBReal>& elC = (pcuMesh ? *pcuMesh->pelC : *pcuaMesh->pelC);
		cuVEC_VC<cuReal3>& S = (pcuMesh ? *pcuMesh->pS : *pcuaMesh->pS);
		cuVEC_VC<cuReal3>& M = (pcuMesh ? *pcuMesh->pM : *pcuaMesh->pM1);

		if (stsolve == STSOLVE_TUNNELING) return -((V.grad_diri(cell1_idx) * nshift) * (elC.grad_sided(cell1_idx) * nshift)) / elC[cell1_idx];

		cuBReal iSHA = (pcuMesh ? *pcuMesh->piSHA : 0.0);
		cuBReal De = (pcuMesh ? *pcuMesh->pDe : *pcuaMesh->pDe);

		//The Poisson solver calls this method to evaluate the RHS of this equation
		cuBReal value = 0.0;

		if (stsolve == STSOLVE_NORMALMETAL || stsolve == STSOLVE_NONE) {

			//non-magnetic mesh

			if (cuIsZ(iSHA) || stsolve == STSOLVE_NONE) {

				//1. no iSHE contribution.
				value = -((V.grad_diri(cell1_idx) * nshift) * (elC.grad_sided(cell1_idx) * nshift)) / elC[cell1_idx];
			}
			else {

				//pcuMesh only, not pcuaMesh, but best to check
				if (pcuMesh) {

					pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pDe, De, *pcuMesh->piSHA, iSHA);

					//1. iSHE enabled, must use non-homogeneous Neumann boundary condition for grad V -> Note homogeneous Neumann boundary conditions apply when calculating S differentials here (due to Jc.n = 0 at boundaries)
					value = -((V.grad_diri_nneu(cell1_idx, (iSHA * De / ((cuBReal)MUB_E * elC[cell1_idx])) * S.curl_neu(cell1_idx)) * nshift) * (elC.grad_sided(cell1_idx) * nshift)) / elC[cell1_idx];
				}
			}
		}
		else {

			//magnetic mesh

			//homogeneous Neumann boundary condition applies to V in magnetic meshes
			cuReal3 grad_V = V.grad_diri(cell1_idx);

			//1. principal term : always present
			value = -((grad_V * nshift) * (elC.grad_sided(cell1_idx) * nshift)) / elC[cell1_idx];

			cuBReal the_eff = (pcuMesh ? *pcuMesh->pthe_eff : *pcuaMesh->pthe_eff);

			//2. topological Hall effect contribution
			if (cuIsNZ(the_eff)) {

				cuBReal the_eff0 = (pcuMesh ? pcuMesh->pthe_eff->get0() : pcuaMesh->pthe_eff->get0());
				cuBReal P = (pcuMesh ? *pcuMesh->pP : *pcuaMesh->pP);
				cuBReal n_density = (pcuMesh ? *pcuMesh->pn_density : *pcuaMesh->pn_density);

				if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pP, P, *pcuMesh->pn_density, n_density);
				else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pP, P, *pcuaMesh->pn_density, n_density);

				int idx_M = M.position_to_cellidx(V.cellidx_to_position(cell1_idx));
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

				value -= (the_eff0 * P * elC[cell1_idx] * (cuBReal)HBAR_E / ((cuBReal)ECHARGE * n_density)) * ((grad_V * nshift) * (B_the * nshift));
			}

			//only calculate current on non-empty cells - empty cells have already been assigned 0 at UpdateConfiguration
			if (V.is_not_empty(cell1_idx)) {

				bool cppgmr_enabled = cuIsNZ(pcuMesh ? pcuMesh->pbetaD->get0() : pcuaMesh->pbetaD->get0());
				bool cpump_enabled = cuIsNZ(pcuMesh ? pcuMesh->pcpump_eff->get0() : pcuaMesh->pcpump_eff->get0());

				if (cppgmr_enabled || cpump_enabled) {

					int idx_M = M.position_to_cellidx(V.cellidx_to_position(cell1_idx));
					cuReal33 grad_m = cu_normalize(M.grad_neu(idx_M), M[idx_M]);
					cuReal3 m = cu_normalize(M[idx_M]);

					//CPP-GMR contribution
					if (cppgmr_enabled) {

						cuBReal De = (pcuMesh ? *pcuMesh->pDe : *pcuaMesh->pDe);
						cuBReal betaD = (pcuMesh ? *pcuMesh->pbetaD : *pcuaMesh->pbetaD);
						if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pDe, De, *pcuMesh->pbetaD, betaD);
						else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pDe, De, *pcuaMesh->pbetaD, betaD);

						cuReal33 grad_S = S.grad_neu(cell1_idx);
						cuReal3 delsq_S = S.delsq_neu(cell1_idx);
						cuBReal div_grad_S_m = (cuReal3(grad_S.i * grad_m.i, grad_S.j * grad_m.j, grad_S.k * grad_m.k) * nshift) + ((m * nshift) * (delsq_S * nshift));

						value += div_grad_S_m * betaD * De / ((cuBReal)MUB_E * elC[cell1_idx]);
					}

					//Charge pumping pre-calculation
					if (cpump_enabled) {

						cuReal33 grad_dm_dt = cu_normalize((*pdM_dt).grad_neu(idx_M), M[idx_M]);
						cuReal3 dm_dt = cu_normalize((*pdM_dt)[idx_M], M[idx_M]);

						cuBReal P = (pcuMesh ? *pcuMesh->pP : *pcuaMesh->pP);
						if (pcuMesh) pcuMesh->update_parameters_ecoarse(cell1_idx, *pcuMesh->pP, P);
						else pcuaMesh->update_parameters_ecoarse(cell1_idx, *pcuaMesh->pP, P);

						cuReal3 dx_m = grad_m.x;
						cuReal3 dy_m = grad_m.y;
						cuReal3 dxx_m = cu_normalize(M.dxx_neu(idx_M), M[idx_M]);
						cuReal3 dyy_m = cu_normalize(M.dyy_neu(idx_M), M[idx_M]);

						cuBReal cpump_eff = (pcuMesh ? pcuMesh->pcpump_eff->get0() : pcuaMesh->pcpump_eff->get0());
						value += (cpump_eff * P * (cuBReal)HBAR_E / 2) * ((grad_dm_dt.x ^ dx_m) + (grad_dm_dt.y ^ dy_m) + (dm_dt ^ (dxx_m + dyy_m))) * m;
					}
				}
			}
		}

		return value;
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//This is held as a mcu_obj managed class in TransportCUDA modules with an associated policy class
//It provides methods and access to mesh data for use in cuVEC_VC cmbnd methods
//The a_func, b_func and diff2_func methods are used to set CMBND conditions based on the continuity of a quantity and a flux.
//If V is the potential, then the flux is the function f(V) = a_func + b_func * V', where the V' differential direction is perpendicular to the interface.
//TransportCUDA_Spin_V_CMBND_Sec is used for access on the secondary side, where we are setting cmbnd values on the primary side for a given device.
//However, on the secondary side we may need access from a different device. TransportCUDA_Spin_V_CMBND_Sec allows access to data on any device using UVA.
//Thus TransportCUDA_Spin_V_CMBND_Sec differs from TransportCUDA_Spin_V_CMBND_Pri to allow access to all devices, and identify which device we need depending on the relpos_m1 passed in (which is relative to the entire cuVEC)
class TransportCUDA_Spin_V_CMBND_Sec {

public:

	//micromagnetic version : managed mesh for access to all required mesh VECs and material parameters (if nullptr then not used)
	ManagedMeshCUDA** ppcuMesh;

	//atomistic version : managed mesh for access to all required mesh VECs and material parameters (if nullptr then not used)
	ManagedAtom_MeshCUDA** ppcuaMesh;

	//dM_dt VEC when we need to do vector calculus operations on it
	//points to cuVEC in TransportCUDA or Atom_TransportCUDA
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

	//CMBND for V
	//flux = a + b V' at the interface, b = -sigma, a = betaD * (De*e/muB) * (grad S)m + (SHA*De*e/muB) * curl S + charge pumping + topological Hall effect
	//Note, the topological Hall effect term includes E, thus gradients in V, but we can include these in the a term for 2 reasons:
	//1. these CMBND functions especially with the topological Hall effect enabled is used for interfaces along z direction normally, and here Ez is zero 
	//(for such interfaces both charge pumping and topological Hall effect have zero contribution to z direction charge current)
	//2. even if the interface is along x or y we can still use the previously calculated E field, and the solution will converge to the same value (but might take more iterations).
	__device__ cuBReal a_func_sec(cuReal3 relpos_m1, cuReal3 shift, cuReal3 stencil)
	{
		if (stsolve == STSOLVE_TUNNELING) return 0.0;

		int device = 0;
		cuReal3 devrelpos_m1 = global_relpos_to_device_relpos(relpos_m1, device);

		cuVEC_VC<cuReal3>& S = (ppcuMesh ? *ppcuMesh[device]->pS : *ppcuaMesh[device]->pS);
		cuVEC_VC<cuBReal>& V = (ppcuMesh ? *ppcuMesh[device]->pV : *ppcuaMesh[device]->pV);
		cuVEC_VC<cuBReal>& elC = (ppcuMesh ? *ppcuMesh[device]->pelC : *ppcuaMesh[device]->pelC);
		cuVEC_VC<cuReal3>& M = (ppcuMesh ? *ppcuMesh[device]->pM : *ppcuaMesh[device]->pM1);

		cuBReal a = 0.0;

		cuReal3 u = shift.normalized() * -1;

		bool cppgmr_enabled = (ppcuMesh ? cuIsNZ(ppcuMesh[device]->pbetaD->get0()) : cuIsNZ(ppcuaMesh[device]->pbetaD->get0()));

		cuBReal cpump_eff0 = (ppcuMesh ? ppcuMesh[device]->pcpump_eff->get0() : ppcuaMesh[device]->pcpump_eff->get0());
		bool cpump_enabled = cuIsNZ(cpump_eff0) && cuIsZ(shift.z);

		cuBReal the_eff0 = (ppcuMesh ? ppcuMesh[device]->pthe_eff->get0() : ppcuaMesh[device]->pthe_eff->get0());
		bool the_enabled = cuIsNZ(the_eff0) && cuIsZ(shift.z);

		if ((stsolve == STSOLVE_FERROMAGNETIC || stsolve == STSOLVE_FERROMAGNETIC_ATOM) && (cppgmr_enabled || cpump_enabled || the_enabled)) {

			//magnetic mesh

			cuReal3 m1 = cu_normalize(M.weighted_average(devrelpos_m1, stencil));
			cuReal3 m2 = cu_normalize(M.weighted_average(devrelpos_m1 + shift, stencil));

			//1. CPP-GMR contribution
			if (cppgmr_enabled) {

				cuBReal betaD = (ppcuMesh ? *ppcuMesh[device]->pbetaD : *ppcuaMesh[device]->pbetaD);
				cuBReal De = (ppcuMesh ? *ppcuMesh[device]->pDe : *ppcuaMesh[device]->pDe);
				if (ppcuMesh) ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pbetaD, betaD, *ppcuMesh[device]->pDe, De);
				else ppcuaMesh[device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[device]->pbetaD, betaD, *ppcuaMesh[device]->pDe, De);

				int idx_S1 = S.position_to_cellidx(devrelpos_m1);
				int idx_S2 = S.position_to_cellidx(devrelpos_m1 + shift);

				//value a1
				cuReal33 grad_S1 = S.grad_neu(idx_S1);

				cuBReal a1 = ((grad_S1 * m1) * betaD * De / (cuBReal)MUB_E) * u;

				//value a2
				cuReal33 grad_S2 = S.grad_neu(idx_S2);

				cuBReal a2 = ((grad_S2 * m2) * betaD * De / (cuBReal)MUB_E) * u;

				//final interpolated a value
				a += (1.5 * a1 - 0.5 * a2);
			}

			//2. Charge pumping
			//3. Topological Hall effect
			if (cpump_enabled || the_enabled) {

				cuBReal P = (ppcuMesh ? *ppcuMesh[device]->pP : *ppcuaMesh[device]->pP);
				cuBReal n_density = (ppcuMesh ? *ppcuMesh[device]->pn_density : *ppcuaMesh[device]->pn_density);
				if (ppcuMesh) ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->pP, P, *ppcuMesh[device]->pn_density, n_density);
				else ppcuaMesh[device]->update_parameters_atposition(relpos_m1, *ppcuaMesh[device]->pP, P, *ppcuaMesh[device]->pn_density, n_density);

				int idx_M1 = M.position_to_cellidx(devrelpos_m1);
				int idx_M2 = M.position_to_cellidx(devrelpos_m1 + shift);

				cuReal33 grad_m1 = cu_normalize(M.grad_neu(idx_M1), M[idx_M1]);
				cuReal33 grad_m2 = cu_normalize(M.grad_neu(idx_M2), M[idx_M2]);

				int idx_V1 = V.position_to_cellidx(devrelpos_m1);
				//do not read off the E field directly as it's only calculated after the spin solver (charge part) has relaxed
				cuReal3 E1 = -1.0 * V.grad_diri(idx_V1);

				int idx_V2 = V.position_to_cellidx(devrelpos_m1 + shift);
				cuReal3 E2 = -1.0 * V.grad_diri(idx_V2);

				cuBReal sigma_1 = elC.weighted_average(devrelpos_m1, stencil);
				cuBReal sigma_2 = elC.weighted_average(devrelpos_m1 + shift, stencil);

				//topological Hall effect contribution
				if (the_enabled) {

					//value a1
					cuBReal Bz_the_1 = (grad_m1.x ^ grad_m1.y) * m1;
					cuBReal a1 = the_eff0 * (-P * sigma_1 * sigma_1 * (cuBReal)HBAR_E / ((cuBReal)ECHARGE * n_density)) * cuReal3(E1.y * Bz_the_1, -E1.x * Bz_the_1, 0.0) * u;

					//value a2
					cuBReal Bz_the_2 = (grad_m2.x ^ grad_m2.y) * m2;
					cuBReal a2 = the_eff0 * (-P * sigma_2 * sigma_2 * (cuBReal)HBAR_E / ((cuBReal)ECHARGE * n_density)) * cuReal3(E1.y * Bz_the_1, -E1.x * Bz_the_1, 0.0) * u;

					//final interpolated a value
					a += (1.5 * a1 - 0.5 * a2);
				}

				//charge pumping contribution
				if (cpump_enabled) {

					//value a1
					cuReal3 dm_dt_1 = cu_normalize(ppdM_dt[device]->weighted_average(devrelpos_m1, stencil), M[idx_M1]);
					cuBReal a1 = cpump_eff0 * (P * sigma_1 * (cuBReal)HBAR_E / 2) * cuReal3((dm_dt_1 ^ grad_m1.x) * m1, (dm_dt_1 ^ grad_m1.y) * m1, 0.0) * u;

					//value a2
					cuReal3 dm_dt_2 = cu_normalize(ppdM_dt[device]->weighted_average(devrelpos_m1 + shift, stencil), M[idx_M2]);
					cuBReal a2 = cpump_eff0 * (P * sigma_2 * (cuBReal)HBAR_E / 2) * cuReal3((dm_dt_2 ^ grad_m2.x) * m2, (dm_dt_2 ^ grad_m2.y) * m2, 0.0) * u;

					//final interpolated a value
					a += (1.5 * a1 - 0.5 * a2);
				}
			}
		}
		else {

			//non-magnetic mesh. Only pcuMesh, but best to check
			if (ppcuMesh) {

				//1. ISHE contribution
				if (cuIsNZ(ppcuMesh[device]->piSHA->get0()) && stsolve == STSOLVE_NORMALMETAL) {

					cuBReal iSHA = *ppcuMesh[device]->piSHA;
					cuBReal SHA = *ppcuMesh[device]->pSHA;
					cuBReal De = *ppcuMesh[device]->pDe;
					ppcuMesh[device]->update_parameters_atposition(relpos_m1, *ppcuMesh[device]->piSHA, iSHA, *ppcuMesh[device]->pSHA, SHA, *ppcuMesh[device]->pDe, De);

					int idx_S1 = V.position_to_cellidx(devrelpos_m1);
					int idx_S2 = V.position_to_cellidx(devrelpos_m1 + shift);

					//need to find value at boundary so use interpolation

					//value a1
					cuBReal sigma_1 = elC.weighted_average(devrelpos_m1, stencil);
					//do not read off the E field directly as it's only calculated after the spin solver (charge part) has relaxed
					cuReal3 E1 = -1.0 * V.grad_diri_nneu(idx_S1, (iSHA * De / ((cuBReal)MUB_E * sigma_1)) * S.curl_neu(idx_S1));
					cuBReal a1 = (iSHA * De / (cuBReal)MUB_E) * S.curl_nneu(idx_S1, cu_epsilon3(E1) * (SHA * sigma_1 * (cuBReal)MUB_E / De)) * u;

					//value a2
					cuBReal sigma_2 = elC.weighted_average(devrelpos_m1 + shift, stencil);
					cuReal3 E2 = -1.0 * V.grad_diri_nneu(idx_S2, (iSHA * De / ((cuBReal)MUB_E * sigma_2)) * S.curl_neu(idx_S2));
					cuBReal a2 = (iSHA * De / (cuBReal)MUB_E) * S.curl_nneu(idx_S2, cu_epsilon3(E2) * (SHA * sigma_2 * (cuBReal)MUB_E / De)) * u;

					//final interpolated a value
					a += (1.5 * a1 - 0.5 * a2);
				}
			}
		}

		return a;
	}

	__device__ cuBReal b_func_sec(cuReal3 relpos_m1, cuReal3 shift, cuReal3 stencil)
	{
		int device = 0;
		cuReal3 devrelpos_m1 = global_relpos_to_device_relpos(relpos_m1, device);

		cuVEC_VC<cuBReal>& elC = (ppcuMesh ? *ppcuMesh[device]->pelC : *ppcuaMesh[device]->pelC);
		return (-1.5 * elC.weighted_average(devrelpos_m1, stencil) + 0.5 * elC.weighted_average(devrelpos_m1 + shift, stencil));
	}

	//second order differential of V along the shift axis
	//this is simply Evaluate_SpinSolver_delsqV_RHS from which we subtract second order differentials orthogonal to the shift axis
	__device__ cuBReal diff2_pri(int cell1_idx, cuReal3 shift, int device)
	{
		//normalized, positive shift: use * operator (dot product) with nshift to eliminate differentials orthogonal to the shift axis
		cuReal3 nshift = cu_mod(cu_normalize(shift));

		cuVEC_VC<cuBReal>& V = (ppcuMesh ? *ppcuMesh[device]->pV : *ppcuaMesh[device]->pV);
		cuVEC_VC<cuBReal>& elC = (ppcuMesh ? *ppcuMesh[device]->pelC : *ppcuaMesh[device]->pelC);
		cuVEC_VC<cuReal3>& S = (ppcuMesh ? *ppcuMesh[device]->pS : *ppcuaMesh[device]->pS);
		cuVEC_VC<cuReal3>& M = (ppcuMesh ? *ppcuMesh[device]->pM : *ppcuaMesh[device]->pM1);

		if (stsolve == STSOLVE_TUNNELING) return -((V.grad_diri(cell1_idx) * nshift) * (elC.grad_sided(cell1_idx) * nshift)) / elC[cell1_idx];

		cuBReal iSHA = (ppcuMesh ? *ppcuMesh[device]->piSHA : 0.0);
		cuBReal De = (ppcuMesh ? *ppcuMesh[device]->pDe : *ppcuaMesh[device]->pDe);

		//The Poisson solver calls this method to evaluate the RHS of this equation
		cuBReal value = 0.0;

		if (stsolve == STSOLVE_NORMALMETAL || stsolve == STSOLVE_NONE) {

			//non-magnetic mesh

			if (cuIsZ(iSHA) || stsolve == STSOLVE_NONE) {

				//1. no iSHE contribution.
				value = -((V.grad_diri(cell1_idx) * nshift) * (elC.grad_sided(cell1_idx) * nshift)) / elC[cell1_idx];
			}
			else {

				//pcuMesh only, not pcuaMesh, but best to check
				if (ppcuMesh) {

					ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pDe, De, *ppcuMesh[device]->piSHA, iSHA);

					//1. iSHE enabled, must use non-homogeneous Neumann boundary condition for grad V -> Note homogeneous Neumann boundary conditions apply when calculating S differentials here (due to Jc.n = 0 at boundaries)
					value = -((V.grad_diri_nneu(cell1_idx, (iSHA * De / ((cuBReal)MUB_E * elC[cell1_idx])) * S.curl_neu(cell1_idx)) * nshift) * (elC.grad_sided(cell1_idx) * nshift)) / elC[cell1_idx];
				}
			}
		}
		else {

			//magnetic mesh

			//homogeneous Neumann boundary condition applies to V in magnetic meshes
			cuReal3 grad_V = V.grad_diri(cell1_idx);

			//1. principal term : always present
			value = -((grad_V * nshift) * (elC.grad_sided(cell1_idx) * nshift)) / elC[cell1_idx];

			cuBReal the_eff = (ppcuMesh ? *ppcuMesh[device]->pthe_eff : *ppcuaMesh[device]->pthe_eff);

			//2. topological Hall effect contribution
			if (cuIsNZ(the_eff)) {

				cuBReal the_eff0 = (ppcuMesh ? ppcuMesh[device]->pthe_eff->get0() : ppcuaMesh[device]->pthe_eff->get0());
				cuBReal P = (ppcuMesh ? *ppcuMesh[device]->pP : *ppcuaMesh[device]->pP);
				cuBReal n_density = (ppcuMesh ? *ppcuMesh[device]->pn_density : *ppcuaMesh[device]->pn_density);

				if (ppcuMesh) ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pP, P, *ppcuMesh[device]->pn_density, n_density);
				else ppcuaMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuaMesh[device]->pP, P, *ppcuaMesh[device]->pn_density, n_density);

				int idx_M = M.position_to_cellidx(V.cellidx_to_position(cell1_idx));
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

				value -= (the_eff0 * P * elC[cell1_idx] * (cuBReal)HBAR_E / ((cuBReal)ECHARGE * n_density)) * ((grad_V * nshift) * (B_the * nshift));
			}

			//only calculate current on non-empty cells - empty cells have already been assigned 0 at UpdateConfiguration
			if (V.is_not_empty(cell1_idx)) {

				bool cppgmr_enabled = cuIsNZ(ppcuMesh ? ppcuMesh[device]->pbetaD->get0() : ppcuaMesh[device]->pbetaD->get0());
				bool cpump_enabled = cuIsNZ(ppcuMesh ? ppcuMesh[device]->pcpump_eff->get0() : ppcuaMesh[device]->pcpump_eff->get0());

				if (cppgmr_enabled || cpump_enabled) {

					int idx_M = M.position_to_cellidx(V.cellidx_to_position(cell1_idx));
					cuReal33 grad_m = cu_normalize(M.grad_neu(idx_M), M[idx_M]);
					cuReal3 m = cu_normalize(M[idx_M]);

					//CPP-GMR contribution
					if (cppgmr_enabled) {

						cuBReal De = (ppcuMesh ? *ppcuMesh[device]->pDe : *ppcuaMesh[device]->pDe);
						cuBReal betaD = (ppcuMesh ? *ppcuMesh[device]->pbetaD : *ppcuaMesh[device]->pbetaD);
						if (ppcuMesh) ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pDe, De, *ppcuMesh[device]->pbetaD, betaD);
						else ppcuaMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuaMesh[device]->pDe, De, *ppcuaMesh[device]->pbetaD, betaD);

						cuReal33 grad_S = S.grad_neu(cell1_idx);
						cuReal3 delsq_S = S.delsq_neu(cell1_idx);
						cuBReal div_grad_S_m = (cuReal3(grad_S.i * grad_m.i, grad_S.j * grad_m.j, grad_S.k * grad_m.k) * nshift) + ((m * nshift) * (delsq_S * nshift));

						value += div_grad_S_m * betaD * De / ((cuBReal)MUB_E * elC[cell1_idx]);
					}

					//Charge pumping pre-calculation
					if (cpump_enabled) {

						cuReal33 grad_dm_dt = cu_normalize((*ppdM_dt[device]).grad_neu(idx_M), M[idx_M]);
						cuReal3 dm_dt = cu_normalize((*ppdM_dt[device])[idx_M], M[idx_M]);

						cuBReal P = (ppcuMesh ? *ppcuMesh[device]->pP : *ppcuaMesh[device]->pP);
						if (ppcuMesh) ppcuMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuMesh[device]->pP, P);
						else ppcuaMesh[device]->update_parameters_ecoarse(cell1_idx, *ppcuaMesh[device]->pP, P);

						cuReal3 dx_m = grad_m.x;
						cuReal3 dy_m = grad_m.y;
						cuReal3 dxx_m = cu_normalize(M.dxx_neu(idx_M), M[idx_M]);
						cuReal3 dyy_m = cu_normalize(M.dyy_neu(idx_M), M[idx_M]);

						cuBReal cpump_eff = (ppcuMesh ? ppcuMesh[device]->pcpump_eff->get0() : ppcuaMesh[device]->pcpump_eff->get0());
						value += (cpump_eff * P * (cuBReal)HBAR_E / 2) * ((grad_dm_dt.x ^ dx_m) + (grad_dm_dt.y ^ dy_m) + (dm_dt ^ (dxx_m + dyy_m))) * m;
					}
				}
			}
		}

		return value;
	}

	//second order differential of V along the shift axis
	//this is simply Evaluate_SpinSolver_delsqV_RHS from which we subtract second order differentials orthogonal to the shift axis
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
