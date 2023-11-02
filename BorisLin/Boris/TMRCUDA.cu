#include "TMRCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TMR

#include "mcuVEC_Managed_avg.cuh"
#include "cuVEC_VC_mcuVEC.cuh"
#include "TEquationCUDA_Function.cuh"
#include "mcuVEC_halo.cuh"

#include "MeshCUDA.h"
#include "MeshParamsControlCUDA.h"
#include "ManagedAtom_MeshCUDA.h"

//--------------------------------------------------------------- TMR computation Launcher

__global__ void CalculateElectricalConductivity_TMR_COS_Kernel(
	ManagedMeshCUDA& cuMesh,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal>* pRAtmr_p_equation,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal>* pRAtmr_ap_equation,
	ManagedMeshCUDA** ppMeshFM_Top, size_t pMeshFM_Top_size, ManagedMeshCUDA** ppMeshFM_Bot, size_t pMeshFM_Bot_size,
	ManagedAtom_MeshCUDA** ppMeshAtom_Top, size_t pMeshAtom_Top_size, ManagedAtom_MeshCUDA** ppMeshAtom_Bot, size_t pMeshAtom_Bot_size)
{
	cuVEC_VC<cuBReal>& elC = *cuMesh.pelC;
	cuVEC_VC<cuBReal>& V = *cuMesh.pV;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n_e = elC.n;
	cuReal3 h_e = elC.h;
	cuRect meshRect = elC.rect;

	if (idx < n_e.x * n_e.y) {

		//expecting elC to be 2D
		int i = idx % n_e.x;
		int j = idx / n_e.x;

		int idx = i + j * n_e.x;

		//skip empty cells
		if (elC.is_not_empty(idx)) {

			//m direction values for top and bottom, used to calculate TMR in this cell
			cuReal3 m_top = cuReal3(), m_bot = cuReal3();

			//TOP

			//Look at FM meshes first, if any
			for (int mesh_idx = 0; mesh_idx < pMeshFM_Top_size; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Top = ppMeshFM_Top[mesh_idx]->pM->mcuvec();
				cuRect tmeshRect = M_Top.rect;

				//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h_e.x + meshRect.s.x - tmeshRect.s.x,
					(j + 0.5) * h_e.y + meshRect.s.y - tmeshRect.s.y,
					M_Top.h.z / 2);

				if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s)) continue;

				m_top = cu_normalize(M_Top.weighted_average(cell_rel_pos, cuReal3(h_e.x, h_e.y, M_Top.h.z)));
				break;
			}

			if (m_top == cuReal3()) {

				//Look at Atomistic meshes, if any
				for (int mesh_idx = 0; mesh_idx < pMeshAtom_Top_size; mesh_idx++) {

					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Top = ppMeshAtom_Top[mesh_idx]->pM1->mcuvec();
					cuRect tmeshRect = M_Top.rect;

					//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
					cuReal3 cell_rel_pos = cuReal3(
						(i + 0.5) * h_e.x + meshRect.s.x - tmeshRect.s.x,
						(j + 0.5) * h_e.y + meshRect.s.y - tmeshRect.s.y,
						M_Top.h.z / 2);

					if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s)) continue;

					m_top = cu_normalize(M_Top.weighted_average(cell_rel_pos, cuReal3(h_e.x, h_e.y, M_Top.h.z)));
					break;
				}
			}

			//BOTTOM

			//Look at FM meshes first, if any
			for (int mesh_idx = 0; mesh_idx < pMeshFM_Bot_size; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bot = ppMeshFM_Bot[mesh_idx]->pM->mcuvec();
				cuRect bmeshRect = M_Bot.rect;

				//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h_e.x + meshRect.s.x - bmeshRect.s.x,
					(j + 0.5) * h_e.y + meshRect.s.y - bmeshRect.s.y,
					bmeshRect.e.z - bmeshRect.s.z - (M_Bot.h.z / 2));

				//can't couple to an empty cell
				if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s)) continue;

				m_bot = cu_normalize(M_Bot.weighted_average(cell_rel_pos, cuReal3(h_e.x, h_e.y, M_Bot.h.z)));
				break;
			}

			if (m_bot == cuReal3()) {

				//Look at Atomistic meshes, if any
				for (int mesh_idx = 0; mesh_idx < pMeshAtom_Bot_size; mesh_idx++) {

					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bot = ppMeshAtom_Bot[mesh_idx]->pM1->mcuvec();
					cuRect bmeshRect = M_Bot.rect;

					//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
					cuReal3 cell_rel_pos = cuReal3(
						(i + 0.5) * h_e.x + meshRect.s.x - bmeshRect.s.x,
						(j + 0.5) * h_e.y + meshRect.s.y - bmeshRect.s.y,
						bmeshRect.e.z - bmeshRect.s.z - (M_Bot.h.z / 2));

					//can't couple to an empty cell
					if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s)) continue;

					m_bot = cu_normalize(M_Bot.weighted_average(cell_rel_pos, cuReal3(h_e.x, h_e.y, M_Bot.h.z)));
					break;
				}
			}

			//now apply TMR formula to store conductivity value

			//cos dependence of RA product (where dRA = RAap - RAp):
			//RA = (RAp + dRA * (1 - m1.m2)/2)
			//so resistivity is (thickness t):
			//ro = (RAp + dRA * (1 - m1.m2)/2) / t
			//so set conductivity as: 1 / ro

			cuBReal RAtmr_p = *cuMesh.pRAtmr_p;
			cuBReal RAtmr_ap = *cuMesh.pRAtmr_ap;
			cuBReal elecCond = *cuMesh.pelecCond;
			cuMesh.update_parameters_ecoarse(idx, *cuMesh.pRAtmr_p, RAtmr_p, *cuMesh.pRAtmr_ap, RAtmr_ap, *cuMesh.pelecCond, elecCond);

			//RA bias dependence if set
			if (pRAtmr_p_equation || pRAtmr_ap_equation) {

				cuBReal bias = 0.0;

				if (V.linear_size()) {

					cuBReal Vt_1 = V[idx + (n_e.z - 1) * n_e.x * n_e.y];
					cuBReal Vt_2 = V[idx + (n_e.z - 2) * n_e.x * n_e.y];

					cuBReal Vb_1 = V[idx];
					cuBReal Vb_2 = V[idx + n_e.x * n_e.y];

					cuBReal Vt = 1.5 * Vt_1 - 0.5 * Vt_2;
					cuBReal Vb = 1.5 * Vb_1 - 0.5 * Vb_2;

					bias = Vt - Vb;
				}

				cuBReal RAtmr_p0 = RAtmr_p;
				cuBReal RAtmr_ap0 = RAtmr_ap;

				if (pRAtmr_p_equation) RAtmr_p = pRAtmr_p_equation->evaluate(RAtmr_p0, RAtmr_ap0, bias);
				if (pRAtmr_ap_equation) RAtmr_ap = pRAtmr_ap_equation->evaluate(RAtmr_p0, RAtmr_ap0, bias);
			}

			for (int k = 0; k < n_e.k; k++) {

				if (elecCond > 0.0) {

					//Metallic pinholes
					elC[idx + k * n_e.x * n_e.y] = elecCond;
				}
				else {

					//TMR
					elC[idx + k * n_e.x * n_e.y] = meshRect.height() / (RAtmr_p + (RAtmr_ap - RAtmr_p) * (1 - m_top * m_bot) / 2);
				}
			}
		}
	}
}

__global__ void CalculateElectricalConductivity_TMR_SLONCZEWSKI_Kernel(
	ManagedMeshCUDA& cuMesh,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal>* pRAtmr_p_equation,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal>* pRAtmr_ap_equation,
	ManagedMeshCUDA** ppMeshFM_Top, size_t pMeshFM_Top_size, ManagedMeshCUDA** ppMeshFM_Bot, size_t pMeshFM_Bot_size,
	ManagedAtom_MeshCUDA** ppMeshAtom_Top, size_t pMeshAtom_Top_size, ManagedAtom_MeshCUDA** ppMeshAtom_Bot, size_t pMeshAtom_Bot_size)
{
	cuVEC_VC<cuBReal>& elC = *cuMesh.pelC;
	cuVEC_VC<cuBReal>& V = *cuMesh.pV;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n_e = elC.n;
	cuReal3 h_e = elC.h;
	cuRect meshRect = elC.rect;

	if (idx < n_e.x * n_e.y) {

		//expecting elC to be 2D
		int i = idx % n_e.x;
		int j = idx / n_e.x;

		int idx = i + j * n_e.x;

		//skip empty cells
		if (elC.is_not_empty(idx)) {

			//m direction values for top and bottom, used to calculate TMR in this cell
			cuReal3 m_top = cuReal3(), m_bot = cuReal3();

			//TOP

			//Look at FM meshes first, if any
			for (int mesh_idx = 0; mesh_idx < pMeshFM_Top_size; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Top = ppMeshFM_Top[mesh_idx]->pM->mcuvec();
				cuRect tmeshRect = M_Top.rect;

				//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h_e.x + meshRect.s.x - tmeshRect.s.x,
					(j + 0.5) * h_e.y + meshRect.s.y - tmeshRect.s.y,
					M_Top.h.z / 2);

				if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s)) continue;

				m_top = cu_normalize(M_Top.weighted_average(cell_rel_pos, cuReal3(h_e.x, h_e.y, M_Top.h.z)));
				break;
			}

			if (m_top == cuReal3()) {

				//Look at Atomistic meshes, if any
				for (int mesh_idx = 0; mesh_idx < pMeshAtom_Top_size; mesh_idx++) {

					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Top = ppMeshAtom_Top[mesh_idx]->pM1->mcuvec();
					cuRect tmeshRect = M_Top.rect;

					//relative coordinates to read value from top mesh (the one we're coupling to here) - relative to top mesh
					cuReal3 cell_rel_pos = cuReal3(
						(i + 0.5) * h_e.x + meshRect.s.x - tmeshRect.s.x,
						(j + 0.5) * h_e.y + meshRect.s.y - tmeshRect.s.y,
						M_Top.h.z / 2);

					if (!tmeshRect.contains(cell_rel_pos + tmeshRect.s)) continue;

					m_top = cu_normalize(M_Top.weighted_average(cell_rel_pos, cuReal3(h_e.x, h_e.y, M_Top.h.z)));
					break;
				}
			}

			//BOTTOM

			//Look at FM meshes first, if any
			for (int mesh_idx = 0; mesh_idx < pMeshFM_Bot_size; mesh_idx++) {

				mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bot = ppMeshFM_Bot[mesh_idx]->pM->mcuvec();
				cuRect bmeshRect = M_Bot.rect;

				//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
				cuReal3 cell_rel_pos = cuReal3(
					(i + 0.5) * h_e.x + meshRect.s.x - bmeshRect.s.x,
					(j + 0.5) * h_e.y + meshRect.s.y - bmeshRect.s.y,
					bmeshRect.e.z - bmeshRect.s.z - (M_Bot.h.z / 2));

				//can't couple to an empty cell
				if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s)) continue;

				m_bot = cu_normalize(M_Bot.weighted_average(cell_rel_pos, cuReal3(h_e.x, h_e.y, M_Bot.h.z)));
				break;
			}

			if (m_bot == cuReal3()) {

				//Look at Atomistic meshes, if any
				for (int mesh_idx = 0; mesh_idx < pMeshAtom_Bot_size; mesh_idx++) {

					mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M_Bot = ppMeshAtom_Bot[mesh_idx]->pM1->mcuvec();
					cuRect bmeshRect = M_Bot.rect;

					//relative coordinates to read value from bottom mesh (the one we're coupling to here) - relative to bottom mesh
					cuReal3 cell_rel_pos = cuReal3(
						(i + 0.5) * h_e.x + meshRect.s.x - bmeshRect.s.x,
						(j + 0.5) * h_e.y + meshRect.s.y - bmeshRect.s.y,
						bmeshRect.e.z - bmeshRect.s.z - (M_Bot.h.z / 2));

					//can't couple to an empty cell
					if (!bmeshRect.contains(cell_rel_pos + bmeshRect.s)) continue;

					m_bot = cu_normalize(M_Bot.weighted_average(cell_rel_pos, cuReal3(h_e.x, h_e.y, M_Bot.h.z)));
					break;
				}
			}

			//now apply TMR formula to store conductivity value

			//Slonczewski form : cos dependence of conductivity
			//RA = 2*RAp / [ (1 + RAp/RAap) + (1 - RAp/RAap)cos(theta) ]

			cuBReal RAtmr_p = *cuMesh.pRAtmr_p;
			cuBReal RAtmr_ap = *cuMesh.pRAtmr_ap;
			cuBReal elecCond = *cuMesh.pelecCond;
			cuMesh.update_parameters_ecoarse(idx, *cuMesh.pRAtmr_p, RAtmr_p, *cuMesh.pRAtmr_ap, RAtmr_ap, *cuMesh.pelecCond, elecCond);

			//RA bias dependence if set
			if (pRAtmr_p_equation || pRAtmr_ap_equation) {

				cuBReal bias = 0.0;

				if (V.linear_size()) {

					cuBReal Vt_1 = V[idx + (n_e.z - 1) * n_e.x * n_e.y];
					cuBReal Vt_2 = V[idx + (n_e.z - 2) * n_e.x * n_e.y];

					cuBReal Vb_1 = V[idx];
					cuBReal Vb_2 = V[idx + n_e.x * n_e.y];

					cuBReal Vt = 1.5 * Vt_1 - 0.5 * Vt_2;
					cuBReal Vb = 1.5 * Vb_1 - 0.5 * Vb_2;

					bias = Vt - Vb;
				}

				cuBReal RAtmr_p0 = RAtmr_p;
				cuBReal RAtmr_ap0 = RAtmr_ap;

				if (pRAtmr_p_equation) RAtmr_p = pRAtmr_p_equation->evaluate(RAtmr_p0, RAtmr_ap0, bias);
				if (pRAtmr_ap_equation) RAtmr_ap = pRAtmr_ap_equation->evaluate(RAtmr_p0, RAtmr_ap0, bias);
			}

			for (int k = 0; k < n_e.k; k++) {

				if (elecCond > 0.0) {

					//Metallic pinholes
					elC[idx + k * n_e.x * n_e.y] = elecCond;
				}
				else {

					//TMR
					elC[idx + k * n_e.x * n_e.y] = meshRect.height() * ((1 + RAtmr_p / RAtmr_ap) + (1 - RAtmr_p / RAtmr_ap) * m_top * m_bot) / (2 * RAtmr_p);
				}
			}
		}
	}
}

//calculate electrical conductivity based on TMR formula
void TMRCUDA::CalculateElectricalConductivity_TMR(TMR_ TMR_type)
{
	switch (TMR_type) {

	case TMR_COS:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			cuSZ3 dn_e = pMeshCUDA->elC.device_n(mGPU);
			CalculateElectricalConductivity_TMR_COS_Kernel <<< (dn_e.x * dn_e.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU),
					(RAtmr_p_equation.is_set() ? &RAtmr_p_equation.get_x(mGPU) : nullptr), (RAtmr_ap_equation.is_set() ? &RAtmr_ap_equation.get_x(mGPU) : nullptr),
					pMeshFM_Top.get_array(mGPU), pMeshFM_Top.size(mGPU), pMeshFM_Bot.get_array(mGPU), pMeshFM_Bot.size(mGPU),
					pMeshAtom_Top.get_array(mGPU), pMeshAtom_Top.size(mGPU), pMeshAtom_Bot.get_array(mGPU), pMeshAtom_Bot.size(mGPU));

		}
		break;

	case TMR_SLONCZEWSKI:

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			cuSZ3 dn_e = pMeshCUDA->elC.device_n(mGPU);
			CalculateElectricalConductivity_TMR_SLONCZEWSKI_Kernel <<< (dn_e.x * dn_e.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU),
					(RAtmr_p_equation.is_set() ? &RAtmr_p_equation.get_x(mGPU) : nullptr), (RAtmr_ap_equation.is_set() ? &RAtmr_ap_equation.get_x(mGPU) : nullptr),
					pMeshFM_Top.get_array(mGPU), pMeshFM_Top.size(mGPU), pMeshFM_Bot.get_array(mGPU), pMeshFM_Bot.size(mGPU),
					pMeshAtom_Top.get_array(mGPU), pMeshAtom_Top.size(mGPU), pMeshAtom_Bot.get_array(mGPU), pMeshAtom_Bot.size(mGPU));
		}
		break;
	}
}

//-------------------Calculation Methods : Electric Field

__global__ void CalculateElectricField_TMR_Kernel(cuVEC<cuReal3>& E, cuVEC_VC<cuBReal>& V)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < V.linear_size()) {

		//only calculate current on non-empty cells - empty cells have already been assigned 0 at UpdateConfiguration
		if (V.is_not_empty(idx)) {

			E[idx] = -1.0 * V.grad_diri(idx);
		}
		else E[idx] = cuReal3(0.0);
	}
}

//calculate electric field as the negative gradient of V
void TMRCUDA::CalculateElectricField(bool open_potential)
{
	pMeshCUDA->V.exchange_halos();

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		CalculateElectricField_TMR_Kernel <<< (pMeshCUDA->elC.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pMeshCUDA->E.get_deviceobject(mGPU), pMeshCUDA->V.get_deviceobject(mGPU));
	}
}

//-------------------Others

BError TMRCUDA::SetBiasEquation_Parallel(const std::vector< std::vector<EqComp::FSPEC> >& fspec)
{
	BError error(CLASS_STR(TMRCUDA));

	if (!fspec.size()) RAtmr_p_equation.clear();
	else if (!RAtmr_p_equation.make_scalar(fspec)) return error(BERROR_OUTOFGPUMEMORY_CRIT);

	return error;
}

BError TMRCUDA::SetBiasEquation_AntiParallel(const std::vector< std::vector<EqComp::FSPEC> >& fspec)
{
	BError error(CLASS_STR(TMRCUDA));

	if (!fspec.size()) RAtmr_ap_equation.clear();
	else if (!RAtmr_ap_equation.make_scalar(fspec)) return error(BERROR_OUTOFGPUMEMORY_CRIT);

	return error;
}

//-------------------Auxiliary

//in order to compute differential operators, halos must be exchange on respective quantites
//these functions exchange all halos
//charge solver only
void TMRCUDA::exchange_all_halos_charge(void)
{
	pMeshCUDA->V.exchange_halos();
	pMeshCUDA->elC.exchange_halos();
}

//spin solver
void TMRCUDA::exchange_all_halos_spin(void)
{
	pMeshCUDA->V.exchange_halos();
	pMeshCUDA->elC.exchange_halos();
	pMeshCUDA->S.exchange_halos();
}

#endif

#endif