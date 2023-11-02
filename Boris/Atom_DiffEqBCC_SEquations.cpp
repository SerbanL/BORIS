#include "stdafx.h"
#include "Atom_DiffEqBCC.h"

#if ATOMISTIC == 1
#ifdef MESH_COMPILATION_ATOM_BCC

#include "Atom_Mesh_BCC.h"
#include "SuperMesh.h"
#include "Atom_MeshParamsControl_Multi.h"

//------------------------------------------------------------------------------------------------------ THERMAL VECs GENERATIONS

//
// Thermal field given as :
//
// Bth = rand * SQRT( 2*alpha* kB*T / (|gamma_e|*mu_s*dT)) (T)

void Atom_DifferentialEquationBCC::GenerateThermalField(void)
{
	//if not in linked dTstoch mode, then only generate stochastic field at a minimum of dTstoch spacing
	if (!link_dTstoch && GetTime() < time_stoch + dTstoch) return;

	double deltaT = (link_dTstoch ? dT : GetTime() - time_stoch);
	time_stoch = GetTime();

	double grel = paMesh->grel.get0();

	if (IsNZ(grel)) {

		double Temperature = paMesh->GetBaseTemperature();

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				if (paMesh->Temp.linear_size()) Temperature = paMesh->Temp[H_Thermal.cellidx_to_position(idx)];
				
				for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

					double mu_s = paMesh->mu_s(sidx);
					double s_eff = paMesh->s_eff(sidx);
					paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s, paMesh->s_eff, s_eff);

					//do not include any damping here - this will be included in the stochastic equations
					double Hth_const = s_eff * sqrt(2 * BOLTZMANN * Temperature / (MUB_MU0 * GAMMA * grel * mu_s * deltaT));

					H_Thermal(sidx, idx) = Hth_const * DBL3(prng.rand_gauss(0, 1), prng.rand_gauss(0, 1), prng.rand_gauss(0, 1));
				}
			}
		}
	}
}

//------------------------------------------------------------------------------------------------------ STOCHASTIC EQUATIONS

DBL3 Atom_DifferentialEquationBCC::SLLG(int sidx, int idx)
{
	//gamma_e is the electron gyromagnetic ratio, which is negative

	//LLG in explicit form : dm/dt = [gamma_e/(1+alpha^2)] * [m*B + alpha * m*(m*B) / mu_s]
	//B is effective field in Tesla; prefer to include the effective field in A/m as it integrates easier in a multiscale simulation; thus gamma = |gamma_e| * mu0
	//m is atomic moment in units of muB : has magnitude mu_s (muB)

	double mu_s = paMesh->mu_s(sidx);
	double alpha = paMesh->alpha(sidx);
	double grel = paMesh->grel(sidx);
	paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s, paMesh->alpha, alpha, paMesh->grel, grel);

	//H_Thermal has same dimensions as M1 in atomistic meshes
	DBL3 H_Thermal_Value = H_Thermal(sidx, idx) * sqrt(alpha);

	return (-GAMMA * grel / (1 + alpha * alpha)) * ((paMesh->M1(sidx, idx) ^ (paMesh->Heff1(sidx, idx) + H_Thermal_Value)) + alpha * ((paMesh->M1(sidx, idx) / mu_s) ^ (paMesh->M1(sidx, idx) ^ (paMesh->Heff1(sidx, idx) + H_Thermal_Value))));
}

DBL3 Atom_DifferentialEquationBCC::SLLGSTT(int sidx, int idx)
{
	//gamma_e is the electron gyromagnetic ratio, which is negative

	//LLG in explicit form : dm/dt = [gamma_e/(1+alpha^2)] * [m*B + alpha * m*(m*B) / mu_s]
	//B is effective field in Tesla; prefer to include the effective field in A/m as it integrates easier in a multiscale simulation; thus gamma = |gamma_e| * mu0
	//m is atomic moment in units of muB : has magnitude mu_s (muB)

	double mu_s = paMesh->mu_s(sidx);
	double alpha = paMesh->alpha(sidx);
	double grel = paMesh->grel(sidx);
	double P = paMesh->P;
	double beta = paMesh->beta;
	paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s, paMesh->alpha, alpha, paMesh->grel, grel, paMesh->P, P, paMesh->beta, beta);

	//H_Thermal has same dimensions as M1 in atomistic meshes
	DBL3 H_Thermal_Value = H_Thermal(sidx, idx) * sqrt(alpha);

	DBL3 LLGSTT_Eval = (-GAMMA * grel / (1 + alpha * alpha)) * ((paMesh->M1(sidx, idx) ^ (paMesh->Heff1(sidx, idx) + H_Thermal_Value)) + alpha * ((paMesh->M1(sidx, idx) / mu_s) ^ (paMesh->M1(sidx, idx) ^ (paMesh->Heff1(sidx, idx) + H_Thermal_Value))));

	if (paMesh->E.linear_size()) {

		DBL33 grad_M1 = paMesh->M1.grad_neu(idx);

		DBL3 position = paMesh->M1.cellidx_to_position(idx);

		double conv = paMesh->M1.h.dim() / MUB;
		DBL3 u = (paMesh->elC[position] * paMesh->E.weighted_average(position, paMesh->h) * P * GMUB_2E * conv) / (mu_s * (1 + beta * beta));

		DBL3 u_dot_del_M1 = (u.x * grad_M1.x) + (u.y * grad_M1.y) + (u.z * grad_M1.z);

		DBL3 stt_value = (((1 + alpha * beta) * u_dot_del_M1) -
			((beta - alpha) * ((paMesh->M1(sidx, idx) / mu_s) ^ u_dot_del_M1))) / (1 + alpha * alpha);

		LLGSTT_Eval += stt_value;
	}

	return LLGSTT_Eval;
}
#endif
#endif