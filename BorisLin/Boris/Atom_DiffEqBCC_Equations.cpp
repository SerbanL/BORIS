#include "stdafx.h"
#include "Atom_DiffEqBCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC

#include "Atom_Mesh_BCC.h"
#include "SuperMesh.h"
#include "Atom_MeshParamsControl_Multi.h"

//------------------------------------------------------------------------------------------------------

DBL3 Atom_DifferentialEquationBCC::LLG(int sidx, int idx)
{
	//gamma_e is the electron gyromagnetic ratio, which is negative

	//LLG in explicit form : dm/dt = [gamma_e/(1+alpha^2)] * [m*B + alpha * m*(m*B) / mu_s]
	//B is effective field in Tesla; prefer to include the effective field in A/m as it integrates easier in a multiscale simulation; thus gamma = |gamma_e| * mu0
	//m is atomic moment in units of muB : has magnitude mu_s (muB)
	
	double mu_s = paMesh->mu_s(sidx);
	double alpha = paMesh->alpha(sidx);
	double grel = paMesh->grel(sidx);
	paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s, paMesh->alpha, alpha, paMesh->grel, grel);

	return (-GAMMA * grel / (1 + alpha*alpha)) * ((paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx)) + alpha * ((paMesh->M1(sidx, idx) / mu_s) ^ (paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx))));
}

//Landau-Lifshitz-Gilbert equation but with no precession term and damping set to 1 : faster relaxation for static problems
DBL3 Atom_DifferentialEquationBCC::LLGStatic(int sidx, int idx)
{
	double mu_s = paMesh->mu_s(sidx);
	double grel = paMesh->grel(sidx);
	paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s, paMesh->grel, grel);

	return (-GAMMA * grel / 2) * ((paMesh->M1(sidx, idx) / mu_s) ^ (paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx)));
}

DBL3 Atom_DifferentialEquationBCC::LLGSTT(int sidx, int idx)
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

	DBL3 LLGSTT_Eval = (-GAMMA * grel / (1 + alpha * alpha)) * ((paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx)) + alpha * ((paMesh->M1(sidx, idx) / mu_s) ^ (paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx))));

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