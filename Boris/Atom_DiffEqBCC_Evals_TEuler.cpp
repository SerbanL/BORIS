#include "stdafx.h"
#include "Atom_DiffEqBCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC
#ifdef ODE_EVAL_COMPILATION_TEULER

#include "Atom_Mesh_BCC.h"
#include "SuperMesh.h"
#include "Atom_MeshParamsControl_Multi.h"

//--------------------------------------------- TRAPEZOIDAL EULER

void Atom_DifferentialEquationBCC::RunTEuler_Step0_withReductions(void)
{
	mxh_av_reduction.new_average_reduction();

	//Trapezoidal Euler can be used for stochastic equations - generate thermal VECs at the start
	if (H_Thermal.linear_size()) GenerateThermalField();

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	double conversion = MUB / paMesh->h.dim();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				//Save current moment for the next step
				sM1(sidx, idx) = paMesh->M1(sidx, idx);

				if (!paMesh->M1.is_skipcell(idx)) {

					//obtained average normalized torque term
					double Mnorm = paMesh->M1(sidx, idx).norm();
					mxh_av_reduction.reduce_average((paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx)) / (conversion * Mnorm * Mnorm));

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment for the next time step
					paMesh->M1(sidx, idx) += rhs * dT;
				}
			}
		}
	}

	if (paMesh->grel.get0()) {

		mxh_reduction.max = GetMagnitude(mxh_av_reduction.average());
	}
	else {

		mxh_reduction.max = 0.0;
	}
}

void Atom_DifferentialEquationBCC::RunTEuler_Step0(void)
{
	//Trapezoidal Euler can be used for stochastic equations - generate thermal VECs at the start
	if (H_Thermal.linear_size()) GenerateThermalField();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				//Save current moment for the next step
				sM1(sidx, idx) = paMesh->M1(sidx, idx);

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment for the next time step
					paMesh->M1(sidx, idx) += rhs * dT;
				}
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunTEuler_Step1_withReductions(void)
{
	dmdt_av_reduction.new_average_reduction();

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	double conversion = MUB / paMesh->h.dim();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment using the second trapezoidal Euler step equation
					paMesh->M1(sidx, idx) = (sM1(sidx, idx) + paMesh->M1(sidx, idx) + rhs * dT) / 2;

					if (renormalize) {

						double mu_s = paMesh->mu_s(sidx);
						paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s);
						paMesh->M1(sidx, idx).renormalize(mu_s);
					}

					//obtained average dmdt term
					double Mnorm = paMesh->M1(sidx, idx).norm();
					dmdt_av_reduction.reduce_average((paMesh->M1(sidx, idx) - sM1(sidx, idx)) / (dT * GAMMA * Mnorm * conversion * Mnorm));
				}
			}
		}
	}

	if (paMesh->grel.get0()) {

		dmdt_reduction.max = GetMagnitude(dmdt_av_reduction.average());
	}
	else {

		dmdt_reduction.max = 0.0;
	}
}

void Atom_DifferentialEquationBCC::RunTEuler_Step1(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment using the second trapezoidal Euler step equation
					paMesh->M1(sidx, idx) = (sM1(sidx, idx) + paMesh->M1(sidx, idx) + rhs * dT) / 2;

					if (renormalize) {

						double mu_s = paMesh->mu_s(sidx);
						paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s);
						paMesh->M1(sidx, idx).renormalize(mu_s);
					}
				}
			}
		}
	}
}

#endif
#endif