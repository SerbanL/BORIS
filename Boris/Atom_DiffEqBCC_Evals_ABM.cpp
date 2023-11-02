#include "stdafx.h"
#include "Atom_DiffEqBCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC
#ifdef ODE_EVAL_COMPILATION_ABM

#include "Atom_Mesh_BCC.h"
#include "SuperMesh.h"
#include "Atom_MeshParamsControl_Multi.h"

//--------------------------------------------- ADAMS-BASHFORTH-MOULTON

void Atom_DifferentialEquationBCC::RunABM_Predictor_withReductions(void)
{
	mxh_reduction.new_minmax_reduction();

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	double conversion = MUB / paMesh->h.dim();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				//Save current moment for the next step
				sM1(sidx, idx) = paMesh->M1(sidx, idx);

				if (!paMesh->M1.is_skipcell(idx)) {

					//obtained maximum normalized torque term
					double Mnorm = paMesh->M1(sidx, idx).norm();
					double _mxh = GetMagnitude(paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx)) / (conversion * Mnorm * Mnorm);
					mxh_reduction.reduce_max(_mxh);

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//ABM predictor : pk+1 = mk + (dt/2) * (3*fk - fk-1)
					if (alternator) {

						paMesh->M1(sidx, idx) += dT * (3 * rhs - sEval0(sidx, idx)) / 2;
						sEval1(sidx, idx) = rhs;
					}
					else {

						paMesh->M1(sidx, idx) += dT * (3 * rhs - sEval1(sidx, idx)) / 2;
						sEval0(sidx, idx) = rhs;
					}
				}
			}
		}
	}

	if (paMesh->grel.get0()) {

		//only reduce if grel is not zero (if it's zero this means magnetization dynamics are disabled in this mesh)
		mxh_reduction.maximum();
	}
	else {

		mxh_reduction.max = 0.0;
	}
}

void Atom_DifferentialEquationBCC::RunABM_Predictor(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				//Save current moment for the next step
				sM1(sidx, idx) = paMesh->M1(sidx, idx);

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//ABM predictor : pk+1 = mk + (dt/2) * (3*fk - fk-1)
					if (alternator) {

						paMesh->M1(sidx, idx) += dT * (3 * rhs - sEval0(sidx, idx)) / 2;
						sEval1(sidx, idx) = rhs;
					}
					else {

						paMesh->M1(sidx, idx) += dT * (3 * rhs - sEval1(sidx, idx)) / 2;
						sEval0(sidx, idx) = rhs;
					}
				}
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunABM_Corrector_withReductions(void)
{
	dmdt_reduction.new_minmax_reduction();
	lte_reduction.new_minmax_reduction();

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	double conversion = MUB / paMesh->h.dim();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//First save predicted moment for lte calculation
					DBL3 saveM = paMesh->M1(sidx, idx);

					//ABM corrector : mk+1 = mk + (dt/2) * (fk+1 + fk)
					if (alternator) {

						paMesh->M1(sidx, idx) = sM1(sidx, idx) + dT * (rhs + sEval1(sidx, idx)) / 2;
					}
					else {

						paMesh->M1(sidx, idx) = sM1(sidx, idx) + dT * (rhs + sEval0(sidx, idx)) / 2;
					}

					if (renormalize) {

						double mu_s = paMesh->mu_s(sidx);
						paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s);
						paMesh->M1(sidx, idx).renormalize(mu_s);
					}

					//obtained maximum dmdt term
					double Mnorm = paMesh->M1(sidx, idx).norm();
					double _dmdt = GetMagnitude(paMesh->M1(sidx, idx) - sM1(sidx, idx)) / (dT * GAMMA * Mnorm * conversion * Mnorm);
					dmdt_reduction.reduce_max(_dmdt);

					//local truncation error (between predicted and corrected)
					double _lte = GetMagnitude(paMesh->M1(sidx, idx) - saveM) / paMesh->M1(sidx, idx).norm();
					lte_reduction.reduce_max(_lte);
				}
			}
		}
	}

	lte_reduction.maximum();

	if (paMesh->grel.get0()) {

		//only reduce for dmdt if grel is not zero (if it's zero this means magnetization dynamics are disabled in this mesh)
		dmdt_reduction.maximum();
	}
	else {

		dmdt_reduction.max = 0.0;
	}
}

void Atom_DifferentialEquationBCC::RunABM_Corrector(void)
{
	lte_reduction.new_minmax_reduction();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//First save predicted moment for lte calculation
					DBL3 saveM = paMesh->M1(sidx, idx);

					//ABM corrector : mk+1 = mk + (dt/2) * (fk+1 + fk)
					if (alternator) {

						paMesh->M1(sidx, idx) = sM1(sidx, idx) + dT * (rhs + sEval1(sidx, idx)) / 2;
					}
					else {

						paMesh->M1(sidx, idx) = sM1(sidx, idx) + dT * (rhs + sEval0(sidx, idx)) / 2;
					}

					if (renormalize) {

						double mu_s = paMesh->mu_s(sidx);
						paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s);
						paMesh->M1(sidx, idx).renormalize(mu_s);
					}

					//local truncation error (between predicted and corrected)
					double _lte = GetMagnitude(paMesh->M1(sidx, idx) - saveM) / paMesh->M1(sidx, idx).norm();
					lte_reduction.reduce_max(_lte);
				}
			}
		}
	}

	lte_reduction.maximum();
}

void Atom_DifferentialEquationBCC::RunABM_TEuler0(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				//Save current moment for the next step
				sM1(sidx, idx) = paMesh->M1(sidx, idx);

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					sEval0(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment for the next time step
					paMesh->M1(sidx, idx) += sEval0(sidx, idx) * dT;
				}
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunABM_TEuler1(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using the second trapezoidal Euler step equation
				paMesh->M1(sidx, idx) = (sM1(sidx, idx) + paMesh->M1(sidx, idx) + rhs * dT) / 2;
			}
		}
	}
}

#endif
#endif