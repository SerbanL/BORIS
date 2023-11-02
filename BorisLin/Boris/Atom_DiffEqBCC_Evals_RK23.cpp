#include "stdafx.h"
#include "Atom_DiffEqBCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC
#ifdef ODE_EVAL_COMPILATION_RK23

#include "Atom_Mesh_BCC.h"
#include "SuperMesh.h"
#include "Atom_MeshParamsControl_Multi.h"

//--------------------------------------------- RUNGE KUTTA 23 (Bogacki-Shampine) (2nd order adaptive step with FSAL, 3rd order evaluation)

void Atom_DifferentialEquationBCC::RunRK23_Step0_withReductions(void)
{
	mxh_reduction.new_minmax_reduction();
	lte_reduction.new_minmax_reduction();

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	double conversion = MUB / paMesh->h.dim();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//obtain maximum normalized torque term
					double Mnorm = paMesh->M1(sidx, idx).norm();
					double _mxh = GetMagnitude(paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx)) / (conversion * Mnorm * Mnorm);
					mxh_reduction.reduce_max(_mxh);

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//2nd order evaluation for adaptive step
					DBL3 prediction = sM1(sidx, idx) + (7 * sEval0(sidx, idx) / 24 + 1 * sEval1(sidx, idx) / 4 + 1 * sEval2(sidx, idx) / 3 + 1 * rhs / 8) * dT;

					//local truncation error (between predicted and corrected)
					double _lte = GetMagnitude(paMesh->M1(sidx, idx) - prediction) / Mnorm;
					lte_reduction.reduce_max(_lte);

					//save evaluation for later use
					sEval0(sidx, idx) = rhs;
				}
			}
		}
	}

	lte_reduction.maximum();
	
	if (paMesh->grel.get0()) {

		mxh_reduction.maximum();
	}
	else {

		mxh_reduction.max = 0.0;
	}
}

void Atom_DifferentialEquationBCC::RunRK23_Step0(void)
{
	//lte reductions needed for adaptive time step
	lte_reduction.new_minmax_reduction();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//2nd order evaluation for adaptive step
					DBL3 prediction = sM1(sidx, idx) + (7 * sEval0(sidx, idx) / 24 + 1 * sEval1(sidx, idx) / 4 + 1 * sEval2(sidx, idx) / 3 + 1 * rhs / 8) * dT;

					//local truncation error (between predicted and corrected)
					double _lte = GetMagnitude(paMesh->M1(sidx, idx) - prediction) / paMesh->M1(sidx, idx).norm();
					lte_reduction.reduce_max(_lte);

					//save evaluation for later use
					sEval0(sidx, idx) = rhs;
				}
			}
		}
	}

	lte_reduction.maximum();
}

void Atom_DifferentialEquationBCC::RunRK23_Step0_Advance(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//Save current moment for later use
					sM1(sidx, idx) = paMesh->M1(sidx, idx);

					//Now estimate moment using RK23 first step
					paMesh->M1(sidx, idx) += sEval0(sidx, idx) * (dT / 2);
				}
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRK23_Step1(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval1(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RK23 midle step 1
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + 3 * sEval1(sidx, idx) * dT / 4;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRK23_Step2_withReductions(void)
{
	dmdt_reduction.new_minmax_reduction();

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	double conversion = MUB / paMesh->h.dim();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					sEval2(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

					//Now calculate 3rd order evaluation
					paMesh->M1(sidx, idx) = sM1(sidx, idx) + (2 * sEval0(sidx, idx) / 9 + 1 * sEval1(sidx, idx) / 3 + 4 * sEval2(sidx, idx) / 9) * dT;

					if (renormalize) {

						double mu_s = paMesh->mu_s(sidx);
						paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s);
						paMesh->M1(sidx, idx).renormalize(mu_s);
					}

					//obtained maximum dmdt term
					double Mnorm = paMesh->M1(sidx, idx).norm();
					double _dmdt = GetMagnitude(paMesh->M1(sidx, idx) - sM1(sidx, idx)) / (dT * GAMMA * Mnorm * conversion * Mnorm);
					dmdt_reduction.reduce_max(_dmdt);
				}
			}
		}
	}

	if (paMesh->grel.get0()) {

		dmdt_reduction.maximum();
	}
	else {
	
		dmdt_reduction.max = 0.0;
	}
}

void Atom_DifferentialEquationBCC::RunRK23_Step2(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					sEval2(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

					//Now calculate 3rd order evaluation
					paMesh->M1(sidx, idx) = sM1(sidx, idx) + (2 * sEval0(sidx, idx) / 9 + 1 * sEval1(sidx, idx) / 3 + 4 * sEval2(sidx, idx) / 9) * dT;

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