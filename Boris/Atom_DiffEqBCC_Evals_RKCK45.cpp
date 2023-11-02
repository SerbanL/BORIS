#include "stdafx.h"
#include "Atom_DiffEqBCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC
#ifdef ODE_EVAL_COMPILATION_RKCK

#include "Atom_Mesh_BCC.h"
#include "SuperMesh.h"
#include "Atom_MeshParamsControl_Multi.h"

//--------------------------------------------- RUNGE KUTTA CASH-KARP (4th order solution, 5th order error)

void Atom_DifferentialEquationBCC::RunRKCK45_Step0_withReductions(void)
{
	mxh_reduction.new_minmax_reduction();

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	double conversion = MUB / paMesh->h.dim();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				//Save current moment for later use
				sM1(sidx, idx) = paMesh->M1(sidx, idx);

				if (!paMesh->M1.is_skipcell(idx)) {

					//obtain maximum normalized torque term
					double Mnorm = paMesh->M1(sidx, idx).norm();
					double _mxh = GetMagnitude(paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx)) / (conversion * Mnorm * Mnorm);
					mxh_reduction.reduce_max(_mxh);

					//First evaluate RHS of set equation at the current time step
					sEval0(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment using RKCK first step
					paMesh->M1(sidx, idx) += sEval0(sidx, idx) * (dT / 5);
				}
			}
		}
	}

	if (paMesh->grel.get0()) {

		mxh_reduction.maximum();
	}
	else {

		mxh_reduction.max = 0.0;
	}
}

void Atom_DifferentialEquationBCC::RunRKCK45_Step0(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				//Save current moment for later use
				sM1(sidx, idx) = paMesh->M1(sidx, idx);

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					sEval0(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment using RKCK first step
					paMesh->M1(sidx, idx) += sEval0(sidx, idx) * (dT / 5);
				}
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKCK45_Step1(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval1(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKCK midle step 1
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (3 * sEval0(sidx, idx) + 9 * sEval1(sidx, idx)) * dT / 40;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKCK45_Step2(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval2(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKCK midle step 2
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (3 * sEval0(sidx, idx) / 10 - 9 * sEval1(sidx, idx) / 10 + 6 * sEval2(sidx, idx) / 5) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKCK45_Step3(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval3(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKCK midle step 3
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (-11 * sEval0(sidx, idx) / 54 + 5 * sEval1(sidx, idx) / 2 - 70 * sEval2(sidx, idx) / 27 + 35 * sEval3(sidx, idx) / 27) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKCK45_Step4(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval4(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKCK midle step 4
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (1631 * sEval0(sidx, idx) / 55296 + 175 * sEval1(sidx, idx) / 512 + 575 * sEval2(sidx, idx) / 13824 + 44275 * sEval3(sidx, idx) / 110592 + 253 * sEval4(sidx, idx) / 4096) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKCK45_Step5_withReductions(void)
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

					//RKCK45 : 4th order evaluation
					paMesh->M1(sidx, idx) = sM1(sidx, idx) + (2825 * sEval0(sidx, idx) / 27648 + 18575 * sEval2(sidx, idx) / 48384 + 13525 * sEval3(sidx, idx) / 55296 + 277 * sEval4(sidx, idx) / 14336 + rhs / 4) * dT;

					//Now calculate 5th order evaluation for adaptive time step
					DBL3 prediction = sM1(sidx, idx) + (37 * sEval0(sidx, idx) / 378 + 250 * sEval2(sidx, idx) / 621 + 125 * sEval3(sidx, idx) / 594 + 512 * rhs / 1771) * dT;

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
					double _lte = GetMagnitude(paMesh->M1(sidx, idx) - prediction) / paMesh->M1(sidx, idx).norm();
					lte_reduction.reduce_max(_lte);
				}
			}
		}
	}

	lte_reduction.maximum();

	if (paMesh->grel.get0()) {

		dmdt_reduction.maximum();
	}
	else {

		dmdt_reduction.max = 0.0;
	}
}

void Atom_DifferentialEquationBCC::RunRKCK45_Step5(void)
{
	lte_reduction.new_minmax_reduction();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//RKCK45 : 4th order evaluation
					paMesh->M1(sidx, idx) = sM1(sidx, idx) + (2825 * sEval0(sidx, idx) / 27648 + 18575 * sEval2(sidx, idx) / 48384 + 13525 * sEval3(sidx, idx) / 55296 + 277 * sEval4(sidx, idx) / 14336 + rhs / 4) * dT;

					//Now calculate 5th order evaluation for adaptive time step
					DBL3 prediction = sM1(sidx, idx) + (37 * sEval0(sidx, idx) / 378 + 250 * sEval2(sidx, idx) / 621 + 125 * sEval3(sidx, idx) / 594 + 512 * rhs / 1771) * dT;

					if (renormalize) {

						double mu_s = paMesh->mu_s(sidx);
						paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s);
						paMesh->M1(sidx, idx).renormalize(mu_s);
					}

					//local truncation error (between predicted and corrected)
					double _lte = GetMagnitude(paMesh->M1(sidx, idx) - prediction) / paMesh->M1(sidx, idx).norm();
					lte_reduction.reduce_max(_lte);
				}
			}
		}
	}

	lte_reduction.maximum();
}

#endif
#endif