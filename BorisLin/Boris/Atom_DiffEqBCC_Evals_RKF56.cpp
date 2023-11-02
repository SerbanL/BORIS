#include "stdafx.h"
#include "Atom_DiffEqBCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC
#ifdef ODE_EVAL_COMPILATION_RKF56

#include "Atom_Mesh_BCC.h"
#include "SuperMesh.h"
#include "Atom_MeshParamsControl_Multi.h"

//--------------------------------------------- RUNGE KUTTA FEHLBERG (5th order solution, 6th order error)

void Atom_DifferentialEquationBCC::RunRKF56_Step0_withReductions(void)
{
	mxh_reduction.new_minmax_reduction();

	//multiplicative conversion factor from atomic moment (units of muB) to A/m
	double conversion = MUB / paMesh->h.dim();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				//Save current moment
				sM1(sidx, idx) = paMesh->M1(sidx, idx);

				if (!paMesh->M1.is_skipcell(idx)) {

					//obtain maximum normalized torque term
					double Mnorm = paMesh->M1(sidx, idx).norm();
					double _mxh = GetMagnitude(paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx)) / (conversion * Mnorm * Mnorm);
					mxh_reduction.reduce_max(_mxh);

					//First evaluate RHS of set equation at the current time step
					sEval0(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment using RKF first step
					paMesh->M1(sidx, idx) += sEval0(sidx, idx) * (dT / 6);
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

void Atom_DifferentialEquationBCC::RunRKF56_Step0(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				//Save current moment
				sM1(sidx, idx) = paMesh->M1(sidx, idx);

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					sEval0(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment using RKF first step
					paMesh->M1(sidx, idx) += sEval0(sidx, idx) * (dT / 6);
				}
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKF56_Step1(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval1(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKF midle step 1
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (4 * sEval0(sidx, idx) + 16 * sEval1(sidx, idx)) * dT / 75;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKF56_Step2(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval2(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKF midle step 2
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (5 * sEval0(sidx, idx) / 6 - 8 * sEval1(sidx, idx) / 3 + 5 * sEval2(sidx, idx) / 2) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKF56_Step3(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval3(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKF midle step 3
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (-8 * sEval0(sidx, idx) / 5 + 144 * sEval1(sidx, idx) / 25 - 4 * sEval2(sidx, idx) + 16 * sEval3(sidx, idx) / 25) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKF56_Step4(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval4(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKF midle step 4
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (361 * sEval0(sidx, idx) / 320 - 18 * sEval1(sidx, idx) / 5 + 407 * sEval2(sidx, idx) / 128 - 11 * sEval3(sidx, idx) / 80 + 55 * sEval4(sidx, idx) / 128) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKF56_Step5(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval5(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (-11 * sEval0(sidx, idx) / 640 + 11 * sEval2(sidx, idx) / 256 - 11 * sEval3(sidx, idx) / 160 + 11 * sEval4(sidx, idx) / 256) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKF56_Step6(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval6(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (93 * sEval0(sidx, idx) / 640 - 18 * sEval1(sidx, idx) / 5 + 803 * sEval2(sidx, idx) / 256 - 11 * sEval3(sidx, idx) / 160 + 99 * sEval4(sidx, idx) / 256 + sEval6(sidx, idx)) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKF56_Step7_withReductions(void)
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

					//5th order evaluation
					paMesh->M1(sidx, idx) = sM1(sidx, idx) + (31 * sEval0(sidx, idx) / 384 + 1125 * sEval2(sidx, idx) / 2816 + 9 * sEval3(sidx, idx) / 32 + 125 * sEval4(sidx, idx) / 768 + 5 * sEval5(sidx, idx) / 66) * dT;

					//local truncation error from 5th order evaluation and 6th order evaluation
					DBL3 lte_diff = 5 * (sEval0(sidx, idx) + sEval5(sidx, idx) - sEval6(sidx, idx) - rhs) * dT / 66;

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
					double _lte = GetMagnitude(lte_diff) / paMesh->M1(sidx, idx).norm();
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

void Atom_DifferentialEquationBCC::RunRKF56_Step7(void)
{
	lte_reduction.new_minmax_reduction();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//5th order evaluation
					paMesh->M1(sidx, idx) = sM1(sidx, idx) + (31 * sEval0(sidx, idx) / 384 + 1125 * sEval2(sidx, idx) / 2816 + 9 * sEval3(sidx, idx) / 32 + 125 * sEval4(sidx, idx) / 768 + 5 * sEval5(sidx, idx) / 66) * dT;

					//local truncation error from 5th order evaluation and 6th order evaluation
					DBL3 lte_diff = 5 * (sEval0(sidx, idx) + sEval5(sidx, idx) - sEval6(sidx, idx) - rhs) * dT / 66;

					if (renormalize) {

						double mu_s = paMesh->mu_s(sidx);
						paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s);
						paMesh->M1(sidx, idx).renormalize(mu_s);
					}

					//local truncation error (between predicted and corrected)
					double _lte = GetMagnitude(lte_diff) / paMesh->M1(sidx, idx).norm();
					lte_reduction.reduce_max(_lte);
				}
			}
		}
	}

	lte_reduction.maximum();
}

#endif
#endif