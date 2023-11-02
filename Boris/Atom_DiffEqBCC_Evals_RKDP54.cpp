#include "stdafx.h"
#include "Atom_DiffEqBCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC
#ifdef ODE_EVAL_COMPILATION_RKDP

#include "Atom_Mesh_BCC.h"
#include "SuperMesh.h"
#include "Atom_MeshParamsControl_Multi.h"

//--------------------------------------------- RUNGE KUTTA DORMAND-PRINCE (4th order solution, 5th order error)

void Atom_DifferentialEquationBCC::RunRKDP54_Step0_withReductions(void)
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

					//Now calculate 5th order evaluation for adaptive time step -> FSAL property (a full pass required for this to be valid)
					DBL3 prediction = sM1(sidx, idx) + (5179 * sEval0(sidx, idx) / 57600 + 7571 * sEval2(sidx, idx) / 16695 + 393 * sEval3(sidx, idx) / 640 - 92097 * sEval4(sidx, idx) / 339200 + 187 * sEval5(sidx, idx) / 2100 + rhs / 40) * dT;

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

void Atom_DifferentialEquationBCC::RunRKDP54_Step0(void)
{
	lte_reduction.new_minmax_reduction();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//Now calculate 5th order evaluation for adaptive time step -> FSAL property (a full pass required for this to be valid)
					DBL3 prediction = sM1(sidx, idx) + (5179 * sEval0(sidx, idx) / 57600 + 7571 * sEval2(sidx, idx) / 16695 + 393 * sEval3(sidx, idx) / 640 - 92097 * sEval4(sidx, idx) / 339200 + 187 * sEval5(sidx, idx) / 2100 + rhs / 40) * dT;

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

void Atom_DifferentialEquationBCC::RunRKDP54_Step0_Advance(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//Save current moment for later use
					sM1(sidx, idx) = paMesh->M1(sidx, idx);

					//Now estimate moment using RKDP first step
					paMesh->M1(sidx, idx) += sEval0(sidx, idx) * (dT / 5);
				}
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKDP54_Step1(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval1(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKDP midle step 1
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (3 * sEval0(sidx, idx) / 40 + 9 * sEval1(sidx, idx) / 40) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKDP54_Step2(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval2(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKDP midle step 2
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (44 * sEval0(sidx, idx) / 45 - 56 * sEval1(sidx, idx) / 15 + 32 * sEval2(sidx, idx) / 9) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKDP54_Step3(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval3(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKDP midle step 3
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (19372 * sEval0(sidx, idx) / 6561 - 25360 * sEval1(sidx, idx) / 2187 + 64448 * sEval2(sidx, idx) / 6561 - 212 * sEval3(sidx, idx) / 729) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKDP54_Step4(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval4(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RKDP midle step 4
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + (9017 * sEval0(sidx, idx) / 3168 - 355 * sEval1(sidx, idx) / 33 + 46732 * sEval2(sidx, idx) / 5247 + 49 * sEval3(sidx, idx) / 176 - 5103 * sEval4(sidx, idx) / 18656) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRKDP54_Step5_withReductions(void)
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
					sEval5(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

					//RKDP54 : 5th order evaluation
					paMesh->M1(sidx, idx) = sM1(sidx, idx) + (35 * sEval0(sidx, idx) / 384 + 500 * sEval2(sidx, idx) / 1113 + 125 * sEval3(sidx, idx) / 192 - 2187 * sEval4(sidx, idx) / 6784 + 11 * sEval5(sidx, idx) / 84) * dT;

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

void Atom_DifferentialEquationBCC::RunRKDP54_Step5(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					sEval5(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

					//RKDP54 : 5th order evaluation
					paMesh->M1(sidx, idx) = sM1(sidx, idx) + (35 * sEval0(sidx, idx) / 384 + 500 * sEval2(sidx, idx) / 1113 + 125 * sEval3(sidx, idx) / 192 - 2187 * sEval4(sidx, idx) / 6784 + 11 * sEval5(sidx, idx) / 84) * dT;

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