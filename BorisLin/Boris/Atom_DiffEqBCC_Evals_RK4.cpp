#include "stdafx.h"
#include "Atom_DiffEqBCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC
#ifdef ODE_EVAL_COMPILATION_RK4

#include "Atom_Mesh_BCC.h"
#include "SuperMesh.h"
#include "Atom_MeshParamsControl_Multi.h"

//--------------------------------------------- RK4

void Atom_DifferentialEquationBCC::RunRK4_Step0_withReductions(void)
{
	bool stochastic = H_Thermal.linear_size() != 0;

	//RK4 can be used for stochastic equations - generate thermal VECs at the start
	if (H_Thermal.linear_size()) GenerateThermalField();

	if (stochastic) {

		mxh_av_reduction.new_average_reduction();

		//multiplicative conversion factor from atomic moment (units of muB) to A/m
		double conversion = MUB / paMesh->h.dim();

		for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
			for (int idx = 0; idx < paMesh->n.dim(); idx++) {

				if (paMesh->M1.is_not_empty(idx)) {

					//Save current moment for later use
					sM1(sidx, idx) = paMesh->M1(sidx, idx);

					if (!paMesh->M1.is_skipcell(idx)) {

						//obtained maximum normalized torque term
						double Mnorm = paMesh->M1(sidx, idx).norm();
						mxh_av_reduction.reduce_average((paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx)) / (conversion * Mnorm * Mnorm));

						//First evaluate RHS of set equation at the current time step
						sEval0(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

						//Now estimate moment using RK4 midle step
						paMesh->M1(sidx, idx) += sEval0(sidx, idx) * (dT / 2);
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
	else {

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

						//obtained maximum normalized torque term
						double Mnorm = paMesh->M1(sidx, idx).norm();
						double _mxh = GetMagnitude(paMesh->M1(sidx, idx) ^ paMesh->Heff1(sidx, idx)) / (conversion * Mnorm * Mnorm);
						mxh_reduction.reduce_max(_mxh);

						//First evaluate RHS of set equation at the current time step
						sEval0(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

						//Now estimate moment using RK4 midle step
						paMesh->M1(sidx, idx) += sEval0(sidx, idx) * (dT / 2);
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
}

void Atom_DifferentialEquationBCC::RunRK4_Step0(void)
{
	//RK4 can be used for stochastic equations - generate thermal VECs at the start
	if (H_Thermal.linear_size()) GenerateThermalField();

	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				//Save current moment for later use
				sM1(sidx, idx) = paMesh->M1(sidx, idx);

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					sEval0(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment using RK4 midle step
					paMesh->M1(sidx, idx) += sEval0(sidx, idx) * (dT / 2);
				}
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRK4_Step1(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval1(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RK4 midle step
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + sEval1(sidx, idx) * (dT / 2);
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRK4_Step2(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx) && !paMesh->M1.is_skipcell(idx)) {

				//First evaluate RHS of set equation at the current time step
				sEval2(sidx, idx) = CALLFP(this, mequation)(sidx, idx);

				//Now estimate moment using RK4 last step
				paMesh->M1(sidx, idx) = sM1(sidx, idx) + sEval2(sidx, idx) * dT;
			}
		}
	}
}

void Atom_DifferentialEquationBCC::RunRK4_Step3_withReductions(void)
{
	bool stochastic = H_Thermal.linear_size() != 0;

	if (stochastic) {

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

						//Now estimate moment using previous RK4 evaluations
						paMesh->M1(sidx, idx) = sM1(sidx, idx) + (sEval0(sidx, idx) + 2 * sEval1(sidx, idx) + 2 * sEval2(sidx, idx) + rhs) * (dT / 6);

						if (renormalize) {

							double mu_s = paMesh->mu_s(sidx);
							paMesh->update_parameters_mcoarse(sidx, idx, paMesh->mu_s, mu_s);
							paMesh->M1(sidx, idx).renormalize(mu_s);
						}

						//obtained maximum dmdt term
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
	else {

		dmdt_reduction.new_minmax_reduction();

		//multiplicative conversion factor from atomic moment (units of muB) to A/m
		double conversion = MUB / paMesh->h.dim();

		for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
			for (int idx = 0; idx < paMesh->n.dim(); idx++) {

				if (paMesh->M1.is_not_empty(idx)) {

					if (!paMesh->M1.is_skipcell(idx)) {

						//First evaluate RHS of set equation at the current time step
						DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

						//Now estimate moment using previous RK4 evaluations
						paMesh->M1(sidx, idx) = sM1(sidx, idx) + (sEval0(sidx, idx) + 2 * sEval1(sidx, idx) + 2 * sEval2(sidx, idx) + rhs) * (dT / 6);

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
}

void Atom_DifferentialEquationBCC::RunRK4_Step3(void)
{
	for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for
		for (int idx = 0; idx < paMesh->n.dim(); idx++) {

			if (paMesh->M1.is_not_empty(idx)) {

				if (!paMesh->M1.is_skipcell(idx)) {

					//First evaluate RHS of set equation at the current time step
					DBL3 rhs = CALLFP(this, mequation)(sidx, idx);

					//Now estimate moment using previous RK4 evaluations
					paMesh->M1(sidx, idx) = sM1(sidx, idx) + (sEval0(sidx, idx) + 2 * sEval1(sidx, idx) + 2 * sEval2(sidx, idx) + rhs) * (dT / 6);

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