#include "stdafx.h"
#include "Atom_DiffEqCUDA.h"

#if COMPILECUDA == 1

#include "Atom_DiffEq.h"

#include "Atom_Mesh.h"
#include "Atom_MeshCUDA.h"

Atom_DifferentialEquationCUDA::Atom_DifferentialEquationCUDA(Atom_DifferentialEquation *pameshODE_) :
	Atom_ODECommonCUDA(),
	sM1(mGPU),
	sEval0(mGPU), sEval1(mGPU), sEval2(mGPU), sEval3(mGPU), sEval4(mGPU), sEval5(mGPU), sEval6(mGPU),
	H_Thermal(mGPU),
	prng(mGPU, GetSystemTickCount(), pameshODE_->paMesh->n.dim() / (128 * mGPU.get_num_devices())),
	deltaTstoch(mGPU)
{
	pameshODE = pameshODE_;
	
	paMesh = pameshODE->paMesh;
	paMeshCUDA = pameshODE->paMesh->paMeshCUDA;
}

//called when using stochastic equations
void Atom_DifferentialEquationCUDA::GenerateThermalField(void)
{
	//if not in linked dTstoch mode, then only generate stochastic field at a minimum of dTstoch spacing
	if (!pameshODE->link_dTstoch && pameshODE->GetTime() < pameshODE->time_stoch + pameshODE->dTstoch) return;

	if (!pameshODE->link_dTstoch) {

		double deltaT = pameshODE->GetTime() - pameshODE->time_stoch;
		pameshODE->time_stoch = pameshODE->GetTime();

		deltaTstoch.from_cpu(deltaT);

		GenerateThermalField_CUDA(deltaTstoch);
	}
	else GenerateThermalField_CUDA(*pdT);
}

#endif