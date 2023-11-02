#include "stdafx.h"
#include "Atom_DiffEq.h"
#include "Atom_Mesh.h"
#include "SuperMesh.h"

///////////////////////////////////////////////////////////////////////
// Static variables in Atom_ODECommon

//-----------------------------------Collection of atomistic magnetic meshes

vector_lut<Atom_DifferentialEquation*> Atom_ODECommon::pODE;

//-----------------------------------Equation and Evaluation method values

int Atom_ODECommon::setODE = ODE_LLG;

Atom_Equation Atom_ODECommon::equation;
Atom_MEquation Atom_ODECommon::mequation;

//-----------------------------------Evaluation method modifiers

bool Atom_ODECommon::renormalize = true;

//-----------------------------------Steepest Descent Solver

double Atom_ODECommon::delta_m_sq = 0.0;
double Atom_ODECommon::delta_G_sq = 0.0;
double Atom_ODECommon::delta_m_dot_delta_G = 0.0;

double Atom_ODECommon::delta_m2_sq = 0.0;
double Atom_ODECommon::delta_G2_sq = 0.0;
double Atom_ODECommon::delta_m2_dot_delta_G2 = 0.0;

//-----------------------------------CUDA version

#if COMPILECUDA == 1
Atom_ODECommonCUDA* Atom_ODECommon::pODECUDA = nullptr;
#endif

///////////////////////////////////////////////////////////////////////

Atom_ODECommon::Atom_ODECommon(bool called_from_derived) :
	ProgramStateNames(this,
		{
			VINFO(iteration), VINFO(stageiteration),
			VINFO(time), VINFO(stagetime),
			VINFO(mxh), VINFO(dmdt),
			VINFO(setODE), VINFO(evalMethod), 
			VINFO(dT), VINFO(dTstoch), VINFO(time_stoch), VINFO(link_dTstoch),
			VINFO(dTspeedup), VINFO(time_speedup), VINFO(link_dTspeedup),
			VINFO(err_high_fail), VINFO(dT_increase), VINFO(dT_min), VINFO(dT_max), VINFO(eval_method_order),
			VINFO(use_evaluation_speedup),
			VINFO(moving_mesh), VINFO(moving_mesh_antisymmetric), VINFO(moving_mesh_threshold), VINFO(moving_mesh_dwshift)
		}, {})
{
	//when a new ferromagnetic mesh is added this constructor is called with called_from_derived = true
	//we only need to set ODE when ODECommon is first created as these are common to all ferromagnetic meshes (ODE can be changed separately of course later)
	if (!called_from_derived) {

		SetODE((ODE_)setODE, (EVAL_)evalMethod);
	}

	//ODECommon is held in SuperMesh. No need to make pODECUDA here as cuda must be switched off when ODECommon is made (start of program)
}

Atom_ODECommon::~Atom_ODECommon()
{
}

void Atom_ODECommon::RepairObjectState(void)
{
	//Here need to make sure everything is correctly conigured from primary data (which was just loaded)

	//must remake equation: do not set eval method yet. As meshes are loaded later, they'll each make their own settings for the current evaluation method
	SetODE((ODE_)setODE, (EVAL_)evalMethod, false);
}

//---------------------------------------- SET-UP METHODS

BError Atom_ODECommon::SetODE(ODE_ setODE_, EVAL_ evalMethod_, bool set_eval_method)
{
	BError error(__FUNCTION__);

	//use a function pointer to assign equation to solve
	//this approach saves on having to write out the evaluation methods for every equation, with possible impact on performance due to equation evaluation not being manually inlined
	//tests for typical problems show this has virtually no impact on performance! Maybe the compiler has inlined the code for all the different equations used? (could check this)
	//Or maybe the overhead from function calls is just insignificant for typical problems

	switch (setODE_) {

	case ODE_LLG:
		setODE = setODE_;
		equation = &Atom_DifferentialEquation::LLG;
		mequation = &Atom_DifferentialEquation::LLG;
		renormalize = true;
		solve_spin_current_a = false;
		break;

	case ODE_LLGSTATIC:
		setODE = setODE_;
		equation = &Atom_DifferentialEquation::LLGStatic;
		mequation = &Atom_DifferentialEquation::LLGStatic;
		renormalize = true;
		solve_spin_current_a = false;
		break;

	case ODE_LLGSTATICSA:
		setODE = setODE_;
		equation = &Atom_DifferentialEquation::LLGStatic;
		mequation = &Atom_DifferentialEquation::LLGStatic;
		renormalize = true;
		solve_spin_current_a = true;
		break;

	case ODE_LLGSTT:
		setODE = setODE_;
		equation = &Atom_DifferentialEquation::LLGSTT;
		mequation = &Atom_DifferentialEquation::LLGSTT;
		renormalize = true;
		solve_spin_current_a = false;
		break;

	case ODE_SLLG:
		setODE = setODE_;
		equation = &Atom_DifferentialEquation::SLLG;
		mequation = &Atom_DifferentialEquation::SLLG;
		renormalize = true;
		solve_spin_current_a = false;
		break;

	case ODE_SLLGSTT:
		setODE = setODE_;
		equation = &Atom_DifferentialEquation::SLLGSTT;
		mequation = &Atom_DifferentialEquation::SLLGSTT;
		renormalize = true;
		solve_spin_current_a = false;
		break;

	case ODE_LLGSA:
		setODE = setODE_;
		equation = &Atom_DifferentialEquation::LLG;
		mequation = &Atom_DifferentialEquation::LLG;
		renormalize = true;
		solve_spin_current_a = true;
		break;

	case ODE_SLLGSA:
		setODE = setODE_;
		equation = &Atom_DifferentialEquation::SLLG;
		mequation = &Atom_DifferentialEquation::SLLG;
		renormalize = true;
		solve_spin_current_a = true;
		break;

	default:
		//no change
		break;
	}

	if (set_eval_method) {

		error = SetEvaluationMethod(evalMethod_);
	}

#if COMPILECUDA == 1
	if (pODECUDA) {

		pODECUDA->SyncODEValues();

		//based on setODE value, setup device method pointers in cuDiffEq held in each DifferentialEquationCUDA object
		for (int idx = 0; idx < (int)pODE.size(); idx++) {

			pODE[idx]->pameshODECUDA->SetODEMethodPointers();
		}
	}
#endif

	return error;
}

//---------------------------------------- GET / SET METHODS

void Atom_ODECommon::Set_mxh(void)
{
	//set mxh as the maximum values from all the set meshes
	for (int idx = 0; idx < pODE.size(); idx++) {

		if (pODE[idx]->mxh_reduction.max > mxh) mxh = pODE[idx]->mxh_reduction.max;
	}
}

void Atom_ODECommon::Set_dmdt(void)
{
	//set dmdt as the maximum values from all the set meshes
	for (int idx = 0; idx < pODE.size(); idx++) {

		if (pODE[idx]->dmdt_reduction.max > dmdt) dmdt = pODE[idx]->dmdt_reduction.max;
	}
}

void Atom_ODECommon::Set_lte(void)
{
	for (int idx = 0; idx < pODE.size(); idx++) {

		if (pODE[idx]->lte_reduction.max > lte) lte = pODE[idx]->lte_reduction.max;
	}
}