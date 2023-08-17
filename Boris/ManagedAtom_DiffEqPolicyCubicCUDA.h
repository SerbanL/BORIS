#pragma once

#include "ManagedAtom_DiffEqCubicCUDA.h"

#if COMPILECUDA == 1

class ManagedAtom_DiffEqPolicyCubicCUDA
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<ManagedAtom_DiffEqCubicCUDA, ManagedAtom_DiffEqPolicyCubicCUDA>& mng;

	//multi-GPU configuration (list of physical devices configured with memory transfer type configuration set)
	mGPUConfig& mGPU;

private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// AUXILIARY (all Policy classes)

	//clear all allocated memory
	void clear_memory_aux(void) {}

public:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// CONSTRUCTORS (all Policy classes)

	//--------------------------------------------CONSTRUCTORS

	//constructor for this policy class : void
	ManagedAtom_DiffEqPolicyCubicCUDA(mcu_obj<ManagedAtom_DiffEqCubicCUDA, ManagedAtom_DiffEqPolicyCubicCUDA>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	ManagedAtom_DiffEqPolicyCubicCUDA& operator=(const ManagedAtom_DiffEqPolicyCubicCUDA& copyThis) { return *this; }

	//destructor
	virtual ~ManagedAtom_DiffEqPolicyCubicCUDA() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// ManagedAtom_DiffEqPolicyCubicCUDA Methods

	BError set_pointers(Atom_DifferentialEquationCubicCUDA* pDiffEqCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(pDiffEqCUDA, mGPU);
		}

		return error;
	}
};

#endif
