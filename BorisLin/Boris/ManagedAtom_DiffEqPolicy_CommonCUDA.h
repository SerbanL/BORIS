#pragma once

#include "ManagedAtom_DiffEq_CommonCUDA.h"

#if COMPILECUDA == 1

class ManagedAtom_DiffEqPolicy_CommonCUDA
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<ManagedAtom_DiffEq_CommonCUDA, ManagedAtom_DiffEqPolicy_CommonCUDA>& mng;

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
	ManagedAtom_DiffEqPolicy_CommonCUDA(mcu_obj<ManagedAtom_DiffEq_CommonCUDA, ManagedAtom_DiffEqPolicy_CommonCUDA>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	ManagedAtom_DiffEqPolicy_CommonCUDA& operator=(const ManagedAtom_DiffEqPolicy_CommonCUDA& copyThis) { return *this; }

	//destructor
	virtual ~ManagedAtom_DiffEqPolicy_CommonCUDA() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// ManagedAtom_DiffEqPolicy_CommonCUDA Methods

	BError set_pointers(Atom_ODECommonCUDA* paDiffEqCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(paDiffEqCUDA, mGPU);
		}

		return error;
	}
};

#endif