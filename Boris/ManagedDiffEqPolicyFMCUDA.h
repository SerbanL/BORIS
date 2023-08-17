#pragma once

#include "ManagedDiffEqFMCUDA.h"

#if COMPILECUDA == 1

class ManagedDiffEqPolicyFMCUDA
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<ManagedDiffEqFMCUDA, ManagedDiffEqPolicyFMCUDA>& mng;

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
	ManagedDiffEqPolicyFMCUDA(mcu_obj<ManagedDiffEqFMCUDA, ManagedDiffEqPolicyFMCUDA>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	ManagedDiffEqPolicyFMCUDA& operator=(const ManagedDiffEqPolicyFMCUDA& copyThis) { return *this; }

	//destructor
	virtual ~ManagedDiffEqPolicyFMCUDA() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// ManagedDiffEqPolicyFMCUDA Methods

	BError set_pointers(DifferentialEquationFMCUDA* pDiffEqCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(pDiffEqCUDA, mGPU);
		}

		return error;
	}
};

#endif
