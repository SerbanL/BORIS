#pragma once

#include "ManagedDiffEqAFMCUDA.h"

#if COMPILECUDA == 1

class ManagedDiffEqPolicyAFMCUDA
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<ManagedDiffEqAFMCUDA, ManagedDiffEqPolicyAFMCUDA>& mng;

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
	ManagedDiffEqPolicyAFMCUDA(mcu_obj<ManagedDiffEqAFMCUDA, ManagedDiffEqPolicyAFMCUDA>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	ManagedDiffEqPolicyAFMCUDA& operator=(const ManagedDiffEqPolicyAFMCUDA& copyThis) { return *this; }

	//destructor
	virtual ~ManagedDiffEqPolicyAFMCUDA() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// ManagedDiffEqPolicyAFMCUDA Methods

	BError set_pointers(DifferentialEquationAFMCUDA* pDiffEqCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(pDiffEqCUDA, mGPU);
		}

		return error;
	}
};

#endif
