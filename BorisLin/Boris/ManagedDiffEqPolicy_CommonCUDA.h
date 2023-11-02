#pragma once

#include "ManagedDiffEq_CommonCUDA.h"

#if COMPILECUDA == 1

class ManagedDiffEqPolicy_CommonCUDA
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<ManagedDiffEq_CommonCUDA, ManagedDiffEqPolicy_CommonCUDA>& mng;

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
	ManagedDiffEqPolicy_CommonCUDA(mcu_obj<ManagedDiffEq_CommonCUDA, ManagedDiffEqPolicy_CommonCUDA>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	ManagedDiffEqPolicy_CommonCUDA& operator=(const ManagedDiffEqPolicy_CommonCUDA& copyThis) { return *this; }

	//destructor
	virtual ~ManagedDiffEqPolicy_CommonCUDA() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// ManagedDiffEqPolicy_CommonCUDA Methods

	BError set_pointers(ODECommonCUDA* pDiffEqCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(pDiffEqCUDA, mGPU);
		}

		return error;
	}
};

#endif