#pragma once

#include "ManagedModulesCUDA.h"

#if COMPILECUDA == 1

class ManagedModulesPolicyCUDA
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<ManagedModulesCUDA, ManagedModulesPolicyCUDA>& mng;

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
	ManagedModulesPolicyCUDA(mcu_obj<ManagedModulesCUDA, ManagedModulesPolicyCUDA>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	ManagedModulesPolicyCUDA& operator=(const ManagedModulesPolicyCUDA& copyThis) { return *this; }

	//destructor
	virtual ~ManagedModulesPolicyCUDA() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// ManagedModulesPolicyCUDA Methods

	BError set_pointers(ModulesCUDA* pModulesCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(pModulesCUDA, mGPU);
		}

		return error;
	}
};

#endif
