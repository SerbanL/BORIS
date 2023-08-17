#pragma once

#include "ManagedMeshCUDA.h"

#if COMPILECUDA == 1

class ManagedMeshPolicyCUDA
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<ManagedMeshCUDA, ManagedMeshPolicyCUDA>& mng;

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
	ManagedMeshPolicyCUDA(mcu_obj<ManagedMeshCUDA, ManagedMeshPolicyCUDA>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	ManagedMeshPolicyCUDA& operator=(const ManagedMeshPolicyCUDA& copyThis) { return *this; }

	//destructor
	virtual ~ManagedMeshPolicyCUDA() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// ManagedMeshPolicyCUDA Methods

	BError set_pointers(MeshCUDA* pMeshCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(pMeshCUDA, mGPU);
		}

		return error;
	}
};

#endif
