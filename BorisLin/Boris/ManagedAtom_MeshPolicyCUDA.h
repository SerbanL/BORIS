#pragma once

#include "ManagedAtom_MeshCUDA.h"

#if COMPILECUDA == 1

class ManagedAtom_MeshPolicyCUDA
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<ManagedAtom_MeshCUDA, ManagedAtom_MeshPolicyCUDA>& mng;

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
	ManagedAtom_MeshPolicyCUDA(mcu_obj<ManagedAtom_MeshCUDA, ManagedAtom_MeshPolicyCUDA>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	ManagedAtom_MeshPolicyCUDA& operator=(const ManagedAtom_MeshPolicyCUDA& copyThis) { return *this; }

	//destructor
	virtual ~ManagedAtom_MeshPolicyCUDA() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// ManagedAtom_MeshPolicyCUDA Methods

	BError set_pointers(Atom_MeshCUDA* paMeshCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(paMeshCUDA, mGPU);
		}

		return error;
	}

	void Set_SC_MCFuncs(std::vector<int>& modules_ids)
	{
		mcu_arr<int> cuaModules(mGPU);
		cuaModules.resize(modules_ids.size());
		cuaModules.copy_from_vector(modules_ids);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->Set_SC_MCFuncs(cuaModules.get_cu_arr(mGPU));
		}
	}
};

#endif
