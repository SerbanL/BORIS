#pragma once

#include "Atom_Mesh_CubicCUDA_Thermalize.h"

#if COMPILECUDA == 1

#if ATOMISTIC == 1

class Thermalize_FM_to_AtomPolicy
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<Thermalize_FM_to_Atom, Thermalize_FM_to_AtomPolicy>& mng;

	//multi-GPU configuration (list of physical devices configured with memory transfer type configuration set)
	mGPUConfig& mGPU;

	//////////////////////////////////////////////////////////////////////////////////
	//
	// POLICY CLASS DATA (specific)


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
	Thermalize_FM_to_AtomPolicy(mcu_obj<Thermalize_FM_to_Atom, Thermalize_FM_to_AtomPolicy>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	Thermalize_FM_to_AtomPolicy& operator=(const Thermalize_FM_to_AtomPolicy& copyThis) { return *this; }

	//destructor
	virtual ~Thermalize_FM_to_AtomPolicy() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// Thermalize_FM_to_AtomPolicy Methods

	//for modules held in micromagnetic meshes
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(paMeshCUDA, mGPU);
		}

		return error;
	}
};

#endif

#endif