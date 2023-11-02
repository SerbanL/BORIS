#pragma once

#include "TransportCUDA_Poisson.h"

#if COMPILECUDA == 1

class TransportCUDA_V_FuncsPolicy
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<TransportCUDA_V_Funcs, TransportCUDA_V_FuncsPolicy>& mng;

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
	TransportCUDA_V_FuncsPolicy(mcu_obj<TransportCUDA_V_Funcs, TransportCUDA_V_FuncsPolicy>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	TransportCUDA_V_FuncsPolicy& operator=(const TransportCUDA_V_FuncsPolicy& copyThis) { return *this; }

	//destructor
	virtual ~TransportCUDA_V_FuncsPolicy() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// TransportCUDA_V_FuncsPolicy Methods

	//for modules held in micromagnetic meshes
	BError set_pointers(MeshCUDA* pMeshCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(pMeshCUDA, mGPU);
		}

		return error;
	}

	//for modules held in atomistic meshes
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(paMeshCUDA, mGPU);
		}

		return error;
	}

	void set_thermoelectric_mesh_flag(bool status) 
	{ 
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_thermoelectric_mesh_flag(status);
		}
	}
};

#endif