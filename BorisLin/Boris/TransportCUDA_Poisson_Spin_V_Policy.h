#pragma once

#include "TransportCUDA_Poisson_Spin_V.h"

#if COMPILECUDA == 1

class TransportCUDA_Spin_V_FuncsPolicy
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<TransportCUDA_Spin_V_Funcs, TransportCUDA_Spin_V_FuncsPolicy>& mng;

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
	TransportCUDA_Spin_V_FuncsPolicy(mcu_obj<TransportCUDA_Spin_V_Funcs, TransportCUDA_Spin_V_FuncsPolicy>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	TransportCUDA_Spin_V_FuncsPolicy& operator=(const TransportCUDA_Spin_V_FuncsPolicy& copyThis) { return *this; }

	//destructor
	virtual ~TransportCUDA_Spin_V_FuncsPolicy() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// TransportCUDA_Spin_V_FuncsPolicy Methods

	//for modules held in micromagnetic meshes
	BError set_pointers(MeshCUDA* pMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(pMeshCUDA, pTransportBaseCUDA, mGPU);
		}

		return error;
	}

	//for modules held in atomistic meshes
	BError set_pointers(Atom_MeshCUDA* paMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			error = mng(mGPU)->set_pointers(paMeshCUDA, pTransportBaseCUDA, mGPU);
		}

		return error;
	}

	void set_stsolve(int stsolve_)
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_stsolve(stsolve_);
		}
	}
};

#endif