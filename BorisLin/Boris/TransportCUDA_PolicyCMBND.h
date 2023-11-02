#pragma once

#include "TransportCUDA_CMBND.h"

#if COMPILECUDA == 1

template <typename TransportCUDA_CMBND>
class TransportCUDA_PolicyCMBND
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<TransportCUDA_CMBND, TransportCUDA_PolicyCMBND>& mng;

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
	TransportCUDA_PolicyCMBND(mcu_obj<TransportCUDA_CMBND, TransportCUDA_PolicyCMBND>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	TransportCUDA_PolicyCMBND& operator=(const TransportCUDA_PolicyCMBND& copyThis) { return *this; }

	//destructor
	virtual ~TransportCUDA_PolicyCMBND() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// TransportCUDA_PolicyCMBND Methods

	BError set_pointers(MeshCUDA* pMeshCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_number_of_devices_mesh(mGPU.get_num_devices());
			error = mng(mGPU)->set_pointers(pMeshCUDA, mGPU);
		}

		return error;
	}

	BError set_pointers(Atom_MeshCUDA* paMeshCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_number_of_devices_amesh(mGPU.get_num_devices());
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

	void set_open_potential_flag(bool status) 
	{ 
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_open_potential_flag(status);
		}
	}
};

#endif