#pragma once

#include "HeatCUDA_CMBND.h"

#if COMPILECUDA == 1

template <typename HeatCUDA_CMBND>
class HeatCUDA_PolicyCMBND
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<HeatCUDA_CMBND, HeatCUDA_PolicyCMBND>& mng;

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
	HeatCUDA_PolicyCMBND(mcu_obj<HeatCUDA_CMBND, HeatCUDA_PolicyCMBND>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	HeatCUDA_PolicyCMBND& operator=(const HeatCUDA_PolicyCMBND& copyThis) { return *this; }

	//destructor
	virtual ~HeatCUDA_PolicyCMBND() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// HeatCUDA_PolicyCMBND Methods

	BError set_pointers(MeshCUDA* pMeshCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_number_of_devices(mGPU.get_num_devices());
			error = mng(mGPU)->set_pointers(pMeshCUDA, mGPU);
		}

		return error;
	}

	BError set_pointers(Atom_MeshCUDA* paMeshCUDA)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_number_of_devices(mGPU.get_num_devices());
			error = mng(mGPU)->set_pointers(paMeshCUDA, mGPU);
		}

		return error;
	}

	//set spatial variation equation from cpu version : scalar version
	BError set_Q_equation(mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Q_equation)
	{
		BError error(__FUNCTION__);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_number_of_devices(mGPU.get_num_devices());
			error = mng(mGPU)->set_Q_equation(Q_equation, mGPU);
		}

		return error;
	}
};

#endif