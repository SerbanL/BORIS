#pragma once

#include "cuObj_Math_Special.h"

//mcu_SpecialFunc replaces cu_obj<ManagedFuncs_Special_CUDA> as multi-device container

class mcu_SpecialFunc
{
private:

	//multi-GPU configuration (list of physical devices configured with memory transfer type configuration set)
	mGPUConfig& mGPU;

	//array of size equal to number of devices, each holding a pointer to cu_obj<ManagedFuncs_Special_CUDA> object constructed for each respective device.
	cu_obj<ManagedFuncs_Special_CUDA>** ppManagedFuncs_Special_CUDA = nullptr;

public:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// CONSTRUCTOR

	//--------------------------------------------CONSTRUCTORS

	//constructor for this policy class : void
	mcu_SpecialFunc(mGPUConfig& mGPU_) :
		mGPU(mGPU_)
	{
		ppManagedFuncs_Special_CUDA = new cu_obj<ManagedFuncs_Special_CUDA>*[mGPU.get_num_devices()];

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			ppManagedFuncs_Special_CUDA[mGPU] = new cu_obj<ManagedFuncs_Special_CUDA>();
		}
	}

	//destructor
	~mcu_SpecialFunc() 
	{ 
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			delete ppManagedFuncs_Special_CUDA[mGPU];
			ppManagedFuncs_Special_CUDA[mGPU] = nullptr;
		}

		delete[] ppManagedFuncs_Special_CUDA;
		ppManagedFuncs_Special_CUDA = nullptr;
	}

	//////////////////////////////////////////////////////////////////////////////////
	//
	// mcu_SpecialFunc Methods

	//set gpu data from cpu data
	bool set_data(std::vector<double>& data_cpu, double start_cpu, int resolution_cpu)
	{
		bool success = true;

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			(*ppManagedFuncs_Special_CUDA[mGPU])()->set_data(data_cpu, start_cpu, resolution_cpu);
		}

		return true;
	}

	//Respective device must be selected before, so use within the typical mGPU for loop construct.
	cu_obj<ManagedFuncs_Special_CUDA>*& get_pcu_object(int device_idx) { return ppManagedFuncs_Special_CUDA[device_idx]; }
};