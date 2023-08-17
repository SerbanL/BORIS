#pragma once

#include "TEquationCUDA.h"
#include "mcuObj_Math_Special.h"

//Wrapper for multi-GPU TEquationCUDA objects - one constructed for each device, but all are the same.
//Has (almost) same public methods as TEquationCUDA - only the ones needed are implemented here, one which are just used in single-device objects not needed; methods here simply call the respective TEquationCUDA methods in turn on each device.

//mTEquationCUDA replaces TEquationCUDA for multi-devices.

template <typename ... BVarType>
class mTEquationCUDA
{

private:

	//multi-GPU configuration (list of physical devices configured with memory transfer type configuration set)
	mGPUConfig& mGPU;

	//array of size equal to number of devices, each holding a pointer to TEquationCUDA object constructed for each respective device.
	TEquationCUDA<BVarType...>** ppTEquationCUDA = nullptr;

public:

	/////////////////////////////////////////////////////////
	//
	// CONSTRUCTOR

	mTEquationCUDA(mGPUConfig& mGPU_) :
		mGPU(mGPU_)
	{
		ppTEquationCUDA = new TEquationCUDA<BVarType...>*[mGPU.get_num_devices()];

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			ppTEquationCUDA[mGPU] = new TEquationCUDA<BVarType...>();
		}
	}

	~mTEquationCUDA()
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			delete ppTEquationCUDA[mGPU];
			ppTEquationCUDA[mGPU] = nullptr;
		}

		delete[] ppTEquationCUDA;
		ppTEquationCUDA = nullptr;
	}

	/////////////////////////////////////////////////////////
	//
	// MAKE EQUATION

	bool make_scalar(const std::vector< std::vector<EqComp::FSPEC> >& fspec)
	{
		bool success = true;

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			success &= ppTEquationCUDA[mGPU]->make_scalar(fspec);
		}

		return success;
	}

	bool make_dual(const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec)
	{
		bool success = true;

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			success &= ppTEquationCUDA[mGPU]->make_dual(fspec);
		}

		return success;
	}

	bool make_vector(const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec)
	{
		bool success = true;

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			success &= ppTEquationCUDA[mGPU]->make_vector(fspec);
		}

		return success;
	}

	/////////////////////////////////////////////////////////
	//
	// DESTROY EQUATION

	void clear(void) 
	{ 
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			ppTEquationCUDA[mGPU]->clear();
		}
	}

	/////////////////////////////////////////////////////////
	//
	// EQUATION TYPE CHECKING

	//just use index 0 (always available) to check, as we don't need to access device data here
	bool is_set(void) const { return ppTEquationCUDA[0]->is_set(); }

	/////////////////////////////////////////////////////////
	//
	// LOGICAL EQUATION GETTERS

	//Used to pass object to CUDA kernel. Respective device must be selected before, so use within the typical mGPU for loop construct.
	__host__ ManagedFunctionCUDA<BVarType...>& get_x(int device_idx) { return ppTEquationCUDA[device_idx]->get_x(); }
	__host__ ManagedFunctionCUDA<BVarType...>& get_y(int device_idx) { return ppTEquationCUDA[device_idx]->get_y(); }
	__host__ ManagedFunctionCUDA<BVarType...>& get_z(int device_idx) { return ppTEquationCUDA[device_idx]->get_z(); }

	/////////////////////////////////////////////////////////
	//
	// DEVICE TEquationCUDA GETTER

	//Respective device must be selected before, so use within the typical mGPU for loop construct.
	TEquationCUDA<BVarType...>& get_managed_object(int device_idx) { return *ppTEquationCUDA[device_idx]; }

	/////////////////////////////////////////////////////////
	//
	// SET SPECIAL FUNCTIONS

	void Set_SpecialFunction(EqComp::FUNC_ type, mcu_SpecialFunc* pSpecialFunc)
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			ppTEquationCUDA[mGPU]->Set_SpecialFunction(type, pSpecialFunc->get_pcu_object(mGPU));
		}
	}
};