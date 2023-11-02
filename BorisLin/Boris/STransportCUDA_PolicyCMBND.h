#pragma once

#include "STransportCUDA_GInterf_V.h"
#include "STransportCUDA_GInterf_S.h"

#if COMPILECUDA == 1

template <typename STransportCUDA_CMBND>
class STransportCUDA_PolicyCMBND
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<STransportCUDA_CMBND, STransportCUDA_PolicyCMBND>& mng;

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
	STransportCUDA_PolicyCMBND(mcu_obj<STransportCUDA_CMBND, STransportCUDA_PolicyCMBND>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	STransportCUDA_PolicyCMBND& operator=(const STransportCUDA_PolicyCMBND& copyThis) { return *this; }

	//destructor
	virtual ~STransportCUDA_PolicyCMBND() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// STransportCUDA_PolicyCMBND Methods
};

#endif