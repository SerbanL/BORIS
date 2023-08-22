#pragma once

#include "MElastic_BoundariesCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MELASTIC

class MElastic_PolicyBoundaryCUDA
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<MElastic_BoundaryCUDA, MElastic_PolicyBoundaryCUDA>& mng;

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
	MElastic_PolicyBoundaryCUDA(mcu_obj<MElastic_BoundaryCUDA, MElastic_PolicyBoundaryCUDA>& mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_)
	{}

	void construct_policy(void) {}

	//destructor
	virtual ~MElastic_PolicyBoundaryCUDA() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// MElastic_PolicyBoundaryCUDA Methods

	////////////////////////////////////// SETUP

	void setup_surface(mcu_VEC_VC(cuReal3)& u_disp, cuBox cells_box, int surf_orientation)
	{
		//cells_box is the entire box which intersect with the mesh of u_disp (relative to entire mesh)
		//split it up between devices so each manages the box which intersects with it (and is relative to each device respectively)
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			cuBox device_box = u_disp.device_box(mGPU);
			cuBox cells_box_device = cells_box.get_intersection(device_box) - device_box.s;

			mng(mGPU)->setup_surface(cells_box_device, u_disp.h, surf_orientation);
		}
	}

	////////////////////////////////////// EQUATION STIMULUS

	//from Fext_equation make the CUDA version
	void make_cuda_equation(mTEquationCUDA<cuBReal, cuBReal, cuBReal>& Fext_equationCUDA)
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->make_cuda_equation(Fext_equationCUDA.get_managed_object(mGPU));
		}
	}

	////////////////////////////////////// FIXED STIMULUS

	void setup_fixed_stimulus(cuReal3 Fext_constant)
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->setup_fixed_stimulus(Fext_constant);
		}
	}
};

#endif

#endif