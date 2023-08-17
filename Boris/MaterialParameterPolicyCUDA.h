#pragma once

#include "MaterialParameterCUDA.h"

#if COMPILECUDA == 1

template <typename PType, typename SType>
class MatPPolicyCUDA
{
private:

	//////////////////////////////////////////////////////////////////////////////////
	//
	// SPECIAL OBJECTS FROM CONSTRUCTOR (all Policy classes)

	//reference to mcu_obj manager for which this is a policy class
	mcu_obj<MatPCUDA<PType, SType>, MatPPolicyCUDA<PType, SType>>& mng;

	//multi-GPU configuration (list of physical devices configured with memory transfer type configuration set)
	mGPUConfig& mGPU;

	//////////////////////////////////////////////////////////////////////////////////
	//
	// POLICY CLASS DATA (specific)

	//spatial scaling as a mcu_VEC (each device will hold a pointer to relevant cuVEC held by s_scaling)
	mcu_VEC(SType) s_scaling;

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
	MatPPolicyCUDA(mcu_obj<MatPCUDA<PType, SType>, MatPPolicyCUDA<PType, SType>> & mng_, mGPUConfig& mGPU_) :
		mng(mng_),
		mGPU(mGPU_),
		s_scaling(mGPU_)
	{}

	void construct_policy(void) {}

	//assignment operator
	MatPPolicyCUDA& operator=(const MatPPolicyCUDA& copyThis) { return *this; }

	//destructor
	virtual ~MatPPolicyCUDA() { clear_memory_aux(); }

	//////////////////////////////////////////////////////////////////////////////////
	//
	// MatPPolicyCUDA Methods

	//---------Set MatPCUDA from cpu version (MatP)

	//full copy of MatP : values, scaling array and equation setting with coefficients
	template <typename MatP_PType_>
	void set_from_cpu(MatP_PType_& matp)
	{
		//1. General
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_from_cpu(matp);
		}
		
		//2. Spatial scaling cuVEC
		//set spatial scaling if present
		s_scaling.set_from_cpuvec(matp.s_scaling_ref());

		//don't need cuVEC if equation set.
		if (matp.Sscaling_CUDAeq_ref(0).is_set()) s_scaling.clear();

		//synchronize ps_scaling on each device with data held in s_scaling
		//(make each ps_scaling pointer point to correct device data in s_caling, else if spatial variation net set just null it)
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			if (s_scaling.linear_size_cpu()) mng(mGPU)->set_pointers_ps_scaling(s_scaling.get_managed_object(mGPU));
			else mng(mGPU)->clear_pointers_ps_scaling();
		}

		//3. Temperature dependence text equation
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_pointers_pTscaling_eq(matp.Tscaling_CUDAeq_ref(mGPU));
		}

		//4. Spatial variation text equation
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_pointers_pSscaling_eq(matp.Sscaling_CUDAeq_ref(mGPU));
		}
	}

	//set both value_at_0K and current_value only from MatP
	template <typename MatP_PType_>
	void set_from_cpu_value(MatP_PType_& matp)
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_from_cpu_value(matp);
		}
	}

	//set temperature equation from cpu version : scalar version
	void set_t_equation_from_cpu(mTEquationCUDA<cuBReal>& Tscaling_CUDAeq, const std::vector< std::vector<EqComp::FSPEC> >& fspec)
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_t_equation_from_cpu(Tscaling_CUDAeq.get_managed_object(mGPU), fspec);
		}
	}

	//set temperature equation from cpu version : dual or vector version
	void set_t_equation_from_cpu(mTEquationCUDA<cuBReal>& Tscaling_CUDAeq, const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec)
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_t_equation_from_cpu(Tscaling_CUDAeq.get_managed_object(mGPU), fspec);
		}
	}

	//set spatial variation equation from cpu version : scalar version
	void set_s_equation_from_cpu(mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq, const std::vector< std::vector<EqComp::FSPEC> >& fspec)
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_s_equation_from_cpu(Sscaling_CUDAeq.get_managed_object(mGPU), fspec);
		}
	}

	//set spatial variation equation from cpu version : vector version
	void set_s_equation_from_cpu(mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq, const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec)
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			mng(mGPU)->set_s_equation_from_cpu(Sscaling_CUDAeq.get_managed_object(mGPU), fspec);
		}
	}

	//return current value in cpu memory
	PType get_current_cpu(void) { return mng(mGPU)->get_current_cpu(); }

	PType get0_cpu(void) { return mng(mGPU)->get0_cpu(); }
};

//Macro to simplify declaration
#define mcu_MatPCUDA(PType, SType) mcu_obj<MatPCUDA<PType, SType>, MatPPolicyCUDA<PType, SType>>

#endif