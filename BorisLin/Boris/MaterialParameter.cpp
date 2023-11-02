#include "stdafx.h"
#include "MaterialParameter.h"

#if COMPILECUDA == 1

//-------------- UPDATE CUDA VALUE ONLY

//update just the value at 0K and current value in the corresponding MatPCUDA
template <>
void MatP<float, double>::update_cuda_value(void)
{
	reinterpret_cast<mcu_MatPCUDA(cuBReal, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu_value(*this);
}

template <>
void MatP<double, double>::update_cuda_value(void)
{
	reinterpret_cast<mcu_MatPCUDA(cuBReal, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu_value(*this);
}

template <>
void MatP<FLT2, double>::update_cuda_value(void)
{
	reinterpret_cast<mcu_MatPCUDA(cuReal2, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu_value(*this);
}

template <>
void MatP<DBL2, double>::update_cuda_value(void)
{
	reinterpret_cast<mcu_MatPCUDA(cuReal2, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu_value(*this);
}

template <>
void MatP<FLT3, double>::update_cuda_value(void)
{
	reinterpret_cast<mcu_MatPCUDA(cuReal3, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu_value(*this);
}

template <>
void MatP<DBL3, double>::update_cuda_value(void)
{
	reinterpret_cast<mcu_MatPCUDA(cuReal3, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu_value(*this);
}

template <>
void MatP<FLT3, FLT3>::update_cuda_value(void)
{
	reinterpret_cast<mcu_MatPCUDA(cuReal3, cuReal3)*>(p_cu_obj_mpcuda)->set_from_cpu_value(*this);
}

template <>
void MatP<DBL3, DBL3>::update_cuda_value(void)
{
	reinterpret_cast<mcu_MatPCUDA(cuReal3, cuReal3)*>(p_cu_obj_mpcuda)->set_from_cpu_value(*this);
}

//-------------- UPDATE FULL CUDA OBJECT

//fully update the corresponding MatPCUDA
template <>
void MatP<float, double>::update_cuda_object(void)
{
	//temperature equation
	if (Tscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuBReal, cuBReal)*>(p_cu_obj_mpcuda)->set_t_equation_from_cpu(*pTscaling_CUDAeq, Tscaling_eq.get_scalar_fspec());
	//clear CUDA version so when we call set_from_cpu below the ManagedFunctionCUDA pointer is nulled.
	else pTscaling_CUDAeq->clear();

	//spatial variation equation
	if (Sscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuBReal, cuBReal)*>(p_cu_obj_mpcuda)->set_s_equation_from_cpu(*pSscaling_CUDAeq, Sscaling_eq.get_scalar_fspec());
	//clear CUDA version so when we call set_from_cpu below the ManagedFunctionCUDA pointer is nulled.
	else pSscaling_CUDAeq->clear();

	reinterpret_cast<mcu_MatPCUDA(cuBReal, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu(*this);
}

template <>
void MatP<double, double>::update_cuda_object(void)
{
	//temperature equation
	if (Tscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuBReal, cuBReal)*>(p_cu_obj_mpcuda)->set_t_equation_from_cpu(*pTscaling_CUDAeq, Tscaling_eq.get_scalar_fspec());
	else pTscaling_CUDAeq->clear();

	//spatial variation equation
	if (Sscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuBReal, cuBReal)*>(p_cu_obj_mpcuda)->set_s_equation_from_cpu(*pSscaling_CUDAeq, Sscaling_eq.get_scalar_fspec());
	else pSscaling_CUDAeq->clear();

	reinterpret_cast<mcu_MatPCUDA(cuBReal, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu(*this);
}

template <>
void MatP<FLT2, double>::update_cuda_object(void)
{
	//temperature equation
	if (Tscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal2, cuBReal)*>(p_cu_obj_mpcuda)->set_t_equation_from_cpu(*pTscaling_CUDAeq, Tscaling_eq.get_dual_fspec());
	else pTscaling_CUDAeq->clear();

	//spatial variation equation
	if (Sscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal2, cuBReal)*>(p_cu_obj_mpcuda)->set_s_equation_from_cpu(*pSscaling_CUDAeq, Sscaling_eq.get_scalar_fspec());
	else pSscaling_CUDAeq->clear();

	reinterpret_cast<mcu_MatPCUDA(cuReal2, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu(*this);
}

template <>
void MatP<DBL2, double>::update_cuda_object(void)
{
	//temperature equation
	if (Tscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal2, cuBReal)*>(p_cu_obj_mpcuda)->set_t_equation_from_cpu(*pTscaling_CUDAeq, Tscaling_eq.get_dual_fspec());
	else pTscaling_CUDAeq->clear();

	//spatial variation equation
	if (Sscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal2, cuBReal)*>(p_cu_obj_mpcuda)->set_s_equation_from_cpu(*pSscaling_CUDAeq, Sscaling_eq.get_scalar_fspec());
	else pSscaling_CUDAeq->clear();

	reinterpret_cast<mcu_MatPCUDA(cuReal2, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu(*this);
}

template <>
void MatP<FLT3, double>::update_cuda_object(void)
{
	//temperature equation
	if (Tscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal3, cuBReal)*>(p_cu_obj_mpcuda)->set_t_equation_from_cpu(*pTscaling_CUDAeq, Tscaling_eq.get_vector_fspec());
	else pTscaling_CUDAeq->clear();

	//spatial variation equation
	if (Sscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal3, cuBReal)*>(p_cu_obj_mpcuda)->set_s_equation_from_cpu(*pSscaling_CUDAeq, Sscaling_eq.get_scalar_fspec());
	else pSscaling_CUDAeq->clear();

	reinterpret_cast<mcu_MatPCUDA(cuReal3, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu(*this);
}

template <>
void MatP<DBL3, double>::update_cuda_object(void)
{
	//temperature equation
	if (Tscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal3, cuBReal)*>(p_cu_obj_mpcuda)->set_t_equation_from_cpu(*pTscaling_CUDAeq, Tscaling_eq.get_vector_fspec());
	else pTscaling_CUDAeq->clear();

	//spatial variation equation
	if (Sscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal3, cuBReal)*>(p_cu_obj_mpcuda)->set_s_equation_from_cpu(*pSscaling_CUDAeq, Sscaling_eq.get_scalar_fspec());
	else pSscaling_CUDAeq->clear();

	reinterpret_cast<mcu_MatPCUDA(cuReal3, cuBReal)*>(p_cu_obj_mpcuda)->set_from_cpu(*this);
}

template <>
void MatP<FLT3, FLT3>::update_cuda_object(void)
{
	//temperature equation
	if (Tscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal3, cuReal3)*>(p_cu_obj_mpcuda)->set_t_equation_from_cpu(*pTscaling_CUDAeq, Tscaling_eq.get_vector_fspec());
	else pTscaling_CUDAeq->clear();

	//spatial variation equation
	if (Sscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal3, cuReal3)*>(p_cu_obj_mpcuda)->set_s_equation_from_cpu(*pSscaling_CUDAeq, Sscaling_eq.get_vector_fspec());
	else pSscaling_CUDAeq->clear();

	reinterpret_cast<mcu_MatPCUDA(cuReal3, cuReal3)*>(p_cu_obj_mpcuda)->set_from_cpu(*this);
}

template <>
void MatP<DBL3, DBL3>::update_cuda_object(void)
{
	//temperature equation
	if (Tscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal3, cuReal3)*>(p_cu_obj_mpcuda)->set_t_equation_from_cpu(*pTscaling_CUDAeq, Tscaling_eq.get_vector_fspec());
	else pTscaling_CUDAeq->clear();

	//spatial variation equation
	if (Sscaling_eq.is_set()) reinterpret_cast<mcu_MatPCUDA(cuReal3, cuReal3)*>(p_cu_obj_mpcuda)->set_s_equation_from_cpu(*pSscaling_CUDAeq, Sscaling_eq.get_vector_fspec());
	else pSscaling_CUDAeq->clear();

	reinterpret_cast<mcu_MatPCUDA(cuReal3, cuReal3)*>(p_cu_obj_mpcuda)->set_from_cpu(*this);
}

#endif