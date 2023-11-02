#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#include "BorisCUDALib.h"

#include "ParametersDefs.h"

#include "ErrorHandler.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//Class holding a single material parameter with any associated temperature dependence.
//This is designed to be used just as the type stored (PType) would be used on its own in mathematical equations.
//This is achieved by updating the parameter output value (current_value) if the temperature changes.
//1. For uniform temperature just call update(Temperature) whenever the uniform base temperature is changed
//2. For non-uniform temperature the value must be updated in every computational cell before being used, using current_value = get(Temperature). 
//This should be done by an external variadic template function if there are multiple parameters to update.

//NOTE : position here is of crelpos type (relative to collection of cuVECs in mcu_VEC)

template <typename PType, typename SType>
class MatPCUDA
{

private:

	//the parameter value at 0K : when constructing it, this is the value to set. This is also the value displayed/changed in the console. Temperature dependence set separately.
	PType value_at_0K;

	//this is the value that is actually read out and is obtained from value_at_0K depending on the set temperature
	PType current_value;

	//array describing temperature scaling of value_at_0K. The index in this vector is the temperature value in K, i.e. values are set at 1K increments starting at 0K
	cuBReal *pt_scaling;
	int scaling_arr_size;

	//additional components for dual or vector quantities if set
	cuBReal *pt_scaling_y, *pt_scaling_z;

	//temperature scaling equation, if set : takes only the parameter T (temperature); Tc constant : Curie temperature, Tb constant : base temperature
	ManagedFunctionCUDA<cuBReal> *pTscaling_eq_x;
	ManagedFunctionCUDA<cuBReal> *pTscaling_eq_y;
	ManagedFunctionCUDA<cuBReal> *pTscaling_eq_z;

	//spatial scaling, copied from CPU version. Points to cuVEC on relevent device (if set, initialized nullptr), held by s_scaling in MatPPolicyCUDA class.
	cuVEC<SType>* ps_scaling;

	//spatial scaling equation, if set. function of x, y, z, t (stage time); takes parameters Lx, Ly, Lz
	//x, y, and z components for vector equations (or use just x if scalar equation)
	
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal> *pSscaling_eq_x;
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal> *pSscaling_eq_y;
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal> *pSscaling_eq_z;

private:

	//---------Helpers for GPU Memory management

	__host__ void set_value_at_0K(PType value);

	__host__ void set_current_value(PType value);

	__host__ cudaError_t set_scaling_arr_size(int size);
	__host__ cudaError_t set_scaling_arr_y_size(int size);
	__host__ cudaError_t set_scaling_arr_z_size(int size);

	__host__ int get_scaling_arr_size(void);

public:

	//---------Constructors / destructor : cu_obj managed constructors, not real constructors as this object is never instantiated in the real sense

	//void constructor
	__host__ void construct_cu_obj(void);

	//parameter constructor
	__host__ void construct_cu_obj(PType value);

	//destructor
	__host__ void destruct_cu_obj(void);

	//invoked by MatPPolicyCUDA if needed (spatial scaling cuVEC set)
	__host__ void set_pointers_ps_scaling(cuVEC<SType>*& ps_scaling_ondevice) { set_gpu_value(ps_scaling, ps_scaling_ondevice); }
	__host__ void clear_pointers_ps_scaling(void) { nullgpuptr(ps_scaling); }
	__host__ void set_pointers_pTscaling_eq(TEquationCUDA<cuBReal>& Tscaling_CUDAeq);
	__host__ void set_pointers_pSscaling_eq(TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq);

	//---------Set MatPCUDA from cpu version (MatP)

	//full copy of MatP : values, scaling array and equation setting with coefficients
	template <typename MatP_PType_>
	__host__ void set_from_cpu(MatP_PType_& matp);

	//set both value_at_0K and current_value only from MatP
	template <typename MatP_PType_>
	__host__ void set_from_cpu_value(MatP_PType_& matp);

	//set temperature equation from cpu version : scalar version
	__host__ void set_t_equation_from_cpu(TEquationCUDA<cuBReal>& Tscaling_CUDAeq, const std::vector< std::vector<EqComp::FSPEC> >& fspec);

	//set temperature equation from cpu version : dual or vector version
	__host__ void set_t_equation_from_cpu(TEquationCUDA<cuBReal>& Tscaling_CUDAeq, const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec);

	//set spatial variation equation from cpu version : scalar version
	__host__ void set_s_equation_from_cpu(TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq, const std::vector< std::vector<EqComp::FSPEC> >& fspec);

	//set spatial variation equation from cpu version : vector version
	__host__ void set_s_equation_from_cpu(TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq, const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec);

	//---------Output value

	//get value at given temperature and no spatial scaling, but do not update output (meant for use with non-uniform temperature; for uniform temperature it's faster to just read the value through the conversion operator)
	//Use with temperature dependence : YES, spatial variation : NO
	__device__ PType get(cuBReal Temperature = 0);

	//get value (at base temperature) with spatial scaling (must be set so check before!).
	//Use with temperature dependence : NO, spatial variation : YES
	__device__ PType get(const cuReal3& position, cuBReal stime);

	//get value with spatial scaling (must be set so check before!) and temperature dependence
	//Use with temperature dependence : YES, spatial variation : YES
	__device__ PType get(const cuReal3& position, cuBReal stime, cuBReal Temperature);

	//get 0K value
	__device__ PType get0(void) const { return value_at_0K; }

	//get current output value (at base temperature with no spatial scaling)
	__device__ PType get_current(void) const { return current_value; }

	//---------Set value

	//set current value for given temperature, but do not update CUDA value
	__device__ void set_current(cuBReal Temperature);

	//---------Read value

	//use the material parameter just like a PType type to read from it
	__device__ operator PType() { return current_value; }

	//return current value in cpu memory
	__host__ PType get_current_cpu(void);

	__host__ PType get0_cpu(void);

	//---------Get Info

	//does it have a temperature dependence?
	__device__ bool is_tdep(void) const { return (pTscaling_eq_x || scaling_arr_size); }
	__device__ bool is_tdep_eq(void) const { return pTscaling_eq_x; }

	//does it have a spatial dependence?
	__device__ bool is_sdep(void) const { return (pSscaling_eq_x || ps_scaling); }
	__device__ bool is_sdep_eq(void) const { return pSscaling_eq_x; }

	//does it have any kind of dependence set by an equation?
	__device__ bool is_dep_eq(void) const { return pSscaling_eq_x || pTscaling_eq_x; }

	//---------Comparison operators

	//Not currently needed

	//---------Arithmetic operators

	//the return type might not simply be PType! (e.g. PType is a DBL3, so '*' for DBL3 is a scalar product which results in a double). 
	//Need to tell the compiler exactly how to determine the return type. Similar problems may occur for the other arithmetic operators

	//multiply with a MatP - '^' is used to signify vector product for VAL3

	template <typename PType_, typename SType_>
	__device__ auto operator^(const MatPCUDA<PType_, SType_>& rhs) const -> decltype(std::declval<PType_>() ^ std::declval<PType_>())
	{
		return current_value ^ rhs.current_value;
	}

	//multiply with a MatP - '&' is used to signify component-by-component product for VAL3
	template <typename PType_, typename SType_>
	__device__ auto operator&(const MatPCUDA<PType_, SType_>& rhs) const -> decltype(std::declval<PType_>() & std::declval<PType_>())
	{
		return current_value & rhs.current_value;
	}

	//multiply with a MatP
	__device__ auto operator*(const MatPCUDA& rhs) const -> decltype(std::declval<PType>() * std::declval<PType>())
	{
		return current_value * rhs.current_value;
	}

	//multiply with a value on the RHS
	template <typename PType_, std::enable_if_t<!std::is_same<PType_, MatPCUDA<PType, SType>>::value>* = nullptr>
	__device__ auto operator*(const PType_& rhs) const -> decltype(std::declval<PType>() * std::declval<PType_>())
	{
		return current_value * rhs;
	}

	//multiply with a value on the LHS
	template <typename _PType, std::enable_if_t<!std::is_same<_PType, MatPCUDA<PType, SType>>::value>* = nullptr>
	__device__ friend auto operator*(const _PType& lhs, const MatPCUDA<PType, SType>& rhs) -> decltype(std::declval<_PType>() * std::declval<PType>())
	{
		return lhs * rhs.current_value;
	}

	//divide by a MatP
	__device__ auto operator/(const MatPCUDA& rhs) const -> decltype(std::declval<PType>() / std::declval<PType>())
	{
		return current_value / rhs.current_value;
	}

	//division with a value on the RHS
	template <typename PType_, std::enable_if_t<!std::is_same<PType_, MatPCUDA<PType, SType>>::value>* = nullptr>
	__device__ auto operator/(const PType_& rhs) const -> decltype(std::declval<PType>() / std::declval<PType_>())
	{
		return current_value / rhs;
	}

	//division with a value on the LHS
	template <typename _PType, std::enable_if_t<!std::is_same<_PType, MatPCUDA<PType, SType>>::value>* = nullptr>
	__device__ friend auto operator/(const _PType& lhs, const MatPCUDA<PType, SType>& rhs) -> decltype(std::declval<_PType>() / std::declval<PType>())
	{
		return lhs / rhs.current_value;
	}

	//addition with a MatP
	__device__ auto operator+(const MatPCUDA& rhs) const -> decltype(std::declval<PType>() + std::declval<PType>())
	{
		return current_value + rhs.current_value;
	}

	//addition with a value on the RHS
	template <typename PType_, std::enable_if_t<!std::is_same<PType_, MatPCUDA<PType, SType>>::value>* = nullptr>
	__device__ auto operator+(const PType_& rhs) const -> decltype(std::declval<PType>() + std::declval<PType_>())
	{
		return current_value + rhs;
	}

	//addition with a value on the LHS
	template <typename _PType, std::enable_if_t<!std::is_same<_PType, MatPCUDA<PType, SType>>::value>* = nullptr>
	__device__ friend auto operator+(const _PType& lhs, const MatPCUDA<PType, SType>& rhs) -> decltype(std::declval<_PType>() + std::declval<PType>())
	{
		return lhs + rhs.current_value;
	}

	//difference with a MatP
	__device__ auto operator-(const MatPCUDA& rhs) const -> decltype(std::declval<PType>() - std::declval<PType>())
	{
		return current_value - rhs.current_value;
	}

	//addition with a value on the RHS
	template <typename PType_, std::enable_if_t<!std::is_same<PType_, MatPCUDA<PType, SType>>::value>* = nullptr>
	__device__ auto operator-(const PType_& rhs) const -> decltype(std::declval<PType>() - std::declval<PType_>())
	{
		return current_value - rhs;
	}

	//addition with a value on the LHS
	template <typename _PType, std::enable_if_t<!std::is_same<_PType, MatPCUDA<PType, SType>>::value>* = nullptr>
	__device__ friend auto operator-(const _PType& lhs, const MatPCUDA<PType, SType>& rhs) -> decltype(std::declval<_PType>() - std::declval<PType>())
	{
		return lhs - rhs.current_value;
	}
};

//---------Helpers for GPU Memory management

template <typename PType, typename SType>
__host__ void MatPCUDA<PType, SType>::set_value_at_0K(PType value)
{
	set_gpu_value(value_at_0K, value);
}

template <typename PType, typename SType>
__host__ void MatPCUDA<PType, SType>::set_current_value(PType value)
{
	set_gpu_value(current_value, value);
}

template <typename PType, typename SType>
__host__ cudaError_t MatPCUDA<PType, SType>::set_scaling_arr_size(int size)
{
	if (size == get_scaling_arr_size()) return cudaSuccess;

	cudaError_t error = gpu_alloc_managed(pt_scaling, size);
	if (error == cudaSuccess) {

		set_gpu_value(scaling_arr_size, size);
	}
	else {

		nullgpuptr(pt_scaling);
		set_gpu_value(scaling_arr_size, (int)0);
	}

	return error;
}

template <typename PType, typename SType>
__host__ cudaError_t MatPCUDA<PType, SType>::set_scaling_arr_y_size(int size)
{
	//only attempt to set array if it matches dimensions of primary array (first component)
	if (size != get_scaling_arr_size()) return cudaErrorInvalidValue;

	cudaError_t error = gpu_alloc_managed(pt_scaling_y, size);
	if (error != cudaSuccess) {

		nullgpuptr(pt_scaling_y);
	}

	return error;
}

template <typename PType, typename SType>
__host__ cudaError_t MatPCUDA<PType, SType>::set_scaling_arr_z_size(int size)
{
	//only attempt to set array if it matches dimensions of primary array (first component)
	if (size != get_scaling_arr_size()) return cudaErrorInvalidValue;

	cudaError_t error = gpu_alloc_managed(pt_scaling_z, size);
	if (error != cudaSuccess) {

		nullgpuptr(pt_scaling_z);
	}

	return error;
}

template <typename PType, typename SType>
__host__ int MatPCUDA<PType, SType>::get_scaling_arr_size(void)
{
	return get_gpu_value(scaling_arr_size);
}

//---------Constructors / destructor : cu_obj managed constructors, not real constructors as this object is never instantiated in the real sense

//void constructor
template <typename PType, typename SType>
__host__ void MatPCUDA<PType, SType>::construct_cu_obj(void)
{
	nullgpuptr(pt_scaling);
	nullgpuptr(pt_scaling_y);
	nullgpuptr(pt_scaling_z);
	set_gpu_value(scaling_arr_size, (int)0);
	set_value_at_0K(PType());
	set_current_value(PType());

	nullgpuptr(ps_scaling);

	nullgpuptr(pTscaling_eq_x);
	nullgpuptr(pTscaling_eq_y);
	nullgpuptr(pTscaling_eq_z);

	nullgpuptr(pSscaling_eq_x);
	nullgpuptr(pSscaling_eq_y);
	nullgpuptr(pSscaling_eq_z);
}

//parameter constructor
template <typename PType, typename SType>
__host__ void MatPCUDA<PType, SType>::construct_cu_obj(PType value)
{
	nullgpuptr(pt_scaling);
	nullgpuptr(pt_scaling_y);
	nullgpuptr(pt_scaling_z);
	set_gpu_value(scaling_arr_size, (int)0);
	set_value_at_0K(value);
	set_current_value(value);

	nullgpuptr(ps_scaling);

	nullgpuptr(pTscaling_eq_x);
	nullgpuptr(pTscaling_eq_y);
	nullgpuptr(pTscaling_eq_z);

	nullgpuptr(pSscaling_eq_x);
	nullgpuptr(pSscaling_eq_y);
	nullgpuptr(pSscaling_eq_z);
}

//destructor
template <typename PType, typename SType>
__host__ void MatPCUDA<PType, SType>::destruct_cu_obj(void)
{
	gpu_free_managed(pt_scaling);
	gpu_free_managed(pt_scaling_y);
	gpu_free_managed(pt_scaling_z);

	//don't need to invoke destruct_cu_obj on ps_scaling, since this is handled in MatPPolicyCUDA, where memory has (possibly) been allocated.
}

//---------Set MatPCUDA from cpu version (MatP)

//full copy of MatP : values, scaling array and equation setting with coefficients
template <typename PType, typename SType>
template <typename MatP_PType_>
__host__ void MatPCUDA<PType, SType>::set_from_cpu(MatP_PType_& matp)
{
	//need : value_at_0K, current_value, t_scaling vector, equation_selector, equation coefficients

	//0K and current value
	set_value_at_0K((PType)matp.get0());
	set_current_value((PType)matp.get_current());

	//the scaling array - we'll need to convert this
	std::vector<double>& t_scaling_cpu = matp.t_scaling_ref();
	std::vector<double>& t_scaling_y_cpu = matp.t_scaling_y_ref();
	std::vector<double>& t_scaling_z_cpu = matp.t_scaling_z_ref();

	cudaError_t error = set_scaling_arr_size(t_scaling_cpu.size());
	if (error == cudaSuccess && t_scaling_cpu.size()) {

		cpu_to_gpu_managed(pt_scaling, t_scaling_cpu.data(), t_scaling_cpu.size());

		if (t_scaling_y_cpu.size() && t_scaling_y_cpu.size() == t_scaling_cpu.size()) {

			cudaError_t error = set_scaling_arr_y_size(t_scaling_cpu.size());
			if (error == cudaSuccess) cpu_to_gpu_managed(pt_scaling_y, t_scaling_y_cpu.data(), t_scaling_y_cpu.size());
		}
		else nullgpuptr(pt_scaling_y);

		if (t_scaling_z_cpu.size() && t_scaling_z_cpu.size() == t_scaling_cpu.size()) {

			cudaError_t error = set_scaling_arr_z_size(t_scaling_cpu.size());
			if (error == cudaSuccess) cpu_to_gpu_managed(pt_scaling_z, t_scaling_z_cpu.data(), t_scaling_z_cpu.size());
		}
		else nullgpuptr(pt_scaling_z);
	}
	else {

		nullgpuptr(pt_scaling);
		nullgpuptr(pt_scaling_y);
		nullgpuptr(pt_scaling_z);
	}

	//spatial scaling set in MatPPolicyCUDA
	nullgpuptr(ps_scaling);

	//text equations set in MatPPolicyCUDA
	nullgpuptr(pTscaling_eq_x);
	nullgpuptr(pTscaling_eq_y);
	nullgpuptr(pTscaling_eq_z);

	nullgpuptr(pSscaling_eq_x);
	nullgpuptr(pSscaling_eq_y);
	nullgpuptr(pSscaling_eq_z);

}

//set both value_at_0K and current_value only from MatP
template <typename PType, typename SType>
template <typename MatP_PType_>
__host__ void MatPCUDA<PType, SType>::set_from_cpu_value(MatP_PType_& matp)
{
	set_value_at_0K(matp.get0());
	set_current_value(matp.get_current());
}

template <typename PType, typename SType>
__host__ void MatPCUDA<PType, SType>::set_pointers_pTscaling_eq(TEquationCUDA<cuBReal>& Tscaling_CUDAeq)
{
	if (Tscaling_CUDAeq.is_set()) {

		//don't need arrays if equation set
		nullgpuptr(pt_scaling);
		nullgpuptr(pt_scaling_y);
		nullgpuptr(pt_scaling_z);

		if (Tscaling_CUDAeq.get_pcu_obj_x()) set_gpu_value(pTscaling_eq_x,Tscaling_CUDAeq.get_pcu_obj_x()->get_managed_object());
		else nullgpuptr(pTscaling_eq_x);

		if (Tscaling_CUDAeq.get_pcu_obj_y()) set_gpu_value(pTscaling_eq_y, Tscaling_CUDAeq.get_pcu_obj_y()->get_managed_object());
		else nullgpuptr(pTscaling_eq_y);

		if (Tscaling_CUDAeq.get_pcu_obj_z()) set_gpu_value(pTscaling_eq_z, Tscaling_CUDAeq.get_pcu_obj_z()->get_managed_object());
		else nullgpuptr(pTscaling_eq_z);
	}
	else {

		nullgpuptr(pTscaling_eq_x);
		nullgpuptr(pTscaling_eq_y);
		nullgpuptr(pTscaling_eq_z);
	}
}

template <typename PType, typename SType>
__host__ void MatPCUDA<PType, SType>::set_pointers_pSscaling_eq(TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq)
{
	if (Sscaling_CUDAeq.is_set()) {

		//don't need cuVEC if equation set.
		nullgpuptr(ps_scaling);

		if (Sscaling_CUDAeq.get_pcu_obj_x()) set_gpu_value(pSscaling_eq_x, Sscaling_CUDAeq.get_pcu_obj_x()->get_managed_object());
		else nullgpuptr(pSscaling_eq_x);

		if (Sscaling_CUDAeq.get_pcu_obj_y()) set_gpu_value(pSscaling_eq_y, Sscaling_CUDAeq.get_pcu_obj_y()->get_managed_object());
		else nullgpuptr(pSscaling_eq_y);

		if (Sscaling_CUDAeq.get_pcu_obj_z()) set_gpu_value(pSscaling_eq_z, Sscaling_CUDAeq.get_pcu_obj_z()->get_managed_object());
		else nullgpuptr(pSscaling_eq_z);
	}
	else {

		nullgpuptr(pSscaling_eq_x);
		nullgpuptr(pSscaling_eq_y);
		nullgpuptr(pSscaling_eq_z);
	}
}

//---------Output value

//---------- TEMPERATURE ONLY

//get value at given temperature but do not update output (use it for non-uniform temperatures)
template <>
__device__ inline cuBReal MatPCUDA<cuBReal, cuBReal>::get(cuBReal Temperature)
{
	if (Temperature > MAX_TEMPERATURE || isnan(Temperature)) return value_at_0K;

	if (pTscaling_eq_x) {

		//use pre-set equation
		return value_at_0K * pTscaling_eq_x->evaluate(Temperature);
	}
	else if (scaling_arr_size && Temperature > 0) {

		//use custom temperature scaling
		int index = (int)cu_floor_epsilon(Temperature);
		if (index + 1 < scaling_arr_size && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1
			return (value_at_0K * (pt_scaling[index] * (cuBReal(index + 1) - Temperature) + pt_scaling[index + 1] * (Temperature - cuBReal(index))));
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else return (value_at_0K * pt_scaling[scaling_arr_size - 1]);
	}
	//no temperature dependence set
	else return value_at_0K;
}

template <>
__device__ inline cuReal2 MatPCUDA<cuReal2, cuBReal>::get(cuBReal Temperature)
{
	if (Temperature > MAX_TEMPERATURE || isnan(Temperature)) return value_at_0K;

	if (pTscaling_eq_x) {

		//component-by-component product if dual equation
		if (pTscaling_eq_y) return (value_at_0K & cuReal2(pTscaling_eq_x->evaluate(Temperature), pTscaling_eq_y->evaluate(Temperature)));

		//just constant multiplication for a scalar equation
		else return value_at_0K * pTscaling_eq_x->evaluate(Temperature);
	}
	else if (scaling_arr_size && Temperature > 0) {

		//use custom temperature scaling
		int index = (int)cu_floor_epsilon(Temperature);
		if (index + 1 < scaling_arr_size && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1

			//component-by-component product if dual equation
			if (pt_scaling_y) return (value_at_0K & (cuReal2(pt_scaling[index], pt_scaling_y[index]) * (cuBReal(index + 1) - Temperature) + cuReal2(pt_scaling[index + 1], pt_scaling_y[index + 1]) * (Temperature - cuBReal(index))));

			//just constant multiplication for a scalar equation
			return (value_at_0K * (pt_scaling[index] * (cuBReal(index + 1) - Temperature) + pt_scaling[index + 1] * (Temperature - cuBReal(index))));
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (pt_scaling_y) return (value_at_0K & cuReal2(pt_scaling[scaling_arr_size - 1], pt_scaling_y[scaling_arr_size - 1]));
			else return (value_at_0K * pt_scaling[scaling_arr_size - 1]);
		}
	}
	//no temperature dependence set
	else return value_at_0K;
}

template <>
__device__ inline cuReal3 MatPCUDA<cuReal3, cuBReal>::get(cuBReal Temperature)
{
	if (Temperature > MAX_TEMPERATURE || isnan(Temperature)) return value_at_0K;

	if (pTscaling_eq_x) {

		//component-by-component product if vector equation
		if (pTscaling_eq_y && pTscaling_eq_z) return (value_at_0K & cuReal3(pTscaling_eq_x->evaluate(Temperature), pTscaling_eq_y->evaluate(Temperature), pTscaling_eq_z->evaluate(Temperature)));

		//just constant multiplication for a scalar equation
		else return value_at_0K * pTscaling_eq_x->evaluate(Temperature);
	}
	else if (scaling_arr_size && Temperature > 0) {

		//use custom temperature scaling
		int index = (int)cu_floor_epsilon(Temperature);
		if (index + 1 < scaling_arr_size && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1

			//component-by-component product if vector equation
			if (pt_scaling_z) return (value_at_0K & (cuReal3(pt_scaling[index], pt_scaling_y[index], pt_scaling_z[index]) * (cuBReal(index + 1) - Temperature) + cuReal3(pt_scaling[index + 1], pt_scaling_y[index + 1], pt_scaling_z[index + 1]) * (Temperature - cuBReal(index))));

			//just constant multiplication for a scalar equation
			return (value_at_0K * (pt_scaling[index] * (cuBReal(index + 1) - Temperature) + pt_scaling[index + 1] * (Temperature - cuBReal(index))));
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (pt_scaling_z) return (value_at_0K & cuReal3(pt_scaling[scaling_arr_size - 1], pt_scaling_y[scaling_arr_size - 1], pt_scaling_z[scaling_arr_size - 1]));
			else return (value_at_0K * pt_scaling[scaling_arr_size - 1]);
		}
	}
	//no temperature dependence set
	else return value_at_0K;
}

template <>
__device__ inline cuReal3 MatPCUDA<cuReal3, cuReal3>::get(cuBReal Temperature)
{
	if (Temperature > MAX_TEMPERATURE || isnan(Temperature)) return value_at_0K;

	if (pTscaling_eq_x) {

		//component-by-component product if vector equation
		if (pTscaling_eq_y && pTscaling_eq_z) return (value_at_0K & cuReal3(pTscaling_eq_x->evaluate(Temperature), pTscaling_eq_y->evaluate(Temperature), pTscaling_eq_z->evaluate(Temperature)));

		//just constant multiplication for a scalar equation
		else return value_at_0K * pTscaling_eq_x->evaluate(Temperature);
	}
	else if (scaling_arr_size && Temperature > 0) {

		//use custom temperature scaling
		int index = (int)cu_floor_epsilon(Temperature);
		if (index + 1 < scaling_arr_size && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1

			//component-by-component product if vector equation
			if (pt_scaling_z) return (value_at_0K & (cuReal3(pt_scaling[index], pt_scaling_y[index], pt_scaling_z[index]) * (cuBReal(index + 1) - Temperature) + cuReal3(pt_scaling[index + 1], pt_scaling_y[index + 1], pt_scaling_z[index + 1]) * (Temperature - cuBReal(index))));

			//just constant multiplication for a scalar equation
			return (value_at_0K * (pt_scaling[index] * (cuBReal(index + 1) - Temperature) + pt_scaling[index + 1] * (Temperature - cuBReal(index))));
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (pt_scaling_z) return (value_at_0K & cuReal3(pt_scaling[scaling_arr_size - 1], pt_scaling_y[scaling_arr_size - 1], pt_scaling_z[scaling_arr_size - 1]));
			else return (value_at_0K * pt_scaling[scaling_arr_size - 1]);
		}
	}
	//no temperature dependence set
	else return value_at_0K;
}

//---------- SPATIAL ONLY

//get value (at base temperature) with spatial scaling (must be set so check before!)
//Use with temperature dependence : NO, spatial variation : YES
template <>
__device__ inline cuBReal MatPCUDA<cuBReal, cuBReal>::get(const cuReal3& position, cuBReal stime)
{
	if (pSscaling_eq_x) return current_value * pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime);
	else return current_value * (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)];
}

template <>
__device__ inline cuReal2 MatPCUDA<cuReal2, cuBReal>::get(const cuReal3& position, cuBReal stime)
{
	if (pSscaling_eq_x) return current_value * pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime);
	else return current_value * (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)];
}


template <>
__device__ inline cuReal3 MatPCUDA<cuReal3, cuBReal>::get(const cuReal3& position, cuBReal stime)
{
	if (pSscaling_eq_x) return current_value * pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime);
	else return current_value * (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)];
}

template <>
__device__ inline cuReal3 MatPCUDA<cuReal3, cuReal3>::get(const cuReal3& position, cuBReal stime)
{
	if (pSscaling_eq_z) return rotate_polar(current_value, 
		cuReal3(
			pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime),
			pSscaling_eq_y->evaluate(position.x, position.y, position.z, stime),
			pSscaling_eq_z->evaluate(position.x, position.y, position.z, stime))
		);
	else return rotate_polar(current_value, (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)]);
}

//---------- TEMPERATURE and SPATIAL

//get value with spatial scaling (must be set so check before!) and temperature dependence
//Use with temperature dependence : YES, spatial variation : YES
template <>
__device__ inline cuBReal MatPCUDA<cuBReal, cuBReal>::get(const cuReal3& position, cuBReal stime, cuBReal Temperature)
{
	if (Temperature > MAX_TEMPERATURE || isnan(Temperature)) return value_at_0K;

	if (pTscaling_eq_x) {

		if (pSscaling_eq_x) return (value_at_0K * pTscaling_eq_x->evaluate(Temperature) * pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime));
		else return (value_at_0K * pTscaling_eq_x->evaluate(Temperature) * (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)]);
	}
	else if (scaling_arr_size && Temperature > 0) {

		//use custom temperature scaling
		int index = (int)cu_floor_epsilon(Temperature);
		if (index + 1 < scaling_arr_size && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1
			if (pSscaling_eq_x) return (value_at_0K * (pt_scaling[index] * (cuBReal(index + 1) - Temperature) + pt_scaling[index + 1] * (Temperature - cuBReal(index)))) * pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime);
			else return (value_at_0K * (pt_scaling[index] * (cuBReal(index + 1) - Temperature) + pt_scaling[index + 1] * (Temperature - cuBReal(index)))) * (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)];
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (pSscaling_eq_x) return (value_at_0K * pt_scaling[scaling_arr_size - 1] * pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime));
			else return (value_at_0K * pt_scaling[scaling_arr_size - 1] * (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)]);
		}
	}
	//no temperature dependence set
	else {

		if (pSscaling_eq_x) return value_at_0K * pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime);
		else return value_at_0K * (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)];
	}
}

template <>
__device__ inline cuReal2 MatPCUDA<cuReal2, cuBReal>::get(const cuReal3& position, cuBReal stime, cuBReal Temperature)
{
	if (Temperature > MAX_TEMPERATURE || isnan(Temperature)) return value_at_0K;

	if (pTscaling_eq_x) {

		cuBReal s = (pSscaling_eq_x ? pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime) : (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)]);

		//component-by-component product if dual equation
		if (pTscaling_eq_y) return (value_at_0K & cuReal2(pTscaling_eq_x->evaluate(Temperature), pTscaling_eq_y->evaluate(Temperature))) * s;

		//just constant multiplication for a scalar equation
		else return value_at_0K * pTscaling_eq_x->evaluate(Temperature) * s;
	}
	else if (scaling_arr_size && Temperature > 0) {

		cuBReal s = (pSscaling_eq_x ? pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime) : (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)]);

		//use custom temperature scaling
		int index = (int)cu_floor_epsilon(Temperature);
		if (index + 1 < scaling_arr_size && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1

			//component-by-component product if dual equation
			if (pt_scaling_y) return (value_at_0K & (cuReal2(pt_scaling[index], pt_scaling_y[index]) * (cuBReal(index + 1) - Temperature) + cuReal2(pt_scaling[index + 1], pt_scaling_y[index + 1]) * (Temperature - cuBReal(index)))) * s;

			//just constant multiplication for a scalar equation
			return (value_at_0K * (pt_scaling[index] * (cuBReal(index + 1) - Temperature) + pt_scaling[index + 1] * (Temperature - cuBReal(index)))) * s;
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (pt_scaling_y) return (value_at_0K & cuReal2(pt_scaling[scaling_arr_size - 1], pt_scaling_y[scaling_arr_size - 1])) * s;
			else return (value_at_0K * pt_scaling[scaling_arr_size - 1]) * s;
		}
	}
	//no temperature dependence set
	else {

		if (pSscaling_eq_x) return value_at_0K * pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime);
		else return value_at_0K * (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)];
	}
}

template <>
__device__ inline cuReal3 MatPCUDA<cuReal3, cuBReal>::get(const cuReal3& position, cuBReal stime, cuBReal Temperature)
{
	if (Temperature > MAX_TEMPERATURE || isnan(Temperature)) return value_at_0K;

	if (pTscaling_eq_x) {

		cuBReal s = (pSscaling_eq_x ? pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime) : (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)]);

		//component-by-component product if vector equation
		if (pTscaling_eq_y && pTscaling_eq_z) return (value_at_0K & cuReal3(pTscaling_eq_x->evaluate(Temperature), pTscaling_eq_y->evaluate(Temperature), pTscaling_eq_z->evaluate(Temperature))) * s;

		//just constant multiplication for a scalar equation
		else return value_at_0K * pTscaling_eq_x->evaluate(Temperature) * s;
	}
	else if (scaling_arr_size && Temperature > 0) {

		cuBReal s = (pSscaling_eq_x ? pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime) : (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)]);

		//use custom temperature scaling
		int index = (int)cu_floor_epsilon(Temperature);
		if (index + 1 < scaling_arr_size && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1

			//component-by-component product if vector equation
			if (pt_scaling_z) return (value_at_0K & (cuReal3(pt_scaling[index], pt_scaling_y[index], pt_scaling_z[index]) * (cuBReal(index + 1) - Temperature) + cuReal3(pt_scaling[index + 1], pt_scaling_y[index + 1], pt_scaling_z[index + 1]) * (Temperature - cuBReal(index)))) * s;

			//just constant multiplication for a scalar equation
			return (value_at_0K * (pt_scaling[index] * (cuBReal(index + 1) - Temperature) + pt_scaling[index + 1] * (Temperature - cuBReal(index)))) * s;
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (pt_scaling_z) return (value_at_0K & cuReal3(pt_scaling[scaling_arr_size - 1], pt_scaling_y[scaling_arr_size - 1], pt_scaling_z[scaling_arr_size - 1])) * s;
			else return (value_at_0K * pt_scaling[scaling_arr_size - 1]) * s;
		}
	}
	//no temperature dependence set
	else {

		if (pSscaling_eq_x) return value_at_0K * pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime);
		else return value_at_0K * (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)];
	}
}

template <>
__device__ inline cuReal3 MatPCUDA<cuReal3, cuReal3>::get(const cuReal3& position, cuBReal stime, cuBReal Temperature)
{
	if (Temperature > MAX_TEMPERATURE || isnan(Temperature)) return value_at_0K;

	if (pTscaling_eq_x) {

		cuReal3 s = (pSscaling_eq_z ? 
			cuReal3(
			pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime),
			pSscaling_eq_y->evaluate(position.x, position.y, position.z, stime),
			pSscaling_eq_z->evaluate(position.x, position.y, position.z, stime)) : (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)]);

		//component-by-component product if vector equation
		if (pTscaling_eq_y && pTscaling_eq_z) return rotate_polar(value_at_0K & cuReal3(pTscaling_eq_x->evaluate(Temperature), pTscaling_eq_y->evaluate(Temperature), pTscaling_eq_z->evaluate(Temperature)), s);

		//just constant multiplication for a scalar equation
		else return rotate_polar(value_at_0K * pTscaling_eq_x->evaluate(Temperature), s);
	}
	else if (scaling_arr_size && Temperature > 0) {

		cuReal3 s = (pSscaling_eq_z ?
			cuReal3(
				pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime),
				pSscaling_eq_y->evaluate(position.x, position.y, position.z, stime),
				pSscaling_eq_z->evaluate(position.x, position.y, position.z, stime)) : (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)]);

		//use custom temperature scaling
		int index = (int)cu_floor_epsilon(Temperature);
		if (index + 1 < scaling_arr_size && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1

			//component-by-component product if vector equation
			if (pt_scaling_z) return rotate_polar(value_at_0K & (cuReal3(pt_scaling[index], pt_scaling_y[index], pt_scaling_z[index]) * (cuBReal(index + 1) - Temperature) + cuReal3(pt_scaling[index + 1], pt_scaling_y[index + 1], pt_scaling_z[index + 1]) * (Temperature - cuBReal(index))), s);

			//just constant multiplication for a scalar equation
			return rotate_polar(value_at_0K * (pt_scaling[index] * (cuBReal(index + 1) - Temperature) + pt_scaling[index + 1] * (Temperature - cuBReal(index))), s);
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (pt_scaling_z) return rotate_polar(value_at_0K & cuReal3(pt_scaling[scaling_arr_size - 1], pt_scaling_y[scaling_arr_size - 1], pt_scaling_z[scaling_arr_size - 1]), s);
			else return rotate_polar(value_at_0K * pt_scaling[scaling_arr_size - 1], s);
		}
	}
	//no temperature dependence set
	else {

		if (pSscaling_eq_z) return rotate_polar(
			value_at_0K, 
			cuReal3(
				pSscaling_eq_x->evaluate(position.x, position.y, position.z, stime),
				pSscaling_eq_y->evaluate(position.x, position.y, position.z, stime),
				pSscaling_eq_z->evaluate(position.x, position.y, position.z, stime)));
		else return rotate_polar(value_at_0K, (*ps_scaling)[ps_scaling->get_relpos_from_crelpos(position)]);
	}
}

//---------Set value

//set current value for given temperature
template <typename PType, typename SType>
__device__ void MatPCUDA<PType, SType>::set_current(cuBReal Temperature)
{
	current_value = get(Temperature);
}

//---------Read value

//return current value in cpu memory
template <typename PType, typename SType>
__host__ PType MatPCUDA<PType, SType>::get_current_cpu(void)
{
	return get_gpu_value(current_value);
}

//return value at 0K in cpu memory
template <typename PType, typename SType>
__host__ PType MatPCUDA<PType, SType>::get0_cpu(void)
{
	return get_gpu_value(value_at_0K);
}

#endif