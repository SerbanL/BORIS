#include "MaterialParameterCUDA.h"

#if COMPILECUDA == 1

//NOTE : including the full BorisCUDALib.cuh is not strictly needed here; only #include "TEquationCUDA_Function.cuh" is needed here.
//However, there are a number of cpp compilation units which require code in cuh files (cannot include cuh in cpp). Such code could be moved to cu files and appropriate includes used there, but this can be messy.
//Instead, since MaterialParameterCUDA.h is included in all relevant compilation units, the required compiled code will be available for linking with the inclusion below.
//Should avoid in the future including the full BorisCUDALib.cuh anywhere else - this is the only place in the entire codebase where it is included.
#include "BorisCUDALib.cuh"

//////////////////// TEMPERATURE SCALING

template <>
void MatPCUDA<cuBReal, cuBReal>::set_t_equation_from_cpu(TEquationCUDA<cuBReal>& Tscaling_CUDAeq, const std::vector< std::vector<EqComp::FSPEC> >& fspec)
{
	//make CUDA version of text equation for temperature dependence
	Tscaling_CUDAeq.make_scalar(fspec);
}

template <>
void MatPCUDA<cuReal2, cuBReal>::set_t_equation_from_cpu(TEquationCUDA<cuBReal>& Tscaling_CUDAeq, const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec)
{
	//make CUDA version of text equation for temperature dependence
	
	//dual
	if (fspec[1].size()) Tscaling_CUDAeq.make_dual(fspec);
	
	//scalar
	else Tscaling_CUDAeq.make_scalar(fspec[0]);
}

template <>
void MatPCUDA<cuReal3, cuBReal>::set_t_equation_from_cpu(TEquationCUDA<cuBReal>& Tscaling_CUDAeq, const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec)
{
	//make CUDA version of text equation for temperature dependence
	
	//vector
	if (fspec[1].size() && fspec[2].size()) Tscaling_CUDAeq.make_vector(fspec);

	//scalar
	else Tscaling_CUDAeq.make_scalar(fspec[0]);
}

template <>
void MatPCUDA<cuReal3, cuReal3>::set_t_equation_from_cpu(TEquationCUDA<cuBReal>& Tscaling_CUDAeq, const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec)
{
	//make CUDA version of text equation for temperature dependence
	//vector
	if (fspec[1].size() && fspec[2].size()) Tscaling_CUDAeq.make_vector(fspec);

	//scalar
	else Tscaling_CUDAeq.make_scalar(fspec[0]);
}

//////////////////// SPATIAL VARIATION

//set spatial variation equation from cpu version : scalar version
template <>
void MatPCUDA<cuBReal, cuBReal>::set_s_equation_from_cpu(TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq, const std::vector< std::vector<EqComp::FSPEC> >& fspec)
{
	//make CUDA version of text equation for spatial variation
	Sscaling_CUDAeq.make_scalar(fspec);
}

//set spatial variation equation from cpu version : scalar version
template <>
void MatPCUDA<cuReal2, cuBReal>::set_s_equation_from_cpu(TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq, const std::vector< std::vector<EqComp::FSPEC> >& fspec)
{
	//make CUDA version of text equation for spatial variation
	Sscaling_CUDAeq.make_scalar(fspec);
}

//set spatial variation equation from cpu version : scalar version
template <>
void MatPCUDA<cuReal3, cuBReal>::set_s_equation_from_cpu(TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq, const std::vector< std::vector<EqComp::FSPEC> >& fspec)
{
	//make CUDA version of text equation for spatial variation
	Sscaling_CUDAeq.make_scalar(fspec);
}

//set spatial variation equation from cpu version : vector version
template <>
void MatPCUDA<cuReal3, cuReal3>::set_s_equation_from_cpu(TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq, const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec)
{
	//make CUDA version of text equation for spatial variation
	Sscaling_CUDAeq.make_vector(fspec);
}

#endif
