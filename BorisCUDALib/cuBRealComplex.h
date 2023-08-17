#pragma once

#include "cuBLib_Flags.h"

#include <cufft.h>

#if SINGLEPRECISION == 1

//float (fp32)
#define cuBReal cufftReal
#define cuBComplex cufftComplex

//half-precision (fp16 as uint16_t)
#define cuBHalf uint16_t

#else

//double (fp64)
#define cuBReal cufftDoubleReal
#define cuBComplex cufftDoubleComplex

//float (fp32)
#define cuBHalf cufftReal

#endif