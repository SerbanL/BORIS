#pragma once

//Only include BorisCUDALib.cuh in CUDA C code files - will not compile in C++ code files

#include "atomics.cuh"
#include "Reduction.cuh"
#include "alloc_cpy.cuh"
#include "cu_prng.cuh"
#include "cuFuncs_Math.cuh"
#include "TEquationCUDA_Function.cuh"

#include "cuFuncs_fp16.cuh"

#include "cuArray_sizing.cuh"
#include "mcuArray_sizing.cuh"

#include "cuVEC_aux.cuh"
#include "cuVEC_mng.cuh"
#include "cuVEC_extract.cuh"
#include "cuVEC_histo.cuh"
#include "cuVEC_generate.cuh"
#include "cuVEC_oper.cuh"
#include "cuVEC_avg.cuh"
#include "cuVEC_nprops.cuh"
#include "cuVEC_arith.cuh"
#include "cuVEC_MeshTransfer.cuh"

#include "cuVEC_VC_mng.cuh"
#include "cuVEC_VC_flags.cuh"
#include "cuVEC_VC_halo.cuh"
#include "cuVEC_VC_shape.cuh"
#include "cuVEC_VC_oper.cuh"
#include "cuVEC_VC_avg.cuh"
#include "cuVEC_VC_nprops.cuh"
#include "cuVEC_VC_arith.cuh"
#include "cuVEC_VC_cmbnd.cuh"
#include "cuVEC_VC_extract.cuh"
#include "cuVEC_VC_solve.cuh"

#include "cuVEC_mcuVEC.cuh"
#include "cuVEC_VC_mcuVEC.cuh"

#include "mGPU_transfer.cuh"

#include "mcuVEC_oper.cuh"
#include "mcuVEC_solve.cuh"
#include "mcuVEC_histo.cuh"
#include "mcuVEC_cmbnd.cuh"

#include "mcuVEC_Managed_avg.cuh"
#include "mcuVEC_MeshTransfer.cuh"
