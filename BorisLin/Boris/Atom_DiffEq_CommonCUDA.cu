#include "Atom_DiffEq_CommonCUDA.h"

#if COMPILECUDA == 1

//defines evaluation methods kernel launchers

//-----------------------------------------

__global__ void Zero_SD_Solver_BB_Values_Atom_kernel(cuBReal& delta_m_sq, cuBReal& delta_G_sq, cuBReal& delta_m_dot_delta_G)
{
	if (threadIdx.x == 0) delta_m_sq = 0.0;
	else if (threadIdx.x == 1) delta_G_sq = 0.0;
	else if (threadIdx.x == 2) delta_m_dot_delta_G = 0.0;
}

void Atom_ODECommonCUDA::Zero_SD_Solver_BB_Values(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Zero_SD_Solver_BB_Values_Atom_kernel <<< 1, CUDATHREADS >>> 
			((*pdelta_M_sq)(mGPU), (*pdelta_G_sq)(mGPU), (*pdelta_M_dot_delta_G)(mGPU));
	}
}

#endif