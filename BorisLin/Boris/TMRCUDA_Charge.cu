#include "TMRCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TMR

#include "MeshCUDA.h"

#include "mcuVEC_solve.cuh"

//-------------------Calculation Methods

void TMRCUDA::IterateChargeSolver_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value)
{
	pMeshCUDA->V.IteratePoisson_SOR(poisson_V, damping, max_error, max_value);
}

#endif

#endif