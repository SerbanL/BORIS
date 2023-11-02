#include "TransportCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "mcuVEC_solve.cuh"
#include "cuVEC_VC_nprops.cuh"
#include "mcuVEC_halo.cuh"

#include "MeshCUDA.h"

//-------------------Calculation Methods

void TransportCUDA::IterateChargeSolver_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value)
{
	//Note, TransportCUDA_V_Funcs covers both thermoelectric and no thermoelectric effect cases (but for thermoelectric effect must use the NNeu Poisson solver as nonhomogeneous Neumann boundary conditions are required for V)
	
	//don't need to exchange halos on V since this is done by the SOR algorithm
	//however need to exchange halos on all quantities for which differential operators are required
	pMeshCUDA->elC.exchange_halos();

	if (!is_thermoelectric_mesh) {

		//no thermoelectric effect
		pMeshCUDA->V.IteratePoisson_SOR(poisson_V, damping, max_error, max_value);
	}
	else {
		
		pMeshCUDA->Temp.exchange_halos();

		//include thermoelectric effect
		pMeshCUDA->V.IteratePoisson_NNeu_SOR(poisson_V, damping, max_error, max_value);
	}
}

#endif

#endif