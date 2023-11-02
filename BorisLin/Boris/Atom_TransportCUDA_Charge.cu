#include "Atom_TransportCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_TRANSPORT) && ATOMISTIC == 1

#include "Atom_MeshCUDA.h"

#include "mcuVEC_solve.cuh"
#include "cuVEC_VC_nprops.cuh"
#include "mcuVEC_halo.cuh"

//-------------------Calculation Methods

void Atom_TransportCUDA::IterateChargeSolver_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value)
{
	//Note, TransportCUDA_V_Funcs covers both thermoelectric and no thermoelectric effect cases (but for thermoelectric effect must use the NNeu Poisson solver as nonhomogeneous Neumann boundary conditions are required for V)

	//don't need to exchange halos on V since this is done by the SOR algorithm
	//however need to exchange halos on all quantities for which differential operators are required
	paMeshCUDA->elC.exchange_halos();

	if (!is_thermoelectric_mesh) {

		//no thermoelectric effect
		paMeshCUDA->V.IteratePoisson_SOR(poisson_V, damping, max_error, max_value);
	}
	else {

		paMeshCUDA->Temp.exchange_halos();

		//include thermoelectric effect
		paMeshCUDA->V.IteratePoisson_NNeu_SOR(poisson_V, damping, max_error, max_value);
	}
}

#endif

#endif