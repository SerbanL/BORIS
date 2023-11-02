#include "SHeatCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_HEAT

#include "MeshCUDA.h"

#include "mcuVEC_cmbnd.cuh"

//calculate and set values at composite media boundaries after all other cells have been computed and set
void SHeatCUDA::set_cmbnd_values(void)
{
	//calculate values at CMBND cells using boundary conditions
	for (int idx1 = 0; idx1 < (int)CMBNDcontacts.size(); idx1++) {

		for (int idx2 = 0; idx2 < (int)CMBNDcontacts[idx1].size(); idx2++) {

			int idx_sec = CMBNDcontacts[idx1][idx2].mesh_idx.i;
			int idx_pri = CMBNDcontacts[idx1][idx2].mesh_idx.j;
			
			(*pTemp[idx_pri]).set_cmbnd_continuous(
				pTemp[idx_sec]->get_managed_mcuvec(), 
				pHeat[idx_sec]->temp_cmbnd_funcs_sec, 
				pHeat[idx_pri]->temp_cmbnd_funcs_pri, 
				CMBNDcontactsCUDA[idx1][idx2]);
		}
	}
}

#endif

#endif