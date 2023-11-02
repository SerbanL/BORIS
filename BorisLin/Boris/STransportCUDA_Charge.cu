#include "STransportCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "mcuVEC_cmbnd.cuh"

void STransportCUDA::set_cmbnd_charge_transport(void)
{
	//calculate values at CMBND cells using boundary conditions
	for (int idx1 = 0; idx1 < (int)CMBNDcontacts.size(); idx1++) {

		for (int idx2 = 0; idx2 < (int)CMBNDcontacts[idx1].size(); idx2++) {

			int idx_sec = CMBNDcontacts[idx1][idx2].mesh_idx.i;
			int idx_pri = CMBNDcontacts[idx1][idx2].mesh_idx.j;
			
			pTransport[idx_pri]->exchange_all_halos_charge();
			pTransport[idx_sec]->exchange_all_halos_charge();
			
			(*pV[idx_pri]).set_cmbnd_continuous(
				pV[idx_sec]->get_managed_mcuvec(),
				pTransport[idx_sec]->charge_V_cmbnd_funcs_sec,
				pTransport[idx_pri]->charge_V_cmbnd_funcs_pri,
				CMBNDcontactsCUDA[idx1][idx2]);
		}
	}
}

#endif

#endif