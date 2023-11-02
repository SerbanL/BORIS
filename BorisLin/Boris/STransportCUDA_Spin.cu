#include "STransportCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "mcuVEC_cmbnd.cuh"

#include "CUDAError.h"

#include "STransportCUDA_GInterf_S.h"

void STransportCUDA::set_cmbnd_spin_transport_V(void)
{
	//calculate values at CMBND cells using boundary conditions
	for (int idx1 = 0; idx1 < (int)CMBNDcontacts.size(); idx1++) {

		for (int idx2 = 0; idx2 < (int)CMBNDcontacts[idx1].size(); idx2++) {

			int idx_sec = CMBNDcontacts[idx1][idx2].mesh_idx.i;
			int idx_pri = CMBNDcontacts[idx1][idx2].mesh_idx.j;

			pTransport[idx_pri]->exchange_all_halos_spin();
			pTransport[idx_sec]->exchange_all_halos_spin();

			//use continuity of Jc and V across interface unless the interface is N-F type (normal metal - ferromagnetic) and the spin mixing conductance is not zero (i.e. continuous method disabled).

			//Is it an N-F contact?
			if (((pTransport[idx_pri]->Get_STSolveType() == STSOLVE_FERROMAGNETIC || pTransport[idx_pri]->Get_STSolveType() == STSOLVE_FERROMAGNETIC_ATOM) && 
				(pTransport[idx_sec]->Get_STSolveType() == STSOLVE_NORMALMETAL || pTransport[idx_sec]->Get_STSolveType() == STSOLVE_TUNNELING)) ||
				((pTransport[idx_sec]->Get_STSolveType() == STSOLVE_FERROMAGNETIC || pTransport[idx_sec]->Get_STSolveType() == STSOLVE_FERROMAGNETIC_ATOM) && 
				(pTransport[idx_pri]->Get_STSolveType() == STSOLVE_NORMALMETAL || pTransport[idx_pri]->Get_STSolveType() == STSOLVE_TUNNELING))) {

				//Yes we have an N(T)-F contact. Is G interface enabled for this contact ? (remember top mesh sets G interface values)

				//if primary is top then we check GInterface in primary. If primary is bottom then we check GInterface in secondary as it must be top.
				if ((CMBNDcontacts[idx1][idx2].IsPrimaryTop() && pTransport[idx_pri]->GInterface_Enabled()) ||
					(!CMBNDcontacts[idx1][idx2].IsPrimaryTop() && pTransport[idx_sec]->GInterface_Enabled())) {
					
					//G interface method
					(*pV[idx_pri]).set_cmbnd_continuousflux(
						pV[idx_sec]->get_managed_mcuvec(),
						pTransport[idx_sec]->spin_V_cmbnd_funcs_sec,
						pTransport[idx_pri]->spin_V_cmbnd_funcs_pri,
						gInterf_V,
						CMBNDcontactsCUDA[idx1][idx2]);
						
					//next contact
					continue;
				}
			}
		
			//continuous interface method - the G interface method check above didn't pass so this is what we have left
			(*pV[idx_pri]).set_cmbnd_continuous(
				pV[idx_sec]->get_managed_mcuvec(),
				pTransport[idx_sec]->spin_V_cmbnd_funcs_sec,
				pTransport[idx_pri]->spin_V_cmbnd_funcs_pri,
				CMBNDcontactsCUDA[idx1][idx2]);
		}
	}
}

void STransportCUDA::set_cmbnd_spin_transport_S(void)
{
	//calculate values at CMBND cells using boundary conditions
	for (int idx1 = 0; idx1 < (int)CMBNDcontacts.size(); idx1++) {

		for (int idx2 = 0; idx2 < (int)CMBNDcontacts[idx1].size(); idx2++) {

			int idx_sec = CMBNDcontacts[idx1][idx2].mesh_idx.i;
			int idx_pri = CMBNDcontacts[idx1][idx2].mesh_idx.j;

			if (pTransport[idx_pri]->Get_STSolveType() == STSOLVE_NONE || pTransport[idx_sec]->Get_STSolveType() == STSOLVE_NONE) continue;

			pTransport[idx_pri]->exchange_all_halos_spin();
			pTransport[idx_sec]->exchange_all_halos_spin();

			//use continuity of Js and S across interface unless the interface is N-F type (normal metal - ferromagnetic) and the spin mixing conductance is not zero (i.e. continuous method disabled).

			//Is it an N-F contact?
			if (((pTransport[idx_pri]->Get_STSolveType() == STSOLVE_FERROMAGNETIC || pTransport[idx_pri]->Get_STSolveType() == STSOLVE_FERROMAGNETIC_ATOM) && 
				(pTransport[idx_sec]->Get_STSolveType() == STSOLVE_NORMALMETAL || pTransport[idx_sec]->Get_STSolveType() == STSOLVE_TUNNELING)) ||
				((pTransport[idx_sec]->Get_STSolveType() == STSOLVE_FERROMAGNETIC || pTransport[idx_sec]->Get_STSolveType() == STSOLVE_FERROMAGNETIC_ATOM) && 
				(pTransport[idx_pri]->Get_STSolveType() == STSOLVE_NORMALMETAL || pTransport[idx_pri]->Get_STSolveType() == STSOLVE_TUNNELING))) {
				
				//Yes we have an N(T)-F contact. Is G interface enabled for this contact ? (remember top mesh sets G interface values)
				
				//if primary is top then we check GInterface in primary. If primary is bottom then we check GInterface in secondary as it must be top.
				if ((CMBNDcontacts[idx1][idx2].IsPrimaryTop() && pTransport[idx_pri]->GInterface_Enabled()) ||
					(!CMBNDcontacts[idx1][idx2].IsPrimaryTop() && pTransport[idx_sec]->GInterface_Enabled())) {

					//G interface method
					
					if (pTransport[idx_pri]->Get_STSolveType() == STSOLVE_FERROMAGNETIC || pTransport[idx_pri]->Get_STSolveType() == STSOLVE_FERROMAGNETIC_ATOM) {

						//interface conductance method with F being the primary mesh
						(*pS[idx_pri]).set_cmbnd_discontinuous(
							pS[idx_sec]->get_managed_mcuvec(),
							pTransport[idx_sec]->spin_S_cmbnd_funcs_sec,
							pTransport[idx_pri]->spin_S_cmbnd_funcs_pri,
							gInterf_S_NF,
							CMBNDcontactsCUDA[idx1][idx2]);
					}
					else {
						
						//interface conductance method with N(T) being the primary mesh
						(*pS[idx_pri]).set_cmbnd_discontinuous(
							pS[idx_sec]->get_managed_mcuvec(),
							pTransport[idx_sec]->spin_S_cmbnd_funcs_sec,
							pTransport[idx_pri]->spin_S_cmbnd_funcs_pri,
							gInterf_S_FN,
							CMBNDcontactsCUDA[idx1][idx2]);
					}
					
					//next contact
					continue;
				}
			}
			
			//continuous interface method - the G interface method check above didn't pass so this is what we have left
			(*pS[idx_pri]).set_cmbnd_continuous(
				pS[idx_sec]->get_managed_mcuvec(),
				pTransport[idx_sec]->spin_S_cmbnd_funcs_sec,
				pTransport[idx_pri]->spin_S_cmbnd_funcs_pri,
				CMBNDcontactsCUDA[idx1][idx2]);
		}
	}
}

#endif

#endif