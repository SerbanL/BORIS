#include "stdafx.h"
#include "Atom_iDMExchangeCUDA.h"
#include "DataDefs.h"

#if defined(MODULE_COMPILATION_IDMEXCHANGE) && ATOMISTIC == 1
#if COMPILECUDA == 1

#include "BorisLib.h"

#include "Atom_MeshCUDA.h"
#include "Atom_iDMExchange.h"

Atom_iDMExchangeCUDA::Atom_iDMExchangeCUDA(Atom_MeshCUDA* paMeshCUDA_, Atom_iDMExchange* pAtom_iDMExchange_) :
	ModulesCUDA()
{
	paMeshCUDA = paMeshCUDA_;
	pAtom_iDMExchange = pAtom_iDMExchange_;
}

Atom_iDMExchangeCUDA::~Atom_iDMExchangeCUDA()
{}

BError Atom_iDMExchangeCUDA::Initialize(void)
{
	BError error(CLASS_STR(Atom_iDMExchangeCUDA));

	//initialize cpu version also (couple to dipole setting)
	pAtom_iDMExchange->Initialize();

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		(cuReal3)paMeshCUDA->h, (cuRect)paMeshCUDA->meshRect, 
		(MOD_)paMeshCUDA->Get_ActualModule_Heff_Display() == MOD_IDMEXCHANGE || paMeshCUDA->IsOutputDataSet_withRect(DATA_E_EXCH),
		(MOD_)paMeshCUDA->Get_ActualModule_Heff_Display() == MOD_IDMEXCHANGE || paMeshCUDA->IsOutputDataSet_withRect(DATA_E_EXCH));
	if (!error)	initialized = true;

	return error;
}

BError Atom_iDMExchangeCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Atom_iDMExchangeCUDA));

	Uninitialize();

	return error;
}

//-------------------Torque methods

cuReal3 Atom_iDMExchangeCUDA::GetTorque(cuRect avRect)
{
	return CalculateTorque(paMeshCUDA->M1, avRect);
}

#endif

#endif

