#include "stdafx.h"
#include "Atom_viDMExchangeCUDA.h"
#include "DataDefs.h"

#if defined(MODULE_COMPILATION_VIDMEXCHANGE) && ATOMISTIC == 1
#if COMPILECUDA == 1

#include "BorisLib.h"

#include "Atom_MeshCUDA.h"
#include "Atom_viDMExchange.h"

Atom_viDMExchangeCUDA::Atom_viDMExchangeCUDA(Atom_MeshCUDA* paMeshCUDA_, Atom_viDMExchange* pAtom_viDMExchange_) :
	ModulesCUDA()
{
	paMeshCUDA = paMeshCUDA_;
	pAtom_viDMExchange = pAtom_viDMExchange_;
}

Atom_viDMExchangeCUDA::~Atom_viDMExchangeCUDA()
{}

BError Atom_viDMExchangeCUDA::Initialize(void)
{
	BError error(CLASS_STR(Atom_viDMExchangeCUDA));

	//initialize cpu version also (couple to dipole setting)
	pAtom_viDMExchange->Initialize();

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		(cuReal3)paMeshCUDA->h, (cuRect)paMeshCUDA->meshRect,
		(MOD_)paMeshCUDA->Get_ActualModule_Heff_Display() == MOD_VIDMEXCHANGE || paMeshCUDA->IsOutputDataSet_withRect(DATA_E_EXCH),
		(MOD_)paMeshCUDA->Get_ActualModule_Heff_Display() == MOD_VIDMEXCHANGE || paMeshCUDA->IsOutputDataSet_withRect(DATA_E_EXCH));
	if (!error)	initialized = true;

	return error;
}

BError Atom_viDMExchangeCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Atom_viDMExchangeCUDA));

	Uninitialize();

	return error;
}

//-------------------Torque methods

cuReal3 Atom_viDMExchangeCUDA::GetTorque(cuRect avRect)
{
	return CalculateTorque(paMeshCUDA->M1, avRect);
}

#endif

#endif