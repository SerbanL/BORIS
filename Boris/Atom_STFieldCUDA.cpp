#include "stdafx.h"
#include "Atom_STFieldCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_STFIELD) && ATOMISTIC == 1

#include "Atom_STField.h"

#include "Atom_Mesh.h"
#include "Atom_MeshCUDA.h"

#include "Mesh.h"
#include "MeshCUDA.h"

#include "MeshDefs.h"

Atom_STFieldCUDA::Atom_STFieldCUDA(Atom_MeshCUDA* paMeshCUDA_, Atom_STField* pSTField_) : 
	ModulesCUDA(),
	paMesh_Bot(mGPU),
	paMesh_Top(mGPU),
	pMeshFM_Bot(mGPU),
	pMeshFM_Top(mGPU)
{
	paMeshCUDA = paMeshCUDA_;
	pSTField = pSTField_;
}

Atom_STFieldCUDA::~Atom_STFieldCUDA()
{}

BError Atom_STFieldCUDA::Initialize(void)
{
	BError error(CLASS_STR(Atom_STFieldCUDA));

	//clear cu_arrs then rebuild them from information in STField module
	paMesh_Bot.clear();
	paMesh_Top.clear();
	pMeshFM_Bot.clear();
	pMeshFM_Top.clear();

	if (pSTField->paMesh->STp.get0() == DBL3()) {

		fixed_polarization = false;

		//make sure information in STField module is up to date
		error = pSTField->Initialize();

		if (!error) {

			for (int idx = 0; idx < pSTField->paMesh_Bot.size(); idx++) {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					paMesh_Bot.push_back(mGPU, pSTField->paMesh_Bot[idx]->paMeshCUDA->cuaMesh.get_managed_object(mGPU));
				}
			}

			for (int idx = 0; idx < pSTField->pMesh_Bot.size(); idx++) {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					pMeshFM_Bot.push_back(mGPU, pSTField->pMesh_Bot[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
				}
			}

			for (int idx = 0; idx < pSTField->paMesh_Top.size(); idx++) {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					paMesh_Top.push_back(mGPU, pSTField->paMesh_Top[idx]->paMeshCUDA->cuaMesh.get_managed_object(mGPU));
				}
			}

			for (int idx = 0; idx < pSTField->pMesh_Top.size(); idx++) {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					pMeshFM_Top.push_back(mGPU, pSTField->pMesh_Top[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
				}
			}

			initialized = true;
		}
	}
	else fixed_polarization = true;

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		(cuReal3)paMeshCUDA->h, (cuRect)paMeshCUDA->meshRect, 
		(MOD_)paMeshCUDA->Get_Module_Heff_Display() == MOD_STFIELD, 
		(MOD_)paMeshCUDA->Get_Module_Energy_Display() == MOD_STFIELD);
	if (!error)	initialized = true;

	//no energy density contribution here
	ZeroEnergy();

	return error;
}

BError Atom_STFieldCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Atom_STFieldCUDA));

	Uninitialize();

	return error;
}

#endif

#endif
