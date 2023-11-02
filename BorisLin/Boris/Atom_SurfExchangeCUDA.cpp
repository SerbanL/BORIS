#include "stdafx.h"
#include "Atom_SurfExchangeCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_SURFEXCHANGE) && ATOMISTIC == 1

#include "Atom_SurfExchange.h"

#include "Atom_Mesh.h"
#include "Atom_MeshCUDA.h"

#include "Mesh.h"
#include "MeshCUDA.h"

#include "DataDefs.h"

Atom_SurfExchangeCUDA::Atom_SurfExchangeCUDA(Atom_MeshCUDA* paMeshCUDA_, Atom_SurfExchange* paSurfExch_) : 
	ModulesCUDA(),
	paMesh_Bot(mGPU), paMesh_Top(mGPU),
	pMeshFM_Bot(mGPU), pMeshFM_Top(mGPU),
	pMeshAFM_Bot(mGPU), pMeshAFM_Top(mGPU),
	paMesh_Bulk(mGPU), pMeshFM_Bulk(mGPU), pMeshAFM_Bulk(mGPU),
	bulk_coupling_mask(mGPU)
{
	paMeshCUDA = paMeshCUDA_;
	paSurfExch = paSurfExch_;
}

Atom_SurfExchangeCUDA::~Atom_SurfExchangeCUDA()
{}

BError Atom_SurfExchangeCUDA::Initialize(void)
{
	BError error(CLASS_STR(Atom_SurfExchangeCUDA));

	if (!initialized) {

		//clear cu_arrs then rebuild them from information in SurfExchange module
		paMesh_Bot.clear();
		paMesh_Top.clear();
		pMeshFM_Bot.clear();
		pMeshFM_Top.clear();
		pMeshAFM_Bot.clear();
		pMeshAFM_Top.clear();

		paMesh_Bulk.clear();
		pMeshFM_Bulk.clear();
		pMeshAFM_Bulk.clear();

		//make sure information in SurfExchange module is up to date
		error = paSurfExch->Initialize();

		if (!error) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				//------------------ SURFACE COUPLING Z STACKING

				//Atomistic meshes on the bottom
				for (int idx = 0; idx < paSurfExch->paMesh_Bot.size(); idx++) {

					paMesh_Bot.push_back(mGPU, paSurfExch->paMesh_Bot[idx]->paMeshCUDA->cuaMesh.get_managed_object(mGPU));
				}

				//FM and AFM meshes on the bottom
				for (int idx = 0; idx < paSurfExch->pMesh_Bot.size(); idx++) {

					if (paSurfExch->pMesh_Bot[idx]->GetMeshType() == MESH_FERROMAGNETIC) {

						pMeshFM_Bot.push_back(mGPU, paSurfExch->pMesh_Bot[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
					}
					else if (paSurfExch->pMesh_Bot[idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						pMeshAFM_Bot.push_back(mGPU, paSurfExch->pMesh_Bot[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
					}
				}

				//Atomistic meshes on the top
				for (int idx = 0; idx < paSurfExch->paMesh_Top.size(); idx++) {

					paMesh_Top.push_back(mGPU, paSurfExch->paMesh_Top[idx]->paMeshCUDA->cuaMesh.get_managed_object(mGPU));
				}

				//FM and AFM meshes on the top
				for (int idx = 0; idx < paSurfExch->pMesh_Top.size(); idx++) {

					if (paSurfExch->pMesh_Top[idx]->GetMeshType() == MESH_FERROMAGNETIC) {

						pMeshFM_Top.push_back(mGPU, paSurfExch->pMesh_Top[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
					}
					else if (paSurfExch->pMesh_Top[idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						pMeshAFM_Top.push_back(mGPU, paSurfExch->pMesh_Top[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
					}
				}

				//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

				for (int idx = 0; idx < paSurfExch->paMesh_Bulk.size(); idx++) {

					paMesh_Bulk.push_back(mGPU, paSurfExch->paMesh_Bulk[idx]->paMeshCUDA->cuaMesh.get_managed_object(mGPU));
				}

				for (int idx = 0; idx < paSurfExch->pMeshFM_Bulk.size(); idx++) {

					pMeshFM_Bulk.push_back(mGPU, paSurfExch->pMeshFM_Bulk[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
				}

				for (int idx = 0; idx < paSurfExch->pMeshAFM_Bulk.size(); idx++) {

					pMeshAFM_Bulk.push_back(mGPU, paSurfExch->pMeshAFM_Bulk[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
				}
			}

			//copy calculated bulk_coupling_mask if needed
			if (paSurfExch->pMeshFM_Bulk.size() + paSurfExch->pMeshAFM_Bulk.size() + paSurfExch->paMesh_Bulk.size()) {

				if (!bulk_coupling_mask.set_from_cpuvec(paSurfExch->bulk_coupling_mask)) error(BERROR_OUTOFGPUMEMORY_CRIT);
			}
			else bulk_coupling_mask.clear();

			initialized = true;
		}
	}

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		(cuReal3)paMeshCUDA->h, (cuRect)paMeshCUDA->meshRect,
		(MOD_)paMeshCUDA->Get_Module_Heff_Display() == MOD_SURFEXCHANGE || paMeshCUDA->IsOutputDataSet_withRect(DATA_E_SURFEXCH) || paMeshCUDA->IsOutputDataSet(DATA_T_SURFEXCH),
		(MOD_)paMeshCUDA->Get_Module_Energy_Display() == MOD_SURFEXCHANGE || paMeshCUDA->IsOutputDataSet_withRect(DATA_E_SURFEXCH));
	if (error) initialized = false;

	if (initialized) set_Atom_SurfExchangeCUDA_pointers();

	return error;
}

BError Atom_SurfExchangeCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Atom_SurfExchangeCUDA));

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHCHANGE, UPDATECONFIG_MESHADDED, UPDATECONFIG_MESHDELETED, UPDATECONFIG_MESHSHAPECHANGE, UPDATECONFIG_MODULEADDED, UPDATECONFIG_MODULEDELETED)) {

		Uninitialize();
	}

	return error;
}

//-------------------Torque methods

cuReal3 Atom_SurfExchangeCUDA::GetTorque(cuRect avRect)
{
	return CalculateTorque(paMeshCUDA->M1, avRect);
}

#endif

#endif

