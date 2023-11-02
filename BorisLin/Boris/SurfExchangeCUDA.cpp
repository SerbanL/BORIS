#include "stdafx.h"
#include "SurfExchangeCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SURFEXCHANGE

#include "SurfExchange.h"

#include "Mesh_Ferromagnetic.h"
#include "Mesh_FerromagneticCUDA.h"

#include "Atom_Mesh.h"
#include "Atom_MeshCUDA.h"

#include "DataDefs.h"

SurfExchangeCUDA::SurfExchangeCUDA(MeshCUDA* pMeshCUDA_, SurfExchange* pSurfExch_) : 
	ModulesCUDA(),
	pMeshFM_Bot(mGPU), pMeshFM_Top(mGPU),
	pMeshAFM_Bot(mGPU), pMeshAFM_Top(mGPU),
	pMeshAtom_Bot(mGPU), pMeshAtom_Top(mGPU),
	pMeshFM_Bulk(mGPU), pMeshAFM_Bulk(mGPU), paMesh_Bulk(mGPU),
	bulk_coupling_mask(mGPU)
{
	pMeshCUDA = pMeshCUDA_;
	pSurfExch = pSurfExch_;
}

SurfExchangeCUDA::~SurfExchangeCUDA()
{}

BError SurfExchangeCUDA::Initialize(void)
{
	BError error(CLASS_STR(SurfExchangeCUDA));

	if (!initialized) {

		//clear cu_arrs then rebuild them from information in SurfExchange module
		pMeshFM_Bot.clear();
		pMeshFM_Top.clear();
		pMeshAFM_Bot.clear();
		pMeshAFM_Top.clear();
		pMeshAtom_Bot.clear();
		pMeshAtom_Top.clear();

		pMeshFM_Bulk.clear();
		pMeshAFM_Bulk.clear();
		paMesh_Bulk.clear();

		//make sure information in SurfExchange module is up to date
		error = pSurfExch->Initialize();

		if (!error) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				//------------------ SURFACE COUPLING Z STACKING

				//FM and AFM meshes on the bottom
				for (int idx = 0; idx < pSurfExch->pMesh_Bot.size(); idx++) {

					if (pSurfExch->pMesh_Bot[idx]->GetMeshType() == MESH_FERROMAGNETIC) {

						pMeshFM_Bot.push_back(mGPU, pSurfExch->pMesh_Bot[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
					}
					else if (pSurfExch->pMesh_Bot[idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						pMeshAFM_Bot.push_back(mGPU, pSurfExch->pMesh_Bot[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
					}
				}

				//FM and AFM meshes on the top
				for (int idx = 0; idx < pSurfExch->pMesh_Top.size(); idx++) {

					if (pSurfExch->pMesh_Top[idx]->GetMeshType() == MESH_FERROMAGNETIC) {

						pMeshFM_Top.push_back(mGPU, pSurfExch->pMesh_Top[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
					}
					else if (pSurfExch->pMesh_Top[idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						pMeshAFM_Top.push_back(mGPU, pSurfExch->pMesh_Top[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
					}
				}

				//Atomistic meshes on the bottom
				for (int idx = 0; idx < pSurfExch->paMesh_Bot.size(); idx++) {

					pMeshAtom_Bot.push_back(mGPU, pSurfExch->paMesh_Bot[idx]->paMeshCUDA->cuaMesh.get_managed_object(mGPU));
				}

				//Atomistic meshes on the top
				for (int idx = 0; idx < pSurfExch->paMesh_Top.size(); idx++) {

					pMeshAtom_Top.push_back(mGPU, pSurfExch->paMesh_Top[idx]->paMeshCUDA->cuaMesh.get_managed_object(mGPU));
				}

				//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

				for (int idx = 0; idx < pSurfExch->pMeshFM_Bulk.size(); idx++) {

					pMeshFM_Bulk.push_back(mGPU, pSurfExch->pMeshFM_Bulk[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
				}

				for (int idx = 0; idx < pSurfExch->pMeshAFM_Bulk.size(); idx++) {

					pMeshAFM_Bulk.push_back(mGPU, pSurfExch->pMeshAFM_Bulk[idx]->pMeshCUDA->cuMesh.get_managed_object(mGPU));
				}

				for (int idx = 0; idx < pSurfExch->paMesh_Bulk.size(); idx++) {

					paMesh_Bulk.push_back(mGPU, pSurfExch->paMesh_Bulk[idx]->paMeshCUDA->cuaMesh.get_managed_object(mGPU));
				}
			}

			//copy calculated bulk_coupling_mask if needed
			if (pSurfExch->pMeshFM_Bulk.size() + pSurfExch->pMeshAFM_Bulk.size() + pSurfExch->paMesh_Bulk.size()) {

				if (!bulk_coupling_mask.set_from_cpuvec(pSurfExch->bulk_coupling_mask)) error(BERROR_OUTOFGPUMEMORY_CRIT);
			}
			else bulk_coupling_mask.clear();

			initialized = true;
		}
	}

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		(cuReal3)pMeshCUDA->h, (cuRect)pMeshCUDA->meshRect, 
		(MOD_)pMeshCUDA->Get_Module_Heff_Display() == MOD_SURFEXCHANGE || pMeshCUDA->IsOutputDataSet_withRect(DATA_E_SURFEXCH) || pMeshCUDA->IsOutputDataSet(DATA_T_SURFEXCH),
		(MOD_)pMeshCUDA->Get_Module_Energy_Display() == MOD_SURFEXCHANGE || pMeshCUDA->IsOutputDataSet_withRect(DATA_E_SURFEXCH));
	if (error) initialized = false;

	//need to set pointers in ManagedMeshCUDA for Monte Carlo usage
	if (initialized) set_SurfExchangeCUDA_pointers();

	return error;
}

BError SurfExchangeCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SurfExchangeCUDA));

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHCHANGE, UPDATECONFIG_MESHADDED, UPDATECONFIG_MESHDELETED, UPDATECONFIG_MESHSHAPECHANGE, UPDATECONFIG_MODULEADDED, UPDATECONFIG_MODULEDELETED)) {

		Uninitialize();
	}

	return error;
}

//-------------------Torque methods

cuReal3 SurfExchangeCUDA::GetTorque(cuRect avRect)
{
	return CalculateTorque(pMeshCUDA->M, avRect);
}

#endif

#endif

