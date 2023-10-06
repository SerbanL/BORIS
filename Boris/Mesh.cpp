#include "stdafx.h"
#include "Mesh.h"
#include "SuperMesh.h"

Mesh::Mesh(MESH_ meshType, SuperMesh *pSMesh_) :
	MeshBase(meshType, pSMesh_),
	//MeshParams constructor after Meshbase, since they both have virtual inheritance from MeshParamsBase : we want MeshParams to control setting values in MeshParamsBase
	MeshParams(params_for_meshtype(meshType))
{
}

Mesh::~Mesh() 
{
	//delete all allocated Modules
	//This has to go here, not in MeshBase destructor, even though pMod is held there
	//The reason for this, some modules can access Mesh data when destructing, and MeshBase destructor is called after Mesh destructor
	//(so Mesh data no longer defined at that point resulting in undefined behaviour)
	clear_vector(pMod);

#if COMPILECUDA == 1
	//free cuda memory by deleting allocated pMeshBaseCUDA
	if (pMeshBaseCUDA) {
		
		//mark implementation of Mesh as destroyed so the CUDA mesh version doesn't attempt to use its data in destructor
		pMeshBaseCUDA->Holder_Mesh_Destroyed();

		delete pMeshBaseCUDA;
		pMeshBaseCUDA = nullptr;
		pMeshCUDA = nullptr;
	}
#endif
}

//calls pSMesh->UpdateConfiguration
BError Mesh::SuperMesh_UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	return pSMesh->UpdateConfiguration(cfgMessage);
}

//----------------------------------- IMPORTANT CONTROL METHODS

#if COMPILECUDA == 1
//check the shape_synchronization_lost flag when starting a simulation (called from SuperMesh::InitializeAllModulesCUDA)
BError Mesh::CheckSynchronization_on_Initialization(void)
{
	BError error(__FUNCTION__);

	if (shape_synchronization_lost) {

		bool success = true;

		if (pMeshCUDA) {

			//make sure all CPU quantities which could have changed shape at run-time are re-synchronized to GPU versions
			//at the moment this can only arise from the track shifting algorithm
			if (M.linear_size()) success &= pMeshCUDA->M.copy_to_cpuvec(M);
			if (M2.linear_size()) success &= pMeshCUDA->M2.copy_to_cpuvec(M2);

			if (!success) return error(BERROR_GPUERROR_CRIT);
		}

		shape_synchronization_lost = false;
	}

	return error;
}
#endif