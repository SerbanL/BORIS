#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SURFEXCHANGE

#include "BorisCUDALib.h"
#include "ModulesCUDA.h"

class MeshCUDA;
class SurfExchange;
class ManagedMeshCUDA;
class ManagedAtom_MeshCUDA;

class SurfExchangeCUDA :
	public ModulesCUDA
{

private:

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	MeshCUDA* pMeshCUDA;

	//pointer to cpu version of SurfExchange
	SurfExchange* pSurfExch;

	//------------------ SURFACE COUPLING Z STACKING

	//cu arrays with pointers to other meshes in surface exchange coupling with the mesh holding this module, top and bottom, (ferromagnetic)
	mcu_parr<ManagedMeshCUDA> pMeshFM_Bot;
	mcu_parr<ManagedMeshCUDA> pMeshFM_Top;

	//cu arrays with pointers to other meshes in surface exchange coupling with the mesh holding this module, top and bottom, (two-sublattice model meshes)
	mcu_parr<ManagedMeshCUDA> pMeshAFM_Bot;
	mcu_parr<ManagedMeshCUDA> pMeshAFM_Top;

	//cu arrays with pointers to other meshes in surface exchange coupling with the mesh holding this module, top and bottom, (atomistic)
	mcu_parr<ManagedAtom_MeshCUDA> pMeshAtom_Bot;
	mcu_parr<ManagedAtom_MeshCUDA> pMeshAtom_Top;

	//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

	//coupling in the "bulk" - i.e. coupling at surfaces of shapes inside overlapping meshes
	//e.g. FM particles embedded in an AFM matrix etc.
	mcu_parr<ManagedMeshCUDA> pMeshFM_Bulk;
	mcu_parr<ManagedMeshCUDA> pMeshAFM_Bulk;
	mcu_parr<ManagedAtom_MeshCUDA> paMesh_Bulk;
	//when surface exchange in the bulk is detected, a coupling mask will be allocated here and calculated
	//this has same dimensions as M, and contains for each cell a mesh index value (+1) for pMesh_Bulk and paMesh_Bulk
	//if the cell is not surface exchange coupled in the bulk then entry of 0 is set
	//otherwise the index ranges as 1, 2, .., pMesh_Bulk.size(), pMesh_Bulk.size() + 1, pMesh_Bulk.size() + 2, ..., pMesh_Bulk.size() + paMesh_Bulk.size()
	//i.e. subtract 1 to get required index etc.
	//INT3 is stored : one int for each x, y, z directions.
	//each int stores in 2 bytes 2 mesh indexes, for + (first 2 bytes) and - (last 2 bytes) directions for each axis
	mcu_VEC(cuINT3) bulk_coupling_mask;

private:

	//Set pointers in ManagedMeshCUDA so we can access them in device code. This is used by MonteCarlo algorithm.
	void set_SurfExchangeCUDA_pointers(void);

public:

	SurfExchangeCUDA(MeshCUDA* pMeshCUDA_, SurfExchange* pSurfExch_);
	~SurfExchangeCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);

	//-------------------Torque methods

	cuReal3 GetTorque(cuRect avRect);
};

#else

class SurfExchangeCUDA
{
};

#endif

#endif
