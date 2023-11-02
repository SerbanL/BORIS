#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_SURFEXCHANGE) && ATOMISTIC == 1

#include "BorisCUDALib.h"
#include "ModulesCUDA.h"

class Atom_SurfExchange;

class Atom_MeshCUDA;
class ManagedAtom_MeshCUDA;
class ManagedMeshCUDA;

class Atom_SurfExchangeCUDA :
	public ModulesCUDA
{

private:

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	Atom_MeshCUDA* paMeshCUDA;

	//pointer to cpu version of SurfExchange
	Atom_SurfExchange* paSurfExch;

	//------------------ SURFACE COUPLING Z STACKING

	//cu arrays with pointers to other atomistic meshes in surface exchange coupling with the mesh holding this module, top and bottom
	mcu_parr<ManagedAtom_MeshCUDA> paMesh_Bot;
	mcu_parr<ManagedAtom_MeshCUDA> paMesh_Top;

	//cu arrays with pointers to micromagnetic ferromagnetic meshes in surface exchange coupling with the mesh holding this module, top and bottom
	mcu_parr<ManagedMeshCUDA> pMeshFM_Bot;
	mcu_parr<ManagedMeshCUDA> pMeshFM_Top;

	//cu arrays with pointers to micromagnetic antiferromagnetic meshes in surface exchange coupling with the mesh holding this module, top and bottom
	mcu_parr<ManagedMeshCUDA> pMeshAFM_Bot;
	mcu_parr<ManagedMeshCUDA> pMeshAFM_Top;

	//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

	//coupling in the "bulk" - i.e. coupling at surfaces of shapes inside overlapping meshes
	//e.g. FM particles embedded in an AFM matrix etc.
	mcu_parr<ManagedAtom_MeshCUDA> paMesh_Bulk;
	mcu_parr<ManagedMeshCUDA> pMeshFM_Bulk;
	mcu_parr<ManagedMeshCUDA> pMeshAFM_Bulk;
	//when surface exchange in the bulk is detected, a coupling mask will be allocated here and calculated
	//this has same dimensions as M, and contains for each cell a mesh index value (+1) for paMesh_Bulk, and pMesh_Bulk
	//if the cell is not surface exchange coupled in the bulk then entry of 0 is set
	//otherwise the index ranges as 1, ..., paMesh_Bulk.size(), paMesh_Bulk.size() + 1, ..., paMesh_Bulk.size() + pMesh_Bulk.size()
	//i.e. subtract 1 to get required index etc.
	//INT3 is stored : one int for each x, y, z directions.
	//each int stores in 2 bytes 2 mesh indexes, for + (first 2 bytes) and - (last 2 bytes) directions for each axis
	mcu_VEC(cuINT3) bulk_coupling_mask;

private:

	//Set pointers in ManagedAtom_MeshCUDA so we can access them in device code. This is used by MonteCarlo algorithm.
	void set_Atom_SurfExchangeCUDA_pointers(void);

public:

	Atom_SurfExchangeCUDA(Atom_MeshCUDA* paMeshCUDA_, Atom_SurfExchange* paSurfExch_);
	~Atom_SurfExchangeCUDA();

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

class Atom_SurfExchangeCUDA
{
};

#endif

#endif