#pragma once

#include "BorisLib.h"
#include "Modules.h"

#if COMPILECUDA == 1
#include "Atom_SurfExchangeCUDA.h"
#endif

class Atom_Mesh;
class Mesh;

#if defined(MODULE_COMPILATION_SURFEXCHANGE) && ATOMISTIC == 1

class Atom_SurfExchange :
	public Modules,
	public ProgramState<Atom_SurfExchange, std::tuple<>, std::tuple<>>
{

#if COMPILECUDA == 1
	friend Atom_SurfExchangeCUDA;
#endif

private:

	//pointer to mesh object holding this effective field module
	Atom_Mesh* paMesh;

	//------------------ SURFACE COUPLING Z STACKING

	//magnetic meshes in surface exchange coupling with the mesh holding this module, top and bottom (atomistic meshes)
	std::vector<Atom_Mesh*> paMesh_Bot, paMesh_Top;

	//magnetic meshes in surface exchange coupling with the mesh holding this module, top and bottom (micromagnetic meshes)
	std::vector<Mesh*> pMesh_Bot, pMesh_Top;

	//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

	//coupling in the "bulk" - i.e. coupling at surfaces of shapes inside overlapping meshes
	//e.g. FM particles embedded in an AFM matrix etc.
	std::vector<Atom_Mesh*> paMesh_Bulk;
	std::vector<Mesh*> pMeshFM_Bulk;
	std::vector<Mesh*> pMeshAFM_Bulk;
	//when surface exchange in the bulk is detected, a coupling mask will be allocated here and calculated
	//this has same dimensions as M, and contains for each cell a mesh index value (+1) for paMesh_Bulk, and pMesh_Bulk
	//if the cell is not surface exchange coupled in the bulk then entry of 0 is set
	//otherwise the index ranges as 1, ..., paMesh_Bulk.size(), paMesh_Bulk.size() + 1, ..., paMesh_Bulk.size() + pMesh_Bulk.size()
	//i.e. subtract 1 to get required index etc.
	//INT3 is stored : one int for each x, y, z directions.
	//each int stores in 2 bytes 2 mesh indexes, for + (first 2 bytes) and - (last 2 bytes) directions for each axis
	VEC<INT3> bulk_coupling_mask;

public:

	Atom_SurfExchange(Atom_Mesh *paMesh_);
	~Atom_SurfExchange();

	//-------------------Implement ProgramState method

	void RepairObjectState(void) {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void);

	double UpdateField(void);

	//-------------------Torque methods

	DBL3 GetTorque(Rect& avRect);

	//-------------------Energy methods

	//For simple cubic mesh spin_index coincides with index in M1
	double Get_EnergyChange(int spin_index, DBL3 Mnew);
};

#else

class Atom_SurfExchange :
	public Modules
{

private:

public:

	Atom_SurfExchange(Atom_Mesh *paMesh_) {}
	~Atom_SurfExchange() {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) {}

	BError Initialize(void) { return BError(); }

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage) { return BError(); }
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void) { return BError(); }

	double UpdateField(void) { return 0.0; }
};

#endif