#pragma once

#include "BorisLib.h"
#include "Modules.h"

#if COMPILECUDA == 1
#include "SurfExchangeCUDA.h"
#endif

class Mesh;
class Atom_Mesh;

#ifdef MODULE_COMPILATION_SURFEXCHANGE

//This SurfExchange module is only used in a ferromagnetic mesh; there is a SurfExchange_AFM module used in a two-sublattice model mesh

class SurfExchange :
	public Modules,
	public ProgramState<SurfExchange, std::tuple<>, std::tuple<>>
{

#if COMPILECUDA == 1
	friend SurfExchangeCUDA;
#endif

private:

	//pointer to mesh object holding this effective field module
	Mesh* pMesh;

	//------------------ SURFACE COUPLING Z STACKING

	//magnetic meshes in surface exchange coupling with the mesh holding this module, top and bottom
	std::vector<Mesh*> pMesh_Bot, pMesh_Top;

	//atomic meshes in surface exchange coupling with the mesh holding this module, top and bottom
	std::vector<Atom_Mesh*> paMesh_Bot, paMesh_Top;

	//------------------ SURFACE COUPLING FOR OVERLAPPING MESHES

	//coupling in the "bulk" - i.e. coupling at surfaces of shapes inside overlapping meshes
	//e.g. FM particles embedded in an AFM matrix etc.
	std::vector<Mesh*> pMeshFM_Bulk;
	std::vector<Mesh*> pMeshAFM_Bulk;
	std::vector<Atom_Mesh*> paMesh_Bulk;
	//when surface exchange in the bulk is detected, a coupling mask will be allocated here and calculated
	//this has same dimensions as M, and contains for each cell a mesh index value (+1) for pMeshFM_Bulk, pMeshAFM_Bulk and paMesh_Bulk
	//if the cell is not surface exchange coupled in the bulk then entry of 0 is set
	//otherwise the index ranges as 1, ..., pMeshFM_Bulk.size(), pMeshFM_Bulk.size() + 1, ..., pMeshFM_Bulk.size() + pMeshAFM_Bulk.size(), pMeshFM_Bulk.size() + pMeshAFM_Bulk.size() + 1, ..., pMeshFM_Bulk.size() + pMeshAFM_Bulk.size() + paMesh_Bulk.size()
	//i.e. subtract 1 to get required index etc.
	//INT3 is stored : one int for each x, y, z directions.
	//each int stores in 2 bytes 2 mesh indexes, for + (first 2 bytes) and - (last 2 bytes) directions for each axis
	VEC<INT3> bulk_coupling_mask;

private:

	//Mh is for "here", and Mc is what we're trying to couple to. xrel_h and yrel_h are relative to here, and zrel_c relative to coupling mesh
	bool check_cell_coupling(VEC_VC<DBL3>& Mh, VEC_VC<DBL3>& Mc, double xrel_h, double yrel_h, double zrel_c)
	{
		//relative coordinates to read value from coupled mesh (the one we're coupling to here)
		DBL3 cell_rel_pos = DBL3(xrel_h + Mh.rect.s.x - Mc.rect.s.x, yrel_h + Mh.rect.s.y - Mc.rect.s.y, zrel_c);
		//can't couple to an empty cell
		if (!Mc.rect.contains(cell_rel_pos + Mc.rect.s) || Mc.is_empty(cell_rel_pos)) return false; 
		else return true;
	};

public:

	SurfExchange(Mesh *pMesh_);
	~SurfExchange();

	//-------------------Implement ProgramState method

	void RepairObjectState(void) {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void);

	double UpdateField(void);

	//-------------------Energy methods

	//FM mesh
	double Get_EnergyChange(int spin_index, DBL3 Mnew);

	//-------------------Torque methods

	DBL3 GetTorque(Rect& avRect);
};

#else

class SurfExchange :
	public Modules
{

private:

public:

	SurfExchange(Mesh *pMesh_) {}
	~SurfExchange() {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) {}

	BError Initialize(void) { return BError(); }

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage) { return BError(); }
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void) { return BError(); }

	double UpdateField(void) { return 0.0; }
};

#endif