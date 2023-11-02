#include "stdafx.h"
#include "Atom_Mesh_BCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC

#include "SuperMesh.h"

//----------------------------------- IMPORTANT CONTROL METHODS

//called at the start of each iteration
void Atom_Mesh_BCC::PrepareNewIteration(void)
{
	if (Is_Dormant()) Track_Shift_Algorithm();
	else {

		if (!IsModuleSet(MOD_ZEEMAN)) Heff1.set(DBL3());
	}
}

#if COMPILECUDA == 1
void Atom_Mesh_BCC::PrepareNewIterationCUDA(void)
{
	if (Is_Dormant()) Track_Shift_Algorithm();
	else {

		if (paMeshCUDA && !IsModuleSet(MOD_ZEEMAN)) paMeshCUDA->Heff1.set(cuReal3());
	}
}
#endif

//----------------------------------- ALGORITHMS

//setup track shifting algoithm for the holder mesh, with simulation window mesh, to be moved at set velocity and clipping during a simulation
BError Atom_Mesh_BCC::Setup_Track_Shifting(std::vector<int> sim_meshIds, DBL3 velocity, DBL3 clipping)
{
	BError error(__FUNCTION__);

	//track shifting not implemented for atomistic meshes

	return error;
}

//implement track shifting - called during PrepareNewIteration if this is a dormant mesh with track shifting configured (non-zero trackWindow_velocity and idTrackShiftMesh vector not empty)
void Atom_Mesh_BCC::Track_Shift_Algorithm(void)
{
	if (!idTrackShiftMesh.size() || trackWindow_velocity == DBL3() || !pSMesh->CurrentTimeStepSolved()) return;

	//track shifting not implemented for atomistic meshes
}

#endif