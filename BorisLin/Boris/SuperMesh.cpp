#include "stdafx.h"
#include "SuperMesh.h"

SuperMesh::SuperMesh(void) :
	ProgramStateNames(this,
		{
			VINFO(displayedPhysicalQuantity), VINFO(vec3rep), 
			VINFO(n_fm), VINFO(h_fm), VINFO(sMeshRect_fm), VINFO(n_e), VINFO(h_e), VINFO(sMeshRect_e),
			VINFO(globalField), VINFO(globalField_rect), VINFO(globalField_h), VINFO(globalField_velocity), VINFO(globalField_shift_clip), VINFO(globalField_shift_debt), VINFO(globalField_last_time),
			VINFO(odeSolver), VINFO(atom_odeSolver),
			VINFO(pMesh),
			VINFO(pSMod),
			VINFO(activeMeshName), VINFO(superMeshHandle),
			VINFO(scale_rects), VINFO(coupled_dipoles), VINFO(dwpos_component),
			VINFO(kernel_initialize_on_gpu),
			VINFO(computefields_if_MC), VINFO(cone_angle_minmax)
		}, 
		{
			//Mesh implementations
			IINFO(FMesh), IINFO(DipoleMesh), IINFO(MetalMesh), IINFO(InsulatorMesh), IINFO(AFMesh),
			IINFO(Atom_Mesh_Cubic),
			IINFO(Atom_Mesh_BCC),
			//Super-mesh modules implementations (for pSMod)
			IINFO(SDemag), IINFO(StrayField), IINFO(STransport), IINFO(Oersted), IINFO(SHeat)
		})
{
	//set pointers in ODECommon_Base SBC
	odeSolver.set_pointers(odeSolver, atom_odeSolver);

	//default ODE settings
	SetODE(ODE_LLG, EVAL_RKF45);

	//default atomistic ODE settings
	SetAtomisticODE(ODE_LLG, EVAL_RKF45);

	AddMesh(activeMeshName, MESH_FERROMAGNETIC, Rect(DBL3(80e-9, 80e-9, 10e-9)));
}

SuperMesh::~SuperMesh() 
{
	//clean-up meshes
	clear_vector(pMesh);
	clear_vector(pSMod);
}

BError SuperMesh::Error_On_Create(void)
{ 
	//go through meshes (and their modules)
	for (int idx = 0; idx < (int)pMesh.size(); idx++) {

		if(!error_on_create) error_on_create = pMesh[idx]->Error_On_Create();
	}

	//go through super-mesh modules
	for (int idx = 0; idx < (int)pSMod.size(); idx++) {

		if (!error_on_create) error_on_create = pSMod[idx]->Error_On_Create();
	}

	BError return_error = error_on_create;

	//clear error_on_create after reading it - the caller will restore program state
	error_on_create.clear();

	return return_error;
}