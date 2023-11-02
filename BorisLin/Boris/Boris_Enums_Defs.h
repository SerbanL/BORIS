#pragma once

#include "CompileFlags.h"

//All widely applicable enums and defines go here. Further smaller scope enums and defines are given where needed.

//when UpdateConfiguration method is called, pass a value from this enum to signal the reason it was called
enum UPDATECONFIG_ {

	////////////////////////////
	//GENERIC
	////////////////////////////

	//no reason given
	UPDATECONFIG_GENERIC,

	////////////////////////////
	//PROGRAM-WIDE SCOPE
	////////////////////////////

	//Brute force update : when this message is received all objects handling UpdateConfiguration method should update fully
	UPDATECONFIG_FORCEUPDATE,

	//Generic message issued by RepairObjectState method
	UPDATECONFIG_REPAIROBJECTSTATE,

	//Cuda state has changed
	UPDATECONFIG_SWITCHCUDASTATE,

	////////////////////////////
	//SUPERMESH
	////////////////////////////

	//A supermesh cellsize has changed
	UPDATECONFIG_SMESH_CELLSIZE,

	//a global field has been set/cleared
	UPDATECONFIG_SMESH_GLOBALFIELD,

	////////////////////////////
	//MESHES
	////////////////////////////

	//mesh shape has changed (not the rectangle but the shape inside the rectangle)
	UPDATECONFIG_MESHSHAPECHANGE,

	//The mesh has changed: cellsize, rectangle, or number of cells, or important mesh status changed
	UPDATECONFIG_MESHCHANGE,

	//a new mesh has been added (all meshes added through the AddMesh method in supermesh so that method would signal this)
	UPDATECONFIG_MESHADDED,

	//a mesh has been deleted (all meshes deleted through the DelMesh method in supermesh so that method would signal this)
	UPDATECONFIG_MESHDELETED,

	////////////////////////////
	//SPECIAL
	////////////////////////////

	//Change to PRNG settings (seed changed, which requires reinitialization of PRNG)
	UPDATECONFIG_PRNG,

	////////////////////////////
	//PARAMETERS
	////////////////////////////

	//Param value changed
	UPDATECONFIG_PARAMVALUECHANGED,

	//Magnetization length changed (e.g. Ms)
	UPDATECONFIG_PARAMVALUECHANGED_MLENGTH,

	//mesh param settings changed
	UPDATECONFIG_PARAMCHANGED,

	////////////////////////////
	//ODE SOLVER
	////////////////////////////

	//Equation or evaluation method or settings changed
	UPDATECONFIG_ODE_SOLVER,

	//Moving mesh algorithm settings changed
	UPDATECONFIG_ODE_MOVEMESH,

	////////////////////////////
	//MODULES
	////////////////////////////

	//A module was added
	UPDATECONFIG_MODULEADDED,

	//A module was deleted
	UPDATECONFIG_MODULEDELETED,

	//Module display (effective field and energy) settings changed
	UPDATECONFIG_MODULEDISPLAYCHANGED,

	////////////////////////////
	//SPECIFIC MODULES
	////////////////////////////

	//SDemag or Demag module convolution type or settings change
	UPDATECONFIG_DEMAG_CONVCHANGE,

	//Change in roughness module
	UPDATECONFIG_ROUGHNESS_CHANGE,

	//Transport module electrode changed
	UPDATECONFIG_TRANSPORT_ELECTRODE,

	//Transport module changed in an unnspecified way, but needs re-initialization
	UPDATECONFIG_TRANSPORT,

	//Heat solver temperature model type changed
	UPDATECONFIG_HEAT_MODELTYPE,

	//Elastodynamics solver settings changes (typically fixed or stress surfaces, but could also be change of crystal structure)
	UPDATECONFIG_MELASTIC,

	////////////////////////////
	//UpdateConfiguration_Values MESSAGES
	////////////////////////////

	//User constants changed for text equation objects
	UPDATECONFIG_TEQUATION_CONSTANTS,

	//Clear all text equations
	UPDATECONFIG_TEQUATION_CLEAR
};

namespace ucfg {

	//version without forcing flags
	template <typename Flag>
	bool __check_cfgflags(UPDATECONFIG_ cfgMessage, Flag flag)
	{
		if (cfgMessage == flag) return true;
		else return false;
	}

	//version without forcing flags
	template <typename Flag, typename ... Flags>
	bool __check_cfgflags(UPDATECONFIG_ cfgMessage, Flag flag, Flags... flags)
	{
		if (cfgMessage == flag) return true;
		else return __check_cfgflags(cfgMessage, flags...);
	}

	//version with forcing flags - use this
	template <typename Flag>
	bool check_cfgflags(UPDATECONFIG_ cfgMessage, Flag flag)
	{
		if (cfgMessage == UPDATECONFIG_FORCEUPDATE ||
			cfgMessage == UPDATECONFIG_SWITCHCUDASTATE ||
			cfgMessage == UPDATECONFIG_REPAIROBJECTSTATE ||
			cfgMessage == UPDATECONFIG_MESHADDED ||
			cfgMessage == UPDATECONFIG_MESHDELETED ||
			cfgMessage == UPDATECONFIG_MODULEADDED ||
			cfgMessage == UPDATECONFIG_MODULEDELETED) return true;

		if (cfgMessage == flag) return true;
		else return false;
	}

	//version with forcing flags - use this
	template <typename Flag, typename ... Flags>
	bool check_cfgflags(UPDATECONFIG_ cfgMessage, Flag flag, Flags... flags)
	{
		if (cfgMessage == UPDATECONFIG_FORCEUPDATE ||
			cfgMessage == UPDATECONFIG_SWITCHCUDASTATE ||
			cfgMessage == UPDATECONFIG_REPAIROBJECTSTATE ||
			cfgMessage == UPDATECONFIG_MESHADDED ||
			cfgMessage == UPDATECONFIG_MESHDELETED ||
			cfgMessage == UPDATECONFIG_MODULEADDED ||
			cfgMessage == UPDATECONFIG_MODULEDELETED) return true;

		if (cfgMessage == flag) return true;
		else return __check_cfgflags(cfgMessage, flags...);
	}
};

#define CONVERSIONPRECISION 6						//Precision when converting to/from strings

#define MAXSIMSPACE		2.0							//Maximum side length of simulation space (m)
#define MINMESHSPACE	5e-11						//Minimum mesh side length (m). Also used to snap mesh rectangles to this resolution.
#define MAXFIELD		1e10						//Maximum field strength (A/m)
#define MAXSTRESS		1e15						//Maximum mechanical stress (Pa)
#define MINODERELERROR		1e-12					//Minimum relative error for ode solver that can be entered
#define MAXODERELERROR		1e-2					//Maximum relative error for ode solver that can be entered
#define MINTIMESTEP		1e-18						//Minimum time step that can be entered (s)
#define MAXTIMESTEP		1e-6						//Maximum time step that can be entered (s)

//default clipping distance for dipole shifting algorithm
#define DIPOLESHIFTCLIP 0.5e-9
//maximum velocity allowed for dipole shifting algorithm
#define DIPOLEMAXVELOCITY 3e8

//when creating a mesh limit the number of cells. The mesh can exceed these values but user will have to adjust the cellsize manually.
#define MAXSTARTINGCELLS_X	2048
#define MAXSTARTINGCELLS_Y	2048
#define MAXSTARTINGCELLS_Z	512

//the default cellsize when creating a mesh (cubic) up to given number of maximum number of cells
#define DEFAULTCELLSIZE	5e-9

//the default atomistic cellsize when creating an atomistic mesh
#define DEFAULTATOMCELLSIZE 5e-10

//minimum and maximum damping values for fixed SOR damping algorithm
#define MINSORDAMPING	0.1
#define MAXSORDAMPING	2.0

//skyrmion definition settings for the skyrmion and skyrmionbloch commands
#define SKYRMION_RING_WIDTH 0.6
#define SKYRMION_TANH_CONST 2.0

//output buffer number of lines for saving data to disk: when end reached buffer content is written to file
#define DISKBUFFERLINES	100

//number of significant figures to use for saving data to file
#define SAVEDATAPRECISION	6

//Monte-Carlo Algorithm : minimum allowed cone angle
#define MONTECARLO_CONEANGLEDEG_MIN		1.0
//Monte-Carlo Algorithm : maximum allowed cone angle
#define MONTECARLO_CONEANGLEDEG_MAX		180.0
//Monte-Carlo Algorithm : change in cone angle per step
#define MONTECARLO_CONEANGLEDEG_DELTA	1.0
//Monte-Carlo Algorithm : target aceptance probability (vary cone angle to reach this)
#define MONTECARLO_TARGETACCEPTANCE		0.5
//If actual acceptance rate is within this tolerance close to target acceptance then don't adjust cone angle
#define MONTECARLO_ACCEPTANCETOLERANCE	0.1
//Try to perform reduction on acceptance rate only every given number of iterations
#define MONTECARLO_REDUCTIONITERS		100