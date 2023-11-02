#pragma once

#include <string>

//maximum allowed temperature (K) for scaling array
#define MAX_TEMPERATURE 3000.0

//enum for all mesh material parameters. Add new parameters at the end to keep compatibility with older simulation files.

enum PARAM_ {

	PARAM_ALL = -1,

	//Magnetic : FM
	PARAM_GREL = 0, PARAM_GDAMPING = 1, PARAM_MS = 2, PARAM_DEMAGXY = 3,
	//Magnetic : AFM
	PARAM_GREL_AFM = 39, PARAM_GDAMPING_AFM = 40, PARAM_MS_AFM = 41,
	//Magnetic : Atomistic
	PARAM_ATOM_DAMPING = 62, PARAM_ATOM_MUS = 63,

	//Exchange : FM
	PARAM_A = 4, PARAM_D = 5, PARAM_J1 = 6, PARAM_J2 = 7,
	//Exchange : AFM
	PARAM_A_AFM = 42, PARAM_A_AFH = 54, PARAM_A_AFNH = 43, PARAM_D_AFM = 44, PARAM_DMI_DH = 80, PARAM_DMI_DH_DIR = 81,
	//Exchange : Atomistic
	PARAM_ATOM_J = 64, PARAM_ATOM_D = 65, PARAM_ATOM_JS = 78, PARAM_ATOM_JS2 = 84,
	//Exchange : special
	PARAM_DMI_DIR = 79,

	//Anisotropy : FM
	PARAM_K1 = 8, PARAM_K2 = 9, PARAM_K3 = 75, PARAM_EA1 = 10, PARAM_EA2 = 11, PARAM_EA3 = 72,
	//Anisotropy : AFM
	PARAM_K1_AFM = 59, PARAM_K2_AFM = 60, PARAM_K3_AFM = 76,
	//Anisotropy : Atomistic
	PARAM_ATOM_K1 = 66, PARAM_ATOM_K2 = 74, PARAM_ATOM_K3 = 77, PARAM_ATOM_EA1 = 67, PARAM_ATOM_EA2 = 68, PARAM_ATOM_EA3 = 73,

	//Susceptibilities
	PARAM_SUSREL = 12, PARAM_SUSPREL = 13, PARAM_SUSREL_AFM = 55,

	//Transport : general
	PARAM_ELC = 14, PARAM_AMR = 15, PARAM_TAMR = 93, PARAM_P = 16, PARAM_BETA = 17, PARAM_FLSOT = 33,

	//Transport : other spin torque coefficients
	PARAM_STQ = 69, PARAM_STA = 70, PARAM_STP = 71,
	PARAM_FLSOT2 = 86, PARAM_STQ2 = 87, PARAM_STA2 = 88,

	//Transport : drift-diffusion
	PARAM_DE = 18, PARAM_BETAD = 19, PARAM_SHA = 20, PARAM_ISHA = 21, PARAM_LSF = 22, PARAM_LEX = 23, PARAM_LPH = 24, PARAM_GI = 25, PARAM_GMIX = 26, PARAM_NDENSITY = 47,

	//Transport : drift-diffusion model spin torque efficiencies 
	PARAM_TSEFF = 27, PARAM_TSIEFF = 28, PARAM_PUMPEFF = 29, PARAM_CPUMP_EFF = 45, PARAM_THE_EFF = 46,

	//Transport - special
	PARAM_RATMR_P = 82, PARAM_RATMR_AP = 83,

	//Transport and Heat combination
	PARAM_SEEBECK = 89, PARAM_JOULE_EFF = 90,

	//Heat
	PARAM_THERMCOND = 30, PARAM_DENSITY = 31, PARAM_SHC = 32, PARAM_SHC_E = 52, PARAM_G_E = 53, PARAM_T = 37, PARAM_Q = 38,

	//Others
	PARAM_HA = 34, PARAM_HMO = 61, PARAM_RESERVED1 = 51,

	//LLB-specific
	PARAM_TC = 35, PARAM_MUB = 36, PARAM_MUB_AFM = 58, PARAM_AFTAU = 56, PARAM_AFTAUCROSS = 57,

	//Mechanical
	PARAM_MECOEFF = 48, PARAM_MMECOEFF = 95,
	PARAM_MECOEFF2 = 99, PARAM_MMECOEFF2 = 100,
	PARAM_MECOEFF3 = 101, PARAM_MMECOEFF3 = 102,

	PARAM_YOUNGSMOD = 49, PARAM_POISSONRATIO = 50,
	PARAM_STIFFC_CUBIC = 91, PARAM_STIFFC_2 = 96, PARAM_STIFFC_3 = 97, PARAM_STIFFC_S = 98,
	PARAM_MDAMPING = 92,

	//Thermoelasticity
	PARAM_THERMEL = 94,

	//Stochasticity special
	PARAM_S_EFF = 85

}; //Current maximum : 102

//classification of parameter
enum PARAMTYPE_ { PARAMTYPE_NONE, PARAMTYPE_MAGNETIC, PARAMTYPE_ELECTRIC, PARAMTYPE_THERMAL, PARAMTYPE_MECHANICAL };

//enum for material parameter temperature dependence type
enum MATPTDEP_ { MATPTDEP_NONE = 0, MATPTDEP_ARRAY, MATPTDEP_EQUATION };

//enum for spatial variation types. Add new parameters at the end to keep compatibility with older simulation files.
enum MATPVAR_ {
	MATPVAR_NONE = 0,
	MATPVAR_MASK,
	MATPVAR_RANDOM, MATPVAR_JAGGED,
	MATPVAR_ABLPOL, MATPVAR_ABLTANH, MATPVAR_ABLEXP,
	MATPVAR_DEFECTS, MATPVAR_FAULTS,
	MATPVAR_VORONOI2D, MATPVAR_VORONOI3D, MATPVAR_VORONOIBND2D, MATPVAR_VORONOIBND3D, MATPVAR_VORONOIROT2D, MATPVAR_VORONOIROT3D,
	MATPVAR_UVORONOI2D, MATPVAR_UVORONOI3D, MATPVAR_UVORONOIBND2D, MATPVAR_UVORONOIBND3D, MATPVAR_UVORONOIROT2D, MATPVAR_UVORONOIROT3D,
	MATPVAR_EQUATION,
	MATPVAR_OVF2,
	MATPVAR_SHAPE
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//MeshParamDescriptor used for console display / control of mesh material parameters
//

struct MeshParamDescriptor {

	//the unit used when converting from a std::string containing units to a numerical value and conversely
	std::string unit;

	PARAMTYPE_ paramType = PARAMTYPE_NONE;

	//display or hide this parameter?
	//e.g. we may enable a parameter for a mesh type, but may want to hide it so it doesn't appear in the usual lists of parameters (params, paramstemp, paramsvar commands).
	bool hidden;

	MeshParamDescriptor(PARAMTYPE_ paramType_, std::string unit_ = "", bool hidden_ = false) :
		unit(unit_), paramType(paramType_), hidden(hidden_)
	{}

	//what parameter is this?
	PARAMTYPE_ get_type(void) { return paramType; }
};
