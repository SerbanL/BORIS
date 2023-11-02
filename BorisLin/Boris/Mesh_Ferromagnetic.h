#pragma once

#include "Mesh.h"

#include "DiffEqFM.h"

#if COMPILECUDA == 1
#include "Mesh_FerromagneticCUDA.h"
#endif

#ifdef MESH_COMPILATION_FERROMAGNETIC

#include "SkyrmionTrack.h"
#include "DWRunTimeFit.h"

#include "Exchange.h"
#include "DMExchange.h"
#include "iDMExchange.h"
#include "viDMExchange.h"
#include "SurfExchange.h"
#include "SurfExchange_AFM.h"
#include "Demag.h"
#include "Demag_N.h"
#include "SDemag_Demag.h"
#include "StrayField_Mesh.h"
#include "Zeeman.h"
#include "MOptical.h"
#include "Anisotropy.h"
#include "AnisotropyCubi.h"
#include "AnisotropyBiaxial.h"
#include "AnisotropyTensorial.h"
#include "MElastic.h"
#include "Transport.h"
#include "Heat.h"
#include "SOTField.h"
#include "STField.h"
#include "Roughness.h"

/////////////////////////////////////////////////////////////////////
//
//Ferromagnetic Material Mesh

class SuperMesh;

class FMesh :
	public Mesh,
	public ProgramState < FMesh,
	std::tuple<
	//Mesh members
	int, int, int,
	int, int, int, int, int, int,
	Rect, SZ3, DBL3, SZ3, DBL3, SZ3, DBL3, SZ3, DBL3, SZ3, DBL3, bool,
	VEC_VC<DBL3>,
	VEC_VC<double>, VEC_VC<DBL3>, VEC_VC<DBL3>, VEC_VC<double>,
	VEC_VC<double>, VEC_VC<double>,
	VEC_VC<DBL3>, VEC_VC<DBL3>, VEC_VC<DBL3>,
	vector_lut<Modules*>,
	bool, bool,
	unsigned,
	DBL3, DBL3, DBL3, double, std::vector<int>,
	//Members in this derived class
	bool, SkyrmionTrack, bool,
	double, double, bool, bool, bool, DBL3,
	//Material Parameters
	MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<DBL2, double>,
	MatP<double, double>, MatP<double, double>, MatP<DBL3, DBL3>, MatP<double, double>, MatP<double, double>,
	MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<DBL3, DBL3>, MatP<DBL3, DBL3>, MatP<DBL3, DBL3>,
	std::vector<DBL4>,
	MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>,
	MatP<double, double>,
	MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>,
	MatP<double, double>, MatP<double, double>, MatP<DBL2, double>, MatP<DBL2, double>, MatP<DBL3, DBL3>, MatP<double, double>, MatP<DBL2, double>, MatP<DBL2, double>,
	MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<DBL2, double>, MatP<DBL2, double>,
	MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>,
	double, TEquation<double>, double, MatP<double, double>, MatP<double, double>,
	MatP<double, double>, MatP<DBL2, double>, MatP<DBL2, double>, MatP<DBL2, double>, MatP<DBL2, double>, MatP<DBL2, double>, MatP<DBL2, double>, MatP<double, double>, MatP<double, double>, MatP<DBL3, double>, MatP<DBL3, double>, MatP<DBL3, double>, MatP<DBL3, double>, MatP<double, double>, MatP<double, double>,
	MatP<double, double>, MatP<double, double>,
	MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>, MatP<double, double>
	> ,
	//Module Implementations
	std::tuple<
	Demag_N, Demag, SDemag_Demag,
	Exch_6ngbr_Neu, DMExchange, iDMExchange, viDMExchange, SurfExchange,
	Zeeman, MOptical, MElastic, Roughness,
	Anisotropy_Uniaxial, Anisotropy_Cubic, Anisotropy_Biaxial, Anisotropy_Tensorial,
	Transport, Heat,
	SOTField, STField,
	StrayField_Mesh> >
{
#if COMPILECUDA == 1
	friend FMeshCUDA;
#endif

private:

	//The set ODE, associated with this ferromagnetic mesh (the ODE type and evaluation method is controlled from SuperMesh)
	DifferentialEquationFM meshODE;

	//is this mesh used to trigger mesh movement? i.e. the CheckMoveMesh method should only be used if this flag is set
	bool move_mesh_trigger = false;

	//object used to track one or more skyrmions in this mesh
	SkyrmionTrack skyShift;

	//domain wall run-time position and width fitting
	DWPosWidth dwPos;

	//direct exchange coupling to neighboring meshes?
	//If true this is applicable for this mesh only for cells at contacts with other ferromagnetic meshes 
	//i.e. if two distinct meshes with the same materials are in contact, setting this flag to true will make the simulation behave as if the two materials are in the same computational mesh (provided the demag field is computed on the supermesh).
	//this is precisely what it is intended for; if two dissimilar materials are in contact then you probably shouldn't be using this mechanism.
	bool exchange_couple_to_meshes = false;

private:

	//Take a Monte Carlo step in this mesh : these functions implement the actual algorithms
	void Iterate_MonteCarlo_Parallel_Classic(void);

public:

	//constructor taking only a SuperMesh pointer (SuperMesh is the owner) only needed for loading : all required values will be set by LoadObjectState method in ProgramState
	FMesh(SuperMesh *pSMesh_);

	FMesh(Rect meshRect_, DBL3 h_, SuperMesh *pSMesh_);

	~FMesh();

	//implement pure virtual method from ProgramState
	void RepairObjectState(void);

	//----------------------------------- INITIALIZATION

	//----------------------------------- IMPORTANT CONTROL METHODS

	//call when a configuration change has occurred - some objects might need to be updated accordingly
	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage);

	BError SwitchCUDAState(bool cudaState);

	//called at the start of each iteration
	void PrepareNewIteration(void);

#if COMPILECUDA == 1
	void PrepareNewIterationCUDA(void);
#endif

	//Take a Monte Carlo step in this mesh
	void Iterate_MonteCarlo(double acceptance_rate);

#if COMPILECUDA == 1
	//Take a Monte Carlo step in this mesh
	void Iterate_MonteCarloCUDA(double acceptance_rate);
#endif

	//Check if mesh needs to be moved (using the MoveMesh method) - return amount of movement required (i.e. parameter to use when calling MoveMesh).
	double CheckMoveMesh(void);

	//couple this mesh to touching dipoles by setting skip cells as required : used for domain wall moving mesh algorithms
	void CoupleToDipoles(bool status);

	//----------------------------------- ALGORITHMS

	//implement track shifting - called during PrepareNewIteration if this is a dormant mesh with track shifting configured (non-zero trackWindow_velocity and idTrackShiftMesh vector not empty)
	void Track_Shift_Algorithm(void);

	//setup track shifting algoithm for the holder mesh, with simulation window mesh, to be moved at set velocity and clipping during a simulation
	BError Setup_Track_Shifting(std::vector<int> sim_meshIds, DBL3 velocity, DBL3 clipping);

	//----------------------------------- MOVING MESH TRIGGER FLAG

	bool GetMoveMeshTrigger(void) { return move_mesh_trigger; }
	void SetMoveMeshTrigger(bool status) { move_mesh_trigger = status; }

	//----------------------------------- ODE METHODS IN (ANTI)FERROMAGNETIC MESH : Mesh_Ferromagnetic_ODEControl.cpp

	//get rate of change of magnetization (overloaded by Ferromagnetic meshes)
	DBL3 dMdt(int idx);

	//Save current magnetization in sM VECs (e.g. useful to reset dM / dt calculation)
	void SaveMagnetization(void);

	//return average dm/dt in the given avRect (relative rect). Here m is the direction vector.
	DBL3 Average_dmdt(Rect avRect);

	//return average m x dm/dt in the given avRect (relative rect). Here m is the direction vector.
	DBL3 Average_mxdmdt(Rect avRect);

	//----------------------------------- FERROMAGNETIC MESH QUANTITIES CONTROL : Mesh_Ferromagnetic_Control.cpp

	//this method is also used by the dipole mesh where it does something else - sets the dipole direction
	void SetMagAngle(double polar, double azim, Rect rectangle = Rect());

	//set magnetization angle only in given shape
	void SetMagAngle_Shape(double polar, double azim, std::vector<MeshShape> shapes);

	//Set magnetization angle in solid object only containing given relative position uniformly using polar coordinates
	void SetMagAngle_Object(double polar, double azim, DBL3 position);

	//Flower state magnetization
	void SetMagFlower(int direction, DBL3 centre, double radius, double thickness);

	//Onion state magnetization
	void SetMagOnion(int direction, DBL3 centre, double radius1, double radius2, double thickness);

	//Crosstie state magnetization
	void SetMagCrosstie(int direction, DBL3 centre, double radius, double thickness);

	//Invert magnetisation direction in given mesh (must be magnetic)
	void SetInvertedMag(bool x, bool y, bool z);

	//Mirror magnetisation in given axis (literal x, y, or z) in given mesh (must be magnetic)
	void SetMirroredMag(std::string axis);

	//Set random magentisation distribution in given mesh (must be magnetic)
	void SetRandomMag(int seed);
	void SetRandomXYMag(int seed);

	//set a domain wall with given width (metric units) at position within mesh (metric units). 
	//Longitudinal and transverse are magnetisation componets as: 1: x, 2: y, 3: z, 1: -x, 2: -y, 3: -z
	void SetMagDomainWall(int longitudinal, int transverse, double width, double position);

	//set Neel skyrmion with given orientation (core is up: 1, core is down: -1), chirality (1 for towards centre, -1 away from it) in given rectangle (relative to mesh), calculated in the x-y plane
	void SetSkyrmion(int orientation, int chirality, Rect skyrmion_rect);

	//set Bloch skyrmion with given chirality (outside is up: 1, outside is down: -1) in given rectangle (relative to mesh), calculated in the x-y plane
	void SetSkyrmionBloch(int orientation, int chirality, Rect skyrmion_rect);

	//set M from given data VEC (0 values mean empty points) -> stretch data to M dimensions if needed.
	void SetMagFromData(VEC<DBL3>& data, const Rect& dstRect = Rect());

	//----------------------------------- OVERLOAD MESH VIRTUAL METHODS

	//Curie temperature for ferromagnetic meshes. Calling this forces recalculation of affected material parameters temperature dependence - any custom dependence set will be overwritten.
	void SetCurieTemperature(double Tc, bool set_default_dependences);

	//atomic moment (as multiple of Bohr magneton) for ferromagnetic meshes. Calling this forces recalculation of affected material parameters temperature dependence - any custom dependence set will be overwritten.
	void SetAtomicMoment(DBL2 atomic_moment_ub);
	
	//get skyrmion shift for a skyrmion initially in the given rectangle (works only with data in data box or output data, not with ShowData)
	//the rectangle must use relative coordinates
	DBL2 Get_skyshift(Rect skyRect) 
	{ 
#if COMPILECUDA == 1
		if(pMeshCUDA) return skyShift.Get_skyshiftCUDA(pMeshCUDA->M, skyRect);
#endif
		return skyShift.Get_skyshift(M, skyRect); 
	}

	//get skyrmion shift for a skyrmion initially in the given rectangle (works only with data in output data, not with ShowData or with data box), as well as diameters along x and y directions.
	DBL4 Get_skypos_diameters(Rect skyRect)
	{ 
#if COMPILECUDA == 1
		if (pMeshCUDA) return skyShift.Get_skypos_diametersCUDA(pMeshCUDA->M, skyRect);
#endif
		return skyShift.Get_skypos_diameters(M, skyRect);
	}

	//set/get skypos tracker rect size diameter multiplier
	double Get_skypos_dmul(void) { return skyShift.Get_skypos_dmul(); }
	void Set_skypos_dmul(double dia_mul_) { skyShift.Set_skypos_dmul(dia_mul_); }

	//Fit domain wall along the x direction through centre of rectangle : fit the component which matches a tanh profile. Return centre position and width.
	DBL2 FitDomainWall_X(Rect rectangle);
	//Fit domain wall along the y direction through centre of rectangle : fit the component which matches a tanh profile. Return centre position and width.
	DBL2 FitDomainWall_Y(Rect rectangle);
	//Fit domain wall along the z direction through centre of rectangle : fit the component which matches a tanh profile. Return centre position and width.
	DBL2 FitDomainWall_Z(Rect rectangle);

	//compute magnitude histogram data
	//extract histogram between magnitudes min and max with given number of bins. if min max not given (set them to zero) then determine them first. 
	//output probabilities in histogram_p, corresponding to values set in histogram_x min, min + bin, ..., max, where bin = (max - min) / (num_bins - 1)
	//if macrocell_dims is not INT3(1) then first average in macrocells containing given number of individual mesh cells, then obtain histogram
	bool Get_Histogram(std::vector<double>& histogram_x, std::vector<double>& histogram_p, int num_bins, double& min, double& max, INT3 macrocell_dims);

	//As for Get_Histogram, but use thermal averaging in each macrocell
	bool Get_ThAvHistogram(std::vector<double>& histogram_x, std::vector<double>& histogram_p, int num_bins, double& min, double& max, INT3 macrocell_dims);

	//angular deviation histogram computed from ndir unit vector direction. If ndir not given (DBL3()), then angular deviation computed from average magnetization direction
	bool Get_AngHistogram(std::vector<double>& histogram_x, std::vector<double>& histogram_p, int num_bins, double& min, double& max, INT3 macrocell_dims, DBL3 ndir);

	//As for Get_AngHistogram, but use thermal averaging in each macrocell
	bool Get_ThAvAngHistogram(std::vector<double>& histogram_x, std::vector<double>& histogram_p, int num_bins, double& min, double& max, INT3 macrocell_dims, DBL3 ndir);

	//calculate thermodynamic average of magnetization
	DBL3 GetThermodynamicAverageMagnetization(Rect rectangle);

	//set/get exchange_couple_to_meshes status flag
	void SetMeshExchangeCoupling(bool status) { exchange_couple_to_meshes = status; }
	bool GetMeshExchangeCoupling(void) { return exchange_couple_to_meshes; }

	//----------------------------------- GETTERS

#if COMPILECUDA == 1
	//get reference to stored differential equation object (meshODE)
	DifferentialEquationFM& Get_DifferentialEquation(void) { return meshODE; }
#endif
};

#else

class FMesh :
	public Mesh
{

private:

	//The set ODE, associated with this ferromagnetic mesh (the ODE type and evaluation method is controlled from SuperMesh)
	DifferentialEquationFM meshODE;

public:

	//constructor taking only a SuperMesh pointer (SuperMesh is the owner) only needed for loading : all required values will be set by LoadObjectState method in ProgramState
	FMesh(SuperMesh *pSMesh_) :
		Mesh(MESH_FERROMAGNETIC, pSMesh_),
		meshODE(this)
	{}

	FMesh(Rect meshRect_, DBL3 h_, SuperMesh *pSMesh_) :
		Mesh(MESH_FERROMAGNETIC, pSMesh_),
		meshODE(this)
	{}

	~FMesh() {}

	//----------------------------------- INITIALIZATION

	//----------------------------------- IMPORTANT CONTROL METHODS

	//call when a configuration change has occurred - some objects might need to be updated accordingly
	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage) { return BError(); }
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError SwitchCUDAState(bool cudaState) { return BError(); }

	//called at the start of each iteration
	void PrepareNewIteration(void) {}

#if COMPILECUDA == 1
	void PrepareNewIterationCUDA(void) {}
#endif

	//----------------------------------- ALGORITHMS

	//implement track shifting - called during PrepareNewIteration if this is a dormant mesh with track shifting configured (non-zero trackWindow_velocity and idTrackShiftMesh vector not empty)
	void Track_Shift_Algorithm(void) {}

	//setup track shifting algoithm for the holder mesh, with simulation window mesh, to be moved at set velocity and clipping during a simulation
	BError Setup_Track_Shifting(std::vector<int> sim_meshIds, DBL3 velocity, DBL3 clipping) { return BError(); }

	//----------------------------------- GETTERS

#if COMPILECUDA == 1
	//get reference to stored differential equation object (meshODE)
	DifferentialEquationFM& Get_DifferentialEquation(void) { return meshODE; }
#endif
};

#endif