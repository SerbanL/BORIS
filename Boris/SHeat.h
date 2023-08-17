#pragma once

#include "BorisLib.h"
#include "Modules.h"

class SuperMesh;
class HeatBase;

#ifdef MODULE_COMPILATION_HEAT

#if COMPILECUDA == 1
#include "SHeatCUDA.h"
#endif

class SHeat :
	public Modules,
	public ProgramState<SHeat, std::tuple<double, VEC_VC<double>, DBL3, DBL3, DBL3, double>, std::tuple<>>
{

#if COMPILECUDA == 1
	friend SHeatCUDA;
#endif

private:

	//pointer to supermesh
	SuperMesh* pSMesh;

	//----------------------

	//set temperature in individual meshes from this (if set and heat solver disabled)
	//globalTemp has a rect in absolute coordinates. If any mesh with heat solver enabled does not overlap with it, then respective cells set to base temperature.
	VEC_VC<double> globalTemp;

	//during a simulation it's possible to shift globalTemp rectangle with fixed velocity (m/s)
	DBL3 globalTemp_velocity = DBL3();
	//when shifting due to non-zero velocity, if displacement is below the cliping distance (globalTemp_shift_clip), then add it to the shift_debt
	//when globalTemp_shift_debt exceeds clipping distance, then perform shift and reduce debt. This is to limit excessive shifting by small factors, since for non-zero velocity shifting is performed every iteration.
	DBL3 globalTemp_shift_clip = DBL3();
	DBL3 globalTemp_shift_debt = DBL3();
	double globalTemp_last_time = 0.0;

	//---------------------- CMBND data

	//CMBND contacts for all contacting transport meshes - these are ordered by first vector index; for each mesh there could be multiple contacting meshes and these are ordered by second vector index
	//CMBNDInfo describes the contact between 2 meshes, allowing calculation of values at cmbnd cells based on continuity of a potential and flux (temperature and heat flux)
	std::vector< std::vector<CMBNDInfo> > CMBNDcontacts;

	//list of all Heat modules in meshes (same ordering as first vector in CMBNDcontacts)
	std::vector<HeatBase*> pHeat;

	//vector of pointers to all Temp - need this to set cmbnd flags (same ordering as first vector in CMBNDcontacts)
	std::vector<VEC_VC<double>*> pTemp;

	//----------------------

	//time step for the heat equation - if in a magnetic mesh must always be smaller or equal to dT (the magnetization equation time-step)
	double heat_dT = 0.2e-12;

	//save the last magnetic dT used: when advancing the heat equation this is the time we need to advance by. 
	//Update magnetic_dT after each heat equation advance (in case an adaptive time-step method is used for the magnetic part).
	double magnetic_dT;

private:

	//calculate and set values at composite media boundaries after all other cells have been computed and set
	void set_cmbnd_values(void);

public:

	SHeat(SuperMesh *pSMesh_);
	~SHeat() {}

	//-------------------Implement ProgramState method

	void RepairObjectState(void) {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void);

	double UpdateField(void);

	//-------------------Getters

	double get_heat_dT(void) { return heat_dT; }

	//-------------------Setters

	void set_heat_dT(double dT) { heat_dT = dT; }

	void set_globalTemp_velocity(DBL3 velocity, DBL3 clipping) { globalTemp_velocity = velocity; globalTemp_shift_clip = clipping; }
	DBL3 get_globalTemp_velocity(void) { return globalTemp_velocity; }
	DBL3 get_globalTemp_clipping(void) { return globalTemp_shift_clip; }

	//-------------------Global Temperature

	//clear globalTemp
	void ClearGlobalTemperature(void);
	
	//shift rectangle of globalTemp if set
	void ShiftGlobalTemperatureRectangle(DBL3 shift);

	BError LoadGlobalTemperature(VEC<double>& globalTemp_);
	Rect GetGlobalTemperatureRect(void) { return globalTemp.rect; }
};

#else

class SHeat :
	public Modules
{

private:

private:

public:

	SHeat(SuperMesh *pSMesh_) {}
	~SHeat() {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) {}

	BError Initialize(void) { return BError(); }

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage) { return BError(); }
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void) { return BError(); }

	double UpdateField(void) { return 0.0; }

	//-------------------Getters

	double get_heat_dT(void) { return 0.0; }

	//-------------------Setters

	void set_heat_dT(double dT) {}

	void set_globalTemp_velocity(DBL3 velocity, DBL3 clipping) { }
	DBL3 get_globalTemp_velocity(void) { return DBL3(); }
	DBL3 get_globalTemp_clipping(void) { return DBL3(); }

	//-------------------Global Temperature

	//clear globalTemp
	void ClearGlobalTemperature(void) {}

	//shift rectangle of globalTemp if set
	void ShiftGlobalTemperatureRectangle(DBL3 shift) {}

	//typically used to load values into globalTemp (e.g. from ovf2 file)
	BError LoadGlobalTemperature(VEC<double>& globalTemp_) { return BError(); }
	Rect GetGlobalTemperatureRect(void) { return Rect(); }

};

#endif

