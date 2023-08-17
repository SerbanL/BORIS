#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_HEAT

#include <vector>
#include <tuple>

#include "BorisCUDALib.h"
#include "ModulesCUDA.h"

#include "HeatBaseCUDA.h"
#include "MeshParamsCUDA.h"

class SuperMesh;
class SHeat;

class SHeatCUDA :
	public ModulesCUDA
{

	friend SHeat;

private:

	//pointer to cpu version of this super-mesh module
	SHeat * pSHeat;

	SuperMesh* pSMesh;

	//----------------------

	//set temperature in individual meshes from this (if set and heat solver disabled)
	//globalTemp has a rect in absolute coordinates. If any mesh with heat solver enabled does not overlap with it, then respective cells set to base temperature.
	mcu_VEC_VC(cuBReal) globalTemp;

	//---------------------- CMBND data

	//CMBND contacts for all contacting heat conduction meshes - these are ordered by first vector index; for each mesh there could be multiple contacting meshes and these are ordered by second vector index
	//CMBNDInfo describes the contact between 2 meshes, allowing calculation of values at cmbnd cells based on continuity of a potential and flux
	std::vector< std::vector<mCMBNDInfoCUDA> > CMBNDcontactsCUDA;
	//...and we also need a cpu-memory version, even though we can access it using pSHeat - the problem is, we need the cpu data in .cu files where we cannot define SHeat (as nvcc will then attempt to compile BorisLib)
	std::vector< std::vector<CMBNDInfoCUDA> > CMBNDcontacts;

	//list of all transport modules in transport meshes (same ordering as first vector in CMBNDcontacts)
	std::vector<HeatBaseCUDA*> pHeat;

	//vector of pointers to all V - need this to set cmbnd flags (same ordering as first vector in CMBNDcontacts)
	std::vector<mcu_VEC_VC(cuBReal)*> pTemp;

private:

	//calculate and set values at composite media boundaries after all other cells have been computed and set
	void set_cmbnd_values(void);

public:

	SHeatCUDA(SuperMesh* pSMesh_, SHeat* pSHeat_);
	~SHeatCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);

	//-------------------Global Temperature

	//clear globalTemp
	void ClearGlobalTemperature(void);

	//shift rectangle of globalTemp if set
	void ShiftGlobalTemperatureRectangle(cuReal3 shift);

	BError LoadGlobalTemperature(void);
};

#else

class SHeatCUDA
{
};

#endif

#endif
