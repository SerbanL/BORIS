#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_ZEEMAN

#include "BorisCUDALib.h"
#include "ModulesCUDA.h"

class Zeeman;
class MeshCUDA;

template <typename VType> class VEC;

class ZeemanCUDA :
	public ModulesCUDA
{

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	MeshCUDA* pMeshCUDA;

	//pointer to Zeeman holder module
	Zeeman* pZeeman;

	//Applied field
	mcu_val<cuReal3> Ha;

	//Applied field using user equation, thus allowing simultaneous spatial (x, y, z), stage time (t); base temperature (Tb) and stage step (Ss) are introduced as user constants.
	//A number of constants are always present : mesh dimensions in m (Lx, Ly, Lz)
	mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal> H_equation;

	//Applied field but as a cuVEC (e.g. loaded from file), with same resolution as M.
	mcu_VEC(cuReal3) Havec;

	//global field as obtained in this mesh
	//When a global field is set in supermesh, then globalField VEC is initialized here with mesh transfered values
	mcu_VEC(cuReal3) globalField;

private:

	void set_ZeemanCUDA_pointers(void);

	//setup globalField transfer
	BError InitializeGlobalField(void);

public:

	ZeemanCUDA(MeshCUDA* pMeshCUDA_, Zeeman* pZeeman_);
	~ZeemanCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage);

	void UpdateField(void);

	//-------------------

	void SetField(cuReal3 Hxyz);

	BError SetFieldEquation(const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec);

	BError SetFieldVEC(VEC<DBL3>& Havec_cpu);

	BError SetFieldVEC_FromcuVEC(mcu_VEC(cuReal3)& Hext);

	//Set globalField from SMeshCUDA::globalField (without mesh transfer, simply read values)
	void SetGlobalField(mcu_VEC(cuReal3)& SMesh_globalField);

	//-------------------Torque methods

	cuReal3 GetTorque(cuRect avRect);
};

#else

class ZeemanCUDA
{
};

#endif

#endif
