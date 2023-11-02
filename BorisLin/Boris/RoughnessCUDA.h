#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_ROUGHNESS

#include "BorisCUDALib.h"
#include "ModulesCUDA.h"

class MeshCUDA;
class Roughness;

class RoughnessCUDA :
	public ModulesCUDA
{

private:

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	MeshCUDA * pMeshCUDA;

	//pointer to non-CUDA version of this module
	Roughness* pRough;

	//multiplicative functions used to obtain roughness field in the coarse mesh
	mcu_VEC(cuReal3) Fmul_rough;
	mcu_VEC(cuReal3) Fomul_rough;

private:

	void set_RoughnessCUDA_pointers(void);

public:

	RoughnessCUDA(MeshCUDA* pMeshCUDA_, Roughness* pRough_);
	~RoughnessCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);

	//-------------------
};

#else

class RoughnessCUDA
{
};

#endif

#endif



