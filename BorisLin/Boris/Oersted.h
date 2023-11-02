#pragma once

#include "BorisLib.h"
#include "Modules.h"

class Mesh;
class FMesh;
class SuperMesh;

#ifdef MODULE_COMPILATION_OERSTED

#include "Convolution.h"
#include "OerstedKernel.h"

#if COMPILECUDA == 1
#include "OerstedCUDA.h"
#endif

class Oersted : 
	public Modules,
	public Convolution<Oersted, OerstedKernel>,
	public ProgramState<Oersted, std::tuple<>, std::tuple<>>
{

#if COMPILECUDA == 1
	friend OerstedCUDA;
#endif

private:

	SuperMesh * pSMesh;

	//super-mesh current density values used for computing Oersted field on the super-mesh
	VEC<DBL3> sm_Vals;

	//don't need to compute Oe field every iteration, only when a significant change in Jc occurs; but do need to compute it initially.
	bool oefield_computed = false;

public:

	Oersted(SuperMesh *pSMesh_);
	~Oersted() {}

	//-------------------Methods associated with saving/loading simulations

	void RepairObjectState(void) {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void);

	double UpdateField(void);

	//-------------------Getters

	VEC<DBL3>& GetOerstedField(void) { return sm_Vals; }

#if COMPILECUDA == 1
	mcu_VEC(cuReal3)& GetOerstedFieldCUDA(void) { return dynamic_cast<OerstedCUDA*>(pModuleCUDA)->GetOerstedField(); }
#endif
};

#else

class Oersted :
	public Modules
{

private:

	//super-mesh current density values used for computing Oersted field on the super-mesh
	VEC<DBL3> sm_Vals;

#if COMPILECUDA == 1
	mcu_VEC(cuReal3) sm_Vals_CUDA;
#endif

public:

	Oersted(SuperMesh *pSMesh_) 
#if COMPILECUDA == 1
		: sm_Vals_CUDA(mGPU)
#endif
	{}
	~Oersted() {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) {}

	BError Initialize(void) { return BError(); }

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage) { return BError(); }
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void) { return BError(); }

	double UpdateField(void) { return 0.0; }

	//-------------------Getters

	VEC<DBL3>& GetOerstedField(void) { return sm_Vals; }

#if COMPILECUDA == 1
	mcu_VEC(cuReal3)& GetOerstedFieldCUDA(void) { return sm_Vals_CUDA; }
#endif
};

#endif
