#pragma once

#include "BorisLib.h"
#include "Modules.h"

#ifdef MODULE_COMPILATION_DEMAG

class Mesh;

#include "DemagBase.h"
#include "EvalSpeedup.h"

#include "Convolution.h"
#include "DemagKernel.h"

#if COMPILECUDA == 1
class DemagMCUDA;
#endif

class Demag : 
	public Modules,
	public DemagBase,
	public Convolution<Demag, DemagKernel>,
	public ProgramState<Demag, std::tuple<INT3>, std::tuple<>>,
	public EvalSpeedup
{

#if COMPILECUDA == 1
	friend DemagMCUDA;
#endif

private:

	//pointer to mesh object holding this effective field module
	Mesh *pMesh;

public:

	Demag(Mesh *pMesh_);
	~Demag();

	//-------------------Implement ProgramState method

	void RepairObjectState(void) {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void);

	double UpdateField(void);

	//-------------------Setters

	//Set PBC
	BError Set_PBC(INT3 demag_pbc_images_);

	//-------------------Energy methods

	//FM mesh
	double Get_EnergyChange(int spin_index, DBL3 Mnew);

	//AFM mesh
	DBL2 Get_EnergyChange(int spin_index, DBL3 Mnew_A, DBL3 Mnew_B);
};

#else

class Demag :
	public Modules
{

private:

	//pointer to mesh object holding this effective field module
	Mesh * pMesh;

public:

	Demag(Mesh *pMesh_) {}
	~Demag() {}

	//-------------------Implement ProgramState method

	void RepairObjectState(void) {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) {}

	BError Initialize(void) { return BError(); }

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage) { return BError(); }
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void) { return BError(); }

	double UpdateField(void) { return 0.0; }

	//-------------------Setters

	//Set PBC
	BError Set_PBC(INT3 demag_pbc_images_) { return BError(); }
};

#endif