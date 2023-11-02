#pragma once

#include "BorisLib.h"
#include "Modules.h"

class Mesh;

#ifdef MODULE_COMPILATION_ANITENS

//Anisotropy modules can only be used in a magnetic mesh

class Anisotropy_Tensorial :
	public Modules,
	public ProgramState<Anisotropy_Tensorial, std::tuple<>, std::tuple<>>
{
private:

	//pointer to mesh object holding this effective field module
	Mesh * pMesh;

public:

	Anisotropy_Tensorial(Mesh *pMesh_);
	~Anisotropy_Tensorial() {}

	//-------------------Implement ProgramState method

	void RepairObjectState(void) {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void);

	double UpdateField(void);

	//-------------------Energy methods

	//FM Mesh
	double Get_EnergyChange(int spin_index, DBL3 Mnew);

	//AFM mesh
	DBL2 Get_EnergyChange(int spin_index, DBL3 Mnew_A, DBL3 Mnew_B);

	//-------------------Torque methods

	DBL3 GetTorque(Rect& avRect);
};

#else

class Anisotropy_Tensorial :
	public Modules
{
private:

	//pointer to mesh object holding this effective field module
	Mesh * pMesh;

public:

	Anisotropy_Tensorial(Mesh *pMesh_) {}
	~Anisotropy_Tensorial() {}

	//-------------------Implement ProgramState method

	void RepairObjectState(void) {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) {}

	BError Initialize(void) { return BError(); }

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage) { return BError(); }
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void) { return BError(); }

	double UpdateField(void) { return 0.0; }
};


#endif

