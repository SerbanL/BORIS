#pragma once

#include "BorisLib.h"
#include "Modules.h"

#include "ZeemanBase.h"

class Mesh;

#ifdef MODULE_COMPILATION_ZEEMAN

//Zeeman module can only be used in a magnetic mesh

class SuperMesh;

class Zeeman : 
	public Modules,
	public ZeemanBase,
	public ProgramState<Zeeman, std::tuple<DBL3, TEquation<double, double, double, double>, VEC<DBL3>>, std::tuple<>>
{

#if COMPILECUDA == 1
	friend class ZeemanCUDA;
#endif

private:

	//pointer to mesh object holding this effective field module
	Mesh *pMesh;

	//pointer to supermesh
	SuperMesh* pSMesh;

private:

	//Update TEquation object with user constants values
	void UpdateTEquationUserConstants(bool makeCuda = true);

	//setup globalField transfer
	BError InitializeGlobalField(void);

public:

	Zeeman(Mesh *pMesh_);
	~Zeeman();

	//-------------------Implement ProgramState method

	void RepairObjectState(void) {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage);

	BError MakeCUDAModule(void);

	double UpdateField(void);

	//-------------------Energy methods

	double Get_EnergyChange(int spin_index, DBL3 Mnew);

	//AFM mesh
	DBL2 Get_EnergyChange(int spin_index, DBL3 Mnew_A, DBL3 Mnew_B);

	//-------------------

	void SetField(DBL3 Hxyz);
	
	//Get currently set field : if a field equation is set then evaluate it at the centre of the mesh
	DBL3 GetField(void);

	BError SetFieldEquation(std::string equation_string, int step);

	BError SetFieldVEC_FromOVF2(std::string fileName);

	BError SetFieldVEC_FromVEC(VEC<DBL3>& Hext);

#if COMPILECUDA == 1
	BError SetFieldVEC_FromVEC_CUDA(mcu_VEC(cuReal3)& Hext);
#endif

	//if base temperature changes we need to adjust Tb in H_equation if it's used.
	void SetBaseTemperature(double Temperature);

	//Set globalField from SMesh::globalField (without mesh transfer, simply read values)
	void SetGlobalField(void);

	//-------------------Torque methods

	DBL3 GetTorque(Rect& avRect);
};

#else

class Zeeman :
	public Modules,
	public ZeemanBase
{

private:

private:

	//Update TEquation object with user constants values
	void UpdateTEquationUserConstants(bool makeCuda = true) {}

	//setup globalField transfer
	BError InitializeGlobalField(void) { return BError(); }

public:

	Zeeman(Mesh *pMesh_) {}
	~Zeeman() {}

	//-------------------Implement ProgramState method

	void RepairObjectState(void) {}

	//-------------------Abstract base class method implementations

	void Uninitialize(void) {}

	BError Initialize(void) { return BError(); }

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage) { return BError(); }
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	BError MakeCUDAModule(void) { return BError(); }

	double UpdateField(void) { return 0.0; }

	//-------------------

	void SetField(DBL3 Hxyz) {}
	DBL3 GetField(void) { return DBL3(); }

	BError SetFieldEquation(std::string equation_string, int step) { return BError(); }

	BError SetFieldVEC_FromOVF2(std::string fileName) { return BError(); }

	BError SetFieldVEC_FromVEC(VEC<DBL3>& Hext);

#if COMPILECUDA == 1
	BError SetFieldVEC_FromVEC_CUDA(mcu_VEC(cuReal3)& Hext);
#endif

	//if base temperature changes we need to adjust Tb in H_equation if it's used.
	void SetBaseTemperature(double Temperature) {}

	//Set globalField from SMesh::globalField (without mesh transfer, simply read values)
	void SetGlobalField(void) {}
};

#endif