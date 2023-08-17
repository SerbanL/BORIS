#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#include "BorisCUDALib.h"

#include "ErrorHandler.h"

#include "Atom_MeshParamsCUDA.h"

#include "ManagedAtom_DiffEqPolicy_CommonCUDA.h"

#include "DiffEq_CommonBaseCUDA.h"

class Atom_ODECommon;
class ODECommon_Base;

class Atom_ODECommonCUDA :
	public ODECommon_BaseCUDA
{
	friend Atom_ODECommon;
	friend ODECommon_Base;
	friend ManagedAtom_DiffEq_CommonCUDA;

private:

	//-----------------------------------CPU version pointer

	//pointer to CPU version
	static Atom_ODECommon *pODE;

	//-----------------------------------Managed DiffEq

	//ManagedDiffEq_CommonCUDA holds pointers to data in ODECommonCUDA in an object in gpu memory.
	//pass cuaDiffEq to a cuda kernel then all gpu data held here in cu_obj objects can be accessed in device code.
	//Initialize ManagedDiffEq_CommonCUDA with all the pointers you need then forget about it - no book-keeping required.
	static mcu_obj<ManagedAtom_DiffEq_CommonCUDA, ManagedAtom_DiffEqPolicy_CommonCUDA>* pcuaDiffEq;

protected:

	//-----------------------------------Equation

	//set equation identifier in GPU memory
	static mcu_val<int>* psetODE;

	//-----------------------------------Evaluation method modifiers

	static mcu_val<bool>* prenormalize;

	//-----------------------------------Steepest Descent Solver

	//quantities used to calculate Barzilai-Borwein stepsizes across multiple meshes
	//Accumulate values in these quantities, then obtain stepsizes as:
	//step1 = delta_m_sq / delta_m_dot_delta_G
	//step2 = delta_m_dot_delta_G / delta_G_sq
	static mcu_val<cuBReal>* pdelta_M_sq;
	static mcu_val<cuBReal>* pdelta_G_sq;
	static mcu_val<cuBReal>* pdelta_M_dot_delta_G;

private:

	//----------------------------------- SET-UP METHODS : Atom_DiffEq_CommonCUDA.cpp

	//Allocate memory for all static data; deletion only happens in the destructor, however allocation can also be triggered by UpdateConfiguration since the static data can be deleted by another instance which inherits same static data
	void AllocateStaticData(void);

	//----------------------------------- Auxiliary : Atom_DiffEq_CommonCUDA.cu, Atom_DiffEq_CommonCUDA.cpp

	//specific to SD solver
	void Zero_SD_Solver_BB_Values(void);

	void Get_SD_Solver_BB_Values(double* pdelta_M_sq_cpu, double* pdelta_G_sq_cpu, double* pdelta_M_dot_delta_G_cpu);

	//----------------------------------- GPU <-> CPU sync : Atom_DiffEq_CommonCUDA.cpp

	//set all cuda values here from their cpu values held in ODECommon
	void SyncODEValues(void);

public:

	Atom_ODECommonCUDA(void) {}
	Atom_ODECommonCUDA(Atom_ODECommon *pODE_);

	virtual ~Atom_ODECommonCUDA();

	//----------------------------------- Important Control Methods

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	//----------------------------------- GETTERS

	//get reference to stored managed cuda differential equation object (cuaDiffEq)
	mcu_obj<ManagedAtom_DiffEq_CommonCUDA, ManagedAtom_DiffEqPolicy_CommonCUDA>& Get_ManagedAtom_DiffEq_CommonCUDA(void) { return *pcuaDiffEq; }
};

#endif