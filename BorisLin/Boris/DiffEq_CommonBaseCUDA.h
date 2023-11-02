#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#include "BorisCUDALib.h"

#include "ErrorHandler.h"

#include "mGPUConfig.h"

class ODECommon_Base;

class ODECommon_BaseCUDA
{

private:

	//-----------------------------------CPU version pointer

	//pointer to CPU version
	static ODECommon_Base *pODEBase;

protected:

	//-----------------------------------Primary Data

	static mcu_val<cuBReal>* ptime;
	static mcu_val<cuBReal>* pstagetime;

	//-----------------------------------Time step

	//these need to be pointers, not cu_obj directly : we only want to make the cuda objects when ODECommonCUDA is made (cuda switched on), not at the start of the program - what if cuda not available on the system?
	static mcu_val<cuBReal>* pdT;
	static mcu_val<cuBReal>* pdT_last;

	//-----------------------------------mxh and dmdt

	static mcu_val<cuBReal>* pmxh;
	static mcu_val<cuReal3>* pmxh_av;
	static mcu_val<size_t>* pavpoints;

	static mcu_val<cuBReal>* pdmdt;
	static mcu_val<cuReal3>* pdmdt_av;
	static mcu_val<size_t>* pavpoints2;

	//----------------------------------Adaptive time step control

	static mcu_val<cuBReal>* plte;

	//-----------------------------------Special evaluation values

	static mcu_val<bool>* palternator;

	//-----------------------------------Special Properties

	static mcu_val<bool>* psolve_spin_current;

protected:

	//----------------------------------- SET-UP METHODS : DiffEq_CommonBaseCUDA.cpp

	//Allocate memory for all static data; deletion only happens in the destructor, however allocation can also be triggered by UpdateConfiguration since the static data can be deleted by another instance which inherits same static data
	void AllocateStaticData(void);

	//----------------------------------- Auxiliary : DiffEq_CommonBaseCUDA.cu
	
	//zero all main reduction values : mxh, dmdt, lte
	void Zero_reduction_values(void);
	void Zero_mxh_lte_values(void);
	void Zero_dmdt_lte_values(void);
	void Zero_lte_value(void);

	void mxhav_to_mxh(void);
	void dmdtav_to_dmdt(void);

	//----------------------------------- GPU <-> CPU sync : DiffEq_CommonBaseCUDA.cpp

	//set all cuda values here from their cpu values held in ODECommon
	void SyncODEValues(void);

	//set specific cuda values (used often)
	void Sync_time(void);
	void Sync_dT(void);
	void Sync_dT_last(void);
	void Sync_alternator(void);

public:

	ODECommon_BaseCUDA(void) {}
	ODECommon_BaseCUDA(ODECommon_Base *pODEBase_);

	virtual ~ODECommon_BaseCUDA();

	//---------------------------------------- GET METHODS

	cuBReal Get_mxh(void) { return pmxh->to_cpu_max(); }
	cuBReal Get_dmdt(void) { return pdmdt->to_cpu_max(); }
	cuBReal Get_lte(void) { return plte->to_cpu_max(); }
};

#endif
