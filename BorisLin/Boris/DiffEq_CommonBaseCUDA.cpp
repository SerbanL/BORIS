#include "stdafx.h"
#include "DiffEq_CommonBaseCUDA.h"
#include "DiffEq_CommonBase.h"

#if COMPILECUDA == 1

//-----------------------------------CPU version pointer

ODECommon_Base* ODECommon_BaseCUDA::pODEBase = nullptr;

//-----------------------------------Primary Data

mcu_val<cuBReal>* ODECommon_BaseCUDA::ptime = nullptr;
mcu_val<cuBReal>* ODECommon_BaseCUDA::pstagetime = nullptr;

//-----------------------------------Time step

mcu_val<cuBReal>* ODECommon_BaseCUDA::pdT = nullptr;
mcu_val<cuBReal>* ODECommon_BaseCUDA::pdT_last = nullptr;

//-----------------------------------mxh and dmdt

mcu_val<cuBReal>* ODECommon_BaseCUDA::pmxh = nullptr;
mcu_val<cuReal3>* ODECommon_BaseCUDA::pmxh_av = nullptr;
mcu_val<size_t>* ODECommon_BaseCUDA::pavpoints = nullptr;

mcu_val<cuBReal>* ODECommon_BaseCUDA::pdmdt = nullptr;
mcu_val<cuReal3>* ODECommon_BaseCUDA::pdmdt_av = nullptr;
mcu_val<size_t>* ODECommon_BaseCUDA::pavpoints2 = nullptr;

//----------------------------------Adaptive time step control

mcu_val<cuBReal>* ODECommon_BaseCUDA::plte = nullptr;

//-----------------------------------Special evaluation values

mcu_val<bool>* ODECommon_BaseCUDA::palternator = nullptr;

//-----------------------------------Special Properties

mcu_val<bool>* ODECommon_BaseCUDA::psolve_spin_current = nullptr;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ODECommon_BaseCUDA::ODECommon_BaseCUDA(ODECommon_Base *pODEBase_)
{
	if (!pODEBase) pODEBase = pODEBase_;

	AllocateStaticData();
}

ODECommon_BaseCUDA::~ODECommon_BaseCUDA()
{
	if (pODEBase) {

		//delete these to prevent memory leaks
		//Note since all these are static, they will be deleted in all meshes holding a ODECommon_BaseCUDA
		//Thus if you delete a mesh, you'll need to remake these and also set pointers in cuDiffEq : do it in UpdateConfiguration
		//Here we are only deleting memory held in this base class

		if (ptime) {

			pODEBase->time = ptime->to_cpu();
			delete ptime;
			ptime = nullptr;
		}

		if (pstagetime) {

			pODEBase->stagetime = pstagetime->to_cpu();
			delete pstagetime;
			pstagetime = nullptr;
		}

		if (pdT) {

			pODEBase->dT = pdT->to_cpu();
			delete pdT;
			pdT = nullptr;
		}

		if (pdT_last) {

			pODEBase->dT_last = pdT_last->to_cpu();
			delete pdT_last;
			pdT_last = nullptr;
		}

		if (pmxh) {

			pODEBase->mxh = pmxh->to_cpu();
			delete pmxh;
			pmxh = nullptr;
		}

		if (pdmdt) {

			pODEBase->dmdt = pdmdt->to_cpu();
			delete pdmdt;
			pdmdt = nullptr;
		}

		if (pmxh_av) { delete pmxh_av; pmxh_av = nullptr; }
		if (pavpoints) { delete pavpoints; pavpoints = nullptr; }

		if (pdmdt_av) { delete pdmdt_av; pdmdt_av = nullptr; }
		if (pavpoints2) { delete pavpoints2; pavpoints2 = nullptr; }
		if (plte) { delete plte; plte = nullptr; }

		if (psolve_spin_current) { delete psolve_spin_current; psolve_spin_current = nullptr; }

		if (palternator) { delete palternator; palternator = nullptr; }
	}
}

//Allocate memory for all static data; deletion only happens in the destructor, however allocation can also be triggered by UpdateConfiguration since the static data can be deleted by another instance which inherits same static data
void ODECommon_BaseCUDA::AllocateStaticData(void)
{
	if (!ptime) ptime = new mcu_val<cuBReal>(mGPU);
	if (!pstagetime) pstagetime = new mcu_val<cuBReal>(mGPU);

	if (!pdT) pdT = new mcu_val<cuBReal>(mGPU);
	if (!pdT_last) pdT_last = new mcu_val<cuBReal>(mGPU);

	if (!pmxh) pmxh = new mcu_val<cuBReal>(mGPU);
	if (!pmxh_av) pmxh_av = new mcu_val<cuReal3>(mGPU);
	if (!pavpoints) pavpoints = new mcu_val<size_t>(mGPU);

	if (!pdmdt) pdmdt = new mcu_val<cuBReal>(mGPU);
	if (!pdmdt_av) pdmdt_av = new mcu_val<cuReal3>(mGPU);
	if (!pavpoints2) pavpoints2 = new mcu_val<size_t>(mGPU);

	if (!plte) plte = new mcu_val<cuBReal>(mGPU);

	if (!psolve_spin_current) psolve_spin_current = new mcu_val<bool>(mGPU);

	if (!palternator) palternator = new mcu_val<bool>(mGPU);
}

void ODECommon_BaseCUDA::SyncODEValues(void)
{
	ptime->from_cpu(pODEBase->time);
	pstagetime->from_cpu(pODEBase->stagetime);

	pdT->from_cpu(pODEBase->dT);
	pdT_last->from_cpu(pODEBase->dT_last);

	pmxh->from_cpu(pODEBase->mxh);
	pdmdt->from_cpu(pODEBase->dmdt);

	palternator->from_cpu(pODEBase->alternator);
}

//set specific cuda values (used often)
void ODECommon_BaseCUDA::Sync_time(void)
{
	ptime->from_cpu(pODEBase->time);
	pstagetime->from_cpu(pODEBase->stagetime);
}

void ODECommon_BaseCUDA::Sync_dT(void)
{
	pdT->from_cpu(pODEBase->dT);
}

void ODECommon_BaseCUDA::Sync_dT_last(void)
{
	pdT_last->from_cpu(pODEBase->dT_last);
}

void ODECommon_BaseCUDA::Sync_alternator(void)
{
	palternator->from_cpu(pODEBase->alternator);
}

#endif