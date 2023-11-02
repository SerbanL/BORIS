#include "stdafx.h"
#include "SHeatCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_HEAT

#include "SHeat.h"
#include "SuperMesh.h"

SHeatCUDA::SHeatCUDA(SuperMesh* pSMesh_, SHeat* pSHeat_) :
	ModulesCUDA(),
	globalTemp(mGPU)
{
	pSMesh = pSMesh_;
	pSHeat = pSHeat_;

	error_on_create = UpdateConfiguration(UPDATECONFIG_FORCEUPDATE);
}

SHeatCUDA::~SHeatCUDA()
{

}

//-------------------Abstract base class method implementations

BError SHeatCUDA::Initialize(void)
{
	BError error(CLASS_STR(SHeatCUDA));

	//no energy density contribution here
	ZeroEnergy();

	error = pSHeat->Initialize();
	if (error) return error;

	//check meshes to set heat boundary flags (NF2_CMBND flags for Temp)

	//clear everything then rebuild
	pHeat.clear();
	CMBNDcontactsCUDA.clear();
	CMBNDcontacts.clear();
	pTemp.clear();

	//now build pHeat (and pTemp)
	for (int idx = 0; idx < pSMesh->size(); idx++) {

		if ((*pSMesh)[idx]->IsModuleSet(MOD_HEAT)) {

			//do not include dormant meshes even if they have heat module enabled
			if ((*pSMesh)[idx]->Is_Dormant()) continue;

			pHeat.push_back(dynamic_cast<HeatBaseCUDA*>((*pSMesh)[idx]->GetCUDAModule(MOD_HEAT)));
			pTemp.push_back(&(*pSMesh)[idx]->pMeshBaseCUDA->Temp);
		}
	}

	//set cmbnd flags
	for (int idx = 0; idx < (int)pHeat.size(); idx++) {

		//it's easier to just copy the flags entirely from the cpu versions.
		//Notes :
		//1. By design the cpu versions are required to keep size and flags up to date (but not mesh values)
		//2. pHeat in SHeat has exactly the same size and order
		//3. SHeat UpdateConfiguration was called just before, which called this CUDA version at the end.

		if (!(*pTemp[idx]).copyflags_from_cpuvec(*pSHeat->pTemp[idx])) error(BERROR_GPUERROR_CRIT);
	}

	for (int idx = 0; idx < pSHeat->CMBNDcontacts.size(); idx++) {

		std::vector<mCMBNDInfoCUDA> mesh_contacts;
		std::vector<CMBNDInfoCUDA> mesh_contacts_cpu;

		for (int idx_contact = 0; idx_contact < pSHeat->CMBNDcontacts[idx].size(); idx_contact++) {

			mCMBNDInfoCUDA contact(mGPU);
			contact.copy_from_CMBNDInfo<CMBNDInfo>(pSHeat->CMBNDcontacts[idx][idx_contact], pTemp[idx]->get_pbox_d_ref());

			mesh_contacts.push_back(contact);

			mesh_contacts_cpu.push_back(pSHeat->CMBNDcontacts[idx][idx_contact]);
		}

		CMBNDcontactsCUDA.push_back(mesh_contacts);
		CMBNDcontacts.push_back(mesh_contacts_cpu);
	}

	initialized = true;

	return error;
}

BError SHeatCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SHeatCUDA));

	Uninitialize();

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHADDED, UPDATECONFIG_MESHDELETED, UPDATECONFIG_MODULEADDED, UPDATECONFIG_MODULEDELETED, UPDATECONFIG_SWITCHCUDASTATE)) {
		
		//clear everything then rebuild
		pHeat.clear();
		pTemp.clear();

		//now build pHeat (and pTemp)
		for (int idx = 0; idx < pSMesh->size(); idx++) {

			if ((*pSMesh)[idx]->IsModuleSet(MOD_HEAT)) {

				//do not include dormant meshes even if they have heat module enabled
				if ((*pSMesh)[idx]->Is_Dormant()) continue;

				pHeat.push_back(dynamic_cast<HeatBaseCUDA*>((*pSMesh)[idx]->GetCUDAModule(MOD_HEAT)));
				pTemp.push_back(&(*pSMesh)[idx]->pMeshBaseCUDA->Temp);
			}
		}
		
		//set globalTemp from cpu version if not set already
		if (pSHeat->globalTemp.linear_size() != globalTemp.size_cpu().dim()) LoadGlobalTemperature();
	}

	return error;
}

void SHeatCUDA::UpdateField(void)
{
	//only need to update this after an entire magnetization equation time step is solved
	//also if heat_dT is set to zero skip the heat equation solver : this will maintain a fixed temperature
	if (!pSMesh->CurrentTimeStepSolved() || pSHeat->heat_dT < MINTIMESTEP) {

		//set from global temperature in each mesh
		if (pSHeat->globalTemp.linear_size()) {

			//if global temperature velocity set then shift it first
			if (pSHeat->globalTemp_velocity != DBL3()) {

				//current time so we can calculate required shift
				double globalTemp_current_time = pSMesh->GetTime();

				//if current time less than stored previous time then something is wrong (e.g. ode was reset - reset shifting debt as well)
				if (globalTemp_current_time < pSHeat->globalTemp_last_time) {

					pSHeat->globalTemp_last_time = globalTemp_current_time;
					pSHeat->globalTemp_shift_debt = DBL3();
				}

				//add to total amount of shifting which hasn't yet been executed (the shift debt)
				pSHeat->globalTemp_shift_debt += (globalTemp_current_time - pSHeat->globalTemp_last_time) * pSHeat->globalTemp_velocity;

				//clip the shift to execute if required
				DBL3 shift = DBL3(
					pSHeat->globalTemp_shift_clip.x > 0.0 ? 0.0 : pSHeat->globalTemp_shift_debt.x,
					pSHeat->globalTemp_shift_clip.y > 0.0 ? 0.0 : pSHeat->globalTemp_shift_debt.y,
					pSHeat->globalTemp_shift_clip.z > 0.0 ? 0.0 : pSHeat->globalTemp_shift_debt.z);

				if (pSHeat->globalTemp_shift_clip.x > 0.0 && fabs(pSHeat->globalTemp_shift_debt.x) > pSHeat->globalTemp_shift_clip.x)
					shift.x = floor(fabs(pSHeat->globalTemp_shift_debt.x) / pSHeat->globalTemp_shift_clip.x) * pSHeat->globalTemp_shift_clip.x * get_sign(pSHeat->globalTemp_shift_debt.x);

				if (pSHeat->globalTemp_shift_clip.y > 0.0 && fabs(pSHeat->globalTemp_shift_debt.y) > pSHeat->globalTemp_shift_clip.y)
					shift.y = floor(fabs(pSHeat->globalTemp_shift_debt.y) / pSHeat->globalTemp_shift_clip.y) * pSHeat->globalTemp_shift_clip.y * get_sign(pSHeat->globalTemp_shift_debt.y);

				if (pSHeat->globalTemp_shift_clip.z > 0.0 && fabs(pSHeat->globalTemp_shift_debt.z) > pSHeat->globalTemp_shift_clip.z)
					shift.z = floor(fabs(pSHeat->globalTemp_shift_debt.z) / pSHeat->globalTemp_shift_clip.z) * pSHeat->globalTemp_shift_clip.z * get_sign(pSHeat->globalTemp_shift_debt.z);

				//execute shift if needed
				if (shift != DBL3()) {

					globalTemp.shift_rect_start(shift);
					pSHeat->globalTemp_shift_debt -= shift;
				}

				pSHeat->globalTemp_last_time = globalTemp_current_time;
			}

			//now set it
			for (int idx = 0; idx < (int)pHeat.size(); idx++) pHeat[idx]->SetFromGlobalTemperature(globalTemp);
		}
		//set from spatial variation of cT (if enabled in respective meshes)
		else {

			for (int idx = 0; idx < (int)pSHeat->pHeat.size(); idx++)  pSHeat->pHeat[idx]->SetBaseTemperature(pSHeat->pHeat[idx]->pMeshBase->base_temperature, false);
		}

		return;
	}
	
	cuBReal dT = pSHeat->heat_dT;

	//number of sub_steps to cover magnetic_dT required when advancing in smaller heat_dT steps
	int sub_steps = (int)floor_epsilon(pSHeat->magnetic_dT / pSHeat->heat_dT);

	//any left-over epsilon_dT < heat_dT
	cuBReal epsilon_dT = pSHeat->magnetic_dT - pSHeat->heat_dT * sub_steps;

	for (int step_idx = 0; step_idx < sub_steps + 1; step_idx++) {

		//the last step may have a different time step - take this epsilon_dT step (if not zero)
		if (step_idx == sub_steps) {

			if (epsilon_dT) dT = epsilon_dT;
			else continue;
		}

		//1. solve Temp in each mesh separately (1 iteration each) - CMBND cells not set yet
		for (int idx = 0; idx < (int)pHeat.size(); idx++) {

			if (pSHeat->pHeat[idx]->Get_TMType() == TMTYPE_1TM) pHeat[idx]->IterateHeatEquation_1TM(dT);
			else if (pSHeat->pHeat[idx]->Get_TMType() == TMTYPE_2TM) pHeat[idx]->IterateHeatEquation_2TM(dT);
		}

		//2. calculate boundary conditions (i.e. temperature values at CMBND cells)
		set_cmbnd_values();
	}

	//3. update the magnetic dT that will be used next time around to increment the heat solver by
	pSHeat->magnetic_dT = pSMesh->GetTimeStep();
}

//-------------------Global Temperature

//clear globalTemp
void SHeatCUDA::ClearGlobalTemperature(void)
{
	globalTemp.clear();
}

//shift rectangle of globalTemp if set
void SHeatCUDA::ShiftGlobalTemperatureRectangle(cuReal3 shift)
{
	globalTemp.shift_rect_start(shift);
}

BError SHeatCUDA::LoadGlobalTemperature(void)
{
	BError error(CLASS_STR(SHeatCUDA));

	if (!pSHeat->globalTemp.linear_size()) {

		globalTemp.clear();
	}
	else if (!globalTemp.set_from_cpuvec(pSHeat->globalTemp)) {

		globalTemp.clear();
		return error(BERROR_OUTOFGPUMEMORY_NCRIT);
	}

	return error;
}

#endif

#endif