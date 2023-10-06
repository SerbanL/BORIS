#include "stdafx.h"
#include "SuperMesh.h"

//---------------------------------------------------------IMPORTANT CONTROL METHODS

//load field in supermesh (globalField) or in named mesh
BError SuperMesh::LoadOVF2Field(std::string meshName, std::string fileName)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName != superMeshHandle) {

		//field in individual mesh
		if (pMesh[meshName]->Magnetism_Enabled() && pMesh[meshName]->IsModuleSet(MOD_ZEEMAN)) {

			pMesh[meshName]->CallModuleMethod(&ZeemanBase::SetFieldVEC_FromOVF2, fileName);
		}
		else return error(BERROR_INCORRECTMODCONFIG);
	}
	else {

		//global supermesh field
		OVF2 ovf2;
		error = ovf2.Read_OVF2_VEC(fileName, globalField);
		if (error) return error;

		globalField_rect = globalField.rect;
		globalField_h = globalField.h;

#if COMPILECUDA == 1
		if (pSMeshCUDA) {

			if (!pSMeshCUDA->GetGlobalField().set_from_cpuvec(globalField)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		}
#endif

		//now that global field is set, must update configuration so each Zeeman module can uninitialize (on reinitialization correct settings will be made in Zeeman modules)
		error = UpdateConfiguration(UPDATECONFIG_SMESH_GLOBALFIELD);
	}

	return error;
}

BError SuperMesh::ClearGlobalField(void)
{
	BError error(__FUNCTION__);

	globalField.clear();

#if COMPILECUDA == 1
	if (pSMeshCUDA) {

		pSMeshCUDA->GetGlobalField().clear();
	}
#endif

	error = UpdateConfiguration(UPDATECONFIG_SMESH_GLOBALFIELD);

	return error;
}

//shift globalField rectangle
void SuperMesh::ShiftGlobalField(DBL3 shift)
{
	globalField.shift_rect_start(shift);
	globalField_rect = globalField.rect;

#if COMPILECUDA == 1
	if (pSMeshCUDA) {

		pSMeshCUDA->GetGlobalField().shift_rect_start(shift);
	}
#endif

	UpdateConfiguration(UPDATECONFIG_SMESH_GLOBALFIELD);
}

//if globalField_velocity is not zero, then implement global field shifting
void SuperMesh::GlobalFieldShifting_Algorithm(void)
{
	//if global field velocity set then shift it first
	if (globalField_velocity != DBL3()) {

		//current time so we can calculate required shift
		double globalField_current_time = GetTime();

		//if current time less than stored previous time then something is wrong (e.g. ode was reset - reset shifting debt as well)
		if (globalField_current_time < globalField_last_time) {

			globalField_last_time = globalField_current_time;
			globalField_shift_debt = DBL3();
		}

		//add to total amount of shifting which hasn't yet been executed (the shift debt)
		globalField_shift_debt += (globalField_current_time - globalField_last_time) * globalField_velocity;

		//clip the shift to execute if required
		DBL3 shift = DBL3(
			globalField_shift_clip.x > 0.0 ? 0.0 : globalField_shift_debt.x,
			globalField_shift_clip.y > 0.0 ? 0.0 : globalField_shift_debt.y,
			globalField_shift_clip.z > 0.0 ? 0.0 : globalField_shift_debt.z);

		if (globalField_shift_clip.x > 0.0 && fabs(globalField_shift_debt.x) > globalField_shift_clip.x)
			shift.x = floor(fabs(globalField_shift_debt.x) / globalField_shift_clip.x) * globalField_shift_clip.x * get_sign(globalField_shift_debt.x);

		if (globalField_shift_clip.y > 0.0 && fabs(globalField_shift_debt.y) > globalField_shift_clip.y)
			shift.y = floor(fabs(globalField_shift_debt.y) / globalField_shift_clip.y) * globalField_shift_clip.y * get_sign(globalField_shift_debt.y);

		if (globalField_shift_clip.z > 0.0 && fabs(globalField_shift_debt.z) > globalField_shift_clip.z)
			shift.z = floor(fabs(globalField_shift_debt.z) / globalField_shift_clip.z) * globalField_shift_clip.z * get_sign(globalField_shift_debt.z);

		//execute shift if needed
		if (shift != DBL3()) {

			globalField.shift_rect_start(shift);
			globalField_shift_debt -= shift;
		}

		globalField_last_time = globalField_current_time;

#if COMPILECUDA == 1
		//same shift if CUDA switched on
		if (pSMeshCUDA && shift != DBL3()) GetGlobalFieldCUDA().shift_rect_start(shift);
#endif

		//now set it in Zeeman modules
		for (int idx = 0; idx < pMesh.size(); idx++) {

			pMesh[idx]->CallModuleMethod(&ZeemanBase::SetGlobalField);
		}
	}
}