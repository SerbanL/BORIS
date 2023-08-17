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