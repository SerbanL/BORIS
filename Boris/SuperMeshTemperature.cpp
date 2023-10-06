#include "stdafx.h"
#include "SuperMesh.h"

//--------------------------------------------------------- TEMPERATURE / HEAT SOLVER CONTROL : SuperMeshTemperature.cpp

BError SuperMesh::SetBaseTemperature(std::string meshName, double Temperature)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName == superMeshHandle) {

		//all meshes
		for (int idx = 0; idx < pMesh.size(); idx++) {

			pMesh[idx]->SetBaseTemperature(Temperature);

			//also set ambient temperature (for heat equation Robin boundary conditions) if heat module set
			pMesh[idx]->CallModuleMethod(&HeatBase::SetAmbientTemperature, Temperature);
		}
	}
	else {

		//named mesh only
		pMesh[meshName]->SetBaseTemperature(Temperature);

		//also set ambient temperature (for heat equation Robin boundary conditions) if heat module set
		pMesh[meshName]->CallModuleMethod(&HeatBase::SetAmbientTemperature, Temperature);
	}

	//when changing temperature it's possible the magnetization length needs to be changed.
	error = UpdateConfiguration(UPDATECONFIG_PARAMVALUECHANGED_MLENGTH);

	return error;
}

//ambient and alpha boundary coefficient for Robin boundary conditions - set in Heat module if active
BError SuperMesh::SetAmbientTemperature(std::string meshName, double T_ambient)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName == superMeshHandle) {

		//all meshes
		for (int idx = 0; idx < pMesh.size(); idx++) {

			pMesh[idx]->CallModuleMethod(&HeatBase::SetAmbientTemperature, T_ambient);
		}
	}
	else {

		//named mesh only
		pMesh[meshName]->CallModuleMethod(&HeatBase::SetAmbientTemperature, T_ambient);
	}

	return error;
}

BError SuperMesh::SetAlphaHeatBoundary(std::string meshName, double alpha_boundary)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName == superMeshHandle) {

		//all meshes
		for (int idx = 0; idx < pMesh.size(); idx++) {

			pMesh[idx]->CallModuleMethod(&HeatBase::SetAlphaBoundary, alpha_boundary);
		}
	}
	else {

		//named mesh only
		pMesh[meshName]->CallModuleMethod(&HeatBase::SetAlphaBoundary, alpha_boundary);
	}

	return error;
}

BError SuperMesh::SetInsulatingSides(std::string meshName, std::string literal, bool status)
{
	BError error(__FUNCTION__);

	if (!contains(meshName)) return error(BERROR_INCORRECTNAME);

	pMesh[meshName]->CallModuleMethod(&HeatBase::SetInsulatingSides, literal, status);

	return error;
}

BError SuperMesh::SetCurieTemperature(std::string meshName, double T_Curie)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	//Setting special functions temperature dependences with cuda switched on is problematic. Easiest just switch cuda off, then after switch it back on.
	bool switch_cuda_back_on = false;
	if (cudaEnabled) {

		SwitchCUDAState(false);
		switch_cuda_back_on = true;
	}

	//applicable for micromagnetic meshes only

	if (meshName == superMeshHandle) {

		//all meshes
		for (int idx = 0; idx < pMesh.size(); idx++) {

			if (!pMesh[idx]->is_atomistic()) {

				dynamic_cast<Mesh*>(pMesh[idx])->SetCurieTemperature(T_Curie, true);
			}
		}
	}
	else {

		//named mesh only
		if (!pMesh[meshName]->is_atomistic()) {

			dynamic_cast<Mesh*>(pMesh[meshName])->SetCurieTemperature(T_Curie, true);
		}
		else error(BERROR_INCORRECTNAME);
	}

	if (switch_cuda_back_on) SwitchCUDAState(true);

	return error;
}

BError SuperMesh::SetCurieTemperatureMaterial(std::string meshName, double T_Curie_material)
{
	BError error(__FUNCTION__);

	if (!contains(meshName)) return error(BERROR_INCORRECTNAME);

	if (!pMesh[meshName]->is_atomistic()) {

		dynamic_cast<Mesh*>(pMesh[meshName])->SetCurieTemperatureMaterial(T_Curie_material);
	}
	else return error(BERROR_INCORRECTNAME);

	return error;
}

BError SuperMesh::SetAtomicMagneticMoment(std::string meshName, DBL2 atomic_moment)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName == superMeshHandle) {

		//all meshes
		for (int idx = 0; idx < pMesh.size(); idx++) {

			if (!pMesh[idx]->is_atomistic()) {

				dynamic_cast<Mesh*>(pMesh[idx])->SetAtomicMoment(atomic_moment);
			}
		}
	}
	else {

		//named mesh only
		if (!pMesh[meshName]->is_atomistic()) {

			dynamic_cast<Mesh*>(pMesh[meshName])->SetAtomicMoment(atomic_moment);
		}
		else return error(BERROR_INCORRECTNAME);
	}

	return error;
}

//set Tc (critical temperature) coupling terms for 2-sublattice model
BError SuperMesh::SetTcCoupling(std::string meshName, DBL2 tau_ii, DBL2 tau_ij)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName == superMeshHandle) {

		//all meshes
		for (int idx = 0; idx < pMesh.size(); idx++) {

			if (!pMesh[idx]->is_atomistic()) {

				dynamic_cast<Mesh*>(pMesh[idx])->SetTcCoupling(tau_ii, tau_ij);
			}
		}
	}
	else {

		//named mesh only
		if (!pMesh[meshName]->is_atomistic()) {

			dynamic_cast<Mesh*>(pMesh[meshName])->SetTcCoupling(tau_ii, tau_ij);
		}
		else return error(BERROR_INCORRECTNAME);
	}

	return error;
}

BError SuperMesh::SetTcCoupling_Intra(std::string meshName, DBL2 tau_ii)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName == superMeshHandle) {

		//all meshes
		for (int idx = 0; idx < pMesh.size(); idx++) {

			if (!pMesh[idx]->is_atomistic()) {

				dynamic_cast<Mesh*>(pMesh[idx])->SetTcCoupling_Intra(tau_ii);
			}
		}
	}
	else {

		//named mesh only
		if (!pMesh[meshName]->is_atomistic()) {

			dynamic_cast<Mesh*>(pMesh[meshName])->SetTcCoupling_Intra(tau_ii);
		}
		else return error(BERROR_INCORRECTNAME);
	}

	return error;
}

BError SuperMesh::SetTcCoupling_Inter(std::string meshName, DBL2 tau_ij)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName == superMeshHandle) {

		//all meshes
		for (int idx = 0; idx < pMesh.size(); idx++) {

			if (!pMesh[idx]->is_atomistic()) {

				dynamic_cast<Mesh*>(pMesh[idx])->SetTcCoupling_Inter(tau_ij);
			}
		}
	}
	else {

		//named mesh only
		if (!pMesh[meshName]->is_atomistic()) {

			dynamic_cast<Mesh*>(pMesh[meshName])->SetTcCoupling_Inter(tau_ij);
		}
		else return error(BERROR_INCORRECTNAME);
	}

	return error;
}

//Set temperature model
BError SuperMesh::SetTemperatureModel(std::string meshName, int tmtype)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName == superMeshHandle) {

		//all meshes
		for (int idx = 0; idx < pMesh.size(); idx++) {

			pMesh[meshName]->CallModuleMethod(&HeatBase::Set_TMType, (TMTYPE_)tmtype);
		}
	}
	else {

		//named mesh only
		pMesh[meshName]->CallModuleMethod(&HeatBase::Set_TMType, (TMTYPE_)tmtype);
	}

	return error;
}

//load temperature in supermesh (SHeat::globalTemp) or in named mesh
BError SuperMesh::LoadOVF2Temp(std::string meshName, std::string fileName)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	VEC<double> data;
	OVF2 ovf2;
	error = ovf2.Read_OVF2_SCA(fileName, data);
	if (error) return error;

	if (meshName != superMeshHandle) {

		//temperature in individual mesh
		if (pMesh[meshName]->TComputation_Enabled()) {

			pMesh[meshName]->SetTempFromData(data);
		}
		else return error(BERROR_INCORRECTMODCONFIG);
	}
	else {

		//global temperature
		error = CallModuleMethod<BError, SHeat, VEC<double>&>(&SHeat::LoadGlobalTemperature, data);
	}

	return error;
}