#include "stdafx.h"
#include "MElasticCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MELASTIC

#include "MElastic.h"
#include "MeshCUDA.h"
#include "Mesh.h"
#include "DataDefs.h"

#include "MElastic_BoundariesCUDA.h"
#include "MElastic_Boundaries.h"

#include "HeatBase.h"

#include "SMElastic.h"

MElasticCUDA::MElasticCUDA(Mesh* pMesh_, MElastic* pMElastic_) :
	ModulesCUDA(),
	vx(mGPU), vy(mGPU), vz(mGPU),
	sdd(mGPU), sxy(mGPU), sxz(mGPU), syz(mGPU),
	vx2(mGPU), vy2(mGPU), vz2(mGPU),
	sdd2(mGPU), sxy2(mGPU), sxz2(mGPU), syz2(mGPU),
	Temp_previous(mGPU), T_ambient(mGPU)
{
	pMesh = pMesh_;
	pMeshCUDA = pMesh->pMeshCUDA;
	pMElastic = pMElastic_;

	error_on_create = copy_VECs_to_GPU();
	error_on_create = UpdateConfiguration(UPDATECONFIG_FORCEUPDATE);
}

MElasticCUDA::~MElasticCUDA()
{
	//Transport managed quantities must be transferred back to their cpu versions - values only as sizes must match
	if (Holder_Module_Available()) {

		//If holder module still available, this means the cpu version of this module has not been deleted.
		//The only way this could happen is if CUDA is being switched off. 
		//In this case we want to copy over to cpu vecs, but no need to clear memory explicitly, as this will be done in the cu-obj managed destructor when these cuVECs go out of scope.
		//This is done in the CUDA version of Mesh where these quantities are held.

		//However we also have quantities held in MElastic modules which must be copied over:
		vx.copy_to_cpuvec(pMElastic->vx);
		vy.copy_to_cpuvec(pMElastic->vy);
		vz.copy_to_cpuvec(pMElastic->vz);

		sdd.copy_to_cpuvec(pMElastic->sdd);
		sxy.copy_to_cpuvec(pMElastic->sxy);
		sxz.copy_to_cpuvec(pMElastic->sxz);
		syz.copy_to_cpuvec(pMElastic->syz);

		if (pMElastic->vx2.linear_size()) {

			vx2.copy_to_cpuvec(pMElastic->vx2);
			vy2.copy_to_cpuvec(pMElastic->vy2);
			vz2.copy_to_cpuvec(pMElastic->vz2);

			sdd2.copy_to_cpuvec(pMElastic->sdd2);
			sxy2.copy_to_cpuvec(pMElastic->sxy2);
			sxz2.copy_to_cpuvec(pMElastic->sxz2);
			syz2.copy_to_cpuvec(pMElastic->syz2);
		}
	}
	else {

		//Holder module not available. This means this module has been deleted entirely, but CUDA must still be switched on.
		//In this case free up GPU memory as these cuVECs will not be going out of scope, but in any case they're not needed anymore.
		pMeshCUDA->u_disp.clear();
		pMeshCUDA->strain_diag.clear();
		pMeshCUDA->strain_odiag.clear();
	}
}

BError MElasticCUDA::Initialize(void)
{
	BError error(CLASS_STR(MElasticCUDA));

	ZeroEnergy();

	//refresh ambient temperature here
	if (thermoelasticity_enabled) T_ambient.from_cpu(pMesh->CallModuleMethod(&HeatBase::GetAmbientTemperature));

	//disabled by setting magnetoelastic coefficient to zero (also disabled in non-magnetic meshes)
	melastic_field_disabled = (IsZ(pMesh->MEc.get0().i) && IsZ(pMesh->MEc.get0().j)) || (pMesh->GetMeshType() != MESH_ANTIFERROMAGNETIC && pMesh->GetMeshType() != MESH_FERROMAGNETIC);

	if (!initialized && pMElastic->pSMEl->get_el_dT() > 0.0) {

		error = pMElastic->Initialize();
		if (error) return error;

		Fext_equationCUDA.clear();
		Fext_equationCUDA.resize(pMElastic->external_stress_surfaces.size());

		external_stress_surfaces.clear();
		external_stress_surfaces.resize(pMElastic->external_stress_surfaces.size());

		external_stress_surfaces_arr.clear();

		//Setup MElastic_BoundariesCUDA
		for (int idx = 0; idx < pMElastic->external_stress_surfaces.size(); idx++) {

			//setup surface
			external_stress_surfaces[idx]()->setup_surface(
				pMElastic->external_stress_surfaces[idx].get_box(),
				pMElastic->external_stress_surfaces[idx].get_cellsize(),
				pMElastic->external_stress_surfaces[idx].get_orientation());

			//now setup force (constant or equation)
			if (pMElastic->external_stress_surfaces[idx].is_constant_force()) {

				external_stress_surfaces[idx]()->setup_fixed_stimulus(pMElastic->external_stress_surfaces[idx].get_constant_force());
				Fext_equationCUDA[idx].clear();
			}
			else if (pMElastic->external_stress_surfaces[idx].is_equation_set()) {

				if (!external_stress_surfaces[idx]()->make_cuda_equation(
					Fext_equationCUDA[idx], pMElastic->external_stress_surfaces[idx].get_equation_fspec())) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			}

			external_stress_surfaces_arr.push_back(external_stress_surfaces[idx].get_managed_object());
		}

		//set Dirichlet conditions for u_disp (zero, i.e. fixed, or zero displacement, points)
		pMeshCUDA->u_disp.clear_dirichlet_flags();
		for (auto& fixed_rect : pMElastic->fixed_u_surfaces) pMeshCUDA->u_disp.set_dirichlet_conditions(fixed_rect, cuReal3());

		//set Dirichlet conditions for strain_diag (external force)
		pMeshCUDA->strain_diag.clear_dirichlet_flags();
		for (auto& external_stress : pMElastic->external_stress_surfaces) pMeshCUDA->strain_diag.set_dirichlet_conditions(external_stress.get_surface(), cuReal3());

		Reset_ElSolver();
	}

	//Make sure display data has memory allocated (or freed) as required - this is only required for magnetic meshes (MElastic module could also be used in metal and insulator meshes)
	if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC || pMeshCUDA->GetMeshType() == MESH_FERROMAGNETIC) {

		//Make sure display data has memory allocated (or freed) as required
		error = Update_Module_Display_VECs(
			(cuReal3)pMeshCUDA->h, (cuRect)pMeshCUDA->meshRect,
			(MOD_)pMeshCUDA->Get_Module_Heff_Display() == MOD_MELASTIC || pMeshCUDA->IsOutputDataSet_withRect(DATA_E_MELASTIC),
			(MOD_)pMeshCUDA->Get_Module_Energy_Display() == MOD_MELASTIC || pMeshCUDA->IsOutputDataSet_withRect(DATA_E_MELASTIC),
			pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC);
		if (!error)	initialized = true;
	}

	return error;
}

BError MElasticCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(MElasticCUDA));

	//is thermoelasticity enabled or status changed?
	if (thermoelasticity_enabled != (pMesh->Temp.linear_size() && pMesh->IsModuleSet(MOD_HEAT) && IsNZ(pMesh->thalpha.get0()))) {

		Uninitialize();
		thermoelasticity_enabled = !thermoelasticity_enabled;

		if (thermoelasticity_enabled) {

			if (Temp_previous.resize(pMeshCUDA->h_t, pMeshCUDA->meshRect)) Temp_previous.set(0.0);
			else return error(BERROR_OUTOFGPUMEMORY_CRIT);
		}
		else Temp_previous.clear();
	}

	//is magnetostriction enabled?
	if (magnetostriction_enabled != (pMesh->Magnetism_Enabled() && (IsNZ(pMesh->mMEc.get0().i) || IsNZ(pMesh->mMEc.get0().j)))) {

		Uninitialize();
		magnetostriction_enabled = !magnetostriction_enabled;
	}

	bool success = true;

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MELASTIC, UPDATECONFIG_MESHSHAPECHANGE, UPDATECONFIG_MESHCHANGE)) {

		Uninitialize();

		if (pMeshCUDA->u_disp.size_cpu().dim()) {

			success &= pMeshCUDA->u_disp.resize(pMeshCUDA->h_m, pMeshCUDA->meshRect);
		}
		else {

			success &= pMeshCUDA->u_disp.set_from_cpuvec(pMesh->u_disp);
		}

		//strain tensor - set empty cells using information in u_disp
		if (pMeshCUDA->u_disp.size_cpu().dim()) {

			success &= pMeshCUDA->strain_diag.resize(pMeshCUDA->h_m, pMeshCUDA->meshRect, pMeshCUDA->strain_diag);
			success &= pMeshCUDA->strain_odiag.resize(pMeshCUDA->h_m, pMeshCUDA->meshRect, pMeshCUDA->strain_odiag);
		}
		else {

			success &= pMeshCUDA->strain_diag.assign(pMeshCUDA->h_m, pMeshCUDA->meshRect, cuReal3(), pMeshCUDA->strain_diag);
			success &= pMeshCUDA->strain_odiag.assign(pMeshCUDA->h_m, pMeshCUDA->meshRect, cuReal3(), pMeshCUDA->strain_odiag);
		}

		//correct size for FDTD data
		success &= vx.resize(cuSZ3(pMeshCUDA->n_m.x, pMeshCUDA->n_m.y + 1, pMeshCUDA->n_m.z + 1));
		success &= vy.resize(cuSZ3(pMeshCUDA->n_m.x + 1, pMeshCUDA->n_m.y, pMeshCUDA->n_m.z + 1));
		success &= vz.resize(cuSZ3(pMeshCUDA->n_m.x + 1, pMeshCUDA->n_m.y + 1, pMeshCUDA->n_m.z));

		success &= sdd.resize(cuSZ3(pMeshCUDA->n_m.x + 1, pMeshCUDA->n_m.y + 1, pMeshCUDA->n_m.z + 1));
		success &= sxy.resize(cuSZ3(pMeshCUDA->n_m.x, pMeshCUDA->n_m.y, pMeshCUDA->n_m.z + 1));
		success &= sxz.resize(cuSZ3(pMeshCUDA->n_m.x, pMeshCUDA->n_m.y + 1, pMeshCUDA->n_m.z));
		success &= syz.resize(cuSZ3(pMeshCUDA->n_m.x + 1, pMeshCUDA->n_m.y, pMeshCUDA->n_m.z));

		//update mesh dimensions in equation constants
		if (pMElastic->Sd_equation.is_set()) Set_Sd_Equation(pMElastic->Sd_equation.get_vector_fspec());
		if (pMElastic->Sod_equation.is_set()) Set_Sod_Equation(pMElastic->Sod_equation.get_vector_fspec());

		//for trigonal crystal system need additional velocity and stress components
		if (pMElastic->crystal == CRYSTAL_TRIGONAL) {

			success &= vx2()->resize(cuSZ3(pMeshCUDA->n_m.x, pMeshCUDA->n_m.y, pMeshCUDA->n_m.z));
			success &= vy2()->resize(cuSZ3(pMeshCUDA->n_m.x + 1, pMeshCUDA->n_m.y + 1, pMeshCUDA->n_m.z));
			success &= vz2()->resize(cuSZ3(pMeshCUDA->n_m.x + 1, pMeshCUDA->n_m.y, pMeshCUDA->n_m.z + 1));

			success &= sdd2()->resize(cuSZ3(pMeshCUDA->n_m.x + 1, pMeshCUDA->n_m.y, pMeshCUDA->n_m.z));
			success &= sxy2()->resize(cuSZ3(pMeshCUDA->n_m.x, pMeshCUDA->n_m.y + 1, pMeshCUDA->n_m.z));
			success &= sxz2()->resize(cuSZ3(pMeshCUDA->n_m.x, pMeshCUDA->n_m.y, pMeshCUDA->n_m.z + 1));
			success &= syz2()->resize(cuSZ3(pMeshCUDA->n_m.x + 1, pMeshCUDA->n_m.y + 1, pMeshCUDA->n_m.z + 1));
		}
		else {

			vx2()->clear();
			vy2()->clear();
			vz2()->clear();
			sdd2()->clear();
			sxy2()->clear();
			sxz2()->clear();
			syz2()->clear();
		}
	}

	if (!success) return error(BERROR_OUTOFGPUMEMORY_CRIT);

	return error;
}

void MElasticCUDA::UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage)
{
	if (cfgMessage == UPDATECONFIG_TEQUATION_CONSTANTS) {

		//if this affects external stress surfaces, equations will need to be remade
		Uninitialize();
	}
	else if (cfgMessage == UPDATECONFIG_TEQUATION_CLEAR) {

	}
}

//------------------- Configuration

//reset stress-strain solver to initial values (zero velocity, displacement and stress)
void MElasticCUDA::Reset_ElSolver(void)
{
	vx()->set(0.0); vy()->set(0.0); vz()->set(0.0);

	pMeshCUDA->u_disp()->set(cuReal3());
	pMeshCUDA->strain_diag()->set(cuReal3());
	pMeshCUDA->strain_odiag()->set(cuReal3());

	if (thermoelasticity_enabled) {

		//refresh ambient temperature here
		T_ambient.from_cpu(pMesh->CallModuleMethod(&HeatBase::GetAmbientTemperature));
	}

	//Additional discretization scheme
	if (pMElastic->vx2.linear_size()) {

		vx2()->set(0.0); vy2()->set(0.0); vz2()->set(0.0);
	}

	//setting the stress depends on if thermoelasticity or magnetostriction are enabled, so not straightforward
	Set_Initial_Stress();
}

//clear text equations
void MElasticCUDA::Clear_Sd_Sod_Equations(void)
{
	if (Sd_equation.is_set()) Sd_equation.clear();
	if (Sod_equation.is_set()) Sod_equation.clear();
}

//copy all required mechanical VECs from their cpu versions
BError MElasticCUDA::copy_VECs_to_GPU(void)
{
	BError error(CLASS_STR(MElasticCUDA));

	bool success = true;

	success &= pMeshCUDA->u_disp()->set_from_cpuvec(pMesh->u_disp);
	success &= pMeshCUDA->strain_diag()->set_from_cpuvec(pMesh->strain_diag);
	success &= pMeshCUDA->strain_odiag()->set_from_cpuvec(pMesh->strain_odiag);

	success &= vx()->set_from_cpuvec(pMElastic->vx);
	success &= vy()->set_from_cpuvec(pMElastic->vy);
	success &= vz()->set_from_cpuvec(pMElastic->vz);

	success &= sdd()->set_from_cpuvec(pMElastic->sdd);
	success &= sxy()->set_from_cpuvec(pMElastic->sxy);
	success &= sxz()->set_from_cpuvec(pMElastic->sxz);
	success &= syz()->set_from_cpuvec(pMElastic->syz);

	//Additional discretization scheme
	if (pMElastic->vx2.linear_size()) {

		success &= vx2()->set_from_cpuvec(pMElastic->vx2);
		success &= vy2()->set_from_cpuvec(pMElastic->vy2);
		success &= vz2()->set_from_cpuvec(pMElastic->vz2);

		success &= sdd2()->set_from_cpuvec(pMElastic->sdd2);
		success &= sxy2()->set_from_cpuvec(pMElastic->sxy2);
		success &= sxz2()->set_from_cpuvec(pMElastic->sxz2);
		success &= syz2()->set_from_cpuvec(pMElastic->syz2);
	}

	if (!success) return error(BERROR_OUTOFGPUMEMORY_CRIT);

	return error;
}

#endif

#endif