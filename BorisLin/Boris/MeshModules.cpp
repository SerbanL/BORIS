#include "stdafx.h"
#include "Mesh.h"
#include "SuperMesh.h"

//----------------------------------- MODULES CONTROL

BError Mesh::AddModule(MOD_ moduleID, bool force_add)
{
	BError error(__FUNCTION__);

	if (moduleID <= MOD_ERROR) return error(BERROR_INCORRECTNAME);

	//first make sure the module can be added to this mesh type
	if (!vector_contains(modules_for_meshtype[INT2(meshType, 0)], moduleID)) return error(BERROR_INCORRECTNAME);

	//if module is already set then don't add another one unless we specifically ask to do it.
	//only one module of each type allowed normally, but we may want to force adding of multiple modules of same type (force_add = true)
	if (!force_add && IsModuleSet(moduleID)) return error(BERROR_INCORRECTACTION_SILENT);

	//now set the module
	switch (moduleID) {

	case MOD_DEMAG_N:
		pMod.push_back(new Demag_N(this), MOD_DEMAG_N);
		break;

	case MOD_DEMAG:
		pMod.push_back(new Demag(this), MOD_DEMAG);
		break;

	//individual mesh demag module used by SDemag super-mesh module - not available in the console, but added when SDemag module is enabled
	case MOD_SDEMAG_DEMAG:
		//there's the option of excluding this mesh from multilayered demag convolution
		if (!Get_Demag_Exclusion()) pMod.push_back(new SDemag_Demag(this), MOD_SDEMAG_DEMAG);
		break;

	case MOD_STRAYFIELD_MESH:
		pMod.push_back(new StrayField_Mesh(this), MOD_STRAYFIELD_MESH);
		break;

	case MOD_EXCHANGE:
		pMod.push_back(new Exch_6ngbr_Neu(this), MOD_EXCHANGE);
		break;

	case MOD_DMEXCHANGE:
		pMod.push_back(new DMExchange(this), MOD_DMEXCHANGE);
		break;

	case MOD_IDMEXCHANGE:
		pMod.push_back(new iDMExchange(this), MOD_IDMEXCHANGE);
		break;

	case MOD_VIDMEXCHANGE:
		pMod.push_back(new viDMExchange(this), MOD_VIDMEXCHANGE);
		break;

	case MOD_SURFEXCHANGE:
		if (meshType == MESH_FERROMAGNETIC) pMod.push_back(new SurfExchange(this), MOD_SURFEXCHANGE);
		else if (meshType == MESH_ANTIFERROMAGNETIC) pMod.push_back(new SurfExchange_AFM(this), MOD_SURFEXCHANGE);
		break;

	case MOD_ZEEMAN:
		pMod.push_back(new Zeeman(this), MOD_ZEEMAN);
		break;

	case MOD_MOPTICAL:
		pMod.push_back(new MOptical(this), MOD_MOPTICAL);
		break;

	case MOD_ANIUNI:
		pMod.push_back(new Anisotropy_Uniaxial(this), MOD_ANIUNI);
		break;

	case MOD_ANICUBI:
		pMod.push_back(new Anisotropy_Cubic(this), MOD_ANICUBI);
		break;

	case MOD_ANIBI:
		pMod.push_back(new Anisotropy_Biaxial(this), MOD_ANIBI);
		break;

	case MOD_ANITENS:
		pMod.push_back(new Anisotropy_Tensorial(this), MOD_ANITENS);
		break;
	
	case MOD_MELASTIC:
		pMod.push_back(new MElastic(this), MOD_MELASTIC);
		break;

	case MOD_TRANSPORT:
		pMod.push_back(new Transport(this), MOD_TRANSPORT);
		break;

	case MOD_TMR:
		pMod.push_back(new TMR(this), MOD_TMR);
		break;

	case MOD_HEAT:
		pMod.push_back(new Heat(this), MOD_HEAT);
		break;

	case MOD_SOTFIELD:
		pMod.push_back(new SOTField(this), MOD_SOTFIELD);
		break;

	case MOD_STFIELD:
		pMod.push_back(new STField(this), MOD_STFIELD);
		break;

	case MOD_ROUGHNESS:
		pMod.push_back(new Roughness(this), MOD_ROUGHNESS);
		break;
	}

	//check the module was created correctly - if not, delete it
	error = pMod.back()->Error_On_Create();

	if (error) pMod.pop_back();
	else {

		//Delete any modules which are exclusive to moduleId
		for (int idx = 0; idx < (int)exclusiveModules[moduleID].size(); idx++) {

			MOD_ module = exclusiveModules[moduleID][idx];
			if (module == moduleID) continue;

			if (IsModuleSet(module)) { delete pMod[pMod.get_index_from_ID(module)]; pMod.erase(INT2(module, 0)); }
		}
	}

	//Make sure Zeeman module is always the first one in the list : Zeeman module sets Heff (if Zeeman module disabled then PrepareIteration clears Heff)
	if (IsModuleSet(MOD_ZEEMAN)) {

		int idxZeeman = pMod.get_index_from_ID(MOD_ZEEMAN);
		if (idxZeeman != 0) pMod.move(idxZeeman);
	}

	return error;
}

//update MOD_TRANSPORT module only if set
void Mesh::UpdateTransportSolver(void)
{
	if (IsModuleSet(MOD_TRANSPORT)) pMod(MOD_TRANSPORT)->UpdateField();
	else if (IsModuleSet(MOD_TMR)) pMod(MOD_TMR)->UpdateField();
}

#if COMPILECUDA == 1
void Mesh::UpdateTransportSolverCUDA(void)
{
	if (IsModuleSet(MOD_TRANSPORT)) pMod(MOD_TRANSPORT)->UpdateFieldCUDA();
	else if (IsModuleSet(MOD_TMR)) pMod(MOD_TMR)->UpdateFieldCUDA();
}
#endif
