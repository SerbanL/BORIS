#include "stdafx.h"
#include "SDemag.h"

#ifdef MODULE_COMPILATION_SDEMAG

#include "SuperMesh.h"

//called from Initialize if using SMesh demag
BError SDemag::Initialize_SMesh_Demag(void)
{
	BError error(CLASS_STR(SDemag));

	error = Calculate_Demag_Kernels();
	if (error) return error;

	Initialize_Mesh_Transfer();

	return error;
}

//initialize transfer object for supermesh convolution
BError SDemag::Initialize_Mesh_Transfer(void)
{
	BError error(CLASS_STR(SDemag));

	//clear transfer objects before remaking them
	sm_Vals.clear_transfer();
	sm_Vals.clear_transfer2();
	non_empty_cells = 0;

	//now calculate data required for mesh transfers, as well as demag corrections
	std::vector< VEC<DBL3>* > pVal_from, pVal_from2;
	std::vector< VEC<DBL3>* > pVal_to, pVal_to2;
	//atomistic meshes input / output
	std::vector< VEC<DBL3>* > pVal_afrom, pVal_ato;

	antiferromagnetic_meshes_present = false;

	//identify all existing magnetic meshes (magnetic computation enabled)
	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled()) {

			if (!(*pSMesh)[idx]->is_atomistic()) {

				//micromagnetic mesh

				pVal_from.push_back(&(dynamic_cast<Mesh*>((*pSMesh)[idx])->M));
				pVal_to.push_back(&(dynamic_cast<Mesh*>((*pSMesh)[idx])->Heff));

				pVal_from2.push_back(&(dynamic_cast<Mesh*>((*pSMesh)[idx])->M2));
				pVal_to2.push_back(&(dynamic_cast<Mesh*>((*pSMesh)[idx])->Heff2));

				if ((*pSMesh)[idx]->GetMeshType() == MESH_ANTIFERROMAGNETIC) antiferromagnetic_meshes_present = true;
			}
			else {

				//atomistic mesh

				pVal_afrom.push_back(&(dynamic_cast<Atom_Mesh*>((*pSMesh)[idx])->M1));
				pVal_ato.push_back(&(dynamic_cast<Atom_Mesh*>((*pSMesh)[idx])->Heff1));
			}
		}
	}

	if (pVal_from.size()) {

		///////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////// ALL FERROMAGNETIC MESHES //////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (!antiferromagnetic_meshes_present) {

			//Initialize the mesh transfer object for convolution on the super-mesh

			//use built-in corrections based on interpolation
			if (!sm_Vals.Initialize_MeshTransfer(pVal_from, pVal_to, MESHTRANSFERTYPE_WEIGHTED)) return error(BERROR_OUTOFMEMORY_CRIT);

			//transfer values from invidual M meshes to sm_Vals - we need this to get number of non-empty cells
			sm_Vals.transfer_in();

			non_empty_cells = sm_Vals.get_nonempty_cells();
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////// AT LEAST ONE ANTIFERROMAGNETIC MESH ///////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		else {

			//Initialize the mesh transfer object for convolution on the super-mesh

			//use built-in corrections based on interpolation
			if (!sm_Vals.Initialize_MeshTransfer_AveragedInputs_DuplicatedOutputs(pVal_from, pVal_from2, pVal_to, pVal_to2, MESHTRANSFERTYPE_WEIGHTED)) return error(BERROR_OUTOFMEMORY_CRIT);

			//transfer values from invidual M meshes to sm_Vals - we need this to get number of non-empty cells
			//NOTE : do not use transfer_in_averaged here as for antiferromagnetic meshes in the ground state this will result in zero values everywhere, looking like there are no non-empty cells
			sm_Vals.transfer_in();

			non_empty_cells = sm_Vals.get_nonempty_cells();
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////// ATOMISTIC MESHES //////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (pVal_afrom.size()) {

		//secondary transfer object for atomistic meshes
		if (!sm_Vals.Initialize_MeshTransfer2(pVal_afrom, pVal_ato, MESHTRANSFERTYPE_WDENSITY, MUB)) return error(BERROR_OUTOFMEMORY_CRIT);

		//transfer in, adding to current values if primary transfer object also present, else clear input
		sm_Vals.transfer2_in(sm_Vals.size_transfer_in() == 0);

		non_empty_cells += sm_Vals.get_nonempty_cells();
	}

	//avoid division by zero
	if (!non_empty_cells) non_empty_cells = 1;

	if (pSMesh->GetEvaluationSpeedup()) {

		std::function<bool(VEC<DBL3>&)> initialize_mesh_transfer = [&](VEC<DBL3>& H) -> bool {

			if (pVal_from.size()) {

				if (!antiferromagnetic_meshes_present) {

					if (!H.Initialize_MeshTransfer(pVal_from, pVal_to, MESHTRANSFERTYPE_WEIGHTED)) return error(BERROR_OUTOFMEMORY_CRIT);
				}

				else {

					if (!H.Initialize_MeshTransfer_AveragedInputs_DuplicatedOutputs(pVal_from, pVal_from2, pVal_to, pVal_to2, MESHTRANSFERTYPE_WEIGHTED)) return error(BERROR_OUTOFMEMORY_CRIT);
				}
			}

			if (pVal_afrom.size()) {

				//secondary transfer object for atomistic meshes
				if (!H.Initialize_MeshTransfer2(pVal_afrom, pVal_ato, MESHTRANSFERTYPE_WDENSITY, MUB)) return error(BERROR_OUTOFMEMORY_CRIT);
			}

			return true;
		};

		EvalSpeedup::Initialize_EvalSpeedup(
			DemagTFunc().SelfDemag_PBC(pSMesh->h_fm, pSMesh->n_fm, Get_PBC()),
			pSMesh->GetEvaluationSpeedup(),
			pSMesh->h_fm, pSMesh->sMeshRect_fm,
			initialize_mesh_transfer);

		EvalSpeedup::Initialize_EvalSpeedup_Mode_Atom(sm_Vals, sm_Vals);
	}

	return error;
}

//called from UpdateConfiguration if using Smesh demag
BError SDemag::UpdateConfiguration_SMesh_Demag(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SDemag));

	//for super-mesh convolution just need a single convolution and sm_Vals to be sized correctly

	//only need to uninitialize if n_fm or h_fm have changed
	if (!CheckDimensions(pSMesh->n_fm, pSMesh->h_fm, demag_pbc_images) || cfgMessage == UPDATECONFIG_DEMAG_CONVCHANGE || cfgMessage == UPDATECONFIG_MESHCHANGE) {

		Uninitialize();
		error = SetDimensions(pSMesh->n_fm, pSMesh->h_fm, true, demag_pbc_images);

		if (!sm_Vals.resize(pSMesh->h_fm, pSMesh->sMeshRect_fm)) return error(BERROR_OUTOFMEMORY_CRIT);
	}

	//Check if h_fm.z divides each magnetic mesh thickness exactly - if not issue a warning to user
	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled()) {

			double start = (*pSMesh)[idx]->meshRect.s.z;
			double end = (*pSMesh)[idx]->meshRect.e.z;

			if (IsNZ(round(start / pSMesh->h_fm.z) - start / pSMesh->h_fm.z)) { error(BWARNING_INCORRECTCELLSIZE); break; }
			if (IsNZ(round(end / pSMesh->h_fm.z) - end / pSMesh->h_fm.z)) { error(BWARNING_INCORRECTCELLSIZE); break; }
		}
	}

	return error;
}

//called from UpdateField if using Smesh demag
void SDemag::UpdateField_SMesh_Demag(void)
{
	if (!EvalSpeedup::Check_if_EvalSpeedup(pSMesh->GetEvaluationSpeedup(), pSMesh->Check_Step_Update())) {

		///////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////// NO SPEEDUP //////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (!antiferromagnetic_meshes_present) {

			//transfer values from invidual M meshes to sm_Vals
			if (sm_Vals.size_transfer_in()) sm_Vals.transfer_in();
			//transfer from atomistic mesh (if any) - clear input only if there was no transfer from micromagnetic meshes, else add in
			if (sm_Vals.size_transfer2_in()) sm_Vals.transfer2_in(sm_Vals.size_transfer_in() == 0);

			//convolution with demag kernels, output overwrites in sm_Vals
			energy = Convolute(sm_Vals, sm_Vals, true);

			//finish off energy value
			energy *= -MU0 / (2 * non_empty_cells);

			//transfer to individual Heff meshes (micromagnetic and atomistc meshes)
			if (sm_Vals.size_transfer_out()) sm_Vals.transfer_out();
			if (sm_Vals.size_transfer2_out()) sm_Vals.transfer2_out();
		}
		else {

			//transfer values from invidual M meshes to sm_Vals
			if (sm_Vals.size_transfer_in()) sm_Vals.transfer_in_averaged();
			//transfer from atomistic mesh (if any) - clear input only if there was no transfer from micromagnetic meshes, else add in
			if (sm_Vals.size_transfer2_in()) sm_Vals.transfer_in(sm_Vals.size_transfer_in() == 0);

			//convolution with demag kernels, output overwrites in sm_Vals
			energy = Convolute(sm_Vals, sm_Vals, true);

			//finish off energy value
			energy *= -MU0 / (2 * non_empty_cells);

			//transfer to individual Heff meshes
			if (sm_Vals.size_transfer_out()) sm_Vals.transfer_out_duplicated();
			if (sm_Vals.size_transfer2_out()) sm_Vals.transfer2_out();
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////// EVAL SPEEDUP /////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	else {

		std::function<void(VEC<DBL3>&)> do_evaluation = [&](VEC<DBL3>& H) -> void {

			if (!antiferromagnetic_meshes_present) {

				//convolution with demag kernels
				energy = Convolute(sm_Vals, H, true);

				//finish off energy value
				energy *= -MU0 / (2 * non_empty_cells);
			}
			else {

				//convolution with demag kernels, output overwrites in sm_Vals
				energy = Convolute(sm_Vals, H, true);

				//finish off energy value
				energy *= -MU0 / (2 * non_empty_cells);
			}
		};

		std::function<void(void)> do_transfer_in = [&](void) -> void {

			if (!antiferromagnetic_meshes_present) {

				//transfer values from invidual M meshes to sm_Vals
				if (sm_Vals.size_transfer_in()) sm_Vals.transfer_in();
				//transfer from atomistic mesh (if any) - clear input only if there was no transfer from micromagnetic meshes, else add in
				if (sm_Vals.size_transfer2_in()) sm_Vals.transfer2_in(sm_Vals.size_transfer_in() == 0);
			}
			else {

				//transfer values from invidual M meshes to sm_Vals
				if (sm_Vals.size_transfer_in()) sm_Vals.transfer_in_averaged();
				//transfer from atomistic mesh (if any) - clear input only if there was no transfer from micromagnetic meshes, else add in
				if (sm_Vals.size_transfer2_in()) sm_Vals.transfer_in(sm_Vals.size_transfer_in() == 0);
			}
		};

		std::function<void(VEC<DBL3>&)> do_transfer_out = [&](VEC<DBL3>& H) -> void {

			if (!antiferromagnetic_meshes_present) {

				//transfer to individual Heff meshes (micromagnetic and atomistc meshes)
				if (sm_Vals.size_transfer_out()) H.transfer_out();
				if (sm_Vals.size_transfer2_out()) H.transfer2_out();
			}
			else {

				//transfer to individual Heff meshes
				if (sm_Vals.size_transfer_out()) H.transfer_out_duplicated();
				if (sm_Vals.size_transfer2_out()) H.transfer2_out();
			}
		};

		EvalSpeedup::UpdateField_EvalSpeedup(
			pSMesh->GetEvaluationSpeedup(), pSMesh->Check_Step_Update(),
			pSMesh->Get_EvalStep_Time(),
			do_evaluation,
			do_transfer_in, do_transfer_out);
	}
}

#endif