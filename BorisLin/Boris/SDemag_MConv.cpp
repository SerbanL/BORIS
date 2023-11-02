#include "stdafx.h"
#include "SDemag.h"

#ifdef MODULE_COMPILATION_SDEMAG

#include "SuperMesh.h"

//------------------ Helpers for multi-layered convolution control

//when SDemag created, it needs to add one SDemag_Demag module to each (anti)ferromagnetic mesh (or multiple if the 2D layering option is enabled).
BError SDemag::Create_SDemag_Demag_Modules(void)
{
	BError error(CLASS_STR(SDemag));

	//identify all existing magnetic meshes (magnetic computation enabled)
	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled() && !(*pSMesh)[idx]->Get_Demag_Exclusion()) {

			if (force_2d_convolution < 2) {

				//not 2d layered convolution, so there can only be 1 SDemag_Demag module in each layer

				//Make sure each (anti)ferromagnetic mesh has one SDemag_Demag module added
				if (!(*pSMesh)[idx]->IsModuleSet(MOD_SDEMAG_DEMAG)) {

					error = (*pSMesh)[idx]->AddModule(MOD_SDEMAG_DEMAG);
				}

				(*pSMesh)[idx]->CallModuleMethod(&SDemag_Demag::Set_SDemag_Pointer, this);
			}
			else {

				//2d layered convolution - in each mesh need Mesh::n.z SDemag_Demag modules - one for each layer.
				//here we must ensure we have at least n.z such modules - if there are more, they will be deleted through UpdateConfiguration method in the respective SDemag_Demag modules.

				//number of layers required
				int num_layers = (*pSMesh)[idx]->n.z;

				while ((*pSMesh)[idx]->GetNumModules(MOD_SDEMAG_DEMAG) < num_layers && !error) {

					//keep adding SDemag_Demag modules until we have enough to cover all layers
					error = (*pSMesh)[idx]->AddModule(MOD_SDEMAG_DEMAG, true);
				}

				//set SDemag pointer in all SDemag_Demag modules
				(*pSMesh)[idx]->CallAllModulesMethod(&SDemag_Demag::Set_SDemag_Pointer, this);

				//switch on 2D layering in all SDemag_Demag modules in this mesh, making sure each one knows exactly what layer it is
				(*pSMesh)[idx]->CallAllModulesMethod(&SDemag_Demag::Set_2D_Layering);
			}
		}
	}

	return error;
}

//delete all SDemag_Demag modules - these are only created by SDemag
void SDemag::Destroy_SDemag_Demag_Modules(void)
{
	//identify all existing magnetic meshes (magnetic computation enabled)
	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled()) {

			//Delete any SDemag_Demag modules in this mesh
			(*pSMesh)[idx]->DelModule(MOD_SDEMAG_DEMAG);

			//Also delete any Demag modules as these should not be present when we have the SDemag module enabled
			//the exception is if the mesh is dormant (e.g. Track shifting algorithm with demag / sdemag combination)
			if (!(*pSMesh)[idx]->Is_Dormant()) (*pSMesh)[idx]->DelModule(MOD_DEMAG);
		}
	}

	FFT_Spaces_Input.clear();
	Rect_collection.clear();
	kernel_collection.clear();
	pSDemag_Demag.clear();
}

//make sure the pSDemag_Demag list is up to date : if any mismatches found return false
bool SDemag::Update_SDemag_Demag_List(void)
{
	std::vector<SDemag_Demag*> pSDemag_Demag_;

	//also make sure the FFT spaces and rectangles list is correct -> rebuild it
	FFT_Spaces_Input.clear();
	kernel_collection.clear();

	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled() && !(*pSMesh)[idx]->Get_Demag_Exclusion()) {

			for (int layer_idx = 0; layer_idx < (*pSMesh)[idx]->GetNumModules(MOD_SDEMAG_DEMAG); layer_idx++) {

				SDemag_Demag* pSDemag_Demag_Module = dynamic_cast<SDemag_Demag*>((*pSMesh)[idx]->GetModule(MOD_SDEMAG_DEMAG, layer_idx));

				pSDemag_Demag_.push_back(pSDemag_Demag_Module);

				//build the fft spaces, rectangles list, and demag kernels
				FFT_Spaces_Input.push_back(pSDemag_Demag_.back()->Get_Input_Scratch_Space());
				kernel_collection.push_back(dynamic_cast<DemagKernelCollection*>(pSDemag_Demag_.back()));
			}
		}
	}

	//before transferring pSDemag_Demag_ to pSDemag_Demag, check no changes have occured in order to determine the return value
	if (pSDemag_Demag_.size() == pSDemag_Demag.size()) {

		for (int idx = 0; idx < pSDemag_Demag.size(); idx++) {

			if (pSDemag_Demag[idx] != pSDemag_Demag_[idx]) {

				pSDemag_Demag = pSDemag_Demag_;
				return false;
			}
		}

		return true;
	}
	else {

		pSDemag_Demag = pSDemag_Demag_;
		return false;
	}
}

//called from Initialize if using multiconvolution demag
BError SDemag::Initialize_MConv_Demag(void)
{
	BError error(CLASS_STR(SDemag));

	//in multi-layered convolution mode must make sure all convolution sizes are set correctly, and rect collections also set
	//SDemag_Demag modules are initialized before SDemag, so they must check if SDemag is not initialized, in which case must call this
	//This will happen in the first SDemag_Demag module to initialize, so after that everything is set correctly to calculate kernels

	//update common discretisation if needed
	if (use_default_n) set_default_n_common();

	//make sure Rect_collection is correct
	set_Rect_collection();

	double h_max = get_maximum_cellsize();

	for (int idx = 0; idx < pSDemag_Demag.size(); idx++) {

		//h_convolution may differ from h_common in 2D mode
		DBL3 h_convolution = Rect_collection[idx] / n_common;

		if (!pSDemag_Demag[idx]->CheckDimensions(n_common, h_convolution, demag_pbc_images)) {

			//set convolution dimensions using the common discretisation
			//kernel collection must be used without multiplcation embedding. Calling this also sets full sizes for S and S2 scratch spaces.
			error = pSDemag_Demag[idx]->SetDimensions(n_common, h_convolution, false, demag_pbc_images);

			if (error) return error;
		}

		//set all rect collections
		error = pSDemag_Demag[idx]->Set_Rect_Collection(Rect_collection, Rect_collection[idx], h_max, idx);
		if (error) return error;
	}

	//now everything is set correctly, ready to calculate demag kernel collections

	return error;
}

//called from UpdateConfiguration if using multiconvolution demag
BError SDemag::UpdateConfiguration_MConv_Demag(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SDemag));

	//don't need memory allocated for supermesh demag
	sm_Vals.clear();
	SetDimensions(SZ3(1), DBL3());

	if (use_default_n) set_default_n_common();

	//for multi-layered convolution need a convolution object for each mesh (make modules) - or multiple if the 2D layering option is enabled.

	//new (anti)ferromagnetic meshes could have been added
	Create_SDemag_Demag_Modules();

	//make sure the list of SDemag_Demag  modules is up to date. If it was not, must re-initialize.
	if (!Update_SDemag_Demag_List()) Uninitialize();

	//If SDemag or any SDemag_Demag modules are uninitialized, then Uninitialize all SDemag_Demag modules
	for (int idx = 0; idx < pSDemag_Demag.size(); idx++) {

		initialized &= pSDemag_Demag[idx]->IsInitialized();
	}

	return error;
}

//called from UpdateField if using multiconvolution demag
void SDemag::UpdateField_MConv_Demag(void)
{
	///////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////// NO SPEEDUP - MULTILAYERED /////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (!pSMesh->GetEvaluationSpeedup() || (num_Hdemag_saved < pSMesh->GetEvaluationSpeedup() && !pSMesh->Check_Step_Update())) {

		//don't use evaluation speedup, so no need to use Hdemag in SDemag_Demag modules (this won't have memory allocated anyway)

		energy = 0;

		//Forward FFT for all ferromagnetic meshes
		for (int idx = 0; idx < pSDemag_Demag.size(); idx++) {

			///////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////// ANTIFERROMAGNETIC MESH ///////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////////////////////

			if (pSDemag_Demag[idx]->pMeshBase->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

				if (pSDemag_Demag[idx]->do_transfer) {

					//transfer from M to common meshing
					pSDemag_Demag[idx]->transfer.transfer_in_averaged();

					//do forward FFT
					pSDemag_Demag[idx]->ForwardFFT(pSDemag_Demag[idx]->transfer);
				}
				else {

					pSDemag_Demag[idx]->ForwardFFT_AveragedInputs(pSDemag_Demag[idx]->pMesh->M, pSDemag_Demag[idx]->pMesh->M2);
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			///////////////////////////////////// OTHER MAGNETIC MESH /////////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////////////////////

			else {

				if (pSDemag_Demag[idx]->do_transfer) {

					//transfer from M to common meshing
					pSDemag_Demag[idx]->transfer.transfer_in();

					//do forward FFT
					pSDemag_Demag[idx]->ForwardFFT(pSDemag_Demag[idx]->transfer);
				}
				else {

					//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh
					pSDemag_Demag[idx]->ForwardFFT(pSDemag_Demag[idx]->pMesh->M);
				}
			}
		}

		//Kernel multiplications for multiple inputs. Reverse loop ordering improves cache use at both ends.
		for (int idx = pSDemag_Demag.size() - 1; idx >= 0; idx--) {

			pSDemag_Demag[idx]->KernelMultiplication_MultipleInputs(FFT_Spaces_Input);
		}

		//Inverse FFT
		for (int idx = 0; idx < pSDemag_Demag.size(); idx++) {

			///////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////// ANTIFERROMAGNETIC MESH ///////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////////////////////

			if (pSDemag_Demag[idx]->pMeshBase->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

				if (pSDemag_Demag[idx]->do_transfer) {

					//do inverse FFT and accumulate energy
					if (pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

						pSDemag_Demag[idx]->energy += (-MU0 / 2) * (pSDemag_Demag[idx]->InverseFFT(
							pSDemag_Demag[idx]->transfer, pSDemag_Demag[idx]->transfer, true, &pSDemag_Demag[idx]->transfer_Module_Heff, &pSDemag_Demag[idx]->transfer_Module_energy) / pSDemag_Demag[idx]->non_empty_cells);

						pSDemag_Demag[idx]->transfer_Module_Heff.transfer_out();
						pSDemag_Demag[idx]->transfer_Module_energy.transfer_out();
					}
					else pSDemag_Demag[idx]->energy += (-MU0 / 2) * (pSDemag_Demag[idx]->InverseFFT(pSDemag_Demag[idx]->transfer, pSDemag_Demag[idx]->transfer, true) / pSDemag_Demag[idx]->non_empty_cells);

					//transfer to Heff in each mesh
					pSDemag_Demag[idx]->transfer.transfer_out_duplicated();
				}
				else {

					//do inverse FFT and accumulate energy
					if (pSDemag_Demag[idx]->Module_Heff.linear_size())
						pSDemag_Demag[idx]->energy += (-MU0 / 2) * (pSDemag_Demag[idx]->InverseFFT_AveragedInputs_DuplicatedOutputs(
							pSDemag_Demag[idx]->pMesh->M, pSDemag_Demag[idx]->pMesh->M2,
							pSDemag_Demag[idx]->pMesh->Heff, pSDemag_Demag[idx]->pMesh->Heff2, false, &pSDemag_Demag[idx]->Module_Heff, &pSDemag_Demag[idx]->Module_energy) / pSDemag_Demag[idx]->non_empty_cells);
					else
						pSDemag_Demag[idx]->energy += (-MU0 / 2) * (pSDemag_Demag[idx]->InverseFFT_AveragedInputs_DuplicatedOutputs(
							pSDemag_Demag[idx]->pMesh->M, pSDemag_Demag[idx]->pMesh->M2,
							pSDemag_Demag[idx]->pMesh->Heff, pSDemag_Demag[idx]->pMesh->Heff2, false) / pSDemag_Demag[idx]->non_empty_cells);
				}
			}

			///////////////////////////////////////////////////////////////////////////////////////////////
			///////////////////////////////////// OTHER MAGNETIC MESH /////////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////////////////////

			else {

				if (pSDemag_Demag[idx]->do_transfer) {

					//do inverse FFT and accumulate energy
					if (pSDemag_Demag[idx]->transfer_Module_Heff.linear_size()) {

						pSDemag_Demag[idx]->energy += (-MU0 / 2) * (pSDemag_Demag[idx]->InverseFFT(
							pSDemag_Demag[idx]->transfer, pSDemag_Demag[idx]->transfer, true, &pSDemag_Demag[idx]->transfer_Module_Heff, &pSDemag_Demag[idx]->transfer_Module_energy) / pSDemag_Demag[idx]->non_empty_cells);

						pSDemag_Demag[idx]->transfer_Module_Heff.transfer_out();
						pSDemag_Demag[idx]->transfer_Module_energy.transfer_out();
					}
					else pSDemag_Demag[idx]->energy += (-MU0 / 2) * (pSDemag_Demag[idx]->InverseFFT(pSDemag_Demag[idx]->transfer, pSDemag_Demag[idx]->transfer, true) / pSDemag_Demag[idx]->non_empty_cells);

					//transfer to Heff in each mesh
					pSDemag_Demag[idx]->transfer.transfer_out();
				}
				else {

					//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh

					//do inverse FFT and accumulate energy
					if (pSDemag_Demag[idx]->Module_Heff.linear_size()) {

						pSDemag_Demag[idx]->energy += (-MU0 / 2) * (pSDemag_Demag[idx]->InverseFFT(
							pSDemag_Demag[idx]->pMesh->M, pSDemag_Demag[idx]->pMesh->Heff, false, &pSDemag_Demag[idx]->Module_Heff, &pSDemag_Demag[idx]->Module_energy) / pSDemag_Demag[idx]->non_empty_cells);
					}
					else {

						pSDemag_Demag[idx]->energy += (-MU0 / 2) * (pSDemag_Demag[idx]->InverseFFT(pSDemag_Demag[idx]->pMesh->M, pSDemag_Demag[idx]->pMesh->Heff, false) / pSDemag_Demag[idx]->non_empty_cells);
					}
				}
			}

			//build total energy
			energy += pSDemag_Demag[idx]->energy * pSDemag_Demag[idx]->energy_density_weight;
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////// EVAL SPEEDUP - MULTILAYERED ////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	else {

		//update if required by ODE solver or if we don't have enough previous evaluations saved to extrapolate
		if (pSMesh->Check_Step_Update() || num_Hdemag_saved < pSMesh->GetEvaluationSpeedup()) {

			energy = 0;

			for (int idx_mesh = 0; idx_mesh < pSDemag_Demag.size(); idx_mesh++) {

				VEC<DBL3>* pHdemag;

				if (num_Hdemag_saved < pSMesh->GetEvaluationSpeedup()) {

					//don't have enough evaluations, so save next one
					switch (num_Hdemag_saved)
					{
					case 0:
						pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag;
						if (idx_mesh == pSDemag_Demag.size() - 1) time_demag1 = pSMesh->Get_EvalStep_Time();
						break;
					case 1:
						pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag2;
						if (idx_mesh == pSDemag_Demag.size() - 1) time_demag2 = pSMesh->Get_EvalStep_Time();
						break;
					case 2:
						pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag3;
						if (idx_mesh == pSDemag_Demag.size() - 1) time_demag3 = pSMesh->Get_EvalStep_Time();
						break;
					case 3:
						pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag4;
						if (idx_mesh == pSDemag_Demag.size() - 1) time_demag4 = pSMesh->Get_EvalStep_Time();
						break;
					case 4:
						pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag5;
						if (idx_mesh == pSDemag_Demag.size() - 1) time_demag5 = pSMesh->Get_EvalStep_Time();
						break;
					case 5:
						pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag6;
						if (idx_mesh == pSDemag_Demag.size() - 1) time_demag6 = pSMesh->Get_EvalStep_Time();
						break;
					}

					if (idx_mesh == pSDemag_Demag.size() - 1) num_Hdemag_saved++;
				}
				else {

					//have enough evaluations saved, so just cycle between them now

					//QUINTIC
					if (pSMesh->GetEvaluationSpeedup() == 6) {

						//1, 2, 3, 4, 5, 6 -> next is 1
						if (time_demag6 > time_demag5 && time_demag5 > time_demag4 && time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag1 = pSMesh->Get_EvalStep_Time();
						}
						//2, 3, 4, 5, 6, 1 -> next is 2
						else if (time_demag1 > time_demag2) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag2;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag2 = pSMesh->Get_EvalStep_Time();
						}
						//3, 4, 5, 6, 1, 2 -> next is 3
						else if (time_demag2 > time_demag3) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag3;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag3 = pSMesh->Get_EvalStep_Time();
						}
						//4, 5, 6, 1, 2, 3 -> next is 4
						else if (time_demag3 > time_demag4) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag4;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag4 = pSMesh->Get_EvalStep_Time();
						}
						//5, 6, 1, 2, 3, 4 -> next is 5
						else if (time_demag4 > time_demag5) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag5;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag5 = pSMesh->Get_EvalStep_Time();
						}
						else {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag6;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag6 = pSMesh->Get_EvalStep_Time();
						}
					}
					//QUARTIC
					else if (pSMesh->GetEvaluationSpeedup() == 5) {

						//1, 2, 3, 4, 5 -> next is 1
						if (time_demag5 > time_demag4 && time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag1 = pSMesh->Get_EvalStep_Time();
						}
						//2, 3, 4, 5, 1 -> next is 2
						else if (time_demag1 > time_demag2) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag2;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag2 = pSMesh->Get_EvalStep_Time();
						}
						//3, 4, 5, 1, 2 -> next is 3
						else if (time_demag2 > time_demag3) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag3;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag3 = pSMesh->Get_EvalStep_Time();
						}
						//4, 5, 1, 2, 3 -> next is 4
						else if (time_demag3 > time_demag4) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag4;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag4 = pSMesh->Get_EvalStep_Time();
						}
						else {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag5;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag5 = pSMesh->Get_EvalStep_Time();
						}
					}
					//CUBIC
					else if (pSMesh->GetEvaluationSpeedup() == 4) {

						//1, 2, 3, 4 -> next is 1
						if (time_demag4 > time_demag3 && time_demag3 > time_demag2 && time_demag2 > time_demag1) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag1 = pSMesh->Get_EvalStep_Time();
						}
						//2, 3, 4, 1 -> next is 2
						else if (time_demag1 > time_demag2) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag2;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag2 = pSMesh->Get_EvalStep_Time();
						}
						//3, 4, 1, 2 -> next is 3
						else if (time_demag2 > time_demag3) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag3;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag3 = pSMesh->Get_EvalStep_Time();
						}
						else {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag4;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag4 = pSMesh->Get_EvalStep_Time();
						}
					}
					//QUADRATIC
					else if (pSMesh->GetEvaluationSpeedup() == 3) {

						//1, 2, 3 -> next is 1
						if (time_demag3 > time_demag2 && time_demag2 > time_demag1) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag1 = pSMesh->Get_EvalStep_Time();
						}
						//2, 3, 1 -> next is 2
						else if (time_demag3 > time_demag2 && time_demag1 > time_demag2) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag2;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag2 = pSMesh->Get_EvalStep_Time();
						}
						//3, 1, 2 -> next is 3, leading to 1, 2, 3 again
						else {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag3;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag3 = pSMesh->Get_EvalStep_Time();
						}
					}
					//LINEAR
					else if (pSMesh->GetEvaluationSpeedup() == 2) {

						//1, 2 -> next is 1
						if (time_demag2 > time_demag1) {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag1 = pSMesh->Get_EvalStep_Time();
						}
						//2, 1 -> next is 2, leading to 1, 2 again
						else {

							pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag2;
							if (idx_mesh == pSDemag_Demag.size() - 1) time_demag2 = pSMesh->Get_EvalStep_Time();
						}
					}
					//STEP
					else {

						pHdemag = &pSDemag_Demag[idx_mesh]->Hdemag;
					}
				}

				//Forward FFT for all ferromagnetic meshes
				for (int idx = 0; idx < pSDemag_Demag.size(); idx++) {

					///////////////////////////////////////////////////////////////////////////////////////////////
					//////////////////////////////////// ANTIFERROMAGNETIC MESH ///////////////////////////////////
					///////////////////////////////////////////////////////////////////////////////////////////////

					if (pSDemag_Demag[idx]->pMeshBase->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						if (pSDemag_Demag[idx]->do_transfer) {

							//transfer from M to common meshing
							pSDemag_Demag[idx]->transfer.transfer_in_averaged();

							//do forward FFT
							pSDemag_Demag[idx]->ForwardFFT(pSDemag_Demag[idx]->transfer);
						}
						else {

							pSDemag_Demag[idx]->ForwardFFT_AveragedInputs(pSDemag_Demag[idx]->pMesh->M, pSDemag_Demag[idx]->pMesh->M2);
						}
					}

					///////////////////////////////////////////////////////////////////////////////////////////////
					///////////////////////////////////// OTHER MAGNETIC MESH /////////////////////////////////////
					///////////////////////////////////////////////////////////////////////////////////////////////

					else {

						if (pSDemag_Demag[idx]->do_transfer) {

							//transfer from M to common meshing
							pSDemag_Demag[idx]->transfer.transfer_in();

							//do forward FFT
							pSDemag_Demag[idx]->ForwardFFT(pSDemag_Demag[idx]->transfer);
						}
						else {

							//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh
							pSDemag_Demag[idx]->ForwardFFT(pSDemag_Demag[idx]->pMesh->M);
						}
					}
				}

				//Kernel multiplications for multiple inputs. Reverse loop ordering improves cache use at both ends.
				for (int idx = pSDemag_Demag.size() - 1; idx >= 0; idx--) {

					pSDemag_Demag[idx]->KernelMultiplication_MultipleInputs(FFT_Spaces_Input);
				}

				//Inverse FFT

				///////////////////////////////////////////////////////////////////////////////////////////////
				//////////////////////////////////// ANTIFERROMAGNETIC MESH ///////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////////////

				if (pSDemag_Demag[idx_mesh]->pMeshBase->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

					if (pSDemag_Demag[idx_mesh]->do_transfer) {

						//do inverse FFT and accumulate energy
						if (pSDemag_Demag[idx_mesh]->transfer_Module_Heff.linear_size()) {

							pSDemag_Demag[idx_mesh]->energy += (-MU0 / 2) * (pSDemag_Demag[idx_mesh]->InverseFFT(
								pSDemag_Demag[idx_mesh]->transfer, *pHdemag, true, &pSDemag_Demag[idx_mesh]->transfer_Module_Heff, &pSDemag_Demag[idx_mesh]->transfer_Module_energy) / pSDemag_Demag[idx_mesh]->non_empty_cells);

							pSDemag_Demag[idx_mesh]->transfer_Module_Heff.transfer_out();
							pSDemag_Demag[idx_mesh]->transfer_Module_energy.transfer_out();
						}
						else {

							pSDemag_Demag[idx_mesh]->energy += (-MU0 / 2) * (pSDemag_Demag[idx_mesh]->InverseFFT(pSDemag_Demag[idx_mesh]->transfer, *pHdemag, true) / pSDemag_Demag[idx_mesh]->non_empty_cells);
						}

						//transfer to Heff in each mesh
						pHdemag->transfer_out_duplicated();

						//remove self demag contribution
#pragma omp parallel for
						for (int idx = 0; idx < pHdemag->linear_size(); idx++) {

							//subtract self demag contribution: we'll add in again for the new magnetization, so it least the self demag is exact
							(*pHdemag)[idx] -= (pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);
						}
					}
					else {

						//do inverse FFT and accumulate energy
						if (pSDemag_Demag[idx_mesh]->Module_Heff.linear_size()) {

							pSDemag_Demag[idx_mesh]->energy += (-MU0 / 2) * (pSDemag_Demag[idx_mesh]->InverseFFT_AveragedInputs(
								pSDemag_Demag[idx_mesh]->pMesh->M, pSDemag_Demag[idx_mesh]->pMesh->M2,
								*pHdemag, true, &pSDemag_Demag[idx_mesh]->Module_Heff, &pSDemag_Demag[idx_mesh]->Module_energy) / pSDemag_Demag[idx_mesh]->non_empty_cells);
						}
						else {

							pSDemag_Demag[idx_mesh]->energy += (-MU0 / 2) * (pSDemag_Demag[idx_mesh]->InverseFFT_AveragedInputs(
								pSDemag_Demag[idx_mesh]->pMesh->M, pSDemag_Demag[idx_mesh]->pMesh->M2,
								*pHdemag, true) / pSDemag_Demag[idx_mesh]->non_empty_cells);
						}

						//add contribution to Heff and Heff2 then remove self demag contribution
#pragma omp parallel for
						for (int idx = 0; idx < pHdemag->linear_size(); idx++) {

							pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] += (*pHdemag)[idx];
							pSDemag_Demag[idx_mesh]->pMesh->Heff2[idx] += (*pHdemag)[idx];
							//subtract self demag contribution: we'll add in again for the new magnetization, so it least the self demag is exact
							(*pHdemag)[idx] -= (pSDemag_Demag[idx_mesh]->selfDemagCoeff & (pSDemag_Demag[idx_mesh]->pMesh->M[idx] + pSDemag_Demag[idx_mesh]->pMesh->M2[idx]) / 2);
						}
					}
				}

				///////////////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////////// OTHER MAGNETIC MESH /////////////////////////////////////
				///////////////////////////////////////////////////////////////////////////////////////////////

				else {

					if (pSDemag_Demag[idx_mesh]->do_transfer) {

						//do inverse FFT and accumulate energy
						if (pSDemag_Demag[idx_mesh]->transfer_Module_Heff.linear_size()) {

							pSDemag_Demag[idx_mesh]->energy += (-MU0 / 2) * (pSDemag_Demag[idx_mesh]->InverseFFT(
								pSDemag_Demag[idx_mesh]->transfer, *pHdemag, true, &pSDemag_Demag[idx_mesh]->transfer_Module_Heff, &pSDemag_Demag[idx_mesh]->transfer_Module_energy) / pSDemag_Demag[idx_mesh]->non_empty_cells);

							pSDemag_Demag[idx_mesh]->transfer_Module_Heff.transfer_out();
							pSDemag_Demag[idx_mesh]->transfer_Module_energy.transfer_out();
						}
						else {

							pSDemag_Demag[idx_mesh]->energy += (-MU0 / 2) * (pSDemag_Demag[idx_mesh]->InverseFFT(pSDemag_Demag[idx_mesh]->transfer, *pHdemag, true) / pSDemag_Demag[idx_mesh]->non_empty_cells);
						}

						//transfer to Heff in each mesh
						pHdemag->transfer_out();

						//remove self demag contribution
#pragma omp parallel for
						for (int idx = 0; idx < pHdemag->linear_size(); idx++) {

							//subtract self demag contribution: we'll add in again for the new magnetization, so it least the self demag is exact
							(*pHdemag)[idx] -= (pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);
						}
					}
					else {

						//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh

						//do inverse FFT and accumulate energy
						if (pSDemag_Demag[idx_mesh]->Module_Heff.linear_size()) {

							pSDemag_Demag[idx_mesh]->energy += (-MU0 / 2) * (pSDemag_Demag[idx_mesh]->InverseFFT(
								pSDemag_Demag[idx_mesh]->pMesh->M, *pHdemag, true, &pSDemag_Demag[idx_mesh]->Module_Heff, &pSDemag_Demag[idx_mesh]->Module_energy) / pSDemag_Demag[idx_mesh]->non_empty_cells);
						}
						else {

							pSDemag_Demag[idx_mesh]->energy += (-MU0 / 2) * (pSDemag_Demag[idx_mesh]->InverseFFT(
								pSDemag_Demag[idx_mesh]->pMesh->M, *pHdemag, true) / pSDemag_Demag[idx_mesh]->non_empty_cells);
						}

						//add contribution to Heff then remove self demag contribution
#pragma omp parallel for
						for (int idx = 0; idx < pHdemag->linear_size(); idx++) {

							pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] += (*pHdemag)[idx];
							//subtract self demag contribution: we'll add in again for the new magnetization, so it least the self demag is exact
							(*pHdemag)[idx] -= (pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->pMesh->M[idx]);
						}
					}
				}

				//build total energy
				energy += pSDemag_Demag[idx_mesh]->energy * pSDemag_Demag[idx_mesh]->energy_density_weight;
			}
		}
		else {

			//not required to update, and we have enough previous evaluations: use previous Hdemag saves to extrapolate for current evaluation

			double a1 = 1.0, a2 = 0.0, a3 = 0.0, a4 = 0.0, a5 = 0.0, a6 = 0.0;
			double time = pSMesh->Get_EvalStep_Time();

			//QUINTIC
			if (pSMesh->GetEvaluationSpeedup() == 6) {

				a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4) * (time_demag1 - time_demag5) * (time_demag1 - time_demag6));
				a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4) * (time_demag2 - time_demag5) * (time_demag2 - time_demag6));
				a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) * (time - time_demag5) * (time - time_demag6) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4) * (time_demag3 - time_demag5) * (time_demag3 - time_demag6));
				a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag5) * (time - time_demag6) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3) * (time_demag4 - time_demag5) * (time_demag4 - time_demag6));
				a5 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag6) / ((time_demag5 - time_demag1) * (time_demag5 - time_demag2) * (time_demag5 - time_demag3) * (time_demag5 - time_demag4) * (time_demag5 - time_demag6));
				a6 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag6 - time_demag1) * (time_demag6 - time_demag2) * (time_demag6 - time_demag3) * (time_demag6 - time_demag4) * (time_demag6 - time_demag5));

				for (int idx_mesh = 0; idx_mesh < pSDemag_Demag.size(); idx_mesh++) {

					//ANTIFERROMAGNETIC
					if (pSDemag_Demag[idx_mesh]->pMeshBase->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 + pSDemag_Demag[idx_mesh]->Hdemag5[idx] * a5 + pSDemag_Demag[idx_mesh]->Hdemag6[idx] * a6 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out_duplicated();
						}
						else {

							//add contribution to Heff and Heff2
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 + pSDemag_Demag[idx_mesh]->Hdemag5[idx] * a5 + pSDemag_Demag[idx_mesh]->Hdemag6[idx] * a6 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & (pSDemag_Demag[idx_mesh]->pMesh->M[idx] + pSDemag_Demag[idx_mesh]->pMesh->M2[idx]) / 2);

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] += Hdemag_value;
								pSDemag_Demag[idx_mesh]->pMesh->Heff2[idx] += Hdemag_value;
							}
						}
					}
					//FERROMAGNETIC
					else {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 + pSDemag_Demag[idx_mesh]->Hdemag5[idx] * a5 + pSDemag_Demag[idx_mesh]->Hdemag6[idx] * a6 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out();
						}
						else {

							//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh

							//add contribution to Heff
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] +=
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 + pSDemag_Demag[idx_mesh]->Hdemag5[idx] * a5 + pSDemag_Demag[idx_mesh]->Hdemag6[idx] * a6 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->pMesh->M[idx]);
							}
						}
					}
				}
			}
			//QUARTIC
			else if (pSMesh->GetEvaluationSpeedup() == 5) {

				a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4) * (time_demag1 - time_demag5));
				a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) * (time - time_demag5) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4) * (time_demag2 - time_demag5));
				a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) * (time - time_demag5) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4) * (time_demag3 - time_demag5));
				a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag5) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3) * (time_demag4 - time_demag5));
				a5 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) * (time - time_demag4) / ((time_demag5 - time_demag1) * (time_demag5 - time_demag2) * (time_demag5 - time_demag3) * (time_demag5 - time_demag4));

				for (int idx_mesh = 0; idx_mesh < pSDemag_Demag.size(); idx_mesh++) {

					//ANTIFERROMAGNETIC
					if (pSDemag_Demag[idx_mesh]->pMeshBase->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 + pSDemag_Demag[idx_mesh]->Hdemag5[idx] * a5 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out_duplicated();
						}
						else {

							//add contribution to Heff and Heff2
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 + pSDemag_Demag[idx_mesh]->Hdemag5[idx] * a5 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & (pSDemag_Demag[idx_mesh]->pMesh->M[idx] + pSDemag_Demag[idx_mesh]->pMesh->M2[idx]) / 2);

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] += Hdemag_value;
								pSDemag_Demag[idx_mesh]->pMesh->Heff2[idx] += Hdemag_value;
							}
						}
					}
					//FERROMAGNETIC
					else {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 + pSDemag_Demag[idx_mesh]->Hdemag5[idx] * a5 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out();
						}
						else {

							//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh

							//add contribution to Heff
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] +=
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 + pSDemag_Demag[idx_mesh]->Hdemag5[idx] * a5 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->pMesh->M[idx]);
							}
						}
					}
				}
			}
			//CUBIC
			else if (pSMesh->GetEvaluationSpeedup() == 4) {

				a1 = (time - time_demag2) * (time - time_demag3) * (time - time_demag4) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3) * (time_demag1 - time_demag4));
				a2 = (time - time_demag1) * (time - time_demag3) * (time - time_demag4) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3) * (time_demag2 - time_demag4));
				a3 = (time - time_demag1) * (time - time_demag2) * (time - time_demag4) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2) * (time_demag3 - time_demag4));
				a4 = (time - time_demag1) * (time - time_demag2) * (time - time_demag3) / ((time_demag4 - time_demag1) * (time_demag4 - time_demag2) * (time_demag4 - time_demag3));

				for (int idx_mesh = 0; idx_mesh < pSDemag_Demag.size(); idx_mesh++) {

					//ANTIFERROMAGNETIC
					if (pSDemag_Demag[idx_mesh]->pMeshBase->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out_duplicated();
						}
						else {

							//add contribution to Heff and Heff2
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & (pSDemag_Demag[idx_mesh]->pMesh->M[idx] + pSDemag_Demag[idx_mesh]->pMesh->M2[idx]) / 2);

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] += Hdemag_value;
								pSDemag_Demag[idx_mesh]->pMesh->Heff2[idx] += Hdemag_value;
							}
						}
					}
					//FERROMAGNETIC
					else {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out();
						}
						else {

							//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh

							//add contribution to Heff
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] +=
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 + pSDemag_Demag[idx_mesh]->Hdemag4[idx] * a4 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->pMesh->M[idx]);
							}
						}
					}
				}
			}
			//QUADRATIC
			else if (pSMesh->GetEvaluationSpeedup() == 3) {

				a1 = (time - time_demag2) * (time - time_demag3) / ((time_demag1 - time_demag2) * (time_demag1 - time_demag3));
				a2 = (time - time_demag1) * (time - time_demag3) / ((time_demag2 - time_demag1) * (time_demag2 - time_demag3));
				a3 = (time - time_demag1) * (time - time_demag2) / ((time_demag3 - time_demag1) * (time_demag3 - time_demag2));

				for (int idx_mesh = 0; idx_mesh < pSDemag_Demag.size(); idx_mesh++) {

					//ANTIFERROMAGNETIC
					if (pSDemag_Demag[idx_mesh]->pMeshBase->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out_duplicated();
						}
						else {

							//add contribution to Heff and Heff2
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & (pSDemag_Demag[idx_mesh]->pMesh->M[idx] + pSDemag_Demag[idx_mesh]->pMesh->M2[idx]) / 2);

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] += Hdemag_value;
								pSDemag_Demag[idx_mesh]->pMesh->Heff2[idx] += Hdemag_value;
							}
						}
					}
					//FERROMAGNETIC
					else {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out();
						}
						else {

							//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh

							//add contribution to Heff
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] +=
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 + pSDemag_Demag[idx_mesh]->Hdemag3[idx] * a3 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->pMesh->M[idx]);
							}
						}
					}
				}
			}
			//LINEAR
			else if (pSMesh->GetEvaluationSpeedup() == 2) {

				a1 = (time - time_demag2) / (time_demag1 - time_demag2);
				a2 = (time - time_demag1) / (time_demag2 - time_demag1);

				for (int idx_mesh = 0; idx_mesh < pSDemag_Demag.size(); idx_mesh++) {

					//ANTIFERROMAGNETIC
					if (pSDemag_Demag[idx_mesh]->pMeshBase->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out_duplicated();
						}
						else {

							//add contribution to Heff and Heff2
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & (pSDemag_Demag[idx_mesh]->pMesh->M[idx] + pSDemag_Demag[idx_mesh]->pMesh->M2[idx]) / 2);

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] += Hdemag_value;
								pSDemag_Demag[idx_mesh]->pMesh->Heff2[idx] += Hdemag_value;
							}
						}
					}
					//FERROMAGNETIC
					else {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out();
						}
						else {

							//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh

							//add contribution to Heff
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] +=
									pSDemag_Demag[idx_mesh]->Hdemag[idx] * a1 + pSDemag_Demag[idx_mesh]->Hdemag2[idx] * a2 +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->pMesh->M[idx]);
							}
						}
					}
				}
			}
			//STEP
			else {

				for (int idx_mesh = 0; idx_mesh < pSDemag_Demag.size(); idx_mesh++) {

					//ANTIFERROMAGNETIC
					if (pSDemag_Demag[idx_mesh]->pMeshBase->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out_duplicated();
						}
						else {

							//add contribution to Heff and Heff2
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & (pSDemag_Demag[idx_mesh]->pMesh->M[idx] + pSDemag_Demag[idx_mesh]->pMesh->M2[idx]) / 2);

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] += Hdemag_value;
								pSDemag_Demag[idx_mesh]->pMesh->Heff2[idx] += Hdemag_value;
							}
						}
					}
					//FERROMAGNETIC
					else {

						if (pSDemag_Demag[idx_mesh]->do_transfer) {

#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								DBL3 Hdemag_value =
									pSDemag_Demag[idx_mesh]->Hdemag[idx] +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->transfer[idx]);

								pSDemag_Demag[idx_mesh]->transfer[idx] = Hdemag_value;
							}

							//transfer to Heff in each mesh
							pSDemag_Demag[idx_mesh]->transfer.transfer_out();
						}
						else {

							//transfer is forced for atomistic meshes, so if no transfer required, this must mean a micromagnetic mesh

							//add contribution to Heff
#pragma omp parallel for
							for (int idx = 0; idx < pSDemag_Demag[idx_mesh]->Hdemag.linear_size(); idx++) {

								pSDemag_Demag[idx_mesh]->pMesh->Heff[idx] +=
									pSDemag_Demag[idx_mesh]->Hdemag[idx] +
									(pSDemag_Demag[idx_mesh]->selfDemagCoeff & pSDemag_Demag[idx_mesh]->pMesh->M[idx]);
							}
						}
					}
				}
			}
		}
	}
}

//-------------------Setters

//change between demag calculation types : super-mesh (status = false) or multilayered (status = true)
BError SDemag::Set_Multilayered_Convolution(bool status)
{
	BError error(CLASS_STR(SDemag));

	use_multilayered_convolution = status;
	error = UpdateConfiguration(UPDATECONFIG_DEMAG_CONVCHANGE);

	if (use_multilayered_convolution && use_default_n) set_default_n_common();

	UninitializeAll();

	return error;
}

//enable multi-layered convolution and force it to 2D for all layers
BError SDemag::Set_2D_Multilayered_Convolution(int status)
{
	BError error(CLASS_STR(SDemag));

	use_multilayered_convolution = true;

	force_2d_convolution = status;

	if (force_2d_convolution) n_common.z = 1;
	else if (use_default_n) set_default_n_common();

	//first clear all currently set SDemag_Demag modules - these will be created as required through the UpdateConfiguration() method below.
	Destroy_SDemag_Demag_Modules();

	error = UpdateConfiguration(UPDATECONFIG_DEMAG_CONVCHANGE);

	UninitializeAll();

	return error;
}

#endif