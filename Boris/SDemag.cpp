#include "stdafx.h"
#include "SDemag.h"

#ifdef MODULE_COMPILATION_SDEMAG

#include "SuperMesh.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////

SDemag::SDemag(SuperMesh *pSMesh_) :
	Modules(),
	Convolution<SDemag, DemagKernel>(),
	ProgramStateNames(this, 
		{ 
			VINFO(use_multilayered_convolution), 
			VINFO(n_common), 
			VINFO(use_default_n), VINFO(force_2d_convolution), 
			VINFO(demag_pbc_images) }, {}),
	EvalSpeedup()
{
	pSMesh = pSMesh_;

	error_on_create = UpdateConfiguration(UPDATECONFIG_FORCEUPDATE);

	//-------------------------- Is CUDA currently enabled?

	//If cuda is enabled we also need to make the cuda module version
	if (pSMesh->cudaEnabled) {

		if (!error_on_create) error_on_create = SwitchCUDAState(true);
	}
}

SDemag::~SDemag()
{
	//when deleting the SDemag module any pbc settings should no longer take effect in all meshes
	//thus must clear pbc flags in all M

	demag_pbc_images = INT3();
	Set_Magnetic_PBC();

	//RAII : SDemag_Demag modules were created in the constructor, so delete them here in any remaining magnetic meshes 
	//(some could have been deleted already if any magnetic mesh was deleted in the mean-time)
	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled()) {

			//Delete any SDemag_Demag modules in this mesh
			(*pSMesh)[idx]->DelModule(MOD_SDEMAG_DEMAG);
		}
	}
}

//------------------ Auxiliary

//set default value for n_common : largest value from all (anti)ferromagnetic meshes
void SDemag::set_default_n_common(void)
{
	SZ3 n_common_old = n_common;

	n_common = SZ3(1);

	bool meshes_2d = true;

	//are all the meshes 2d?
	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled() && !(*pSMesh)[idx]->Get_Demag_Exclusion()) {

			if ((*pSMesh)[idx]->n.z != 1) {

				meshes_2d = false;
				break;
			}
		}
	}

	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled() && !(*pSMesh)[idx]->Get_Demag_Exclusion()) {

			if (n_common.x < (*pSMesh)[idx]->n.x) n_common.x = (*pSMesh)[idx]->n.x;
			if (n_common.y < (*pSMesh)[idx]->n.y) n_common.y = (*pSMesh)[idx]->n.y;
			if (n_common.z < (*pSMesh)[idx]->n.z) n_common.z = (*pSMesh)[idx]->n.z;
		}
	}

	if (meshes_2d || force_2d_convolution) {

		//all 2D meshes, or 2D layered meshes, forced or otherwise : common n.z must be 1, thus enabling exact computation for layers with arbitrary thicknesses.
		n_common.z = 1;
	}

	//uninitialize only if n_common has changed
	if (n_common_old != n_common) Uninitialize();
}

//adjust Rect_collection so the rectangles are matching for multi-layered convolution. n_common should be calculated already.
void SDemag::set_Rect_collection(void)
{
	//first get raw Rect_collection
	Rect_collection.clear();

	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		if ((*pSMesh)[idx]->MComputation_Enabled() && !(*pSMesh)[idx]->Get_Demag_Exclusion()) {

			for (int layer_idx = 0; layer_idx < (*pSMesh)[idx]->GetNumModules(MOD_SDEMAG_DEMAG); layer_idx++) {

				if (force_2d_convolution == 2) {

					//if in 2d layering mode, then the rectangle set in the rect collection must be that of the respective layer, not the entire mesh
					Rect_collection.push_back((*pSMesh)[idx]->GetMeshRect().get_zlayer(layer_idx, (*pSMesh)[idx]->GetMeshCellsize().z));
				}
				else {

					Rect_collection.push_back((*pSMesh)[idx]->GetMeshRect());
				}
			}
		}
	}

	//it's possible there are no modules set for multilayered convolution (e.g. all set to be excluded)
	if (!Rect_collection.size()) return;

	//now adjust it
	DBL3 max_sizes;

	double min_x = Rect_collection[0].s.x;
	double min_y = Rect_collection[0].s.y;
	double min_z = Rect_collection[0].s.z;

	//adjust rectangles so they have matching dimensions for multi-layered convolution
	
	//first find maximum size in each dimension
	//also find smallest starting coordinates along each axis
	for (int idx = 0; idx < Rect_collection.size(); idx++) {

		if (max_sizes.x < Rect_collection[idx].size().x) max_sizes.x = Rect_collection[idx].size().x;
		if (max_sizes.y < Rect_collection[idx].size().y) max_sizes.y = Rect_collection[idx].size().y;
		if (max_sizes.z < Rect_collection[idx].size().z) max_sizes.z = Rect_collection[idx].size().z;

		if (Rect_collection[idx].s.x < min_x) min_x = Rect_collection[idx].s.x;
		if (Rect_collection[idx].s.y < min_y) min_y = Rect_collection[idx].s.y;
		if (Rect_collection[idx].s.z < min_z) min_z = Rect_collection[idx].s.z;
	}

	//now enlarge rectangles so they all have sizes max_sizes (except in 2D where they keep their thickness)
	//enlarge them by setting the starting points as close as possible to the smallest starting points found above, along each axis
	//ideally they all start at the same point, thus making multi-layered convolution most efficient
	
	for (int idx = 0; idx < Rect_collection.size(); idx++) {
		
		if (Rect_collection[idx].e.x - max_sizes.x < min_x) {

			Rect_collection[idx].s.x = min_x;
			Rect_collection[idx].e.x = Rect_collection[idx].s.x + max_sizes.x;
		}
		else Rect_collection[idx].s.x = Rect_collection[idx].e.x - max_sizes.x;

		if (Rect_collection[idx].e.y - max_sizes.y < min_y) {

			Rect_collection[idx].s.y = min_y;
			Rect_collection[idx].e.y = Rect_collection[idx].s.y + max_sizes.y;
		}
		else Rect_collection[idx].s.y = Rect_collection[idx].e.y - max_sizes.y;

		if (n_common.z != 1) {

			//3D convolution so also set the z sizes
			if (Rect_collection[idx].e.z - max_sizes.z < min_z) {

				Rect_collection[idx].s.z = min_z;
				Rect_collection[idx].e.z = Rect_collection[idx].s.z + max_sizes.z;
			}
			else Rect_collection[idx].s.z = Rect_collection[idx].e.z - max_sizes.z;
		}
	}
}

//get maximum cellsize for multi-layered convolution (use it to normalize dimensions)
double SDemag::get_maximum_cellsize(void)
{
	double h_max = 0.0;

	for (int idx = 0; idx < pSDemag_Demag.size(); idx++) {

		if (h_max < pSDemag_Demag[idx]->h.maxdim()) h_max = pSDemag_Demag[idx]->h.maxdim();
	}

	return h_max;
}

//get convolution rectangle for the given SDemag_Demag module (remember this might not be the rectangle of M in that mesh, but an adjusted rectangle to make the convolution work)
Rect SDemag::get_convolution_rect(SDemag_Demag* demag_demag)
{
	for (int idx = 0; idx < pSDemag_Demag.size(); idx++) {

		if (pSDemag_Demag[idx] == demag_demag) return Rect_collection[idx];
	}

	//empty rect : something not right
	return Rect();
}

//set n_common for multi-layered convolution
BError SDemag::Set_n_common(SZ3 n)
{
	BError error(CLASS_STR(SDemag));

	n_common = n;

	use_multilayered_convolution = true;
	use_default_n = false;

	if (n_common.z == 1) force_2d_convolution = 1;
	else force_2d_convolution = 0;

	//first clear all currently set SDemag_Demag modules - these will be created as required through the UpdateConfiguration() method below.
	//we need to do this since it's possible force_2d_convolution mode was changed e.g. from 2 to 1.
	Destroy_SDemag_Demag_Modules();

	error = UpdateConfiguration(UPDATECONFIG_DEMAG_CONVCHANGE);

	UninitializeAll();

	return error;
}

//set status for use_default_n
BError SDemag::Set_Default_n_status(bool status)
{
	BError error(CLASS_STR(SDemag));

	use_multilayered_convolution = true;
	use_default_n = status;

	if (use_default_n) set_default_n_common();

	error = UpdateConfiguration(UPDATECONFIG_DEMAG_CONVCHANGE);

	UninitializeAll();

	return error;
}

//Set PBC images for supermesh demag
BError SDemag::Set_PBC(INT3 demag_pbc_images_)
{
	BError error(__FUNCTION__);

	demag_pbc_images = demag_pbc_images_;

	error = Set_Magnetic_PBC();

	UninitializeAll();

	//update will be needed if pbc settings have changed
	error = UpdateConfiguration(UPDATECONFIG_DEMAG_CONVCHANGE);

	return error;
}

//Set PBC settings for M in all meshes
BError SDemag::Set_Magnetic_PBC(void)
{
	BError error(__FUNCTION__);

	//set pbc conditions in all M : if any are zero then pbc is disabled in that dimension

	for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

		//Set PBC irrespective of demag exclusion setting
		if ((*pSMesh)[idx]->MComputation_Enabled()) {

			(*pSMesh)[idx]->Set_Magnetic_PBC(demag_pbc_images);
		}
	}

	return error;
}

//-------------------Abstract base class method implementations

BError SDemag::Initialize(void)
{
	BError error(CLASS_STR(SDemag));

	//FFT Kernels are not so quick to calculate - if already initialized then we are guaranteed they are correct
	if (!initialized) {

		///////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////// SUPERMESH CONVOLUTION ////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		//calculate kernels for super-mesh convolution
		if (!use_multilayered_convolution) {

			error = Initialize_SMesh_Demag();
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////// MULTILAYERED CONVOLUTION //////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		else {

			error = Initialize_MConv_Demag();
		}

		//initialized ok.
		if (!error) initialized = true;
	}

	//calculate total_nonempty_volume from all meshes participating in convolution
	if (use_multilayered_convolution) {

		total_nonempty_volume = 0.0;

		for (int idx = 0; idx < (int)pSMesh->pMesh.size(); idx++) {

			if ((*pSMesh)[idx]->MComputation_Enabled() && !(*pSMesh)[idx]->Get_Demag_Exclusion()) {

				total_nonempty_volume += pSMesh->pMesh[idx]->Get_NonEmpty_Magnetic_Volume();
			}
		}
	}

	EvalSpeedup::num_Hdemag_saved = 0;

	return error;
}

void SDemag::UninitializeAll(void)
{
	Uninitialize();

	//be careful when using UninitializeAll : pSDemag_Demag must be up to date
	if (use_multilayered_convolution) {

		//Must have called UpdateConfiguration before, which makes sure pSDemag_Demag vector is correct
		for (int idx = 0; idx < pSDemag_Demag.size(); idx++) {

			pSDemag_Demag[idx]->Uninitialize();
		}
	}

#if COMPILECUDA == 1
	if (pModuleCUDA) {

		//Must have called UpdateConfiguration before - in turn it will have called the CUDA version of UpdateConfiguration, which makes sure the pSDemagCUDA_Demag vector is correct
		dynamic_cast<SDemagCUDA*>(pModuleCUDA)->UninitializeAll();
	}
#endif
}

BError SDemag::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SDemag));

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_DEMAG_CONVCHANGE, UPDATECONFIG_SMESH_CELLSIZE, UPDATECONFIG_MESHCHANGE, UPDATECONFIG_MESHADDED, UPDATECONFIG_MESHDELETED, UPDATECONFIG_MODULEADDED, UPDATECONFIG_MODULEDELETED)) {

		///////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////// SUPERMESH CONVOLUTION ////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (!use_multilayered_convolution) {

			//don't need memory allocated for multi-layered convolution
			Destroy_SDemag_Demag_Modules();

			error = UpdateConfiguration_SMesh_Demag(cfgMessage);
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////// MULTILAYERED CONVOLUTION //////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		else {

			error = UpdateConfiguration_MConv_Demag(cfgMessage);
		}

		//if a new mesh has been added we must also set any possible pbc conditions for M
		error = Set_Magnetic_PBC();

		EvalSpeedup::UpdateConfiguration_EvalSpeedup();
	}

	//------------------------ CUDA UpdateConfiguration if set

#if COMPILECUDA == 1
	if (pModuleCUDA) {

		if (!error) error = pModuleCUDA->UpdateConfiguration(cfgMessage);
	}
#endif

	//important this is at the end, so the CUDA version of UpdateConfiguration is executed before
	if (!initialized) UninitializeAll();

	return error;
}

BError SDemag::MakeCUDAModule(void)
{
	BError error(CLASS_STR(SDemag));

#if COMPILECUDA == 1

	pModuleCUDA = new SDemagCUDA(pSMesh, this);
	error = pModuleCUDA->Error_On_Create();

	Uninitialize();

#endif

	return error;
}

double SDemag::UpdateField(void)
{
	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////// SUPERMESH CONVOLUTION ////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (!use_multilayered_convolution) {

		UpdateField_SMesh_Demag();
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////// MULTILAYERED CONVOLUTION //////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	else {

		UpdateField_MConv_Demag();
	}

	return energy;
}

#endif