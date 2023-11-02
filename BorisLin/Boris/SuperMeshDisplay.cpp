#include "stdafx.h"
#include "SuperMesh.h"

//----------------------------------- DISPLAY-ASSOCIATED GET/SET METHODS

std::vector<PhysQ> SuperMesh::FetchOnScreenPhysicalQuantity(double detail_level)
{
	std::vector<PhysQ> physQ;

	bool cudaSupermesh = false;

#if COMPILECUDA == 1
	
	//Commands are executed on newly spawned threads, so if cuda is on and we are not using device 0 (default device) we must switch to required device, otherwise 0 will be used
	if (cudaEnabled && cudaDeviceSelect != 0) {

		int device = 0;
		cudaGetDevice(&device);
		if (device != cudaDeviceSelect) cudaSetDevice(cudaDeviceSelect);
	}
	
	if (pSMeshCUDA) {

		//if super-mesh display quantities are set with CUDA enabled then get them from PSMeshCUDA
		physQ = pSMeshCUDA->FetchOnScreenPhysicalQuantity(detail_level);
		cudaSupermesh = true;
	}
#endif

	//get anything displayed on super-mesh
	if (!cudaSupermesh) {

		switch (displayedPhysicalQuantity) {

		case MESHDISPLAY_NONE:
			break;

		case MESHDISPLAY_SM_DEMAG:

			if (IsSuperMeshModuleSet(MODS_SDEMAG)) {

				physQ.push_back(PhysQ(&(dynamic_cast<SDemag*>(pSMod(MODS_SDEMAG))->GetDemagField()), displayedPhysicalQuantity, (VEC3REP_)vec3rep).set_focus(true, superMeshHandle));
			}
			break;

		case MESHDISPLAY_SM_OERSTED:

			if (IsSuperMeshModuleSet(MODS_OERSTED)) {

				physQ.push_back(PhysQ(&(dynamic_cast<Oersted*>(pSMod(MODS_OERSTED))->GetOerstedField()), displayedPhysicalQuantity, (VEC3REP_)vec3rep).set_focus(true, superMeshHandle));
			}
			break;

		case MESHDISPLAY_SM_STRAYH:

			if (IsSuperMeshModuleSet(MODS_STRAYFIELD)) {

				physQ.push_back(PhysQ(&(dynamic_cast<StrayField*>(pSMod(MODS_STRAYFIELD))->GetStrayField()), displayedPhysicalQuantity, (VEC3REP_)vec3rep).set_focus(true, superMeshHandle));
			}
			break;
		}
	}
	
	bool supermeshDisplay = physQ.size();
	//allow dual display with supermesh display as the foreground - need to check if any individual meshes have a background display enabled.
	if (supermeshDisplay) physQ.back().set_display_props(displayTransparency.i, displayThresholds, displayThresholdTrigger);

	//get anything displayed in individual meshes
	for (int idx = 0; idx < (int)pMesh.size(); idx++) {

		if (pMesh[idx]->IsDisplayBackgroundEnabled()) {

			//get foreground and set required transparency and thresholds : not applicable if supermesh quantity is in the foreground
			if (!supermeshDisplay) physQ.push_back(pMesh[idx]->FetchOnScreenPhysicalQuantity(detail_level).set_focus(pMesh.get_key_from_index(idx) == activeMeshName, pMesh.get_key_from_index(idx)).set_display_props(displayTransparency.i, displayThresholds, displayThresholdTrigger));

			//get background and set required transparency (thresholds do not apply)
			physQ.push_back(pMesh[idx]->FetchOnScreenPhysicalQuantity(detail_level, true).set_focus(false, pMesh.get_key_from_index(idx)).set_transparency(displayTransparency.j));
		}
		else if (!supermeshDisplay) {

			//get quantity and set thresholds (transparency does not apply) : not applicable if supermesh quantity is in the foreground
			physQ.push_back(pMesh[idx]->FetchOnScreenPhysicalQuantity(detail_level).set_focus(pMesh.get_key_from_index(idx) == activeMeshName, pMesh.get_key_from_index(idx)).set_thresholds(displayThresholds, displayThresholdTrigger));
		}
	}
	
	return physQ;
}

//save the quantity currently displayed on screen for named mesh in an ovf2 file using the specified format
BError SuperMesh::SaveOnScreenPhysicalQuantity(std::string meshName, std::string fileName, std::string ovf2_dataType, MESHDISPLAY_ quantity)
{
#if COMPILECUDA == 1
	if (pSMeshCUDA) { return pSMeshCUDA->SaveOnScreenPhysicalQuantity(meshName, fileName, ovf2_dataType, quantity); }
#endif

	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	OVF2 ovf2;

	//If quantity is MESHDISPLAY_NONE (default) then use displayedPhysicalQuantity instead, which could be a supermesh quantity or an individual mesh one.
	//If quantity is not MESHDISPLAY_NONE then get that quantity from the indicated mesh (individual one, or supermesh)
	switch ((quantity == MESHDISPLAY_NONE ? displayedPhysicalQuantity : quantity)) {

	default:
	case MESHDISPLAY_NONE:
		if (contains(meshName)) error = pMesh[meshName]->SaveOnScreenPhysicalQuantity(fileName, ovf2_dataType, quantity);
		break;

	case MESHDISPLAY_SM_DEMAG:

		if (IsSuperMeshModuleSet(MODS_SDEMAG)) {

			error = ovf2.Write_OVF2_VEC(fileName, dynamic_cast<SDemag*>(pSMod(MODS_SDEMAG))->GetDemagField(), ovf2_dataType);
		}
		break;

	case MESHDISPLAY_SM_OERSTED:

		if (IsSuperMeshModuleSet(MODS_OERSTED)) {

			error = ovf2.Write_OVF2_VEC(fileName, dynamic_cast<Oersted*>(pSMod(MODS_OERSTED))->GetOerstedField(), ovf2_dataType);
		}
		break;

	case MESHDISPLAY_SM_STRAYH:

		if (IsSuperMeshModuleSet(MODS_STRAYFIELD)) {

			error = ovf2.Write_OVF2_VEC(fileName, dynamic_cast<StrayField*>(pSMod(MODS_STRAYFIELD))->GetStrayField(), ovf2_dataType);
		}
		break;
	}

	return error;
}

//extract profile from named mesh, from currently display mesh quantity, but reading directly from the quantity
//Displayed mesh quantity can be scalar or a vector; pass in std::vector pointers, then check for nullptr to determine what type is displayed
//if do_average = true then build average and don't return anything, else return just a single-shot profile. If read_average = true then simply read out the internally stored averaged profile by assigning to pointer.
void SuperMesh::GetPhysicalQuantityProfile(
	DBL3 start, DBL3 end, double step, DBL3 stencil, 
	std::vector<DBL3>*& pprofile_dbl3, std::vector<double>*& pprofile_dbl, 
	std::string meshName, bool do_average, bool read_average, MESHDISPLAY_ quantity)
{
#if COMPILECUDA == 1
	if (pSMeshCUDA) {

		//if super-mesh display quantities are set with CUDA enabled then get value from pSMeshCUDA
		return pSMeshCUDA->GetPhysicalQuantityProfile(
			start, end, step, stencil, 
			pprofile_dbl3, pprofile_dbl, 
			meshName, do_average, read_average, quantity);
	}
#endif

	//If quantity is MESHDISPLAY_NONE (default) then use displayedPhysicalQuantity instead, which could be a supermesh quantity or an individual mesh one.
	//If quantity is not MESHDISPLAY_NONE then get that quantity from the indicated mesh (individual one, or supermesh)
	switch ((quantity == MESHDISPLAY_NONE ? displayedPhysicalQuantity : quantity)) {

		////////////////
		//no quantity displayed on the supermesh, so use individual mesh displayed quantities
		////////////////

	default:
	case MESHDISPLAY_NONE:
		if (contains(meshName)) pMesh[meshName]->GetPhysicalQuantityProfile(
			start, end, step, stencil, 
			pprofile_dbl3, pprofile_dbl, 
			do_average, read_average, quantity);
		break;

	////////////////
	//use a quantity displayed on the supermesh
	////////////////

	case MESHDISPLAY_SM_DEMAG:

		if (IsSuperMeshModuleSet(MODS_SDEMAG)) {

			profile_storage_dbl3 = dynamic_cast<SDemag*>(pSMod(MODS_SDEMAG))->GetDemagField().extract_profile(start, end, step, stencil);
			pprofile_dbl3 = &profile_storage_dbl3;
		}
		break;

	case MESHDISPLAY_SM_OERSTED:

		if (IsSuperMeshModuleSet(MODS_OERSTED)) {

			profile_storage_dbl3 = dynamic_cast<Oersted*>(pSMod(MODS_OERSTED))->GetOerstedField().extract_profile(start, end, step, stencil);
			pprofile_dbl3 = &profile_storage_dbl3;
		}
		break;

	case MESHDISPLAY_SM_STRAYH:

		if (IsSuperMeshModuleSet(MODS_STRAYFIELD)) {

			profile_storage_dbl3 = dynamic_cast<StrayField*>(pSMod(MODS_STRAYFIELD))->GetStrayField().extract_profile(start, end, step, stencil);
			pprofile_dbl3 = &profile_storage_dbl3;
		}
		break;
	}
}

//return average value for currently displayed mesh quantity for named mesh in the given relative rectangle
Any SuperMesh::GetAverageDisplayedMeshValue(std::string meshName, Rect rel_rect, std::vector<MeshShape> shapes, MESHDISPLAY_ quantity)
{
	if (!contains(meshName) && meshName != superMeshHandle) return Any();

#if COMPILECUDA == 1
	if (pSMeshCUDA) {

		//if super-mesh display quantities are set with CUDA enabled then get value from pSMeshCUDA
		return pSMeshCUDA->GetAverageDisplayedMeshValue(meshName, rel_rect, shapes, quantity);
	}
#endif

	//If quantity is MESHDISPLAY_NONE (default) then use displayedPhysicalQuantity instead, which could be a supermesh quantity or an individual mesh one.
	//If quantity is not MESHDISPLAY_NONE then get that quantity from the indicated mesh (individual one, or supermesh)
	switch ((quantity == MESHDISPLAY_NONE ? displayedPhysicalQuantity : quantity)) {

		////////////////
		//no quantity displayed on the supermesh, so use individual mesh displayed quantities
		////////////////

	default:
	case MESHDISPLAY_NONE:
	{
		if (contains(meshName)) return pMesh[meshName]->GetAverageDisplayedMeshValue(rel_rect, shapes, quantity);
	}
	break;

	////////////////
	//use a quantity displayed on the supermesh
	////////////////

	case MESHDISPLAY_SM_DEMAG:

		if (IsSuperMeshModuleSet(MODS_SDEMAG)) {

			return dynamic_cast<SDemag*>(pSMod(MODS_SDEMAG))->GetDemagField().average_nonempty_omp(rel_rect);
		}
		break;

	case MESHDISPLAY_SM_OERSTED:

		if (IsSuperMeshModuleSet(MODS_OERSTED)) {

			return dynamic_cast<Oersted*>(pSMod(MODS_OERSTED))->GetOerstedField().average_nonempty_omp(rel_rect);
		}
		break;

	case MESHDISPLAY_SM_STRAYH:

		if (IsSuperMeshModuleSet(MODS_STRAYFIELD)) {

			return dynamic_cast<StrayField*>(pSMod(MODS_STRAYFIELD))->GetStrayField().average_nonempty_omp(rel_rect);
		}
		break;
	}
	
	return Any();
}

BError SuperMesh::SetDisplayedPhysicalQuantity(std::string meshName, int displayedPhysicalQuantity_)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName != superMeshHandle) {

		MESH_ meshType = pMesh[meshName]->GetMeshType();

		if (displayedPhysicalQuantity_ >= MESHDISPLAY_NONE && vector_contains(meshAllowedDisplay(meshType), (MESHDISPLAY_)displayedPhysicalQuantity_)) {

			pMesh[meshName]->SetDisplayedPhysicalQuantity(displayedPhysicalQuantity_);
		}
	}
	else {

		if (displayedPhysicalQuantity_ >= MESHDISPLAY_NONE && vector_contains(meshAllowedDisplay(MESH_SUPERMESH), (MESHDISPLAY_)displayedPhysicalQuantity_)) {

			displayedPhysicalQuantity = displayedPhysicalQuantity_;
		}
	}

	return error;
}

BError SuperMesh::SetDisplayedBackgroundPhysicalQuantity(std::string meshName, int displayedBackgroundPhysicalQuantity_)
{
	BError error(__FUNCTION__);

	if (!contains(meshName)) return error(BERROR_INCORRECTNAME);

	MESH_ meshType = pMesh[meshName]->GetMeshType();

	if (displayedBackgroundPhysicalQuantity_ >= MESHDISPLAY_NONE && vector_contains(meshAllowedDisplay(meshType), (MESHDISPLAY_)displayedBackgroundPhysicalQuantity_)) {

		pMesh[meshName]->SetDisplayedBackgroundPhysicalQuantity(displayedBackgroundPhysicalQuantity_);
	}

	return error;
}

//Get/Set vectorial quantity representation options in named mesh (which could be the supermesh)
BError SuperMesh::SetVEC3Rep(std::string meshName, int vec3rep_)
{
	BError error(__FUNCTION__);

	if (!contains(meshName) && meshName != superMeshHandle) return error(BERROR_INCORRECTNAME);

	if (meshName != superMeshHandle) {

		 pMesh[meshName]->SetVEC3Rep(vec3rep_);
	}
	else {

		vec3rep = vec3rep_;
	}

	return error;
}

int SuperMesh::GetVEC3Rep(std::string meshName)
{
	if (!contains(meshName) && meshName != superMeshHandle) return vec3rep;

	if (meshName != superMeshHandle) {

		return pMesh[meshName]->GetVEC3Rep();
	}
	else {

		return vec3rep;
	}
}