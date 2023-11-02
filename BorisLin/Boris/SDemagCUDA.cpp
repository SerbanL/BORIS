#include "stdafx.h"
#include "SDemagCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SDEMAG

#include "ManagedMeshCUDA.h"

#include "SuperMesh.h"
#include "SDemag.h"

SDemagCUDA::SDemagCUDA(SuperMesh* pSMesh_, SDemag* pSDemag_) :
	ModulesCUDA(),
	EvalSpeedupCUDA(),
	sm_Vals(mGPU)
{
	Uninitialize();

	pSMesh = pSMesh_;
	pSDemag = pSDemag_;

	error_on_create = UpdateConfiguration(UPDATECONFIG_FORCEUPDATE);
}

SDemagCUDA::~SDemagCUDA()
{
	//copy values back to cpu version
	if (Holder_Module_Available()) {

		if (!pSDemag->use_multilayered_convolution) {

			sm_Vals.copy_to_cpuvec(pSDemag->sm_Vals);
		}

		//must force SDemag module to re-initialize as it's not properly initialized when CUDA module active even though its initialized flag is true.
		pSDemag->Uninitialize();
	}

	Clear_SMesh_Demag();
}

void SDemagCUDA::UninitializeAll(void)
{
	Uninitialize();

	//be careful when using UninitializeAll : pSDemagCUDA_Demag must be up to date
	if (pSDemag->use_multilayered_convolution) {

		for (int idx = 0; idx < pSDemagCUDA_Demag.size(); idx++) {

			pSDemagCUDA_Demag[idx]->Uninitialize();
		}
	}
}

BError SDemagCUDA::Initialize(void)
{
	BError error(CLASS_STR(SDemagCUDA));

	//this module only works with subvec axis x (multiple devices)
	if (mGPU.get_num_devices() > 1 && mGPU.get_subvec_axis() != 0) {

		Uninitialize();
		return error(BERROR_MGPU_MUSTBEXAXIS);
	}

	//FFT Kernels are not so quick to calculate - if already initialized then we are guaranteed they are correct
	if (!initialized) {
		
		///////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////// SUPERMESH CONVOLUTION ////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		if (!pSDemag->use_multilayered_convolution) {

			error = Initialize_SMesh_Demag();
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////// MULTILAYERED CONVOLUTION //////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////
		
		else {

			error = Initialize_MConv_Demag();
		}

		if (!error) {

			//initialized ok.
			initialized = true;

			//mirror SDemag initialized flag
			pSDemag->initialized = true;
		}
	}

	EvalSpeedupCUDA::num_Hdemag_saved = 0;

	return error;
}

BError SDemagCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SDemagCUDA));
	
	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_DEMAG_CONVCHANGE, UPDATECONFIG_SMESH_CELLSIZE, UPDATECONFIG_MESHCHANGE, UPDATECONFIG_MESHADDED, UPDATECONFIG_MESHDELETED, UPDATECONFIG_MODULEADDED, UPDATECONFIG_MODULEDELETED)) {

		///////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////// SUPERMESH CONVOLUTION ////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////
		
		if (!pSDemag->use_multilayered_convolution) {

			Make_SMesh_Demag();

			error = UpdateConfiguration_SMesh_Demag(cfgMessage);

			Uninitialize();
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////// MULTILAYERED CONVOLUTION //////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////

		else {

			Clear_SMesh_Demag();

			error = UpdateConfiguration_MConv_Demag(cfgMessage);
		}

		EvalSpeedupCUDA::UpdateConfiguration_EvalSpeedup();
	}

	return error;
}

void SDemagCUDA::UpdateField(void)
{
	///////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////// SUPERMESH CONVOLUTION ////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	if (!pSDemag->use_multilayered_convolution) {
		
		UpdateField_SMesh_Demag();
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////// MULTILAYERED CONVOLUTION //////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	else {

		UpdateField_MConv_Demag();
	}
}

#endif

#endif