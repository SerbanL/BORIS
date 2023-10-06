#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_OERSTED

#include "ModulesCUDA.h"

#include "ConvolutionCUDA.h"
#include "OerstedKernelCUDA.h"

class SuperMesh;
class Oersted;

class OerstedMCUDA_single;

class OerstedCUDA :
	public ModulesCUDA
{
	friend OerstedMCUDA_single;

private:

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	SuperMesh * pSMesh;

	//the SDemag module (cpu version) holding this CUDA version
	Oersted* pOersted;

	////////////////////////////////////////////////////

	//one OerstedCUDA_single object per GPU
	std::vector<OerstedMCUDA_single*> pOerstedMCUDA;

	//transfer data before x-FFTs
	std::vector<std::vector<mGPU_Transfer<cuReal3>*>> J_Input_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> J_Input_transfer_half;

	std::vector<std::vector<mGPU_Transfer<cuBComplex>*>> xFFT_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> xFFT_Data_transfer_half;

	//transfer data before x-IFFTs
	std::vector<std::vector<mGPU_Transfer<cuBComplex>*>> xIFFT_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> xIFFT_Data_transfer_half;

	//transfer data after x-IFFTs
	std::vector<std::vector<mGPU_Transfer<cuReal3>*>> Out_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> Out_Data_transfer_half;

	//use a normalization constant to avoid exponent limit for half-precision, i.e. before precision reduction divide by this, then multiply result when converting back up
	double normalization = 1.0, normalization_J = 1.0;

	////////////////////////////////////////////////////

	//super-mesh magnetization values used for computing demag field on the super-mesh
	mcu_VEC(cuReal3) sm_Vals;

private:

	//check if all pOerstedMCUDA modules are initialized
	bool Submodules_Initialized(void);

public:

	OerstedCUDA(SuperMesh* pSMesh_, Oersted* pOersted_);
	~OerstedCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);

	//-------------------

	mcu_VEC(cuReal3)& GetOerstedField(void) { return sm_Vals; }
};

#else

class OerstedCUDA
{
};

#endif

#endif


