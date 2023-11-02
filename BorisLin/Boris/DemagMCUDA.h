#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_DEMAG

#include "ModulesCUDA.h"

#include "EvalSpeedupCUDA.h"

class MeshCUDA;
class DemagMCUDA_single;

class DemagMCUDA :
	public ModulesCUDA,
	public EvalSpeedupCUDA
{
	friend DemagMCUDA_single;

private:

	////////////////////////////////////////////////////

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	MeshCUDA* pMeshCUDA;

	////////////////////////////////////////////////////

	//one DemagMCUDA_single object per GPU
	std::vector<DemagMCUDA_single*> pDemagMCUDA;

	//transfer data before x-FFTs
	std::vector<std::vector<mGPU_Transfer<cuReal3>*>> M_Input_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> M_Input_transfer_half;

	std::vector<std::vector<mGPU_Transfer<cuBComplex>*>> xFFT_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> xFFT_Data_transfer_half;

	//transfer data before x-IFFTs
	std::vector<std::vector<mGPU_Transfer<cuBComplex>*>> xIFFT_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> xIFFT_Data_transfer_half;
	
	//transfer data after x-IFFTs
	std::vector<std::vector<mGPU_Transfer<cuReal3>*>> Out_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> Out_Data_transfer_half;

private:

	//check if all pDemagMCUDA modules are initialized
	bool Submodules_Initialized(void);

	void set_DemagCUDA_pointers(void);
	
public:

	DemagMCUDA(MeshCUDA* pMeshCUDA_);
	~DemagMCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);

	//-------------------Configuration

};

#else

class DemagMCUDA
{
};

#endif

#endif


