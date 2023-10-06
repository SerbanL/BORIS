#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SDEMAG

#include "ModulesCUDA.h"
#include "EvalSpeedupCUDA.h"

class SuperMesh;
class SDemag;
class ManagedMeshCUDA;

class SDemagCUDA_Demag;
class SDemagMCUDA_single;

class DemagKernelCollectionCUDA;

class SDemagCUDA :
	public ModulesCUDA,
	public EvalSpeedupCUDA
{
	friend SDemagMCUDA_single;
	friend SDemagCUDA_Demag;

private:

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	SuperMesh* pSMesh;

	//the SDemag module (cpu version) holding this CUDA version
	SDemag* pSDemag;

	//-------- SUPERMESH DEMAG

	//super-mesh magnetization values used for computing demag field on the super-mesh
	mcu_VEC(cuReal3) sm_Vals;

	//value used by SDemagMCUDA_single modules for half-precision transfer data normalization (set as largest Ms from all participating meshes)
	cuBReal normalization_Ms = 1.0;

	//total non-empty volume from all meshes participating in convolution
	double total_nonempty_volume = 0.0;

	//--------

	//one SDemagMCUDA_single object per GPU
	std::vector<SDemagMCUDA_single*> pSDemagMCUDA;

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

	//-------- MULTICONVOLUTION DEMAG

	//collection of all SDemagCUDA_Demag modules in individual magnetic meshes
	std::vector<SDemagCUDA_Demag*> pSDemagCUDA_Demag;

	//collect FFT input spaces : after Forward FFT the ffts of M from the individual meshes will be found here
	//These are used as inputs to kernel multiplications. Same order as pSDemag_Demag.
	//outer vector dimension is number of GPUs used
	std::vector<std::vector<cu_arr<cuBComplex>*>> FFT_Spaces_x_Input, FFT_Spaces_y_Input, FFT_Spaces_z_Input;

	//collection of rectangles of meshes, same ordering as for pSDemag_Demag and FFT_Spaces, used in multi-layered convolution
	//these are not necessarily the rectangles of the input M meshes, but are the rectangles of the transfer meshes (M -> transfer -> convolution)
	//for instance in 3D mode, all rectangles in multi-layered convolution must have same size
	//in 2D mode the rectangles can differ in thickness but must have the same xy size
	//thus in 3D mode find largest one and extend all the other rectangles to match (if possible try to have them overlapping in xy-plane projections so we can use kernel symmetries)
	//in 2D mode find largest xy dimension and extend all xy dimensions -> again try to overlap their xy-plane projections
	std::vector<cuRect> Rect_collection;

	//demag kernels used for multilayered convolution, one collection per mesh/SDemag_Demag module. Don't recalculate redundant kernels in the collection.
	//outer vector dimension is number of GPUs used
	std::vector<std::vector<DemagKernelCollectionCUDA*>> kernel_collection;

private:

	//-------- SUPERMESH DEMAG

	//construct objects for supermesh demag
	void Make_SMesh_Demag(void);
	
	//destruct objects for supermesh demag
	void Clear_SMesh_Demag(void);

	//check if all pSDemagMCUDA modules are initialized
	bool SDemagCUDA_Submodules_Initialized(void);

	//called from Initialize if using SMesh demag
	BError Initialize_SMesh_Demag(void);

	//called from UpdateConfiguration if using Smesh demag
	BError UpdateConfiguration_SMesh_Demag(UPDATECONFIG_ cfgMessage);

	//called from UpdateField if using Smesh demag
	void UpdateField_SMesh_Demag(void);

	//-------- MULTICONVOLUTION DEMAG

	//called from Initialize if using multiconvolution demag
	BError Initialize_MConv_Demag(void);

	//called from UpdateConfiguration if using multiconvolution demag
	BError UpdateConfiguration_MConv_Demag(UPDATECONFIG_ cfgMessage);

	//called from UpdateField if using multiconvolution demag
	void UpdateField_MConv_Demag(void);

public:

	SDemagCUDA(SuperMesh* pSMesh_, SDemag* pSDemag_);
	~SDemagCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	void UninitializeAll(void);

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);

	//-------------------

	mcu_VEC(cuReal3)& GetDemagField(void) { return sm_Vals; }
};

#else

class SDemagCUDA
{
};

#endif

#endif


