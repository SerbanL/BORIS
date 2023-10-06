#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SDEMAG

#include "BorisCUDALib.h"

#include "ConvolutionCUDA.h"
#include "DemagKernelCollectionCUDA.h"

class SDemagCUDA_Demag;
class SDemagCUDA;

class SDemagMCUDA_Demag_single :
	public ConvolutionCUDA<SDemagMCUDA_Demag_single, DemagKernelCollectionCUDA>
{
	friend SDemagCUDA_Demag;
	friend SDemagCUDA;

private:

	////////////////////////////////////////////////////

	SDemagCUDA_Demag* pSDemagCUDA_Demag = nullptr;

	////////////////////////////////////////////////////

	//Real and Complex arrays for collecting and transferring data between devices in between convolution steps
	//xRegion : arrays have same xRegion
	//yRegion : arrays have same yRegion
	//half : half-precision arrays when half-precision transfer mode is enabled
	//_arr : cu_arr of cu_arrs used to collect them so they can be passed to cuda kernels
	//vector size is number of devices, and all indexes != device_index will store data from other devices after transfers,
	std::vector<cu_arr<cuReal3>*> Real_xRegion;
	std::vector<cu_arr<cuReal3>*> Real_yRegion;
	std::vector<cu_arr<cuBHalf>*> Real_xRegion_half;
	std::vector<cu_arr<cuBHalf>*> Real_yRegion_half;

	cu_arr<cuReal3*> Real_xRegion_arr;
	cu_arr<cuReal3*> Real_yRegion_arr;
	cu_arr<cuBHalf*> Real_xRegion_half_arr;
	cu_arr<cuBHalf*> Real_yRegion_half_arr;

	std::vector<cu_arr<cuBComplex>*> Complex_xRegion;
	std::vector<cu_arr<cuBComplex>*> Complex_yRegion;
	std::vector<cu_arr<cuBHalf>*> Complex_xRegion_half;
	std::vector<cu_arr<cuBHalf>*> Complex_yRegion_half;

	cu_arr<cuBComplex*> Complex_xRegion_arr;
	cu_arr<cuBComplex*> Complex_yRegion_arr;
	cu_arr<cuBHalf*> Complex_xRegion_half_arr;
	cu_arr<cuBHalf*> Complex_yRegion_half_arr;

	//use a normalization constant to avoid exponent limit for half-precision, i.e. before precision reduction divide by this, then multiply result when converting back up
	cu_obj<cuBReal> normalization, normalization_M;

	//device index for which this SDemagMCUDA_single does calculations
	int device_index;

	//is this module initialized?
	bool initialized = false;

private:

	//Copy M data on this device to linear regions so we can transfer
	void Copy_M_Input_xRegion(bool half_precision);

public:

	SDemagMCUDA_Demag_single(SDemagCUDA_Demag* pSDemagCUDA_Demag_, int device_index_);
	~SDemagMCUDA_Demag_single();

	//-------------------

	BError Initialize(std::vector<DemagKernelCollectionCUDA*>& kernelCollection, int n_z, bool initialize_on_gpu);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
};

#else

class SDemagMCUDA_Demag_single
{
};

#endif

#endif


