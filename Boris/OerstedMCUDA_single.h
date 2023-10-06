#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_OERSTED

#include "BorisCUDALib.h"

#include "ConvolutionCUDA.h"
#include "OerstedKernelCUDA.h"

//computation on a single GPU for a portion of a mcuVEC, intended as part of a multi-GPU computation, and directed by OerstedCUDA.

class OerstedCUDA;

class OerstedMCUDA_single :
	public ConvolutionCUDA<OerstedMCUDA_single, OerstedKernelCUDA>
{
	friend OerstedCUDA;

private:

	////////////////////////////////////////////////////

	OerstedCUDA* pOerstedCUDA = nullptr;

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
	cu_obj<cuBReal> normalization, normalization_J;

	//device index for which this OerstedCUDA_single does calculations
	int device_index;

	//is this module initialized?
	bool initialized = false;

private:

	//Copy J data on this device to linear regions so we can transfer
	void Copy_J_Input_xRegion(bool half_precision);

public:

	OerstedMCUDA_single(OerstedCUDA* pOerstedCUDA_, int device_index_);
	~OerstedMCUDA_single();

	//-------------------

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);
};

#else

class OerstedMCUDA_single
{
};

#endif

#endif


