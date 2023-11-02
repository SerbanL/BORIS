#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_ATOM_DIPOLEDIPOLE) && ATOMISTIC == 1

#include "BorisCUDALib.h"

#include "ConvolutionCUDA.h"
#include "DipoleDipoleKernelCUDA.h"

//DipoleDipole computation on a single GPU for a portion of a mcuVEC, intended as part of a multi-GPU computation, and directed by Atom_DipoleDipoleMCUDA.

class Atom_MeshCUDA;
class Atom_DipoleDipoleMCUDA;

class Atom_DipoleDipoleMCUDA_single :
	public ConvolutionCUDA<Atom_DipoleDipoleMCUDA_single, DipoleDipoleKernelCUDA>
{
	friend Atom_DipoleDipoleMCUDA;

private:

	////////////////////////////////////////////////////

	Atom_MeshCUDA* paMeshCUDA = nullptr;
	Atom_DipoleDipoleMCUDA* pDipoleDipoleMCUDA = nullptr;

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

	//device index for which this DemagMCUDA_single does calculations
	int device_index;

	//is this module initialized?
	bool initialized = false;

private:

	//Copy M data on this device to linear regions so we can transfer
	void Copy_M_Input_xRegion(bool half_precision);

public:

	Atom_DipoleDipoleMCUDA_single(Atom_MeshCUDA* paMeshCUDA_, Atom_DipoleDipoleMCUDA* pDipoleDipoleMCUDA_, int device_index_);
	~Atom_DipoleDipoleMCUDA_single();

	//-------------------

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);
};

#else

class Atom_DipoleDipoleMCUDA_single
{
};

#endif

#endif


