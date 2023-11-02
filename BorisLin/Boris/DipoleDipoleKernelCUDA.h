#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_ATOM_DIPOLEDIPOLE)

#include "BorisCUDALib.h"

#include "ConvolutionDataCUDA.h"

#include "ErrorHandler.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Dipole-dipole Kernel calculated from dipole-dipole tensor

//This must be used as a template parameter in Convolution class.

class DipoleDipoleKernelCUDA :
	public virtual ConvolutionDataCUDA		//virtual to solve the diamond problem as it's also inherited by the Convolution class which inherits from this Kernel
{

private:

	//off-diagonal Kernel used for 2D only (real parts only, imaginary parts are zero)
	cu_obj<cuVEC<cuBReal>> K2D_odiag;

	//Kernels for 3D : GPU memory
	//Kdiag : Kx, Ky, Kz; Kodiag : Kxy, Kxz, Kyz; (real parts only, imaginary parts are zero)
	cu_obj<cuVEC<cuReal3>> Kdiag, Kodiag;

private:

	//-------------------------- KERNEL CALCULATION

	BError Calculate_DipoleDipole_Kernels_2D(bool include_self_demag);
	BError Calculate_DipoleDipole_Kernels_3D(bool include_self_demag);

protected:

	//-------------------------- CONSTRUCTOR

	DipoleDipoleKernelCUDA(void) {}

	virtual ~DipoleDipoleKernelCUDA() {}

	//-------------------------- MEMORY ALLOCATION

	//Called by SetDimensions in ConvolutionCUDA class
	BError AllocateKernelMemory(void);

	//-------------------------- KERNEL CALCULATION

	//this initializes the convolution kernels for the given mesh dimensions. 2D is for n.z == 1.
	BError Calculate_DipoleDipole_Kernels(bool include_self_demag = true)
	{
		if (n.z == 1) return Calculate_DipoleDipole_Kernels_2D(include_self_demag);
		else return Calculate_DipoleDipole_Kernels_3D(include_self_demag);
	}

	//-------------------------- RUN-TIME KERNEL MULTIPLICATION

	//Called by Convolute_2D/Convolute_3D methods in ConvolutionCUDA class : define pointwise multiplication with Kernels (using the cuSx, cuSy and cuSz arrays)
	void KernelMultiplication_2D(void);
	void KernelMultiplication_3D(void);

	//Kernel multiplication in quasi-2D mode : z-axis fft / kernel multiplication / z-axis ifft rolled into one (but do not divide by N for the ifft)
	void KernelMultiplication_q2D(int q2D_level);
};

#endif

#endif


