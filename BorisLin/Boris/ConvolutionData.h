#pragma once

#include "BorisLib.h"

#include "ErrorHandler.h"

#include "fftw3.h"

#pragma comment(lib, "libfftw3-3.lib")

class ConvolutionData
{

protected:
	
	int OmpThreads;

	//input mesh dimensions
	SZ3 n;

	//intput mesh cellsize
	DBL3 h;

	//dimensions of FFT spaces, even so can always divide by 2 :
	//In directions without PBC these are double the n values, but not necessarily a power of 2.
	//In directions with PBC these are the same as the n values.
	//For pbc all we have to do is calculate the kernel differently and set N value to n rather than double that of n.
	SZ3 N;

	//periodic boundary conditions -> setting these changes convolution dimensions and workflow for kernel calculation.
	//Set to zero to remove pbc conditions. This gives the number of images to use when computing tensors with pbc conditions.
	INT3 pbc_images;

	//FFT calculation spaces : dimensions (N.x/2 + 1) * N.y * n.z. for 3D and (N.x/2 + 1) * n.y * 1 for 2D
	VEC<ReIm3> F;

	//by default embed the kernel multiplication with the z-axis fft / ifft
	//if false then allocate additional scratch spaces so that kernel multiplication stores result in them. then ifft proceeds from these additional spaces (this is the un-embedded pipeline)
	bool embed_multiplication = true;

	//additional scratch space used when embed_multiplication == false
	VEC<ReIm3> F2;

	std::vector<fftw_plan> plan_fwd_x, plan_fwd_y, plan_fwd_z;
	std::vector<fftw_plan> plan_inv_x, plan_inv_y, plan_inv_z;

	//forward fft lines with constant zero padding
	std::vector<double*> pline_zp_x;
	std::vector<fftw_complex*> pline_zp_y, pline_zp_z;

	//fft and ifft line without zero padding
	std::vector<fftw_complex*> pline;

	//ifft line for real output, to be truncated
	std::vector<double*> pline_rev_x;
	
	//the flow is:
	//input -> pline_zp_x -fft-> pline -> F
	//F -> pline_zp_y -fft-> pline -> F
	//F -> pline_zp_z -fft-> pline -*K> pline -> -ifft-> pline -> F
	//
	//F -> pline -ifft-> pline -> F
	//F -> pline -ifft-> pline_rev_x -> output

	bool fftw_plans_created = false;

private:

	//-------------------------- HELPERS

	//free memmory allocated for fftw
	void free_memory(void);

	//Allocate memory for F and F2 (if needed) scratch spaces)
	BError AllocateScratchSpaces(void);

protected:

	//-------------------------- CONSTRUCTORS

	ConvolutionData(void);

	virtual ~ConvolutionData();

	//-------------------------- CONFIGURATION

	BError SetConvolutionDimensions(SZ3 n_, DBL3 h_, bool embed_multiplication_ = true, INT3 pbc_images = INT3());

	//zero fftw memory
	void zero_fft_lines(void);

	//-------------------------- GETTERS

	//Get pointer to the F scratch space
	VEC<ReIm3>* Get_Input_Scratch_Space(void) { return &F; }

	//-------------------------- RUN-TIME METHODS

};
