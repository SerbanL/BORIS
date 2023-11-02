#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#include "BorisCUDALib.h"

#include "ErrorHandler.h"

#include "fftw3.h"

#pragma comment(lib, "libfftw3-3.lib")

//size of N.y required to switch to transpose_xy mode
//This is also the threshold of N.x/2 required to switch to interleaving transposing operations for the 3 vector components, rather than launching a single kernel call
#define TRANSPOSE_XY_YTHRESHOLD	2048

//convolution configuration
enum CONV_ { 
	
	//embed_multiplication = true, additional_spaces = false; can only be used if a single mesh (and single gpu) is present
	CONV_SINGLEMESH = 0,
	//embed_multiplication = true, additional_spaces = true; suitable for multi-mesh (and multi-gpu) convolution
	CONV_MULTIMESH = 1,
	//not embedded test modes, where q2d mode is disabled
	//embed_multiplication = false, additional_spaces = false; can only be used if a single mesh (and single gpu) is present
	CONV_SINGLEMESH_NOTEMBEDDED = 2,
	//embed_multiplication = true, additional_spaces = true; suitable for multi-mesh (and multi-gpu) convolution
	CONV_MULTIMESH_NOTEMBEDDED = 3
};

//macros so the correct cufft type is launched depending on set precision
#if SINGLEPRECISION == 1
#define cufftR2C cufftExecR2C
#define cufftC2C cufftExecC2C
#define cufftC2R cufftExecC2R
#else
#define cufftR2C cufftExecD2Z
#define cufftC2C cufftExecZ2Z
#define cufftC2R cufftExecZ2D
#endif

class ConvolutionDataCUDA
{

private:

	//use additional scratch spaces for forward FFTs so zero padding regions are not polluted. This saves on having to launch additional zero padding kernels at the cost of extra memory.
	bool preserve_zero_padding = true;

protected:

	//mesh dimensions
	cuSZ3 n;

	//mesh cellsize
	cuReal3 h;

	//dimensions of FFT spaces
	cuSZ3 N;

	//periodic boundary conditions -> setting these changes convolution dimensions and workflow for kernel calculation.
	//Set to zero to remove pbc conditions. This gives the number of images to use when computing tensors with pbc conditions.
	cuINT3 pbc_images;

	//If xRegion is not null, then perform convolution only in the specified x region
	//This is useful for multi-GPU (mGPU) demag, where n, h, N will be values for the entire space, but each device will be assigned a different xRegion to work on
	//In this mode y and z (I)FFTs will only be done in the specified xRegion to full length along respective axes.
	//For the x (I)FFTs, they will also be done to full length along the x axis, but in the respective yRegion only so their computation times is also reduced with more devices
	//note, xRegion_R is for real numbers, which is the input xRegion with maximum value n.x. xRegion is for complex numbers with maximum value N.x / 2 + 1 (only last xRegion.j differs from xRegion_R, rest are the same)
	cuINT2 xRegion, xRegion_R, yRegion;

	//is either (N.x/2 + 1) (xRegion null) or xRegion.j - xRegion.i
	int nxRegion;
	//is either n.x (xRegion null) or xRegion_R.j - xRegion_R.i
	int nxRegion_R;
	//is either n.y (xRegion null) or yRegion.j - yRegion.i
	int nyRegion;

	//the above values in GPU memory
	cu_obj<cuINT2> cuxRegion, cuxRegion_R, cuyRegion;

	int num_devices = 1;
	cu_obj<int> cunum_devices;

	//mesh dimensions in GPU memory
	cu_obj<cuSZ3> cun;
	//dimensions of FFT space in GPU memory
	cu_obj<cuSZ3> cuN;
	
	//(N.x/2 + 1) * N.y * N.z or (xRegion.j - xRegion.i) * N.y * N.z. This is useful for kernel multiplications.
	cu_obj<cuSZ3> cuN_xRegion;

	//dimensions of complex fft spaces in GPU memory (N.x/2 + 1)*N.y*N.z, and N.y*(N.x/2 + 1)*N.z respectively
	cu_obj<cuSZ3> cuNc_xy, cuNc_yx;
	//dimensions of complex fft spaces in GPU memory for q2d mode (N.x/2 + 1)*N.y*N.z/2, and N.y*(N.x/2 + 1)*N.z/2 respectively
	cu_obj<cuSZ3> cuNc_xy_q2d, cuNc_yx_q2d;
	//quarter size scratch space sizes as: (N.x/2 + 1)*n.y*n.z for xy transpose and n*y*(N.x/2 + 1)*n.z for yx transpose respectively
	cu_obj<cuSZ3> cuNcquart_xy, cuNcquart_yx;

	//regions for zero padding
	cu_obj<cuRect> cuUpper_y_region, cuUpper_y_transposed_region, cuUpper_z_region, cuUpper_z_region_q2d;

	//cuFFT plans : 2D
	cufftHandle plan2D_fwd_x, plan2D_y;
	cufftHandle plan2D_inv_x;
	
	//cuFFT plans : 3D
	cufftHandle plan3D_fwd_x, plan3D_y, plan3D_z;
	cufftHandle plan3D_inv_x;

	//FFT arrays in GPU memory

	//Input / output space of size N.x * n.y * n.z; zero padding for input from n.x to N.x must be kept by computations
	cu_arr<cuBReal> cuIn_x, cuIn_y, cuIn_z;
	cu_arr<cuBReal> cuOut_x, cuOut_y, cuOut_z;

	//Scratch space (N.x / 2 + 1) * N.y * N.z (except in q2D mode where last dimension is n.z, same as in 2D mode -> q2D mode is disabled if not using the embedded convolution pipeline)
	cu_arr<cuBComplex> cuS_x, cuS_y, cuS_z;

	//Scratch space with zero padding above ny, dimensions (N.x / 2 + 1) * N.y * n.z. Forward FFT only. Not used if y pbc is on (in this case they will point to cuS).
	cu_arr<cuBComplex> *pcuSzpy_x = nullptr, *pcuSzpy_y = nullptr, *pcuSzpy_z = nullptr;
	//Scratch space with zero padding above nz, dimensions (N.x / 2 + 1) * N.y * N.z. Forward FFT only. Not used if z pbc is on, or if using q2D or 2D mode (in this case they will point to cuS).
	cu_arr<cuBComplex> *pcuSzpz_x = nullptr, *pcuSzpz_y = nullptr, *pcuSzpz_z = nullptr;

	//the embedded convolution pipeline should be used (CONV_SINGLEMESH or CONV_MULTIMESH modes)
	//(i.e. last dimension FFT -> Mult -> iFFT are done one after another in same kernel call; this is the q2d mode, which requires less memory and is faster if last dimension not large)
	//if you want to break down the convolution into separate calls then set embed_multiplication = false with the CONV_SINGLEMESH_NOTEMBEDDED or CONV_MULTIMESH_NOTEMBEDDED modes. This disables q2D mode.
	bool embed_multiplication = true;

	//allocate additional scratch spaces so that kernel multiplication stores result in them. then ifft proceeds from these additional spaces.
	//this is required for multi-mesh convolution as we cannot write result back into base scratch base, as these are needed to complete all kernel multiplications.
	bool additional_spaces = false;

	//additional scratch space used if additional_spaces == true
	cu_arr<cuBComplex> cuS2_x, cuS2_y, cuS2_z;

	//Quarter-size scratch space used for x fft/ifft when the transpose_xy mode is used.
	//After the x ffts this space is transposed into the main scratch space.
	//Similarly before the x iffts, the required part from the main scratch space is transposed into the quarter scratch space.
	cu_arr<cuBComplex> cuSquart_x, cuSquart_y, cuSquart_z;

	//transpose xy planes before doing the y direction fft/ifft?
	//for 2D mode we have a choice : transpose_xy mode triggered if N.y is above a set threshold
	//for 3D (and q2D) mode always use transpose_xy as it turns out it's a very good catch-all approach
	bool transpose_xy = true;

	//quasi 2D mode : 3D mode but with the z-axis fft / kernel multiplication / z-axis ifft rolled into one step
	//currently handled values:
	//0 : q2D disabled
	//4 : N.z = 4 (n.z = 2)
	//8 : N.z = 8 (n.z = 3, 4)
	//16 : N.z = 16 (n.z = 5, 6, 7, 8)
	//32 : N.z = 32 (n.z = 9, 10, ..., 16)
	int q2D_level = 0;

private:

	BError AllocateConvolutionSpaces(void);

	void DeallocateConvolutionSpaces(void);

protected:

	//-------------------------- CONSTRUCTORS

	ConvolutionDataCUDA(void) {}

	virtual ~ConvolutionDataCUDA();

	//-------------------------- CONFIGURATION

	BError SetConvolutionDimensions(cuSZ3 n_, cuReal3 h_, CONV_ convtype, cuINT3 pbc_images_ = INT3(), cuINT2 xRegion_ = cuINT2(), std::pair<int, int> devices_cfg = {0, 1});

	//Similar to SetConvolutionDimensions but use a pre-determined N value
	BError SetConvolutionDimensions(cuSZ3 n_, cuReal3 h_, cuSZ3 N_, CONV_ convtype, cuINT3 pbc_images_ = INT3(), cuINT2 xRegion_ = cuINT2(), std::pair<int, int> devices_cfg = { 0, 1 });

	//for given n value and pbc images recommend N value to use
	cuSZ3 Recommend_N_from_n(cuSZ3 n_, cuINT3 pbc_images_ = INT3());

	BError Set_Preserve_Zero_Padding(bool status)
	{
		preserve_zero_padding = status;
		return AllocateConvolutionSpaces();
	}

	//-------------------------- RUN-TIME METHODS

	//Input

	//Copy data to cuIn_x, cuIn_y, cuIn_z arrays at start of convolution iteration
	template <typename cuVECIn>
	void CopyInputData(cuVECIn& In);

	//Copy input data from array components collected from all devices in yRegion (but different x regions), as well as from In in the current xRegion and yRegion, to cuIn_x, cuIn_y, cuIn_z arrays at start of convolution iteration
	template <typename cuVECIn>
	void CopyInputData_mGPU(cuVECIn& In, cu_arr<cuReal3*>& M_Input_yRegion);
	template <typename cuVECIn>
	void CopyInputData_mGPU(cuVECIn& In, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M);

	template <typename cuVECIn>
	void CopyInputData_AveragedInputs_mGPU(cuVECIn& In1, cuVECIn& In2, cu_arr<cuReal3*>& M_Input_yRegion);
	template <typename cuVECIn>
	void CopyInputData_AveragedInputs_mGPU(cuVECIn& In1, cuVECIn& In2, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M);

	//Average input data - (In1 + In2 ) / 2 - to cuSx, cuSy, cuSz arrays at start of convolution iteration
	template <typename cuVECIn>
	void AverageInputData(cuVECIn& In1, cuVECIn& In2);

	//FFTs

	void forward_fft_2D(void);
	void inverse_fft_2D(void);
	void inverse_fft_2D_2(void);

	void forward_fft_3D(void);
	void inverse_fft_3D(void);
	void inverse_fft_3D_2(void);

	void forward_fft_q2D(void);
	void inverse_fft_q2D(void);
	void inverse_fft_q2D_2(void);

	//FFTs for mGPU use

	//perform x step of 2D forward FFTs in current yRegion, after having copied in components from other devices, then transpose and copy data to xIFFT_Data_yRegion so we can transfer to other devices
	void forward_fft_2D_mGPU_xstep(cu_arr<cuBComplex*>& xIFFT_Data_yRegion);
	void forward_fft_2D_mGPU_xstep(cu_arr<cuBHalf*>& xIFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization);
	//copy in data from x FFT obtained from other devices, so we can perform last steps of forward FFT
	void forward_fft_2D_mGPU_ystep(cu_arr<cuBComplex*>& xFFT_Data_xRegion);
	void forward_fft_2D_mGPU_ystep(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization);
	//perform x step of 3D forward FFTs in current yRegion, after having copied in components from other devices, then transpose and copy data to xIFFT_Data_yRegion so we can transfer to other devices
	void forward_fft_3D_mGPU_xstep(cu_arr<cuBComplex*>& xIFFT_Data_yRegion);
	void forward_fft_3D_mGPU_xstep(cu_arr<cuBHalf*>& xIFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization);
	//copy in data from x FFT obtained from other devices, so we can perform last steps of forward FFT
	void forward_fft_3D_mGPU_yzsteps(cu_arr<cuBComplex*>& xFFT_Data_xRegion);
	void forward_fft_3D_mGPU_yzsteps(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization);
	//as above but only y step of q2D (z not required separately in q2D)
	void forward_fft_q2D_mGPU_ystep(cu_arr<cuBComplex*>& xFFT_Data_xRegion);
	void forward_fft_q2D_mGPU_ystep(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization);
	
	//perform y step of IFFT in 2D mode (before x IFFT need to transfer data between devices, which is the data copied to linear spaces xIFFT_Data - this must have dimensions nxRegion * (corresponding nyRegion) * n.z)
	void inverse_fft_2D_mGPU_ystep(cu_arr<cuBComplex*>& xIFFT_Data_xRegion);
	void inverse_fft_2D_mGPU_ystep(cu_arr<cuBHalf*>& xIFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization);
	//perform x IFFT step after copying in xIFFT data from all components not in the current xRegion; xIFFT_Data has size the total number of convolution components, but the one matching the current component doesn't need to be copied over
	//after x IFFT copy results for other devices in Out_Data_yRegion
	void inverse_fft_2D_mGPU_xstep(cu_arr<cuBComplex*>& xIFFT_Data_yRegion, cu_arr<cuReal3*>& Out_Data_yRegion);
	void inverse_fft_2D_mGPU_xstep(cu_arr<cuBHalf*>& xIFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xIFFT, cu_arr<cuBHalf*>& Out_Data_yRegion_half, cu_obj<cuBReal>& normalization_Out);

	//perform z and y steps of IFFT (before x IFFT need to transfer data between devices, which is the data copied to linear spaces xIFFT_Data_xRegion - this must have dimensions nxRegion * (corresponding nyRegion) * n.z)
	void inverse_fft_3D_mGPU_zysteps(cu_arr<cuBComplex*>& xIFFT_Data_xRegion);
	void inverse_fft_3D_mGPU_zysteps(cu_arr<cuBHalf*>& xIFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization);
	//perform y step of IFFT in q2D mode (before x IFFT need to transfer data between devices, which is the data copied to linear spaces xIFFT_Data_xRegion - this must have dimensions nxRegion * (corresponding nyRegion)* n.z)
	void inverse_fft_q2D_mGPU_ystep(cu_arr<cuBComplex*>& xIFFT_Data_xRegion);
	void inverse_fft_q2D_mGPU_ystep(cu_arr<cuBHalf*>& xIFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization);
	//perform x IFFT step after copying in xIFFT data from all components not in the current xRegion; xIFFT_Data_yRegion has size the total number of convolution components, but the one matching the current component doesn't need to be copied over
	//after x IFFT copy results for other devices in Out_Data_yRegion
	void inverse_fft_3D_mGPU_xstep(cu_arr<cuBComplex*>& xIFFT_Data_yRegion, cu_arr<cuReal3*>& Out_Data_yRegion);
	void inverse_fft_3D_mGPU_xstep(cu_arr<cuBHalf*>& xIFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xIFFT, cu_arr<cuBHalf*>& Out_Data_yRegion_half, cu_obj<cuBReal>& normalization_Out);
	
	//Output

	//SINGLE INPUT, SINGLE OUTPUT

	//Copy convolution result (in cuOut arrays) to output and obtain energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set(
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);
	
	//Add convolution result (in cuOut arrays) to output and obtain energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add(
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//Copy convolution result (in cuOut arrays) to output and obtain energy value : weighted product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set(
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, cuBReal& energy_weight, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//Add convolution result (in cuOut arrays) to output and obtain energy value : weighted product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add(
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, cuBReal& energy_weight, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version
	//Copy convolution result (in cuOut arrays and Out_Data_xRegion) to output and obtain energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set_mGPU(
		cu_arr<cuReal3*>& Out_Data_xRegion,
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version
	//Add convolution result (in cuOut arrays and Out_Data_xRegion) to output and obtain energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add_mGPU(
		cu_arr<cuReal3*>& Out_Data_xRegion,
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);
	
	//mGPU version with mixed precision
	//Copy convolution result (in cuOut arrays and Out_Data_xRegion) to output and obtain energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set_mGPU(
		cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version with mixed precision
	//Add convolution result (in cuOut arrays and Out_Data_xRegion) to output and obtain energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add_mGPU(
		cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//AVERAGED INPUTS, SINGLE OUTPUT

	//same as above but for averaged input
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, cuBReal& energy_weight, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, cuBReal& energy_weight, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set_mGPU(
		cu_arr<cuReal3*>& Out_Data_xRegion,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version
	//Add convolution result (in cuS arrays) to output and obtain energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add_mGPU(
		cu_arr<cuReal3*>& Out_Data_xRegion,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version with mixed precision
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set_mGPU(
		cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version with mixed precision
	//Add convolution result (in cuS arrays) to output and obtain energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add_mGPU(
		cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//AVERAGED INPUTS, DUPLICATED OUTPUTS

	//same as above but for averaged input and duplicated output
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);
	
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, cuBReal& energy_weight,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, cuBReal& energy_weight, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set_mGPU(
		cu_arr<cuReal3*>& Out_Data_xRegion,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version
	//Add convolution result (in cuS arrays) to output and obtain energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add_mGPU(
		cu_arr<cuReal3*>& Out_Data_xRegion,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version with mixed precision
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Set_mGPU(
		cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//mGPU version with mixed precision
	//Add convolution result (in cuS arrays) to output and obtain energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	template <typename cuVECIn, typename cuVECOut>
	void FinishConvolution_Add_mGPU(
		cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//-------------------------- GETTERS

	//Get pointer to the S scratch space
	cu_arr<cuBComplex>* Get_Input_Scratch_Space_x(void) { return &cuS_x; }
	cu_arr<cuBComplex>* Get_Input_Scratch_Space_y(void) { return &cuS_y; }
	cu_arr<cuBComplex>* Get_Input_Scratch_Space_z(void) { return &cuS_z; }

	//Get pointer to the S scratch space
	cu_arr<cuBComplex>* Get_Scratch_Quarter_Space_x(void) { return &cuSquart_x; }
	cu_arr<cuBComplex>* Get_Scratch_Quarter_Space_y(void) { return &cuSquart_y; }
	cu_arr<cuBComplex>* Get_Scratch_Quarter_Space_z(void) { return &cuSquart_z; }

	cu_arr<cuBReal>* Get_Input_Space_x(void) { return &cuIn_x; }
	cu_arr<cuBReal>* Get_Input_Space_y(void) { return &cuIn_y; }
	cu_arr<cuBReal>* Get_Input_Space_z(void) { return &cuIn_z; }

	//-------------------------- INFO

	bool is_q2d(void) { return q2D_level != 0; }
};

#endif