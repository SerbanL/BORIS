#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#include "BorisCUDALib.h"

#include "ConvolutionDataCUDA.h"

#include "ErrorHandler.h"

template <typename Owner, typename Kernel>
class ConvolutionCUDA :
	public virtual ConvolutionDataCUDA,		//virtual to solve the diamond problem as it's also inherited by Kernel
	public Kernel
{

private:

	//if the object couldn't be created properly in the constructor an error is set here
	BError convolution_error_on_create;

private:

	//Embedded (default)

	//convolute In with kernels, set output in Out. 2D is for n.z == 1.
	//set energy value : product of In with Out times -MU0 / (2 * non_empty_points), where non_empty_points = In.get_nonempty_points();
	//If clearOut flag is true then Out is set, otherwise Out is added into.
	//SINGLE INPUT, SINGLE OUTPUT
	template <typename cuVECIn, typename cuVECOut>
	void Convolute_2D(
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy, bool clearOut, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);
	
	template <typename cuVECIn, typename cuVECOut>
	void Convolute_3D(
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy, bool clearOut, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//AVERAGED INPUTS, SINGLE OUTPUT

	//Same as Convolution with (In1 + In2) / 2 as input.
	template <typename cuVECIn, typename cuVECOut>
	void Convolute_2D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy, bool clearOut, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	template <typename cuVECIn, typename cuVECOut>
	void Convolute_3D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy, bool clearOut, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//AVERAGED INPUTS, DUPLICATED OUTPUTS
	//Same as Convolution with (In1 + In2) / 2 as input and output copied to both Out1 and Out2.
	template <typename cuVECIn, typename cuVECOut>
	void Convolute_2D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	template <typename cuVECIn, typename cuVECOut>
	void Convolute_3D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//Not embedded

	//Forward FFT. In -> cuS

	//SINGLE INPUT

	template <typename cuVECIn>
	void ForwardFFT_2D(cuVECIn& In);

	template <typename cuVECIn>
	void ForwardFFT_3D(cuVECIn& In);

	//multi-GPU versions (broken down in 2 steps to allow for transfers in between)
	template <typename cuVECIn>
	void ForwardFFT_2D_mGPU_first(cuVECIn& In, cu_arr<cuReal3*>& M_Input_yRegion, cu_arr<cuBComplex*>& xFFT_Data_yRegion);
	template <typename cuVECIn>
	void ForwardFFT_2D_mGPU_first(cuVECIn& In, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M, cu_arr<cuBHalf*>& xFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xFFT);
	void ForwardFFT_2D_mGPU_last(cu_arr<cuBComplex*>& xFFT_Data_xRegion);
	void ForwardFFT_2D_mGPU_last(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization);
	template <typename cuVECIn>
	void ForwardFFT_3D_mGPU_first(cuVECIn& In, cu_arr<cuReal3*>& M_Input_yRegion, cu_arr<cuBComplex*>& xFFT_Data_yRegion);
	template <typename cuVECIn>
	void ForwardFFT_3D_mGPU_first(cuVECIn& In, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M, cu_arr<cuBHalf*>& xFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xFFT);
	void ForwardFFT_3D_mGPU_last(cu_arr<cuBComplex*>& xFFT_Data_xRegion);
	void ForwardFFT_3D_mGPU_last(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization);

	//AVERAGED INPUTS

	template <typename cuVECIn>
	void ForwardFFT_2D(cuVECIn& In1, cuVECIn& In2);

	template <typename cuVECIn>
	void ForwardFFT_3D(cuVECIn& In1, cuVECIn& In2);

	//multi-GPU versions (broken down in 2 steps to allow for transfers in between)
	template <typename cuVECIn>
	void ForwardFFT_2D_AveragedInputs_mGPU_first(cuVECIn& In1, cuVECIn& In2, cu_arr<cuReal3*>& M_Input_yRegion, cu_arr<cuBComplex*>& xFFT_Data_yRegion);
	template <typename cuVECIn>
	void ForwardFFT_2D_AveragedInputs_mGPU_first(cuVECIn& In1, cuVECIn& In2, cu_arr<cuBHalf*>& M_Input_yRegion, cu_obj<cuBReal>& normalization_M, cu_arr<cuBHalf*>& xFFT_Data_yRegion, cu_obj<cuBReal>& normalization_xFFT);
	
	template <typename cuVECIn>
	void ForwardFFT_3D_AveragedInputs_mGPU_first(cuVECIn& In1, cuVECIn& In2, cu_arr<cuReal3*>& M_Input_yRegion, cu_arr<cuBComplex*>& xFFT_Data_yRegion);
	template <typename cuVECIn>
	void ForwardFFT_3D_AveragedInputs_mGPU_first(cuVECIn& In1, cuVECIn& In2, cu_arr<cuBHalf*>& M_Input_yRegion, cu_obj<cuBReal>& normalization_M, cu_arr<cuBHalf*>& xFFT_Data_yRegion, cu_obj<cuBReal>& normalization_xFFT);

	//Inverse FFT. cuS or cuS2 -> Out

	//SINGLE INPUT, SINGLE OUTPUT

	//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_2D(
		cuVECIn& In, cuVECOut& Out, 
		cuBReal& energy, bool get_energy, bool clearOut, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_3D(
		cuVECIn& In, cuVECOut& Out, 
		cuBReal& energy, bool get_energy, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
	//This is useful when calculating an energy total from multiple meshes which might need different weights.
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_2D(
		cuVECIn& In, cuVECOut& Out, 
		cuBReal& energy, cuBReal& energy_weight, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
	//This is useful when calculating an energy total from multiple meshes which might need different weights.
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_3D(
		cuVECIn& In, cuVECOut& Out, 
		cuBReal& energy, cuBReal& energy_weight, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//AVERAGED INPUTS, SINGLE OUTPUT

	//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_2D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, 
		cuBReal& energy, bool get_energy, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_3D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, 
		cuBReal& energy, bool get_energy, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
	//This is useful when calculating an energy total from multiple meshes which might need different weights.
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_2D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, 
		cuBReal& energy, cuBReal& energy_weight, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
	//This is useful when calculating an energy total from multiple meshes which might need different weights.
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_3D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, 
		cuBReal& energy, cuBReal& energy_weight, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//AVERAGED INPUTS, DUPLICATED OUTPUTS

	//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_2D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, 
		cuBReal& energy, bool get_energy, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_3D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, 
		cuBReal& energy, bool get_energy, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
	//This is useful when calculating an energy total from multiple meshes which might need different weights.
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_2D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, 
		cuBReal& energy, cuBReal& energy_weight, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

	//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
	//This is useful when calculating an energy total from multiple meshes which might need different weights.
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_3D(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, 
		cuBReal& energy, cuBReal& energy_weight, bool clearOut,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr);

protected:

	//-------------------------- CONSTRUCTORS

	ConvolutionCUDA(void) :
		ConvolutionDataCUDA(),
		Kernel()
	{}

	ConvolutionCUDA(cuSZ3 n_, cuReal3 h_, CONV_ convtype, cuINT2 xRegion_ = cuINT2());

	virtual ~ConvolutionCUDA() {}

	//-------------------------- ERROR CHECKER

	BError Convolution_Error_on_Create(void) { return convolution_error_on_create; }

	//-------------------------- CONFIGURATION

	//This method sets all values from n and h, including allocating memory - call this before initializing kernels or doing any convolutions
	BError SetDimensions(cuSZ3 n_, cuReal3 h_, CONV_ convtype, cuINT3 pbc_images_ = cuINT3(), cuINT2 xRegion_ = cuINT2(), std::pair<int, int> devices_cfg = { 0, 1 });

	//Similar to SetDimensions but use a pre-determined N value
	BError SetDimensions(cuSZ3 n_, cuReal3 h_, cuSZ3 N_, CONV_ convtype, cuINT3 pbc_images_ = cuINT3(), cuINT2 xRegion_ = cuINT2(), std::pair<int, int> devices_cfg = { 0, 1 });

	//-------------------------- CHECK

	//return true only if both n_ and h_ match the current dimensions (n and h); also number of pbc images must match
	bool CheckDimensions(cuSZ3 n_, cuReal3 h_, cuINT3 pbc_images_) { return (n == n_ && h == h_ && pbc_images == pbc_images_); }

	//-------------------------- RUN-TIME CONVOLUTION

	//SINGLE INPUT, SINGLE OUTPUT

	//Run a 2D or 3D convolution depending on set n (n.z == 1 for 2D)
	template <typename cuVECIn, typename cuVECOut>
	void Convolute(
		cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy, bool clearOut = true, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (n.z == 1) Convolute_2D(In, Out, energy, get_energy, clearOut, pH, penergy);
		else Convolute_3D(In, Out, energy, get_energy, clearOut, pH, penergy);
	}

	//AVERAGED INPUTS, SINGLE OUTPUT

	//Same as Convolution with (In1 + In2) / 2 as input.
	template <typename cuVECIn, typename cuVECOut>
	void Convolute_AveragedInputs(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy, bool clearOut = true, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (n.z == 1) Convolute_2D(In1, In2, Out, energy, get_energy, clearOut, pH, penergy);
		else Convolute_3D(In1, In2, Out, energy, get_energy, clearOut, pH, penergy);
	}

	//AVERAGED INPUTS, DUPLICATED OUTPUTS

	//Same as Convolution with (In1 + In2) / 2 as input and output copied to both Out1 and Out2.
	template <typename cuVECIn, typename cuVECOut>
	void Convolute_AveragedInputs_DuplicatedOutputs(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy, bool clearOut = true, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (n.z == 1) Convolute_2D(In1, In2, Out1, Out2, energy, get_energy, clearOut, pH, penergy);
		else Convolute_3D(In1, In2, Out1, Out2, energy, get_energy, clearOut, pH, penergy);
	}

	//Convolution broken down into parts : forward FFT, Kernel Multiplication, inverse FFT

	//SINGLE INPUT

	template <typename cuVECIn>
	void ForwardFFT(cuVECIn& In)
	{
		if (n.z == 1) ForwardFFT_2D(In);
		else ForwardFFT_3D(In);
	}

	//multi-GPU version
	template <typename cuVECIn>
	void ForwardFFT_mGPU_first(cuVECIn& In, cu_arr<cuReal3*>& M_Input_yRegion, cu_arr<cuBComplex*>& xFFT_Data_yRegion)
	{
		if (n.z == 1) ForwardFFT_2D_mGPU_first(In, M_Input_yRegion, xFFT_Data_yRegion);
		else ForwardFFT_3D_mGPU_first(In, M_Input_yRegion, xFFT_Data_yRegion);
	}

	//multi-GPU version with mixed precision
	template <typename cuVECIn>
	void ForwardFFT_mGPU_first(cuVECIn& In, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M, cu_arr<cuBHalf*>& xFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xFFT)
	{
		if (n.z == 1) ForwardFFT_2D_mGPU_first(In, M_Input_yRegion_half, normalization_M, xFFT_Data_yRegion_half, normalization_xFFT);
		else ForwardFFT_3D_mGPU_first(In, M_Input_yRegion_half, normalization_M, xFFT_Data_yRegion_half, normalization_xFFT);
	}

	void ForwardFFT_mGPU_last(cu_arr<cuBComplex*>& xFFT_Data_xRegion)
	{
		if (n.z == 1) ForwardFFT_2D_mGPU_last(xFFT_Data_xRegion);
		else ForwardFFT_3D_mGPU_last(xFFT_Data_xRegion);
	}

	void ForwardFFT_mGPU_last(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization)
	{
		if (n.z == 1) ForwardFFT_2D_mGPU_last(xFFT_Data_xRegion_half, normalization);
		else ForwardFFT_3D_mGPU_last(xFFT_Data_xRegion_half, normalization);
	}

	//AVERAGED INPUTS

	template <typename cuVECIn>
	void ForwardFFT_AveragedInputs(cuVECIn& In1, cuVECIn& In2)
	{
		if (n.z == 1) ForwardFFT_2D(In1, In2);
		else ForwardFFT_3D(In1, In2);
	}

	//multi-GPU version
	template <typename cuVECIn>
	void ForwardFFT_AveragedInputs_mGPU_first(cuVECIn& In1, cuVECIn& In2, cu_arr<cuReal3*>& M_Input_yRegion, cu_arr<cuBComplex*>& xFFT_Data_yRegion)
	{
		if (n.z == 1) ForwardFFT_2D_AveragedInputs_mGPU_first(In1, In2, M_Input_yRegion, xFFT_Data_yRegion);
		else ForwardFFT_3D_AveragedInputs_mGPU_first(In1, In2, M_Input_yRegion, xFFT_Data_yRegion);
	}

	//multi-GPU version with mixed precision
	template <typename cuVECIn>
	void ForwardFFT_AveragedInputs_mGPU_first(cuVECIn& In1, cuVECIn& In2, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M, cu_arr<cuBHalf*>& xFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xFFT)
	{
		if (n.z == 1) ForwardFFT_2D_AveragedInputs_mGPU_first(In1, In2, M_Input_yRegion_half, normalization_M, xFFT_Data_yRegion_half, normalization_xFFT);
		else ForwardFFT_3D_AveragedInputs_mGPU_first(In1, In2, M_Input_yRegion_half, normalization_M, xFFT_Data_yRegion_half, normalization_xFFT);
	}

	//MULTIPLICATION
	
	void KernelMultiplication(void)
	{
		if (n.z == 1) static_cast<Owner*>(this)->KernelMultiplication_2D();
		else {

			if (!q2D_level) static_cast<Owner*>(this)->KernelMultiplication_3D();
			else static_cast<Owner*>(this)->KernelMultiplication_q2D(q2D_level);
		}
	}

	//2. (S -> S or S2) -> multiple input spaces version using a collection of FFT spaces (Kernel must be configured for this).
	//Also additional_spaces must be true, unless this method is used for multi-GPU convolution for a single mesh, since then S does not need to be reused after memory transfer to other GPUs.
	void KernelMultiplication_MultipleInputs(std::vector<cu_arr<cuBComplex>*>& Scol_x, std::vector<cu_arr<cuBComplex>*>& Scol_y, std::vector<cu_arr<cuBComplex>*>& Scol_z)
	{
		if (additional_spaces) {

			if (n.z == 1) static_cast<Owner*>(this)->KernelMultiplication_2D(Scol_x, Scol_y, Scol_z, cuS2_x, cuS2_y, cuS2_z);
			else {

				if (!q2D_level) {

					static_cast<Owner*>(this)->KernelMultiplication_3D(Scol_x, Scol_y, Scol_z, cuS2_x, cuS2_y, cuS2_z);
				}
				else {

					static_cast<Owner*>(this)->KernelMultiplication_q2D(Scol_x, Scol_y, Scol_z, cuS2_x, cuS2_y, cuS2_z);
				}
			}
		}
		else {

			if (n.z == 1) static_cast<Owner*>(this)->KernelMultiplication_2D(Scol_x, Scol_y, Scol_z, cuS_x, cuS_y, cuS_z);
			else {

				if (!q2D_level) {

					static_cast<Owner*>(this)->KernelMultiplication_3D(Scol_x, Scol_y, Scol_z, cuS_x, cuS_y, cuS_z);
				}
				else {

					static_cast<Owner*>(this)->KernelMultiplication_q2D(Scol_x, Scol_y, Scol_z, cuS_x, cuS_y, cuS_z);
				}
			}
		}
	}

	//SINGLE INPUT, SINGLE OUTPUT

	//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT(
		cuVECIn& In, cuVECOut& Out, 
		cuBReal& energy, bool get_energy, bool clearOut = true, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (n.z == 1) InverseFFT_2D(In, Out, energy, get_energy, clearOut, pH, penergy);
		else InverseFFT_3D(In, Out, energy, get_energy, clearOut, pH, penergy);
	}

	//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
	//This is useful when calculating an energy total from multiple meshes which might need different weights.
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT(
		cuVECIn& In, cuVECOut& Out, 
		cuBReal& energy, cuBReal& energy_weight, bool clearOut = true, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (n.z == 1) InverseFFT_2D(In, Out, energy, energy_weight, clearOut, pH, penergy);
		else InverseFFT_3D(In, Out, energy, energy_weight, clearOut, pH, penergy);
	}

	//multi-GPU version (needs to be broken down into 2 steps, first and last, to allow for data transfer after the two steps)
	//in the first step perform everything up to x-IFFT, capturing device data into linear spaces xIFFT_Data_xRegion
	void InverseFFT_mGPU_first(cu_arr<cuBComplex*>& xIFFT_Data_xRegion)
	{
		if (n.z == 1) {

			inverse_fft_2D_mGPU_ystep(xIFFT_Data_xRegion);
		}
		else {

			if (q2D_level) inverse_fft_q2D_mGPU_ystep(xIFFT_Data_xRegion);
			else inverse_fft_3D_mGPU_zysteps(xIFFT_Data_xRegion);
		}
	}
	
	//multi-GPU version with mixed precision
	void InverseFFT_mGPU_first(cu_arr<cuBHalf*>& xIFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization)
	{
		if (n.z == 1) {

			inverse_fft_2D_mGPU_ystep(xIFFT_Data_xRegion_half, normalization);
		}
		else {

			if (q2D_level) inverse_fft_q2D_mGPU_ystep(xIFFT_Data_xRegion_half, normalization);
			else inverse_fft_3D_mGPU_zysteps(xIFFT_Data_xRegion_half, normalization);
		}
	}

	//after first step, copy in data transferred from xIFFT_Data_xRegion to xIFFT_Data_yRegion, and perfrom x IFFT in each y region
	//copy results to Out_Data_yRegion so we can transfer to Out_Data_xRegion after
	void InverseFFT_mGPU_last(cu_arr<cuBComplex*>& xIFFT_Data_yRegion, cu_arr<cuReal3*>& Out_Data_yRegion)
	{
		if (n.z == 1) {

			inverse_fft_2D_mGPU_xstep(xIFFT_Data_yRegion, Out_Data_yRegion);
		}
		else {

			inverse_fft_3D_mGPU_xstep(xIFFT_Data_yRegion, Out_Data_yRegion);
		}
	}

	void InverseFFT_mGPU_last(cu_arr<cuBHalf*>& xIFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xIFFT, cu_arr<cuBHalf*>& Out_Data_yRegion_half, cu_obj<cuBReal>& normalization_Out)
	{
		if (n.z == 1) {

			inverse_fft_2D_mGPU_xstep(xIFFT_Data_yRegion_half, normalization_xIFFT, Out_Data_yRegion_half, normalization_Out);
		}
		else {

			inverse_fft_3D_mGPU_xstep(xIFFT_Data_yRegion_half, normalization_xIFFT, Out_Data_yRegion_half, normalization_Out);
		}
	}

	//mGPU version. In the last step perform data copying which was transferred from other devices, xIFFT, and finally setting output
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_mGPU_finish(
		cu_arr<cuReal3*>& Out_Data_xRegion,
		cuVECIn& In, cuVECOut& Out,
		cuBReal& energy, bool get_energy, bool clearOut = true,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (!clearOut) FinishConvolution_Add_mGPU(Out_Data_xRegion, In, Out, energy, get_energy, pH, penergy);
		else FinishConvolution_Set_mGPU(Out_Data_xRegion, In, Out, energy, get_energy, pH, penergy);
	}

	//mGPU version with mixed precision
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_mGPU_finish(
		cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
		cuVECIn& In, cuVECOut& Out,
		cuBReal& energy, bool get_energy, bool clearOut = true,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (!clearOut) FinishConvolution_Add_mGPU(Out_Data_xRegion_half, normalization, In, Out, energy, get_energy, pH, penergy);
		else FinishConvolution_Set_mGPU(Out_Data_xRegion_half, normalization, In, Out, energy, get_energy, pH, penergy);
	}

	//AVERAGED INPUTS, SINGLE OUTPUT

	//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_AveragedInputs(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, 
		cuBReal& energy, bool get_energy, bool clearOut = true, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (n.z == 1) InverseFFT_2D(In1, In2, Out, energy, get_energy, clearOut, pH, penergy);
		else InverseFFT_3D(In1, In2, Out, energy, get_energy, clearOut, pH, penergy);
	}

	//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
	//This is useful when calculating an energy total from multiple meshes which might need different weights.
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_AveragedInputs(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, 
		cuBReal& energy, cuBReal& energy_weight, bool clearOut = true, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (n.z == 1) InverseFFT_2D(In1, In2, Out, energy, energy_weight, clearOut, pH, penergy);
		else InverseFFT_3D(In1, In2, Out, energy, energy_weight, clearOut, pH, penergy);
	}

	//mGPU version. In the last step perform data copying which was transferred from other devices, xIFFT, and finally setting output
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_AveragedInputs_SingleOutput_mGPU_finish(
		cu_arr<cuReal3*>& Out_Data_xRegion,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out,
		cuBReal& energy, bool get_energy, bool clearOut = true,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (!clearOut) FinishConvolution_Add_mGPU(Out_Data_xRegion, In1, In2, Out, energy, get_energy, pH, penergy);
		else FinishConvolution_Set_mGPU(Out_Data_xRegion, In1, In2, Out, energy, get_energy, pH, penergy);
	}

	//mGPU version with mixed precision
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_AveragedInputs_SingleOutput_mGPU_finish(
		cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out,
		cuBReal& energy, bool get_energy, bool clearOut = true,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (!clearOut) FinishConvolution_Add_mGPU(Out_Data_xRegion_half, normalization, In1, In2, Out, energy, get_energy, pH, penergy);
		else FinishConvolution_Set_mGPU(Out_Data_xRegion_half, normalization, In1, In2, Out, energy, get_energy, pH, penergy);
	}

	//AVERAGED INPUTS, DUPLICATED OUTPUTS

	//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_AveragedInputs_DuplicatedOutputs(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, 
		cuBReal& energy, bool get_energy, bool clearOut = true, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (n.z == 1) InverseFFT_2D(In1, In2, Out1, Out2, energy, get_energy, clearOut, pH, penergy);
		else InverseFFT_3D(In1, In2, Out1, Out2, energy, get_energy, clearOut, pH, penergy);
	}

	//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
	//This is useful when calculating an energy total from multiple meshes which might need different weights.
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_AveragedInputs_DuplicatedOutputs(
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, 
		cuBReal& energy, cuBReal& energy_weight, bool clearOut = true, 
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (n.z == 1) InverseFFT_2D(In1, In2, Out1, Out2, energy, energy_weight, clearOut, pH, penergy);
		else InverseFFT_3D(In1, In2, Out1, Out2, energy, energy_weight, clearOut, pH, penergy);
	}

	//mGPU version. In the last step perform data copying which was transferred from other devices, xIFFT, and finally setting output
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_AveragedInputs_DuplicatedOutputs_mGPU_finish(
		cu_arr<cuReal3*>& Out_Data_xRegion,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2,
		cuBReal& energy, bool get_energy, bool clearOut = true,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (!clearOut) FinishConvolution_Add_mGPU(Out_Data_xRegion, In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
		else FinishConvolution_Set_mGPU(Out_Data_xRegion, In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
	}

	//mGPU version with mixed precision
	template <typename cuVECIn, typename cuVECOut>
	void InverseFFT_AveragedInputs_DuplicatedOutputs_mGPU_finish(
		cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
		cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2,
		cuBReal& energy, bool get_energy, bool clearOut = true,
		cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
	{
		if (!clearOut) FinishConvolution_Add_mGPU(Out_Data_xRegion_half, normalization, In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
		else FinishConvolution_Set_mGPU(Out_Data_xRegion_half, normalization, In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
	}
};

//-------------------------- CONSTRUCTORS

template <typename Owner, typename Kernel>
ConvolutionCUDA<Owner, Kernel>::ConvolutionCUDA(cuSZ3 n_, cuReal3 h_, CONV_ convtype, cuINT2 xRegion_) :
	ConvolutionDataCUDA(),
	Kernel()
{
	convolution_error_on_create = SetDimensions(n_, h_, convtype, cuINT3(), xRegion_);
}

//-------------------------- CONFIGURATION

template <typename Owner, typename Kernel>
BError ConvolutionCUDA<Owner, Kernel>::SetDimensions(cuSZ3 n_, cuReal3 h_, CONV_ convtype, cuINT3 pbc_images_, cuINT2 xRegion_, std::pair<int, int> devices_cfg)
{
	BError error(__FUNCTION__);

	error = SetConvolutionDimensions(n_, h_, convtype, pbc_images_, xRegion_, devices_cfg);
	if (!error) error = static_cast<Owner*>(this)->AllocateKernelMemory();
	if (error) { 
		
		error.reset(); 
		error = Set_Preserve_Zero_Padding(false); 
		if (!error) error = static_cast<Owner*>(this)->AllocateKernelMemory(); 
	}

	return error;
}

//Similar to SetDimensions but use a pre-determined N value
template <typename Owner, typename Kernel>
BError ConvolutionCUDA<Owner, Kernel>::SetDimensions(cuSZ3 n_, cuReal3 h_, cuSZ3 N_, CONV_ convtype, cuINT3 pbc_images_, cuINT2 xRegion_, std::pair<int, int> devices_cfg)
{
	BError error(__FUNCTION__);

	error = SetConvolutionDimensions(n_, h_, N_, convtype, pbc_images_, xRegion_, devices_cfg);
	if (!error) error = static_cast<Owner*>(this)->AllocateKernelMemory();
	if (error) {

		error.reset();
		error = Set_Preserve_Zero_Padding(false);
		if (!error) error = static_cast<Owner*>(this)->AllocateKernelMemory();
	}

	return error;
}

//-------------------------- RUN-TIME CONVOLUTION : 2D

//SINGLE INPUT, SINGLE OUTPUT

template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::Convolute_2D(
	cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Copy In to cufft arrays, setting all other points to zero
	CopyInputData(In);

	//Forward 2D FFT
	forward_fft_2D();

	//Multiplication with kernels
	static_cast<Owner*>(this)->KernelMultiplication_2D();

	//Inverse 2D FFT
	if (!additional_spaces) inverse_fft_2D();
	else inverse_fft_2D_2();

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In, Out, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In, Out, energy, get_energy, pH, penergy);
	}
}

//AVERAGED INPUTS, SINGLE OUTPUT

template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::Convolute_2D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Copy In to cufft arrays, setting all other points to zero
	AverageInputData(In1, In2);

	//Forward 2D FFT
	forward_fft_2D();

	//Multiplication with kernels
	static_cast<Owner*>(this)->KernelMultiplication_2D();

	//Inverse 2D FFT
	if (!additional_spaces) inverse_fft_2D();
	else inverse_fft_2D_2();

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out, energy, get_energy, pH, penergy);
	}
}

//AVERAGED INPUTS, DUPLICATED OUTPUTS

template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::Convolute_2D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Copy In to cufft arrays, setting all other points to zero
	AverageInputData(In1, In2);

	//Forward 2D FFT
	forward_fft_2D();

	//Multiplication with kernels
	static_cast<Owner*>(this)->KernelMultiplication_2D();

	//Inverse 2D FFT
	if (!additional_spaces) inverse_fft_2D();
	else inverse_fft_2D_2();

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
	}
}

//-------------------------- RUN-TIME CONVOLUTION : 3D

//SINGLE INPUT, SINGLE OUTPUT

template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::Convolute_3D(
	cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Copy In to cufft arrays, setting all other points to zero
	CopyInputData(In);

	if (!q2D_level) {

		//Forward 3D FFT
		forward_fft_3D();

		//Multiplication with kernels
		static_cast<Owner*>(this)->KernelMultiplication_3D();

		//Inverse 3D FFT
		if (!additional_spaces) inverse_fft_3D();
		else inverse_fft_3D_2();
	}
	else {

		//Forward q2D FFT
		forward_fft_q2D();

		//Multiplication with kernels with z-axis fft and ifft rolled into one step
		static_cast<Owner*>(this)->KernelMultiplication_q2D(q2D_level);

		//Inverse q2D FFT
		if (!additional_spaces) inverse_fft_q2D();
		else inverse_fft_q2D_2();
	}

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In, Out, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In, Out, energy, get_energy, pH, penergy);
	}
}

//AVERAGED INPUTS, SINGLE OUTPUT

template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::Convolute_3D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Copy In to cufft arrays, setting all other points to zero
	AverageInputData(In1, In2);

	if (!q2D_level) {

		//Forward 3D FFT
		forward_fft_3D();

		//Multiplication with kernels
		static_cast<Owner*>(this)->KernelMultiplication_3D();

		//Inverse 3D FFT
		if (!additional_spaces) inverse_fft_3D();
		else inverse_fft_3D_2();
	}
	else {

		//Forward q2D FFT
		forward_fft_q2D();

		//Multiplication with kernels with z-axis fft and ifft rolled into one step
		static_cast<Owner*>(this)->KernelMultiplication_q2D(q2D_level);

		//Inverse q2D FFT
		if (!additional_spaces) inverse_fft_q2D();
		else inverse_fft_q2D_2();
	}

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out, energy, get_energy, pH, penergy);
	}
}

//AVERAGED INPUTS, DUPLICATED OUTPUTS

template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::Convolute_3D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Copy In to cufft arrays, setting all other points to zero
	AverageInputData(In1, In2);

	if (!q2D_level) {

		//Forward 3D FFT
		forward_fft_3D();

		//Multiplication with kernels
		static_cast<Owner*>(this)->KernelMultiplication_3D();

		//Inverse 3D FFT
		if (!additional_spaces) inverse_fft_3D();
		else inverse_fft_3D_2();
	}
	else {

		//Forward q2D FFT
		forward_fft_q2D();

		//Multiplication with kernels with z-axis fft and ifft rolled into one step
		static_cast<Owner*>(this)->KernelMultiplication_q2D(q2D_level);

		//Inverse q2D FFT
		if (!additional_spaces) inverse_fft_q2D();
		else inverse_fft_q2D_2();
	}

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
	}
}

//-------------------------- RUN-TIME CONVOLUTION : 2D

//SINGLE INPUT

template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_2D(cuVECIn& In)
{
	//Copy In to cufft arrays, setting all other points to zero
	CopyInputData(In);

	//Forward 2D FFT
	forward_fft_2D();
}

//multi-GPU version
template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_2D_mGPU_first(cuVECIn& In, cu_arr<cuReal3*>& M_Input_yRegion, cu_arr<cuBComplex*>& xFFT_Data_yRegion)
{
	//Copy In and M_Input_yRegion to cufft arrays, setting all other points to zero
	CopyInputData_mGPU(In, M_Input_yRegion);

	//Forward 2D FFT
	forward_fft_2D_mGPU_xstep(xFFT_Data_yRegion);
}

//multi-GPU version with mixed precision
template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_2D_mGPU_first(cuVECIn& In, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M, cu_arr<cuBHalf*>& xFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xFFT)
{
	//Copy In and M_Input_yRegion to cufft arrays, setting all other points to zero
	CopyInputData_mGPU(In, M_Input_yRegion_half, normalization_M);

	//Forward 2D FFT
	forward_fft_2D_mGPU_xstep(xFFT_Data_yRegion_half, normalization_xFFT);
}

template <typename Owner, typename Kernel>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_2D_mGPU_last(cu_arr<cuBComplex*>& xFFT_Data_xRegion)
{
	forward_fft_2D_mGPU_ystep(xFFT_Data_xRegion);
}

template <typename Owner, typename Kernel>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_2D_mGPU_last(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization)
{
	forward_fft_2D_mGPU_ystep(xFFT_Data_xRegion_half, normalization);
}

//AVERAGED INPUTS

template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_2D(cuVECIn& In1, cuVECIn& In2)
{
	//Copy In to cufft arrays, setting all other points to zero
	AverageInputData(In1, In2);

	//Forward 2D FFT
	forward_fft_2D();
}

//multi-GPU version
template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_2D_AveragedInputs_mGPU_first(cuVECIn& In1, cuVECIn& In2, cu_arr<cuReal3*>& M_Input_yRegion, cu_arr<cuBComplex*>& xFFT_Data_yRegion)
{
	//Copy In and M_Input_yRegion to cufft arrays, setting all other points to zero
	CopyInputData_AveragedInputs_mGPU(In1, In2, M_Input_yRegion);

	//Forward 2D FFT
	forward_fft_2D_mGPU_xstep(xFFT_Data_yRegion);
}

//multi-GPU version with mixed precision
template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_2D_AveragedInputs_mGPU_first(cuVECIn& In1, cuVECIn& In2, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M, cu_arr<cuBHalf*>& xFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xFFT)
{
	//Copy In and M_Input_yRegion to cufft arrays, setting all other points to zero
	CopyInputData_AveragedInputs_mGPU(In1, In2, M_Input_yRegion_half, normalization_M);

	//Forward 2D FFT
	forward_fft_2D_mGPU_xstep(xFFT_Data_yRegion_half, normalization_xFFT);
}

//SINGLE INPUT, SINGLE OUTPUT

//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_2D(
	cuVECIn& In, cuVECOut& Out, 
	cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Inverse 2D FFT
	if (!additional_spaces) inverse_fft_2D();
	else inverse_fft_2D_2();

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In, Out, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In, Out, energy, get_energy, pH, penergy);
	}
}

//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
//This is useful when calculating an energy total from multiple meshes which might need different weights.
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_2D(
	cuVECIn& In, cuVECOut& Out, cuBReal& energy, 
	cuBReal& energy_weight, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Inverse 2D FFT
	if (!additional_spaces) inverse_fft_2D();
	else inverse_fft_2D_2();

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In, Out, energy, energy_weight, pH, penergy);
	}
	else {

		FinishConvolution_Add(In, Out, energy, energy_weight, pH, penergy);
	}
}

//AVERAGED INPUTS, SINGLE OUTPUT

//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_2D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, 
	cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Inverse 2D FFT
	if (!additional_spaces) inverse_fft_2D();
	else inverse_fft_2D_2();

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out, energy, get_energy, pH, penergy);
	}
}

//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
//This is useful when calculating an energy total from multiple meshes which might need different weights.
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_2D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, 
	cuBReal& energy, cuBReal& energy_weight, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Inverse 2D FFT
	if (!additional_spaces) inverse_fft_2D();
	else inverse_fft_2D_2();

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out, energy, energy_weight, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out, energy, energy_weight, pH, penergy);
	}
}

//AVERAGED INPUTS, DUPLICATED OUTPUTS

//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_2D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, 
	cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Inverse 2D FFT
	if (!additional_spaces) inverse_fft_2D();
	else inverse_fft_2D_2();

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
	}
}

//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
//This is useful when calculating an energy total from multiple meshes which might need different weights.
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_2D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, 
	cuBReal& energy, cuBReal& energy_weight, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	//Inverse 2D FFT
	if (!additional_spaces) inverse_fft_2D();
	else inverse_fft_2D_2();

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out1, Out2, energy, energy_weight, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out1, Out2, energy, energy_weight, pH, penergy);
	}
}

//-------------------------- RUN-TIME CONVOLUTION : 3D

//SINGLE INPUT

template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_3D(cuVECIn& In)
{
	//Copy In to cufft arrays, setting all other points to zero
	CopyInputData(In);

	if (!q2D_level) {

		//Forward 3D FFT
		forward_fft_3D();
	}
	else {

		//Forward q2D FFT
		forward_fft_q2D();
	}
}

//multi-GPU versions
template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_3D_mGPU_first(cuVECIn& In, cu_arr<cuReal3*>& M_Input_yRegion, cu_arr<cuBComplex*>& xFFT_Data_yRegion)
{
	//Copy In and M_Input_yRegion to cufft arrays, setting all other points to zero
	CopyInputData_mGPU(In, M_Input_yRegion);

	//Forward 3D FFT
	forward_fft_3D_mGPU_xstep(xFFT_Data_yRegion);
}

//mGPU mixed precision version
template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_3D_mGPU_first(cuVECIn& In, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M, cu_arr<cuBHalf*>& xFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xFFT)
{
	//Copy In and M_Input_yRegion to cufft arrays, setting all other points to zero
	CopyInputData_mGPU(In, M_Input_yRegion_half, normalization_M);

	//Forward 3D FFT
	forward_fft_3D_mGPU_xstep(xFFT_Data_yRegion_half, normalization_xFFT);
}

template <typename Owner, typename Kernel>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_3D_mGPU_last(cu_arr<cuBComplex*>& xFFT_Data_xRegion)
{
	if (!q2D_level) {

		//Forward 3D FFT
		forward_fft_3D_mGPU_yzsteps(xFFT_Data_xRegion);
	}
	else {

		//Forward q2D FFT
		forward_fft_q2D_mGPU_ystep(xFFT_Data_xRegion);
	}
}

template <typename Owner, typename Kernel>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_3D_mGPU_last(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization)
{
	if (!q2D_level) {

		//Forward 3D FFT
		forward_fft_3D_mGPU_yzsteps(xFFT_Data_xRegion_half, normalization);
	}
	else {

		//Forward q2D FFT
		forward_fft_q2D_mGPU_ystep(xFFT_Data_xRegion_half, normalization);
	}
}

//AVERAGED INPUTS

template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_3D(cuVECIn& In1, cuVECIn& In2)
{
	//Copy In to cufft arrays, setting all other points to zero
	AverageInputData(In1, In2);

	if (!q2D_level) {

		//Forward 3D FFT
		forward_fft_3D();
	}
	else {

		//Forward q2D FFT
		forward_fft_q2D();
	}
}

//multi-GPU versions
template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_3D_AveragedInputs_mGPU_first(cuVECIn& In1, cuVECIn& In2, cu_arr<cuReal3*>& M_Input_yRegion, cu_arr<cuBComplex*>& xFFT_Data_yRegion)
{
	//Copy In and M_Input_yRegion to cufft arrays, setting all other points to zero
	CopyInputData_AveragedInputs_mGPU(In1, In2, M_Input_yRegion);

	//Forward 3D FFT
	forward_fft_3D_mGPU_xstep(xFFT_Data_yRegion);
}

//multi-GPU version with mixed precision
template <typename Owner, typename Kernel>
template <typename cuVECIn>
void ConvolutionCUDA<Owner, Kernel>::ForwardFFT_3D_AveragedInputs_mGPU_first(cuVECIn& In1, cuVECIn& In2, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M, cu_arr<cuBHalf*>& xFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xFFT)
{
	//Copy In and M_Input_yRegion to cufft arrays, setting all other points to zero
	CopyInputData_AveragedInputs_mGPU(In1, In2, M_Input_yRegion_half, normalization_M);

	//Forward 2D FFT
	forward_fft_3D_mGPU_xstep(xFFT_Data_yRegion_half, normalization_xFFT);
}

//SINGLE INPUT, SINGLE OUTPUT

//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_3D(
	cuVECIn& In, cuVECOut& Out, 
	cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (!q2D_level) {

		//Inverse 3D FFT
		if (!additional_spaces) inverse_fft_3D();
		else inverse_fft_3D_2();
	}
	else {

		//Inverse q2D FFT
		if (!additional_spaces) inverse_fft_q2D();
		else inverse_fft_q2D_2();
	}

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In, Out, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In, Out, energy, get_energy, pH, penergy);
	}
}


//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
//This is useful when calculating an energy total from multiple meshes which might need different weights.
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_3D(
	cuVECIn& In, cuVECOut& Out, 
	cuBReal& energy, cuBReal& energy_weight, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (!q2D_level) {

		//Inverse 3D FFT
		if (!additional_spaces) inverse_fft_3D();
		else inverse_fft_3D_2();
	}
	else {

		//Inverse q2D FFT
		if (!additional_spaces) inverse_fft_q2D();
		else inverse_fft_q2D_2();
	}

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In, Out, energy, energy_weight, pH, penergy);
	}
	else {

		FinishConvolution_Add(In, Out, energy, energy_weight, pH, penergy);
	}
}

//AVERAGED INPUTS, SINGLE OUTPUT

//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_3D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, 
	cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (!q2D_level) {

		//Inverse 3D FFT
		if (!additional_spaces) inverse_fft_3D();
		else inverse_fft_3D_2();
	}
	else {

		//Inverse q2D FFT
		if (!additional_spaces) inverse_fft_q2D();
		else inverse_fft_q2D_2();
	}

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out, energy, get_energy, pH, penergy);
	}
}


//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
//This is useful when calculating an energy total from multiple meshes which might need different weights.
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_3D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, 
	cuBReal& energy, cuBReal& energy_weight, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (!q2D_level) {

		//Inverse 3D FFT
		if (!additional_spaces) inverse_fft_3D();
		else inverse_fft_3D_2();
	}
	else {

		//Inverse q2D FFT
		if (!additional_spaces) inverse_fft_q2D();
		else inverse_fft_q2D_2();
	}

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out, energy, energy_weight, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out, energy, energy_weight, pH, penergy);
	}
}

//AVERAGED INPUTS, DUPLICATED OUTPUTS

//inverse FFT with the option of calculating an energy contribution ((-MU0/2)(In * Out) / input_non_empty_cells
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_3D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, 
	cuBReal& energy, bool get_energy, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (!q2D_level) {

		//Inverse 3D FFT
		if (!additional_spaces) inverse_fft_3D();
		else inverse_fft_3D_2();
	}
	else {

		//Inverse q2D FFT
		if (!additional_spaces) inverse_fft_q2D();
		else inverse_fft_q2D_2();
	}

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out1, Out2, energy, get_energy, pH, penergy);
	}
}


//inverse FFT with calculation of a weighted energy contribution: weight*((-MU0/2)(In * Out) / input_non_empty_cells.
//This is useful when calculating an energy total from multiple meshes which might need different weights.
template <typename Owner, typename Kernel>
template <typename cuVECIn, typename cuVECOut>
void ConvolutionCUDA<Owner, Kernel>::InverseFFT_3D(
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, 
	cuBReal& energy, cuBReal& energy_weight, bool clearOut,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (!q2D_level) {

		//Inverse 3D FFT
		if (!additional_spaces) inverse_fft_3D();
		else inverse_fft_3D_2();
	}
	else {

		//Inverse q2D FFT
		if (!additional_spaces) inverse_fft_q2D();
		else inverse_fft_q2D_2();
	}

	//Copy cufft arrays to Heff
	if (clearOut) {

		FinishConvolution_Set(In1, In2, Out1, Out2, energy, energy_weight, pH, penergy);
	}
	else {

		FinishConvolution_Add(In1, In2, Out1, Out2, energy, energy_weight, pH, penergy);
	}
}

#endif