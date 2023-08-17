#include "DemagKernelCollectionCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_SDEMAG) || defined(MODULE_COMPILATION_DEMAG)

#include "DemagKernelCollectionCUDA_KerType.h"

//THESE ARE IN CURRENT USE

//-------------------------- STAND-ALONE VERSIONS FOR MULTIPLE INPUTS

//-------------------------- CONVOLUTION PRODUCT CUDA KERNELS

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// 2D

//Self demag kernel multiplication - real and use symmetries
//N = (N.x/2 + 1, N.y, 1)
__global__ void cu_KernelMultiplication_2D_Self(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz, 
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	//above N.y/2 use kernel symmetries to recover kernel values
	//diagonal components are even about the N.y/2 point
	//off-diagonal values are odd about the N.y/2 point

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		int j = (idx / (N.x / 2 + 1)) % N.y;

		if (j <= N.y / 2) {

			cuS2x[idx] = (kernel.Kdiag_real[idx].x  * FMx) + (kernel.K2D_odiag[idx] * FMy);
			cuS2y[idx] = (kernel.K2D_odiag[idx] * FMx) + (kernel.Kdiag_real[idx].y  * FMy);
			cuS2z[idx] = (kernel.Kdiag_real[idx].z  * FMz);
		}
		else {

			int i = idx % (N.x / 2 + 1);

			int ker_idx = i + (N.y - j) * (N.x / 2 + 1);

			cuS2x[idx] = (kernel.Kdiag_real[ker_idx].x  * FMx) + (-kernel.K2D_odiag[ker_idx] * FMy);
			cuS2y[idx] = (-kernel.K2D_odiag[ker_idx] * FMx) + (kernel.Kdiag_real[ker_idx].y  * FMy);
			cuS2z[idx] = (kernel.Kdiag_real[ker_idx].z  * FMz);
		}
	}
}

//as above, but add to output, not set it
__global__ void cu_KernelMultiplication_2D_Self_add(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	//above N.y/2 use kernel symmetries to recover kernel values
	//diagonal components are even about the N.y/2 point
	//off-diagonal values are odd about the N.y/2 point

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		int j = (idx / (N.x / 2 + 1)) % N.y;

		if (j <= N.y / 2) {

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (kernel.Kdiag_real[idx].x  * FMx) + (kernel.K2D_odiag[idx] * FMy);
			cuS2y[idx] = (cuReIm)cuS2y[idx] + (kernel.K2D_odiag[idx] * FMx) + (kernel.Kdiag_real[idx].y  * FMy);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + (kernel.Kdiag_real[idx].z  * FMz);
		}
		else {

			int i = idx % (N.x / 2 + 1);

			int ker_idx = i + (N.y - j) * (N.x / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (kernel.Kdiag_real[ker_idx].x  * FMx) + (-kernel.K2D_odiag[ker_idx] * FMy);
			cuS2y[idx] = (cuReIm)cuS2y[idx] + (-kernel.K2D_odiag[ker_idx] * FMx) + (kernel.Kdiag_real[ker_idx].y  * FMy);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + (kernel.Kdiag_real[ker_idx].z  * FMz);
		}
	}
}

//Self demag kernel multiplication - real and use symmetries
__global__ void cu_KernelMultiplication_2D_Self_transpose_xy(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	//above N.y/2 use kernel symmetries to recover kernel values
	//diagonal components are even about the N.y/2 point
	//off-diagonal values are odd about the N.y/2 point

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_idx = i + j * (N.y / 2 + 1);

			cuS2x[idx] = (kernel.Kdiag_real[ker_idx].x  * FMx) + (kernel.K2D_odiag[ker_idx] * FMy);
			cuS2y[idx] = (kernel.K2D_odiag[ker_idx] * FMx) + (kernel.Kdiag_real[ker_idx].y  * FMy);
			cuS2z[idx] = (kernel.Kdiag_real[ker_idx].z  * FMz);
		}
		else {

			int ker_idx = (N.y - i) + j * (N.y / 2 + 1);

			cuS2x[idx] = (kernel.Kdiag_real[ker_idx].x  * FMx) + (-kernel.K2D_odiag[ker_idx] * FMy);
			cuS2y[idx] = (-kernel.K2D_odiag[ker_idx] * FMx) + (kernel.Kdiag_real[ker_idx].y  * FMy);
			cuS2z[idx] = (kernel.Kdiag_real[ker_idx].z  * FMz);
		}
	}
}

//as above, but add to output, not set it
__global__ void cu_KernelMultiplication_2D_Self_transpose_xy_add(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	//above N.y/2 use kernel symmetries to recover kernel values
	//diagonal components are even about the N.y/2 point
	//off-diagonal values are odd about the N.y/2 point

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_idx = i + j * (N.y / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (kernel.Kdiag_real[ker_idx].x  * FMx) + (kernel.K2D_odiag[ker_idx] * FMy);
			cuS2y[idx] = (cuReIm)cuS2y[idx] + (kernel.K2D_odiag[ker_idx] * FMx) + (kernel.Kdiag_real[ker_idx].y  * FMy);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + (kernel.Kdiag_real[ker_idx].z  * FMz);
		}
		else {

			int ker_idx = (N.y - i) + j * (N.y / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (kernel.Kdiag_real[ker_idx].x  * FMx) + (-kernel.K2D_odiag[ker_idx] * FMy);
			cuS2y[idx] = (cuReIm)cuS2y[idx] + (-kernel.K2D_odiag[ker_idx] * FMx) + (kernel.Kdiag_real[ker_idx].y  * FMy);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + (kernel.Kdiag_real[ker_idx].z  * FMz);
		}
	}
}

//Self demag kernel multiplication - real and use symmetries
//N = (N.x/2 + 1, N.y, 1)
__global__ void cu_KernelMultiplication_2D_zShifted(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		cuVEC<cuReal3>& Kdiag = kernel.Kdiag_real;
		cuVEC<cuReal3>& Kodiag = kernel.Kodiag_real;

		int j = (idx / (N.x / 2 + 1)) % N.y;

		if (j <= N.y / 2) {

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[idx].x * FMx) + (Kodiag[idx].x * FMy) + !(Kodiag[idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[idx].x * FMx) + (Kdiag[idx].y * FMy) + !(Kodiag[idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + !(Kodiag[idx].y * FMx) + !(Kodiag[idx].z * FMy) + (Kdiag[idx].z * FMz);
		}
		else {

			int i = idx % (N.x / 2 + 1);

			int ker_idx = i + (N.y - j) * (N.x / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) - (Kodiag[ker_idx].x * FMy) + !(Kodiag[ker_idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] - (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) - !(Kodiag[ker_idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + !(Kodiag[ker_idx].y * FMx) - !(Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
		}
	}
}

__global__ void cu_KernelMultiplication_2D_zShifted_transpose_xy(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		cuVEC<cuReal3>& Kdiag = kernel.Kdiag_real;
		cuVEC<cuReal3>& Kodiag = kernel.Kodiag_real;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_idx = i + j * (N.y / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) + (Kodiag[ker_idx].x * FMy) + !(Kodiag[ker_idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + !(Kodiag[ker_idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + !(Kodiag[ker_idx].y * FMx) + !(Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
		}
		else {

			int ker_idx = (N.y - i) + j * (N.y / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) - (Kodiag[ker_idx].x * FMy) + !(Kodiag[ker_idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] - (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) - !(Kodiag[ker_idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + !(Kodiag[ker_idx].y * FMx) - !(Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
		}
	}
}

//Self demag kernel multiplication - real and use symmetries, adjusting for inverse z shift
//N = (N.x/2 + 1, N.y, 1)
__global__ void cu_KernelMultiplication_2D_inversezShifted(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		cuVEC<cuReal3>& Kdiag = kernel.Kdiag_real;
		cuVEC<cuReal3>& Kodiag = kernel.Kodiag_real;

		int j = (idx / (N.x / 2 + 1)) % N.y;

		if (j <= N.y / 2) {

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[idx].x * FMx) + (Kodiag[idx].x * FMy) - !(Kodiag[idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[idx].x * FMx) + (Kdiag[idx].y * FMy) - !(Kodiag[idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] - !(Kodiag[idx].y * FMx) - !(Kodiag[idx].z * FMy) + (Kdiag[idx].z * FMz);
		}
		else {

			int i = idx % (N.x / 2 + 1);

			int ker_idx = i + (N.y - j) * (N.x / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) - (Kodiag[ker_idx].x * FMy) - !(Kodiag[ker_idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] - (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + !(Kodiag[ker_idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] - !(Kodiag[ker_idx].y * FMx) + !(Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
		}
	}
}

__global__ void cu_KernelMultiplication_2D_inversezShifted_transpose_xy(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		cuVEC<cuReal3>& Kdiag = kernel.Kdiag_real;
		cuVEC<cuReal3>& Kodiag = kernel.Kodiag_real;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_idx = i + j * (N.y / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) + (Kodiag[ker_idx].x * FMy) - !(Kodiag[ker_idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) - !(Kodiag[ker_idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] - !(Kodiag[ker_idx].y * FMx) - !(Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
		}
		else {

			int ker_idx = (N.y - i) + j * (N.y / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) - (Kodiag[ker_idx].x * FMy) - !(Kodiag[ker_idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] - (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + !(Kodiag[ker_idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] - !(Kodiag[ker_idx].y * FMx) + !(Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
		}
	}
}

//Self demag kernel multiplication - complex but use symmetries
//N = (N.x/2 + 1, N.y, 1)
__global__ void cu_KernelMultiplication_2D_xShifted(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		cuVEC<cuReIm3>& Kdiag = kernel.Kdiag_cmpl;
		cuVEC<cuReIm3>& Kodiag = kernel.Kodiag_cmpl;

		int j = (idx / (N.x / 2 + 1)) % N.y;

		if (j <= N.y / 2) {

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[idx].x * FMx) + (Kodiag[idx].x * FMy) + (Kodiag[idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[idx].x * FMx) + (Kdiag[idx].y * FMy) + (Kodiag[idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[idx].y * FMx) + (Kodiag[idx].z * FMy) + (Kdiag[idx].z * FMz);
		}
		else {

			int i = idx % (N.x / 2 + 1);

			int ker_idx = i + (N.y - j) * (N.x / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) - (Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] - (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) - (Kodiag[ker_idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[ker_idx].y * FMx) - (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
		}
	}
}

__global__ void cu_KernelMultiplication_2D_xShifted_transpose_xy(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		cuVEC<cuReIm3>& Kdiag = kernel.Kdiag_cmpl;
		cuVEC<cuReIm3>& Kodiag = kernel.Kodiag_cmpl;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_idx = i + j * (N.y / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) + (Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (Kodiag[ker_idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[ker_idx].y * FMx) + (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
		}
		else {

			int ker_idx = (N.y - i) + j * (N.y / 2 + 1);

			cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) - (Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
			cuS2y[idx] = (cuReIm)cuS2y[idx] - (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) - (Kodiag[ker_idx].z * FMz);
			cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[ker_idx].y * FMx) - (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
		}
	}
}

//Complex kernel multiplication with no symmetries used
//N = (N.x/2 + 1, N.y, 1)
__global__ void cu_KernelMultiplication_2D_Regular(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y) {

		cuVEC<cuReIm3>& Kdiag = kernel.Kdiag_cmpl;
		cuVEC<cuReIm3>& Kodiag = kernel.Kodiag_cmpl;

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[idx].x * FMx) + (Kodiag[idx].x * FMy) + (Kodiag[idx].y * FMz);
		cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[idx].x * FMx) + (Kdiag[idx].y * FMy) + (Kodiag[idx].z * FMz);
		cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[idx].y * FMx) + (Kodiag[idx].z * FMy) + (Kdiag[idx].z * FMz);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// 3D

//N = (N.x/2 + 1, N.y, N.z)
__global__ void cu_KernelMultiplication_3D_Self_transpose_xy(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz, 
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	//above N.z/2 and N.y/2 use kernel symmetries to recover kernel values
	//diagonal components are even about the N.z/2 and N.y/2 points
	//Kxy is even about N.z/2 and odd about N.y/2
	//Kxz is odd about N.z/2 and even about N.y/2
	//Kyz is odd about N.z/2 and odd about N.y/2

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y * N.z) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);
		int k = idx / ((N.x / 2 + 1) * N.y);

		cuVEC<cuReal3>& Kdiag = kernel.Kdiag_real;
		cuVEC<cuReal3>& Kodiag = kernel.Kodiag_real;

		if (k <= N.z / 2) {

			if (i <= N.y / 2) {

				int ker_idx = i + j * (N.y / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);

				cuS2x[idx] = (Kdiag[ker_idx].x * FMx) + (Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (Kodiag[ker_idx].y * FMx) + (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
			else {

				int ker_idx = (N.y - i) + j * (N.y / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);

				cuS2x[idx] = (Kdiag[ker_idx].x * FMx) + (-Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (-Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (-Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (Kodiag[ker_idx].y * FMx) + (-Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
		}
		else {

			if (i <= N.y / 2) {

				int ker_idx = i + j * (N.y / 2 + 1) + (N.z - k) * (N.x / 2 + 1) * (N.y / 2 + 1);

				cuS2x[idx] = (Kdiag[ker_idx].x * FMx) + (Kodiag[ker_idx].x * FMy) + (-Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (-Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (-Kodiag[ker_idx].y * FMx) + (-Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
			else {

				int ker_idx = (N.y - i) + j * (N.y / 2 + 1) + (N.z - k) * (N.x / 2 + 1) * (N.y / 2 + 1);

				cuS2x[idx] = (Kdiag[ker_idx].x * FMx) + (-Kodiag[ker_idx].x * FMy) + (-Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (-Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (-Kodiag[ker_idx].y * FMx) + (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
		}
	}
}

//as above, but add to output, not set
__global__ void cu_KernelMultiplication_3D_Self_transpose_xy_add(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	//above N.z/2 and N.y/2 use kernel symmetries to recover kernel values
	//diagonal components are even about the N.z/2 and N.y/2 points
	//Kxy is even about N.z/2 and odd about N.y/2
	//Kxz is odd about N.z/2 and even about N.y/2
	//Kyz is odd about N.z/2 and odd about N.y/2

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y * N.z) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);
		int k = idx / ((N.x / 2 + 1) * N.y);

		cuVEC<cuReal3>& Kdiag = kernel.Kdiag_real;
		cuVEC<cuReal3>& Kodiag = kernel.Kodiag_real;

		if (k <= N.z / 2) {

			if (i <= N.y / 2) {

				int ker_idx = i + j * (N.y / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);

				cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) + (Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[ker_idx].y * FMx) + (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
			else {

				int ker_idx = (N.y - i) + j * (N.y / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);

				cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) + (-Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] + (-Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (-Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[ker_idx].y * FMx) + (-Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
		}
		else {

			if (i <= N.y / 2) {

				int ker_idx = i + j * (N.y / 2 + 1) + (N.z - k) * (N.x / 2 + 1) * (N.y / 2 + 1);

				cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) + (Kodiag[ker_idx].x * FMy) + (-Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (-Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] + (-Kodiag[ker_idx].y * FMx) + (-Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
			else {

				int ker_idx = (N.y - i) + j * (N.y / 2 + 1) + (N.z - k) * (N.x / 2 + 1) * (N.y / 2 + 1);

				cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) + (-Kodiag[ker_idx].x * FMy) + (-Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] + (-Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] + (-Kodiag[ker_idx].y * FMx) + (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
		}
	}
}

//z shifted for 3D : complex kernels, but use kernel symmetries
//N = (N.x/2 + 1, N.y, N.z)
__global__ void cu_KernelMultiplication_3D_zShifted_transpose_xy(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	//z shifted for 3D : can use kernels of reduced dimensions but must be complex
	//
	//Kxx : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//Kyy : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//Kzz : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//
	//Kxy : y - inv. symmetric (-), z - Re part symmetrical  (+), Im part inv. symmetric (-)
	//Kxz : y - symmetrical  (+), z - Re part inv. symmetric (-), Im part symmetrical  (+)
	//Kyz : y - inv. symmetric (-), z - Re part inv. symmetric (-), Im part symmetrical  (+)

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y * N.z) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);
		int k = idx / ((N.x / 2 + 1) * N.y);

		cuVEC<cuReIm3>& Kdiag = kernel.Kdiag_cmpl;
		cuVEC<cuReIm3>& Kodiag = kernel.Kodiag_cmpl;

		if (k <= N.z / 2) {

			if (i <= N.y / 2) {

				int ker_idx = i + j * (N.y / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);
				
				//lower z, lower y
				cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) + (Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[ker_idx].y * FMx) + (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
			else {

				int ker_idx = (N.y - i) + j * (N.y / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);

				//lower z, upper y
				cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) - (Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] - (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) - (Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[ker_idx].y * FMx) - (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
		}
		else {

			if (i <= N.y / 2) {

				int ker_idx = i + j * (N.y / 2 + 1) + (N.z - k) * (N.x / 2 + 1) * (N.y / 2 + 1);

				//upper z, lower y
				cuS2x[idx] = (cuReIm)cuS2x[idx] + ((~Kdiag[ker_idx].x) * FMx) + ((~Kodiag[ker_idx].x) * FMy) - ((~Kodiag[ker_idx].y) * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] + ((~Kodiag[ker_idx].x) * FMx) + ((~Kdiag[ker_idx].y) * FMy) - ((~Kodiag[ker_idx].z) * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] - ((~Kodiag[ker_idx].y) * FMx) - ((~Kodiag[ker_idx].z) * FMy) + ((~Kdiag[ker_idx].z) * FMz);
			}
			else {

				int ker_idx = (N.y - i) + j * (N.y / 2 + 1) + (N.z - k) * (N.x / 2 + 1) * (N.y / 2 + 1);

				//upper z, upper y
				cuS2x[idx] = (cuReIm)cuS2x[idx] + ((~Kdiag[ker_idx].x) * FMx) - ((~Kodiag[ker_idx].x) * FMy) - ((~Kodiag[ker_idx].y) * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] - ((~Kodiag[ker_idx].x) * FMx) + ((~Kdiag[ker_idx].y) * FMy) + ((~Kodiag[ker_idx].z) * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] - ((~Kodiag[ker_idx].y) * FMx) + ((~Kodiag[ker_idx].z) * FMy) + ((~Kdiag[ker_idx].z) * FMz);
			}
		}
	}
}

//x shifted for 3D : complex kernels, but use kernel symmetries
//N = (N.x/2 + 1, N.y, N.z)
__global__ void cu_KernelMultiplication_3D_xShifted_transpose_xy(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	//x shifted for 3D : can use kernels of reduced dimensions but must be complex
	//
	//Kxx, Kyy, Kzz : symmetrical (+) in y and z directions.
	//
	//Kxy : y - inv. symmetric (-), z - symmetrical  (+)
	//Kxz : y - symmetrical  (+), z - inv. symmetric (-)
	//Kyz : y - inv. symmetric (-), z - inv. symmetric (-)

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y * N.z) {

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);
		int k = idx / ((N.x / 2 + 1) * N.y);

		cuVEC<cuReIm3>& Kdiag = kernel.Kdiag_cmpl;
		cuVEC<cuReIm3>& Kodiag = kernel.Kodiag_cmpl;

		if (k <= N.z / 2) {

			if (i <= N.y / 2) {

				int ker_idx = i + j * (N.y / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);

				//lower z, lower y
				cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) + (Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[ker_idx].y * FMx) + (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
			else {

				int ker_idx = (N.y - i) + j * (N.y / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);

				//lower z, upper y
				cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) - (Kodiag[ker_idx].x * FMy) + (Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] - (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) - (Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[ker_idx].y * FMx) - (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
		}
		else {

			if (i <= N.y / 2) {

				int ker_idx = i + j * (N.y / 2 + 1) + (N.z - k) * (N.x / 2 + 1) * (N.y / 2 + 1);

				//upper z, lower y
				cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) + (Kodiag[ker_idx].x * FMy) - (Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) - (Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] - (Kodiag[ker_idx].y * FMx) - (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
			else {

				int ker_idx = (N.y - i) + j * (N.y / 2 + 1) + (N.z - k) * (N.x / 2 + 1) * (N.y / 2 + 1);

				//upper z, upper y
				cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[ker_idx].x * FMx) - (Kodiag[ker_idx].x * FMy) - (Kodiag[ker_idx].y * FMz);
				cuS2y[idx] = (cuReIm)cuS2y[idx] - (Kodiag[ker_idx].x * FMx) + (Kdiag[ker_idx].y * FMy) + (Kodiag[ker_idx].z * FMz);
				cuS2z[idx] = (cuReIm)cuS2z[idx] - (Kodiag[ker_idx].y * FMx) + (Kodiag[ker_idx].z * FMy) + (Kdiag[ker_idx].z * FMz);
			}
		}
	}
}

//Complex kernel multiplication with no symmetries used
//N = (N.x/2 + 1, N.y, N.z)
__global__ void cu_KernelMultiplication_3D_Regular(
	cuKerType& kernel,
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuBComplex* cuS2x, cuBComplex* cuS2y, cuBComplex* cuS2z,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < (N.x / 2 + 1) * N.y * N.z) {

		cuVEC<cuReIm3>& Kdiag = kernel.Kdiag_cmpl;
		cuVEC<cuReIm3>& Kodiag = kernel.Kodiag_cmpl;

		cuReIm FMx = cuSx[idx];
		cuReIm FMy = cuSy[idx];
		cuReIm FMz = cuSz[idx];

		cuS2x[idx] = (cuReIm)cuS2x[idx] + (Kdiag[idx].x * FMx) + (Kodiag[idx].x * FMy) + (Kodiag[idx].y * FMz);
		cuS2y[idx] = (cuReIm)cuS2y[idx] + (Kodiag[idx].x * FMx) + (Kdiag[idx].y * FMy) + (Kodiag[idx].z * FMz);
		cuS2z[idx] = (cuReIm)cuS2z[idx] + (Kodiag[idx].y * FMx) + (Kodiag[idx].z * FMy) + (Kdiag[idx].z * FMz);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// Q2D - SELF DEMAG

//N = (N.x/2 + 1, N.y, 4)
//xy is transposed
//Real kernels : this is a self demag version which sets output (unless set_output is false), but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_4_transpose_xy(
	cuVEC<cuReal3>& Kdiag, cuVEC<cuReal3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N, bool set_output = true)
{
	//above N.z/2 and N.y/2 use kernel symmetries to recover kernel values
	//diagonal components are even about the N.z/2 and N.y/2 points
	//Kxy is even about N.z/2 and odd about N.y/2
	//Kxz is odd about N.z/2 and even about N.y/2
	//Kyz is odd about N.z/2 and odd about N.y/2

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 4, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

		//the z-axis points (the others are zero)
		cuReIm3 a = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 b = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);

		//forward z-axis fft
		//NOTE: cuda fft uses -i for the forward fft and +i for the inverse fft.
		//The kernels are purely real so you would get the same result by taking +i for the forward and -i for the inverse, but better to keep it consistent : use the cuda fft convention here.
		cuReIm3 X0 = a + b;
		cuReIm3 X1 = a - !b;
		cuReIm3 X2 = a - b;
		cuReIm3 X3 = a + !b;

		//kernel multiplication
		cuReIm3 F0, F1, F2, F3;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F1.x = (Kdiag[ker_baseidx + kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + kerplanecount].y * X1.z);
			F1.y = (Kodiag[ker_baseidx + kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + kerplanecount].z * X1.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			F3.x = (Kdiag[ker_baseidx + kerplanecount].x * X3.x) + (Kodiag[ker_baseidx + kerplanecount].x * X3.y) + (-Kodiag[ker_baseidx + kerplanecount].y * X3.z);
			F3.y = (Kodiag[ker_baseidx + kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + kerplanecount].y * X3.y) + (-Kodiag[ker_baseidx + kerplanecount].z * X3.z);
			F3.z = (-Kodiag[ker_baseidx + kerplanecount].y * X3.x) + (-Kodiag[ker_baseidx + kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + kerplanecount].z * X3.z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (-Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (-Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (-Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (-Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F1.x = (Kdiag[ker_baseidx + kerplanecount].x * X1.x) + (-Kodiag[ker_baseidx + kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + kerplanecount].y * X1.z);
			F1.y = (-Kodiag[ker_baseidx + kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + kerplanecount].y * X1.y) + (-Kodiag[ker_baseidx + kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + kerplanecount].y * X1.x) + (-Kodiag[ker_baseidx + kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + kerplanecount].z * X1.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			F3.x = (Kdiag[ker_baseidx + kerplanecount].x * X3.x) + (-Kodiag[ker_baseidx + kerplanecount].x * X3.y) + (-Kodiag[ker_baseidx + kerplanecount].y * X3.z);
			F3.y = (-Kodiag[ker_baseidx + kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + kerplanecount].y * X3.y) + (Kodiag[ker_baseidx + kerplanecount].z * X3.z);
			F3.z = (-Kodiag[ker_baseidx + kerplanecount].y * X3.x) + (Kodiag[ker_baseidx + kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + kerplanecount].z * X3.z);
		}

		//store result in output ready for ifft (but will need to accumulate all other contributions before ifft so cannot compute it here)
		
		if (set_output) {

			cuSx_out[idx] = F0.x;
			cuSy_out[idx] = F0.y;
			cuSz_out[idx] = F0.z;

			cuSx_out[idx + planecount] = F1.x;
			cuSy_out[idx + planecount] = F1.y;
			cuSz_out[idx + planecount] = F1.z;

			cuSx_out[idx + 2 * planecount] = F2.x;
			cuSy_out[idx + 2 * planecount] = F2.y;
			cuSz_out[idx + 2 * planecount] = F2.z;

			cuSx_out[idx + 3 * planecount] = F3.x;
			cuSy_out[idx + 3 * planecount] = F3.y;
			cuSz_out[idx + 3 * planecount] = F3.z;
		}
		else {

			cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
			cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
			cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

			cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
			cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
			cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

			cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
			cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
			cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

			cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
			cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
			cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;
		}
	}
}

//N = (N.x/2 + 1, N.y, 8)
//xy is transposed
//Real kernels : this is a self demag version which sets output (unless set_output is false), but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_8_transpose_xy(
	cuVEC<cuReal3>& Kdiag, cuVEC<cuReal3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N, bool set_output = true)
{
	//above N.z/2 and N.y/2 use kernel symmetries to recover kernel values
	//diagonal components are even about the N.z/2 and N.y/2 points
	//Kxy is even about N.z/2 and odd about N.y/2
	//Kxz is odd about N.z/2 and even about N.y/2
	//Kyz is odd about N.z/2 and odd about N.y/2

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 8, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

#define a (cuBReal)0.7071067811865

		//the z-axis points (the others are zero)
		cuReIm3 x0 = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 x1 = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);
		cuReIm3 x2 = cuReIm3(cuSx_in[idx + 2 * planecount], cuSy_in[idx + 2 * planecount], cuSz_in[idx + 2 * planecount]);
		cuReIm3 x3 = cuReIm3(cuSx_in[idx + 3 * planecount], cuSy_in[idx + 3 * planecount], cuSz_in[idx + 3 * planecount]);

		//Radix-4 step
		cuReIm3 X0 = x0 + x2;
		cuReIm3 X2 = x0 - x2;
		cuReIm3 X4 = x0 - !x2;
		cuReIm3 X6 = x0 + !x2;

		cuReIm3 X1 = x1 + x3;
		cuReIm3 X3 = !(x3 - x1);
		cuReIm3 X5 = (x1 - !x3) * cuReIm(a, -a);
		cuReIm3 X7 = (x1 + !x3) * cuReIm(-a, -a);

		//Radix-2 step
		cuReIm3 temp = X0 - X1;
		X0 = X0 + X1;
		X1 = temp;

		temp = X2 - X3;
		X2 = X2 + X3;
		X3 = temp;

		temp = X4 - X5;
		X4 = X4 + X5;
		X5 = temp;

		temp = X6 - X7;
		X6 = X6 + X7;
		X7 = temp;

		//data set in shuffled order:
		//X0, X4, X2, X6, X1, X5, X3, X7

		cuReIm3 F0, F1, F2, F3, F4, F5, F6, F7;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F4.x = (Kdiag[ker_baseidx + kerplanecount].x * X4.x) + (Kodiag[ker_baseidx + kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + kerplanecount].y * X4.z);
			F4.y = (Kodiag[ker_baseidx + kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + kerplanecount].y * X4.y) + (Kodiag[ker_baseidx + kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + kerplanecount].y * X4.x) + (Kodiag[ker_baseidx + kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + kerplanecount].z * X4.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			F6.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.z);
			F6.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X6.z);

			F1.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.z);
			F1.y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X1.z);

			F5.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X5.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X5.y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X5.z);
			F5.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X5.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X5.y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X5.z);
			F5.z = (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X5.x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X5.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X5.z);

			F3.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X3.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X3.y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X3.z);
			F3.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X3.y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X3.z);
			F3.z = (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X3.x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X3.z);

			F7.x = (Kdiag[ker_baseidx + kerplanecount].x * X7.x) + (Kodiag[ker_baseidx + kerplanecount].x * X7.y) + (-Kodiag[ker_baseidx + kerplanecount].y * X7.z);
			F7.y = (Kodiag[ker_baseidx + kerplanecount].x * X7.x) + (Kdiag[ker_baseidx + kerplanecount].y * X7.y) + (-Kodiag[ker_baseidx + kerplanecount].z * X7.z);
			F7.z = (-Kodiag[ker_baseidx + kerplanecount].y * X7.x) + (-Kodiag[ker_baseidx + kerplanecount].z * X7.y) + (Kdiag[ker_baseidx + kerplanecount].z * X7.z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (-Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (-Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (-Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (-Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F4.x = (Kdiag[ker_baseidx + kerplanecount].x * X4.x) + (-Kodiag[ker_baseidx + kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + kerplanecount].y * X4.z);
			F4.y = (-Kodiag[ker_baseidx + kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + kerplanecount].y * X4.y) + (-Kodiag[ker_baseidx + kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + kerplanecount].y * X4.x) + (-Kodiag[ker_baseidx + kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + kerplanecount].z * X4.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			F6.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.z);
			F6.y = (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X6.y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X6.z);

			F1.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (-Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.z);
			F1.y = (-Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X1.y) + (-Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.x) + (-Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X1.z);

			F5.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X5.x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X5.y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X5.z);
			F5.y = (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X5.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X5.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X5.z);
			F5.z = (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X5.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X5.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X5.z);

			F3.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X3.x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X3.y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X3.z);
			F3.y = (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X3.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X3.z);
			F3.z = (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X3.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X3.z);

			F7.x = (Kdiag[ker_baseidx + kerplanecount].x * X7.x) + (-Kodiag[ker_baseidx + kerplanecount].x * X7.y) + (-Kodiag[ker_baseidx + kerplanecount].y * X7.z);
			F7.y = (-Kodiag[ker_baseidx + kerplanecount].x * X7.x) + (Kdiag[ker_baseidx + kerplanecount].y * X7.y) + (Kodiag[ker_baseidx + kerplanecount].z * X7.z);
			F7.z = (-Kodiag[ker_baseidx + kerplanecount].y * X7.x) + (Kodiag[ker_baseidx + kerplanecount].z * X7.y) + (Kdiag[ker_baseidx + kerplanecount].z * X7.z);
		}

		//store result in output ready for ifft (but will need to accumulate all other contributions before ifft so cannot compute it here)

		if (set_output) {

			cuSx_out[idx] = F0.x;
			cuSy_out[idx] = F0.y;
			cuSz_out[idx] = F0.z;

			cuSx_out[idx + planecount] = F1.x;
			cuSy_out[idx + planecount] = F1.y;
			cuSz_out[idx + planecount] = F1.z;

			cuSx_out[idx + 2 * planecount] = F2.x;
			cuSy_out[idx + 2 * planecount] = F2.y;
			cuSz_out[idx + 2 * planecount] = F2.z;

			cuSx_out[idx + 3 * planecount] = F3.x;
			cuSy_out[idx + 3 * planecount] = F3.y;
			cuSz_out[idx + 3 * planecount] = F3.z;

			cuSx_out[idx + 4 * planecount] = F4.x;
			cuSy_out[idx + 4 * planecount] = F4.y;
			cuSz_out[idx + 4 * planecount] = F4.z;

			cuSx_out[idx + 5 * planecount] = F5.x;
			cuSy_out[idx + 5 * planecount] = F5.y;
			cuSz_out[idx + 5 * planecount] = F5.z;

			cuSx_out[idx + 6 * planecount] = F6.x;
			cuSy_out[idx + 6 * planecount] = F6.y;
			cuSz_out[idx + 6 * planecount] = F6.z;

			cuSx_out[idx + 7 * planecount] = F7.x;
			cuSy_out[idx + 7 * planecount] = F7.y;
			cuSz_out[idx + 7 * planecount] = F7.z;
		}
		else {

			cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
			cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
			cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

			cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
			cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
			cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

			cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
			cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
			cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

			cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
			cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
			cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;

			cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F4.x;
			cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F4.y;
			cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F4.z;

			cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F5.x;
			cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F5.y;
			cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F5.z;

			cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F6.x;
			cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F6.y;
			cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F6.z;

			cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F7.x;
			cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F7.y;
			cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F7.z;
		}

#undef a
	}
}

//N = (N.x/2 + 1, N.y, 16)
//xy is transposed
//Real kernels : this is a self demag version which sets output (unless set_output is false), but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_16_transpose_xy(
	cuVEC<cuReal3>& Kdiag, cuVEC<cuReal3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N, bool set_output = true)
{
	//above N.z/2 and N.y/2 use kernel symmetries to recover kernel values
	//diagonal components are even about the N.z/2 and N.y/2 points
	//Kxy is even about N.z/2 and odd about N.y/2
	//Kxz is odd about N.z/2 and even about N.y/2
	//Kyz is odd about N.z/2 and odd about N.y/2

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 16, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

		//the z-axis points (the others are zero)
		cuReIm3 x0 = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 x1 = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);
		cuReIm3 x2 = cuReIm3(cuSx_in[idx + 2 * planecount], cuSy_in[idx + 2 * planecount], cuSz_in[idx + 2 * planecount]);
		cuReIm3 x3 = cuReIm3(cuSx_in[idx + 3 * planecount], cuSy_in[idx + 3 * planecount], cuSz_in[idx + 3 * planecount]);
		cuReIm3 x4 = cuReIm3(cuSx_in[idx + 4 * planecount], cuSy_in[idx + 4 * planecount], cuSz_in[idx + 4 * planecount]);
		cuReIm3 x5 = cuReIm3(cuSx_in[idx + 5 * planecount], cuSy_in[idx + 5 * planecount], cuSz_in[idx + 5 * planecount]);
		cuReIm3 x6 = cuReIm3(cuSx_in[idx + 6 * planecount], cuSy_in[idx + 6 * planecount], cuSz_in[idx + 6 * planecount]);
		cuReIm3 x7 = cuReIm3(cuSx_in[idx + 7 * planecount], cuSy_in[idx + 7 * planecount], cuSz_in[idx + 7 * planecount]);

#define a	(cuBReal)9.238795325113E-01
#define b	(cuBReal)3.826834323651E-01
#define c	(cuBReal)7.071067811865E-01

		//First stage
		cuReIm3 X0 = x0 + x4;
		cuReIm3 X4 = x0 - x4;
		cuReIm3 X8 = x0 - !x4;
		cuReIm3 X12 = x0 + !x4;

		cuReIm3 X1 = x1 + x5;
		cuReIm3 X5 = (x1 - x5) * cuReIm(c, -c);
		cuReIm3 X9 = (x1 - !x5) * cuReIm(a, -b);
		cuReIm3 X13 = (x1 + !x5) * cuReIm(b, -a);

		cuReIm3 X2 = x2 + x6;
		cuReIm3 X6 = !(x6 - x2);
		cuReIm3 X10 = (x2 - !x6) * cuReIm(c, -c);
		cuReIm3 X14 = (x2 + !x6) * cuReIm(-c, -c);

		cuReIm3 X3 = x3 + x7;
		cuReIm3 X7 = (x3 - x7) * cuReIm(-c, -c);
		cuReIm3 X11 = (x3 - !x7) * cuReIm(b, -a);
		cuReIm3 X15 = (x3 + !x7) * cuReIm(-a, b);

		//Second stage
		cuReIm3 t0 = X0 + X2;
		cuReIm3 t1 = X0 - X2;
		cuReIm3 t2 = X1 + X3;
		cuReIm3 t3 = !(X3 - X1);

		X0 = t0 + t2;
		X1 = t0 - t2;
		X2 = t1 + t3;
		X3 = t1 - t3;

		t0 = X4 + X6;
		t1 = X4 - X6;
		t2 = X5 + X7;
		t3 = !(X7 - X5);

		X4 = t0 + t2;
		X5 = t0 - t2;
		X6 = t1 + t3;
		X7 = t1 - t3;

		t0 = X8 + X10;
		t1 = X8 - X10;
		t2 = X9 + X11;
		t3 = !(X11 - X9);

		X8 = t0 + t2;
		X9 = t0 - t2;
		X10 = t1 + t3;
		X11 = t1 - t3;

		t0 = X12 + X14;
		t1 = X12 - X14;
		t2 = X13 + X15;
		t3 = !(X15 - X13);

		X12 = t0 + t2;
		X13 = t0 - t2;
		X14 = t1 + t3;
		X15 = t1 - t3;

		//output is shuffled now, i.e. it is ordered as:
		//X0, X8, X4, X12, X2, X10, X6, X14, X1, X9, X5, X13, X3, X11, X7, X15

		cuReIm3 F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F8.x = (Kdiag[ker_baseidx + kerplanecount].x * X8.x) + (Kodiag[ker_baseidx + kerplanecount].x * X8.y) + (Kodiag[ker_baseidx + kerplanecount].y * X8.z);
			F8.y = (Kodiag[ker_baseidx + kerplanecount].x * X8.x) + (Kdiag[ker_baseidx + kerplanecount].y * X8.y) + (Kodiag[ker_baseidx + kerplanecount].z * X8.z);
			F8.z = (Kodiag[ker_baseidx + kerplanecount].y * X8.x) + (Kodiag[ker_baseidx + kerplanecount].z * X8.y) + (Kdiag[ker_baseidx + kerplanecount].z * X8.z);

			F4.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.z);
			F4.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X4.z);

			F12.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.z);
			F12.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.z);
			F12.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X12.z);

			F2.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.z);
			F2.y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X2.z);

			F10.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.z);
			F10.y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.z);
			F10.z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X10.z);

			F6.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.z);
			F6.y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X6.z);

			F14.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.z);
			F14.y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.z);
			F14.z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X14.z);

			F1.x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.z);
			F1.y = (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X1.z);

			F9.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X9.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X9.y) + (-Kodiag[ker_baseidx + 7 * kerplanecount].y * X9.z);
			F9.y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X9.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X9.y) + (-Kodiag[ker_baseidx + 7 * kerplanecount].z * X9.z);
			F9.z = (-Kodiag[ker_baseidx + 7 * kerplanecount].y * X9.x) + (-Kodiag[ker_baseidx + 7 * kerplanecount].z * X9.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X9.z);

			F5.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X5.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X5.y) + (-Kodiag[ker_baseidx + 6 * kerplanecount].y * X5.z);
			F5.y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X5.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X5.y) + (-Kodiag[ker_baseidx + 6 * kerplanecount].z * X5.z);
			F5.z = (-Kodiag[ker_baseidx + 6 * kerplanecount].y * X5.x) + (-Kodiag[ker_baseidx + 6 * kerplanecount].z * X5.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X5.z);

			F13.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X13.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X13.y) + (-Kodiag[ker_baseidx + 5 * kerplanecount].y * X13.z);
			F13.y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X13.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X13.y) + (-Kodiag[ker_baseidx + 5 * kerplanecount].z * X13.z);
			F13.z = (-Kodiag[ker_baseidx + 5 * kerplanecount].y * X13.x) + (-Kodiag[ker_baseidx + 5 * kerplanecount].z * X13.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X13.z);

			F3.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X3.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X3.y) + (-Kodiag[ker_baseidx + 4 * kerplanecount].y * X3.z);
			F3.y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X3.y) + (-Kodiag[ker_baseidx + 4 * kerplanecount].z * X3.z);
			F3.z = (-Kodiag[ker_baseidx + 4 * kerplanecount].y * X3.x) + (-Kodiag[ker_baseidx + 4 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X3.z);

			F11.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X11.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X11.y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X11.z);
			F11.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X11.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X11.y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X11.z);
			F11.z = (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X11.x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X11.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X11.z);

			F7.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X7.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X7.y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X7.z);
			F7.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X7.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X7.y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X7.z);
			F7.z = (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X7.x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X7.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X7.z);

			F15.x = (Kdiag[ker_baseidx + kerplanecount].x * X15.x) + (Kodiag[ker_baseidx + kerplanecount].x * X15.y) + (-Kodiag[ker_baseidx + kerplanecount].y * X15.z);
			F15.y = (Kodiag[ker_baseidx + kerplanecount].x * X15.x) + (Kdiag[ker_baseidx + kerplanecount].y * X15.y) + (-Kodiag[ker_baseidx + kerplanecount].z * X15.z);
			F15.z = (-Kodiag[ker_baseidx + kerplanecount].y * X15.x) + (-Kodiag[ker_baseidx + kerplanecount].z * X15.y) + (Kdiag[ker_baseidx + kerplanecount].z * X15.z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (-Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (-Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (-Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (-Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F8.x = (Kdiag[ker_baseidx + kerplanecount].x * X8.x) + (-Kodiag[ker_baseidx + kerplanecount].x * X8.y) + (Kodiag[ker_baseidx + kerplanecount].y * X8.z);
			F8.y = (-Kodiag[ker_baseidx + kerplanecount].x * X8.x) + (Kdiag[ker_baseidx + kerplanecount].y * X8.y) + (-Kodiag[ker_baseidx + kerplanecount].z * X8.z);
			F8.z = (Kodiag[ker_baseidx + kerplanecount].y * X8.x) + (-Kodiag[ker_baseidx + kerplanecount].z * X8.y) + (Kdiag[ker_baseidx + kerplanecount].z * X8.z);

			F4.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.z);
			F4.y = (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X4.y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X4.z);

			F12.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.z);
			F12.y = (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X12.y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.z);
			F12.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X12.z);

			F2.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (-Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.z);
			F2.y = (-Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X2.y) + (-Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.x) + (-Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X2.z);

			F10.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (-Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.z);
			F10.y = (-Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X10.y) + (-Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.z);
			F10.z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.x) + (-Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X10.z);

			F6.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (-Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.z);
			F6.y = (-Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X6.y) + (-Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.x) + (-Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X6.z);

			F14.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (-Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.z);
			F14.y = (-Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X14.y) + (-Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.z);
			F14.z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.x) + (-Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X14.z);

			F1.x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (-Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.z);
			F1.y = (-Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X1.y) + (-Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.x) + (-Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X1.z);

			F9.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X9.x) + (-Kodiag[ker_baseidx + 7 * kerplanecount].x * X9.y) + (-Kodiag[ker_baseidx + 7 * kerplanecount].y * X9.z);
			F9.y = (-Kodiag[ker_baseidx + 7 * kerplanecount].x * X9.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X9.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X9.z);
			F9.z = (-Kodiag[ker_baseidx + 7 * kerplanecount].y * X9.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X9.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X9.z);

			F5.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X5.x) + (-Kodiag[ker_baseidx + 6 * kerplanecount].x * X5.y) + (-Kodiag[ker_baseidx + 6 * kerplanecount].y * X5.z);
			F5.y = (-Kodiag[ker_baseidx + 6 * kerplanecount].x * X5.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X5.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X5.z);
			F5.z = (-Kodiag[ker_baseidx + 6 * kerplanecount].y * X5.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X5.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X5.z);

			F13.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X13.x) + (-Kodiag[ker_baseidx + 5 * kerplanecount].x * X13.y) + (-Kodiag[ker_baseidx + 5 * kerplanecount].y * X13.z);
			F13.y = (-Kodiag[ker_baseidx + 5 * kerplanecount].x * X13.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X13.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X13.z);
			F13.z = (-Kodiag[ker_baseidx + 5 * kerplanecount].y * X13.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X13.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X13.z);

			F3.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X3.x) + (-Kodiag[ker_baseidx + 4 * kerplanecount].x * X3.y) + (-Kodiag[ker_baseidx + 4 * kerplanecount].y * X3.z);
			F3.y = (-Kodiag[ker_baseidx + 4 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X3.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X3.z);
			F3.z = (-Kodiag[ker_baseidx + 4 * kerplanecount].y * X3.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X3.z);

			F11.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X11.x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X11.y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X11.z);
			F11.y = (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X11.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X11.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X11.z);
			F11.z = (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X11.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X11.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X11.z);

			F7.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X7.x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X7.y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X7.z);
			F7.y = (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X7.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X7.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X7.z);
			F7.z = (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X7.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X7.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X7.z);

			F15.x = (Kdiag[ker_baseidx + kerplanecount].x * X15.x) + (-Kodiag[ker_baseidx + kerplanecount].x * X15.y) + (-Kodiag[ker_baseidx + kerplanecount].y * X15.z);
			F15.y = (-Kodiag[ker_baseidx + kerplanecount].x * X15.x) + (Kdiag[ker_baseidx + kerplanecount].y * X15.y) + (Kodiag[ker_baseidx + kerplanecount].z * X15.z);
			F15.z = (-Kodiag[ker_baseidx + kerplanecount].y * X15.x) + (Kodiag[ker_baseidx + kerplanecount].z * X15.y) + (Kdiag[ker_baseidx + kerplanecount].z * X15.z);
		}

		//store result in output ready for ifft (but will need to accumulate all other contributions before ifft so cannot compute it here)

		if (set_output) {

			cuSx_out[idx] = F0.x;
			cuSy_out[idx] = F0.y;
			cuSz_out[idx] = F0.z;

			cuSx_out[idx + planecount] = F1.x;
			cuSy_out[idx + planecount] = F1.y;
			cuSz_out[idx + planecount] = F1.z;

			cuSx_out[idx + 2 * planecount] = F2.x;
			cuSy_out[idx + 2 * planecount] = F2.y;
			cuSz_out[idx + 2 * planecount] = F2.z;

			cuSx_out[idx + 3 * planecount] = F3.x;
			cuSy_out[idx + 3 * planecount] = F3.y;
			cuSz_out[idx + 3 * planecount] = F3.z;

			cuSx_out[idx + 4 * planecount] = F4.x;
			cuSy_out[idx + 4 * planecount] = F4.y;
			cuSz_out[idx + 4 * planecount] = F4.z;

			cuSx_out[idx + 5 * planecount] = F5.x;
			cuSy_out[idx + 5 * planecount] = F5.y;
			cuSz_out[idx + 5 * planecount] = F5.z;

			cuSx_out[idx + 6 * planecount] = F6.x;
			cuSy_out[idx + 6 * planecount] = F6.y;
			cuSz_out[idx + 6 * planecount] = F6.z;

			cuSx_out[idx + 7 * planecount] = F7.x;
			cuSy_out[idx + 7 * planecount] = F7.y;
			cuSz_out[idx + 7 * planecount] = F7.z;

			cuSx_out[idx + 8 * planecount] = F8.x;
			cuSy_out[idx + 8 * planecount] = F8.y;
			cuSz_out[idx + 8 * planecount] = F8.z;

			cuSx_out[idx + 9 * planecount] = F9.x;
			cuSy_out[idx + 9 * planecount] = F9.y;
			cuSz_out[idx + 9 * planecount] = F9.z;

			cuSx_out[idx + 10 * planecount] = F10.x;
			cuSy_out[idx + 10 * planecount] = F10.y;
			cuSz_out[idx + 10 * planecount] = F10.z;

			cuSx_out[idx + 11 * planecount] = F11.x;
			cuSy_out[idx + 11 * planecount] = F11.y;
			cuSz_out[idx + 11 * planecount] = F11.z;

			cuSx_out[idx + 12 * planecount] = F12.x;
			cuSy_out[idx + 12 * planecount] = F12.y;
			cuSz_out[idx + 12 * planecount] = F12.z;

			cuSx_out[idx + 13 * planecount] = F13.x;
			cuSy_out[idx + 13 * planecount] = F13.y;
			cuSz_out[idx + 13 * planecount] = F13.z;

			cuSx_out[idx + 14 * planecount] = F14.x;
			cuSy_out[idx + 14 * planecount] = F14.y;
			cuSz_out[idx + 14 * planecount] = F14.z;

			cuSx_out[idx + 15 * planecount] = F15.x;
			cuSy_out[idx + 15 * planecount] = F15.y;
			cuSz_out[idx + 15 * planecount] = F15.z;
		}
		else {

			cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
			cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
			cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

			cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
			cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
			cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

			cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
			cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
			cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

			cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
			cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
			cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;

			cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F4.x;
			cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F4.y;
			cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F4.z;

			cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F5.x;
			cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F5.y;
			cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F5.z;

			cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F6.x;
			cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F6.y;
			cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F6.z;

			cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F7.x;
			cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F7.y;
			cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F7.z;

			cuSx_out[idx + 8 * planecount] = (cuReIm)cuSx_out[idx + 8 * planecount] + F8.x;
			cuSy_out[idx + 8 * planecount] = (cuReIm)cuSy_out[idx + 8 * planecount] + F8.y;
			cuSz_out[idx + 8 * planecount] = (cuReIm)cuSz_out[idx + 8 * planecount] + F8.z;

			cuSx_out[idx + 9 * planecount] = (cuReIm)cuSx_out[idx + 9 * planecount] + F9.x;
			cuSy_out[idx + 9 * planecount] = (cuReIm)cuSy_out[idx + 9 * planecount] + F9.y;
			cuSz_out[idx + 9 * planecount] = (cuReIm)cuSz_out[idx + 9 * planecount] + F9.z;

			cuSx_out[idx + 10 * planecount] = (cuReIm)cuSx_out[idx + 10 * planecount] + F10.x;
			cuSy_out[idx + 10 * planecount] = (cuReIm)cuSy_out[idx + 10 * planecount] + F10.y;
			cuSz_out[idx + 10 * planecount] = (cuReIm)cuSz_out[idx + 10 * planecount] + F10.z;

			cuSx_out[idx + 11 * planecount] = (cuReIm)cuSx_out[idx + 11 * planecount] + F11.x;
			cuSy_out[idx + 11 * planecount] = (cuReIm)cuSy_out[idx + 11 * planecount] + F11.y;
			cuSz_out[idx + 11 * planecount] = (cuReIm)cuSz_out[idx + 11 * planecount] + F11.z;

			cuSx_out[idx + 12 * planecount] = (cuReIm)cuSx_out[idx + 12 * planecount] + F12.x;
			cuSy_out[idx + 12 * planecount] = (cuReIm)cuSy_out[idx + 12 * planecount] + F12.y;
			cuSz_out[idx + 12 * planecount] = (cuReIm)cuSz_out[idx + 12 * planecount] + F12.z;

			cuSx_out[idx + 13 * planecount] = (cuReIm)cuSx_out[idx + 13 * planecount] + F13.x;
			cuSy_out[idx + 13 * planecount] = (cuReIm)cuSy_out[idx + 13 * planecount] + F13.y;
			cuSz_out[idx + 13 * planecount] = (cuReIm)cuSz_out[idx + 13 * planecount] + F13.z;

			cuSx_out[idx + 14 * planecount] = (cuReIm)cuSx_out[idx + 14 * planecount] + F14.x;
			cuSy_out[idx + 14 * planecount] = (cuReIm)cuSy_out[idx + 14 * planecount] + F14.y;
			cuSz_out[idx + 14 * planecount] = (cuReIm)cuSz_out[idx + 14 * planecount] + F14.z;

			cuSx_out[idx + 15 * planecount] = (cuReIm)cuSx_out[idx + 15 * planecount] + F15.x;
			cuSy_out[idx + 15 * planecount] = (cuReIm)cuSy_out[idx + 15 * planecount] + F15.y;
			cuSz_out[idx + 15 * planecount] = (cuReIm)cuSz_out[idx + 15 * planecount] + F15.z;
		}

#undef a
#undef b
#undef c
	}
}

//N = (N.x/2 + 1, N.y, 32)
//xy is transposed
//Real kernels : this is a self demag version which sets output (unless set_output is false), but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_32_transpose_xy(
	cuVEC<cuReal3>& Kdiag, cuVEC<cuReal3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N, bool set_output = true)
{
	//above N.z/2 and N.y/2 use kernel symmetries to recover kernel values
	//diagonal components are even about the N.z/2 and N.y/2 points
	//Kxy is even about N.z/2 and odd about N.y/2
	//Kxz is odd about N.z/2 and even about N.y/2
	//Kyz is odd about N.z/2 and odd about N.y/2

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 32, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

		//input data
#define x(n)	(cuReIm3(cuSx_in[idx + (n) * planecount], cuSy_in[idx + (n) * planecount], cuSz_in[idx + (n) * planecount]))

		//no performance gain to be had from setting these as X0, X1, ... etc.
		//unrolling loops does make a slight difference though - probably last case for which you want to unroll loops
		cuReIm3 X[32];

		cuReIm3 t0, t1, t2, t3;

		//input stage

#define a	(cuBReal)0.980785280403230
#define b	(cuBReal)0.195090322016128
#define c	(cuBReal)0.923879532511287
#define d	(cuBReal)0.382683432365090
#define e	(cuBReal)0.831469612302545
#define f	(cuBReal)0.555570233019602
#define g	(cuBReal)0.707106781186548

		//j = 0
		X[0] = (x(0) + x(8));
		X[8] = (x(0) - x(8));
		X[16] = (x(0) - !x(8));
		X[24] = (x(0) + !x(8));

		//j = 1
		X[1] = (x(1) + x(9));
		X[9] = (x(1) - x(9)) * cuReIm(c, -d);
		X[17] = (x(1) - !x(9)) * cuReIm(a, -b);
		X[25] = (x(1) + !x(9)) * cuReIm(e, -f);

		//j = 2
		X[2] = (x(2) + x(10));
		X[10] = (x(2) - x(10)) * cuReIm(g, -g);
		X[18] = (x(2) - !x(10)) * cuReIm(c, -d);
		X[26] = (x(2) + !x(10)) * cuReIm(d, -c);

		//j = 3
		X[3] = (x(3) + x(11));
		X[11] = (x(3) - x(11)) * cuReIm(d, -c);
		X[19] = (x(3) - !x(11)) * cuReIm(e, -f);
		X[27] = (x(3) + !x(11)) * cuReIm(-b, -a);

		//j = 4
		X[4] = (x(4) + x(12));
		X[12] = !(x(12) - x(4));
		X[20] = (x(4) - !x(12)) * cuReIm(g, -g);
		X[28] = (x(4) + !x(12)) * cuReIm(-g, -g);

		//j = 5
		X[5] = (x(5) + x(13));
		X[13] = (x(5) - x(13)) * cuReIm(-d, -c);
		X[21] = (x(5) - !x(13)) * cuReIm(f, -e);
		X[29] = (x(5) + !x(13)) * cuReIm(-a, -b);

		//j = 6
		X[6] = (x(6) + x(14));
		X[14] = (x(6) - x(14)) * cuReIm(-g, -g);
		X[22] = (x(6) - !x(14)) * cuReIm(d, -c);
		X[30] = (x(6) + !x(14)) * cuReIm(-c, d);

		//j = 7
		X[7] = (x(7) + x(15));
		X[15] = (x(7) - x(15)) * cuReIm(-c, -d);
		X[23] = (x(7) - !x(15)) * cuReIm(b, -a);
		X[31] = (x(7) + !x(15)) * cuReIm(-f, e);

#undef x

		//final radix4 stage

		//j = 0
		t0 = (X[0] + X[4]);
		t1 = (X[0] - X[4]);
		t2 = (X[2] + X[6]);
		t3 = !(X[6] - X[2]);

		X[0] = (t0 + t2);
		X[2] = (t0 - t2);
		X[4] = (t1 + t3);
		X[6] = (t1 - t3);

		t0 = (X[8] + X[12]);
		t1 = (X[8] - X[12]);
		t2 = (X[10] + X[14]);
		t3 = !(X[14] - X[10]);

		X[8] = (t0 + t2);
		X[10] = (t0 - t2);
		X[12] = (t1 + t3);
		X[14] = (t1 - t3);

		t0 = (X[16] + X[20]);
		t1 = (X[16] - X[20]);
		t2 = (X[18] + X[22]);
		t3 = !(X[22] - X[18]);

		X[16] = (t0 + t2);
		X[18] = (t0 - t2);
		X[20] = (t1 + t3);
		X[22] = (t1 - t3);

		t0 = (X[24] + X[28]);
		t1 = (X[24] - X[28]);
		t2 = (X[26] + X[30]);
		t3 = !(X[30] - X[26]);

		X[24] = (t0 + t2);
		X[26] = (t0 - t2);
		X[28] = (t1 + t3);
		X[30] = (t1 - t3);

		//j = 1
		t0 = (X[1] + X[5]);
		t1 = (X[1] - X[5]);
		t2 = (X[3] + X[7]);
		t3 = !(X[7] - X[3]);

		X[1] = (t0 + t2);
		X[3] = !(t2 - t0);
		X[5] = (t1 + t3) * cuReIm(g, -g);
		X[7] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[9] + X[13]);
		t1 = (X[9] - X[13]);
		t2 = (X[11] + X[15]);
		t3 = !(X[15] - X[11]);

		X[9] = (t0 + t2);
		X[11] = !(t2 - t0);
		X[13] = (t1 + t3) * cuReIm(g, -g);
		X[15] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[17] + X[21]);
		t1 = (X[17] - X[21]);
		t2 = (X[19] + X[23]);
		t3 = !(X[23] - X[19]);

		X[17] = (t0 + t2);
		X[19] = !(t2 - t0);
		X[21] = (t1 + t3) * cuReIm(g, -g);
		X[23] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[25] + X[29]);
		t1 = (X[25] - X[29]);
		t2 = (X[27] + X[31]);
		t3 = !(X[31] - X[27]);

		X[25] = (t0 + t2);
		X[27] = !(t2 - t0);
		X[29] = (t1 + t3) * cuReIm(g, -g);
		X[31] = (t1 - t3) * cuReIm(-g, -g);

		//radix-2 step to finish
		t0 = X[0] - X[1];
		X[0] = X[0] + X[1];
		X[1] = t0;

		t0 = X[2] - X[3];
		X[2] = X[2] + X[3];
		X[3] = t0;

		t0 = X[4] - X[5];
		X[4] = X[4] + X[5];
		X[5] = t0;

		t0 = X[6] - X[7];
		X[6] = X[6] + X[7];
		X[7] = t0;

		t0 = X[8] - X[9];
		X[8] = X[8] + X[9];
		X[9] = t0;

		t0 = X[10] - X[11];
		X[10] = X[10] + X[11];
		X[11] = t0;

		t0 = X[12] - X[13];
		X[12] = X[12] + X[13];
		X[13] = t0;

		t0 = X[14] - X[15];
		X[14] = X[14] + X[15];
		X[15] = t0;

		t0 = X[16] - X[17];
		X[16] = X[16] + X[17];
		X[17] = t0;

		t0 = X[18] - X[19];
		X[18] = X[18] + X[19];
		X[19] = t0;

		t0 = X[20] - X[21];
		X[20] = X[20] + X[21];
		X[21] = t0;

		t0 = X[22] - X[23];
		X[22] = X[22] + X[23];
		X[23] = t0;

		t0 = X[24] - X[25];
		X[24] = X[24] + X[25];
		X[25] = t0;

		t0 = X[26] - X[27];
		X[26] = X[26] + X[27];
		X[27] = t0;

		t0 = X[28] - X[29];
		X[28] = X[28] + X[29];
		X[29] = t0;

		t0 = X[30] - X[31];
		X[30] = X[30] + X[31];
		X[31] = t0;

		//output is shuffled now, i.e. it is ordered as:
		//0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		cuReIm3 F[32];

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			F[0].x = (Kdiag[ker_baseidx].x * X[0].x) + (Kodiag[ker_baseidx].x * X[0].y) + (Kodiag[ker_baseidx].y * X[0].z);
			F[0].y = (Kodiag[ker_baseidx].x * X[0].x) + (Kdiag[ker_baseidx].y * X[0].y) + (Kodiag[ker_baseidx].z * X[0].z);
			F[0].z = (Kodiag[ker_baseidx].y * X[0].x) + (Kodiag[ker_baseidx].z * X[0].y) + (Kdiag[ker_baseidx].z * X[0].z);

			F[16].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].z);
			F[16].y = (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);
			F[16].z = (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);

			F[8].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].z);
			F[8].y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);
			F[8].z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);

			F[24].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].z);
			F[24].y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);
			F[24].z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);

			F[4].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].z);
			F[4].y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);
			F[4].z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);

			F[20].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].z);
			F[20].y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);
			F[20].z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);

			F[12].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].z);
			F[12].y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);
			F[12].z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);

			F[28].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].z);
			F[28].y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);
			F[28].z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);

			F[2].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].z);
			F[2].y = (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);
			F[2].z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);

			F[18].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].z);
			F[18].y = (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);
			F[18].z = (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);

			F[10].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].z);
			F[10].y = (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);
			F[10].z = (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);

			F[26].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].z);
			F[26].y = (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);
			F[26].z = (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);

			F[6].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].z);
			F[6].y = (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);
			F[6].z = (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);

			F[22].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].z);
			F[22].y = (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);
			F[22].z = (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);

			F[14].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].z);
			F[14].y = (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);
			F[14].z = (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);

			F[30].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].z);
			F[30].y = (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);
			F[30].z = (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);

			F[1].x = (Kdiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].z);
			F[1].y = (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kdiag[ker_baseidx + 16 * kerplanecount].y * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);
			F[1].z = (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].x) + (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].y) + (Kdiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);

			F[17].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[17].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[17].y) + (-Kodiag[ker_baseidx + 15 * kerplanecount].y * X[17].z);
			F[17].y = (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[17].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[17].y) + (-Kodiag[ker_baseidx + 15 * kerplanecount].z * X[17].z);
			F[17].z = (-Kodiag[ker_baseidx + 15 * kerplanecount].y * X[17].x) + (-Kodiag[ker_baseidx + 15 * kerplanecount].z * X[17].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[17].z);

			F[9].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[9].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[9].y) + (-Kodiag[ker_baseidx + 14 * kerplanecount].y * X[9].z);
			F[9].y = (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[9].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[9].y) + (-Kodiag[ker_baseidx + 14 * kerplanecount].z * X[9].z);
			F[9].z = (-Kodiag[ker_baseidx + 14 * kerplanecount].y * X[9].x) + (-Kodiag[ker_baseidx + 14 * kerplanecount].z * X[9].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[9].z);

			F[25].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[25].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[25].y) + (-Kodiag[ker_baseidx + 13 * kerplanecount].y * X[25].z);
			F[25].y = (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[25].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[25].y) + (-Kodiag[ker_baseidx + 13 * kerplanecount].z * X[25].z);
			F[25].z = (-Kodiag[ker_baseidx + 13 * kerplanecount].y * X[25].x) + (-Kodiag[ker_baseidx + 13 * kerplanecount].z * X[25].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[25].z);

			F[5].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[5].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[5].y) + (-Kodiag[ker_baseidx + 12 * kerplanecount].y * X[5].z);
			F[5].y = (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[5].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[5].y) + (-Kodiag[ker_baseidx + 12 * kerplanecount].z * X[5].z);
			F[5].z = (-Kodiag[ker_baseidx + 12 * kerplanecount].y * X[5].x) + (-Kodiag[ker_baseidx + 12 * kerplanecount].z * X[5].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[5].z);

			F[21].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[21].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[21].y) + (-Kodiag[ker_baseidx + 11 * kerplanecount].y * X[21].z);
			F[21].y = (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[21].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[21].y) + (-Kodiag[ker_baseidx + 11 * kerplanecount].z * X[21].z);
			F[21].z = (-Kodiag[ker_baseidx + 11 * kerplanecount].y * X[21].x) + (-Kodiag[ker_baseidx + 11 * kerplanecount].z * X[21].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[21].z);

			F[13].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[13].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[13].y) + (-Kodiag[ker_baseidx + 10 * kerplanecount].y * X[13].z);
			F[13].y = (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[13].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[13].y) + (-Kodiag[ker_baseidx + 10 * kerplanecount].z * X[13].z);
			F[13].z = (-Kodiag[ker_baseidx + 10 * kerplanecount].y * X[13].x) + (-Kodiag[ker_baseidx + 10 * kerplanecount].z * X[13].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[13].z);

			F[29].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[29].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[29].y) + (-Kodiag[ker_baseidx + 9 * kerplanecount].y * X[29].z);
			F[29].y = (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[29].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[29].y) + (-Kodiag[ker_baseidx + 9 * kerplanecount].z * X[29].z);
			F[29].z = (-Kodiag[ker_baseidx + 9 * kerplanecount].y * X[29].x) + (-Kodiag[ker_baseidx + 9 * kerplanecount].z * X[29].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[29].z);

			F[3].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[3].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[3].y) + (-Kodiag[ker_baseidx + 8 * kerplanecount].y * X[3].z);
			F[3].y = (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[3].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[3].y) + (-Kodiag[ker_baseidx + 8 * kerplanecount].z * X[3].z);
			F[3].z = (-Kodiag[ker_baseidx + 8 * kerplanecount].y * X[3].x) + (-Kodiag[ker_baseidx + 8 * kerplanecount].z * X[3].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[3].z);

			F[19].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[19].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[19].y) + (-Kodiag[ker_baseidx + 7 * kerplanecount].y * X[19].z);
			F[19].y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[19].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[19].y) + (-Kodiag[ker_baseidx + 7 * kerplanecount].z * X[19].z);
			F[19].z = (-Kodiag[ker_baseidx + 7 * kerplanecount].y * X[19].x) + (-Kodiag[ker_baseidx + 7 * kerplanecount].z * X[19].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[19].z);

			F[11].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[11].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[11].y) + (-Kodiag[ker_baseidx + 6 * kerplanecount].y * X[11].z);
			F[11].y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[11].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[11].y) + (-Kodiag[ker_baseidx + 6 * kerplanecount].z * X[11].z);
			F[11].z = (-Kodiag[ker_baseidx + 6 * kerplanecount].y * X[11].x) + (-Kodiag[ker_baseidx + 6 * kerplanecount].z * X[11].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[11].z);

			F[27].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[27].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[27].y) + (-Kodiag[ker_baseidx + 5 * kerplanecount].y * X[27].z);
			F[27].y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[27].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[27].y) + (-Kodiag[ker_baseidx + 5 * kerplanecount].z * X[27].z);
			F[27].z = (-Kodiag[ker_baseidx + 5 * kerplanecount].y * X[27].x) + (-Kodiag[ker_baseidx + 5 * kerplanecount].z * X[27].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[27].z);

			F[7].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[7].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[7].y) + (-Kodiag[ker_baseidx + 4 * kerplanecount].y * X[7].z);
			F[7].y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[7].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[7].y) + (-Kodiag[ker_baseidx + 4 * kerplanecount].z * X[7].z);
			F[7].z = (-Kodiag[ker_baseidx + 4 * kerplanecount].y * X[7].x) + (-Kodiag[ker_baseidx + 4 * kerplanecount].z * X[7].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[7].z);

			F[23].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[23].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[23].y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X[23].z);
			F[23].y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[23].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[23].y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X[23].z);
			F[23].z = (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X[23].x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X[23].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[23].z);

			F[15].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[15].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[15].y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X[15].z);
			F[15].y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[15].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[15].y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X[15].z);
			F[15].z = (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X[15].x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X[15].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[15].z);

			F[31].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[31].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[31].y) + (-Kodiag[ker_baseidx + 1 * kerplanecount].y * X[31].z);
			F[31].y = (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[31].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[31].y) + (-Kodiag[ker_baseidx + 1 * kerplanecount].z * X[31].z);
			F[31].z = (-Kodiag[ker_baseidx + 1 * kerplanecount].y * X[31].x) + (-Kodiag[ker_baseidx + 1 * kerplanecount].z * X[31].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[31].z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			F[0].x = (Kdiag[ker_baseidx].x * X[0].x) + (-Kodiag[ker_baseidx].x * X[0].y) + (Kodiag[ker_baseidx].y * X[0].z);
			F[0].y = (-Kodiag[ker_baseidx].x * X[0].x) + (Kdiag[ker_baseidx].y * X[0].y) + (-Kodiag[ker_baseidx].z * X[0].z);
			F[0].z = (Kodiag[ker_baseidx].y * X[0].x) + (-Kodiag[ker_baseidx].z * X[0].y) + (Kdiag[ker_baseidx].z * X[0].z);

			F[16].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (-Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].z);
			F[16].y = (-Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[16].y) + (-Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);
			F[16].z = (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].x) + (-Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);

			F[8].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].z);
			F[8].y = (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[8].y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);
			F[8].z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);

			F[24].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].z);
			F[24].y = (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[24].y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);
			F[24].z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);

			F[4].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (-Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].z);
			F[4].y = (-Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[4].y) + (-Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);
			F[4].z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].x) + (-Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);

			F[20].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (-Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].z);
			F[20].y = (-Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[20].y) + (-Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);
			F[20].z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].x) + (-Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);

			F[12].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (-Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].z);
			F[12].y = (-Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[12].y) + (-Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);
			F[12].z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].x) + (-Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);

			F[28].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (-Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].z);
			F[28].y = (-Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[28].y) + (-Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);
			F[28].z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].x) + (-Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);

			F[2].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (-Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].z);
			F[2].y = (-Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[2].y) + (-Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);
			F[2].z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].x) + (-Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);

			F[18].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (-Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].z);
			F[18].y = (-Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[18].y) + (-Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);
			F[18].z = (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].x) + (-Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);

			F[10].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (-Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].z);
			F[10].y = (-Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[10].y) + (-Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);
			F[10].z = (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].x) + (-Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);

			F[26].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (-Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].z);
			F[26].y = (-Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[26].y) + (-Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);
			F[26].z = (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].x) + (-Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);

			F[6].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (-Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].z);
			F[6].y = (-Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[6].y) + (-Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);
			F[6].z = (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].x) + (-Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);

			F[22].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (-Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].z);
			F[22].y = (-Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[22].y) + (-Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);
			F[22].z = (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].x) + (-Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);

			F[14].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (-Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].z);
			F[14].y = (-Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[14].y) + (-Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);
			F[14].z = (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].x) + (-Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);

			F[30].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (-Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].z);
			F[30].y = (-Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[30].y) + (-Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);
			F[30].z = (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].x) + (-Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);

			F[1].x = (Kdiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (-Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].z);
			F[1].y = (-Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kdiag[ker_baseidx + 16 * kerplanecount].y * X[1].y) + (-Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);
			F[1].z = (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].x) + (-Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].y) + (Kdiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);

			F[17].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[17].x) + (-Kodiag[ker_baseidx + 15 * kerplanecount].x * X[17].y) + (-Kodiag[ker_baseidx + 15 * kerplanecount].y * X[17].z);
			F[17].y = (-Kodiag[ker_baseidx + 15 * kerplanecount].x * X[17].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[17].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[17].z);
			F[17].z = (-Kodiag[ker_baseidx + 15 * kerplanecount].y * X[17].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[17].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[17].z);

			F[9].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[9].x) + (-Kodiag[ker_baseidx + 14 * kerplanecount].x * X[9].y) + (-Kodiag[ker_baseidx + 14 * kerplanecount].y * X[9].z);
			F[9].y = (-Kodiag[ker_baseidx + 14 * kerplanecount].x * X[9].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[9].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[9].z);
			F[9].z = (-Kodiag[ker_baseidx + 14 * kerplanecount].y * X[9].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[9].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[9].z);

			F[25].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[25].x) + (-Kodiag[ker_baseidx + 13 * kerplanecount].x * X[25].y) + (-Kodiag[ker_baseidx + 13 * kerplanecount].y * X[25].z);
			F[25].y = (-Kodiag[ker_baseidx + 13 * kerplanecount].x * X[25].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[25].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[25].z);
			F[25].z = (-Kodiag[ker_baseidx + 13 * kerplanecount].y * X[25].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[25].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[25].z);

			F[5].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[5].x) + (-Kodiag[ker_baseidx + 12 * kerplanecount].x * X[5].y) + (-Kodiag[ker_baseidx + 12 * kerplanecount].y * X[5].z);
			F[5].y = (-Kodiag[ker_baseidx + 12 * kerplanecount].x * X[5].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[5].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[5].z);
			F[5].z = (-Kodiag[ker_baseidx + 12 * kerplanecount].y * X[5].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[5].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[5].z);

			F[21].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[21].x) + (-Kodiag[ker_baseidx + 11 * kerplanecount].x * X[21].y) + (-Kodiag[ker_baseidx + 11 * kerplanecount].y * X[21].z);
			F[21].y = (-Kodiag[ker_baseidx + 11 * kerplanecount].x * X[21].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[21].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[21].z);
			F[21].z = (-Kodiag[ker_baseidx + 11 * kerplanecount].y * X[21].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[21].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[21].z);

			F[13].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[13].x) + (-Kodiag[ker_baseidx + 10 * kerplanecount].x * X[13].y) + (-Kodiag[ker_baseidx + 10 * kerplanecount].y * X[13].z);
			F[13].y = (-Kodiag[ker_baseidx + 10 * kerplanecount].x * X[13].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[13].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[13].z);
			F[13].z = (-Kodiag[ker_baseidx + 10 * kerplanecount].y * X[13].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[13].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[13].z);

			F[29].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[29].x) + (-Kodiag[ker_baseidx + 9 * kerplanecount].x * X[29].y) + (-Kodiag[ker_baseidx + 9 * kerplanecount].y * X[29].z);
			F[29].y = (-Kodiag[ker_baseidx + 9 * kerplanecount].x * X[29].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[29].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[29].z);
			F[29].z = (-Kodiag[ker_baseidx + 9 * kerplanecount].y * X[29].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[29].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[29].z);

			F[3].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[3].x) + (-Kodiag[ker_baseidx + 8 * kerplanecount].x * X[3].y) + (-Kodiag[ker_baseidx + 8 * kerplanecount].y * X[3].z);
			F[3].y = (-Kodiag[ker_baseidx + 8 * kerplanecount].x * X[3].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[3].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[3].z);
			F[3].z = (-Kodiag[ker_baseidx + 8 * kerplanecount].y * X[3].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[3].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[3].z);

			F[19].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[19].x) + (-Kodiag[ker_baseidx + 7 * kerplanecount].x * X[19].y) + (-Kodiag[ker_baseidx + 7 * kerplanecount].y * X[19].z);
			F[19].y = (-Kodiag[ker_baseidx + 7 * kerplanecount].x * X[19].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[19].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[19].z);
			F[19].z = (-Kodiag[ker_baseidx + 7 * kerplanecount].y * X[19].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[19].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[19].z);

			F[11].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[11].x) + (-Kodiag[ker_baseidx + 6 * kerplanecount].x * X[11].y) + (-Kodiag[ker_baseidx + 6 * kerplanecount].y * X[11].z);
			F[11].y = (-Kodiag[ker_baseidx + 6 * kerplanecount].x * X[11].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[11].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[11].z);
			F[11].z = (-Kodiag[ker_baseidx + 6 * kerplanecount].y * X[11].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[11].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[11].z);

			F[27].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[27].x) + (-Kodiag[ker_baseidx + 5 * kerplanecount].x * X[27].y) + (-Kodiag[ker_baseidx + 5 * kerplanecount].y * X[27].z);
			F[27].y = (-Kodiag[ker_baseidx + 5 * kerplanecount].x * X[27].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[27].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[27].z);
			F[27].z = (-Kodiag[ker_baseidx + 5 * kerplanecount].y * X[27].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[27].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[27].z);

			F[7].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[7].x) + (-Kodiag[ker_baseidx + 4 * kerplanecount].x * X[7].y) + (-Kodiag[ker_baseidx + 4 * kerplanecount].y * X[7].z);
			F[7].y = (-Kodiag[ker_baseidx + 4 * kerplanecount].x * X[7].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[7].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[7].z);
			F[7].z = (-Kodiag[ker_baseidx + 4 * kerplanecount].y * X[7].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[7].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[7].z);

			F[23].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[23].x) + (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X[23].y) + (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X[23].z);
			F[23].y = (-Kodiag[ker_baseidx + 3 * kerplanecount].x * X[23].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[23].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[23].z);
			F[23].z = (-Kodiag[ker_baseidx + 3 * kerplanecount].y * X[23].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[23].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[23].z);

			F[15].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[15].x) + (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X[15].y) + (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X[15].z);
			F[15].y = (-Kodiag[ker_baseidx + 2 * kerplanecount].x * X[15].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[15].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[15].z);
			F[15].z = (-Kodiag[ker_baseidx + 2 * kerplanecount].y * X[15].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[15].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[15].z);

			F[31].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[31].x) + (-Kodiag[ker_baseidx + 1 * kerplanecount].x * X[31].y) + (-Kodiag[ker_baseidx + 1 * kerplanecount].y * X[31].z);
			F[31].y = (-Kodiag[ker_baseidx + 1 * kerplanecount].x * X[31].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[31].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[31].z);
			F[31].z = (-Kodiag[ker_baseidx + 1 * kerplanecount].y * X[31].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[31].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[31].z);
		}

		//store result in output ready for ifft (but will need to accumulate all other contributions before ifft so cannot compute it here)

		if (set_output) {

			cuSx_out[idx] = F[0].x;
			cuSy_out[idx] = F[0].y;
			cuSz_out[idx] = F[0].z;

			cuSx_out[idx + planecount] = F[1].x;
			cuSy_out[idx + planecount] = F[1].y;
			cuSz_out[idx + planecount] = F[1].z;

			cuSx_out[idx + 2 * planecount] = F[2].x;
			cuSy_out[idx + 2 * planecount] = F[2].y;
			cuSz_out[idx + 2 * planecount] = F[2].z;

			cuSx_out[idx + 3 * planecount] = F[3].x;
			cuSy_out[idx + 3 * planecount] = F[3].y;
			cuSz_out[idx + 3 * planecount] = F[3].z;

			cuSx_out[idx + 4 * planecount] = F[4].x;
			cuSy_out[idx + 4 * planecount] = F[4].y;
			cuSz_out[idx + 4 * planecount] = F[4].z;

			cuSx_out[idx + 5 * planecount] = F[5].x;
			cuSy_out[idx + 5 * planecount] = F[5].y;
			cuSz_out[idx + 5 * planecount] = F[5].z;

			cuSx_out[idx + 6 * planecount] = F[6].x;
			cuSy_out[idx + 6 * planecount] = F[6].y;
			cuSz_out[idx + 6 * planecount] = F[6].z;

			cuSx_out[idx + 7 * planecount] = F[7].x;
			cuSy_out[idx + 7 * planecount] = F[7].y;
			cuSz_out[idx + 7 * planecount] = F[7].z;

			cuSx_out[idx + 8 * planecount] = F[8].x;
			cuSy_out[idx + 8 * planecount] = F[8].y;
			cuSz_out[idx + 8 * planecount] = F[8].z;

			cuSx_out[idx + 9 * planecount] = F[9].x;
			cuSy_out[idx + 9 * planecount] = F[9].y;
			cuSz_out[idx + 9 * planecount] = F[9].z;

			cuSx_out[idx + 10 * planecount] = F[10].x;
			cuSy_out[idx + 10 * planecount] = F[10].y;
			cuSz_out[idx + 10 * planecount] = F[10].z;

			cuSx_out[idx + 11 * planecount] = F[11].x;
			cuSy_out[idx + 11 * planecount] = F[11].y;
			cuSz_out[idx + 11 * planecount] = F[11].z;

			cuSx_out[idx + 12 * planecount] = F[12].x;
			cuSy_out[idx + 12 * planecount] = F[12].y;
			cuSz_out[idx + 12 * planecount] = F[12].z;

			cuSx_out[idx + 13 * planecount] = F[13].x;
			cuSy_out[idx + 13 * planecount] = F[13].y;
			cuSz_out[idx + 13 * planecount] = F[13].z;

			cuSx_out[idx + 14 * planecount] = F[14].x;
			cuSy_out[idx + 14 * planecount] = F[14].y;
			cuSz_out[idx + 14 * planecount] = F[14].z;

			cuSx_out[idx + 15 * planecount] = F[15].x;
			cuSy_out[idx + 15 * planecount] = F[15].y;
			cuSz_out[idx + 15 * planecount] = F[15].z;

			cuSx_out[idx + 16 * planecount] = F[16].x;
			cuSy_out[idx + 16 * planecount] = F[16].y;
			cuSz_out[idx + 16 * planecount] = F[16].z;

			cuSx_out[idx + 17 * planecount] = F[17].x;
			cuSy_out[idx + 17 * planecount] = F[17].y;
			cuSz_out[idx + 17 * planecount] = F[17].z;

			cuSx_out[idx + 18 * planecount] = F[18].x;
			cuSy_out[idx + 18 * planecount] = F[18].y;
			cuSz_out[idx + 18 * planecount] = F[18].z;

			cuSx_out[idx + 19 * planecount] = F[19].x;
			cuSy_out[idx + 19 * planecount] = F[19].y;
			cuSz_out[idx + 19 * planecount] = F[19].z;

			cuSx_out[idx + 20 * planecount] = F[20].x;
			cuSy_out[idx + 20 * planecount] = F[20].y;
			cuSz_out[idx + 20 * planecount] = F[20].z;

			cuSx_out[idx + 21 * planecount] = F[21].x;
			cuSy_out[idx + 21 * planecount] = F[21].y;
			cuSz_out[idx + 21 * planecount] = F[21].z;

			cuSx_out[idx + 22 * planecount] = F[22].x;
			cuSy_out[idx + 22 * planecount] = F[22].y;
			cuSz_out[idx + 22 * planecount] = F[22].z;

			cuSx_out[idx + 23 * planecount] = F[23].x;
			cuSy_out[idx + 23 * planecount] = F[23].y;
			cuSz_out[idx + 23 * planecount] = F[23].z;

			cuSx_out[idx + 24 * planecount] = F[24].x;
			cuSy_out[idx + 24 * planecount] = F[24].y;
			cuSz_out[idx + 24 * planecount] = F[24].z;

			cuSx_out[idx + 25 * planecount] = F[25].x;
			cuSy_out[idx + 25 * planecount] = F[25].y;
			cuSz_out[idx + 25 * planecount] = F[25].z;

			cuSx_out[idx + 26 * planecount] = F[26].x;
			cuSy_out[idx + 26 * planecount] = F[26].y;
			cuSz_out[idx + 26 * planecount] = F[26].z;

			cuSx_out[idx + 27 * planecount] = F[27].x;
			cuSy_out[idx + 27 * planecount] = F[27].y;
			cuSz_out[idx + 27 * planecount] = F[27].z;

			cuSx_out[idx + 28 * planecount] = F[28].x;
			cuSy_out[idx + 28 * planecount] = F[28].y;
			cuSz_out[idx + 28 * planecount] = F[28].z;

			cuSx_out[idx + 29 * planecount] = F[29].x;
			cuSy_out[idx + 29 * planecount] = F[29].y;
			cuSz_out[idx + 29 * planecount] = F[29].z;

			cuSx_out[idx + 30 * planecount] = F[30].x;
			cuSy_out[idx + 30 * planecount] = F[30].y;
			cuSz_out[idx + 30 * planecount] = F[30].z;

			cuSx_out[idx + 31 * planecount] = F[31].x;
			cuSy_out[idx + 31 * planecount] = F[31].y;
			cuSz_out[idx + 31 * planecount] = F[31].z;
		}
		else {

			cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F[0].x;
			cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F[0].y;
			cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F[0].z;

			cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F[1].x;
			cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F[1].y;
			cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F[1].z;

			cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F[2].x;
			cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F[2].y;
			cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F[2].z;

			cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F[3].x;
			cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F[3].y;
			cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F[3].z;

			cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F[4].x;
			cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F[4].y;
			cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F[4].z;

			cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F[5].x;
			cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F[5].y;
			cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F[5].z;

			cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F[6].x;
			cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F[6].y;
			cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F[6].z;

			cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F[7].x;
			cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F[7].y;
			cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F[7].z;

			cuSx_out[idx + 8 * planecount] = (cuReIm)cuSx_out[idx + 8 * planecount] + F[8].x;
			cuSy_out[idx + 8 * planecount] = (cuReIm)cuSy_out[idx + 8 * planecount] + F[8].y;
			cuSz_out[idx + 8 * planecount] = (cuReIm)cuSz_out[idx + 8 * planecount] + F[8].z;

			cuSx_out[idx + 9 * planecount] = (cuReIm)cuSx_out[idx + 9 * planecount] + F[9].x;
			cuSy_out[idx + 9 * planecount] = (cuReIm)cuSy_out[idx + 9 * planecount] + F[9].y;
			cuSz_out[idx + 9 * planecount] = (cuReIm)cuSz_out[idx + 9 * planecount] + F[9].z;

			cuSx_out[idx + 10 * planecount] = (cuReIm)cuSx_out[idx + 10 * planecount] + F[10].x;
			cuSy_out[idx + 10 * planecount] = (cuReIm)cuSy_out[idx + 10 * planecount] + F[10].y;
			cuSz_out[idx + 10 * planecount] = (cuReIm)cuSz_out[idx + 10 * planecount] + F[10].z;

			cuSx_out[idx + 11 * planecount] = (cuReIm)cuSx_out[idx + 11 * planecount] + F[11].x;
			cuSy_out[idx + 11 * planecount] = (cuReIm)cuSy_out[idx + 11 * planecount] + F[11].y;
			cuSz_out[idx + 11 * planecount] = (cuReIm)cuSz_out[idx + 11 * planecount] + F[11].z;

			cuSx_out[idx + 12 * planecount] = (cuReIm)cuSx_out[idx + 12 * planecount] + F[12].x;
			cuSy_out[idx + 12 * planecount] = (cuReIm)cuSy_out[idx + 12 * planecount] + F[12].y;
			cuSz_out[idx + 12 * planecount] = (cuReIm)cuSz_out[idx + 12 * planecount] + F[12].z;

			cuSx_out[idx + 13 * planecount] = (cuReIm)cuSx_out[idx + 13 * planecount] + F[13].x;
			cuSy_out[idx + 13 * planecount] = (cuReIm)cuSy_out[idx + 13 * planecount] + F[13].y;
			cuSz_out[idx + 13 * planecount] = (cuReIm)cuSz_out[idx + 13 * planecount] + F[13].z;

			cuSx_out[idx + 14 * planecount] = (cuReIm)cuSx_out[idx + 14 * planecount] + F[14].x;
			cuSy_out[idx + 14 * planecount] = (cuReIm)cuSy_out[idx + 14 * planecount] + F[14].y;
			cuSz_out[idx + 14 * planecount] = (cuReIm)cuSz_out[idx + 14 * planecount] + F[14].z;

			cuSx_out[idx + 15 * planecount] = (cuReIm)cuSx_out[idx + 15 * planecount] + F[15].x;
			cuSy_out[idx + 15 * planecount] = (cuReIm)cuSy_out[idx + 15 * planecount] + F[15].y;
			cuSz_out[idx + 15 * planecount] = (cuReIm)cuSz_out[idx + 15 * planecount] + F[15].z;

			cuSx_out[idx + 16 * planecount] = (cuReIm)cuSx_out[idx + 16 * planecount] + F[16].x;
			cuSy_out[idx + 16 * planecount] = (cuReIm)cuSy_out[idx + 16 * planecount] + F[16].y;
			cuSz_out[idx + 16 * planecount] = (cuReIm)cuSz_out[idx + 16 * planecount] + F[16].z;

			cuSx_out[idx + 17 * planecount] = (cuReIm)cuSx_out[idx + 17 * planecount] + F[17].x;
			cuSy_out[idx + 17 * planecount] = (cuReIm)cuSy_out[idx + 17 * planecount] + F[17].y;
			cuSz_out[idx + 17 * planecount] = (cuReIm)cuSz_out[idx + 17 * planecount] + F[17].z;

			cuSx_out[idx + 18 * planecount] = (cuReIm)cuSx_out[idx + 18 * planecount] + F[18].x;
			cuSy_out[idx + 18 * planecount] = (cuReIm)cuSy_out[idx + 18 * planecount] + F[18].y;
			cuSz_out[idx + 18 * planecount] = (cuReIm)cuSz_out[idx + 18 * planecount] + F[18].z;

			cuSx_out[idx + 19 * planecount] = (cuReIm)cuSx_out[idx + 19 * planecount] + F[19].x;
			cuSy_out[idx + 19 * planecount] = (cuReIm)cuSy_out[idx + 19 * planecount] + F[19].y;
			cuSz_out[idx + 19 * planecount] = (cuReIm)cuSz_out[idx + 19 * planecount] + F[19].z;

			cuSx_out[idx + 20 * planecount] = (cuReIm)cuSx_out[idx + 20 * planecount] + F[20].x;
			cuSy_out[idx + 20 * planecount] = (cuReIm)cuSy_out[idx + 20 * planecount] + F[20].y;
			cuSz_out[idx + 20 * planecount] = (cuReIm)cuSz_out[idx + 20 * planecount] + F[20].z;

			cuSx_out[idx + 21 * planecount] = (cuReIm)cuSx_out[idx + 21 * planecount] + F[21].x;
			cuSy_out[idx + 21 * planecount] = (cuReIm)cuSy_out[idx + 21 * planecount] + F[21].y;
			cuSz_out[idx + 21 * planecount] = (cuReIm)cuSz_out[idx + 21 * planecount] + F[21].z;

			cuSx_out[idx + 22 * planecount] = (cuReIm)cuSx_out[idx + 22 * planecount] + F[22].x;
			cuSy_out[idx + 22 * planecount] = (cuReIm)cuSy_out[idx + 22 * planecount] + F[22].y;
			cuSz_out[idx + 22 * planecount] = (cuReIm)cuSz_out[idx + 22 * planecount] + F[22].z;

			cuSx_out[idx + 23 * planecount] = (cuReIm)cuSx_out[idx + 23 * planecount] + F[23].x;
			cuSy_out[idx + 23 * planecount] = (cuReIm)cuSy_out[idx + 23 * planecount] + F[23].y;
			cuSz_out[idx + 23 * planecount] = (cuReIm)cuSz_out[idx + 23 * planecount] + F[23].z;

			cuSx_out[idx + 24 * planecount] = (cuReIm)cuSx_out[idx + 24 * planecount] + F[24].x;
			cuSy_out[idx + 24 * planecount] = (cuReIm)cuSy_out[idx + 24 * planecount] + F[24].y;
			cuSz_out[idx + 24 * planecount] = (cuReIm)cuSz_out[idx + 24 * planecount] + F[24].z;

			cuSx_out[idx + 25 * planecount] = (cuReIm)cuSx_out[idx + 25 * planecount] + F[25].x;
			cuSy_out[idx + 25 * planecount] = (cuReIm)cuSy_out[idx + 25 * planecount] + F[25].y;
			cuSz_out[idx + 25 * planecount] = (cuReIm)cuSz_out[idx + 25 * planecount] + F[25].z;

			cuSx_out[idx + 26 * planecount] = (cuReIm)cuSx_out[idx + 26 * planecount] + F[26].x;
			cuSy_out[idx + 26 * planecount] = (cuReIm)cuSy_out[idx + 26 * planecount] + F[26].y;
			cuSz_out[idx + 26 * planecount] = (cuReIm)cuSz_out[idx + 26 * planecount] + F[26].z;

			cuSx_out[idx + 27 * planecount] = (cuReIm)cuSx_out[idx + 27 * planecount] + F[27].x;
			cuSy_out[idx + 27 * planecount] = (cuReIm)cuSy_out[idx + 27 * planecount] + F[27].y;
			cuSz_out[idx + 27 * planecount] = (cuReIm)cuSz_out[idx + 27 * planecount] + F[27].z;

			cuSx_out[idx + 28 * planecount] = (cuReIm)cuSx_out[idx + 28 * planecount] + F[28].x;
			cuSy_out[idx + 28 * planecount] = (cuReIm)cuSy_out[idx + 28 * planecount] + F[28].y;
			cuSz_out[idx + 28 * planecount] = (cuReIm)cuSz_out[idx + 28 * planecount] + F[28].z;

			cuSx_out[idx + 29 * planecount] = (cuReIm)cuSx_out[idx + 29 * planecount] + F[29].x;
			cuSy_out[idx + 29 * planecount] = (cuReIm)cuSy_out[idx + 29 * planecount] + F[29].y;
			cuSz_out[idx + 29 * planecount] = (cuReIm)cuSz_out[idx + 29 * planecount] + F[29].z;

			cuSx_out[idx + 30 * planecount] = (cuReIm)cuSx_out[idx + 30 * planecount] + F[30].x;
			cuSy_out[idx + 30 * planecount] = (cuReIm)cuSy_out[idx + 30 * planecount] + F[30].y;
			cuSz_out[idx + 30 * planecount] = (cuReIm)cuSz_out[idx + 30 * planecount] + F[30].z;

			cuSx_out[idx + 31 * planecount] = (cuReIm)cuSx_out[idx + 31 * planecount] + F[31].x;
			cuSy_out[idx + 31 * planecount] = (cuReIm)cuSy_out[idx + 31 * planecount] + F[31].y;
			cuSz_out[idx + 31 * planecount] = (cuReIm)cuSz_out[idx + 31 * planecount] + F[31].z;
		}

#undef a
#undef b
#undef c
#undef d
#undef e
#undef f
#undef g
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// Q2D - Z SHIFTED

//N = (N.x/2 + 1, N.y, 4)
//xy is transposed
//Complex kernels : this is a z shifted demag version using kernel symmetries, which accumulates in output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_4_transpose_xy_zshifted(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	//z shifted for 3D : can use kernels of reduced dimensions but must be complex
	//
	//Kxx : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//Kyy : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//Kzz : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//
	//Kxy : y - inv. symmetric (-), z - Re part symmetrical  (+), Im part inv. symmetric (-)
	//Kxz : y - symmetrical  (+), z - Re part inv. symmetric (-), Im part symmetrical  (+)
	//Kyz : y - inv. symmetric (-), z - Re part inv. symmetric (-), Im part symmetrical  (+)

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 4, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

		//the z-axis points (the others are zero)
		cuReIm3 a = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 b = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);

		//forward z-axis fft
		//NOTE: cuda fft uses -i for the forward fft and +i for the inverse fft.
		//The kernels are purely real so you would get the same result by taking +i for the forward and -i for the inverse, but better to keep it consistent : use the cuda fft convention here.
		cuReIm3 X0 = a + b;
		cuReIm3 X1 = a - !b;
		cuReIm3 X2 = a - b;
		cuReIm3 X3 = a + !b;

		//kernel multiplication
		cuReIm3 F0, F1, F2, F3;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			//lower z, lower y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F1.x = (Kdiag[ker_baseidx + kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + kerplanecount].y * X1.z);
			F1.y = (Kodiag[ker_baseidx + kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + kerplanecount].z * X1.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			//upper z, lower y
			F3.x = ((~Kdiag[ker_baseidx + kerplanecount].x) * X3.x) + ((~Kodiag[ker_baseidx + kerplanecount].x) * X3.y) - ((~Kodiag[ker_baseidx + kerplanecount].y) * X3.z);
			F3.y = ((~Kodiag[ker_baseidx + kerplanecount].x) * X3.x) + ((~Kdiag[ker_baseidx + kerplanecount].y) * X3.y) - ((~Kodiag[ker_baseidx + kerplanecount].z) * X3.z);
			F3.z = -1 * ((~Kodiag[ker_baseidx + kerplanecount].y) * X3.x) - ((~Kodiag[ker_baseidx + kerplanecount].z) * X3.y) + ((~Kdiag[ker_baseidx + kerplanecount].z) * X3.z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			//lower z, upper y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) - (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = -1 * (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) - (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) - (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F1.x = (Kdiag[ker_baseidx + kerplanecount].x * X1.x) - (Kodiag[ker_baseidx + kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + kerplanecount].y * X1.z);
			F1.y = -1 * (Kodiag[ker_baseidx + kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + kerplanecount].y * X1.y) - (Kodiag[ker_baseidx + kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + kerplanecount].y * X1.x) - (Kodiag[ker_baseidx + kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + kerplanecount].z * X1.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			//upper z, upper y
			F3.x = ((~Kdiag[ker_baseidx + kerplanecount].x) * X3.x) - ((~Kodiag[ker_baseidx + kerplanecount].x) * X3.y) - ((~Kodiag[ker_baseidx + kerplanecount].y) * X3.z);
			F3.y = -1 * ((~Kodiag[ker_baseidx + kerplanecount].x) * X3.x) + ((~Kdiag[ker_baseidx + kerplanecount].y) * X3.y) + ((~Kodiag[ker_baseidx + kerplanecount].z) * X3.z);
			F3.z = -1 * ((~Kodiag[ker_baseidx + kerplanecount].y) * X3.x) + ((~Kodiag[ker_baseidx + kerplanecount].z) * X3.y) + ((~Kdiag[ker_baseidx + kerplanecount].z) * X3.z);
		}

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;
	}
}

//N = (N.x/2 + 1, N.y, 8)
//xy is transposed
//Complex kernels : this is a z shifted demag version using kernel symmetries, which accumulates in output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_8_transpose_xy_zshifted(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	//z shifted for 3D : can use kernels of reduced dimensions but must be complex
	//
	//Kxx : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//Kyy : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//Kzz : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//
	//Kxy : y - inv. symmetric (-), z - Re part symmetrical  (+), Im part inv. symmetric (-)
	//Kxz : y - symmetrical  (+), z - Re part inv. symmetric (-), Im part symmetrical  (+)
	//Kyz : y - inv. symmetric (-), z - Re part inv. symmetric (-), Im part symmetrical  (+)

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 8, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

#define a (cuBReal)0.7071067811865

		//the z-axis points (the others are zero)
		cuReIm3 x0 = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 x1 = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);
		cuReIm3 x2 = cuReIm3(cuSx_in[idx + 2 * planecount], cuSy_in[idx + 2 * planecount], cuSz_in[idx + 2 * planecount]);
		cuReIm3 x3 = cuReIm3(cuSx_in[idx + 3 * planecount], cuSy_in[idx + 3 * planecount], cuSz_in[idx + 3 * planecount]);

		//Radix-4 step
		cuReIm3 X0 = x0 + x2;
		cuReIm3 X2 = x0 - x2;
		cuReIm3 X4 = x0 - !x2;
		cuReIm3 X6 = x0 + !x2;

		cuReIm3 X1 = x1 + x3;
		cuReIm3 X3 = !(x3 - x1);
		cuReIm3 X5 = (x1 - !x3) * cuReIm(a, -a);
		cuReIm3 X7 = (x1 + !x3) * cuReIm(-a, -a);

		//Radix-2 step
		cuReIm3 temp = X0 - X1;
		X0 = X0 + X1;
		X1 = temp;

		temp = X2 - X3;
		X2 = X2 + X3;
		X3 = temp;

		temp = X4 - X5;
		X4 = X4 + X5;
		X5 = temp;

		temp = X6 - X7;
		X6 = X6 + X7;
		X7 = temp;

		//data set in shuffled order:
		//X0, X4, X2, X6, X1, X5, X3, X7

		cuReIm3 F0, F1, F2, F3, F4, F5, F6, F7;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);
		
		if (i <= N.y / 2) {
			
			int ker_baseidx = i + j * (N.y / 2 + 1);

			//lower z, lower y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F4.x = (Kdiag[ker_baseidx + kerplanecount].x * X4.x) + (Kodiag[ker_baseidx + kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + kerplanecount].y * X4.z);
			F4.y = (Kodiag[ker_baseidx + kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + kerplanecount].y * X4.y) + (Kodiag[ker_baseidx + kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + kerplanecount].y * X4.x) + (Kodiag[ker_baseidx + kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + kerplanecount].z * X4.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			F6.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.z);
			F6.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X6.z);

			F1.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.z);
			F1.y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X1.z);

			//upper z, lower y
			F5.x = ((~Kdiag[ker_baseidx + 3 * kerplanecount].x) * X5.x) + ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X5.y) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X5.z);
			F5.y = ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X5.x) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].y) * X5.y) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X5.z);
			F5.z = -1 * ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X5.x) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X5.y) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].z) * X5.z);

			F3.x = ((~Kdiag[ker_baseidx + 2 * kerplanecount].x) * X3.x) + ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X3.y) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X3.z);
			F3.y = ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X3.x) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].y) * X3.y) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X3.z);
			F3.z = -1 * ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X3.x) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X3.y) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].z) * X3.z);

			F7.x = ((~Kdiag[ker_baseidx + kerplanecount].x) * X7.x) + ((~Kodiag[ker_baseidx + kerplanecount].x) * X7.y) - ((~Kodiag[ker_baseidx + kerplanecount].y) * X7.z);
			F7.y = ((~Kodiag[ker_baseidx + kerplanecount].x) * X7.x) + ((~Kdiag[ker_baseidx + kerplanecount].y) * X7.y) - ((~Kodiag[ker_baseidx + kerplanecount].z) * X7.z);
			F7.z = -1 * ((~Kodiag[ker_baseidx + kerplanecount].y) * X7.x) - ((~Kodiag[ker_baseidx + kerplanecount].z) * X7.y) + ((~Kdiag[ker_baseidx + kerplanecount].z) * X7.z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			//lower z, upper y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) - (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = -1 * (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) - (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) - (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F4.x = (Kdiag[ker_baseidx + kerplanecount].x * X4.x) - (Kodiag[ker_baseidx + kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + kerplanecount].y * X4.z);
			F4.y = -1 * (Kodiag[ker_baseidx + kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + kerplanecount].y * X4.y) - (Kodiag[ker_baseidx + kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + kerplanecount].y * X4.x) - (Kodiag[ker_baseidx + kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + kerplanecount].z * X4.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			F6.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X6.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.z);
			F6.y = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X6.y) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X6.z);

			F1.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X1.x) - (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.z);
			F1.y = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X1.y) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.x) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X1.z);

			//upper z, upper y
			F5.x = ((~Kdiag[ker_baseidx + 3 * kerplanecount].x) * X5.x) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X5.y) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X5.z);
			F5.y = -1 * ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X5.x) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].y) * X5.y) + ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X5.z);
			F5.z = -1 * ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X5.x) + ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X5.y) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].z) * X5.z);

			F3.x = ((~Kdiag[ker_baseidx + 2 * kerplanecount].x) * X3.x) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X3.y) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X3.z);
			F3.y = -1 * ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X3.x) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].y) * X3.y) + ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X3.z);
			F3.z = -1 * ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X3.x) + ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X3.y) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].z) * X3.z);

			F7.x = ((~Kdiag[ker_baseidx + kerplanecount].x) * X7.x) - ((~Kodiag[ker_baseidx + kerplanecount].x) * X7.y) - ((~Kodiag[ker_baseidx + kerplanecount].y) * X7.z);
			F7.y = -1 * ((~Kodiag[ker_baseidx + kerplanecount].x) * X7.x) + ((~Kdiag[ker_baseidx + kerplanecount].y) * X7.y) + ((~Kodiag[ker_baseidx + kerplanecount].z) * X7.z);
			F7.z = -1 * ((~Kodiag[ker_baseidx + kerplanecount].y) * X7.x) + ((~Kodiag[ker_baseidx + kerplanecount].z) * X7.y) + ((~Kdiag[ker_baseidx + kerplanecount].z) * X7.z);
		}
		
		//accumulate result in output ready for ifft
		
		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;

		cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F4.x;
		cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F4.y;
		cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F4.z;

		cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F5.x;
		cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F5.y;
		cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F5.z;

		cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F6.x;
		cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F6.y;
		cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F6.z;

		cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F7.x;
		cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F7.y;
		cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F7.z;

#undef a
	}
}

//N = (N.x/2 + 1, N.y, 16)
//xy is transposed
//Complex kernels : this is a z shifted demag version using kernel symmetries, which accumulates in output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_16_transpose_xy_zshifted(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	//z shifted for 3D : can use kernels of reduced dimensions but must be complex
	//
	//Kxx : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//Kyy : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//Kzz : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//
	//Kxy : y - inv. symmetric (-), z - Re part symmetrical  (+), Im part inv. symmetric (-)
	//Kxz : y - symmetrical  (+), z - Re part inv. symmetric (-), Im part symmetrical  (+)
	//Kyz : y - inv. symmetric (-), z - Re part inv. symmetric (-), Im part symmetrical  (+)

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 16, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

		//the z-axis points (the others are zero)
		cuReIm3 x0 = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 x1 = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);
		cuReIm3 x2 = cuReIm3(cuSx_in[idx + 2 * planecount], cuSy_in[idx + 2 * planecount], cuSz_in[idx + 2 * planecount]);
		cuReIm3 x3 = cuReIm3(cuSx_in[idx + 3 * planecount], cuSy_in[idx + 3 * planecount], cuSz_in[idx + 3 * planecount]);
		cuReIm3 x4 = cuReIm3(cuSx_in[idx + 4 * planecount], cuSy_in[idx + 4 * planecount], cuSz_in[idx + 4 * planecount]);
		cuReIm3 x5 = cuReIm3(cuSx_in[idx + 5 * planecount], cuSy_in[idx + 5 * planecount], cuSz_in[idx + 5 * planecount]);
		cuReIm3 x6 = cuReIm3(cuSx_in[idx + 6 * planecount], cuSy_in[idx + 6 * planecount], cuSz_in[idx + 6 * planecount]);
		cuReIm3 x7 = cuReIm3(cuSx_in[idx + 7 * planecount], cuSy_in[idx + 7 * planecount], cuSz_in[idx + 7 * planecount]);

#define a	(cuBReal)9.238795325113E-01
#define b	(cuBReal)3.826834323651E-01
#define c	(cuBReal)7.071067811865E-01

		//First stage
		cuReIm3 X0 = x0 + x4;
		cuReIm3 X4 = x0 - x4;
		cuReIm3 X8 = x0 - !x4;
		cuReIm3 X12 = x0 + !x4;

		cuReIm3 X1 = x1 + x5;
		cuReIm3 X5 = (x1 - x5) * cuReIm(c, -c);
		cuReIm3 X9 = (x1 - !x5) * cuReIm(a, -b);
		cuReIm3 X13 = (x1 + !x5) * cuReIm(b, -a);

		cuReIm3 X2 = x2 + x6;
		cuReIm3 X6 = !(x6 - x2);
		cuReIm3 X10 = (x2 - !x6) * cuReIm(c, -c);
		cuReIm3 X14 = (x2 + !x6) * cuReIm(-c, -c);

		cuReIm3 X3 = x3 + x7;
		cuReIm3 X7 = (x3 - x7) * cuReIm(-c, -c);
		cuReIm3 X11 = (x3 - !x7) * cuReIm(b, -a);
		cuReIm3 X15 = (x3 + !x7) * cuReIm(-a, b);

		//Second stage
		cuReIm3 t0 = X0 + X2;
		cuReIm3 t1 = X0 - X2;
		cuReIm3 t2 = X1 + X3;
		cuReIm3 t3 = !(X3 - X1);

		X0 = t0 + t2;
		X1 = t0 - t2;
		X2 = t1 + t3;
		X3 = t1 - t3;

		t0 = X4 + X6;
		t1 = X4 - X6;
		t2 = X5 + X7;
		t3 = !(X7 - X5);

		X4 = t0 + t2;
		X5 = t0 - t2;
		X6 = t1 + t3;
		X7 = t1 - t3;

		t0 = X8 + X10;
		t1 = X8 - X10;
		t2 = X9 + X11;
		t3 = !(X11 - X9);

		X8 = t0 + t2;
		X9 = t0 - t2;
		X10 = t1 + t3;
		X11 = t1 - t3;

		t0 = X12 + X14;
		t1 = X12 - X14;
		t2 = X13 + X15;
		t3 = !(X15 - X13);

		X12 = t0 + t2;
		X13 = t0 - t2;
		X14 = t1 + t3;
		X15 = t1 - t3;

		//output is shuffled now, i.e. it is ordered as:
		//X0, X8, X4, X12, X2, X10, X6, X14, X1, X9, X5, X13, X3, X11, X7, X15

		cuReIm3 F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			//lower z, lower y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F8.x = (Kdiag[ker_baseidx + kerplanecount].x * X8.x) + (Kodiag[ker_baseidx + kerplanecount].x * X8.y) + (Kodiag[ker_baseidx + kerplanecount].y * X8.z);
			F8.y = (Kodiag[ker_baseidx + kerplanecount].x * X8.x) + (Kdiag[ker_baseidx + kerplanecount].y * X8.y) + (Kodiag[ker_baseidx + kerplanecount].z * X8.z);
			F8.z = (Kodiag[ker_baseidx + kerplanecount].y * X8.x) + (Kodiag[ker_baseidx + kerplanecount].z * X8.y) + (Kdiag[ker_baseidx + kerplanecount].z * X8.z);

			F4.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.z);
			F4.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X4.z);

			F12.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.z);
			F12.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.z);
			F12.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X12.z);

			F2.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.z);
			F2.y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X2.z);

			F10.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.z);
			F10.y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.z);
			F10.z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X10.z);

			F6.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.z);
			F6.y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X6.z);

			F14.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.z);
			F14.y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.z);
			F14.z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X14.z);

			F1.x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.z);
			F1.y = (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X1.z);

			//upper z, lower y
			F9.x = ((~Kdiag[ker_baseidx + 7 * kerplanecount].x) * X9.x) + ((~Kodiag[ker_baseidx + 7 * kerplanecount].x) * X9.y) - ((~Kodiag[ker_baseidx + 7 * kerplanecount].y) * X9.z);
			F9.y = ((~Kodiag[ker_baseidx + 7 * kerplanecount].x) * X9.x) + ((~Kdiag[ker_baseidx + 7 * kerplanecount].y) * X9.y) - ((~Kodiag[ker_baseidx + 7 * kerplanecount].z) * X9.z);
			F9.z = -1 * ((~Kodiag[ker_baseidx + 7 * kerplanecount].y) * X9.x) - ((~Kodiag[ker_baseidx + 7 * kerplanecount].z) * X9.y) + ((~Kdiag[ker_baseidx + 7 * kerplanecount].z) * X9.z);

			F5.x = ((~Kdiag[ker_baseidx + 6 * kerplanecount].x) * X5.x) + ((~Kodiag[ker_baseidx + 6 * kerplanecount].x) * X5.y) - ((~Kodiag[ker_baseidx + 6 * kerplanecount].y) * X5.z);
			F5.y = ((~Kodiag[ker_baseidx + 6 * kerplanecount].x) * X5.x) + ((~Kdiag[ker_baseidx + 6 * kerplanecount].y) * X5.y) - ((~Kodiag[ker_baseidx + 6 * kerplanecount].z) * X5.z);
			F5.z = -1 * ((~Kodiag[ker_baseidx + 6 * kerplanecount].y) * X5.x) - ((~Kodiag[ker_baseidx + 6 * kerplanecount].z) * X5.y) + ((~Kdiag[ker_baseidx + 6 * kerplanecount].z) * X5.z);

			F13.x = ((~Kdiag[ker_baseidx + 5 * kerplanecount].x) * X13.x) + ((~Kodiag[ker_baseidx + 5 * kerplanecount].x) * X13.y) - ((~Kodiag[ker_baseidx + 5 * kerplanecount].y) * X13.z);
			F13.y = ((~Kodiag[ker_baseidx + 5 * kerplanecount].x) * X13.x) + ((~Kdiag[ker_baseidx + 5 * kerplanecount].y) * X13.y) - ((~Kodiag[ker_baseidx + 5 * kerplanecount].z) * X13.z);
			F13.z = -1 * ((~Kodiag[ker_baseidx + 5 * kerplanecount].y) * X13.x) - ((~Kodiag[ker_baseidx + 5 * kerplanecount].z) * X13.y) + ((~Kdiag[ker_baseidx + 5 * kerplanecount].z) * X13.z);

			F3.x = ((~Kdiag[ker_baseidx + 4 * kerplanecount].x) * X3.x) + ((~Kodiag[ker_baseidx + 4 * kerplanecount].x) * X3.y) - ((~Kodiag[ker_baseidx + 4 * kerplanecount].y) * X3.z);
			F3.y = ((~Kodiag[ker_baseidx + 4 * kerplanecount].x) * X3.x) + ((~Kdiag[ker_baseidx + 4 * kerplanecount].y) * X3.y) - ((~Kodiag[ker_baseidx + 4 * kerplanecount].z) * X3.z);
			F3.z = -1 * ((~Kodiag[ker_baseidx + 4 * kerplanecount].y) * X3.x) - ((~Kodiag[ker_baseidx + 4 * kerplanecount].z) * X3.y) + ((~Kdiag[ker_baseidx + 4 * kerplanecount].z) * X3.z);

			F11.x = ((~Kdiag[ker_baseidx + 3 * kerplanecount].x) * X11.x) + ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X11.y) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X11.z);
			F11.y = ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X11.x) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].y) * X11.y) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X11.z);
			F11.z = -1 * ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X11.x) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X11.y) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].z) * X11.z);

			F7.x = ((~Kdiag[ker_baseidx + 2 * kerplanecount].x) * X7.x) + ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X7.y) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X7.z);
			F7.y = ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X7.x) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].y) * X7.y) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X7.z);
			F7.z = -1 * ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X7.x) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X7.y) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].z) * X7.z);

			F15.x = ((~Kdiag[ker_baseidx + kerplanecount].x) * X15.x) + ((~Kodiag[ker_baseidx + kerplanecount].x) * X15.y) - ((~Kodiag[ker_baseidx + kerplanecount].y) * X15.z);
			F15.y = ((~Kodiag[ker_baseidx + kerplanecount].x) * X15.x) + ((~Kdiag[ker_baseidx + kerplanecount].y) * X15.y) - ((~Kodiag[ker_baseidx + kerplanecount].z) * X15.z);
			F15.z = -1 * ((~Kodiag[ker_baseidx + kerplanecount].y) * X15.x) - ((~Kodiag[ker_baseidx + kerplanecount].z) * X15.y) + ((~Kdiag[ker_baseidx + kerplanecount].z) * X15.z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			//lower z, upper y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) - (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = -1 * (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) - (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) - (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F8.x = (Kdiag[ker_baseidx + kerplanecount].x * X8.x) - (Kodiag[ker_baseidx + kerplanecount].x * X8.y) + (Kodiag[ker_baseidx + kerplanecount].y * X8.z);
			F8.y = -1 * (Kodiag[ker_baseidx + kerplanecount].x * X8.x) + (Kdiag[ker_baseidx + kerplanecount].y * X8.y) - (Kodiag[ker_baseidx + kerplanecount].z * X8.z);
			F8.z = (Kodiag[ker_baseidx + kerplanecount].y * X8.x) - (Kodiag[ker_baseidx + kerplanecount].z * X8.y) + (Kdiag[ker_baseidx + kerplanecount].z * X8.z);

			F4.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X4.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.z);
			F4.y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X4.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X4.z);

			F12.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X12.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.z);
			F12.y = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X12.y) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.z);
			F12.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X12.z);

			F2.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X2.x) - (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.z);
			F2.y = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X2.y) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.x) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X2.z);

			F10.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X10.x) - (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.z);
			F10.y = -1 * (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X10.y) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.z);
			F10.z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.x) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X10.z);

			F6.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X6.x) - (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.z);
			F6.y = -1 * (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X6.y) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.x) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X6.z);

			F14.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X14.x) - (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.z);
			F14.y = -1 * (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X14.y) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.z);
			F14.z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.x) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X14.z);

			F1.x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X1.x) - (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.z);
			F1.y = -1 * (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X1.y) - (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.x) - (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X1.z);

			//upper z, upper y
			F9.x = ((~Kdiag[ker_baseidx + 7 * kerplanecount].x) * X9.x) - ((~Kodiag[ker_baseidx + 7 * kerplanecount].x) * X9.y) - ((~Kodiag[ker_baseidx + 7 * kerplanecount].y) * X9.z);
			F9.y = -1 * ((~Kodiag[ker_baseidx + 7 * kerplanecount].x) * X9.x) + ((~Kdiag[ker_baseidx + 7 * kerplanecount].y) * X9.y) + ((~Kodiag[ker_baseidx + 7 * kerplanecount].z) * X9.z);
			F9.z = -1 * ((~Kodiag[ker_baseidx + 7 * kerplanecount].y) * X9.x) + ((~Kodiag[ker_baseidx + 7 * kerplanecount].z) * X9.y) + ((~Kdiag[ker_baseidx + 7 * kerplanecount].z) * X9.z);

			F5.x = ((~Kdiag[ker_baseidx + 6 * kerplanecount].x) * X5.x) - ((~Kodiag[ker_baseidx + 6 * kerplanecount].x) * X5.y) - ((~Kodiag[ker_baseidx + 6 * kerplanecount].y) * X5.z);
			F5.y = -1 * ((~Kodiag[ker_baseidx + 6 * kerplanecount].x) * X5.x) + ((~Kdiag[ker_baseidx + 6 * kerplanecount].y) * X5.y) + ((~Kodiag[ker_baseidx + 6 * kerplanecount].z) * X5.z);
			F5.z = -1 * ((~Kodiag[ker_baseidx + 6 * kerplanecount].y) * X5.x) + ((~Kodiag[ker_baseidx + 6 * kerplanecount].z) * X5.y) + ((~Kdiag[ker_baseidx + 6 * kerplanecount].z) * X5.z);

			F13.x = ((~Kdiag[ker_baseidx + 5 * kerplanecount].x) * X13.x) - ((~Kodiag[ker_baseidx + 5 * kerplanecount].x) * X13.y) - ((~Kodiag[ker_baseidx + 5 * kerplanecount].y) * X13.z);
			F13.y = -1 * ((~Kodiag[ker_baseidx + 5 * kerplanecount].x) * X13.x) + ((~Kdiag[ker_baseidx + 5 * kerplanecount].y) * X13.y) + ((~Kodiag[ker_baseidx + 5 * kerplanecount].z) * X13.z);
			F13.z = -1 * ((~Kodiag[ker_baseidx + 5 * kerplanecount].y) * X13.x) + ((~Kodiag[ker_baseidx + 5 * kerplanecount].z) * X13.y) + ((~Kdiag[ker_baseidx + 5 * kerplanecount].z) * X13.z);

			F3.x = ((~Kdiag[ker_baseidx + 4 * kerplanecount].x) * X3.x) - ((~Kodiag[ker_baseidx + 4 * kerplanecount].x) * X3.y) - ((~Kodiag[ker_baseidx + 4 * kerplanecount].y) * X3.z);
			F3.y = -1 * ((~Kodiag[ker_baseidx + 4 * kerplanecount].x) * X3.x) + ((~Kdiag[ker_baseidx + 4 * kerplanecount].y) * X3.y) + ((~Kodiag[ker_baseidx + 4 * kerplanecount].z) * X3.z);
			F3.z = -1 * ((~Kodiag[ker_baseidx + 4 * kerplanecount].y) * X3.x) + ((~Kodiag[ker_baseidx + 4 * kerplanecount].z) * X3.y) + ((~Kdiag[ker_baseidx + 4 * kerplanecount].z) * X3.z);

			F11.x = ((~Kdiag[ker_baseidx + 3 * kerplanecount].x) * X11.x) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X11.y) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X11.z);
			F11.y = -1 * ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X11.x) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].y) * X11.y) + ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X11.z);
			F11.z = -1 * ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X11.x) + ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X11.y) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].z) * X11.z);

			F7.x = ((~Kdiag[ker_baseidx + 2 * kerplanecount].x) * X7.x) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X7.y) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X7.z);
			F7.y = -1 * ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X7.x) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].y) * X7.y) + ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X7.z);
			F7.z = -1 * ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X7.x) + ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X7.y) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].z) * X7.z);

			F15.x = ((~Kdiag[ker_baseidx + kerplanecount].x) * X15.x) - ((~Kodiag[ker_baseidx + kerplanecount].x) * X15.y) - ((~Kodiag[ker_baseidx + kerplanecount].y) * X15.z);
			F15.y = -1 * ((~Kodiag[ker_baseidx + kerplanecount].x) * X15.x) + ((~Kdiag[ker_baseidx + kerplanecount].y) * X15.y) + ((~Kodiag[ker_baseidx + kerplanecount].z) * X15.z);
			F15.z = -1 * ((~Kodiag[ker_baseidx + kerplanecount].y) * X15.x) + ((~Kodiag[ker_baseidx + kerplanecount].z) * X15.y) + ((~Kdiag[ker_baseidx + kerplanecount].z) * X15.z);
		}

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;

		cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F4.x;
		cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F4.y;
		cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F4.z;

		cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F5.x;
		cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F5.y;
		cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F5.z;

		cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F6.x;
		cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F6.y;
		cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F6.z;

		cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F7.x;
		cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F7.y;
		cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F7.z;

		cuSx_out[idx + 8 * planecount] = (cuReIm)cuSx_out[idx + 8 * planecount] + F8.x;
		cuSy_out[idx + 8 * planecount] = (cuReIm)cuSy_out[idx + 8 * planecount] + F8.y;
		cuSz_out[idx + 8 * planecount] = (cuReIm)cuSz_out[idx + 8 * planecount] + F8.z;

		cuSx_out[idx + 9 * planecount] = (cuReIm)cuSx_out[idx + 9 * planecount] + F9.x;
		cuSy_out[idx + 9 * planecount] = (cuReIm)cuSy_out[idx + 9 * planecount] + F9.y;
		cuSz_out[idx + 9 * planecount] = (cuReIm)cuSz_out[idx + 9 * planecount] + F9.z;

		cuSx_out[idx + 10 * planecount] = (cuReIm)cuSx_out[idx + 10 * planecount] + F10.x;
		cuSy_out[idx + 10 * planecount] = (cuReIm)cuSy_out[idx + 10 * planecount] + F10.y;
		cuSz_out[idx + 10 * planecount] = (cuReIm)cuSz_out[idx + 10 * planecount] + F10.z;

		cuSx_out[idx + 11 * planecount] = (cuReIm)cuSx_out[idx + 11 * planecount] + F11.x;
		cuSy_out[idx + 11 * planecount] = (cuReIm)cuSy_out[idx + 11 * planecount] + F11.y;
		cuSz_out[idx + 11 * planecount] = (cuReIm)cuSz_out[idx + 11 * planecount] + F11.z;

		cuSx_out[idx + 12 * planecount] = (cuReIm)cuSx_out[idx + 12 * planecount] + F12.x;
		cuSy_out[idx + 12 * planecount] = (cuReIm)cuSy_out[idx + 12 * planecount] + F12.y;
		cuSz_out[idx + 12 * planecount] = (cuReIm)cuSz_out[idx + 12 * planecount] + F12.z;

		cuSx_out[idx + 13 * planecount] = (cuReIm)cuSx_out[idx + 13 * planecount] + F13.x;
		cuSy_out[idx + 13 * planecount] = (cuReIm)cuSy_out[idx + 13 * planecount] + F13.y;
		cuSz_out[idx + 13 * planecount] = (cuReIm)cuSz_out[idx + 13 * planecount] + F13.z;

		cuSx_out[idx + 14 * planecount] = (cuReIm)cuSx_out[idx + 14 * planecount] + F14.x;
		cuSy_out[idx + 14 * planecount] = (cuReIm)cuSy_out[idx + 14 * planecount] + F14.y;
		cuSz_out[idx + 14 * planecount] = (cuReIm)cuSz_out[idx + 14 * planecount] + F14.z;

		cuSx_out[idx + 15 * planecount] = (cuReIm)cuSx_out[idx + 15 * planecount] + F15.x;
		cuSy_out[idx + 15 * planecount] = (cuReIm)cuSy_out[idx + 15 * planecount] + F15.y;
		cuSz_out[idx + 15 * planecount] = (cuReIm)cuSz_out[idx + 15 * planecount] + F15.z;

#undef a
#undef b
#undef c
	}
}

//N = (N.x/2 + 1, N.y, 32)
//xy is transposed
//Complex kernels : this is a z shifted demag version using kernel symmetries, which accumulates in output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_32_transpose_xy_zshifted(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	//z shifted for 3D : can use kernels of reduced dimensions but must be complex
	//
	//Kxx : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//Kyy : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//Kzz : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
	//
	//Kxy : y - inv. symmetric (-), z - Re part symmetrical  (+), Im part inv. symmetric (-)
	//Kxz : y - symmetrical  (+), z - Re part inv. symmetric (-), Im part symmetrical  (+)
	//Kyz : y - inv. symmetric (-), z - Re part inv. symmetric (-), Im part symmetrical  (+)

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 32, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

		//input data
#define x(n)	(cuReIm3(cuSx_in[idx + (n) * planecount], cuSy_in[idx + (n) * planecount], cuSz_in[idx + (n) * planecount]))

		//no performance gain to be had from setting these as X0, X1, ... etc.
		//unrolling loops does make a slight difference though - probably last case for which you want to unroll loops
		cuReIm3 X[32];

		cuReIm3 t0, t1, t2, t3;

		//input stage

#define a	(cuBReal)0.980785280403230
#define b	(cuBReal)0.195090322016128
#define c	(cuBReal)0.923879532511287
#define d	(cuBReal)0.382683432365090
#define e	(cuBReal)0.831469612302545
#define f	(cuBReal)0.555570233019602
#define g	(cuBReal)0.707106781186548

		//j = 0
		X[0] = (x(0) + x(8));
		X[8] = (x(0) - x(8));
		X[16] = (x(0) - !x(8));
		X[24] = (x(0) + !x(8));

		//j = 1
		X[1] = (x(1) + x(9));
		X[9] = (x(1) - x(9)) * cuReIm(c, -d);
		X[17] = (x(1) - !x(9)) * cuReIm(a, -b);
		X[25] = (x(1) + !x(9)) * cuReIm(e, -f);

		//j = 2
		X[2] = (x(2) + x(10));
		X[10] = (x(2) - x(10)) * cuReIm(g, -g);
		X[18] = (x(2) - !x(10)) * cuReIm(c, -d);
		X[26] = (x(2) + !x(10)) * cuReIm(d, -c);

		//j = 3
		X[3] = (x(3) + x(11));
		X[11] = (x(3) - x(11)) * cuReIm(d, -c);
		X[19] = (x(3) - !x(11)) * cuReIm(e, -f);
		X[27] = (x(3) + !x(11)) * cuReIm(-b, -a);

		//j = 4
		X[4] = (x(4) + x(12));
		X[12] = !(x(12) - x(4));
		X[20] = (x(4) - !x(12)) * cuReIm(g, -g);
		X[28] = (x(4) + !x(12)) * cuReIm(-g, -g);

		//j = 5
		X[5] = (x(5) + x(13));
		X[13] = (x(5) - x(13)) * cuReIm(-d, -c);
		X[21] = (x(5) - !x(13)) * cuReIm(f, -e);
		X[29] = (x(5) + !x(13)) * cuReIm(-a, -b);

		//j = 6
		X[6] = (x(6) + x(14));
		X[14] = (x(6) - x(14)) * cuReIm(-g, -g);
		X[22] = (x(6) - !x(14)) * cuReIm(d, -c);
		X[30] = (x(6) + !x(14)) * cuReIm(-c, d);

		//j = 7
		X[7] = (x(7) + x(15));
		X[15] = (x(7) - x(15)) * cuReIm(-c, -d);
		X[23] = (x(7) - !x(15)) * cuReIm(b, -a);
		X[31] = (x(7) + !x(15)) * cuReIm(-f, e);

#undef x

		//final radix4 stage

		//j = 0
		t0 = (X[0] + X[4]);
		t1 = (X[0] - X[4]);
		t2 = (X[2] + X[6]);
		t3 = !(X[6] - X[2]);

		X[0] = (t0 + t2);
		X[2] = (t0 - t2);
		X[4] = (t1 + t3);
		X[6] = (t1 - t3);

		t0 = (X[8] + X[12]);
		t1 = (X[8] - X[12]);
		t2 = (X[10] + X[14]);
		t3 = !(X[14] - X[10]);

		X[8] = (t0 + t2);
		X[10] = (t0 - t2);
		X[12] = (t1 + t3);
		X[14] = (t1 - t3);

		t0 = (X[16] + X[20]);
		t1 = (X[16] - X[20]);
		t2 = (X[18] + X[22]);
		t3 = !(X[22] - X[18]);

		X[16] = (t0 + t2);
		X[18] = (t0 - t2);
		X[20] = (t1 + t3);
		X[22] = (t1 - t3);

		t0 = (X[24] + X[28]);
		t1 = (X[24] - X[28]);
		t2 = (X[26] + X[30]);
		t3 = !(X[30] - X[26]);

		X[24] = (t0 + t2);
		X[26] = (t0 - t2);
		X[28] = (t1 + t3);
		X[30] = (t1 - t3);

		//j = 1
		t0 = (X[1] + X[5]);
		t1 = (X[1] - X[5]);
		t2 = (X[3] + X[7]);
		t3 = !(X[7] - X[3]);

		X[1] = (t0 + t2);
		X[3] = !(t2 - t0);
		X[5] = (t1 + t3) * cuReIm(g, -g);
		X[7] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[9] + X[13]);
		t1 = (X[9] - X[13]);
		t2 = (X[11] + X[15]);
		t3 = !(X[15] - X[11]);

		X[9] = (t0 + t2);
		X[11] = !(t2 - t0);
		X[13] = (t1 + t3) * cuReIm(g, -g);
		X[15] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[17] + X[21]);
		t1 = (X[17] - X[21]);
		t2 = (X[19] + X[23]);
		t3 = !(X[23] - X[19]);

		X[17] = (t0 + t2);
		X[19] = !(t2 - t0);
		X[21] = (t1 + t3) * cuReIm(g, -g);
		X[23] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[25] + X[29]);
		t1 = (X[25] - X[29]);
		t2 = (X[27] + X[31]);
		t3 = !(X[31] - X[27]);

		X[25] = (t0 + t2);
		X[27] = !(t2 - t0);
		X[29] = (t1 + t3) * cuReIm(g, -g);
		X[31] = (t1 - t3) * cuReIm(-g, -g);

		//radix-2 step to finish
		t0 = X[0] - X[1];
		X[0] = X[0] + X[1];
		X[1] = t0;

		t0 = X[2] - X[3];
		X[2] = X[2] + X[3];
		X[3] = t0;

		t0 = X[4] - X[5];
		X[4] = X[4] + X[5];
		X[5] = t0;

		t0 = X[6] - X[7];
		X[6] = X[6] + X[7];
		X[7] = t0;

		t0 = X[8] - X[9];
		X[8] = X[8] + X[9];
		X[9] = t0;

		t0 = X[10] - X[11];
		X[10] = X[10] + X[11];
		X[11] = t0;

		t0 = X[12] - X[13];
		X[12] = X[12] + X[13];
		X[13] = t0;

		t0 = X[14] - X[15];
		X[14] = X[14] + X[15];
		X[15] = t0;

		t0 = X[16] - X[17];
		X[16] = X[16] + X[17];
		X[17] = t0;

		t0 = X[18] - X[19];
		X[18] = X[18] + X[19];
		X[19] = t0;

		t0 = X[20] - X[21];
		X[20] = X[20] + X[21];
		X[21] = t0;

		t0 = X[22] - X[23];
		X[22] = X[22] + X[23];
		X[23] = t0;

		t0 = X[24] - X[25];
		X[24] = X[24] + X[25];
		X[25] = t0;

		t0 = X[26] - X[27];
		X[26] = X[26] + X[27];
		X[27] = t0;

		t0 = X[28] - X[29];
		X[28] = X[28] + X[29];
		X[29] = t0;

		t0 = X[30] - X[31];
		X[30] = X[30] + X[31];
		X[31] = t0;

		//output is shuffled now, i.e. it is ordered as:
		//0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		cuReIm3 F[32];

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			//lower z, lower y
			F[0].x = (Kdiag[ker_baseidx].x * X[0].x) + (Kodiag[ker_baseidx].x * X[0].y) + (Kodiag[ker_baseidx].y * X[0].z);
			F[0].y = (Kodiag[ker_baseidx].x * X[0].x) + (Kdiag[ker_baseidx].y * X[0].y) + (Kodiag[ker_baseidx].z * X[0].z);
			F[0].z = (Kodiag[ker_baseidx].y * X[0].x) + (Kodiag[ker_baseidx].z * X[0].y) + (Kdiag[ker_baseidx].z * X[0].z);

			F[16].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].z);
			F[16].y = (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);
			F[16].z = (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);

			F[8].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].z);
			F[8].y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);
			F[8].z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);

			F[24].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].z);
			F[24].y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);
			F[24].z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);

			F[4].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].z);
			F[4].y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);
			F[4].z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);

			F[20].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].z);
			F[20].y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);
			F[20].z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);

			F[12].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].z);
			F[12].y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);
			F[12].z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);

			F[28].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].z);
			F[28].y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);
			F[28].z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);

			F[2].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].z);
			F[2].y = (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);
			F[2].z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);

			F[18].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].z);
			F[18].y = (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);
			F[18].z = (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);

			F[10].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].z);
			F[10].y = (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);
			F[10].z = (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);

			F[26].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].z);
			F[26].y = (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);
			F[26].z = (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);

			F[6].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].z);
			F[6].y = (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);
			F[6].z = (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);

			F[22].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].z);
			F[22].y = (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);
			F[22].z = (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);

			F[14].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].z);
			F[14].y = (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);
			F[14].z = (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);

			F[30].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].z);
			F[30].y = (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);
			F[30].z = (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);

			F[1].x = (Kdiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].z);
			F[1].y = (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kdiag[ker_baseidx + 16 * kerplanecount].y * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);
			F[1].z = (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].x) + (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].y) + (Kdiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);

			//upper z, lower y
			F[17].x = ((~Kdiag[ker_baseidx + 15 * kerplanecount].x) * X[17].x) + ((~Kodiag[ker_baseidx + 15 * kerplanecount].x) * X[17].y) - ((~Kodiag[ker_baseidx + 15 * kerplanecount].y) * X[17].z);
			F[17].y = ((~Kodiag[ker_baseidx + 15 * kerplanecount].x) * X[17].x) + ((~Kdiag[ker_baseidx + 15 * kerplanecount].y) * X[17].y) - ((~Kodiag[ker_baseidx + 15 * kerplanecount].z) * X[17].z);
			F[17].z = -1 * ((~Kodiag[ker_baseidx + 15 * kerplanecount].y) * X[17].x) - ((~Kodiag[ker_baseidx + 15 * kerplanecount].z) * X[17].y) + ((~Kdiag[ker_baseidx + 15 * kerplanecount].z) * X[17].z);

			F[9].x = ((~Kdiag[ker_baseidx + 14 * kerplanecount].x) * X[9].x) + ((~Kodiag[ker_baseidx + 14 * kerplanecount].x) * X[9].y) - ((~Kodiag[ker_baseidx + 14 * kerplanecount].y) * X[9].z);
			F[9].y = ((~Kodiag[ker_baseidx + 14 * kerplanecount].x) * X[9].x) + ((~Kdiag[ker_baseidx + 14 * kerplanecount].y) * X[9].y) - ((~Kodiag[ker_baseidx + 14 * kerplanecount].z) * X[9].z);
			F[9].z = -1 * ((~Kodiag[ker_baseidx + 14 * kerplanecount].y) * X[9].x) - ((~Kodiag[ker_baseidx + 14 * kerplanecount].z) * X[9].y) + ((~Kdiag[ker_baseidx + 14 * kerplanecount].z) * X[9].z);

			F[25].x = ((~Kdiag[ker_baseidx + 13 * kerplanecount].x) * X[25].x) + ((~Kodiag[ker_baseidx + 13 * kerplanecount].x) * X[25].y) - ((~Kodiag[ker_baseidx + 13 * kerplanecount].y) * X[25].z);
			F[25].y = ((~Kodiag[ker_baseidx + 13 * kerplanecount].x) * X[25].x) + ((~Kdiag[ker_baseidx + 13 * kerplanecount].y) * X[25].y) - ((~Kodiag[ker_baseidx + 13 * kerplanecount].z) * X[25].z);
			F[25].z = -1 * ((~Kodiag[ker_baseidx + 13 * kerplanecount].y) * X[25].x) - ((~Kodiag[ker_baseidx + 13 * kerplanecount].z) * X[25].y) + ((~Kdiag[ker_baseidx + 13 * kerplanecount].z) * X[25].z);

			F[5].x = ((~Kdiag[ker_baseidx + 12 * kerplanecount].x) * X[5].x) + ((~Kodiag[ker_baseidx + 12 * kerplanecount].x) * X[5].y) - ((~Kodiag[ker_baseidx + 12 * kerplanecount].y) * X[5].z);
			F[5].y = ((~Kodiag[ker_baseidx + 12 * kerplanecount].x) * X[5].x) + ((~Kdiag[ker_baseidx + 12 * kerplanecount].y) * X[5].y) - ((~Kodiag[ker_baseidx + 12 * kerplanecount].z) * X[5].z);
			F[5].z = -1 * ((~Kodiag[ker_baseidx + 12 * kerplanecount].y) * X[5].x) - ((~Kodiag[ker_baseidx + 12 * kerplanecount].z) * X[5].y) + ((~Kdiag[ker_baseidx + 12 * kerplanecount].z) * X[5].z);

			F[21].x = ((~Kdiag[ker_baseidx + 11 * kerplanecount].x) * X[21].x) + ((~Kodiag[ker_baseidx + 11 * kerplanecount].x) * X[21].y) - ((~Kodiag[ker_baseidx + 11 * kerplanecount].y) * X[21].z);
			F[21].y = ((~Kodiag[ker_baseidx + 11 * kerplanecount].x) * X[21].x) + ((~Kdiag[ker_baseidx + 11 * kerplanecount].y) * X[21].y) - ((~Kodiag[ker_baseidx + 11 * kerplanecount].z) * X[21].z);
			F[21].z = -1 * ((~Kodiag[ker_baseidx + 11 * kerplanecount].y) * X[21].x) - ((~Kodiag[ker_baseidx + 11 * kerplanecount].z) * X[21].y) + ((~Kdiag[ker_baseidx + 11 * kerplanecount].z) * X[21].z);

			F[13].x = ((~Kdiag[ker_baseidx + 10 * kerplanecount].x) * X[13].x) + ((~Kodiag[ker_baseidx + 10 * kerplanecount].x) * X[13].y) - ((~Kodiag[ker_baseidx + 10 * kerplanecount].y) * X[13].z);
			F[13].y = ((~Kodiag[ker_baseidx + 10 * kerplanecount].x) * X[13].x) + ((~Kdiag[ker_baseidx + 10 * kerplanecount].y) * X[13].y) - ((~Kodiag[ker_baseidx + 10 * kerplanecount].z) * X[13].z);
			F[13].z = -1 * ((~Kodiag[ker_baseidx + 10 * kerplanecount].y) * X[13].x) - ((~Kodiag[ker_baseidx + 10 * kerplanecount].z) * X[13].y) + ((~Kdiag[ker_baseidx + 10 * kerplanecount].z) * X[13].z);

			F[29].x = ((~Kdiag[ker_baseidx + 9 * kerplanecount].x) * X[29].x) + ((~Kodiag[ker_baseidx + 9 * kerplanecount].x) * X[29].y) - ((~Kodiag[ker_baseidx + 9 * kerplanecount].y) * X[29].z);
			F[29].y = ((~Kodiag[ker_baseidx + 9 * kerplanecount].x) * X[29].x) + ((~Kdiag[ker_baseidx + 9 * kerplanecount].y) * X[29].y) - ((~Kodiag[ker_baseidx + 9 * kerplanecount].z) * X[29].z);
			F[29].z = -1 * ((~Kodiag[ker_baseidx + 9 * kerplanecount].y) * X[29].x) - ((~Kodiag[ker_baseidx + 9 * kerplanecount].z) * X[29].y) + ((~Kdiag[ker_baseidx + 9 * kerplanecount].z) * X[29].z);

			F[3].x = ((~Kdiag[ker_baseidx + 8 * kerplanecount].x) * X[3].x) + ((~Kodiag[ker_baseidx + 8 * kerplanecount].x) * X[3].y) - ((~Kodiag[ker_baseidx + 8 * kerplanecount].y) * X[3].z);
			F[3].y = ((~Kodiag[ker_baseidx + 8 * kerplanecount].x) * X[3].x) + ((~Kdiag[ker_baseidx + 8 * kerplanecount].y) * X[3].y) - ((~Kodiag[ker_baseidx + 8 * kerplanecount].z) * X[3].z);
			F[3].z = -1 * ((~Kodiag[ker_baseidx + 8 * kerplanecount].y) * X[3].x) - ((~Kodiag[ker_baseidx + 8 * kerplanecount].z) * X[3].y) + ((~Kdiag[ker_baseidx + 8 * kerplanecount].z) * X[3].z);

			F[19].x = ((~Kdiag[ker_baseidx + 7 * kerplanecount].x) * X[19].x) + ((~Kodiag[ker_baseidx + 7 * kerplanecount].x) * X[19].y) - ((~Kodiag[ker_baseidx + 7 * kerplanecount].y) * X[19].z);
			F[19].y = ((~Kodiag[ker_baseidx + 7 * kerplanecount].x) * X[19].x) + ((~Kdiag[ker_baseidx + 7 * kerplanecount].y) * X[19].y) - ((~Kodiag[ker_baseidx + 7 * kerplanecount].z) * X[19].z);
			F[19].z = -1 * ((~Kodiag[ker_baseidx + 7 * kerplanecount].y) * X[19].x) - ((~Kodiag[ker_baseidx + 7 * kerplanecount].z) * X[19].y) + ((~Kdiag[ker_baseidx + 7 * kerplanecount].z) * X[19].z);

			F[11].x = ((~Kdiag[ker_baseidx + 6 * kerplanecount].x) * X[11].x) + ((~Kodiag[ker_baseidx + 6 * kerplanecount].x) * X[11].y) - ((~Kodiag[ker_baseidx + 6 * kerplanecount].y) * X[11].z);
			F[11].y = ((~Kodiag[ker_baseidx + 6 * kerplanecount].x) * X[11].x) + ((~Kdiag[ker_baseidx + 6 * kerplanecount].y) * X[11].y) - ((~Kodiag[ker_baseidx + 6 * kerplanecount].z) * X[11].z);
			F[11].z = -1 * ((~Kodiag[ker_baseidx + 6 * kerplanecount].y) * X[11].x) - ((~Kodiag[ker_baseidx + 6 * kerplanecount].z) * X[11].y) + ((~Kdiag[ker_baseidx + 6 * kerplanecount].z) * X[11].z);

			F[27].x = ((~Kdiag[ker_baseidx + 5 * kerplanecount].x) * X[27].x) + ((~Kodiag[ker_baseidx + 5 * kerplanecount].x) * X[27].y) - ((~Kodiag[ker_baseidx + 5 * kerplanecount].y) * X[27].z);
			F[27].y = ((~Kodiag[ker_baseidx + 5 * kerplanecount].x) * X[27].x) + ((~Kdiag[ker_baseidx + 5 * kerplanecount].y) * X[27].y) - ((~Kodiag[ker_baseidx + 5 * kerplanecount].z) * X[27].z);
			F[27].z = -1 * ((~Kodiag[ker_baseidx + 5 * kerplanecount].y) * X[27].x) - ((~Kodiag[ker_baseidx + 5 * kerplanecount].z) * X[27].y) + ((~Kdiag[ker_baseidx + 5 * kerplanecount].z) * X[27].z);

			F[7].x = ((~Kdiag[ker_baseidx + 4 * kerplanecount].x) * X[7].x) + ((~Kodiag[ker_baseidx + 4 * kerplanecount].x) * X[7].y) - ((~Kodiag[ker_baseidx + 4 * kerplanecount].y) * X[7].z);
			F[7].y = ((~Kodiag[ker_baseidx + 4 * kerplanecount].x) * X[7].x) + ((~Kdiag[ker_baseidx + 4 * kerplanecount].y) * X[7].y) - ((~Kodiag[ker_baseidx + 4 * kerplanecount].z) * X[7].z);
			F[7].z = -1 * ((~Kodiag[ker_baseidx + 4 * kerplanecount].y) * X[7].x) - ((~Kodiag[ker_baseidx + 4 * kerplanecount].z) * X[7].y) + ((~Kdiag[ker_baseidx + 4 * kerplanecount].z) * X[7].z);

			F[23].x = ((~Kdiag[ker_baseidx + 3 * kerplanecount].x) * X[23].x) + ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X[23].y) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X[23].z);
			F[23].y = ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X[23].x) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].y) * X[23].y) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X[23].z);
			F[23].z = -1 * ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X[23].x) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X[23].y) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].z) * X[23].z);

			F[15].x = ((~Kdiag[ker_baseidx + 2 * kerplanecount].x) * X[15].x) + ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X[15].y) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X[15].z);
			F[15].y = ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X[15].x) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].y) * X[15].y) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X[15].z);
			F[15].z = -1 * ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X[15].x) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X[15].y) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].z) * X[15].z);

			F[31].x = ((~Kdiag[ker_baseidx + 1 * kerplanecount].x) * X[31].x) + ((~Kodiag[ker_baseidx + 1 * kerplanecount].x) * X[31].y) - ((~Kodiag[ker_baseidx + 1 * kerplanecount].y) * X[31].z);
			F[31].y = ((~Kodiag[ker_baseidx + 1 * kerplanecount].x) * X[31].x) + ((~Kdiag[ker_baseidx + 1 * kerplanecount].y) * X[31].y) - ((~Kodiag[ker_baseidx + 1 * kerplanecount].z) * X[31].z);
			F[31].z = -1 * ((~Kodiag[ker_baseidx + 1 * kerplanecount].y) * X[31].x) - ((~Kodiag[ker_baseidx + 1 * kerplanecount].z) * X[31].y) + ((~Kdiag[ker_baseidx + 1 * kerplanecount].z) * X[31].z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			//lower z, upper y
			F[0].x = (Kdiag[ker_baseidx].x * X[0].x) - (Kodiag[ker_baseidx].x * X[0].y) + (Kodiag[ker_baseidx].y * X[0].z);
			F[0].y = -1 * (Kodiag[ker_baseidx].x * X[0].x) + (Kdiag[ker_baseidx].y * X[0].y) - (Kodiag[ker_baseidx].z * X[0].z);
			F[0].z = (Kodiag[ker_baseidx].y * X[0].x) - (Kodiag[ker_baseidx].z * X[0].y) + (Kdiag[ker_baseidx].z * X[0].z);

			F[16].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) - (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].z);
			F[16].y = -1 * (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[16].y) - (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);
			F[16].z = (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].x) - (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);

			F[8].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].z);
			F[8].y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[8].y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);
			F[8].z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);

			F[24].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) - (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].z);
			F[24].y = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[24].y) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);
			F[24].z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].x) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);

			F[4].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) - (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].z);
			F[4].y = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[4].y) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);
			F[4].z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].x) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);

			F[20].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) - (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].z);
			F[20].y = -1 * (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[20].y) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);
			F[20].z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].x) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);

			F[12].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) - (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].z);
			F[12].y = -1 * (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[12].y) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);
			F[12].z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].x) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);

			F[28].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) - (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].z);
			F[28].y = -1 * (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[28].y) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);
			F[28].z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].x) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);

			F[2].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) - (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].z);
			F[2].y = -1 * (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[2].y) - (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);
			F[2].z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].x) - (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);

			F[18].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) - (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].z);
			F[18].y = -1 * (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[18].y) - (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);
			F[18].z = (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].x) - (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);

			F[10].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) - (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].z);
			F[10].y = -1 * (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[10].y) - (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);
			F[10].z = (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].x) - (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);

			F[26].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) - (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].z);
			F[26].y = -1 * (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[26].y) - (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);
			F[26].z = (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].x) - (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);

			F[6].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) - (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].z);
			F[6].y = -1 * (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[6].y) - (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);
			F[6].z = (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].x) - (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);

			F[22].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) - (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].z);
			F[22].y = -1 * (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[22].y) - (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);
			F[22].z = (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].x) - (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);

			F[14].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) - (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].z);
			F[14].y = -1 * (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[14].y) - (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);
			F[14].z = (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].x) - (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);

			F[30].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) - (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].z);
			F[30].y = -1 * (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[30].y) - (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);
			F[30].z = (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].x) - (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);

			F[1].x = (Kdiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) - (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].z);
			F[1].y = -1 * (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kdiag[ker_baseidx + 16 * kerplanecount].y * X[1].y) - (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);
			F[1].z = (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].x) - (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].y) + (Kdiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);

			//upper z, upper y
			F[17].x = ((~Kdiag[ker_baseidx + 15 * kerplanecount].x) * X[17].x) - ((~Kodiag[ker_baseidx + 15 * kerplanecount].x) * X[17].y) - ((~Kodiag[ker_baseidx + 15 * kerplanecount].y) * X[17].z);
			F[17].y = -1 * ((~Kodiag[ker_baseidx + 15 * kerplanecount].x) * X[17].x) + ((~Kdiag[ker_baseidx + 15 * kerplanecount].y) * X[17].y) + ((~Kodiag[ker_baseidx + 15 * kerplanecount].z) * X[17].z);
			F[17].z = -1 * ((~Kodiag[ker_baseidx + 15 * kerplanecount].y) * X[17].x) + ((~Kodiag[ker_baseidx + 15 * kerplanecount].z) * X[17].y) + ((~Kdiag[ker_baseidx + 15 * kerplanecount].z) * X[17].z);

			F[9].x = ((~Kdiag[ker_baseidx + 14 * kerplanecount].x) * X[9].x) - ((~Kodiag[ker_baseidx + 14 * kerplanecount].x) * X[9].y) - ((~Kodiag[ker_baseidx + 14 * kerplanecount].y) * X[9].z);
			F[9].y = -1 * ((~Kodiag[ker_baseidx + 14 * kerplanecount].x) * X[9].x) + ((~Kdiag[ker_baseidx + 14 * kerplanecount].y) * X[9].y) + ((~Kodiag[ker_baseidx + 14 * kerplanecount].z) * X[9].z);
			F[9].z = -1 * ((~Kodiag[ker_baseidx + 14 * kerplanecount].y) * X[9].x) + ((~Kodiag[ker_baseidx + 14 * kerplanecount].z) * X[9].y) + ((~Kdiag[ker_baseidx + 14 * kerplanecount].z) * X[9].z);

			F[25].x = ((~Kdiag[ker_baseidx + 13 * kerplanecount].x) * X[25].x) - ((~Kodiag[ker_baseidx + 13 * kerplanecount].x) * X[25].y) - ((~Kodiag[ker_baseidx + 13 * kerplanecount].y) * X[25].z);
			F[25].y = -1 * ((~Kodiag[ker_baseidx + 13 * kerplanecount].x) * X[25].x) + ((~Kdiag[ker_baseidx + 13 * kerplanecount].y) * X[25].y) + ((~Kodiag[ker_baseidx + 13 * kerplanecount].z) * X[25].z);
			F[25].z = -1 * ((~Kodiag[ker_baseidx + 13 * kerplanecount].y) * X[25].x) + ((~Kodiag[ker_baseidx + 13 * kerplanecount].z) * X[25].y) + ((~Kdiag[ker_baseidx + 13 * kerplanecount].z) * X[25].z);

			F[5].x = ((~Kdiag[ker_baseidx + 12 * kerplanecount].x) * X[5].x) - ((~Kodiag[ker_baseidx + 12 * kerplanecount].x) * X[5].y) - ((~Kodiag[ker_baseidx + 12 * kerplanecount].y) * X[5].z);
			F[5].y = -1 * ((~Kodiag[ker_baseidx + 12 * kerplanecount].x) * X[5].x) + ((~Kdiag[ker_baseidx + 12 * kerplanecount].y) * X[5].y) + ((~Kodiag[ker_baseidx + 12 * kerplanecount].z) * X[5].z);
			F[5].z = -1 * ((~Kodiag[ker_baseidx + 12 * kerplanecount].y) * X[5].x) + ((~Kodiag[ker_baseidx + 12 * kerplanecount].z) * X[5].y) + ((~Kdiag[ker_baseidx + 12 * kerplanecount].z) * X[5].z);

			F[21].x = ((~Kdiag[ker_baseidx + 11 * kerplanecount].x) * X[21].x) - ((~Kodiag[ker_baseidx + 11 * kerplanecount].x) * X[21].y) - ((~Kodiag[ker_baseidx + 11 * kerplanecount].y) * X[21].z);
			F[21].y = -1 * ((~Kodiag[ker_baseidx + 11 * kerplanecount].x) * X[21].x) + ((~Kdiag[ker_baseidx + 11 * kerplanecount].y) * X[21].y) + ((~Kodiag[ker_baseidx + 11 * kerplanecount].z) * X[21].z);
			F[21].z = -1 * ((~Kodiag[ker_baseidx + 11 * kerplanecount].y) * X[21].x) + ((~Kodiag[ker_baseidx + 11 * kerplanecount].z) * X[21].y) + ((~Kdiag[ker_baseidx + 11 * kerplanecount].z) * X[21].z);

			F[13].x = ((~Kdiag[ker_baseidx + 10 * kerplanecount].x) * X[13].x) - ((~Kodiag[ker_baseidx + 10 * kerplanecount].x) * X[13].y) - ((~Kodiag[ker_baseidx + 10 * kerplanecount].y) * X[13].z);
			F[13].y = -1 * ((~Kodiag[ker_baseidx + 10 * kerplanecount].x) * X[13].x) + ((~Kdiag[ker_baseidx + 10 * kerplanecount].y) * X[13].y) + ((~Kodiag[ker_baseidx + 10 * kerplanecount].z) * X[13].z);
			F[13].z = -1 * ((~Kodiag[ker_baseidx + 10 * kerplanecount].y) * X[13].x) + ((~Kodiag[ker_baseidx + 10 * kerplanecount].z) * X[13].y) + ((~Kdiag[ker_baseidx + 10 * kerplanecount].z) * X[13].z);

			F[29].x = ((~Kdiag[ker_baseidx + 9 * kerplanecount].x) * X[29].x) - ((~Kodiag[ker_baseidx + 9 * kerplanecount].x) * X[29].y) - ((~Kodiag[ker_baseidx + 9 * kerplanecount].y) * X[29].z);
			F[29].y = -1 * ((~Kodiag[ker_baseidx + 9 * kerplanecount].x) * X[29].x) + ((~Kdiag[ker_baseidx + 9 * kerplanecount].y) * X[29].y) + ((~Kodiag[ker_baseidx + 9 * kerplanecount].z) * X[29].z);
			F[29].z = -1 * ((~Kodiag[ker_baseidx + 9 * kerplanecount].y) * X[29].x) + ((~Kodiag[ker_baseidx + 9 * kerplanecount].z) * X[29].y) + ((~Kdiag[ker_baseidx + 9 * kerplanecount].z) * X[29].z);

			F[3].x = ((~Kdiag[ker_baseidx + 8 * kerplanecount].x) * X[3].x) - ((~Kodiag[ker_baseidx + 8 * kerplanecount].x) * X[3].y) - ((~Kodiag[ker_baseidx + 8 * kerplanecount].y) * X[3].z);
			F[3].y = -1 * ((~Kodiag[ker_baseidx + 8 * kerplanecount].x) * X[3].x) + ((~Kdiag[ker_baseidx + 8 * kerplanecount].y) * X[3].y) + ((~Kodiag[ker_baseidx + 8 * kerplanecount].z) * X[3].z);
			F[3].z = -1 * ((~Kodiag[ker_baseidx + 8 * kerplanecount].y) * X[3].x) + ((~Kodiag[ker_baseidx + 8 * kerplanecount].z) * X[3].y) + ((~Kdiag[ker_baseidx + 8 * kerplanecount].z) * X[3].z);

			F[19].x = ((~Kdiag[ker_baseidx + 7 * kerplanecount].x) * X[19].x) - ((~Kodiag[ker_baseidx + 7 * kerplanecount].x) * X[19].y) - ((~Kodiag[ker_baseidx + 7 * kerplanecount].y) * X[19].z);
			F[19].y = -1 * ((~Kodiag[ker_baseidx + 7 * kerplanecount].x) * X[19].x) + ((~Kdiag[ker_baseidx + 7 * kerplanecount].y) * X[19].y) + ((~Kodiag[ker_baseidx + 7 * kerplanecount].z) * X[19].z);
			F[19].z = -1 * ((~Kodiag[ker_baseidx + 7 * kerplanecount].y) * X[19].x) + ((~Kodiag[ker_baseidx + 7 * kerplanecount].z) * X[19].y) + ((~Kdiag[ker_baseidx + 7 * kerplanecount].z) * X[19].z);

			F[11].x = ((~Kdiag[ker_baseidx + 6 * kerplanecount].x) * X[11].x) - ((~Kodiag[ker_baseidx + 6 * kerplanecount].x) * X[11].y) - ((~Kodiag[ker_baseidx + 6 * kerplanecount].y) * X[11].z);
			F[11].y = -1 * ((~Kodiag[ker_baseidx + 6 * kerplanecount].x) * X[11].x) + ((~Kdiag[ker_baseidx + 6 * kerplanecount].y) * X[11].y) + ((~Kodiag[ker_baseidx + 6 * kerplanecount].z) * X[11].z);
			F[11].z = -1 * ((~Kodiag[ker_baseidx + 6 * kerplanecount].y) * X[11].x) + ((~Kodiag[ker_baseidx + 6 * kerplanecount].z) * X[11].y) + ((~Kdiag[ker_baseidx + 6 * kerplanecount].z) * X[11].z);

			F[27].x = ((~Kdiag[ker_baseidx + 5 * kerplanecount].x) * X[27].x) - ((~Kodiag[ker_baseidx + 5 * kerplanecount].x) * X[27].y) - ((~Kodiag[ker_baseidx + 5 * kerplanecount].y) * X[27].z);
			F[27].y = -1 * ((~Kodiag[ker_baseidx + 5 * kerplanecount].x) * X[27].x) + ((~Kdiag[ker_baseidx + 5 * kerplanecount].y) * X[27].y) + ((~Kodiag[ker_baseidx + 5 * kerplanecount].z) * X[27].z);
			F[27].z = -1 * ((~Kodiag[ker_baseidx + 5 * kerplanecount].y) * X[27].x) + ((~Kodiag[ker_baseidx + 5 * kerplanecount].z) * X[27].y) + ((~Kdiag[ker_baseidx + 5 * kerplanecount].z) * X[27].z);

			F[7].x = ((~Kdiag[ker_baseidx + 4 * kerplanecount].x) * X[7].x) - ((~Kodiag[ker_baseidx + 4 * kerplanecount].x) * X[7].y) - ((~Kodiag[ker_baseidx + 4 * kerplanecount].y) * X[7].z);
			F[7].y = -1 * ((~Kodiag[ker_baseidx + 4 * kerplanecount].x) * X[7].x) + ((~Kdiag[ker_baseidx + 4 * kerplanecount].y) * X[7].y) + ((~Kodiag[ker_baseidx + 4 * kerplanecount].z) * X[7].z);
			F[7].z = -1 * ((~Kodiag[ker_baseidx + 4 * kerplanecount].y) * X[7].x) + ((~Kodiag[ker_baseidx + 4 * kerplanecount].z) * X[7].y) + ((~Kdiag[ker_baseidx + 4 * kerplanecount].z) * X[7].z);

			F[23].x = ((~Kdiag[ker_baseidx + 3 * kerplanecount].x) * X[23].x) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X[23].y) - ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X[23].z);
			F[23].y = -1 * ((~Kodiag[ker_baseidx + 3 * kerplanecount].x) * X[23].x) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].y) * X[23].y) + ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X[23].z);
			F[23].z = -1 * ((~Kodiag[ker_baseidx + 3 * kerplanecount].y) * X[23].x) + ((~Kodiag[ker_baseidx + 3 * kerplanecount].z) * X[23].y) + ((~Kdiag[ker_baseidx + 3 * kerplanecount].z) * X[23].z);

			F[15].x = ((~Kdiag[ker_baseidx + 2 * kerplanecount].x) * X[15].x) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X[15].y) - ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X[15].z);
			F[15].y = -1 * ((~Kodiag[ker_baseidx + 2 * kerplanecount].x) * X[15].x) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].y) * X[15].y) + ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X[15].z);
			F[15].z = -1 * ((~Kodiag[ker_baseidx + 2 * kerplanecount].y) * X[15].x) + ((~Kodiag[ker_baseidx + 2 * kerplanecount].z) * X[15].y) + ((~Kdiag[ker_baseidx + 2 * kerplanecount].z) * X[15].z);

			F[31].x = ((~Kdiag[ker_baseidx + 1 * kerplanecount].x) * X[31].x) - ((~Kodiag[ker_baseidx + 1 * kerplanecount].x) * X[31].y) - ((~Kodiag[ker_baseidx + 1 * kerplanecount].y) * X[31].z);
			F[31].y = -1 * ((~Kodiag[ker_baseidx + 1 * kerplanecount].x) * X[31].x) + ((~Kdiag[ker_baseidx + 1 * kerplanecount].y) * X[31].y) + ((~Kodiag[ker_baseidx + 1 * kerplanecount].z) * X[31].z);
			F[31].z = -1 * ((~Kodiag[ker_baseidx + 1 * kerplanecount].y) * X[31].x) + ((~Kodiag[ker_baseidx + 1 * kerplanecount].z) * X[31].y) + ((~Kdiag[ker_baseidx + 1 * kerplanecount].z) * X[31].z);
		}

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F[0].x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F[0].y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F[0].z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F[1].x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F[1].y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F[1].z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F[2].x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F[2].y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F[2].z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F[3].x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F[3].y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F[3].z;

		cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F[4].x;
		cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F[4].y;
		cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F[4].z;

		cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F[5].x;
		cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F[5].y;
		cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F[5].z;

		cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F[6].x;
		cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F[6].y;
		cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F[6].z;

		cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F[7].x;
		cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F[7].y;
		cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F[7].z;

		cuSx_out[idx + 8 * planecount] = (cuReIm)cuSx_out[idx + 8 * planecount] + F[8].x;
		cuSy_out[idx + 8 * planecount] = (cuReIm)cuSy_out[idx + 8 * planecount] + F[8].y;
		cuSz_out[idx + 8 * planecount] = (cuReIm)cuSz_out[idx + 8 * planecount] + F[8].z;

		cuSx_out[idx + 9 * planecount] = (cuReIm)cuSx_out[idx + 9 * planecount] + F[9].x;
		cuSy_out[idx + 9 * planecount] = (cuReIm)cuSy_out[idx + 9 * planecount] + F[9].y;
		cuSz_out[idx + 9 * planecount] = (cuReIm)cuSz_out[idx + 9 * planecount] + F[9].z;

		cuSx_out[idx + 10 * planecount] = (cuReIm)cuSx_out[idx + 10 * planecount] + F[10].x;
		cuSy_out[idx + 10 * planecount] = (cuReIm)cuSy_out[idx + 10 * planecount] + F[10].y;
		cuSz_out[idx + 10 * planecount] = (cuReIm)cuSz_out[idx + 10 * planecount] + F[10].z;

		cuSx_out[idx + 11 * planecount] = (cuReIm)cuSx_out[idx + 11 * planecount] + F[11].x;
		cuSy_out[idx + 11 * planecount] = (cuReIm)cuSy_out[idx + 11 * planecount] + F[11].y;
		cuSz_out[idx + 11 * planecount] = (cuReIm)cuSz_out[idx + 11 * planecount] + F[11].z;

		cuSx_out[idx + 12 * planecount] = (cuReIm)cuSx_out[idx + 12 * planecount] + F[12].x;
		cuSy_out[idx + 12 * planecount] = (cuReIm)cuSy_out[idx + 12 * planecount] + F[12].y;
		cuSz_out[idx + 12 * planecount] = (cuReIm)cuSz_out[idx + 12 * planecount] + F[12].z;

		cuSx_out[idx + 13 * planecount] = (cuReIm)cuSx_out[idx + 13 * planecount] + F[13].x;
		cuSy_out[idx + 13 * planecount] = (cuReIm)cuSy_out[idx + 13 * planecount] + F[13].y;
		cuSz_out[idx + 13 * planecount] = (cuReIm)cuSz_out[idx + 13 * planecount] + F[13].z;

		cuSx_out[idx + 14 * planecount] = (cuReIm)cuSx_out[idx + 14 * planecount] + F[14].x;
		cuSy_out[idx + 14 * planecount] = (cuReIm)cuSy_out[idx + 14 * planecount] + F[14].y;
		cuSz_out[idx + 14 * planecount] = (cuReIm)cuSz_out[idx + 14 * planecount] + F[14].z;

		cuSx_out[idx + 15 * planecount] = (cuReIm)cuSx_out[idx + 15 * planecount] + F[15].x;
		cuSy_out[idx + 15 * planecount] = (cuReIm)cuSy_out[idx + 15 * planecount] + F[15].y;
		cuSz_out[idx + 15 * planecount] = (cuReIm)cuSz_out[idx + 15 * planecount] + F[15].z;

		cuSx_out[idx + 16 * planecount] = (cuReIm)cuSx_out[idx + 16 * planecount] + F[16].x;
		cuSy_out[idx + 16 * planecount] = (cuReIm)cuSy_out[idx + 16 * planecount] + F[16].y;
		cuSz_out[idx + 16 * planecount] = (cuReIm)cuSz_out[idx + 16 * planecount] + F[16].z;

		cuSx_out[idx + 17 * planecount] = (cuReIm)cuSx_out[idx + 17 * planecount] + F[17].x;
		cuSy_out[idx + 17 * planecount] = (cuReIm)cuSy_out[idx + 17 * planecount] + F[17].y;
		cuSz_out[idx + 17 * planecount] = (cuReIm)cuSz_out[idx + 17 * planecount] + F[17].z;

		cuSx_out[idx + 18 * planecount] = (cuReIm)cuSx_out[idx + 18 * planecount] + F[18].x;
		cuSy_out[idx + 18 * planecount] = (cuReIm)cuSy_out[idx + 18 * planecount] + F[18].y;
		cuSz_out[idx + 18 * planecount] = (cuReIm)cuSz_out[idx + 18 * planecount] + F[18].z;

		cuSx_out[idx + 19 * planecount] = (cuReIm)cuSx_out[idx + 19 * planecount] + F[19].x;
		cuSy_out[idx + 19 * planecount] = (cuReIm)cuSy_out[idx + 19 * planecount] + F[19].y;
		cuSz_out[idx + 19 * planecount] = (cuReIm)cuSz_out[idx + 19 * planecount] + F[19].z;

		cuSx_out[idx + 20 * planecount] = (cuReIm)cuSx_out[idx + 20 * planecount] + F[20].x;
		cuSy_out[idx + 20 * planecount] = (cuReIm)cuSy_out[idx + 20 * planecount] + F[20].y;
		cuSz_out[idx + 20 * planecount] = (cuReIm)cuSz_out[idx + 20 * planecount] + F[20].z;

		cuSx_out[idx + 21 * planecount] = (cuReIm)cuSx_out[idx + 21 * planecount] + F[21].x;
		cuSy_out[idx + 21 * planecount] = (cuReIm)cuSy_out[idx + 21 * planecount] + F[21].y;
		cuSz_out[idx + 21 * planecount] = (cuReIm)cuSz_out[idx + 21 * planecount] + F[21].z;

		cuSx_out[idx + 22 * planecount] = (cuReIm)cuSx_out[idx + 22 * planecount] + F[22].x;
		cuSy_out[idx + 22 * planecount] = (cuReIm)cuSy_out[idx + 22 * planecount] + F[22].y;
		cuSz_out[idx + 22 * planecount] = (cuReIm)cuSz_out[idx + 22 * planecount] + F[22].z;

		cuSx_out[idx + 23 * planecount] = (cuReIm)cuSx_out[idx + 23 * planecount] + F[23].x;
		cuSy_out[idx + 23 * planecount] = (cuReIm)cuSy_out[idx + 23 * planecount] + F[23].y;
		cuSz_out[idx + 23 * planecount] = (cuReIm)cuSz_out[idx + 23 * planecount] + F[23].z;

		cuSx_out[idx + 24 * planecount] = (cuReIm)cuSx_out[idx + 24 * planecount] + F[24].x;
		cuSy_out[idx + 24 * planecount] = (cuReIm)cuSy_out[idx + 24 * planecount] + F[24].y;
		cuSz_out[idx + 24 * planecount] = (cuReIm)cuSz_out[idx + 24 * planecount] + F[24].z;

		cuSx_out[idx + 25 * planecount] = (cuReIm)cuSx_out[idx + 25 * planecount] + F[25].x;
		cuSy_out[idx + 25 * planecount] = (cuReIm)cuSy_out[idx + 25 * planecount] + F[25].y;
		cuSz_out[idx + 25 * planecount] = (cuReIm)cuSz_out[idx + 25 * planecount] + F[25].z;

		cuSx_out[idx + 26 * planecount] = (cuReIm)cuSx_out[idx + 26 * planecount] + F[26].x;
		cuSy_out[idx + 26 * planecount] = (cuReIm)cuSy_out[idx + 26 * planecount] + F[26].y;
		cuSz_out[idx + 26 * planecount] = (cuReIm)cuSz_out[idx + 26 * planecount] + F[26].z;

		cuSx_out[idx + 27 * planecount] = (cuReIm)cuSx_out[idx + 27 * planecount] + F[27].x;
		cuSy_out[idx + 27 * planecount] = (cuReIm)cuSy_out[idx + 27 * planecount] + F[27].y;
		cuSz_out[idx + 27 * planecount] = (cuReIm)cuSz_out[idx + 27 * planecount] + F[27].z;

		cuSx_out[idx + 28 * planecount] = (cuReIm)cuSx_out[idx + 28 * planecount] + F[28].x;
		cuSy_out[idx + 28 * planecount] = (cuReIm)cuSy_out[idx + 28 * planecount] + F[28].y;
		cuSz_out[idx + 28 * planecount] = (cuReIm)cuSz_out[idx + 28 * planecount] + F[28].z;

		cuSx_out[idx + 29 * planecount] = (cuReIm)cuSx_out[idx + 29 * planecount] + F[29].x;
		cuSy_out[idx + 29 * planecount] = (cuReIm)cuSy_out[idx + 29 * planecount] + F[29].y;
		cuSz_out[idx + 29 * planecount] = (cuReIm)cuSz_out[idx + 29 * planecount] + F[29].z;

		cuSx_out[idx + 30 * planecount] = (cuReIm)cuSx_out[idx + 30 * planecount] + F[30].x;
		cuSy_out[idx + 30 * planecount] = (cuReIm)cuSy_out[idx + 30 * planecount] + F[30].y;
		cuSz_out[idx + 30 * planecount] = (cuReIm)cuSz_out[idx + 30 * planecount] + F[30].z;

		cuSx_out[idx + 31 * planecount] = (cuReIm)cuSx_out[idx + 31 * planecount] + F[31].x;
		cuSy_out[idx + 31 * planecount] = (cuReIm)cuSy_out[idx + 31 * planecount] + F[31].y;
		cuSz_out[idx + 31 * planecount] = (cuReIm)cuSz_out[idx + 31 * planecount] + F[31].z;

#undef a
#undef b
#undef c
#undef d
#undef e
#undef f
#undef g
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// Q2D - X SHIFTED

//N = (N.x/2 + 1, N.y, 4)
//xy is transposed
//Complex kernels : this is a x shifted demag version using kernel symmetries, which accumulates in output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_4_transpose_xy_xshifted(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	//x shifted for 3D : can use kernels of reduced dimensions but must be complex
	//
	//Kxx, Kyy, Kzz : symmetrical (+) in y and z directions.
	//
	//Kxy : y - inv. symmetric (-), z - symmetrical  (+)
	//Kxz : y - symmetrical  (+), z - inv. symmetric (-)
	//Kyz : y - inv. symmetric (-), z - inv. symmetric (-)

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 4, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

		//the z-axis points (the others are zero)
		cuReIm3 a = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 b = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);

		//forward z-axis fft
		//NOTE: cuda fft uses -i for the forward fft and +i for the inverse fft.
		//The kernels are purely real so you would get the same result by taking +i for the forward and -i for the inverse, but better to keep it consistent : use the cuda fft convention here.
		cuReIm3 X0 = a + b;
		cuReIm3 X1 = a - !b;
		cuReIm3 X2 = a - b;
		cuReIm3 X3 = a + !b;

		//kernel multiplication
		cuReIm3 F0, F1, F2, F3;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			//lower z, lower y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F1.x = (Kdiag[ker_baseidx + kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + kerplanecount].y * X1.z);
			F1.y = (Kodiag[ker_baseidx + kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + kerplanecount].z * X1.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			//upper z, lower y
			F3.x = (Kdiag[ker_baseidx + kerplanecount].x * X3.x) + (Kodiag[ker_baseidx + kerplanecount].x * X3.y) - (Kodiag[ker_baseidx + kerplanecount].y * X3.z);
			F3.y = (Kodiag[ker_baseidx + kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + kerplanecount].y * X3.y) - (Kodiag[ker_baseidx + kerplanecount].z * X3.z);
			F3.z = -1 * (Kodiag[ker_baseidx + kerplanecount].y * X3.x) - (Kodiag[ker_baseidx + kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + kerplanecount].z * X3.z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			//lower z, upper y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) - (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = -1 * (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) - (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) - (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F1.x = (Kdiag[ker_baseidx + kerplanecount].x * X1.x) - (Kodiag[ker_baseidx + kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + kerplanecount].y * X1.z);
			F1.y = -1 * (Kodiag[ker_baseidx + kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + kerplanecount].y * X1.y) - (Kodiag[ker_baseidx + kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + kerplanecount].y * X1.x) - (Kodiag[ker_baseidx + kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + kerplanecount].z * X1.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			//upper z, upper y
			F3.x = (Kdiag[ker_baseidx + kerplanecount].x * X3.x) - (Kodiag[ker_baseidx + kerplanecount].x * X3.y) - (Kodiag[ker_baseidx + kerplanecount].y * X3.z);
			F3.y = -1 * (Kodiag[ker_baseidx + kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + kerplanecount].y * X3.y) + (Kodiag[ker_baseidx + kerplanecount].z * X3.z);
			F3.z = -1 * (Kodiag[ker_baseidx + kerplanecount].y * X3.x) + (Kodiag[ker_baseidx + kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + kerplanecount].z * X3.z);
		}

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;
	}
}

//N = (N.x/2 + 1, N.y, 8)
//xy is transposed
//Complex kernels : this is a x shifted demag version using kernel symmetries, which accumulates in output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_8_transpose_xy_xshifted(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	//x shifted for 3D : can use kernels of reduced dimensions but must be complex
	//
	//Kxx, Kyy, Kzz : symmetrical (+) in y and z directions.
	//
	//Kxy : y - inv. symmetric (-), z - symmetrical  (+)
	//Kxz : y - symmetrical  (+), z - inv. symmetric (-)
	//Kyz : y - inv. symmetric (-), z - inv. symmetric (-)

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 8, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

#define a (cuBReal)0.7071067811865

		//the z-axis points (the others are zero)
		cuReIm3 x0 = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 x1 = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);
		cuReIm3 x2 = cuReIm3(cuSx_in[idx + 2 * planecount], cuSy_in[idx + 2 * planecount], cuSz_in[idx + 2 * planecount]);
		cuReIm3 x3 = cuReIm3(cuSx_in[idx + 3 * planecount], cuSy_in[idx + 3 * planecount], cuSz_in[idx + 3 * planecount]);

		//Radix-4 step
		cuReIm3 X0 = x0 + x2;
		cuReIm3 X2 = x0 - x2;
		cuReIm3 X4 = x0 - !x2;
		cuReIm3 X6 = x0 + !x2;

		cuReIm3 X1 = x1 + x3;
		cuReIm3 X3 = !(x3 - x1);
		cuReIm3 X5 = (x1 - !x3) * cuReIm(a, -a);
		cuReIm3 X7 = (x1 + !x3) * cuReIm(-a, -a);

		//Radix-2 step
		cuReIm3 temp = X0 - X1;
		X0 = X0 + X1;
		X1 = temp;

		temp = X2 - X3;
		X2 = X2 + X3;
		X3 = temp;

		temp = X4 - X5;
		X4 = X4 + X5;
		X5 = temp;

		temp = X6 - X7;
		X6 = X6 + X7;
		X7 = temp;

		//data set in shuffled order:
		//X0, X4, X2, X6, X1, X5, X3, X7

		cuReIm3 F0, F1, F2, F3, F4, F5, F6, F7;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			//lower z, lower y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F4.x = (Kdiag[ker_baseidx + kerplanecount].x * X4.x) + (Kodiag[ker_baseidx + kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + kerplanecount].y * X4.z);
			F4.y = (Kodiag[ker_baseidx + kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + kerplanecount].y * X4.y) + (Kodiag[ker_baseidx + kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + kerplanecount].y * X4.x) + (Kodiag[ker_baseidx + kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + kerplanecount].z * X4.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			F6.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.z);
			F6.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X6.z);

			F1.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.z);
			F1.y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X1.z);

			//upper z, lower y
			F5.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X5.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X5.y) - (Kodiag[ker_baseidx + 3 * kerplanecount].y * X5.z);
			F5.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X5.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X5.y) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X5.z);
			F5.z = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].y * X5.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X5.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X5.z);

			F3.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X3.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X3.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].y * X3.z);
			F3.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X3.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X3.z);
			F3.z = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].y * X3.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X3.z);

			F7.x = (Kdiag[ker_baseidx + kerplanecount].x * X7.x) + (Kodiag[ker_baseidx + kerplanecount].x * X7.y) - (Kodiag[ker_baseidx + kerplanecount].y * X7.z);
			F7.y = (Kodiag[ker_baseidx + kerplanecount].x * X7.x) + (Kdiag[ker_baseidx + kerplanecount].y * X7.y) - (Kodiag[ker_baseidx + kerplanecount].z * X7.z);
			F7.z = -1 * (Kodiag[ker_baseidx + kerplanecount].y * X7.x) - (Kodiag[ker_baseidx + kerplanecount].z * X7.y) + (Kdiag[ker_baseidx + kerplanecount].z * X7.z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			//lower z, upper y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) - (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = -1 * (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) - (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) - (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F4.x = (Kdiag[ker_baseidx + kerplanecount].x * X4.x) - (Kodiag[ker_baseidx + kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + kerplanecount].y * X4.z);
			F4.y = -1 * (Kodiag[ker_baseidx + kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + kerplanecount].y * X4.y) - (Kodiag[ker_baseidx + kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + kerplanecount].y * X4.x) - (Kodiag[ker_baseidx + kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + kerplanecount].z * X4.z);

			F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
			F2.y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

			F6.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X6.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.z);
			F6.y = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X6.y) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X6.z);

			F1.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X1.x) - (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.z);
			F1.y = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X1.y) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.x) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X1.z);

			//upper z, upper y
			F5.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X5.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].x * X5.y) - (Kodiag[ker_baseidx + 3 * kerplanecount].y * X5.z);
			F5.y = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].x * X5.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X5.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X5.z);
			F5.z = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].y * X5.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X5.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X5.z);

			F3.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X3.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X3.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].y * X3.z);
			F3.y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X3.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X3.z);
			F3.z = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].y * X3.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X3.z);

			F7.x = (Kdiag[ker_baseidx + kerplanecount].x * X7.x) - (Kodiag[ker_baseidx + kerplanecount].x * X7.y) - (Kodiag[ker_baseidx + kerplanecount].y * X7.z);
			F7.y = -1 * (Kodiag[ker_baseidx + kerplanecount].x * X7.x) + (Kdiag[ker_baseidx + kerplanecount].y * X7.y) + (Kodiag[ker_baseidx + kerplanecount].z * X7.z);
			F7.z = -1 * (Kodiag[ker_baseidx + kerplanecount].y * X7.x) + (Kodiag[ker_baseidx + kerplanecount].z * X7.y) + (Kdiag[ker_baseidx + kerplanecount].z * X7.z);
		}

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;

		cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F4.x;
		cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F4.y;
		cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F4.z;

		cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F5.x;
		cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F5.y;
		cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F5.z;

		cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F6.x;
		cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F6.y;
		cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F6.z;

		cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F7.x;
		cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F7.y;
		cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F7.z;

#undef a
	}
}

//N = (N.x/2 + 1, N.y, 16)
//xy is transposed
//Complex kernels : this is a x shifted demag version using kernel symmetries, which accumulates in output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_16_transpose_xy_xshifted(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	//x shifted for 3D : can use kernels of reduced dimensions but must be complex
	//
	//Kxx, Kyy, Kzz : symmetrical (+) in y and z directions.
	//
	//Kxy : y - inv. symmetric (-), z - symmetrical  (+)
	//Kxz : y - symmetrical  (+), z - inv. symmetric (-)
	//Kyz : y - inv. symmetric (-), z - inv. symmetric (-)

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 16, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

		//the z-axis points (the others are zero)
		cuReIm3 x0 = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 x1 = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);
		cuReIm3 x2 = cuReIm3(cuSx_in[idx + 2 * planecount], cuSy_in[idx + 2 * planecount], cuSz_in[idx + 2 * planecount]);
		cuReIm3 x3 = cuReIm3(cuSx_in[idx + 3 * planecount], cuSy_in[idx + 3 * planecount], cuSz_in[idx + 3 * planecount]);
		cuReIm3 x4 = cuReIm3(cuSx_in[idx + 4 * planecount], cuSy_in[idx + 4 * planecount], cuSz_in[idx + 4 * planecount]);
		cuReIm3 x5 = cuReIm3(cuSx_in[idx + 5 * planecount], cuSy_in[idx + 5 * planecount], cuSz_in[idx + 5 * planecount]);
		cuReIm3 x6 = cuReIm3(cuSx_in[idx + 6 * planecount], cuSy_in[idx + 6 * planecount], cuSz_in[idx + 6 * planecount]);
		cuReIm3 x7 = cuReIm3(cuSx_in[idx + 7 * planecount], cuSy_in[idx + 7 * planecount], cuSz_in[idx + 7 * planecount]);

#define a	(cuBReal)9.238795325113E-01
#define b	(cuBReal)3.826834323651E-01
#define c	(cuBReal)7.071067811865E-01

		//First stage
		cuReIm3 X0 = x0 + x4;
		cuReIm3 X4 = x0 - x4;
		cuReIm3 X8 = x0 - !x4;
		cuReIm3 X12 = x0 + !x4;

		cuReIm3 X1 = x1 + x5;
		cuReIm3 X5 = (x1 - x5) * cuReIm(c, -c);
		cuReIm3 X9 = (x1 - !x5) * cuReIm(a, -b);
		cuReIm3 X13 = (x1 + !x5) * cuReIm(b, -a);

		cuReIm3 X2 = x2 + x6;
		cuReIm3 X6 = !(x6 - x2);
		cuReIm3 X10 = (x2 - !x6) * cuReIm(c, -c);
		cuReIm3 X14 = (x2 + !x6) * cuReIm(-c, -c);

		cuReIm3 X3 = x3 + x7;
		cuReIm3 X7 = (x3 - x7) * cuReIm(-c, -c);
		cuReIm3 X11 = (x3 - !x7) * cuReIm(b, -a);
		cuReIm3 X15 = (x3 + !x7) * cuReIm(-a, b);

		//Second stage
		cuReIm3 t0 = X0 + X2;
		cuReIm3 t1 = X0 - X2;
		cuReIm3 t2 = X1 + X3;
		cuReIm3 t3 = !(X3 - X1);

		X0 = t0 + t2;
		X1 = t0 - t2;
		X2 = t1 + t3;
		X3 = t1 - t3;

		t0 = X4 + X6;
		t1 = X4 - X6;
		t2 = X5 + X7;
		t3 = !(X7 - X5);

		X4 = t0 + t2;
		X5 = t0 - t2;
		X6 = t1 + t3;
		X7 = t1 - t3;

		t0 = X8 + X10;
		t1 = X8 - X10;
		t2 = X9 + X11;
		t3 = !(X11 - X9);

		X8 = t0 + t2;
		X9 = t0 - t2;
		X10 = t1 + t3;
		X11 = t1 - t3;

		t0 = X12 + X14;
		t1 = X12 - X14;
		t2 = X13 + X15;
		t3 = !(X15 - X13);

		X12 = t0 + t2;
		X13 = t0 - t2;
		X14 = t1 + t3;
		X15 = t1 - t3;

		//output is shuffled now, i.e. it is ordered as:
		//X0, X8, X4, X12, X2, X10, X6, X14, X1, X9, X5, X13, X3, X11, X7, X15

		cuReIm3 F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15;

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			//lower z, lower y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F8.x = (Kdiag[ker_baseidx + kerplanecount].x * X8.x) + (Kodiag[ker_baseidx + kerplanecount].x * X8.y) + (Kodiag[ker_baseidx + kerplanecount].y * X8.z);
			F8.y = (Kodiag[ker_baseidx + kerplanecount].x * X8.x) + (Kdiag[ker_baseidx + kerplanecount].y * X8.y) + (Kodiag[ker_baseidx + kerplanecount].z * X8.z);
			F8.z = (Kodiag[ker_baseidx + kerplanecount].y * X8.x) + (Kodiag[ker_baseidx + kerplanecount].z * X8.y) + (Kdiag[ker_baseidx + kerplanecount].z * X8.z);

			F4.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.z);
			F4.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X4.z);

			F12.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.z);
			F12.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.z);
			F12.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X12.z);

			F2.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.z);
			F2.y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X2.z);

			F10.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.z);
			F10.y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.z);
			F10.z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X10.z);

			F6.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.z);
			F6.y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X6.z);

			F14.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.z);
			F14.y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.z);
			F14.z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X14.z);

			F1.x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.z);
			F1.y = (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X1.z);

			//upper z, lower y
			F9.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X9.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X9.y) - (Kodiag[ker_baseidx + 7 * kerplanecount].y * X9.z);
			F9.y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X9.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X9.y) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X9.z);
			F9.z = -1 * (Kodiag[ker_baseidx + 7 * kerplanecount].y * X9.x) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X9.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X9.z);

			F5.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X5.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X5.y) - (Kodiag[ker_baseidx + 6 * kerplanecount].y * X5.z);
			F5.y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X5.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X5.y) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X5.z);
			F5.z = -1 * (Kodiag[ker_baseidx + 6 * kerplanecount].y * X5.x) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X5.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X5.z);

			F13.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X13.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X13.y) - (Kodiag[ker_baseidx + 5 * kerplanecount].y * X13.z);
			F13.y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X13.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X13.y) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X13.z);
			F13.z = -1 * (Kodiag[ker_baseidx + 5 * kerplanecount].y * X13.x) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X13.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X13.z);

			F3.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X3.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X3.y) - (Kodiag[ker_baseidx + 4 * kerplanecount].y * X3.z);
			F3.y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X3.y) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X3.z);
			F3.z = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].y * X3.x) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X3.z);

			F11.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X11.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X11.y) - (Kodiag[ker_baseidx + 3 * kerplanecount].y * X11.z);
			F11.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X11.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X11.y) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X11.z);
			F11.z = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].y * X11.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X11.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X11.z);

			F7.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X7.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X7.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].y * X7.z);
			F7.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X7.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X7.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X7.z);
			F7.z = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].y * X7.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X7.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X7.z);

			F15.x = (Kdiag[ker_baseidx + kerplanecount].x * X15.x) + (Kodiag[ker_baseidx + kerplanecount].x * X15.y) - (Kodiag[ker_baseidx + kerplanecount].y * X15.z);
			F15.y = (Kodiag[ker_baseidx + kerplanecount].x * X15.x) + (Kdiag[ker_baseidx + kerplanecount].y * X15.y) - (Kodiag[ker_baseidx + kerplanecount].z * X15.z);
			F15.z = -1 * (Kodiag[ker_baseidx + kerplanecount].y * X15.x) - (Kodiag[ker_baseidx + kerplanecount].z * X15.y) + (Kdiag[ker_baseidx + kerplanecount].z * X15.z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			//lower z, upper y
			F0.x = (Kdiag[ker_baseidx].x * X0.x) - (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
			F0.y = -1 * (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) - (Kodiag[ker_baseidx].z * X0.z);
			F0.z = (Kodiag[ker_baseidx].y * X0.x) - (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

			F8.x = (Kdiag[ker_baseidx + kerplanecount].x * X8.x) - (Kodiag[ker_baseidx + kerplanecount].x * X8.y) + (Kodiag[ker_baseidx + kerplanecount].y * X8.z);
			F8.y = -1 * (Kodiag[ker_baseidx + kerplanecount].x * X8.x) + (Kdiag[ker_baseidx + kerplanecount].y * X8.y) - (Kodiag[ker_baseidx + kerplanecount].z * X8.z);
			F8.z = (Kodiag[ker_baseidx + kerplanecount].y * X8.x) - (Kodiag[ker_baseidx + kerplanecount].z * X8.y) + (Kdiag[ker_baseidx + kerplanecount].z * X8.z);

			F4.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X4.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.z);
			F4.y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X4.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.z);
			F4.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X4.z);

			F12.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X12.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.z);
			F12.y = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X12.y) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.z);
			F12.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X12.z);

			F2.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X2.x) - (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.z);
			F2.y = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X2.y) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.z);
			F2.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.x) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X2.z);

			F10.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X10.x) - (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.z);
			F10.y = -1 * (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X10.y) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.z);
			F10.z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.x) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X10.z);

			F6.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X6.x) - (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.z);
			F6.y = -1 * (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X6.y) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.z);
			F6.z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.x) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X6.z);

			F14.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X14.x) - (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.z);
			F14.y = -1 * (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X14.y) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.z);
			F14.z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.x) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X14.z);

			F1.x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X1.x) - (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.z);
			F1.y = -1 * (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X1.y) - (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.z);
			F1.z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.x) - (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X1.z);

			//upper z, upper y
			F9.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X9.x) - (Kodiag[ker_baseidx + 7 * kerplanecount].x * X9.y) - (Kodiag[ker_baseidx + 7 * kerplanecount].y * X9.z);
			F9.y = -1 * (Kodiag[ker_baseidx + 7 * kerplanecount].x * X9.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X9.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X9.z);
			F9.z = -1 * (Kodiag[ker_baseidx + 7 * kerplanecount].y * X9.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X9.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X9.z);

			F5.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X5.x) - (Kodiag[ker_baseidx + 6 * kerplanecount].x * X5.y) - (Kodiag[ker_baseidx + 6 * kerplanecount].y * X5.z);
			F5.y = -1 * (Kodiag[ker_baseidx + 6 * kerplanecount].x * X5.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X5.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X5.z);
			F5.z = -1 * (Kodiag[ker_baseidx + 6 * kerplanecount].y * X5.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X5.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X5.z);

			F13.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X13.x) - (Kodiag[ker_baseidx + 5 * kerplanecount].x * X13.y) - (Kodiag[ker_baseidx + 5 * kerplanecount].y * X13.z);
			F13.y = -1 * (Kodiag[ker_baseidx + 5 * kerplanecount].x * X13.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X13.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X13.z);
			F13.z = -1 * (Kodiag[ker_baseidx + 5 * kerplanecount].y * X13.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X13.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X13.z);

			F3.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X3.x) - (Kodiag[ker_baseidx + 4 * kerplanecount].x * X3.y) - (Kodiag[ker_baseidx + 4 * kerplanecount].y * X3.z);
			F3.y = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X3.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X3.z);
			F3.z = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].y * X3.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X3.z);

			F11.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X11.x) - (Kodiag[ker_baseidx + 3 * kerplanecount].x * X11.y) - (Kodiag[ker_baseidx + 3 * kerplanecount].y * X11.z);
			F11.y = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].x * X11.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X11.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X11.z);
			F11.z = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].y * X11.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X11.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X11.z);

			F7.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X7.x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X7.y) - (Kodiag[ker_baseidx + 2 * kerplanecount].y * X7.z);
			F7.y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X7.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X7.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X7.z);
			F7.z = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].y * X7.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X7.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X7.z);

			F15.x = (Kdiag[ker_baseidx + kerplanecount].x * X15.x) - (Kodiag[ker_baseidx + kerplanecount].x * X15.y) - (Kodiag[ker_baseidx + kerplanecount].y * X15.z);
			F15.y = -1 * (Kodiag[ker_baseidx + kerplanecount].x * X15.x) + (Kdiag[ker_baseidx + kerplanecount].y * X15.y) + (Kodiag[ker_baseidx + kerplanecount].z * X15.z);
			F15.z = -1 * (Kodiag[ker_baseidx + kerplanecount].y * X15.x) + (Kodiag[ker_baseidx + kerplanecount].z * X15.y) + (Kdiag[ker_baseidx + kerplanecount].z * X15.z);
		}

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;

		cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F4.x;
		cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F4.y;
		cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F4.z;

		cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F5.x;
		cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F5.y;
		cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F5.z;

		cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F6.x;
		cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F6.y;
		cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F6.z;

		cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F7.x;
		cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F7.y;
		cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F7.z;

		cuSx_out[idx + 8 * planecount] = (cuReIm)cuSx_out[idx + 8 * planecount] + F8.x;
		cuSy_out[idx + 8 * planecount] = (cuReIm)cuSy_out[idx + 8 * planecount] + F8.y;
		cuSz_out[idx + 8 * planecount] = (cuReIm)cuSz_out[idx + 8 * planecount] + F8.z;

		cuSx_out[idx + 9 * planecount] = (cuReIm)cuSx_out[idx + 9 * planecount] + F9.x;
		cuSy_out[idx + 9 * planecount] = (cuReIm)cuSy_out[idx + 9 * planecount] + F9.y;
		cuSz_out[idx + 9 * planecount] = (cuReIm)cuSz_out[idx + 9 * planecount] + F9.z;

		cuSx_out[idx + 10 * planecount] = (cuReIm)cuSx_out[idx + 10 * planecount] + F10.x;
		cuSy_out[idx + 10 * planecount] = (cuReIm)cuSy_out[idx + 10 * planecount] + F10.y;
		cuSz_out[idx + 10 * planecount] = (cuReIm)cuSz_out[idx + 10 * planecount] + F10.z;

		cuSx_out[idx + 11 * planecount] = (cuReIm)cuSx_out[idx + 11 * planecount] + F11.x;
		cuSy_out[idx + 11 * planecount] = (cuReIm)cuSy_out[idx + 11 * planecount] + F11.y;
		cuSz_out[idx + 11 * planecount] = (cuReIm)cuSz_out[idx + 11 * planecount] + F11.z;

		cuSx_out[idx + 12 * planecount] = (cuReIm)cuSx_out[idx + 12 * planecount] + F12.x;
		cuSy_out[idx + 12 * planecount] = (cuReIm)cuSy_out[idx + 12 * planecount] + F12.y;
		cuSz_out[idx + 12 * planecount] = (cuReIm)cuSz_out[idx + 12 * planecount] + F12.z;

		cuSx_out[idx + 13 * planecount] = (cuReIm)cuSx_out[idx + 13 * planecount] + F13.x;
		cuSy_out[idx + 13 * planecount] = (cuReIm)cuSy_out[idx + 13 * planecount] + F13.y;
		cuSz_out[idx + 13 * planecount] = (cuReIm)cuSz_out[idx + 13 * planecount] + F13.z;

		cuSx_out[idx + 14 * planecount] = (cuReIm)cuSx_out[idx + 14 * planecount] + F14.x;
		cuSy_out[idx + 14 * planecount] = (cuReIm)cuSy_out[idx + 14 * planecount] + F14.y;
		cuSz_out[idx + 14 * planecount] = (cuReIm)cuSz_out[idx + 14 * planecount] + F14.z;

		cuSx_out[idx + 15 * planecount] = (cuReIm)cuSx_out[idx + 15 * planecount] + F15.x;
		cuSy_out[idx + 15 * planecount] = (cuReIm)cuSy_out[idx + 15 * planecount] + F15.y;
		cuSz_out[idx + 15 * planecount] = (cuReIm)cuSz_out[idx + 15 * planecount] + F15.z;

#undef a
#undef b
#undef c
	}
}

//N = (N.x/2 + 1, N.y, 32)
//xy is transposed
//Complex kernels : this is a x shifted demag version using kernel symmetries, which accumulates in output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_32_transpose_xy_xshifted(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	//x shifted for 3D : can use kernels of reduced dimensions but must be complex
	//
	//Kxx, Kyy, Kzz : symmetrical (+) in y and z directions.
	//
	//Kxy : y - inv. symmetric (-), z - symmetrical  (+)
	//Kxz : y - symmetrical  (+), z - inv. symmetric (-)
	//Kyz : y - inv. symmetric (-), z - inv. symmetric (-)

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 32, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.y / 2 + 1) * (N.x / 2 + 1) size
	int kerplanecount = (N.x / 2 + 1) * (N.y / 2 + 1);

	if (idx < planecount) {

		//input data
#define x(n)	(cuReIm3(cuSx_in[idx + (n) * planecount], cuSy_in[idx + (n) * planecount], cuSz_in[idx + (n) * planecount]))

		//no performance gain to be had from setting these as X0, X1, ... etc.
		//unrolling loops does make a slight difference though - probably last case for which you want to unroll loops
		cuReIm3 X[32];

		cuReIm3 t0, t1, t2, t3;

		//input stage

#define a	(cuBReal)0.980785280403230
#define b	(cuBReal)0.195090322016128
#define c	(cuBReal)0.923879532511287
#define d	(cuBReal)0.382683432365090
#define e	(cuBReal)0.831469612302545
#define f	(cuBReal)0.555570233019602
#define g	(cuBReal)0.707106781186548

		//j = 0
		X[0] = (x(0) + x(8));
		X[8] = (x(0) - x(8));
		X[16] = (x(0) - !x(8));
		X[24] = (x(0) + !x(8));

		//j = 1
		X[1] = (x(1) + x(9));
		X[9] = (x(1) - x(9)) * cuReIm(c, -d);
		X[17] = (x(1) - !x(9)) * cuReIm(a, -b);
		X[25] = (x(1) + !x(9)) * cuReIm(e, -f);

		//j = 2
		X[2] = (x(2) + x(10));
		X[10] = (x(2) - x(10)) * cuReIm(g, -g);
		X[18] = (x(2) - !x(10)) * cuReIm(c, -d);
		X[26] = (x(2) + !x(10)) * cuReIm(d, -c);

		//j = 3
		X[3] = (x(3) + x(11));
		X[11] = (x(3) - x(11)) * cuReIm(d, -c);
		X[19] = (x(3) - !x(11)) * cuReIm(e, -f);
		X[27] = (x(3) + !x(11)) * cuReIm(-b, -a);

		//j = 4
		X[4] = (x(4) + x(12));
		X[12] = !(x(12) - x(4));
		X[20] = (x(4) - !x(12)) * cuReIm(g, -g);
		X[28] = (x(4) + !x(12)) * cuReIm(-g, -g);

		//j = 5
		X[5] = (x(5) + x(13));
		X[13] = (x(5) - x(13)) * cuReIm(-d, -c);
		X[21] = (x(5) - !x(13)) * cuReIm(f, -e);
		X[29] = (x(5) + !x(13)) * cuReIm(-a, -b);

		//j = 6
		X[6] = (x(6) + x(14));
		X[14] = (x(6) - x(14)) * cuReIm(-g, -g);
		X[22] = (x(6) - !x(14)) * cuReIm(d, -c);
		X[30] = (x(6) + !x(14)) * cuReIm(-c, d);

		//j = 7
		X[7] = (x(7) + x(15));
		X[15] = (x(7) - x(15)) * cuReIm(-c, -d);
		X[23] = (x(7) - !x(15)) * cuReIm(b, -a);
		X[31] = (x(7) + !x(15)) * cuReIm(-f, e);

#undef x

		//final radix4 stage

		//j = 0
		t0 = (X[0] + X[4]);
		t1 = (X[0] - X[4]);
		t2 = (X[2] + X[6]);
		t3 = !(X[6] - X[2]);

		X[0] = (t0 + t2);
		X[2] = (t0 - t2);
		X[4] = (t1 + t3);
		X[6] = (t1 - t3);

		t0 = (X[8] + X[12]);
		t1 = (X[8] - X[12]);
		t2 = (X[10] + X[14]);
		t3 = !(X[14] - X[10]);

		X[8] = (t0 + t2);
		X[10] = (t0 - t2);
		X[12] = (t1 + t3);
		X[14] = (t1 - t3);

		t0 = (X[16] + X[20]);
		t1 = (X[16] - X[20]);
		t2 = (X[18] + X[22]);
		t3 = !(X[22] - X[18]);

		X[16] = (t0 + t2);
		X[18] = (t0 - t2);
		X[20] = (t1 + t3);
		X[22] = (t1 - t3);

		t0 = (X[24] + X[28]);
		t1 = (X[24] - X[28]);
		t2 = (X[26] + X[30]);
		t3 = !(X[30] - X[26]);

		X[24] = (t0 + t2);
		X[26] = (t0 - t2);
		X[28] = (t1 + t3);
		X[30] = (t1 - t3);

		//j = 1
		t0 = (X[1] + X[5]);
		t1 = (X[1] - X[5]);
		t2 = (X[3] + X[7]);
		t3 = !(X[7] - X[3]);

		X[1] = (t0 + t2);
		X[3] = !(t2 - t0);
		X[5] = (t1 + t3) * cuReIm(g, -g);
		X[7] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[9] + X[13]);
		t1 = (X[9] - X[13]);
		t2 = (X[11] + X[15]);
		t3 = !(X[15] - X[11]);

		X[9] = (t0 + t2);
		X[11] = !(t2 - t0);
		X[13] = (t1 + t3) * cuReIm(g, -g);
		X[15] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[17] + X[21]);
		t1 = (X[17] - X[21]);
		t2 = (X[19] + X[23]);
		t3 = !(X[23] - X[19]);

		X[17] = (t0 + t2);
		X[19] = !(t2 - t0);
		X[21] = (t1 + t3) * cuReIm(g, -g);
		X[23] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[25] + X[29]);
		t1 = (X[25] - X[29]);
		t2 = (X[27] + X[31]);
		t3 = !(X[31] - X[27]);

		X[25] = (t0 + t2);
		X[27] = !(t2 - t0);
		X[29] = (t1 + t3) * cuReIm(g, -g);
		X[31] = (t1 - t3) * cuReIm(-g, -g);

		//radix-2 step to finish
		t0 = X[0] - X[1];
		X[0] = X[0] + X[1];
		X[1] = t0;

		t0 = X[2] - X[3];
		X[2] = X[2] + X[3];
		X[3] = t0;

		t0 = X[4] - X[5];
		X[4] = X[4] + X[5];
		X[5] = t0;

		t0 = X[6] - X[7];
		X[6] = X[6] + X[7];
		X[7] = t0;

		t0 = X[8] - X[9];
		X[8] = X[8] + X[9];
		X[9] = t0;

		t0 = X[10] - X[11];
		X[10] = X[10] + X[11];
		X[11] = t0;

		t0 = X[12] - X[13];
		X[12] = X[12] + X[13];
		X[13] = t0;

		t0 = X[14] - X[15];
		X[14] = X[14] + X[15];
		X[15] = t0;

		t0 = X[16] - X[17];
		X[16] = X[16] + X[17];
		X[17] = t0;

		t0 = X[18] - X[19];
		X[18] = X[18] + X[19];
		X[19] = t0;

		t0 = X[20] - X[21];
		X[20] = X[20] + X[21];
		X[21] = t0;

		t0 = X[22] - X[23];
		X[22] = X[22] + X[23];
		X[23] = t0;

		t0 = X[24] - X[25];
		X[24] = X[24] + X[25];
		X[25] = t0;

		t0 = X[26] - X[27];
		X[26] = X[26] + X[27];
		X[27] = t0;

		t0 = X[28] - X[29];
		X[28] = X[28] + X[29];
		X[29] = t0;

		t0 = X[30] - X[31];
		X[30] = X[30] + X[31];
		X[31] = t0;

		//output is shuffled now, i.e. it is ordered as:
		//0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31

		int i = idx % N.y;
		int j = (idx / N.y) % (N.x / 2 + 1);

		cuReIm3 F[32];

		if (i <= N.y / 2) {

			int ker_baseidx = i + j * (N.y / 2 + 1);

			//lower z, lower y
			F[0].x = (Kdiag[ker_baseidx].x * X[0].x) + (Kodiag[ker_baseidx].x * X[0].y) + (Kodiag[ker_baseidx].y * X[0].z);
			F[0].y = (Kodiag[ker_baseidx].x * X[0].x) + (Kdiag[ker_baseidx].y * X[0].y) + (Kodiag[ker_baseidx].z * X[0].z);
			F[0].z = (Kodiag[ker_baseidx].y * X[0].x) + (Kodiag[ker_baseidx].z * X[0].y) + (Kdiag[ker_baseidx].z * X[0].z);

			F[16].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].z);
			F[16].y = (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);
			F[16].z = (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);

			F[8].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].z);
			F[8].y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);
			F[8].z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);

			F[24].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].z);
			F[24].y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);
			F[24].z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);

			F[4].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].z);
			F[4].y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);
			F[4].z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);

			F[20].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].z);
			F[20].y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);
			F[20].z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);

			F[12].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].z);
			F[12].y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);
			F[12].z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);

			F[28].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].z);
			F[28].y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);
			F[28].z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);

			F[2].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].z);
			F[2].y = (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);
			F[2].z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);

			F[18].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].z);
			F[18].y = (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);
			F[18].z = (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);

			F[10].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].z);
			F[10].y = (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);
			F[10].z = (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);

			F[26].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].z);
			F[26].y = (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);
			F[26].z = (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);

			F[6].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].z);
			F[6].y = (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);
			F[6].z = (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);

			F[22].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].z);
			F[22].y = (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);
			F[22].z = (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);

			F[14].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].z);
			F[14].y = (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);
			F[14].z = (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);

			F[30].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].z);
			F[30].y = (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);
			F[30].z = (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);

			F[1].x = (Kdiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].z);
			F[1].y = (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kdiag[ker_baseidx + 16 * kerplanecount].y * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);
			F[1].z = (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].x) + (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].y) + (Kdiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);

			//upper z, lower y
			F[17].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[17].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[17].y) - (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[17].z);
			F[17].y = (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[17].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[17].y) - (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[17].z);
			F[17].z = -1 * (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[17].x) - (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[17].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[17].z);

			F[9].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[9].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[9].y) - (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[9].z);
			F[9].y = (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[9].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[9].y) - (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[9].z);
			F[9].z = -1 * (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[9].x) - (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[9].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[9].z);

			F[25].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[25].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[25].y) - (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[25].z);
			F[25].y = (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[25].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[25].y) - (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[25].z);
			F[25].z = -1 * (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[25].x) - (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[25].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[25].z);

			F[5].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[5].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[5].y) - (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[5].z);
			F[5].y = (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[5].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[5].y) - (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[5].z);
			F[5].z = -1 * (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[5].x) - (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[5].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[5].z);

			F[21].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[21].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[21].y) - (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[21].z);
			F[21].y = (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[21].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[21].y) - (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[21].z);
			F[21].z = -1 * (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[21].x) - (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[21].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[21].z);

			F[13].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[13].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[13].y) - (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[13].z);
			F[13].y = (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[13].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[13].y) - (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[13].z);
			F[13].z = -1 * (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[13].x) - (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[13].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[13].z);

			F[29].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[29].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[29].y) - (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[29].z);
			F[29].y = (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[29].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[29].y) - (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[29].z);
			F[29].z = -1 * (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[29].x) - (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[29].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[29].z);

			F[3].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[3].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[3].y) - (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[3].z);
			F[3].y = (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[3].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[3].y) - (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[3].z);
			F[3].z = -1 * (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[3].x) - (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[3].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[3].z);

			F[19].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[19].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[19].y) - (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[19].z);
			F[19].y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[19].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[19].y) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[19].z);
			F[19].z = -1 * (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[19].x) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[19].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[19].z);

			F[11].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[11].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[11].y) - (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[11].z);
			F[11].y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[11].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[11].y) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[11].z);
			F[11].z = -1 * (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[11].x) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[11].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[11].z);

			F[27].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[27].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[27].y) - (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[27].z);
			F[27].y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[27].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[27].y) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[27].z);
			F[27].z = -1 * (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[27].x) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[27].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[27].z);

			F[7].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[7].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[7].y) - (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[7].z);
			F[7].y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[7].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[7].y) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[7].z);
			F[7].z = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[7].x) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[7].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[7].z);

			F[23].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[23].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[23].y) - (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[23].z);
			F[23].y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[23].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[23].y) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[23].z);
			F[23].z = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[23].x) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[23].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[23].z);

			F[15].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[15].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[15].y) - (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[15].z);
			F[15].y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[15].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[15].y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[15].z);
			F[15].z = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[15].x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[15].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[15].z);

			F[31].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[31].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[31].y) - (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[31].z);
			F[31].y = (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[31].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[31].y) - (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[31].z);
			F[31].z = -1 * (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[31].x) - (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[31].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[31].z);
		}
		else {

			int ker_baseidx = (N.y - i) + j * (N.y / 2 + 1);

			//lower z, upper y
			F[0].x = (Kdiag[ker_baseidx].x * X[0].x) - (Kodiag[ker_baseidx].x * X[0].y) + (Kodiag[ker_baseidx].y * X[0].z);
			F[0].y = -1 * (Kodiag[ker_baseidx].x * X[0].x) + (Kdiag[ker_baseidx].y * X[0].y) - (Kodiag[ker_baseidx].z * X[0].z);
			F[0].z = (Kodiag[ker_baseidx].y * X[0].x) - (Kodiag[ker_baseidx].z * X[0].y) + (Kdiag[ker_baseidx].z * X[0].z);

			F[16].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) - (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].z);
			F[16].y = -1 * (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[16].y) - (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);
			F[16].z = (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].x) - (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);

			F[8].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].z);
			F[8].y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[8].y) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);
			F[8].z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].x) - (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);

			F[24].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) - (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].z);
			F[24].y = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[24].y) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);
			F[24].z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].x) - (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);

			F[4].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) - (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].z);
			F[4].y = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[4].y) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);
			F[4].z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].x) - (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);

			F[20].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) - (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].z);
			F[20].y = -1 * (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[20].y) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);
			F[20].z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].x) - (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);

			F[12].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) - (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].z);
			F[12].y = -1 * (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[12].y) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);
			F[12].z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].x) - (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);

			F[28].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) - (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].z);
			F[28].y = -1 * (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[28].y) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);
			F[28].z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].x) - (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);

			F[2].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) - (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].z);
			F[2].y = -1 * (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[2].y) - (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);
			F[2].z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].x) - (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);

			F[18].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) - (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].z);
			F[18].y = -1 * (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[18].y) - (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);
			F[18].z = (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].x) - (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);

			F[10].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) - (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].z);
			F[10].y = -1 * (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[10].y) - (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);
			F[10].z = (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].x) - (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);

			F[26].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) - (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].z);
			F[26].y = -1 * (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[26].y) - (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);
			F[26].z = (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].x) - (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);

			F[6].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) - (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].z);
			F[6].y = -1 * (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[6].y) - (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);
			F[6].z = (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].x) - (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);

			F[22].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) - (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].z);
			F[22].y = -1 * (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[22].y) - (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);
			F[22].z = (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].x) - (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);

			F[14].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) - (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].z);
			F[14].y = -1 * (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[14].y) - (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);
			F[14].z = (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].x) - (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);

			F[30].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) - (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].z);
			F[30].y = -1 * (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[30].y) - (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);
			F[30].z = (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].x) - (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);

			F[1].x = (Kdiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) - (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].z);
			F[1].y = -1 * (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kdiag[ker_baseidx + 16 * kerplanecount].y * X[1].y) - (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);
			F[1].z = (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].x) - (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].y) + (Kdiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);

			//upper z, upper y
			F[17].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[17].x) - (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[17].y) - (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[17].z);
			F[17].y = -1 * (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[17].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[17].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[17].z);
			F[17].z = -1 * (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[17].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[17].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[17].z);

			F[9].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[9].x) - (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[9].y) - (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[9].z);
			F[9].y = -1 * (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[9].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[9].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[9].z);
			F[9].z = -1 * (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[9].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[9].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[9].z);

			F[25].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[25].x) - (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[25].y) - (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[25].z);
			F[25].y = -1 * (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[25].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[25].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[25].z);
			F[25].z = -1 * (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[25].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[25].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[25].z);

			F[5].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[5].x) - (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[5].y) - (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[5].z);
			F[5].y = -1 * (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[5].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[5].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[5].z);
			F[5].z = -1 * (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[5].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[5].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[5].z);

			F[21].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[21].x) - (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[21].y) - (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[21].z);
			F[21].y = -1 * (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[21].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[21].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[21].z);
			F[21].z = -1 * (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[21].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[21].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[21].z);

			F[13].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[13].x) - (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[13].y) - (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[13].z);
			F[13].y = -1 * (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[13].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[13].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[13].z);
			F[13].z = -1 * (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[13].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[13].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[13].z);

			F[29].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[29].x) - (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[29].y) - (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[29].z);
			F[29].y = -1 * (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[29].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[29].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[29].z);
			F[29].z = -1 * (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[29].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[29].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[29].z);

			F[3].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[3].x) - (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[3].y) - (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[3].z);
			F[3].y = -1 * (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[3].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[3].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[3].z);
			F[3].z = -1 * (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[3].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[3].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[3].z);

			F[19].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[19].x) - (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[19].y) - (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[19].z);
			F[19].y = -1 * (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[19].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[19].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[19].z);
			F[19].z = -1 * (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[19].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[19].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[19].z);

			F[11].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[11].x) - (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[11].y) - (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[11].z);
			F[11].y = -1 * (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[11].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[11].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[11].z);
			F[11].z = -1 * (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[11].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[11].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[11].z);

			F[27].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[27].x) - (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[27].y) - (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[27].z);
			F[27].y = -1 * (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[27].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[27].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[27].z);
			F[27].z = -1 * (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[27].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[27].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[27].z);

			F[7].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[7].x) - (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[7].y) - (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[7].z);
			F[7].y = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[7].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[7].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[7].z);
			F[7].z = -1 * (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[7].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[7].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[7].z);

			F[23].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[23].x) - (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[23].y) - (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[23].z);
			F[23].y = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[23].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[23].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[23].z);
			F[23].z = -1 * (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[23].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[23].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[23].z);

			F[15].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[15].x) - (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[15].y) - (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[15].z);
			F[15].y = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[15].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[15].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[15].z);
			F[15].z = -1 * (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[15].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[15].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[15].z);

			F[31].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[31].x) - (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[31].y) - (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[31].z);
			F[31].y = -1 * (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[31].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[31].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[31].z);
			F[31].z = -1 * (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[31].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[31].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[31].z);
		}

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F[0].x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F[0].y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F[0].z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F[1].x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F[1].y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F[1].z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F[2].x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F[2].y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F[2].z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F[3].x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F[3].y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F[3].z;

		cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F[4].x;
		cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F[4].y;
		cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F[4].z;

		cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F[5].x;
		cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F[5].y;
		cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F[5].z;

		cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F[6].x;
		cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F[6].y;
		cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F[6].z;

		cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F[7].x;
		cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F[7].y;
		cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F[7].z;

		cuSx_out[idx + 8 * planecount] = (cuReIm)cuSx_out[idx + 8 * planecount] + F[8].x;
		cuSy_out[idx + 8 * planecount] = (cuReIm)cuSy_out[idx + 8 * planecount] + F[8].y;
		cuSz_out[idx + 8 * planecount] = (cuReIm)cuSz_out[idx + 8 * planecount] + F[8].z;

		cuSx_out[idx + 9 * planecount] = (cuReIm)cuSx_out[idx + 9 * planecount] + F[9].x;
		cuSy_out[idx + 9 * planecount] = (cuReIm)cuSy_out[idx + 9 * planecount] + F[9].y;
		cuSz_out[idx + 9 * planecount] = (cuReIm)cuSz_out[idx + 9 * planecount] + F[9].z;

		cuSx_out[idx + 10 * planecount] = (cuReIm)cuSx_out[idx + 10 * planecount] + F[10].x;
		cuSy_out[idx + 10 * planecount] = (cuReIm)cuSy_out[idx + 10 * planecount] + F[10].y;
		cuSz_out[idx + 10 * planecount] = (cuReIm)cuSz_out[idx + 10 * planecount] + F[10].z;

		cuSx_out[idx + 11 * planecount] = (cuReIm)cuSx_out[idx + 11 * planecount] + F[11].x;
		cuSy_out[idx + 11 * planecount] = (cuReIm)cuSy_out[idx + 11 * planecount] + F[11].y;
		cuSz_out[idx + 11 * planecount] = (cuReIm)cuSz_out[idx + 11 * planecount] + F[11].z;

		cuSx_out[idx + 12 * planecount] = (cuReIm)cuSx_out[idx + 12 * planecount] + F[12].x;
		cuSy_out[idx + 12 * planecount] = (cuReIm)cuSy_out[idx + 12 * planecount] + F[12].y;
		cuSz_out[idx + 12 * planecount] = (cuReIm)cuSz_out[idx + 12 * planecount] + F[12].z;

		cuSx_out[idx + 13 * planecount] = (cuReIm)cuSx_out[idx + 13 * planecount] + F[13].x;
		cuSy_out[idx + 13 * planecount] = (cuReIm)cuSy_out[idx + 13 * planecount] + F[13].y;
		cuSz_out[idx + 13 * planecount] = (cuReIm)cuSz_out[idx + 13 * planecount] + F[13].z;

		cuSx_out[idx + 14 * planecount] = (cuReIm)cuSx_out[idx + 14 * planecount] + F[14].x;
		cuSy_out[idx + 14 * planecount] = (cuReIm)cuSy_out[idx + 14 * planecount] + F[14].y;
		cuSz_out[idx + 14 * planecount] = (cuReIm)cuSz_out[idx + 14 * planecount] + F[14].z;

		cuSx_out[idx + 15 * planecount] = (cuReIm)cuSx_out[idx + 15 * planecount] + F[15].x;
		cuSy_out[idx + 15 * planecount] = (cuReIm)cuSy_out[idx + 15 * planecount] + F[15].y;
		cuSz_out[idx + 15 * planecount] = (cuReIm)cuSz_out[idx + 15 * planecount] + F[15].z;

		cuSx_out[idx + 16 * planecount] = (cuReIm)cuSx_out[idx + 16 * planecount] + F[16].x;
		cuSy_out[idx + 16 * planecount] = (cuReIm)cuSy_out[idx + 16 * planecount] + F[16].y;
		cuSz_out[idx + 16 * planecount] = (cuReIm)cuSz_out[idx + 16 * planecount] + F[16].z;

		cuSx_out[idx + 17 * planecount] = (cuReIm)cuSx_out[idx + 17 * planecount] + F[17].x;
		cuSy_out[idx + 17 * planecount] = (cuReIm)cuSy_out[idx + 17 * planecount] + F[17].y;
		cuSz_out[idx + 17 * planecount] = (cuReIm)cuSz_out[idx + 17 * planecount] + F[17].z;

		cuSx_out[idx + 18 * planecount] = (cuReIm)cuSx_out[idx + 18 * planecount] + F[18].x;
		cuSy_out[idx + 18 * planecount] = (cuReIm)cuSy_out[idx + 18 * planecount] + F[18].y;
		cuSz_out[idx + 18 * planecount] = (cuReIm)cuSz_out[idx + 18 * planecount] + F[18].z;

		cuSx_out[idx + 19 * planecount] = (cuReIm)cuSx_out[idx + 19 * planecount] + F[19].x;
		cuSy_out[idx + 19 * planecount] = (cuReIm)cuSy_out[idx + 19 * planecount] + F[19].y;
		cuSz_out[idx + 19 * planecount] = (cuReIm)cuSz_out[idx + 19 * planecount] + F[19].z;

		cuSx_out[idx + 20 * planecount] = (cuReIm)cuSx_out[idx + 20 * planecount] + F[20].x;
		cuSy_out[idx + 20 * planecount] = (cuReIm)cuSy_out[idx + 20 * planecount] + F[20].y;
		cuSz_out[idx + 20 * planecount] = (cuReIm)cuSz_out[idx + 20 * planecount] + F[20].z;

		cuSx_out[idx + 21 * planecount] = (cuReIm)cuSx_out[idx + 21 * planecount] + F[21].x;
		cuSy_out[idx + 21 * planecount] = (cuReIm)cuSy_out[idx + 21 * planecount] + F[21].y;
		cuSz_out[idx + 21 * planecount] = (cuReIm)cuSz_out[idx + 21 * planecount] + F[21].z;

		cuSx_out[idx + 22 * planecount] = (cuReIm)cuSx_out[idx + 22 * planecount] + F[22].x;
		cuSy_out[idx + 22 * planecount] = (cuReIm)cuSy_out[idx + 22 * planecount] + F[22].y;
		cuSz_out[idx + 22 * planecount] = (cuReIm)cuSz_out[idx + 22 * planecount] + F[22].z;

		cuSx_out[idx + 23 * planecount] = (cuReIm)cuSx_out[idx + 23 * planecount] + F[23].x;
		cuSy_out[idx + 23 * planecount] = (cuReIm)cuSy_out[idx + 23 * planecount] + F[23].y;
		cuSz_out[idx + 23 * planecount] = (cuReIm)cuSz_out[idx + 23 * planecount] + F[23].z;

		cuSx_out[idx + 24 * planecount] = (cuReIm)cuSx_out[idx + 24 * planecount] + F[24].x;
		cuSy_out[idx + 24 * planecount] = (cuReIm)cuSy_out[idx + 24 * planecount] + F[24].y;
		cuSz_out[idx + 24 * planecount] = (cuReIm)cuSz_out[idx + 24 * planecount] + F[24].z;

		cuSx_out[idx + 25 * planecount] = (cuReIm)cuSx_out[idx + 25 * planecount] + F[25].x;
		cuSy_out[idx + 25 * planecount] = (cuReIm)cuSy_out[idx + 25 * planecount] + F[25].y;
		cuSz_out[idx + 25 * planecount] = (cuReIm)cuSz_out[idx + 25 * planecount] + F[25].z;

		cuSx_out[idx + 26 * planecount] = (cuReIm)cuSx_out[idx + 26 * planecount] + F[26].x;
		cuSy_out[idx + 26 * planecount] = (cuReIm)cuSy_out[idx + 26 * planecount] + F[26].y;
		cuSz_out[idx + 26 * planecount] = (cuReIm)cuSz_out[idx + 26 * planecount] + F[26].z;

		cuSx_out[idx + 27 * planecount] = (cuReIm)cuSx_out[idx + 27 * planecount] + F[27].x;
		cuSy_out[idx + 27 * planecount] = (cuReIm)cuSy_out[idx + 27 * planecount] + F[27].y;
		cuSz_out[idx + 27 * planecount] = (cuReIm)cuSz_out[idx + 27 * planecount] + F[27].z;

		cuSx_out[idx + 28 * planecount] = (cuReIm)cuSx_out[idx + 28 * planecount] + F[28].x;
		cuSy_out[idx + 28 * planecount] = (cuReIm)cuSy_out[idx + 28 * planecount] + F[28].y;
		cuSz_out[idx + 28 * planecount] = (cuReIm)cuSz_out[idx + 28 * planecount] + F[28].z;

		cuSx_out[idx + 29 * planecount] = (cuReIm)cuSx_out[idx + 29 * planecount] + F[29].x;
		cuSy_out[idx + 29 * planecount] = (cuReIm)cuSy_out[idx + 29 * planecount] + F[29].y;
		cuSz_out[idx + 29 * planecount] = (cuReIm)cuSz_out[idx + 29 * planecount] + F[29].z;

		cuSx_out[idx + 30 * planecount] = (cuReIm)cuSx_out[idx + 30 * planecount] + F[30].x;
		cuSy_out[idx + 30 * planecount] = (cuReIm)cuSy_out[idx + 30 * planecount] + F[30].y;
		cuSz_out[idx + 30 * planecount] = (cuReIm)cuSz_out[idx + 30 * planecount] + F[30].z;

		cuSx_out[idx + 31 * planecount] = (cuReIm)cuSx_out[idx + 31 * planecount] + F[31].x;
		cuSy_out[idx + 31 * planecount] = (cuReIm)cuSy_out[idx + 31 * planecount] + F[31].y;
		cuSz_out[idx + 31 * planecount] = (cuReIm)cuSz_out[idx + 31 * planecount] + F[31].z;

#undef a
#undef b
#undef c
#undef d
#undef e
#undef f
#undef g
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// Q2D - REGULAR

//N = (N.x/2 + 1, N.y, 4)
//xy is transposed
//Complex kernels : this is a inter-mesh demag version which accumulates into output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_4_transpose_xy_Regular(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	//Regular version : no kernel symmetries

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 4, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.x / 2 + 1)*N.y size
	int kerplanecount = (N.x / 2 + 1) * N.y;

	if (idx < planecount) {

		//the z-axis points (the others are zero)
		cuReIm3 a = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 b = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);

		//forward z-axis fft
		//NOTE: cuda fft uses -i for the forward fft and +i for the inverse fft.
		//The kernels are purely real so you would get the same result by taking +i for the forward and -i for the inverse, but better to keep it consistent : use the cuda fft convention here.
		cuReIm3 X0 = a + b;
		cuReIm3 X1 = a - !b;
		cuReIm3 X2 = a - b;
		cuReIm3 X3 = a + !b;

		//kernel multiplication
		cuReIm3 F0, F1, F2, F3;

		int ker_baseidx = idx;

		F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
		F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
		F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

		F1.x = (Kdiag[ker_baseidx + kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + kerplanecount].y * X1.z);
		F1.y = (Kodiag[ker_baseidx + kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + kerplanecount].z * X1.z);
		F1.z = (Kodiag[ker_baseidx + kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + kerplanecount].z * X1.z);

		F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
		F2.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
		F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

		F3.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X3.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X3.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X3.z);
		F3.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X3.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X3.z);
		F3.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X3.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X3.z);

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;
	}
}

//N = (N.x/2 + 1, N.y, 8)
//xy is transposed
//Complex kernels : this is a inter-mesh demag version which accumulates into output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_8_transpose_xy_Regular(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 8, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.x / 2 + 1)*N.y size
	int kerplanecount = (N.x / 2 + 1) * N.y;

	if (idx < planecount) {

#define a (cuBReal)0.7071067811865

		//the z-axis points (the others are zero)
		cuReIm3 x0 = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 x1 = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);
		cuReIm3 x2 = cuReIm3(cuSx_in[idx + 2 * planecount], cuSy_in[idx + 2 * planecount], cuSz_in[idx + 2 * planecount]);
		cuReIm3 x3 = cuReIm3(cuSx_in[idx + 3 * planecount], cuSy_in[idx + 3 * planecount], cuSz_in[idx + 3 * planecount]);

		//Radix-4 step
		cuReIm3 X0 = x0 + x2;
		cuReIm3 X2 = x0 - x2;
		cuReIm3 X4 = x0 - !x2;
		cuReIm3 X6 = x0 + !x2;

		cuReIm3 X1 = x1 + x3;
		cuReIm3 X3 = !(x3 - x1);
		cuReIm3 X5 = (x1 - !x3) * cuReIm(a, -a);
		cuReIm3 X7 = (x1 + !x3) * cuReIm(-a, -a);

		//Radix-2 step
		cuReIm3 temp = X0 - X1;
		X0 = X0 + X1;
		X1 = temp;

		temp = X2 - X3;
		X2 = X2 + X3;
		X3 = temp;

		temp = X4 - X5;
		X4 = X4 + X5;
		X5 = temp;

		temp = X6 - X7;
		X6 = X6 + X7;
		X7 = temp;

		//data set in shuffled order:
		//X0, X4, X2, X6, X1, X5, X3, X7

		cuReIm3 F0, F1, F2, F3, F4, F5, F6, F7;

		int ker_baseidx = idx;

		F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
		F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
		F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

		F4.x = (Kdiag[ker_baseidx + kerplanecount].x * X4.x) + (Kodiag[ker_baseidx + kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + kerplanecount].y * X4.z);
		F4.y = (Kodiag[ker_baseidx + kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + kerplanecount].y * X4.y) + (Kodiag[ker_baseidx + kerplanecount].z * X4.z);
		F4.z = (Kodiag[ker_baseidx + kerplanecount].y * X4.x) + (Kodiag[ker_baseidx + kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + kerplanecount].z * X4.z);

		F2.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.z);
		F2.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.z);
		F2.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X2.z);

		F6.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.z);
		F6.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X6.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.z);
		F6.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X6.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X6.z);

		F1.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.z);
		F1.y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.z);
		F1.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X1.z);

		F5.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X5.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X5.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X5.z);
		F5.y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X5.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X5.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X5.z);
		F5.z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X5.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X5.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X5.z);

		F3.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X3.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X3.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X3.z);
		F3.y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X3.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X3.z);
		F3.z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X3.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X3.z);

		F7.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X7.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X7.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X7.z);
		F7.y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X7.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X7.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X7.z);
		F7.z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X7.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X7.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X7.z);

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;

		cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F4.x;
		cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F4.y;
		cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F4.z;

		cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F5.x;
		cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F5.y;
		cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F5.z;

		cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F6.x;
		cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F6.y;
		cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F6.z;

		cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F7.x;
		cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F7.y;
		cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F7.z;

#undef a
	}
}

//N = (N.x/2 + 1, N.y, 16)
//xy is transposed
//Complex kernels : this is a inter-mesh demag version which accumulates into output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_16_transpose_xy_Regular(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 16, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.x / 2 + 1)*N.y size
	int kerplanecount = (N.x / 2 + 1) * N.y;

	if (idx < planecount) {

		//the z-axis points (the others are zero)
		cuReIm3 x0 = cuReIm3(cuSx_in[idx], cuSy_in[idx], cuSz_in[idx]);
		cuReIm3 x1 = cuReIm3(cuSx_in[idx + planecount], cuSy_in[idx + planecount], cuSz_in[idx + planecount]);
		cuReIm3 x2 = cuReIm3(cuSx_in[idx + 2 * planecount], cuSy_in[idx + 2 * planecount], cuSz_in[idx + 2 * planecount]);
		cuReIm3 x3 = cuReIm3(cuSx_in[idx + 3 * planecount], cuSy_in[idx + 3 * planecount], cuSz_in[idx + 3 * planecount]);
		cuReIm3 x4 = cuReIm3(cuSx_in[idx + 4 * planecount], cuSy_in[idx + 4 * planecount], cuSz_in[idx + 4 * planecount]);
		cuReIm3 x5 = cuReIm3(cuSx_in[idx + 5 * planecount], cuSy_in[idx + 5 * planecount], cuSz_in[idx + 5 * planecount]);
		cuReIm3 x6 = cuReIm3(cuSx_in[idx + 6 * planecount], cuSy_in[idx + 6 * planecount], cuSz_in[idx + 6 * planecount]);
		cuReIm3 x7 = cuReIm3(cuSx_in[idx + 7 * planecount], cuSy_in[idx + 7 * planecount], cuSz_in[idx + 7 * planecount]);

#define a	(cuBReal)9.238795325113E-01
#define b	(cuBReal)3.826834323651E-01
#define c	(cuBReal)7.071067811865E-01

		//First stage
		cuReIm3 X0 = x0 + x4;
		cuReIm3 X4 = x0 - x4;
		cuReIm3 X8 = x0 - !x4;
		cuReIm3 X12 = x0 + !x4;

		cuReIm3 X1 = x1 + x5;
		cuReIm3 X5 = (x1 - x5) * cuReIm(c, -c);
		cuReIm3 X9 = (x1 - !x5) * cuReIm(a, -b);
		cuReIm3 X13 = (x1 + !x5) * cuReIm(b, -a);

		cuReIm3 X2 = x2 + x6;
		cuReIm3 X6 = !(x6 - x2);
		cuReIm3 X10 = (x2 - !x6) * cuReIm(c, -c);
		cuReIm3 X14 = (x2 + !x6) * cuReIm(-c, -c);

		cuReIm3 X3 = x3 + x7;
		cuReIm3 X7 = (x3 - x7) * cuReIm(-c, -c);
		cuReIm3 X11 = (x3 - !x7) * cuReIm(b, -a);
		cuReIm3 X15 = (x3 + !x7) * cuReIm(-a, b);

		//Second stage
		cuReIm3 t0 = X0 + X2;
		cuReIm3 t1 = X0 - X2;
		cuReIm3 t2 = X1 + X3;
		cuReIm3 t3 = !(X3 - X1);

		X0 = t0 + t2;
		X1 = t0 - t2;
		X2 = t1 + t3;
		X3 = t1 - t3;

		t0 = X4 + X6;
		t1 = X4 - X6;
		t2 = X5 + X7;
		t3 = !(X7 - X5);

		X4 = t0 + t2;
		X5 = t0 - t2;
		X6 = t1 + t3;
		X7 = t1 - t3;

		t0 = X8 + X10;
		t1 = X8 - X10;
		t2 = X9 + X11;
		t3 = !(X11 - X9);

		X8 = t0 + t2;
		X9 = t0 - t2;
		X10 = t1 + t3;
		X11 = t1 - t3;

		t0 = X12 + X14;
		t1 = X12 - X14;
		t2 = X13 + X15;
		t3 = !(X15 - X13);

		X12 = t0 + t2;
		X13 = t0 - t2;
		X14 = t1 + t3;
		X15 = t1 - t3;

		//output is shuffled now, i.e. it is ordered as:
		//X0, X8, X4, X12, X2, X10, X6, X14, X1, X9, X5, X13, X3, X11, X7, X15

		cuReIm3 F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15;

		int ker_baseidx = idx;

		F0.x = (Kdiag[ker_baseidx].x * X0.x) + (Kodiag[ker_baseidx].x * X0.y) + (Kodiag[ker_baseidx].y * X0.z);
		F0.y = (Kodiag[ker_baseidx].x * X0.x) + (Kdiag[ker_baseidx].y * X0.y) + (Kodiag[ker_baseidx].z * X0.z);
		F0.z = (Kodiag[ker_baseidx].y * X0.x) + (Kodiag[ker_baseidx].z * X0.y) + (Kdiag[ker_baseidx].z * X0.z);

		F8.x = (Kdiag[ker_baseidx + kerplanecount].x * X8.x) + (Kodiag[ker_baseidx + kerplanecount].x * X8.y) + (Kodiag[ker_baseidx + kerplanecount].y * X8.z);
		F8.y = (Kodiag[ker_baseidx + kerplanecount].x * X8.x) + (Kdiag[ker_baseidx + kerplanecount].y * X8.y) + (Kodiag[ker_baseidx + kerplanecount].z * X8.z);
		F8.z = (Kodiag[ker_baseidx + kerplanecount].y * X8.x) + (Kodiag[ker_baseidx + kerplanecount].z * X8.y) + (Kdiag[ker_baseidx + kerplanecount].z * X8.z);

		F4.x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.z);
		F4.y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X4.x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X4.y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.z);
		F4.z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X4.x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X4.y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X4.z);

		F12.x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.z);
		F12.y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X12.x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X12.y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.z);
		F12.z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X12.x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X12.y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X12.z);

		F2.x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.z);
		F2.y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X2.x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X2.y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.z);
		F2.z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X2.x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X2.y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X2.z);

		F10.x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.z);
		F10.y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X10.x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X10.y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.z);
		F10.z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X10.x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X10.y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X10.z);

		F6.x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.z);
		F6.y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X6.x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X6.y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.z);
		F6.z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X6.x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X6.y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X6.z);

		F14.x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.z);
		F14.y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X14.x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X14.y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.z);
		F14.z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X14.x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X14.y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X14.z);

		F1.x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.z);
		F1.y = (Kodiag[ker_baseidx + 8 * kerplanecount].x * X1.x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X1.y) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.z);
		F1.z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X1.x) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X1.y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X1.z);

		F9.x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X9.x) + (Kodiag[ker_baseidx + 9 * kerplanecount].x * X9.y) + (Kodiag[ker_baseidx + 9 * kerplanecount].y * X9.z);
		F9.y = (Kodiag[ker_baseidx + 9 * kerplanecount].x * X9.x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X9.y) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X9.z);
		F9.z = (Kodiag[ker_baseidx + 9 * kerplanecount].y * X9.x) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X9.y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X9.z);

		F5.x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X5.x) + (Kodiag[ker_baseidx + 10 * kerplanecount].x * X5.y) + (Kodiag[ker_baseidx + 10 * kerplanecount].y * X5.z);
		F5.y = (Kodiag[ker_baseidx + 10 * kerplanecount].x * X5.x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X5.y) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X5.z);
		F5.z = (Kodiag[ker_baseidx + 10 * kerplanecount].y * X5.x) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X5.y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X5.z);

		F13.x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X13.x) + (Kodiag[ker_baseidx + 11 * kerplanecount].x * X13.y) + (Kodiag[ker_baseidx + 11 * kerplanecount].y * X13.z);
		F13.y = (Kodiag[ker_baseidx + 11 * kerplanecount].x * X13.x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X13.y) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X13.z);
		F13.z = (Kodiag[ker_baseidx + 11 * kerplanecount].y * X13.x) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X13.y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X13.z);

		F3.x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X3.x) + (Kodiag[ker_baseidx + 12 * kerplanecount].x * X3.y) + (Kodiag[ker_baseidx + 12 * kerplanecount].y * X3.z);
		F3.y = (Kodiag[ker_baseidx + 12 * kerplanecount].x * X3.x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X3.y) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X3.z);
		F3.z = (Kodiag[ker_baseidx + 12 * kerplanecount].y * X3.x) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X3.y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X3.z);

		F11.x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X11.x) + (Kodiag[ker_baseidx + 13 * kerplanecount].x * X11.y) + (Kodiag[ker_baseidx + 13 * kerplanecount].y * X11.z);
		F11.y = (Kodiag[ker_baseidx + 13 * kerplanecount].x * X11.x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X11.y) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X11.z);
		F11.z = (Kodiag[ker_baseidx + 13 * kerplanecount].y * X11.x) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X11.y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X11.z);

		F7.x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X7.x) + (Kodiag[ker_baseidx + 14 * kerplanecount].x * X7.y) + (Kodiag[ker_baseidx + 14 * kerplanecount].y * X7.z);
		F7.y = (Kodiag[ker_baseidx + 14 * kerplanecount].x * X7.x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X7.y) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X7.z);
		F7.z = (Kodiag[ker_baseidx + 14 * kerplanecount].y * X7.x) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X7.y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X7.z);

		F15.x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X15.x) + (Kodiag[ker_baseidx + 15 * kerplanecount].x * X15.y) + (Kodiag[ker_baseidx + 15 * kerplanecount].y * X15.z);
		F15.y = (Kodiag[ker_baseidx + 15 * kerplanecount].x * X15.x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X15.y) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X15.z);
		F15.z = (Kodiag[ker_baseidx + 15 * kerplanecount].y * X15.x) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X15.y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X15.z);

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F0.x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F0.y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F0.z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F1.x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F1.y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F1.z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F2.x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F2.y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F2.z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F3.x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F3.y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F3.z;

		cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F4.x;
		cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F4.y;
		cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F4.z;

		cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F5.x;
		cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F5.y;
		cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F5.z;

		cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F6.x;
		cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F6.y;
		cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F6.z;

		cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F7.x;
		cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F7.y;
		cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F7.z;

		cuSx_out[idx + 8 * planecount] = (cuReIm)cuSx_out[idx + 8 * planecount] + F8.x;
		cuSy_out[idx + 8 * planecount] = (cuReIm)cuSy_out[idx + 8 * planecount] + F8.y;
		cuSz_out[idx + 8 * planecount] = (cuReIm)cuSz_out[idx + 8 * planecount] + F8.z;

		cuSx_out[idx + 9 * planecount] = (cuReIm)cuSx_out[idx + 9 * planecount] + F9.x;
		cuSy_out[idx + 9 * planecount] = (cuReIm)cuSy_out[idx + 9 * planecount] + F9.y;
		cuSz_out[idx + 9 * planecount] = (cuReIm)cuSz_out[idx + 9 * planecount] + F9.z;

		cuSx_out[idx + 10 * planecount] = (cuReIm)cuSx_out[idx + 10 * planecount] + F10.x;
		cuSy_out[idx + 10 * planecount] = (cuReIm)cuSy_out[idx + 10 * planecount] + F10.y;
		cuSz_out[idx + 10 * planecount] = (cuReIm)cuSz_out[idx + 10 * planecount] + F10.z;

		cuSx_out[idx + 11 * planecount] = (cuReIm)cuSx_out[idx + 11 * planecount] + F11.x;
		cuSy_out[idx + 11 * planecount] = (cuReIm)cuSy_out[idx + 11 * planecount] + F11.y;
		cuSz_out[idx + 11 * planecount] = (cuReIm)cuSz_out[idx + 11 * planecount] + F11.z;

		cuSx_out[idx + 12 * planecount] = (cuReIm)cuSx_out[idx + 12 * planecount] + F12.x;
		cuSy_out[idx + 12 * planecount] = (cuReIm)cuSy_out[idx + 12 * planecount] + F12.y;
		cuSz_out[idx + 12 * planecount] = (cuReIm)cuSz_out[idx + 12 * planecount] + F12.z;

		cuSx_out[idx + 13 * planecount] = (cuReIm)cuSx_out[idx + 13 * planecount] + F13.x;
		cuSy_out[idx + 13 * planecount] = (cuReIm)cuSy_out[idx + 13 * planecount] + F13.y;
		cuSz_out[idx + 13 * planecount] = (cuReIm)cuSz_out[idx + 13 * planecount] + F13.z;

		cuSx_out[idx + 14 * planecount] = (cuReIm)cuSx_out[idx + 14 * planecount] + F14.x;
		cuSy_out[idx + 14 * planecount] = (cuReIm)cuSy_out[idx + 14 * planecount] + F14.y;
		cuSz_out[idx + 14 * planecount] = (cuReIm)cuSz_out[idx + 14 * planecount] + F14.z;

		cuSx_out[idx + 15 * planecount] = (cuReIm)cuSx_out[idx + 15 * planecount] + F15.x;
		cuSy_out[idx + 15 * planecount] = (cuReIm)cuSy_out[idx + 15 * planecount] + F15.y;
		cuSz_out[idx + 15 * planecount] = (cuReIm)cuSz_out[idx + 15 * planecount] + F15.z;

#undef a
#undef b
#undef c
	}
}

//N = (N.x/2 + 1, N.y, 32)
//xy is transposed
//Complex kernels : this is a inter-mesh demag version which accumulates into output, but ifft not performed yet
__global__ void cu_MultiDemag_ConvProd_q2D_32_transpose_xy_Regular(
	cuVEC<cuReIm3>& Kdiag, cuVEC<cuReIm3>& Kodiag,
	cuBComplex* cuSx_in, cuBComplex* cuSy_in, cuBComplex* cuSz_in,
	cuBComplex* cuSx_out, cuBComplex* cuSy_out, cuBComplex* cuSz_out,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 32, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	//kernels packed into planes of (N.x / 2 + 1)*N.y size
	int kerplanecount = (N.x / 2 + 1) * N.y;

	if (idx < planecount) {

		//input data
#define x(n)	(cuReIm3(cuSx_in[idx + (n) * planecount], cuSy_in[idx + (n) * planecount], cuSz_in[idx + (n) * planecount]))

		//no performance gain to be had from setting these as X0, X1, ... etc.
		//unrolling loops does make a slight difference though - probably last case for which you want to unroll loops
		cuReIm3 X[32];

		cuReIm3 t0, t1, t2, t3;

		//input stage

#define a	(cuBReal)0.980785280403230
#define b	(cuBReal)0.195090322016128
#define c	(cuBReal)0.923879532511287
#define d	(cuBReal)0.382683432365090
#define e	(cuBReal)0.831469612302545
#define f	(cuBReal)0.555570233019602
#define g	(cuBReal)0.707106781186548

		//j = 0
		X[0] = (x(0) + x(8));
		X[8] = (x(0) - x(8));
		X[16] = (x(0) - !x(8));
		X[24] = (x(0) + !x(8));

		//j = 1
		X[1] = (x(1) + x(9));
		X[9] = (x(1) - x(9)) * cuReIm(c, -d);
		X[17] = (x(1) - !x(9)) * cuReIm(a, -b);
		X[25] = (x(1) + !x(9)) * cuReIm(e, -f);

		//j = 2
		X[2] = (x(2) + x(10));
		X[10] = (x(2) - x(10)) * cuReIm(g, -g);
		X[18] = (x(2) - !x(10)) * cuReIm(c, -d);
		X[26] = (x(2) + !x(10)) * cuReIm(d, -c);

		//j = 3
		X[3] = (x(3) + x(11));
		X[11] = (x(3) - x(11)) * cuReIm(d, -c);
		X[19] = (x(3) - !x(11)) * cuReIm(e, -f);
		X[27] = (x(3) + !x(11)) * cuReIm(-b, -a);

		//j = 4
		X[4] = (x(4) + x(12));
		X[12] = !(x(12) - x(4));
		X[20] = (x(4) - !x(12)) * cuReIm(g, -g);
		X[28] = (x(4) + !x(12)) * cuReIm(-g, -g);

		//j = 5
		X[5] = (x(5) + x(13));
		X[13] = (x(5) - x(13)) * cuReIm(-d, -c);
		X[21] = (x(5) - !x(13)) * cuReIm(f, -e);
		X[29] = (x(5) + !x(13)) * cuReIm(-a, -b);

		//j = 6
		X[6] = (x(6) + x(14));
		X[14] = (x(6) - x(14)) * cuReIm(-g, -g);
		X[22] = (x(6) - !x(14)) * cuReIm(d, -c);
		X[30] = (x(6) + !x(14)) * cuReIm(-c, d);

		//j = 7
		X[7] = (x(7) + x(15));
		X[15] = (x(7) - x(15)) * cuReIm(-c, -d);
		X[23] = (x(7) - !x(15)) * cuReIm(b, -a);
		X[31] = (x(7) + !x(15)) * cuReIm(-f, e);

#undef x

		//final radix4 stage

		//j = 0
		t0 = (X[0] + X[4]);
		t1 = (X[0] - X[4]);
		t2 = (X[2] + X[6]);
		t3 = !(X[6] - X[2]);

		X[0] = (t0 + t2);
		X[2] = (t0 - t2);
		X[4] = (t1 + t3);
		X[6] = (t1 - t3);

		t0 = (X[8] + X[12]);
		t1 = (X[8] - X[12]);
		t2 = (X[10] + X[14]);
		t3 = !(X[14] - X[10]);

		X[8] = (t0 + t2);
		X[10] = (t0 - t2);
		X[12] = (t1 + t3);
		X[14] = (t1 - t3);

		t0 = (X[16] + X[20]);
		t1 = (X[16] - X[20]);
		t2 = (X[18] + X[22]);
		t3 = !(X[22] - X[18]);

		X[16] = (t0 + t2);
		X[18] = (t0 - t2);
		X[20] = (t1 + t3);
		X[22] = (t1 - t3);

		t0 = (X[24] + X[28]);
		t1 = (X[24] - X[28]);
		t2 = (X[26] + X[30]);
		t3 = !(X[30] - X[26]);

		X[24] = (t0 + t2);
		X[26] = (t0 - t2);
		X[28] = (t1 + t3);
		X[30] = (t1 - t3);

		//j = 1
		t0 = (X[1] + X[5]);
		t1 = (X[1] - X[5]);
		t2 = (X[3] + X[7]);
		t3 = !(X[7] - X[3]);

		X[1] = (t0 + t2);
		X[3] = !(t2 - t0);
		X[5] = (t1 + t3) * cuReIm(g, -g);
		X[7] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[9] + X[13]);
		t1 = (X[9] - X[13]);
		t2 = (X[11] + X[15]);
		t3 = !(X[15] - X[11]);

		X[9] = (t0 + t2);
		X[11] = !(t2 - t0);
		X[13] = (t1 + t3) * cuReIm(g, -g);
		X[15] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[17] + X[21]);
		t1 = (X[17] - X[21]);
		t2 = (X[19] + X[23]);
		t3 = !(X[23] - X[19]);

		X[17] = (t0 + t2);
		X[19] = !(t2 - t0);
		X[21] = (t1 + t3) * cuReIm(g, -g);
		X[23] = (t1 - t3) * cuReIm(-g, -g);

		t0 = (X[25] + X[29]);
		t1 = (X[25] - X[29]);
		t2 = (X[27] + X[31]);
		t3 = !(X[31] - X[27]);

		X[25] = (t0 + t2);
		X[27] = !(t2 - t0);
		X[29] = (t1 + t3) * cuReIm(g, -g);
		X[31] = (t1 - t3) * cuReIm(-g, -g);

		//radix-2 step to finish
		t0 = X[0] - X[1];
		X[0] = X[0] + X[1];
		X[1] = t0;

		t0 = X[2] - X[3];
		X[2] = X[2] + X[3];
		X[3] = t0;

		t0 = X[4] - X[5];
		X[4] = X[4] + X[5];
		X[5] = t0;

		t0 = X[6] - X[7];
		X[6] = X[6] + X[7];
		X[7] = t0;

		t0 = X[8] - X[9];
		X[8] = X[8] + X[9];
		X[9] = t0;

		t0 = X[10] - X[11];
		X[10] = X[10] + X[11];
		X[11] = t0;

		t0 = X[12] - X[13];
		X[12] = X[12] + X[13];
		X[13] = t0;

		t0 = X[14] - X[15];
		X[14] = X[14] + X[15];
		X[15] = t0;

		t0 = X[16] - X[17];
		X[16] = X[16] + X[17];
		X[17] = t0;

		t0 = X[18] - X[19];
		X[18] = X[18] + X[19];
		X[19] = t0;

		t0 = X[20] - X[21];
		X[20] = X[20] + X[21];
		X[21] = t0;

		t0 = X[22] - X[23];
		X[22] = X[22] + X[23];
		X[23] = t0;

		t0 = X[24] - X[25];
		X[24] = X[24] + X[25];
		X[25] = t0;

		t0 = X[26] - X[27];
		X[26] = X[26] + X[27];
		X[27] = t0;

		t0 = X[28] - X[29];
		X[28] = X[28] + X[29];
		X[29] = t0;

		t0 = X[30] - X[31];
		X[30] = X[30] + X[31];
		X[31] = t0;

		//output is shuffled now, i.e. it is ordered as:
		//0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31

		cuReIm3 F[32];

		int ker_baseidx = idx;

		F[0].x = (Kdiag[ker_baseidx].x * X[0].x) + (Kodiag[ker_baseidx].x * X[0].y) + (Kodiag[ker_baseidx].y * X[0].z);
		F[0].y = (Kodiag[ker_baseidx].x * X[0].x) + (Kdiag[ker_baseidx].y * X[0].y) + (Kodiag[ker_baseidx].z * X[0].z);
		F[0].z = (Kodiag[ker_baseidx].y * X[0].x) + (Kodiag[ker_baseidx].z * X[0].y) + (Kdiag[ker_baseidx].z * X[0].z);

		F[16].x = (Kdiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].z);
		F[16].y = (Kodiag[ker_baseidx + 1 * kerplanecount].x * X[16].x) + (Kdiag[ker_baseidx + 1 * kerplanecount].y * X[16].y) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);
		F[16].z = (Kodiag[ker_baseidx + 1 * kerplanecount].y * X[16].x) + (Kodiag[ker_baseidx + 1 * kerplanecount].z * X[16].y) + (Kdiag[ker_baseidx + 1 * kerplanecount].z * X[16].z);

		F[8].x = (Kdiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].z);
		F[8].y = (Kodiag[ker_baseidx + 2 * kerplanecount].x * X[8].x) + (Kdiag[ker_baseidx + 2 * kerplanecount].y * X[8].y) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);
		F[8].z = (Kodiag[ker_baseidx + 2 * kerplanecount].y * X[8].x) + (Kodiag[ker_baseidx + 2 * kerplanecount].z * X[8].y) + (Kdiag[ker_baseidx + 2 * kerplanecount].z * X[8].z);

		F[24].x = (Kdiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].z);
		F[24].y = (Kodiag[ker_baseidx + 3 * kerplanecount].x * X[24].x) + (Kdiag[ker_baseidx + 3 * kerplanecount].y * X[24].y) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);
		F[24].z = (Kodiag[ker_baseidx + 3 * kerplanecount].y * X[24].x) + (Kodiag[ker_baseidx + 3 * kerplanecount].z * X[24].y) + (Kdiag[ker_baseidx + 3 * kerplanecount].z * X[24].z);

		F[4].x = (Kdiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].z);
		F[4].y = (Kodiag[ker_baseidx + 4 * kerplanecount].x * X[4].x) + (Kdiag[ker_baseidx + 4 * kerplanecount].y * X[4].y) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);
		F[4].z = (Kodiag[ker_baseidx + 4 * kerplanecount].y * X[4].x) + (Kodiag[ker_baseidx + 4 * kerplanecount].z * X[4].y) + (Kdiag[ker_baseidx + 4 * kerplanecount].z * X[4].z);

		F[20].x = (Kdiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].z);
		F[20].y = (Kodiag[ker_baseidx + 5 * kerplanecount].x * X[20].x) + (Kdiag[ker_baseidx + 5 * kerplanecount].y * X[20].y) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);
		F[20].z = (Kodiag[ker_baseidx + 5 * kerplanecount].y * X[20].x) + (Kodiag[ker_baseidx + 5 * kerplanecount].z * X[20].y) + (Kdiag[ker_baseidx + 5 * kerplanecount].z * X[20].z);

		F[12].x = (Kdiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].z);
		F[12].y = (Kodiag[ker_baseidx + 6 * kerplanecount].x * X[12].x) + (Kdiag[ker_baseidx + 6 * kerplanecount].y * X[12].y) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);
		F[12].z = (Kodiag[ker_baseidx + 6 * kerplanecount].y * X[12].x) + (Kodiag[ker_baseidx + 6 * kerplanecount].z * X[12].y) + (Kdiag[ker_baseidx + 6 * kerplanecount].z * X[12].z);

		F[28].x = (Kdiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].z);
		F[28].y = (Kodiag[ker_baseidx + 7 * kerplanecount].x * X[28].x) + (Kdiag[ker_baseidx + 7 * kerplanecount].y * X[28].y) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);
		F[28].z = (Kodiag[ker_baseidx + 7 * kerplanecount].y * X[28].x) + (Kodiag[ker_baseidx + 7 * kerplanecount].z * X[28].y) + (Kdiag[ker_baseidx + 7 * kerplanecount].z * X[28].z);

		F[2].x = (Kdiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].z);
		F[2].y = (Kodiag[ker_baseidx + 8 * kerplanecount].x * X[2].x) + (Kdiag[ker_baseidx + 8 * kerplanecount].y * X[2].y) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);
		F[2].z = (Kodiag[ker_baseidx + 8 * kerplanecount].y * X[2].x) + (Kodiag[ker_baseidx + 8 * kerplanecount].z * X[2].y) + (Kdiag[ker_baseidx + 8 * kerplanecount].z * X[2].z);

		F[18].x = (Kdiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].z);
		F[18].y = (Kodiag[ker_baseidx + 9 * kerplanecount].x * X[18].x) + (Kdiag[ker_baseidx + 9 * kerplanecount].y * X[18].y) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);
		F[18].z = (Kodiag[ker_baseidx + 9 * kerplanecount].y * X[18].x) + (Kodiag[ker_baseidx + 9 * kerplanecount].z * X[18].y) + (Kdiag[ker_baseidx + 9 * kerplanecount].z * X[18].z);

		F[10].x = (Kdiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].z);
		F[10].y = (Kodiag[ker_baseidx + 10 * kerplanecount].x * X[10].x) + (Kdiag[ker_baseidx + 10 * kerplanecount].y * X[10].y) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);
		F[10].z = (Kodiag[ker_baseidx + 10 * kerplanecount].y * X[10].x) + (Kodiag[ker_baseidx + 10 * kerplanecount].z * X[10].y) + (Kdiag[ker_baseidx + 10 * kerplanecount].z * X[10].z);

		F[26].x = (Kdiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].z);
		F[26].y = (Kodiag[ker_baseidx + 11 * kerplanecount].x * X[26].x) + (Kdiag[ker_baseidx + 11 * kerplanecount].y * X[26].y) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);
		F[26].z = (Kodiag[ker_baseidx + 11 * kerplanecount].y * X[26].x) + (Kodiag[ker_baseidx + 11 * kerplanecount].z * X[26].y) + (Kdiag[ker_baseidx + 11 * kerplanecount].z * X[26].z);

		F[6].x = (Kdiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].z);
		F[6].y = (Kodiag[ker_baseidx + 12 * kerplanecount].x * X[6].x) + (Kdiag[ker_baseidx + 12 * kerplanecount].y * X[6].y) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);
		F[6].z = (Kodiag[ker_baseidx + 12 * kerplanecount].y * X[6].x) + (Kodiag[ker_baseidx + 12 * kerplanecount].z * X[6].y) + (Kdiag[ker_baseidx + 12 * kerplanecount].z * X[6].z);

		F[22].x = (Kdiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].z);
		F[22].y = (Kodiag[ker_baseidx + 13 * kerplanecount].x * X[22].x) + (Kdiag[ker_baseidx + 13 * kerplanecount].y * X[22].y) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);
		F[22].z = (Kodiag[ker_baseidx + 13 * kerplanecount].y * X[22].x) + (Kodiag[ker_baseidx + 13 * kerplanecount].z * X[22].y) + (Kdiag[ker_baseidx + 13 * kerplanecount].z * X[22].z);

		F[14].x = (Kdiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].z);
		F[14].y = (Kodiag[ker_baseidx + 14 * kerplanecount].x * X[14].x) + (Kdiag[ker_baseidx + 14 * kerplanecount].y * X[14].y) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);
		F[14].z = (Kodiag[ker_baseidx + 14 * kerplanecount].y * X[14].x) + (Kodiag[ker_baseidx + 14 * kerplanecount].z * X[14].y) + (Kdiag[ker_baseidx + 14 * kerplanecount].z * X[14].z);

		F[30].x = (Kdiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].z);
		F[30].y = (Kodiag[ker_baseidx + 15 * kerplanecount].x * X[30].x) + (Kdiag[ker_baseidx + 15 * kerplanecount].y * X[30].y) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);
		F[30].z = (Kodiag[ker_baseidx + 15 * kerplanecount].y * X[30].x) + (Kodiag[ker_baseidx + 15 * kerplanecount].z * X[30].y) + (Kdiag[ker_baseidx + 15 * kerplanecount].z * X[30].z);

		F[1].x = (Kdiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].z);
		F[1].y = (Kodiag[ker_baseidx + 16 * kerplanecount].x * X[1].x) + (Kdiag[ker_baseidx + 16 * kerplanecount].y * X[1].y) + (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);
		F[1].z = (Kodiag[ker_baseidx + 16 * kerplanecount].y * X[1].x) + (Kodiag[ker_baseidx + 16 * kerplanecount].z * X[1].y) + (Kdiag[ker_baseidx + 16 * kerplanecount].z * X[1].z);

		F[17].x = (Kdiag[ker_baseidx + 17 * kerplanecount].x * X[17].x) + (Kodiag[ker_baseidx + 17 * kerplanecount].x * X[17].y) + (Kodiag[ker_baseidx + 17 * kerplanecount].y * X[17].z);
		F[17].y = (Kodiag[ker_baseidx + 17 * kerplanecount].x * X[17].x) + (Kdiag[ker_baseidx + 17 * kerplanecount].y * X[17].y) + (Kodiag[ker_baseidx + 17 * kerplanecount].z * X[17].z);
		F[17].z = (Kodiag[ker_baseidx + 17 * kerplanecount].y * X[17].x) + (Kodiag[ker_baseidx + 17 * kerplanecount].z * X[17].y) + (Kdiag[ker_baseidx + 17 * kerplanecount].z * X[17].z);

		F[9].x = (Kdiag[ker_baseidx + 18 * kerplanecount].x * X[9].x) + (Kodiag[ker_baseidx + 18 * kerplanecount].x * X[9].y) + (Kodiag[ker_baseidx + 18 * kerplanecount].y * X[9].z);
		F[9].y = (Kodiag[ker_baseidx + 18 * kerplanecount].x * X[9].x) + (Kdiag[ker_baseidx + 18 * kerplanecount].y * X[9].y) + (Kodiag[ker_baseidx + 18 * kerplanecount].z * X[9].z);
		F[9].z = (Kodiag[ker_baseidx + 18 * kerplanecount].y * X[9].x) + (Kodiag[ker_baseidx + 18 * kerplanecount].z * X[9].y) + (Kdiag[ker_baseidx + 18 * kerplanecount].z * X[9].z);

		F[25].x = (Kdiag[ker_baseidx + 19 * kerplanecount].x * X[25].x) + (Kodiag[ker_baseidx + 19 * kerplanecount].x * X[25].y) + (Kodiag[ker_baseidx + 19 * kerplanecount].y * X[25].z);
		F[25].y = (Kodiag[ker_baseidx + 19 * kerplanecount].x * X[25].x) + (Kdiag[ker_baseidx + 19 * kerplanecount].y * X[25].y) + (Kodiag[ker_baseidx + 19 * kerplanecount].z * X[25].z);
		F[25].z = (Kodiag[ker_baseidx + 19 * kerplanecount].y * X[25].x) + (Kodiag[ker_baseidx + 19 * kerplanecount].z * X[25].y) + (Kdiag[ker_baseidx + 19 * kerplanecount].z * X[25].z);

		F[5].x = (Kdiag[ker_baseidx + 20 * kerplanecount].x * X[5].x) + (Kodiag[ker_baseidx + 20 * kerplanecount].x * X[5].y) + (Kodiag[ker_baseidx + 20 * kerplanecount].y * X[5].z);
		F[5].y = (Kodiag[ker_baseidx + 20 * kerplanecount].x * X[5].x) + (Kdiag[ker_baseidx + 20 * kerplanecount].y * X[5].y) + (Kodiag[ker_baseidx + 20 * kerplanecount].z * X[5].z);
		F[5].z = (Kodiag[ker_baseidx + 20 * kerplanecount].y * X[5].x) + (Kodiag[ker_baseidx + 20 * kerplanecount].z * X[5].y) + (Kdiag[ker_baseidx + 20 * kerplanecount].z * X[5].z);

		F[21].x = (Kdiag[ker_baseidx + 21 * kerplanecount].x * X[21].x) + (Kodiag[ker_baseidx + 21 * kerplanecount].x * X[21].y) + (Kodiag[ker_baseidx + 21 * kerplanecount].y * X[21].z);
		F[21].y = (Kodiag[ker_baseidx + 21 * kerplanecount].x * X[21].x) + (Kdiag[ker_baseidx + 21 * kerplanecount].y * X[21].y) + (Kodiag[ker_baseidx + 21 * kerplanecount].z * X[21].z);
		F[21].z = (Kodiag[ker_baseidx + 21 * kerplanecount].y * X[21].x) + (Kodiag[ker_baseidx + 21 * kerplanecount].z * X[21].y) + (Kdiag[ker_baseidx + 21 * kerplanecount].z * X[21].z);

		F[13].x = (Kdiag[ker_baseidx + 22 * kerplanecount].x * X[13].x) + (Kodiag[ker_baseidx + 22 * kerplanecount].x * X[13].y) + (Kodiag[ker_baseidx + 22 * kerplanecount].y * X[13].z);
		F[13].y = (Kodiag[ker_baseidx + 22 * kerplanecount].x * X[13].x) + (Kdiag[ker_baseidx + 22 * kerplanecount].y * X[13].y) + (Kodiag[ker_baseidx + 22 * kerplanecount].z * X[13].z);
		F[13].z = (Kodiag[ker_baseidx + 22 * kerplanecount].y * X[13].x) + (Kodiag[ker_baseidx + 22 * kerplanecount].z * X[13].y) + (Kdiag[ker_baseidx + 22 * kerplanecount].z * X[13].z);

		F[29].x = (Kdiag[ker_baseidx + 23 * kerplanecount].x * X[29].x) + (Kodiag[ker_baseidx + 23 * kerplanecount].x * X[29].y) + (Kodiag[ker_baseidx + 23 * kerplanecount].y * X[29].z);
		F[29].y = (Kodiag[ker_baseidx + 23 * kerplanecount].x * X[29].x) + (Kdiag[ker_baseidx + 23 * kerplanecount].y * X[29].y) + (Kodiag[ker_baseidx + 23 * kerplanecount].z * X[29].z);
		F[29].z = (Kodiag[ker_baseidx + 23 * kerplanecount].y * X[29].x) + (Kodiag[ker_baseidx + 23 * kerplanecount].z * X[29].y) + (Kdiag[ker_baseidx + 23 * kerplanecount].z * X[29].z);

		F[3].x = (Kdiag[ker_baseidx + 24 * kerplanecount].x * X[3].x) + (Kodiag[ker_baseidx + 24 * kerplanecount].x * X[3].y) + (Kodiag[ker_baseidx + 24 * kerplanecount].y * X[3].z);
		F[3].y = (Kodiag[ker_baseidx + 24 * kerplanecount].x * X[3].x) + (Kdiag[ker_baseidx + 24 * kerplanecount].y * X[3].y) + (Kodiag[ker_baseidx + 24 * kerplanecount].z * X[3].z);
		F[3].z = (Kodiag[ker_baseidx + 24 * kerplanecount].y * X[3].x) + (Kodiag[ker_baseidx + 24 * kerplanecount].z * X[3].y) + (Kdiag[ker_baseidx + 24 * kerplanecount].z * X[3].z);

		F[19].x = (Kdiag[ker_baseidx + 25 * kerplanecount].x * X[19].x) + (Kodiag[ker_baseidx + 25 * kerplanecount].x * X[19].y) + (Kodiag[ker_baseidx + 25 * kerplanecount].y * X[19].z);
		F[19].y = (Kodiag[ker_baseidx + 25 * kerplanecount].x * X[19].x) + (Kdiag[ker_baseidx + 25 * kerplanecount].y * X[19].y) + (Kodiag[ker_baseidx + 25 * kerplanecount].z * X[19].z);
		F[19].z = (Kodiag[ker_baseidx + 25 * kerplanecount].y * X[19].x) + (Kodiag[ker_baseidx + 25 * kerplanecount].z * X[19].y) + (Kdiag[ker_baseidx + 25 * kerplanecount].z * X[19].z);

		F[11].x = (Kdiag[ker_baseidx + 26 * kerplanecount].x * X[11].x) + (Kodiag[ker_baseidx + 26 * kerplanecount].x * X[11].y) + (Kodiag[ker_baseidx + 26 * kerplanecount].y * X[11].z);
		F[11].y = (Kodiag[ker_baseidx + 26 * kerplanecount].x * X[11].x) + (Kdiag[ker_baseidx + 26 * kerplanecount].y * X[11].y) + (Kodiag[ker_baseidx + 26 * kerplanecount].z * X[11].z);
		F[11].z = (Kodiag[ker_baseidx + 26 * kerplanecount].y * X[11].x) + (Kodiag[ker_baseidx + 26 * kerplanecount].z * X[11].y) + (Kdiag[ker_baseidx + 26 * kerplanecount].z * X[11].z);

		F[27].x = (Kdiag[ker_baseidx + 27 * kerplanecount].x * X[27].x) + (Kodiag[ker_baseidx + 27 * kerplanecount].x * X[27].y) + (Kodiag[ker_baseidx + 27 * kerplanecount].y * X[27].z);
		F[27].y = (Kodiag[ker_baseidx + 27 * kerplanecount].x * X[27].x) + (Kdiag[ker_baseidx + 27 * kerplanecount].y * X[27].y) + (Kodiag[ker_baseidx + 27 * kerplanecount].z * X[27].z);
		F[27].z = (Kodiag[ker_baseidx + 27 * kerplanecount].y * X[27].x) + (Kodiag[ker_baseidx + 27 * kerplanecount].z * X[27].y) + (Kdiag[ker_baseidx + 27 * kerplanecount].z * X[27].z);

		F[7].x = (Kdiag[ker_baseidx + 28 * kerplanecount].x * X[7].x) + (Kodiag[ker_baseidx + 28 * kerplanecount].x * X[7].y) + (Kodiag[ker_baseidx + 28 * kerplanecount].y * X[7].z);
		F[7].y = (Kodiag[ker_baseidx + 28 * kerplanecount].x * X[7].x) + (Kdiag[ker_baseidx + 28 * kerplanecount].y * X[7].y) + (Kodiag[ker_baseidx + 28 * kerplanecount].z * X[7].z);
		F[7].z = (Kodiag[ker_baseidx + 28 * kerplanecount].y * X[7].x) + (Kodiag[ker_baseidx + 28 * kerplanecount].z * X[7].y) + (Kdiag[ker_baseidx + 28 * kerplanecount].z * X[7].z);

		F[23].x = (Kdiag[ker_baseidx + 29 * kerplanecount].x * X[23].x) + (Kodiag[ker_baseidx + 29 * kerplanecount].x * X[23].y) + (Kodiag[ker_baseidx + 29 * kerplanecount].y * X[23].z);
		F[23].y = (Kodiag[ker_baseidx + 29 * kerplanecount].x * X[23].x) + (Kdiag[ker_baseidx + 29 * kerplanecount].y * X[23].y) + (Kodiag[ker_baseidx + 29 * kerplanecount].z * X[23].z);
		F[23].z = (Kodiag[ker_baseidx + 29 * kerplanecount].y * X[23].x) + (Kodiag[ker_baseidx + 29 * kerplanecount].z * X[23].y) + (Kdiag[ker_baseidx + 29 * kerplanecount].z * X[23].z);

		F[15].x = (Kdiag[ker_baseidx + 30 * kerplanecount].x * X[15].x) + (Kodiag[ker_baseidx + 30 * kerplanecount].x * X[15].y) + (Kodiag[ker_baseidx + 30 * kerplanecount].y * X[15].z);
		F[15].y = (Kodiag[ker_baseidx + 30 * kerplanecount].x * X[15].x) + (Kdiag[ker_baseidx + 30 * kerplanecount].y * X[15].y) + (Kodiag[ker_baseidx + 30 * kerplanecount].z * X[15].z);
		F[15].z = (Kodiag[ker_baseidx + 30 * kerplanecount].y * X[15].x) + (Kodiag[ker_baseidx + 30 * kerplanecount].z * X[15].y) + (Kdiag[ker_baseidx + 30 * kerplanecount].z * X[15].z);

		F[31].x = (Kdiag[ker_baseidx + 31 * kerplanecount].x * X[31].x) + (Kodiag[ker_baseidx + 31 * kerplanecount].x * X[31].y) + (Kodiag[ker_baseidx + 31 * kerplanecount].y * X[31].z);
		F[31].y = (Kodiag[ker_baseidx + 31 * kerplanecount].x * X[31].x) + (Kdiag[ker_baseidx + 31 * kerplanecount].y * X[31].y) + (Kodiag[ker_baseidx + 31 * kerplanecount].z * X[31].z);
		F[31].z = (Kodiag[ker_baseidx + 31 * kerplanecount].y * X[31].x) + (Kodiag[ker_baseidx + 31 * kerplanecount].z * X[31].y) + (Kdiag[ker_baseidx + 31 * kerplanecount].z * X[31].z);

		//accumulate result in output ready for ifft

		cuSx_out[idx] = (cuReIm)cuSx_out[idx] + F[0].x;
		cuSy_out[idx] = (cuReIm)cuSy_out[idx] + F[0].y;
		cuSz_out[idx] = (cuReIm)cuSz_out[idx] + F[0].z;

		cuSx_out[idx + planecount] = (cuReIm)cuSx_out[idx + planecount] + F[1].x;
		cuSy_out[idx + planecount] = (cuReIm)cuSy_out[idx + planecount] + F[1].y;
		cuSz_out[idx + planecount] = (cuReIm)cuSz_out[idx + planecount] + F[1].z;

		cuSx_out[idx + 2 * planecount] = (cuReIm)cuSx_out[idx + 2 * planecount] + F[2].x;
		cuSy_out[idx + 2 * planecount] = (cuReIm)cuSy_out[idx + 2 * planecount] + F[2].y;
		cuSz_out[idx + 2 * planecount] = (cuReIm)cuSz_out[idx + 2 * planecount] + F[2].z;

		cuSx_out[idx + 3 * planecount] = (cuReIm)cuSx_out[idx + 3 * planecount] + F[3].x;
		cuSy_out[idx + 3 * planecount] = (cuReIm)cuSy_out[idx + 3 * planecount] + F[3].y;
		cuSz_out[idx + 3 * planecount] = (cuReIm)cuSz_out[idx + 3 * planecount] + F[3].z;

		cuSx_out[idx + 4 * planecount] = (cuReIm)cuSx_out[idx + 4 * planecount] + F[4].x;
		cuSy_out[idx + 4 * planecount] = (cuReIm)cuSy_out[idx + 4 * planecount] + F[4].y;
		cuSz_out[idx + 4 * planecount] = (cuReIm)cuSz_out[idx + 4 * planecount] + F[4].z;

		cuSx_out[idx + 5 * planecount] = (cuReIm)cuSx_out[idx + 5 * planecount] + F[5].x;
		cuSy_out[idx + 5 * planecount] = (cuReIm)cuSy_out[idx + 5 * planecount] + F[5].y;
		cuSz_out[idx + 5 * planecount] = (cuReIm)cuSz_out[idx + 5 * planecount] + F[5].z;

		cuSx_out[idx + 6 * planecount] = (cuReIm)cuSx_out[idx + 6 * planecount] + F[6].x;
		cuSy_out[idx + 6 * planecount] = (cuReIm)cuSy_out[idx + 6 * planecount] + F[6].y;
		cuSz_out[idx + 6 * planecount] = (cuReIm)cuSz_out[idx + 6 * planecount] + F[6].z;

		cuSx_out[idx + 7 * planecount] = (cuReIm)cuSx_out[idx + 7 * planecount] + F[7].x;
		cuSy_out[idx + 7 * planecount] = (cuReIm)cuSy_out[idx + 7 * planecount] + F[7].y;
		cuSz_out[idx + 7 * planecount] = (cuReIm)cuSz_out[idx + 7 * planecount] + F[7].z;

		cuSx_out[idx + 8 * planecount] = (cuReIm)cuSx_out[idx + 8 * planecount] + F[8].x;
		cuSy_out[idx + 8 * planecount] = (cuReIm)cuSy_out[idx + 8 * planecount] + F[8].y;
		cuSz_out[idx + 8 * planecount] = (cuReIm)cuSz_out[idx + 8 * planecount] + F[8].z;

		cuSx_out[idx + 9 * planecount] = (cuReIm)cuSx_out[idx + 9 * planecount] + F[9].x;
		cuSy_out[idx + 9 * planecount] = (cuReIm)cuSy_out[idx + 9 * planecount] + F[9].y;
		cuSz_out[idx + 9 * planecount] = (cuReIm)cuSz_out[idx + 9 * planecount] + F[9].z;

		cuSx_out[idx + 10 * planecount] = (cuReIm)cuSx_out[idx + 10 * planecount] + F[10].x;
		cuSy_out[idx + 10 * planecount] = (cuReIm)cuSy_out[idx + 10 * planecount] + F[10].y;
		cuSz_out[idx + 10 * planecount] = (cuReIm)cuSz_out[idx + 10 * planecount] + F[10].z;

		cuSx_out[idx + 11 * planecount] = (cuReIm)cuSx_out[idx + 11 * planecount] + F[11].x;
		cuSy_out[idx + 11 * planecount] = (cuReIm)cuSy_out[idx + 11 * planecount] + F[11].y;
		cuSz_out[idx + 11 * planecount] = (cuReIm)cuSz_out[idx + 11 * planecount] + F[11].z;

		cuSx_out[idx + 12 * planecount] = (cuReIm)cuSx_out[idx + 12 * planecount] + F[12].x;
		cuSy_out[idx + 12 * planecount] = (cuReIm)cuSy_out[idx + 12 * planecount] + F[12].y;
		cuSz_out[idx + 12 * planecount] = (cuReIm)cuSz_out[idx + 12 * planecount] + F[12].z;

		cuSx_out[idx + 13 * planecount] = (cuReIm)cuSx_out[idx + 13 * planecount] + F[13].x;
		cuSy_out[idx + 13 * planecount] = (cuReIm)cuSy_out[idx + 13 * planecount] + F[13].y;
		cuSz_out[idx + 13 * planecount] = (cuReIm)cuSz_out[idx + 13 * planecount] + F[13].z;

		cuSx_out[idx + 14 * planecount] = (cuReIm)cuSx_out[idx + 14 * planecount] + F[14].x;
		cuSy_out[idx + 14 * planecount] = (cuReIm)cuSy_out[idx + 14 * planecount] + F[14].y;
		cuSz_out[idx + 14 * planecount] = (cuReIm)cuSz_out[idx + 14 * planecount] + F[14].z;

		cuSx_out[idx + 15 * planecount] = (cuReIm)cuSx_out[idx + 15 * planecount] + F[15].x;
		cuSy_out[idx + 15 * planecount] = (cuReIm)cuSy_out[idx + 15 * planecount] + F[15].y;
		cuSz_out[idx + 15 * planecount] = (cuReIm)cuSz_out[idx + 15 * planecount] + F[15].z;

		cuSx_out[idx + 16 * planecount] = (cuReIm)cuSx_out[idx + 16 * planecount] + F[16].x;
		cuSy_out[idx + 16 * planecount] = (cuReIm)cuSy_out[idx + 16 * planecount] + F[16].y;
		cuSz_out[idx + 16 * planecount] = (cuReIm)cuSz_out[idx + 16 * planecount] + F[16].z;

		cuSx_out[idx + 17 * planecount] = (cuReIm)cuSx_out[idx + 17 * planecount] + F[17].x;
		cuSy_out[idx + 17 * planecount] = (cuReIm)cuSy_out[idx + 17 * planecount] + F[17].y;
		cuSz_out[idx + 17 * planecount] = (cuReIm)cuSz_out[idx + 17 * planecount] + F[17].z;

		cuSx_out[idx + 18 * planecount] = (cuReIm)cuSx_out[idx + 18 * planecount] + F[18].x;
		cuSy_out[idx + 18 * planecount] = (cuReIm)cuSy_out[idx + 18 * planecount] + F[18].y;
		cuSz_out[idx + 18 * planecount] = (cuReIm)cuSz_out[idx + 18 * planecount] + F[18].z;

		cuSx_out[idx + 19 * planecount] = (cuReIm)cuSx_out[idx + 19 * planecount] + F[19].x;
		cuSy_out[idx + 19 * planecount] = (cuReIm)cuSy_out[idx + 19 * planecount] + F[19].y;
		cuSz_out[idx + 19 * planecount] = (cuReIm)cuSz_out[idx + 19 * planecount] + F[19].z;

		cuSx_out[idx + 20 * planecount] = (cuReIm)cuSx_out[idx + 20 * planecount] + F[20].x;
		cuSy_out[idx + 20 * planecount] = (cuReIm)cuSy_out[idx + 20 * planecount] + F[20].y;
		cuSz_out[idx + 20 * planecount] = (cuReIm)cuSz_out[idx + 20 * planecount] + F[20].z;

		cuSx_out[idx + 21 * planecount] = (cuReIm)cuSx_out[idx + 21 * planecount] + F[21].x;
		cuSy_out[idx + 21 * planecount] = (cuReIm)cuSy_out[idx + 21 * planecount] + F[21].y;
		cuSz_out[idx + 21 * planecount] = (cuReIm)cuSz_out[idx + 21 * planecount] + F[21].z;

		cuSx_out[idx + 22 * planecount] = (cuReIm)cuSx_out[idx + 22 * planecount] + F[22].x;
		cuSy_out[idx + 22 * planecount] = (cuReIm)cuSy_out[idx + 22 * planecount] + F[22].y;
		cuSz_out[idx + 22 * planecount] = (cuReIm)cuSz_out[idx + 22 * planecount] + F[22].z;

		cuSx_out[idx + 23 * planecount] = (cuReIm)cuSx_out[idx + 23 * planecount] + F[23].x;
		cuSy_out[idx + 23 * planecount] = (cuReIm)cuSy_out[idx + 23 * planecount] + F[23].y;
		cuSz_out[idx + 23 * planecount] = (cuReIm)cuSz_out[idx + 23 * planecount] + F[23].z;

		cuSx_out[idx + 24 * planecount] = (cuReIm)cuSx_out[idx + 24 * planecount] + F[24].x;
		cuSy_out[idx + 24 * planecount] = (cuReIm)cuSy_out[idx + 24 * planecount] + F[24].y;
		cuSz_out[idx + 24 * planecount] = (cuReIm)cuSz_out[idx + 24 * planecount] + F[24].z;

		cuSx_out[idx + 25 * planecount] = (cuReIm)cuSx_out[idx + 25 * planecount] + F[25].x;
		cuSy_out[idx + 25 * planecount] = (cuReIm)cuSy_out[idx + 25 * planecount] + F[25].y;
		cuSz_out[idx + 25 * planecount] = (cuReIm)cuSz_out[idx + 25 * planecount] + F[25].z;

		cuSx_out[idx + 26 * planecount] = (cuReIm)cuSx_out[idx + 26 * planecount] + F[26].x;
		cuSy_out[idx + 26 * planecount] = (cuReIm)cuSy_out[idx + 26 * planecount] + F[26].y;
		cuSz_out[idx + 26 * planecount] = (cuReIm)cuSz_out[idx + 26 * planecount] + F[26].z;

		cuSx_out[idx + 27 * planecount] = (cuReIm)cuSx_out[idx + 27 * planecount] + F[27].x;
		cuSy_out[idx + 27 * planecount] = (cuReIm)cuSy_out[idx + 27 * planecount] + F[27].y;
		cuSz_out[idx + 27 * planecount] = (cuReIm)cuSz_out[idx + 27 * planecount] + F[27].z;

		cuSx_out[idx + 28 * planecount] = (cuReIm)cuSx_out[idx + 28 * planecount] + F[28].x;
		cuSy_out[idx + 28 * planecount] = (cuReIm)cuSy_out[idx + 28 * planecount] + F[28].y;
		cuSz_out[idx + 28 * planecount] = (cuReIm)cuSz_out[idx + 28 * planecount] + F[28].z;

		cuSx_out[idx + 29 * planecount] = (cuReIm)cuSx_out[idx + 29 * planecount] + F[29].x;
		cuSy_out[idx + 29 * planecount] = (cuReIm)cuSy_out[idx + 29 * planecount] + F[29].y;
		cuSz_out[idx + 29 * planecount] = (cuReIm)cuSz_out[idx + 29 * planecount] + F[29].z;

		cuSx_out[idx + 30 * planecount] = (cuReIm)cuSx_out[idx + 30 * planecount] + F[30].x;
		cuSy_out[idx + 30 * planecount] = (cuReIm)cuSy_out[idx + 30 * planecount] + F[30].y;
		cuSz_out[idx + 30 * planecount] = (cuReIm)cuSz_out[idx + 30 * planecount] + F[30].z;

		cuSx_out[idx + 31 * planecount] = (cuReIm)cuSx_out[idx + 31 * planecount] + F[31].x;
		cuSy_out[idx + 31 * planecount] = (cuReIm)cuSy_out[idx + 31 * planecount] + F[31].y;
		cuSz_out[idx + 31 * planecount] = (cuReIm)cuSz_out[idx + 31 * planecount] + F[31].z;

#undef a
#undef b
#undef c
#undef d
#undef e
#undef f
#undef g
	}
}

//N = (N.x/2 + 1, N.y, 4)
//IFFT part
__global__ void cu_MultiDemag_ConvProd_q2D_4_IFFT(
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 4, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	if (idx < planecount) {

		//inverse z-axis fft (but without division by 4). Also only keep first 2 points.

		cuReIm3 F0, F1, F2, F3;
		F0.x = cuSx[idx]; F0.y = cuSy[idx]; F0.z = cuSz[idx];
		F1.x = cuSx[idx + planecount]; F1.y = cuSy[idx + planecount]; F1.z = cuSz[idx + planecount];
		F2.x = cuSx[idx + 2 * planecount]; F2.y = cuSy[idx + 2 * planecount]; F2.z = cuSz[idx + 2 * planecount];
		F3.x = cuSx[idx + 3 * planecount]; F3.y = cuSy[idx + 3 * planecount]; F3.z = cuSz[idx + 3 * planecount];

		cuSx[idx] = F0.x + F1.x + F2.x + F3.x;
		cuSy[idx] = F0.y + F1.y + F2.y + F3.y;
		cuSz[idx] = F0.z + F1.z + F2.z + F3.z;

		cuReIm3 F1c = !F1;
		cuReIm3 F3c = !F3;

		cuSx[idx + planecount] = F0.x + F1c.x - F2.x - F3c.x;
		cuSy[idx + planecount] = F0.y + F1c.y - F2.y - F3c.y;
		cuSz[idx + planecount] = F0.z + F1c.z - F2.z - F3c.z;
	}
}

//N = (N.x/2 + 1, N.y, 8)
//IFFT part
__global__ void cu_MultiDemag_ConvProd_q2D_8_IFFT(
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 8, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	if (idx < planecount) {

		cuReIm3 F0, F1, F2, F3, F4, F5, F6, F7;
		F0.x = cuSx[idx]; F0.y = cuSy[idx]; F0.z = cuSz[idx];
		F1.x = cuSx[idx + planecount]; F1.y = cuSy[idx + planecount]; F1.z = cuSz[idx + planecount];
		F2.x = cuSx[idx + 2 * planecount]; F2.y = cuSy[idx + 2 * planecount]; F2.z = cuSz[idx + 2 * planecount];
		F3.x = cuSx[idx + 3 * planecount]; F3.y = cuSy[idx + 3 * planecount]; F3.z = cuSz[idx + 3 * planecount];
		F4.x = cuSx[idx + 4 * planecount]; F4.y = cuSy[idx + 4 * planecount]; F4.z = cuSz[idx + 4 * planecount];
		F5.x = cuSx[idx + 5 * planecount]; F5.y = cuSy[idx + 5 * planecount]; F5.z = cuSz[idx + 5 * planecount];
		F6.x = cuSx[idx + 6 * planecount]; F6.y = cuSy[idx + 6 * planecount]; F6.z = cuSz[idx + 6 * planecount];
		F7.x = cuSx[idx + 7 * planecount]; F7.y = cuSy[idx + 7 * planecount]; F7.z = cuSz[idx + 7 * planecount];

#define a (cuBReal)0.7071067811865

		//inverse z-axis fft (but without division by 8). Also only keep first 4 points.

		//Radix-2 step
		cuReIm3 X0 = F0 + F1;
		cuReIm3 X1 = F0 - F1;

		cuReIm3 X2 = F2 + F3;
		cuReIm3 X3 = F2 - F3;

		cuReIm3 X4 = F4 + F5;
		cuReIm3 X5 = F4 - F5;

		cuReIm3 X6 = F6 + F7;
		cuReIm3 X7 = F6 - F7;

		//Radix-4 step
		cuReIm3 t0 = X0 + X2;
		cuReIm3 t1 = X0 - X2;
		cuReIm3 t2 = X4 + X6;
		cuReIm3 t3 = !(X6 - X4);

		X0 = (t0 + t2);
		X2 = (t1 - t3);

		t0 = X1 + !X3;
		t1 = X1 - !X3;
		t2 = X5 * cuReIm(a, a) + X7 * cuReIm(-a, a);
		t3 = X7 * cuReIm(-a, -a) - X5 * cuReIm(-a, a);

		X1 = (t0 + t2);
		X3 = (t1 - t3);

		cuSx[idx] = X0.x;
		cuSy[idx] = X0.y;
		cuSz[idx] = X0.z;

		cuSx[idx + planecount] = X1.x;
		cuSy[idx + planecount] = X1.y;
		cuSz[idx + planecount] = X1.z;

		cuSx[idx + 2 * planecount] = X2.x;
		cuSy[idx + 2 * planecount] = X2.y;
		cuSz[idx + 2 * planecount] = X2.z;

		cuSx[idx + 3 * planecount] = X3.x;
		cuSy[idx + 3 * planecount] = X3.y;
		cuSz[idx + 3 * planecount] = X3.z;

#undef a
	}
}

//N = (N.x/2 + 1, N.y, 16)
//IFFT part
__global__ void cu_MultiDemag_ConvProd_q2D_16_IFFT(
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 16, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	if (idx < planecount) {

		cuReIm3 F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15;
		F0.x = cuSx[idx]; F0.y = cuSy[idx]; F0.z = cuSz[idx];
		F1.x = cuSx[idx + planecount]; F1.y = cuSy[idx + planecount]; F1.z = cuSz[idx + planecount];
		F2.x = cuSx[idx + 2 * planecount]; F2.y = cuSy[idx + 2 * planecount]; F2.z = cuSz[idx + 2 * planecount];
		F3.x = cuSx[idx + 3 * planecount]; F3.y = cuSy[idx + 3 * planecount]; F3.z = cuSz[idx + 3 * planecount];
		F4.x = cuSx[idx + 4 * planecount]; F4.y = cuSy[idx + 4 * planecount]; F4.z = cuSz[idx + 4 * planecount];
		F5.x = cuSx[idx + 5 * planecount]; F5.y = cuSy[idx + 5 * planecount]; F5.z = cuSz[idx + 5 * planecount];
		F6.x = cuSx[idx + 6 * planecount]; F6.y = cuSy[idx + 6 * planecount]; F6.z = cuSz[idx + 6 * planecount];
		F7.x = cuSx[idx + 7 * planecount]; F7.y = cuSy[idx + 7 * planecount]; F7.z = cuSz[idx + 7 * planecount];
		F8.x = cuSx[idx + 8 * planecount]; F8.y = cuSy[idx + 8 * planecount]; F8.z = cuSz[idx + 8 * planecount];
		F9.x = cuSx[idx + 9 * planecount]; F9.y = cuSy[idx + 9 * planecount]; F9.z = cuSz[idx + 9 * planecount];
		F10.x = cuSx[idx + 10 * planecount]; F10.y = cuSy[idx + 10 * planecount]; F10.z = cuSz[idx + 10 * planecount];
		F11.x = cuSx[idx + 11 * planecount]; F11.y = cuSy[idx + 11 * planecount]; F11.z = cuSz[idx + 11 * planecount];
		F12.x = cuSx[idx + 12 * planecount]; F12.y = cuSy[idx + 12 * planecount]; F12.z = cuSz[idx + 12 * planecount];
		F13.x = cuSx[idx + 13 * planecount]; F13.y = cuSy[idx + 13 * planecount]; F13.z = cuSz[idx + 13 * planecount];
		F14.x = cuSx[idx + 14 * planecount]; F14.y = cuSy[idx + 14 * planecount]; F14.z = cuSz[idx + 14 * planecount];
		F15.x = cuSx[idx + 15 * planecount]; F15.y = cuSy[idx + 15 * planecount]; F15.z = cuSz[idx + 15 * planecount];

		//inverse z-axis fft (but without division by 16). Also only keep first 8 points.

#define a	(cuBReal)9.238795325113E-01
#define b	(cuBReal)3.826834323651E-01
#define c	(cuBReal)7.071067811865E-01

		//First stage
		cuReIm3 t0 = F0 + F1;
		cuReIm3 t1 = F0 - F1;
		cuReIm3 t2 = F2 + F3;
		cuReIm3 t3 = !(F3 - F2);

		cuReIm3 X0 = t0 + t2;
		cuReIm3 X1 = t1 - t3;
		cuReIm3 X2 = t0 - t2;
		cuReIm3 X3 = t1 + t3;

		t0 = F4 + F5;
		t1 = F4 - F5;
		t2 = F6 + F7;
		t3 = !(F7 - F6);

		cuReIm3 X4 = t0 + t2;
		cuReIm3 X5 = t1 - t3;
		cuReIm3 X6 = t0 - t2;
		cuReIm3 X7 = t1 + t3;

		t0 = F8 + F9;
		t1 = F8 - F9;
		t2 = F10 + F11;
		t3 = !(F11 - F10);

		cuReIm3 X8 = t0 + t2;
		cuReIm3 X9 = t1 - t3;
		cuReIm3 X10 = t0 - t2;
		cuReIm3 X11 = t1 + t3;

		t0 = F12 + F13;
		t1 = F12 - F13;
		t2 = F14 + F15;
		t3 = !(F15 - F14);

		cuReIm3 X12 = t0 + t2;
		cuReIm3 X13 = t1 - t3;
		cuReIm3 X14 = t0 - t2;
		cuReIm3 X15 = t1 + t3;

		//Second stage

		t0 = X0 + X4;
		t1 = X0 - X4;
		t2 = X8 + X12;
		t3 = !(X12 - X8);

		X0 = t0 + t2;
		X4 = t1 - t3;

		t0 = X1 + X5 * cuReIm(c, c);
		t1 = X1 - X5 * cuReIm(c, c);
		t2 = X9 * cuReIm(a, b) + X13 * cuReIm(b, a);
		t3 = (X13 * cuReIm(-a, b) - X9 * cuReIm(-b, a));

		X1 = t0 + t2;
		X5 = t1 - t3;

		t0 = X2 + !X6;
		t1 = X2 - !X6;
		t2 = X10 * cuReIm(c, c) + X14 * cuReIm(-c, c);
		t3 = (X14 * cuReIm(-c, -c) - X10 * cuReIm(-c, c));

		X2 = t0 + t2;
		X6 = t1 - t3;

		t0 = X3 + X7 * cuReIm(-c, c);
		t1 = X3 - X7 * cuReIm(-c, c);
		t2 = X11 * cuReIm(b, a) + X15 * cuReIm(-a, -b);
		t3 = (X15 * cuReIm(b, -a) - X11 * cuReIm(-a, b));

		X3 = t0 + t2;
		X7 = t1 - t3;

		cuSx[idx] = X0.x;
		cuSy[idx] = X0.y;
		cuSz[idx] = X0.z;
		cuSx[idx + 4 * planecount] = X4.x;
		cuSy[idx + 4 * planecount] = X4.y;
		cuSz[idx + 4 * planecount] = X4.z;

		cuSx[idx + 1 * planecount] = X1.x;
		cuSy[idx + 1 * planecount] = X1.y;
		cuSz[idx + 1 * planecount] = X1.z;
		cuSx[idx + 5 * planecount] = X5.x;
		cuSy[idx + 5 * planecount] = X5.y;
		cuSz[idx + 5 * planecount] = X5.z;

		cuSx[idx + 2 * planecount] = X2.x;
		cuSy[idx + 2 * planecount] = X2.y;
		cuSz[idx + 2 * planecount] = X2.z;
		cuSx[idx + 6 * planecount] = X6.x;
		cuSy[idx + 6 * planecount] = X6.y;
		cuSz[idx + 6 * planecount] = X6.z;

		cuSx[idx + 3 * planecount] = X3.x;
		cuSy[idx + 3 * planecount] = X3.y;
		cuSz[idx + 3 * planecount] = X3.z;
		cuSx[idx + 7 * planecount] = X7.x;
		cuSy[idx + 7 * planecount] = X7.y;
		cuSz[idx + 7 * planecount] = X7.z;

#undef a
#undef b
#undef c
	}
}

//N = (N.x/2 + 1, N.y, 32)
//IFFT part
__global__ void cu_MultiDemag_ConvProd_q2D_32_IFFT(
	cuBComplex* cuSx, cuBComplex* cuSy, cuBComplex* cuSz,
	cuSZ3& N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//N.z = 32, and this kernel was called with (N.x/2 + 1) * N.y points: handle all z points in one go
	int planecount = (N.x / 2 + 1) * N.y;

	if (idx < planecount) {

		//input data
#define x(n)	(cuReIm3(cuSx[idx + (n) * planecount], cuSy[idx + (n) * planecount], cuSz[idx + (n) * planecount]))

		//no performance gain to be had from setting these as X0, X1, ... etc.
		//unrolling loops does make a slight difference though - probably last case for which you want to unroll loops
		cuReIm3 X[32];
		cuReIm3 t0, t1, t2, t3;

		cuReIm3 F[32];
		F[0] = x(0);
		F[1] = x(1);
		F[2] = x(2);
		F[3] = x(3);
		F[4] = x(4);
		F[5] = x(5);
		F[6] = x(6);
		F[7] = x(7);
		F[8] = x(8);
		F[9] = x(9);
		F[10] = x(10);
		F[11] = x(11);
		F[12] = x(12);
		F[13] = x(13);
		F[14] = x(14);
		F[15] = x(15);
		F[16] = x(16);
		F[17] = x(17);
		F[18] = x(18);
		F[19] = x(19);
		F[20] = x(20);
		F[21] = x(21);
		F[22] = x(22);
		F[23] = x(23);
		F[24] = x(24);
		F[25] = x(25);
		F[26] = x(26);
		F[27] = x(27);
		F[28] = x(28);
		F[29] = x(29);
		F[30] = x(30);
		F[31] = x(31);

		//inverse z-axis fft (but without division by 32). Also only keep first 16 points.

#define a	(cuBReal)0.980785280403230
#define b	(cuBReal)0.195090322016128
#define c	(cuBReal)0.923879532511287
#define d	(cuBReal)0.382683432365090
#define e	(cuBReal)0.831469612302545
#define f	(cuBReal)0.555570233019602
#define g	(cuBReal)0.707106781186548

		//radix-2 stage to start
		X[0] = F[0] + F[1];
		X[1] = F[0] - F[1];

		X[2] = F[2] + F[3];
		X[3] = F[2] - F[3];

		X[4] = F[4] + F[5];
		X[5] = F[4] - F[5];

		X[6] = F[6] + F[7];
		X[7] = F[6] - F[7];

		X[8] = F[8] + F[9];
		X[9] = F[8] - F[9];

		X[10] = F[10] + F[11];
		X[11] = F[10] - F[11];

		X[12] = F[12] + F[13];
		X[13] = F[12] - F[13];

		X[14] = F[14] + F[15];
		X[15] = F[14] - F[15];

		X[16] = F[16] + F[17];
		X[17] = F[16] - F[17];

		X[18] = F[18] + F[19];
		X[19] = F[18] - F[19];

		X[20] = F[20] + F[21];
		X[21] = F[20] - F[21];

		X[22] = F[22] + F[23];
		X[23] = F[22] - F[23];

		X[24] = F[24] + F[25];
		X[25] = F[24] - F[25];

		X[26] = F[26] + F[27];
		X[27] = F[26] - F[27];

		X[28] = F[28] + F[29];
		X[29] = F[28] - F[29];

		X[30] = F[30] + F[31];
		X[31] = F[30] - F[31];

		//First radix-4 stage

		//j = 0 (no multiplications)
		t0 = (X[0] + X[2]);
		t1 = (X[0] - X[2]);
		t2 = (X[4] + X[6]);
		t3 = !(X[6] - X[4]);

		X[0] = t0 + t2;
		X[2] = t1 - t3;
		X[4] = t0 - t2;
		X[6] = t1 + t3;

		t0 = (X[8] + X[10]);
		t1 = (X[8] - X[10]);
		t2 = (X[12] + X[14]);
		t3 = !(X[14] - X[12]);

		X[8] = t0 + t2;
		X[10] = t1 - t3;
		X[12] = t0 - t2;
		X[14] = t1 + t3;

		t0 = (X[16] + X[18]);
		t1 = (X[16] - X[18]);
		t2 = (X[20] + X[22]);
		t3 = !(X[22] - X[20]);

		X[16] = t0 + t2;
		X[18] = t1 - t3;
		X[20] = t0 - t2;
		X[22] = t1 + t3;

		t0 = (X[24] + X[26]);
		t1 = (X[24] - X[26]);
		t2 = (X[28] + X[30]);
		t3 = !(X[30] - X[28]);

		X[24] = t0 + t2;
		X[26] = t1 - t3;
		X[28] = t0 - t2;
		X[30] = t1 + t3;

		//j = 1
		t0 = (X[1] + !X[3]);
		t1 = (X[1] - !X[3]);
		t2 = (X[5] * cuReIm(g, g) + X[7] * cuReIm(-g, g));
		t3 = (X[7] * cuReIm(-g, -g) - X[5] * cuReIm(-g, g));

		X[1] = t0 + t2;
		X[3] = t1 - t3;
		X[5] = t0 - t2;
		X[7] = t1 + t3;

		t0 = (X[9] + !X[11]);
		t1 = (X[9] - !X[11]);
		t2 = (X[13] * cuReIm(g, g) + X[15] * cuReIm(-g, g));
		t3 = (X[15] * cuReIm(-g, -g) - X[13] * cuReIm(-g, g));

		X[9] = t0 + t2;
		X[11] = t1 - t3;
		X[13] = t0 - t2;
		X[15] = t1 + t3;

		t0 = (X[17] + !X[19]);
		t1 = (X[17] - !X[19]);
		t2 = (X[21] * cuReIm(g, g) + X[23] * cuReIm(-g, g));
		t3 = (X[23] * cuReIm(-g, -g) - X[21] * cuReIm(-g, g));

		X[17] = t0 + t2;
		X[19] = t1 - t3;
		X[21] = t0 - t2;
		X[23] = t1 + t3;

		t0 = (X[25] + !X[27]);
		t1 = (X[25] - !X[27]);
		t2 = (X[29] * cuReIm(g, g) + X[31] * cuReIm(-g, g));
		t3 = (X[31] * cuReIm(-g, -g) - X[29] * cuReIm(-g, g));

		X[25] = t0 + t2;
		X[27] = t1 - t3;
		X[29] = t0 - t2;
		X[31] = t1 + t3;

		//Output radix-4 stage (truncated output)
		//j = 0
		t0 = (X[0] + X[8]);
		t1 = (X[0] - X[8]);
		t2 = (X[16] + X[24]);
		t3 = !(X[24] - X[16]);

		cuReIm3 l = t0 + t2;
		cuReIm3 h = t1 - t3;

		cuSx[idx] = l.x;
		cuSy[idx] = l.y;
		cuSz[idx] = l.z;
		cuSx[idx + 8 * planecount] = h.x;
		cuSy[idx + 8 * planecount] = h.y;
		cuSz[idx + 8 * planecount] = h.z;

		//j = 1
		t0 = (X[1] + X[9] * cuReIm(c, d));
		t1 = (X[1] - X[9] * cuReIm(c, d));
		t2 = (X[17] * cuReIm(a, b) + X[25] * cuReIm(e, f));
		t3 = (X[25] * cuReIm(-f, e) - X[17] * cuReIm(-b, a));

		l = t0 + t2;
		h = t1 - t3;

		cuSx[idx + planecount] = l.x;
		cuSy[idx + planecount] = l.y;
		cuSz[idx + planecount] = l.z;
		cuSx[idx + 9 * planecount] = h.x;
		cuSy[idx + 9 * planecount] = h.y;
		cuSz[idx + 9 * planecount] = h.z;

		//j = 2
		t0 = (X[2] + X[10] * cuReIm(g, g));
		t1 = (X[2] - X[10] * cuReIm(g, g));
		t2 = (X[18] * cuReIm(c, d) + X[26] * cuReIm(d, c));
		t3 = (X[26] * cuReIm(-c, d) - X[18] * cuReIm(-d, c));

		l = t0 + t2;
		h = t1 - t3;

		cuSx[idx + 2 * planecount] = l.x;
		cuSy[idx + 2 * planecount] = l.y;
		cuSz[idx + 2 * planecount] = l.z;
		cuSx[idx + 10 * planecount] = h.x;
		cuSy[idx + 10 * planecount] = h.y;
		cuSz[idx + 10 * planecount] = h.z;

		//j = 3
		t0 = (X[3] + X[11] * cuReIm(d, c));
		t1 = (X[3] - X[11] * cuReIm(d, c));
		t2 = (X[19] * cuReIm(e, f) + X[27] * cuReIm(-b, a));
		t3 = (X[27] * cuReIm(-a, -b) - X[19] * cuReIm(-f, e));

		l = t0 + t2;
		h = t1 - t3;

		cuSx[idx + 3 * planecount] = l.x;
		cuSy[idx + 3 * planecount] = l.y;
		cuSz[idx + 3 * planecount] = l.z;
		cuSx[idx + 11 * planecount] = h.x;
		cuSy[idx + 11 * planecount] = h.y;
		cuSz[idx + 11 * planecount] = h.z;

		//j = 4
		t0 = (X[4] + !X[12]);
		t1 = (X[4] - !X[12]);
		t2 = (X[20] * cuReIm(g, g) + X[28] * cuReIm(-g, g));
		t3 = (X[28] * cuReIm(-g, -g) - X[20] * cuReIm(-g, g));

		l = t0 + t2;
		h = t1 - t3;

		cuSx[idx + 4 * planecount] = l.x;
		cuSy[idx + 4 * planecount] = l.y;
		cuSz[idx + 4 * planecount] = l.z;
		cuSx[idx + 12 * planecount] = h.x;
		cuSy[idx + 12 * planecount] = h.y;
		cuSz[idx + 12 * planecount] = h.z;

		//j = 5
		t0 = (X[5] + X[13] * cuReIm(-d, c));
		t1 = (X[5] - X[13] * cuReIm(-d, c));
		t2 = (X[21] * cuReIm(f, e) + X[29] * cuReIm(-a, b));
		t3 = (X[29] * cuReIm(-b, -a) - X[21] * cuReIm(-e, f));

		l = t0 + t2;
		h = t1 - t3;

		cuSx[idx + 5 * planecount] = l.x;
		cuSy[idx + 5 * planecount] = l.y;
		cuSz[idx + 5 * planecount] = l.z;
		cuSx[idx + 13 * planecount] = h.x;
		cuSy[idx + 13 * planecount] = h.y;
		cuSz[idx + 13 * planecount] = h.z;

		//j = 6
		t0 = (X[6] + X[14] * cuReIm(-g, g));
		t1 = (X[6] - X[14] * cuReIm(-g, g));
		t2 = (X[22] * cuReIm(d, c) + X[30] * cuReIm(-c, -d));
		t3 = (X[30] * cuReIm(d, -c) - X[22] * cuReIm(-c, d));

		l = t0 + t2;
		h = t1 - t3;

		cuSx[idx + 6 * planecount] = l.x;
		cuSy[idx + 6 * planecount] = l.y;
		cuSz[idx + 6 * planecount] = l.z;
		cuSx[idx + 14 * planecount] = h.x;
		cuSy[idx + 14 * planecount] = h.y;
		cuSz[idx + 14 * planecount] = h.z;

		//j = 7
		t0 = (X[7] + X[15] * cuReIm(-c, d));
		t1 = (X[7] - X[15] * cuReIm(-c, d));
		t2 = (X[23] * cuReIm(b, a) + X[31] * cuReIm(-f, -e));
		t3 = (X[31] * cuReIm(e, -f) - X[23] * cuReIm(-a, b));

		l = t0 + t2;
		h = t1 - t3;

		cuSx[idx + 7 * planecount] = l.x;
		cuSy[idx + 7 * planecount] = l.y;
		cuSz[idx + 7 * planecount] = l.z;
		cuSx[idx + 15 * planecount] = h.x;
		cuSy[idx + 15 * planecount] = h.y;
		cuSz[idx + 15 * planecount] = h.z;

#undef a
#undef b
#undef c
#undef d
#undef e
#undef f
#undef g
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// LAUNCHERS

//-------------------------- RUN-TIME KERNEL MULTIPLICATION - MULTIPLE INPUTS TO SINGLE OUTPUT

void DemagKernelCollectionCUDA::KernelMultiplication_2D(
	std::vector<cu_arr<cuBComplex>*>& Incol_x, std::vector<cu_arr<cuBComplex>*>& Incol_y, std::vector<cu_arr<cuBComplex>*>& Incol_z,
	cu_arr<cuBComplex>& Out_x, cu_arr<cuBComplex>& Out_y, cu_arr<cuBComplex>& Out_z)
{
	//first compute the self contribution -> this sets Out
	if (transpose_xy) {

		cu_KernelMultiplication_2D_Self_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
			*kernels[self_contribution_index],
			*Incol_x[self_contribution_index], *Incol_y[self_contribution_index], *Incol_z[self_contribution_index],
			Out_x, Out_y, Out_z, cuN);
	}
	else {

		cu_KernelMultiplication_2D_Self <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
			*kernels[self_contribution_index],
			*Incol_x[self_contribution_index], *Incol_y[self_contribution_index], *Incol_z[self_contribution_index],
			Out_x, Out_y, Out_z, cuN);
	}

	//the rest add to Out
	for (int mesh_index = 0; mesh_index < Incol_x.size(); mesh_index++) {

		//now compute the other contributions by adding to Out, but skip the self contribution already included
		if (self_contribution_index == mesh_index) continue;

		//z-shifted : use symmetries
		if (zshifted[mesh_index]) {

			//inverse : adjust signs
			if (inverse_shifted[mesh_index]) {

				if (transpose_xy) {

					cu_KernelMultiplication_2D_inversezShifted_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
						*kernels[mesh_index],
						*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
						Out_x, Out_y, Out_z, cuN);
				}
				else {
					
					cu_KernelMultiplication_2D_inversezShifted <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
						*kernels[mesh_index],
						*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
						Out_x, Out_y, Out_z, cuN);
				}
			}
			//z-shifted regular
			else {

				if (transpose_xy) {

					cu_KernelMultiplication_2D_zShifted_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
						*kernels[mesh_index],
						*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
						Out_x, Out_y, Out_z, cuN);
				}
				else {

					cu_KernelMultiplication_2D_zShifted <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
						*kernels[mesh_index],
						*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
						Out_x, Out_y, Out_z, cuN);
				}
			}
		}
		//x-shifted : use symmetries
		else if (xshifted[mesh_index]) {

			if (transpose_xy) {

				cu_KernelMultiplication_2D_xShifted_transpose_xy <<< ((N.x / 2 + 1) * N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					*kernels[mesh_index],
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
					Out_x, Out_y, Out_z, cuN);
			}
			else {

				cu_KernelMultiplication_2D_xShifted <<< ((N.x / 2 + 1) * N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					*kernels[mesh_index],
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
					Out_x, Out_y, Out_z, cuN);
			}
		}
		//it's possible this rect coincides with another rect in the collection (and not self contribution), in which case use 2D Self multiplication
		else if (Rect_collection[mesh_index] == this_rect) {

			if (transpose_xy) {

				cu_KernelMultiplication_2D_Self_transpose_xy_add <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					*kernels[mesh_index],
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
					Out_x, Out_y, Out_z, cuN);
			}
			else {

				cu_KernelMultiplication_2D_Self_add <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					*kernels[mesh_index],
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
					Out_x, Out_y, Out_z, cuN);
			}
		}
		//now compute the other contributions by adding to Out : general kernel multiplication without any symmetries used
		else {
			
			cu_KernelMultiplication_2D_Regular <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
				*kernels[mesh_index],
				*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
				Out_x, Out_y, Out_z, cuN);
		}
	}
}

void DemagKernelCollectionCUDA::KernelMultiplication_3D(
	std::vector<cu_arr<cuBComplex>*>& Incol_x, std::vector<cu_arr<cuBComplex>*>& Incol_y, std::vector<cu_arr<cuBComplex>*>& Incol_z,
	cu_arr<cuBComplex>& Out_x, cu_arr<cuBComplex>& Out_y, cu_arr<cuBComplex>& Out_z)
{
	//transpose_xy always true in 3D
	
	//first compute the self contribution -> this sets Out
	cu_KernelMultiplication_3D_Self_transpose_xy <<< ((N.x / 2 + 1)*N.y*N.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
		*kernels[self_contribution_index],
		*Incol_x[self_contribution_index], *Incol_y[self_contribution_index], *Incol_z[self_contribution_index],
		Out_x, Out_y, Out_z, cuN);
	
	//now compute the other contribution by adding to Out
	for (int mesh_index = 0; mesh_index < Incol_x.size(); mesh_index++) {

		//now compute the other contributions by adding to Out, but skip the self contribution already included
		if (self_contribution_index == mesh_index) continue;

		//z-shifted : use symmetries
		if (zshifted[mesh_index]) {
			
			cu_KernelMultiplication_3D_zShifted_transpose_xy <<< ((N.x / 2 + 1)*N.y*N.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
				*kernels[mesh_index],
				*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
				Out_x, Out_y, Out_z, cuN);
		}
		//x-shifted : use symmetries
		else if (xshifted[mesh_index]) {

			cu_KernelMultiplication_3D_xShifted_transpose_xy <<< ((N.x / 2 + 1) * N.y * N.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
				*kernels[mesh_index],
				*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
				Out_x, Out_y, Out_z, cuN);
		}
		//it's possible this rect coincides with another rect in the collection (and not self contribution), in which case use 3D Self multiplication
		else if (Rect_collection[mesh_index] == this_rect) {

			cu_KernelMultiplication_3D_Self_transpose_xy_add <<< ((N.x / 2 + 1)*N.y*N.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
				*kernels[mesh_index],
				*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
				Out_x, Out_y, Out_z, cuN);
		}
		//general kernel multiplication without any symmetries used
		else {
			
			cu_KernelMultiplication_3D_Regular <<< ((N.x / 2 + 1)*N.y*N.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
				*kernels[mesh_index],
				*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index],
				Out_x, Out_y, Out_z, cuN);
		}
	}
}

void DemagKernelCollectionCUDA::KernelMultiplication_q2D(
	std::vector<cu_arr<cuBComplex>*>& Incol_x, std::vector<cu_arr<cuBComplex>*>& Incol_y, std::vector<cu_arr<cuBComplex>*>& Incol_z,
	cu_arr<cuBComplex>& Out_x, cu_arr<cuBComplex>& Out_y, cu_arr<cuBComplex>& Out_z)
{
	//transpose_xy always true in 3D (including q2D)

	//first compute the self contribution -> this sets Out

	switch (q2D_level)
	{
		//N.z = 4, n.z = 2
	case 4:
		cu_MultiDemag_ConvProd_q2D_4_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
			(*kernels[self_contribution_index])()->Kdiag_real, (*kernels[self_contribution_index])()->Kodiag_real,
			*Incol_x[self_contribution_index], *Incol_y[self_contribution_index], *Incol_z[self_contribution_index], Out_x, Out_y, Out_z, cuN);
		break;
		
		//N.z = 8, n.z = 3, 4
	case 8:
		cu_MultiDemag_ConvProd_q2D_8_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
			(*kernels[self_contribution_index])()->Kdiag_real, (*kernels[self_contribution_index])()->Kodiag_real,
			*Incol_x[self_contribution_index], *Incol_y[self_contribution_index], *Incol_z[self_contribution_index], Out_x, Out_y, Out_z, cuN);
		break;

		//N.z = 16, n.z = 5, 6, 7, 8
	case 16:
		cu_MultiDemag_ConvProd_q2D_16_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
			(*kernels[self_contribution_index])()->Kdiag_real, (*kernels[self_contribution_index])()->Kodiag_real,
			*Incol_x[self_contribution_index], *Incol_y[self_contribution_index], *Incol_z[self_contribution_index], Out_x, Out_y, Out_z, cuN);
		break;

		//N.z = 32, n.z = 9, 10, ..., 16
	case 32:
		cu_MultiDemag_ConvProd_q2D_32_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
			(*kernels[self_contribution_index])()->Kdiag_real, (*kernels[self_contribution_index])()->Kodiag_real,
			*Incol_x[self_contribution_index], *Incol_y[self_contribution_index], *Incol_z[self_contribution_index], Out_x, Out_y, Out_z, cuN);
		break;
		
		//higher values not handled in q2D mode as they are slower than full 3D mode
	}

	//now compute the other contribution by adding to Out
	for (int mesh_index = 0; mesh_index < Incol_x.size(); mesh_index++) {

		//now compute the other contributions by adding to Out, but skip the self contribution already included
		if (self_contribution_index == mesh_index) continue;
		
		//z-shifted : use symmetries
		if (zshifted[mesh_index]) {

			switch (q2D_level)
			{
				//N.z = 4, n.z = 2
			case 4:
				cu_MultiDemag_ConvProd_q2D_4_transpose_xy_zshifted <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;

				//N.z = 8, n.z = 3, 4
			case 8:
				cu_MultiDemag_ConvProd_q2D_8_transpose_xy_zshifted <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;

				//N.z = 8, n.z = 5, 6, 7, 8
			case 16:
				cu_MultiDemag_ConvProd_q2D_16_transpose_xy_zshifted <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;

				//N.z = 32, n.z = 9, 10, ..., 16
			case 32:
				cu_MultiDemag_ConvProd_q2D_32_transpose_xy_zshifted <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;
			}
		}
		//x-shifted : use symmetries
		else if (xshifted[mesh_index]) {

			switch (q2D_level)
			{
				//N.z = 4, n.z = 2
			case 4:
				cu_MultiDemag_ConvProd_q2D_4_transpose_xy_xshifted <<< ((N.x / 2 + 1) * N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;

				//N.z = 8, n.z = 3, 4
			case 8:
				cu_MultiDemag_ConvProd_q2D_8_transpose_xy_xshifted <<< ((N.x / 2 + 1) * N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;

				//N.z = 8, n.z = 5, 6, 7, 8
			case 16:
				cu_MultiDemag_ConvProd_q2D_16_transpose_xy_xshifted <<< ((N.x / 2 + 1) * N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;

				//N.z = 32, n.z = 9, 10, ..., 16
			case 32:
				cu_MultiDemag_ConvProd_q2D_32_transpose_xy_xshifted <<< ((N.x / 2 + 1) * N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;
			}
		}
		//it's possible this rect coincides with another rect in the collection (and not self contribution), in which case use Q2D Self multiplication (but add to output)
		else if (Rect_collection[mesh_index] == this_rect) {
			
			switch (q2D_level)
			{
				//N.z = 4, n.z = 2
			case 4:
				cu_MultiDemag_ConvProd_q2D_4_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_real, (*kernels[mesh_index])()->Kodiag_real,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN, false);
				break;

				//N.z = 8, n.z = 3, 4
			case 8:
				cu_MultiDemag_ConvProd_q2D_8_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_real, (*kernels[mesh_index])()->Kodiag_real,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN, false);
				break;

				//N.z = 16, n.z = 5, 6, 7, 8
			case 16:
				cu_MultiDemag_ConvProd_q2D_16_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_real, (*kernels[mesh_index])()->Kodiag_real,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN, false);
				break;

				//N.z = 32, n.z = 9, 10, ..., 16
			case 32:
				cu_MultiDemag_ConvProd_q2D_32_transpose_xy <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_real, (*kernels[mesh_index])()->Kodiag_real,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN, false);
				break;

				//higher values not handled in q2D mode as they are slower than full 3D mode
			}
		}
		//general kernel multiplication without any symmetries used
		else {

			switch (q2D_level)
			{
				//N.z = 4, n.z = 2
			case 4:

				cu_MultiDemag_ConvProd_q2D_4_transpose_xy_Regular <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;

				//N.z = 8, n.z = 3, 4
			case 8:
				cu_MultiDemag_ConvProd_q2D_8_transpose_xy_Regular <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;

				//N.z = 16, n.z = 5, 6, 7, 8
			case 16:
				cu_MultiDemag_ConvProd_q2D_16_transpose_xy_Regular <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;

				//N.z = 32, n.z = 9, 10, ..., 16
			case 32:
				
				cu_MultiDemag_ConvProd_q2D_32_transpose_xy_Regular <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
					(*kernels[mesh_index])()->Kdiag_cmpl, (*kernels[mesh_index])()->Kodiag_cmpl,
					*Incol_x[mesh_index], *Incol_y[mesh_index], *Incol_z[mesh_index], Out_x, Out_y, Out_z, cuN);
				break;

				//higher values not handled in q2D mode as they are slower than full 3D mode
			}
		}
	}
	
	//IFFT on accumulated scratch space
	switch (q2D_level)
	{
		//N.z = 4, n.z = 2
	case 4:
		
		cu_MultiDemag_ConvProd_q2D_4_IFFT <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
			Out_x, Out_y, Out_z, cuN);
		break;
		
		//N.z = 8, n.z = 3, 4
	case 8:
		cu_MultiDemag_ConvProd_q2D_8_IFFT <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
			Out_x, Out_y, Out_z, cuN);
		break;

		//N.z = 16, n.z = 5, 6, 7, 8
	case 16:
		cu_MultiDemag_ConvProd_q2D_16_IFFT <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
			Out_x, Out_y, Out_z, cuN);
		break;

		//N.z = 32, n.z = 9, 10, ..., 16
	case 32:
		cu_MultiDemag_ConvProd_q2D_32_IFFT <<< ((N.x / 2 + 1)*N.y + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (
			Out_x, Out_y, Out_z, cuN);
		break;
		
		//higher values not handled in q2D mode as they are slower than full 3D mode
	}
}

#endif

#endif