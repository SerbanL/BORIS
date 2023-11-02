#include "DemagKernelCollectionCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_SDEMAG) || defined(MODULE_COMPILATION_DEMAG)

#include <cuda_runtime.h>

//Auxiliary for kernel computations on the GPU

//--------------------------

//copy Re or Im parts of cuOut to cuIn

__global__ void cuOut_to_cuIn_Re_collection_kernel(
	size_t size,
	cufftDoubleReal* cuIn, cufftDoubleComplex* cuOut,
	int xStride, int yStride, int xStart, int xEnd)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size) {

		//ijk in Out
		cuINT3 ijk = cuINT3(idx % (xEnd - xStart) + xStart, (idx / (xEnd - xStart)) % yStride, idx / ((xEnd - xStart) * yStride));

		cuIn[idx] = cuOut[ijk.i + ijk.j * xStride + ijk.k * xStride * yStride].x;
	}
}

void DemagKernelCollectionCUDA::cuOut_to_cuIn_Re(
	size_t size,
	cu_arr<cufftDoubleReal>& cuIn, cu_arr<cufftDoubleComplex>& cuOut,
	int xStride, int yStride, int xStart, int xEnd)
{
	cuOut_to_cuIn_Re_collection_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(size, cuIn, cuOut, xStride, yStride, xStart, xEnd);
}

__global__ void cuOut_to_cuIn_Im_collection_kernel(
	size_t size,
	cufftDoubleReal* cuIn, cufftDoubleComplex* cuOut,
	int xStride, int yStride, int xStart, int xEnd)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < size) {

		//ijk in Out
		cuINT3 ijk = cuINT3(idx % (xEnd - xStart) + xStart, (idx / (xEnd - xStart)) % yStride, idx / ((xEnd - xStart) * yStride));

		cuIn[idx] = cuOut[ijk.i + ijk.j * xStride + ijk.k * xStride * yStride].y;
	}
}

void DemagKernelCollectionCUDA::cuOut_to_cuIn_Im(
	size_t size,
	cu_arr<cufftDoubleReal>& cuIn, cu_arr<cufftDoubleComplex>& cuOut,
	int xStride, int yStride, int xStart, int xEnd)
{
	cuOut_to_cuIn_Im_collection_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(size, cuIn, cuOut, xStride, yStride, xStart, xEnd);
}

#endif
#endif