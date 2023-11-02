#include "DemagKernelCUDA.h"

#if COMPILECUDA == 1

#if defined MODULE_COMPILATION_DEMAG || defined MODULE_COMPILATION_SDEMAG

#include <cuda_runtime.h>

//Auxiliary for kernel computations on the GPU

//--------------------------

//copy Re or Im parts of cuOut to cuIn

__global__ void cuOut_to_cuIn_Re_kernel(
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

void DemagKernelCUDA::cuOut_to_cuIn_Re(
	size_t size, 
	cu_arr<cufftDoubleReal>& cuIn, cu_arr<cufftDoubleComplex>& cuOut, 
	int xStride, int yStride, int xStart, int xEnd)
{
	cuOut_to_cuIn_Re_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
		(size, cuIn, cuOut, xStride, yStride, xStart, xEnd);
}

__global__ void cuOut_to_cuIn_Im_kernel(
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

void DemagKernelCUDA::cuOut_to_cuIn_Im(
	size_t size, 
	cu_arr<cufftDoubleReal>& cuIn, cu_arr<cufftDoubleComplex>& cuOut,
	int xStride, int yStride, int xStart, int xEnd)
{
	cuOut_to_cuIn_Im_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
		(size, cuIn, cuOut, xStride, yStride, xStart, xEnd);
}

//--------------------------

//Copy Re parts of cuOut to Kdiag component (1: Kx, 2: Ky, 3: Kz)

__global__ void cuOut_to_Kdiagcomponent_kernel(cuVEC<cuReal3>& Kdiag, cufftDoubleComplex* cuOut, cuSZ3& N, cuINT2& xRegion, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int nxRegion = (xRegion.IsNull() ? N.x / 2 + 1 : xRegion.j - xRegion.i);

	if (idx < nxRegion * (N.y / 2 + 1) * (N.z / 2 + 1)) {

		if (component == 1) Kdiag[idx].x = cuOut[idx].x;
		else if (component == 2) Kdiag[idx].y = cuOut[idx].x;
		else if (component == 3) Kdiag[idx].z = cuOut[idx].x;
	}
}

__global__ void cuOut_to_Kdiagcomponent_transpose_kernel(cuVEC<cuReal3>& Kdiag, cufftDoubleComplex* cuOut, cuSZ3& N, cuINT2& xRegion, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int nxRegion = (xRegion.IsNull() ? N.x / 2 + 1 : xRegion.j - xRegion.i);

	if (idx < nxRegion * (N.y / 2 + 1) * (N.z / 2 + 1)) {

		int i = idx % nxRegion;
		int j = (idx / nxRegion) % (N.y / 2 + 1);
		int k = idx / (nxRegion * (N.y / 2 + 1));

		if (component == 1) Kdiag[j + i * (N.y / 2 + 1) + k * nxRegion * (N.y / 2 + 1)].x = cuOut[idx].x;
		else if (component == 2) Kdiag[j + i * (N.y / 2 + 1) + k * nxRegion * (N.y / 2 + 1)].y = cuOut[idx].x;
		else if (component == 3) Kdiag[j + i * (N.y / 2 + 1) + k * nxRegion * (N.y / 2 + 1)].z = cuOut[idx].x;
	}
}

void DemagKernelCUDA::cuOut_to_Kdiagcomponent(cu_arr<cufftDoubleComplex>& cuOut, int component)
{
	if (!transpose_xy) {

		cuOut_to_Kdiagcomponent_kernel <<< (nxRegion * (N.y / 2 + 1) * (N.z / 2 + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kdiag, cuOut, cuN, cuxRegion, component);
	}
	else {

		cuOut_to_Kdiagcomponent_transpose_kernel <<< (nxRegion * (N.y / 2 + 1) * (N.z / 2 + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kdiag, cuOut, cuN, cuxRegion, component);
	}
}

//--------------------------

//Copy -Im parts of cuOut to K2D_odiag

__global__ void cuOut_to_K2D_odiag_kernel(cuVEC<cuBReal>& K2D_odiag, cufftDoubleComplex* cuOut, cuSZ3& N, cuINT2& xRegion)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int nxRegion = (xRegion.IsNull() ? N.x / 2 + 1 : xRegion.j - xRegion.i);

	if (idx < nxRegion * (N.y / 2 + 1)) {

		K2D_odiag[idx] = -cuOut[idx].y;
	}
}

__global__ void cuOut_to_K2D_odiag_transpose_kernel(cuVEC<cuBReal>& K2D_odiag, cufftDoubleComplex* cuOut, cuSZ3& N, cuINT2& xRegion)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int nxRegion = (xRegion.IsNull() ? N.x / 2 + 1 : xRegion.j - xRegion.i);

	if (idx < nxRegion * (N.y / 2 + 1)) {

		int i = idx % nxRegion;
		int j = (idx / nxRegion) % (N.y / 2 + 1);

		K2D_odiag[j + i * (N.y / 2 + 1)] = -cuOut[idx].y;
	}
}

void DemagKernelCUDA::cuOut_to_K2D_odiag(cu_arr<cufftDoubleComplex>& cuOut)
{
	if (!transpose_xy) {

		cuOut_to_K2D_odiag_kernel <<< (nxRegion * (N.y / 2 + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (K2D_odiag, cuOut, cuN, cuxRegion);
	}
	else {

		cuOut_to_K2D_odiag_transpose_kernel <<< (nxRegion * (N.y / 2 + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (K2D_odiag, cuOut, cuN, cuxRegion);
	}
}

//--------------------------

__global__ void cuOut_to_Kodiagcomponent_kernel(cuVEC<cuReal3>& Kodiag, cufftDoubleComplex* cuOut, cuSZ3& N, cuINT2& xRegion, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int nxRegion = (xRegion.IsNull() ? N.x / 2 + 1 : xRegion.j - xRegion.i);

	if (idx < nxRegion * (N.y / 2 + 1) * (N.z / 2 + 1)) {

		if (component == 1) Kodiag[idx].x = -cuOut[idx].x;
		else if (component == 2) Kodiag[idx].y = -cuOut[idx].y;
		else if (component == 3) Kodiag[idx].z = -cuOut[idx].y;
	}
}

__global__ void cuOut_to_Kodiagcomponent_transpose_kernel(cuVEC<cuReal3>& Kodiag, cufftDoubleComplex* cuOut, cuSZ3& N, cuINT2& xRegion, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int nxRegion = (xRegion.IsNull() ? N.x / 2 + 1 : xRegion.j - xRegion.i);

	if (idx < nxRegion * (N.y / 2 + 1) * (N.z / 2 + 1)) {

		int i = idx % nxRegion;
		int j = (idx / nxRegion) % (N.y / 2 + 1);
		int k = idx / (nxRegion * (N.y / 2 + 1));

		if (component == 1) Kodiag[j + i * (N.y / 2 + 1) + k * nxRegion * (N.y / 2 + 1)].x = -cuOut[idx].x;
		else if (component == 2) Kodiag[j + i * (N.y / 2 + 1) + k * nxRegion * (N.y / 2 + 1)].y = -cuOut[idx].y;
		else if (component == 3) Kodiag[j + i * (N.y / 2 + 1) + k * nxRegion * (N.y / 2 + 1)].z = -cuOut[idx].y;
	}
}

//Copy -(Re, Im, Im) parts of cuOut to Kodiag component (1: Kxy, 2: Kxz, 3: Kyz). Takes into account transpose_xy flag.
void DemagKernelCUDA::cuOut_to_Kodiagcomponent(cu_arr<cufftDoubleComplex>& cuOut, int component)
{
	if (!transpose_xy) {

		cuOut_to_Kodiagcomponent_kernel <<< (nxRegion * (N.y / 2 + 1) * (N.z / 2 + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kodiag, cuOut, cuN, cuxRegion, component);
	}
	else {

		cuOut_to_Kodiagcomponent_transpose_kernel <<< (nxRegion * (N.y / 2 + 1) * (N.z / 2 + 1) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kodiag, cuOut, cuN, cuxRegion, component);
	}
}

#endif
#endif