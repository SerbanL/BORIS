#include "DemagKernelCollectionCUDA_KerType.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_SDEMAG) || defined(MODULE_COMPILATION_DEMAG)

#include <cuda_runtime.h>

//--------------------------

//Copy Re parts of cuOut to Kdiag component (1: Kx, 2: Ky, 3: Kz)

__global__ void Set_Kdiag_realcomponent_kernel(cuVEC<cuReal3>& Kdiag, cufftDoubleComplex* cuOut, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (idx < Kdiag.n.dim()) {

		if (component == 1) Kdiag[idx].x = cuOut[idx].x;
		else if (component == 2) Kdiag[idx].y = cuOut[idx].x;
		else if (component == 3) Kdiag[idx].z = cuOut[idx].x;
	}
}

__global__ void Set_Kdiag_realcomponent_transpose_kernel(cuVEC<cuReal3>& Kdiag, cufftDoubleComplex* cuOut, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < Kdiag.n.dim()) {

		int i = idx % Kdiag.n.x;
		int j = (idx / Kdiag.n.x) % Kdiag.n.y;
		int k = idx / (Kdiag.n.x * Kdiag.n.y);

		if (component == 1) Kdiag[j + i * Kdiag.n.y + k * Kdiag.n.x * Kdiag.n.y].x = cuOut[idx].x;
		else if (component == 2) Kdiag[j + i * Kdiag.n.y + k * Kdiag.n.x * Kdiag.n.y].y = cuOut[idx].x;
		else if (component == 3) Kdiag[j + i * Kdiag.n.y + k * Kdiag.n.x * Kdiag.n.y].z = cuOut[idx].x;
	}
}

void cuKerType::Set_Kdiag_real(size_t size, cu_arr<cufftDoubleComplex>& cuOut, int component, int transpose_xy)
{
	if (!transpose_xy) {

		Set_Kdiag_realcomponent_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> > (Kdiag_real, cuOut, component);
	}
	else {

		Set_Kdiag_realcomponent_transpose_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kdiag_real, cuOut, component);
	}
}

//--------------------------

__global__ void Set_K2D_odiag_kernel(cuVEC<cuBReal>& K2D_odiag, cufftDoubleComplex* cuOut)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < K2D_odiag.n.dim()) {

		K2D_odiag[idx] = -cuOut[idx].y;
	}
}

__global__ void Set_K2D_odiag_transpose_kernel(cuVEC<cuBReal>& K2D_odiag, cufftDoubleComplex* cuOut)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < K2D_odiag.n.dim()) {

		int i = idx % K2D_odiag.n.x;
		int j = (idx / K2D_odiag.n.x) % K2D_odiag.n.y;

		K2D_odiag[j + i * K2D_odiag.n.y] = -cuOut[idx].y;
	}
}

//Copy -(Re, Im, Im) parts of cuOut to Kodiag component (1: Kxy, 2: Kxz, 3: Kyz). Takes into account transpose_xy flag.
void cuKerType::Set_K2D_odiag(size_t size, cu_arr<cufftDoubleComplex>& cuOut, int transpose_xy)
{
	if (!transpose_xy) {

		Set_K2D_odiag_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (K2D_odiag, cuOut);
	}
	else {

		Set_K2D_odiag_transpose_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (K2D_odiag, cuOut);
	}
}

//--------------------------

//Copy -Im, Re, Im parts of cuOut to Kodiag component (1: Kxy, 2: Kxz, 3: Kyz)

__global__ void Set_Kodiag_real_2D_component_kernel(cuVEC<cuReal3>& Kodiag, cufftDoubleComplex* cuOut, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < Kodiag.n.dim()) {

		if (component == 1) Kodiag[idx].x = -cuOut[idx].y;
		else if (component == 2) Kodiag[idx].y = cuOut[idx].x;
		else if (component == 3) Kodiag[idx].z = cuOut[idx].y;
	}
}

__global__ void Set_Kodiag_real_2D_component_transpose_kernel(cuVEC<cuReal3>& Kodiag, cufftDoubleComplex* cuOut, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < Kodiag.n.dim()) {

		int i = idx % Kodiag.n.x;
		int j = (idx / Kodiag.n.x) % Kodiag.n.y;

		if (component == 1) Kodiag[j + i * Kodiag.n.y].x = -cuOut[idx].y;
		else if (component == 2) Kodiag[j + i * Kodiag.n.y].y = cuOut[idx].x;
		else if (component == 3) Kodiag[j + i * Kodiag.n.y].z = cuOut[idx].y;
	}
}

void cuKerType::Set_Kodiag_real_2D(size_t size, cu_arr<cufftDoubleComplex>& cuOut, int component, int transpose_xy)
{
	if (!transpose_xy) {

		Set_Kodiag_real_2D_component_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kodiag_real, cuOut, component);
	}
	else {

		Set_Kodiag_real_2D_component_transpose_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kodiag_real, cuOut, component);
	}
}

//Copy -Re, -Im, -Im parts of cuOut to Kodiag component (1: Kxy, 2: Kxz, 3: Kyz)

__global__ void Set_Kodiag_real_3D_component_kernel(cuVEC<cuReal3>& Kodiag, cufftDoubleComplex* cuOut, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < Kodiag.n.dim()) {

		if (component == 1) Kodiag[idx].x = -cuOut[idx].x;
		else if (component == 2) Kodiag[idx].y = -cuOut[idx].y;
		else if (component == 3) Kodiag[idx].z = -cuOut[idx].y;
	}
}

__global__ void Set_Kodiag_real_3D_component_transpose_kernel(cuVEC<cuReal3>& Kodiag, cufftDoubleComplex* cuOut, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < Kodiag.n.dim()) {

		int i = idx % Kodiag.n.x;
		int j = (idx / Kodiag.n.x) % Kodiag.n.y;
		int k = idx / (Kodiag.n.x * Kodiag.n.y);

		if (component == 1) Kodiag[j + i * Kodiag.n.y + k * Kodiag.n.x * Kodiag.n.y].x = -cuOut[idx].x;
		else if (component == 2) Kodiag[j + i * Kodiag.n.y + k * Kodiag.n.x * Kodiag.n.y].y = -cuOut[idx].y;
		else if (component == 3) Kodiag[j + i * Kodiag.n.y + k * Kodiag.n.x * Kodiag.n.y].z = -cuOut[idx].y;
	}
}

void cuKerType::Set_Kodiag_real_3D(size_t size, cu_arr<cufftDoubleComplex>& cuOut, int component, int transpose_xy)
{
	if (!transpose_xy) {

		Set_Kodiag_real_3D_component_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kodiag_real, cuOut, component);
	}
	else {

		Set_Kodiag_real_3D_component_transpose_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kodiag_real, cuOut, component);
	}
}

//--------------------------

//Copy complex to complex for given kernel component

__global__ void Set_K_cmpl_kernel(cuVEC<cuReIm3>& Kdiag, cufftDoubleComplex* cuOut, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < Kdiag.n.dim()) {

		if (component == 1) Kdiag[idx].x = cuReIm(cuOut[idx].x, cuOut[idx].y);
		else if (component == 2) Kdiag[idx].y = cuReIm(cuOut[idx].x, cuOut[idx].y);
		else if (component == 3) Kdiag[idx].z = cuReIm(cuOut[idx].x, cuOut[idx].y);
	}
}

__global__ void Set_K_cmpl_transpose_kernel(cuVEC<cuReIm3>& Kdiag, cufftDoubleComplex* cuOut, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < Kdiag.n.dim()) {

		int i = idx % Kdiag.n.x;
		int j = (idx / Kdiag.n.x) % Kdiag.n.y;
		int k = idx / (Kdiag.n.x * Kdiag.n.y);

		if (component == 1) Kdiag[j + i * Kdiag.n.y + k * Kdiag.n.x * Kdiag.n.y].x = cuReIm(cuOut[idx].x, cuOut[idx].y);
		else if (component == 2) Kdiag[j + i * Kdiag.n.y + k * Kdiag.n.x * Kdiag.n.y].y = cuReIm(cuOut[idx].x, cuOut[idx].y);
		else if (component == 3) Kdiag[j + i * Kdiag.n.y + k * Kdiag.n.x * Kdiag.n.y].z = cuReIm(cuOut[idx].x, cuOut[idx].y);
	}
}

void cuKerType::Set_Kdiag_cmpl(size_t size, cu_arr<cufftDoubleComplex>& cuOut, int component, int transpose_xy)
{
	if (!transpose_xy) {

		Set_K_cmpl_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kdiag_cmpl, cuOut, component);
	}
	else {

		Set_K_cmpl_transpose_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kdiag_cmpl, cuOut, component);
	}
}

void cuKerType::Set_Kodiag_cmpl(size_t size, cu_arr<cufftDoubleComplex>& cuOut, int component, int transpose_xy)
{
	if (!transpose_xy) {

		Set_K_cmpl_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kodiag_cmpl, cuOut, component);
	}
	else {

		Set_K_cmpl_transpose_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kodiag_cmpl, cuOut, component);
	}
}

//--------------------------

//Copy complex to complex for given kernel component

__global__ void Set_K_cmpl_reduced_kernel(cuVEC<cuReIm3>& Kdiag, cufftDoubleComplex* cuOut, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < Kdiag.n.dim()) {

		if (component == 1) Kdiag[idx].x = cuReIm(cuOut[idx].x, cuOut[idx].y);
		else if (component == 2) Kdiag[idx].y = cuReIm(cuOut[idx].x, cuOut[idx].y);
		else if (component == 3) Kdiag[idx].z = cuReIm(cuOut[idx].x, cuOut[idx].y);
	}
}

__global__ void Set_K_cmpl_reduced_transpose_kernel(cuVEC<cuReIm3>& Kdiag, cufftDoubleComplex* cuOut, int component)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < Kdiag.n.dim()) {

		int i = idx % Kdiag.n.x;
		int j = (idx / Kdiag.n.x) % Kdiag.n.y;
		int k = idx / (Kdiag.n.x * Kdiag.n.y);

		if (component == 1) Kdiag[j + i * Kdiag.n.y + k * Kdiag.n.x * Kdiag.n.y].x = cuReIm(cuOut[idx].x, cuOut[idx].y);
		else if (component == 2) Kdiag[j + i * Kdiag.n.y + k * Kdiag.n.x * Kdiag.n.y].y = cuReIm(cuOut[idx].x, cuOut[idx].y);
		else if (component == 3) Kdiag[j + i * Kdiag.n.y + k * Kdiag.n.x * Kdiag.n.y].z = cuReIm(cuOut[idx].x, cuOut[idx].y);
	}
}

void cuKerType::Set_Kdiag_cmpl_reduced(size_t size, cu_arr<cufftDoubleComplex>& cuOut, int component, int transpose_xy)
{
	if (!transpose_xy) {

		Set_K_cmpl_reduced_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kdiag_cmpl, cuOut, component);
	}
	else {

		Set_K_cmpl_reduced_transpose_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kdiag_cmpl, cuOut, component);
	}
}

void cuKerType::Set_Kodiag_cmpl_reduced(size_t size, cu_arr<cufftDoubleComplex>& cuOut, int component, int transpose_xy)
{
	if (!transpose_xy) {

		Set_K_cmpl_reduced_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kodiag_cmpl, cuOut, component);
	}
	else {

		Set_K_cmpl_reduced_transpose_kernel <<< (size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (Kodiag_cmpl, cuOut, component);
	}
}

#endif
#endif