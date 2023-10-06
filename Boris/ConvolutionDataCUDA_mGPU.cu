#include "ConvolutionDataCUDA.h"

#if COMPILECUDA == 1

#include "BorisCUDALib.cuh"

//--------------------------------------------------- Device Auxiliary

__device__ bool to_nx_in_yRegion(
	int& idx, int& idx_out, int& component_idx, int& idx_in,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices)
{
	int nxRegion = xRegion_R.j - xRegion_R.i;
	int nyRegion = yRegion.j - yRegion.i;

	if (idx < n.x * nyRegion * n.z) {

		//ijk index in (n.x, nyRegion, n.z) space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % nyRegion, idx / (n.x * nyRegion));

		//linear index in cu_In, which has dimensions (Nx, nyRegion, nz)
		idx_out = ijk.i + ijk.j * N.x + ijk.k * N.x * nyRegion;

		//if element is inside xRegion then copy from In
		if (ijk.i >= xRegion_R.i && ijk.i < xRegion_R.j) {

			//linear index in In, which has dimensions (nxRegion, n.y, n.z)
			idx_in = ijk.i - xRegion_R.i + (ijk.j + yRegion.i) * nxRegion + ijk.k * nxRegion * n.y;

			//indicate In should be used
			component_idx = -1;
		}
		//if outside xRegion then read from M_Input_yRegion
		else {

			//there are num_devices blocks in M_Input_yRegion, so find which one this element belongs to
			int device_xcells_first = (xRegion_R.j == n.x ? xRegion_R.i / (num_devices - 1) : nxRegion);
			int device_xcells = device_xcells_first;
			component_idx = ijk.i / device_xcells_first;
			//last block can have more than device_xcells cells, so adjust
			if (component_idx == num_devices) component_idx--;
			if (component_idx == num_devices - 1) device_xcells = n.x - device_xcells_first * (num_devices - 1);
			int device_xoffset = device_xcells_first * component_idx;

			//form input index in M_Input_yRegion for device_block
			idx_in = ijk.i - device_xoffset + ijk.j * device_xcells + ijk.k * device_xcells * nyRegion;
		}

		return true;
	}
	else return false;
}

__device__ bool from_yRegion_transpose(
	int& idx, int& component_idx, int& idx_out, int& idx_in,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	int nxRegion = (xRegion.j - xRegion.i);
	int nyRegion = (yRegion.j - yRegion.i);

	if (idx < (N.x / 2 + 1) * nyRegion * n.z) {

		//ijk index in (N.x / 2 + 1, nyRegion, n.z) space
		cuINT3 ijk = cuINT3(idx % (N.x / 2 + 1), (idx / (N.x / 2 + 1)) % nyRegion, idx / ((N.x / 2 + 1) * nyRegion));

		idx_in = idx;

		//if element is inside xRegion then transpose to cuS position
		if (ijk.i >= xRegion.i && ijk.i < xRegion.j) {

			//linear index in transposed cuS space (destination)
			idx_out = (ijk.j + yRegion.i) + (ijk.i - xRegion.i) * N.y + ijk.k * N.y * nxRegion;
			
			//indicate destination should be cuS
			component_idx = -1;
		}
		//if outside xRegion then place it in xFFT_Data_yRegion
		else {

			//there are num_devices blocks in xFFT_Data_yRegion, so find which one this element belongs to
			int device_xcells_first = (xRegion.j == N.x / 2 + 1 ? xRegion.i / (num_devices - 1) : nxRegion);
			int device_xcells = device_xcells_first;
			component_idx = ijk.i / device_xcells_first;
			//last block can have more than device_xcells cells, so adjust
			if (component_idx == num_devices) component_idx--;
			if (component_idx == num_devices - 1) device_xcells = N.x / 2 + 1 - device_xcells_first * (num_devices - 1);
			int device_xoffset = device_xcells_first * component_idx;

			//form output index in xFFT_Data_yRegion for device_block
			idx_out = ijk.j + (ijk.i - device_xoffset) * nyRegion + ijk.k * nyRegion * device_xcells;
		}

		return true;
	}
	else return false;
}

__device__ bool from_xRegion_transpose(
	int& idx, int& component_idx, int& idx_out, int& idx_in,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	int nxRegion = (xRegion.j - xRegion.i);
	int nyRegion = (yRegion.j - yRegion.i);

	if (idx < nxRegion * n.y * n.z) {

		//ijk index in (n.y, nxRegion, n.z) space
		cuINT3 ijk = cuINT3(idx % n.y, (idx / n.y) % nxRegion, idx / (n.y * nxRegion));

		//linear index in input transposed cuS space
		idx_in = ijk.i + ijk.j * N.y + ijk.k * N.y * nxRegion;

		//if element is inside yRegion then transpose to cuSquart position
		if (ijk.i >= yRegion.i && ijk.i < yRegion.j) {

			//linear index in un-transposed cuSquart space (destination)
			idx_out = ijk.j + xRegion.i + (ijk.i - yRegion.i) * (N.x / 2 + 1) + ijk.k * (N.x / 2 + 1) * nyRegion;

			//indicate destination should be cuSquart
			component_idx = -1;
		}
		//if outside yRegion then place it in xIFFT_Data_xRegion
		else {

			//there are num_devices blocks in xIFFT_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			component_idx = ijk.i / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (component_idx == num_devices) component_idx--;
			if (component_idx == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * component_idx;

			//form output index in xIFFT_Data_xRegion for device_block
			idx_out = ijk.j + (ijk.i - device_yoffset) * nxRegion + ijk.k * nxRegion * device_ycells;
		}

		return true;
	}
	else return false;
}

__device__ bool to_notyRegion_inxRegion_transposed(
	int& idx, int& idx_out, int& component_idx, int& idx_in,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	//xFFT_Data_xRegion has the missing yRegions data in the current xRegion
	//cuS arrays have dimensions (transposed) (Ny, nxRegion, Nz)

	int nxRegion = xRegion.j - xRegion.i;
	int nyRegion = yRegion.j - yRegion.i;
	int nCy = (n.y - nyRegion);

	if (idx < nxRegion * nCy * n.z) {

		//more than 2 components, use the general routine
		if (num_devices > 2) {

			//form ijk in complement space, noting that kernel was launched with nCy cells along y, since nyRegion data is already in place
			cuINT3 ijk = cuINT3(idx % nCy, (idx / nCy) % nxRegion, idx / (nCy * nxRegion));

			//find which xFFT_Data component we need to use
			//if yRegion is not the last region, then yRegion.j - yRegion.i is the n.y value common to all xFFT_Data components apart from the last one
			//In this case the last xFFT_Data component has n.y value of n.y - nyRegion * (num_devices - 1)
			//If yRegion is the last region, then the common n.y value of xFFT_Data components before it is yRegion.i / (num_devices - 1)

			//yRegion is the last region
			if (yRegion.j == n.y) {

				//y dimension of components (they are all the same in this case as only component in last region can be different)
				int component_ny = yRegion.i / (num_devices - 1);
				component_idx = ijk.i / component_ny;
				int component_i = ijk.i % component_ny;

				idx_out = ijk.i + ijk.j * N.y + ijk.k * N.y * nxRegion;
				idx_in = component_i + ijk.j * component_ny + ijk.k * component_ny * nxRegion;
			}
			//yRegion is not the last region
			else {

				//component is to the left of yRegion, hence itself not the last region
				if (ijk.i < yRegion.i) {

					int component_ny = nyRegion;
					component_idx = ijk.i / component_ny;
					int component_i = ijk.i % component_ny;

					idx_out = ijk.i + ijk.j * N.y + ijk.k * N.y * nxRegion;
					idx_in = component_i + ijk.j * component_ny + ijk.k * component_ny * nxRegion;
				}
				//component is to the right of yRegion, and could be the last region
				else {

					//x index in entire cuS space is : ijk.i + yRegion.j - yRegion.i (i.e. skip over yRegion)
					component_idx = (ijk.i + nyRegion) / nyRegion;
					if (component_idx == num_devices) component_idx--;
					int component_i = (ijk.i + nyRegion) - nyRegion * component_idx;
					int component_ny;

					//if this is the last component, it is also the last region, hence find its ny value as :
					if (component_idx == num_devices - 1) component_ny = n.y - nyRegion * (num_devices - 1);
					//otherwise we have its ny value already as:
					else component_ny = nyRegion;

					idx_out = ijk.i + nyRegion + ijk.j * N.y + ijk.k * N.y * nxRegion;
					idx_in = component_i + ijk.j * component_ny + ijk.k * component_ny * nxRegion;
				}
			}
		}

		//if number of components is 2 use a simpler routine
		else if (num_devices == 2) {

			if (yRegion.j == n.y) {

				//form ijk in full space
				cuINT3 ijk = cuINT3(idx % nCy, (idx / nCy) % nxRegion, idx / (nCy * nxRegion));
				idx_out = ijk.i + ijk.j * N.y + ijk.k * N.y * nxRegion;

				component_idx = 0;
				idx_in = idx;
			}
			else {

				//form ijk in full space
				cuINT3 ijk = cuINT3(idx % nCy + yRegion.j, (idx / nCy) % nxRegion, idx / (nCy * nxRegion));
				idx_out = ijk.i + ijk.j * N.y + ijk.k * N.y * nxRegion;

				component_idx = 1;
				idx_in = idx;
			}
		}

		return true;
	}
	else return false;
}

__device__ bool to_notxRegion_inyRegion(
	int& idx, int& idx_out, int& component_idx, int& idx_in,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	//xIIFT_Data_Components has the missing xRegions data in the current yRegion
	//cuSquart arrays have dimensions (Nx/2 + 1), nyRegion, nz

	int nxRegion = xRegion.j - xRegion.i;
	int nyRegion = yRegion.j - yRegion.i;
	int nCx = ((N.x / 2 + 1) - nxRegion);

	if (idx < nCx * nyRegion * n.z) {

		//more than 2 components, use the general routine
		if (num_devices > 2) {

			//form ijk in complement space, noting that kernel was launched with ((N.x / 2 + 1) - nxRegion) cells along x, since nxRegion data is already in place
			cuINT3 ijk = cuINT3(idx % nCx, (idx / nCx) % nyRegion, idx / (nCx * nyRegion));

			//find which xIFFT_Data component we need to use
			//if xRegion is not the last region, then xRegion.j - xRegion.i is the n.x value common to all xIFFT_Data components apart from the last one
			//In this case the last xIFFT_Data component has n.x value of (N.x / 2 + 1) - nxRegion * (num_devices - 1)
			//If xRegion is the last region, then the common n.x value of xIFFT_Data components before it is xRegion.i / (num_devices - 1)

			//xRegion is the last region
			if (xRegion.j == (N.x / 2 + 1)) {

				//x dimension of components (they are all the same in this case as only component in last region can be different)
				int component_nx = xRegion.i / (num_devices - 1);
				component_idx = ijk.i / component_nx;
				int component_i = ijk.i % component_nx;

				idx_out = ijk.i + ijk.j * (N.x / 2 + 1) + ijk.k * (N.x / 2 + 1) * nyRegion;
				idx_in = component_i + ijk.j * component_nx + ijk.k * component_nx * nyRegion;
			}
			//xRegion is not the last region
			else {

				//component is to the left of xRegion, hence itself not the last region
				if (ijk.i < xRegion.i) {

					int component_nx = nxRegion;
					component_idx = ijk.i / component_nx;
					int component_i = ijk.i % component_nx;

					idx_out = ijk.i + ijk.j * (N.x / 2 + 1) + ijk.k * (N.x / 2 + 1) * nyRegion;
					idx_in = component_i + ijk.j * component_nx + ijk.k * component_nx * nyRegion;
				}
				//component is to the right of xRegion, and could be the last region
				else {

					//x index in entire cuSquart space is : ijk.i + xRegion.j - xRegion.i (i.e. skip over xRegion)
					component_idx = (ijk.i + nxRegion) / nxRegion;
					if (component_idx == num_devices) component_idx--;
					int component_i = (ijk.i + nxRegion) - nxRegion * component_idx;
					int component_nx;

					//if this is the last component, it is also the last region, hence find its nx value as :
					if (component_idx == num_devices - 1) component_nx = (N.x / 2 + 1) - nxRegion * (num_devices - 1);
					//otherwise we have its nx value already as:
					else component_nx = nxRegion;

					idx_out = ijk.i + nxRegion + ijk.j * (N.x / 2 + 1) + ijk.k * (N.x / 2 + 1) * nyRegion;
					idx_in = component_i + ijk.j * component_nx + ijk.k * component_nx * nyRegion;
				}
			}
		}

		//if number of components is 2 use a simpler routine
		else if (num_devices == 2) {

			idx_in = idx;

			if (xRegion.j == (N.x / 2 + 1)) {

				//form ijk in full space
				cuINT3 ijk = cuINT3(idx % nCx, (idx / nCx) % nyRegion, idx / (nCx * nyRegion));
				idx_out = ijk.i + ijk.j * (N.x / 2 + 1) + ijk.k * (N.x / 2 + 1) * nyRegion;

				component_idx = 0;
			}
			else {

				//form ijk in full space
				cuINT3 ijk = cuINT3(idx % nCx + xRegion.j, (idx / nCx) % nyRegion, idx / (nCx * nyRegion));
				idx_out = ijk.i + ijk.j * (N.x / 2 + 1) + ijk.k * (N.x / 2 + 1) * nyRegion;

				component_idx = 1;
			}
		}

		return true;
	}
	else return false;
}

__device__ bool from_notxRegion_inyRegion(
	int& idx, int& component_idx, int& idx_out, int& idx_in,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices)
{
	int nxRegion = xRegion_R.j - xRegion_R.i;
	int nyRegion = (yRegion.j - yRegion.i);
	int nCx = n.x - nxRegion;

	if (idx < nCx * nyRegion * n.z) {

		//more than 2 components, use the general routine
		if (num_devices > 2) {

			//form ijk in complement space, noting that kernel was launched with (n.x - nxRegion) cells along x, since nxRegion data is already in place
			cuINT3 ijk = cuINT3(idx % nCx, (idx / nCx) % nyRegion, idx / (nCx * nyRegion));

			//find which Out_Data_Components component we need to use
			//if xRegion is not the last region, then xRegion.j - xRegion.i is the n.x value common to all Out_Data_Components components apart from the last one
			//In this case the last Out_Data_Components component has n.x value of n.x - nxRegion * (num_devices - 1)
			//If xRegion is the last region, then the common n.x value of Out_Data_Components components before it is xRegion.i / (num_devices - 1)

			//xRegion is the last region
			if (xRegion_R.j == n.x) {

				//x dimension of components (they are all the same in this case as only component in last region can be different)
				int component_nx = xRegion_R.i / (num_devices - 1);
				component_idx = ijk.i / component_nx;
				int component_i = ijk.i % component_nx;

				idx_in = ijk.i + ijk.j * N.x + ijk.k * N.x * nyRegion;
				idx_out = component_i + ijk.j * component_nx + ijk.k * component_nx * nyRegion;
			}
			//xRegion is not the last region
			else {

				//component is to the left of xRegion, hence itself not the last region
				if (ijk.i < xRegion_R.i) {

					int component_nx = nxRegion;
					component_idx = ijk.i / component_nx;
					int component_i = ijk.i % component_nx;

					idx_in = ijk.i + ijk.j * N.x + ijk.k * N.x * nyRegion;
					idx_out = component_i + ijk.j * component_nx + ijk.k * component_nx * nyRegion;
				}
				//component is to the right of xRegion, and could be the last region
				else {

					//x index in entire cuSquart space is : ijk.i + xRegion.j - xRegion.i (i.e. skip over xRegion)
					component_idx = (ijk.i + nxRegion) / nxRegion;
					if (component_idx == num_devices) component_idx--;
					int component_i = (ijk.i + nxRegion) - nxRegion * component_idx;
					int component_nx;

					//if this is the last component, it is also the last region, hence find its nx value as :
					if (component_idx == num_devices - 1) component_nx = n.x - nxRegion * (num_devices - 1);
					//otherwise we have its nx value already as:
					else component_nx = nxRegion;

					idx_in = ijk.i + nxRegion + ijk.j * N.x + ijk.k * N.x * nyRegion;
					idx_out = component_i + ijk.j * component_nx + ijk.k * component_nx * nyRegion;
				}
			}
		}

		//if number of components is 2 use a simpler routine
		else if (num_devices == 2) {

			idx_out = idx;

			if (xRegion_R.j == n.x) {

				//form ijk in full cuOut space
				cuINT3 ijk = cuINT3(idx % nCx, (idx / nCx) % nyRegion, idx / (nCx * nyRegion));
				idx_in = ijk.i + ijk.j * N.x + ijk.k * N.x * nyRegion;

				component_idx = 0;
			}
			else {

				//form ijk in full cuOut space
				cuINT3 ijk = cuINT3(idx % nCx + xRegion_R.j, (idx / nCx) % nyRegion, idx / (nCx * nyRegion));
				idx_in = ijk.i + ijk.j * N.x + ijk.k * N.x * nyRegion;

				component_idx = 1;
			}
		}

		return true;
	}
	else return false;
}

//--------------------------------------------------- Auxiliary

//cuS has dimensions (nxRegion, Ny, Nz), and when transposed (Ny, nxRegion, Nz)
//cuSquart has dimensions (Nx/2 + 1, nyRegion, nz)
__global__ void cu_transpose_xy_copycomponents_forward_kernel(
	cuBComplex* cuS_x, cuBComplex* cuS_y, cuBComplex* cuS_z,
	cuBComplex* cuSquart_x, cuBComplex* cuSquart_y, cuBComplex* cuSquart_z,
	cuBComplex** xFFT_Data_yRegion,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//form output index in cuS or xFFT_Data_yRegion spaces
	int idx_out;
	//form linear index in cuSquart
	int idx_in;
	//the component index to use in xFFT_Data_yRegion, or if negative use cuS
	int component_idx;

	if (from_yRegion_transpose(idx, component_idx, idx_out, idx_in, n, N, xRegion, yRegion, num_devices)) {

		if (component_idx < 0) {

			cuS_x[idx_out] = cuSquart_x[idx_in];
			cuS_y[idx_out] = cuSquart_y[idx_in];
			cuS_z[idx_out] = cuSquart_z[idx_in];
		}
		else {

			xFFT_Data_yRegion[component_idx][3 * idx_out + 0] = cuSquart_x[idx_in];
			xFFT_Data_yRegion[component_idx][3 * idx_out + 1] = cuSquart_y[idx_in];
			xFFT_Data_yRegion[component_idx][3 * idx_out + 2] = cuSquart_z[idx_in];
		}
	}
}

__global__ void cu_transpose_xy_copycomponents_forward_kernel(
	cuBComplex* cuS_x, cuBComplex* cuS_y, cuBComplex* cuS_z,
	cuBComplex* cuSquart_x, cuBComplex* cuSquart_y, cuBComplex* cuSquart_z,
	cuBHalf** xFFT_Data_yRegion, cuBReal& normalization,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	int idx3 = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = idx3 / 3;

	//form output index in cuS or xFFT_Data_yRegion spaces
	int idx_out;
	//form linear index in cuSquart
	int idx_in;
	//the component index to use in xFFT_Data_yRegion, or if negative use cuS
	int component_idx;

	if (from_yRegion_transpose(idx, component_idx, idx_out, idx_in, n, N, xRegion, yRegion, num_devices)) {

		if (component_idx < 0) {

			if (idx3 % 3 == 0) cuS_x[idx_out] = cuSquart_x[idx_in];
			else if (idx3 % 3 == 1) cuS_y[idx_out] = cuSquart_y[idx_in];
			else if (idx3 % 3 == 2) cuS_z[idx_out] = cuSquart_z[idx_in];
		}
		else {

#if SINGLEPRECISION
			if (idx3 % 3 == 0) {
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 0) * 2 + 0] = float2half_as_uint16(cuSquart_x[idx_in].x / normalization);
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 0) * 2 + 1] = float2half_as_uint16(cuSquart_x[idx_in].y / normalization);
			}
			else if (idx3 % 3 == 1) {
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 1) * 2 + 0] = float2half_as_uint16(cuSquart_y[idx_in].x / normalization);
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 1) * 2 + 1] = float2half_as_uint16(cuSquart_y[idx_in].y / normalization);
			}
			else if (idx3 % 3 == 2) {
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 2) * 2 + 0] = float2half_as_uint16(cuSquart_z[idx_in].x / normalization);
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 2) * 2 + 1] = float2half_as_uint16(cuSquart_z[idx_in].y / normalization);
			}
#else
			if (idx3 % 3 == 0) {
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 0) * 2 + 0] = cuSquart_x[idx_in].x / normalization;
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 0) * 2 + 1] = cuSquart_x[idx_in].y / normalization;
			}
			else if (idx3 % 3 == 1) {
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 1) * 2 + 0] = cuSquart_y[idx_in].x / normalization;
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 1) * 2 + 1] = cuSquart_y[idx_in].y / normalization;
			}
			else if (idx3 % 3 == 2) {
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 2) * 2 + 0] = cuSquart_z[idx_in].x / normalization;
				xFFT_Data_yRegion[component_idx][(3 * idx_out + 2) * 2 + 1] = cuSquart_z[idx_in].y / normalization;
			}
#endif
		}
	}
}

//cuS has dimensions (nxRegion, Ny, Nz)
//cuSquart has dimensions (Nx/2 + 1, nyRegion, nz)
__global__ void cu_transpose_xy_copycomponents_inverse_kernel(
	cuBComplex* cuS_x, cuBComplex* cuS_y, cuBComplex* cuS_z,
	cuBComplex* cuSquart_x, cuBComplex* cuSquart_y, cuBComplex* cuSquart_z,
	cuBComplex** xIFFT_Data_xRegion,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//form output index in cuSquart or xIFFT_Data_xRegion spaces
	int idx_out;
	//form linear index in cuS
	int idx_in;
	//the component index to use in xIFFT_Data_xRegion, or if negative use cuSquart
	int component_idx;

	if (from_xRegion_transpose(idx, component_idx, idx_out, idx_in, n, N, xRegion, yRegion, num_devices)) {

		if (component_idx < 0) {

			cuSquart_x[idx_out] = cuS_x[idx_in];
			cuSquart_y[idx_out] = cuS_y[idx_in];
			cuSquart_z[idx_out] = cuS_z[idx_in];
		}
		else {

			xIFFT_Data_xRegion[component_idx][3 * idx_out + 0] = cuS_x[idx_in];
			xIFFT_Data_xRegion[component_idx][3 * idx_out + 1] = cuS_y[idx_in];
			xIFFT_Data_xRegion[component_idx][3 * idx_out + 2] = cuS_z[idx_in];
		}
	}
}

//cuS has dimensions (nxRegion, Ny, Nz)
//cuSquart has dimensions (Nx/2 + 1, nyRegion, nz)
__global__ void cu_transpose_xy_copycomponents_inverse_kernel(
	cuBComplex* cuS_x, cuBComplex* cuS_y, cuBComplex* cuS_z,
	cuBComplex* cuSquart_x, cuBComplex* cuSquart_y, cuBComplex* cuSquart_z,
	cuBHalf** xIFFT_Data_xRegion, cuBReal& normalization,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	int idx3 = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = idx3 / 3;

	//form output index in cuSquart or xIFFT_Data_xRegion spaces
	int idx_out;
	//form linear index in cuS
	int idx_in;
	//the component index to use in xIFFT_Data_xRegion, or if negative use cuSquart
	int component_idx;

	if (from_xRegion_transpose(idx, component_idx, idx_out, idx_in, n, N, xRegion, yRegion, num_devices)) {

		if (component_idx < 0) {

			if (idx3 % 3 == 0) cuSquart_x[idx_out] = cuS_x[idx_in];
			else if (idx3 % 3 == 1) cuSquart_y[idx_out] = cuS_y[idx_in];
			else if (idx3 % 3 == 2) cuSquart_z[idx_out] = cuS_z[idx_in];
		}
		else {

#if SINGLEPRECISION
			//use a normalization constant to avoid exponent limit for half-precision

			if (idx3 % 3 == 0) {
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 0) * 2 + 0] = float2half_as_uint16(cuS_x[idx_in].x / normalization);
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 0) * 2 + 1] = float2half_as_uint16(cuS_x[idx_in].y / normalization);
			}
			else if (idx3 % 3 == 1) {
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 1) * 2 + 0] = float2half_as_uint16(cuS_y[idx_in].x / normalization);
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 1) * 2 + 1] = float2half_as_uint16(cuS_y[idx_in].y / normalization);
			}
			else if (idx3 % 3 == 2) {
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 2) * 2 + 0] = float2half_as_uint16(cuS_z[idx_in].x / normalization);
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 2) * 2 + 1] = float2half_as_uint16(cuS_z[idx_in].y / normalization);
			}
#else
			if (idx3 % 3 == 0) {
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 0) * 2 + 0] = cuS_x[idx_in].x / normalization;
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 0) * 2 + 1] = cuS_x[idx_in].y / normalization;
			}
			else if (idx3 % 3 == 1) {
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 1) * 2 + 0] = cuS_y[idx_in].x / normalization;
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 1) * 2 + 1] = cuS_y[idx_in].y / normalization;
			}
			else if (idx3 % 3 == 2) {
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 2) * 2 + 0] = cuS_z[idx_in].x / normalization;
				xIFFT_Data_xRegion[component_idx][(3 * idx_out + 2) * 2 + 1] = cuS_z[idx_in].y / normalization;
			}
#endif
		}
	}
}

//xIFFT_Data_Components has size the total number of convolution components, but the one matching the current component xRegion doesn't need to be copied over
__global__ void xIFFT_Data_to_cuSquart_mGPU(
	cuBComplex** xIFFT_Data_Components,
	cuBComplex* cuSquart_x, cuBComplex* cuSquart_y, cuBComplex* cuSquart_z,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//the component index to read from
	int component_idx;
	//form output index in cuSquart spaces, which are of dimension (N.x / 2 + 1) * nyRegion * nz
	int idx_out;
	//form linear index in input xIFFT_Data component
	int idx_in;

	if (to_notxRegion_inyRegion(idx, idx_out, component_idx, idx_in, n, N, xRegion, yRegion, num_devices)) {

		cuSquart_x[idx_out] = xIFFT_Data_Components[component_idx][3 * idx_in + 0];
		cuSquart_y[idx_out] = xIFFT_Data_Components[component_idx][3 * idx_in + 1];
		cuSquart_z[idx_out] = xIFFT_Data_Components[component_idx][3 * idx_in + 2];
	}
}

__global__ void xIFFT_Data_to_cuSquart_mGPU(
	cuBHalf** xIFFT_Data_Components, cuBReal& normalization,
	cuBComplex* cuSquart_x, cuBComplex* cuSquart_y, cuBComplex* cuSquart_z,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	int idx3 = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = idx3 / 3;

	//the component index to read from
	int component_idx;
	//form output index in cuSquart spaces, which are of dimension (N.x / 2 + 1) * nyRegion * nz
	int idx_out;
	//form linear index in input xIFFT_Data component
	int idx_in;

	if (to_notxRegion_inyRegion(idx, idx_out, component_idx, idx_in, n, N, xRegion, yRegion, num_devices)) {

#if SINGLEPRECISION
		if (idx3 % 3 == 0) {
			cuSquart_x[idx_out].x = half_as_uint16_2float(xIFFT_Data_Components[component_idx][(3 * idx_in + 0) * 2 + 0]) * normalization;
			cuSquart_x[idx_out].y = half_as_uint16_2float(xIFFT_Data_Components[component_idx][(3 * idx_in + 0) * 2 + 1]) * normalization;
		}
		else if (idx3 % 3 == 1) {
			cuSquart_y[idx_out].x = half_as_uint16_2float(xIFFT_Data_Components[component_idx][(3 * idx_in + 1) * 2 + 0]) * normalization;
			cuSquart_y[idx_out].y = half_as_uint16_2float(xIFFT_Data_Components[component_idx][(3 * idx_in + 1) * 2 + 1]) * normalization;
		}
		else if (idx3 % 3 == 2) {
			cuSquart_z[idx_out].x = half_as_uint16_2float(xIFFT_Data_Components[component_idx][(3 * idx_in + 2) * 2 + 0]) * normalization;
			cuSquart_z[idx_out].y = half_as_uint16_2float(xIFFT_Data_Components[component_idx][(3 * idx_in + 2) * 2 + 1]) * normalization;
		}
#else
		if (idx3 % 3 == 0) {
			cuSquart_x[idx_out].x = xIFFT_Data_Components[component_idx][(3 * idx_in + 0) * 2 + 0] * normalization;
			cuSquart_x[idx_out].y = xIFFT_Data_Components[component_idx][(3 * idx_in + 0) * 2 + 1] * normalization;
		}
		else if (idx3 % 3 == 1) {
			cuSquart_y[idx_out].x = xIFFT_Data_Components[component_idx][(3 * idx_in + 1) * 2 + 0] * normalization;
			cuSquart_y[idx_out].y = xIFFT_Data_Components[component_idx][(3 * idx_in + 1) * 2 + 1] * normalization;
		}
		else if (idx3 % 3 == 2) {
			cuSquart_z[idx_out].x = xIFFT_Data_Components[component_idx][(3 * idx_in + 2) * 2 + 0] * normalization;
			cuSquart_z[idx_out].y = xIFFT_Data_Components[component_idx][(3 * idx_in + 2) * 2 + 1] * normalization;
		}
#endif
	}
}

//before FFTs copy components from xFFT_Data_xRegion (fixed xRegion) into cuS in missing y regions as they've been transferred in from all other devices
__global__ void xFFT_Data_to_cuS_mGPU(
	cuBComplex** xFFT_Data_xRegion,
	cuBComplex* cuS_x, cuBComplex* cuS_y, cuBComplex* cuS_z,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//the component index to read from
	int component_idx;
	//form output index in cuS spaces, which are of dimension N.y, nxRegion, Nz
	int idx_out;
	//form linear index in input xFFT_Data component
	int idx_in;

	if (to_notyRegion_inxRegion_transposed(idx, idx_out, component_idx, idx_in, n, N, xRegion, yRegion, num_devices)) {

		cuS_x[idx_out] = xFFT_Data_xRegion[component_idx][3 * idx_in + 0];
		cuS_y[idx_out] = xFFT_Data_xRegion[component_idx][3 * idx_in + 1];
		cuS_z[idx_out] = xFFT_Data_xRegion[component_idx][3 * idx_in + 2];
	}
}

__global__ void xFFT_Data_to_cuS_mGPU(
	cuBHalf** xFFT_Data_xRegion, cuBReal& normalization,
	cuBComplex* cuS_x, cuBComplex* cuS_y, cuBComplex* cuS_z,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion, cuINT2& yRegion, int& num_devices)
{
	//kernel was launched with 3x the size
	int idx3 = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = idx3 / 3;

	//the component index to read from
	int component_idx;
	//form output index in cuS spaces, which are of dimension N.y, nxRegion, Nz
	int idx_out;
	//form linear index in input xFFT_Data component
	int idx_in;

	if (to_notyRegion_inxRegion_transposed(idx, idx_out, component_idx, idx_in, n, N, xRegion, yRegion, num_devices)) {

#if SINGLEPRECISION
		if (idx3 % 3 == 0) {

			cuS_x[idx_out].x = half_as_uint16_2float(xFFT_Data_xRegion[component_idx][(3 * idx_in + 0) * 2 + 0]) * normalization;
			cuS_x[idx_out].y = half_as_uint16_2float(xFFT_Data_xRegion[component_idx][(3 * idx_in + 0) * 2 + 1]) * normalization;
		}
		else if (idx3 % 3 == 1) {

			cuS_y[idx_out].x = half_as_uint16_2float(xFFT_Data_xRegion[component_idx][(3 * idx_in + 1) * 2 + 0]) * normalization;
			cuS_y[idx_out].y = half_as_uint16_2float(xFFT_Data_xRegion[component_idx][(3 * idx_in + 1) * 2 + 1]) * normalization;
		}
		else if (idx3 % 3 == 2) {

			cuS_z[idx_out].x = half_as_uint16_2float(xFFT_Data_xRegion[component_idx][(3 * idx_in + 2) * 2 + 0]) * normalization;
			cuS_z[idx_out].y = half_as_uint16_2float(xFFT_Data_xRegion[component_idx][(3 * idx_in + 2) * 2 + 1]) * normalization;
		}
#else
		if (idx3 % 3 == 0) {

			cuS_x[idx_out].x = xFFT_Data_xRegion[component_idx][(3 * idx_in + 0) * 2 + 0] * normalization;
			cuS_x[idx_out].y = xFFT_Data_xRegion[component_idx][(3 * idx_in + 0) * 2 + 1] * normalization;
		}
		else if (idx3 % 3 == 1) {

			cuS_y[idx_out].x = xFFT_Data_xRegion[component_idx][(3 * idx_in + 1) * 2 + 0] * normalization;
			cuS_y[idx_out].y = xFFT_Data_xRegion[component_idx][(3 * idx_in + 1) * 2 + 1] * normalization;
		}
		else if (idx3 % 3 == 2) {

			cuS_z[idx_out].x = xFFT_Data_xRegion[component_idx][(3 * idx_in + 2) * 2 + 0] * normalization;
			cuS_z[idx_out].y = xFFT_Data_xRegion[component_idx][(3 * idx_in + 2) * 2 + 1] * normalization;
		}
#endif
	}
}

//cuOut has dimensions (Nx, nyRegion, nz). Above nx we are not interested in (zero).
__global__ void cu_copyoutputcomponents_kernel(
	cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuReal3** Out_Data_Components,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	//the component index to read from
	int component_idx;
	//form linear index in cuOut
	int idx_in;
	//form linear index in Out_Data_Components component
	int idx_out;

	if (from_notxRegion_inyRegion(idx, component_idx, idx_out, idx_in, n, N, xRegion_R, yRegion, num_devices)) {

		Out_Data_Components[component_idx][idx_out].x = cuOut_x[idx_in];
		Out_Data_Components[component_idx][idx_out].y = cuOut_y[idx_in];
		Out_Data_Components[component_idx][idx_out].z = cuOut_z[idx_in];
	}
}

__global__ void cu_copyoutputcomponents_kernel(
	cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuBHalf** Out_Data_Components, cuBReal& normalization,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices)
{
	int idx3 = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = idx3 / 3;

	//the component index to read from
	int component_idx;
	//form linear index in cuOut
	int idx_in;
	//form linear index in Out_Data_Components component
	int idx_out;

	if (from_notxRegion_inyRegion(idx, component_idx, idx_out, idx_in, n, N, xRegion_R, yRegion, num_devices)) {

#if SINGLEPRECISION
		//use a normalization constant to avoid exponent limit for half-precision
		if (idx3 % 3 == 0) Out_Data_Components[component_idx][3 * idx_out + 0] = float2half_as_uint16(cuOut_x[idx_in] / normalization);
		if (idx3 % 3 == 1) Out_Data_Components[component_idx][3 * idx_out + 1] = float2half_as_uint16(cuOut_y[idx_in] / normalization);
		if (idx3 % 3 == 2) Out_Data_Components[component_idx][3 * idx_out + 2] = float2half_as_uint16(cuOut_z[idx_in] / normalization);
#else
		if (idx3 % 3 == 0) Out_Data_Components[component_idx][3 * idx_out + 0] = cuOut_x[idx_in] / normalization;
		if (idx3 % 3 == 1) Out_Data_Components[component_idx][3 * idx_out + 1] = cuOut_y[idx_in] / normalization;
		if (idx3 % 3 == 2) Out_Data_Components[component_idx][3 * idx_out + 2] = cuOut_z[idx_in] / normalization;
#endif
	}
}

//---------------------------------------------------Copy input arrays from cuVEC or cuVEC_VC ( all <cuReal3> )

template <typename cuVECIn>
__global__ void InComponents_to_cuFFTArrays_forOutOfPlace_mGPU(
	cuVECIn& In, 
	cuReal3** M_Input_yRegion, 
	cuBReal* cuIn_x, cuBReal* cuIn_y, cuBReal* cuIn_z, 
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//form output index in cuIn spaces
	int idx_out;
	//form linear index in input data (In or M_Input_yRegion)
	int idx_in;
	//the component index to use in M_Input_yRegion, or if negative use In
	int component_idx;

	if (to_nx_in_yRegion(idx, idx_out, component_idx, idx_in, n, N, xRegion_R, yRegion, num_devices)) {

		if (component_idx < 0) {

			cuReal3 value_in = In[idx_in];
			cuIn_x[idx_out] = value_in.x;
			cuIn_y[idx_out] = value_in.y;
			cuIn_z[idx_out] = value_in.z;
		}
		else {
			
			cuReal3 value_in = M_Input_yRegion[component_idx][idx_in];
			cuIn_x[idx_out] = value_in.x;
			cuIn_y[idx_out] = value_in.y;
			cuIn_z[idx_out] = value_in.z;
		}
	}
}

template <typename cuVECIn>
__global__ void InComponents_to_cuFFTArrays_forOutOfPlace_mGPU(
	cuVECIn& In, 
	cuBHalf **M_Input_yRegion_half, cuBReal& normalization_M, 
	cuBReal* cuIn_x, cuBReal* cuIn_y, cuBReal* cuIn_z,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//form output index in cuIn spaces
	int idx_out;
	//form linear index in input data (In or M_Input_yRegion)
	int idx_in;
	//the component index to use in M_Input_yRegion, or if negative use In
	int component_idx;

	if (to_nx_in_yRegion(idx, idx_out, component_idx, idx_in, n, N, xRegion_R, yRegion, num_devices)) {
		
		if (component_idx < 0) {

			cuReal3 value_in = In[idx_in];
			cuIn_x[idx_out] = value_in.x;
			cuIn_y[idx_out] = value_in.y;
			cuIn_z[idx_out] = value_in.z;
		}
		else {

#if SINGLEPRECISION
			cuReal3 value_in = cuReal3(
				half_as_uint16_2float(M_Input_yRegion_half[component_idx][3 * idx_in + 0]),
				half_as_uint16_2float(M_Input_yRegion_half[component_idx][3 * idx_in + 1]),
				half_as_uint16_2float(M_Input_yRegion_half[component_idx][3 * idx_in + 2])) * normalization_M;
#else
			cuReal3 value_in = cuReal3(
				M_Input_yRegion_half[component_idx][3 * idx_in + 0], 
				M_Input_yRegion_half[component_idx][3 * idx_in + 1], 
				M_Input_yRegion_half[component_idx][3 * idx_in + 2]) * normalization_M;
#endif
			cuIn_x[idx_out] = value_in.x;
			cuIn_y[idx_out] = value_in.y;
			cuIn_z[idx_out] = value_in.z;
		}
	}
}

template <typename cuVECIn>
__global__ void InComponents_to_cuFFTArrays_forOutOfPlace_mGPU(
	cuVECIn& In1, cuVECIn& In2, 
	cuReal3** M_Input_yRegion, 
	cuBReal* cuIn_x, cuBReal* cuIn_y, cuBReal* cuIn_z,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//form output index in cuIn spaces
	int idx_out;
	//form linear index in input data (In or M_Input_yRegion)
	int idx_in;
	//the component index to use in M_Input_yRegion, or if negative use In
	int component_idx;

	if (to_nx_in_yRegion(idx, idx_out, component_idx, idx_in, n, N, xRegion_R, yRegion, num_devices)) {

		if (component_idx < 0) {

			cuReal3 value_in = (In1[idx_in] + In2[idx_in]) / 2;
			cuIn_x[idx_out] = value_in.x;
			cuIn_y[idx_out] = value_in.y;
			cuIn_z[idx_out] = value_in.z;
		}
		else {

			cuReal3 value_in = M_Input_yRegion[component_idx][idx_in];
			cuIn_x[idx_out] = value_in.x;
			cuIn_y[idx_out] = value_in.y;
			cuIn_z[idx_out] = value_in.z;
		}
	}
}

template <typename cuVECIn>
__global__ void InComponents_to_cuFFTArrays_forOutOfPlace_mGPU(
	cuVECIn& In1, cuVECIn& In2, 
	cuBHalf** M_Input_yRegion_half, cuBReal& normalization_M, 
	cuBReal* cuIn_x, cuBReal* cuIn_y, cuBReal* cuIn_z,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//form output index in cuIn spaces
	int idx_out;
	//form linear index in input data (In or M_Input_yRegion)
	int idx_in;
	//the component index to use in M_Input_yRegion, or if negative use In
	int component_idx;

	if (to_nx_in_yRegion(idx, idx_out, component_idx, idx_in, n, N, xRegion_R, yRegion, num_devices)) {

		if (component_idx < 0) {

			cuReal3 value_in = (In1[idx_in] + In2[idx_in]) / 2;
			cuIn_x[idx_out] = value_in.x;
			cuIn_y[idx_out] = value_in.y;
			cuIn_z[idx_out] = value_in.z;
		}
		else {

#if SINGLEPRECISION
			cuReal3 value_in = cuReal3(
				half_as_uint16_2float(M_Input_yRegion_half[component_idx][3 * idx_in + 0]),
				half_as_uint16_2float(M_Input_yRegion_half[component_idx][3 * idx_in + 1]),
				half_as_uint16_2float(M_Input_yRegion_half[component_idx][3 * idx_in + 2])) * normalization_M;
#else
			cuReal3 value_in = cuReal3(
				M_Input_yRegion_half[component_idx][3 * idx_in + 0], 
				M_Input_yRegion_half[component_idx][3 * idx_in + 1], 
				M_Input_yRegion_half[component_idx][3 * idx_in + 2]) * normalization_M;
#endif

			cuIn_x[idx_out] = value_in.x;
			cuIn_y[idx_out] = value_in.y;
			cuIn_z[idx_out] = value_in.z;
		}
	}
}

//---------------------------------------------------Set/Add cuVEC_VC inputs to output ( all <cuReal3> )

//SINGLE INPUT, SINGLE OUTPUT

template <typename cuVECOut>
__global__ void cuFFTArrays_to_Out_Add_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In, cuVECOut& Out, 
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {
			
			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * In[idx] * Heff_value / (2 * non_empty_cells);
		}

		Out[idx] += Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * (In[idx] * Heff_value) / 2;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_to_Out_Add_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In, cuVECOut& Out,
	cuBHalf** Out_Data_xRegion, cuBReal& normalization,
	cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

#if SINGLEPRECISION
			Heff_value = cuReal3(
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 0]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 1]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 2])) * normalization / N.dim();
#else
			Heff_value = cuReal3(
				Out_Data_xRegion[device_block][3 * idx_component + 0],
				Out_Data_xRegion[device_block][3 * idx_component + 1],
				Out_Data_xRegion[device_block][3 * idx_component + 2]) * normalization / N.dim();
#endif
		}

		if (do_reduction) {

			size_t non_empty_cells = In.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * In[idx] * Heff_value / (2 * non_empty_cells);
		}

		Out[idx] += Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * (In[idx] * Heff_value) / 2;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_to_Out_Set_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In, cuVECOut& Out,
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * In[idx] * Heff_value / (2 * non_empty_cells);
		}

		Out[idx] = Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * (In[idx] * Heff_value) / 2;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_to_Out_Set_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In, cuVECOut& Out,
	cuBHalf** Out_Data_xRegion, cuBReal& normalization,
	cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

#if SINGLEPRECISION
			Heff_value = cuReal3(
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 0]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 1]), 
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 2])) * normalization / N.dim();
#else
			Heff_value = cuReal3(
				Out_Data_xRegion[device_block][3 * idx_component + 0],
				Out_Data_xRegion[device_block][3 * idx_component + 1],
				Out_Data_xRegion[device_block][3 * idx_component + 2]) * normalization / N.dim();
#endif
		}

		if (do_reduction) {

			size_t non_empty_cells = In.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * In[idx] * Heff_value / (2 * non_empty_cells);
		}

		Out[idx] = Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * (In[idx] * Heff_value) / 2;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

//AVERAGED INPUTS, SINGLE OUTPUT

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Add_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVECOut& Out, 
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out[idx] += Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Add_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVECOut& Out,
	cuBHalf** Out_Data_xRegion, cuBReal& normalization, 
	cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

#if SINGLEPRECISION
			Heff_value = cuReal3(
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 0]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 1]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 2])) * normalization / N.dim();
#else
			Heff_value = cuReal3(
				Out_Data_xRegion[device_block][3 * idx_component + 0],
				Out_Data_xRegion[device_block][3 * idx_component + 1],
				Out_Data_xRegion[device_block][3 * idx_component + 2]) * normalization / N.dim();
#endif
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out[idx] += Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Set_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVECOut& Out, 
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out[idx] = Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Set_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVECOut& Out,
	cuBHalf** Out_Data_xRegion, cuBReal& normalization,
	cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

#if SINGLEPRECISION
			Heff_value = cuReal3(
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 0]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 1]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 2])) * normalization / N.dim();
#else
			Heff_value = cuReal3(
				Out_Data_xRegion[device_block][3 * idx_component + 0],
				Out_Data_xRegion[device_block][3 * idx_component + 1],
				Out_Data_xRegion[device_block][3 * idx_component + 2]) * normalization / N.dim();
#endif
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out[idx] = Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

//AVERAGED INPUTS, DUPLICATED OUTPUTS

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Duplicated_Add_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVECOut& Out1, cuVECOut& Out2, 
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out1.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out1.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out1[idx] += Heff_value;
		Out2[idx] += Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Duplicated_Add_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVECOut& Out1, cuVECOut& Out2,
	cuBHalf** Out_Data_xRegion, cuBReal& normalization,
	cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out1.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out1.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

#if SINGLEPRECISION
			Heff_value = cuReal3(
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 0]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 1]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 2])) * normalization / N.dim();
#else
			Heff_value = cuReal3(
				Out_Data_xRegion[device_block][3 * idx_component + 0],
				Out_Data_xRegion[device_block][3 * idx_component + 1],
				Out_Data_xRegion[device_block][3 * idx_component + 2]) * normalization / N.dim();
#endif
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out1[idx] += Heff_value;
		Out2[idx] += Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Duplicated_Set_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVECOut& Out1, cuVECOut& Out2, 
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out1.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out1.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out1[idx] = Heff_value;
		Out2[idx] = Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Duplicated_Set_mGPU_forOutOfPlace(
	cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVECOut& Out1, cuVECOut& Out2,
	cuBHalf** Out_Data_xRegion, cuBReal& normalization,
	cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out1.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out1.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

#if SINGLEPRECISION
			Heff_value = cuReal3(
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 0]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 1]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 2])) * normalization / N.dim();
#else
			Heff_value = cuReal3(
				Out_Data_xRegion[device_block][3 * idx_component + 0],
				Out_Data_xRegion[device_block][3 * idx_component + 1],
				Out_Data_xRegion[device_block][3 * idx_component + 2]) * normalization / N.dim();
#endif
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out1[idx] = Heff_value;
		Out2[idx] = Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

//---------------------------------------------------Set/Add cuVEC inputs to output ( all <cuReal3> )

//SINGLE INPUT, SINGLE OUTPUT

template <typename cuVECOut>
__global__ void cuFFTArrays_to_Out_Add_mGPU_forOutOfPlace(
	cuVEC<cuReal3>& In, cuVECOut& Out,
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In.get_aux_integer();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * In[idx] * Heff_value / (2 * non_empty_cells);
		}

		Out[idx] += Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * (In[idx] * Heff_value) / 2;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_to_Out_Set_mGPU_forOutOfPlace(
	cuVEC<cuReal3>& In, cuVECOut& Out,
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In.get_aux_integer();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * In[idx] * Heff_value / (2 * non_empty_cells);
		}

		Out[idx] = Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * (In[idx] * Heff_value) / 2;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_to_Out_Add_mGPU_forOutOfPlace(
	cuVEC<cuReal3>& In, cuVECOut& Out,
	cuBHalf** Out_Data_xRegion, cuBReal& normalization,
	cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

#if SINGLEPRECISION
			Heff_value = cuReal3(
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 0]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 1]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 2])) * normalization / N.dim();
#else
			Heff_value = cuReal3(
				Out_Data_xRegion[device_block][3 * idx_component + 0],
				Out_Data_xRegion[device_block][3 * idx_component + 1],
				Out_Data_xRegion[device_block][3 * idx_component + 2]) * normalization / N.dim();
#endif
		}

		if (do_reduction) {

			size_t non_empty_cells = In.get_aux_integer();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * In[idx] * Heff_value / (2 * non_empty_cells);
		}

		Out[idx] += Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * (In[idx] * Heff_value) / 2;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_to_Out_Set_mGPU_forOutOfPlace(
	cuVEC<cuReal3>& In, cuVECOut& Out,
	cuBHalf** Out_Data_xRegion, cuBReal& normalization,
	cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

#if SINGLEPRECISION
			Heff_value = cuReal3(
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 0]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 1]),
				half_as_uint16_2float(Out_Data_xRegion[device_block][3 * idx_component + 2])) * normalization / N.dim();
#else
			Heff_value = cuReal3(
				Out_Data_xRegion[device_block][3 * idx_component + 0],
				Out_Data_xRegion[device_block][3 * idx_component + 1],
				Out_Data_xRegion[device_block][3 * idx_component + 2]) * normalization / N.dim();
#endif
		}

		if (do_reduction) {

			size_t non_empty_cells = In.get_aux_integer();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * In[idx] * Heff_value / (2 * non_empty_cells);
		}

		Out[idx] = Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * (In[idx] * Heff_value) / 2;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

//AVERAGED INPUTS, SINGLE OUTPUT

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Add_mGPU_forOutOfPlace(
	cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVECOut& Out, 
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_aux_integer();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out[idx] += Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Set_mGPU_forOutOfPlace(
	cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVECOut& Out, 
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_aux_integer();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out[idx] = Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

//AVERAGED INPUTS, DUPLICATED OUTPUTS

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Duplicated_Add_mGPU_forOutOfPlace(
	cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVECOut& Out1, cuVECOut& Out2, 
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out1.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out1.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_aux_integer();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out1[idx] += Heff_value;
		Out2[idx] += Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

template <typename cuVECOut>
__global__ void cuFFTArrays_Averaged_to_Out_Duplicated_Set_mGPU_forOutOfPlace(
	cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVECOut& Out1, cuVECOut& Out2, 
	cuReal3** Out_Data_xRegion, cuBReal* cuOut_x, cuBReal* cuOut_y, cuBReal* cuOut_z,
	cuINT2& xRegion_R, cuINT2& yRegion, cuSZ3& N, int& num_devices, cuBReal& energy, bool do_reduction,
	cuVEC<cuReal3>* pH = nullptr, cuVEC<cuBReal>* penergy = nullptr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuSZ3 n = Out1.n;

	int nyRegion = yRegion.j - yRegion.i;

	cuBReal energy_ = 0.0;

	if (idx < Out1.linear_size()) {

		//ijk in cuOut space
		cuINT3 ijk = cuINT3(idx % n.x, (idx / n.x) % n.y, idx / (n.x * n.y));

		cuReal3 Heff_value;

		//inside yRegion : read from cuOut directly
		if (ijk.j >= yRegion.i && ijk.j < yRegion.j) {

			Heff_value.x = cuOut_x[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.y = cuOut_y[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
			Heff_value.z = cuOut_z[ijk.i + xRegion_R.i + (ijk.j - yRegion.i) * N.x + ijk.k * N.x * nyRegion] / N.dim();
		}
		//outside yRegion : read from respective Out_Data_xRegion
		else {

			//there are num_devices blocks in Out_Data_xRegion, so find which one this element belongs to
			int device_ycells_first = (n.y / num_devices);
			int device_ycells = device_ycells_first;
			int device_block = ijk.j / device_ycells_first;
			//last block can have more than device_ycells cells, so adjust
			if (device_block == num_devices) device_block--;
			if (device_block == num_devices - 1) device_ycells = n.y - device_ycells_first * (num_devices - 1);
			int device_yoffset = device_ycells_first * device_block;

			//form index in Out_Data_xRegion for device_block
			int idx_component = ijk.i + (ijk.j - device_yoffset) * n.x + ijk.k * n.x * device_ycells;

			Heff_value = Out_Data_xRegion[device_block][idx_component] / N.dim();
		}

		if (do_reduction) {

			size_t non_empty_cells = In1.get_aux_integer();
			if (non_empty_cells) energy_ = -(cuBReal)MU0 * (In1[idx] + In2[idx]) * Heff_value / (4 * non_empty_cells);
		}

		Out1[idx] = Heff_value;
		Out2[idx] = Heff_value;

		if (pH) (*pH)[idx] = Heff_value;
		if (penergy) (*penergy)[idx] = -(cuBReal)MU0 * ((In1[idx] + In2[idx]) * Heff_value) / 4;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, energy);
}

//-------------------------- RUN-TIME METHODS

template void ConvolutionDataCUDA::CopyInputData_mGPU(cuVEC<cuReal3>& In, cu_arr<cuReal3*>& M_Input_yRegion);
template void ConvolutionDataCUDA::CopyInputData_mGPU(cuVEC_VC<cuReal3>& In, cu_arr<cuReal3*>& M_Input_yRegion);

//Copy input data from array components collected from all devices in yRegion (but different x regions), as well as from In in the current xRegion and yRegion, to cuIn_x, cuIn_y, cuIn_z arrays at start of convolution iteration
template <typename cuVECIn>
void ConvolutionDataCUDA::CopyInputData_mGPU(cuVECIn& In, cu_arr<cuReal3*>& M_Input_yRegion)
{
	InComponents_to_cuFFTArrays_forOutOfPlace_mGPU<cuVECIn> <<< (n.x * nyRegion * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(In, M_Input_yRegion, cuIn_x, cuIn_y, cuIn_z, cun, cuN, cuxRegion_R, cuyRegion, cunum_devices);
	
	//if preserve_zero_padding mode is not enabled, then we need to launch zero padding kernels to zero the parts above n.z and n.y each iteration as zero padding will be polluted in cuS spaces.
	if (!preserve_zero_padding) {

		if (n.z > 1 && !q2D_level) {

			//with pbc enabled n.z will be equal to N.z so no need to launch zero padding kernel
			if (N.z - n.z) {

				cu_zeropad(nxRegion * N.y * (N.z - n.z), cuS_x, cuS_y, cuS_z, cuNc_xy, cuUpper_z_region);
			}
		}
		else if (n.z > 1 && q2D_level && n.z != N.z / 2) {

			//if using q2D level, make sure to zero pad from n.z up to N.z / 2 as these values might not be the same
			cu_zeropad(nxRegion * N.y * (N.z / 2 - n.z), cuS_x, cuS_y, cuS_z, cuNc_xy_q2d, cuUpper_z_region_q2d);
		}
	}
}

template void ConvolutionDataCUDA::CopyInputData_mGPU(cuVEC<cuReal3>& In, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M);
template void ConvolutionDataCUDA::CopyInputData_mGPU(cuVEC_VC<cuReal3>& In, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M);

template <typename cuVECIn>
void ConvolutionDataCUDA::CopyInputData_mGPU(cuVECIn& In, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M)
{
	InComponents_to_cuFFTArrays_forOutOfPlace_mGPU<cuVECIn> <<< (n.x * nyRegion * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(In, M_Input_yRegion_half, normalization_M, cuIn_x, cuIn_y, cuIn_z, cun, cuN, cuxRegion_R, cuyRegion, cunum_devices);
	
	//if preserve_zero_padding mode is not enabled, then we need to launch zero padding kernels to zero the parts above n.z and n.y each iteration as zero padding will be polluted in cuS spaces.
	if (!preserve_zero_padding) {

		if (n.z > 1 && !q2D_level) {

			//with pbc enabled n.z will be equal to N.z so no need to launch zero padding kernel
			if (N.z - n.z) {

				cu_zeropad(nxRegion * N.y * (N.z - n.z), cuS_x, cuS_y, cuS_z, cuNc_xy, cuUpper_z_region);
			}
		}
		else if (n.z > 1 && q2D_level && n.z != N.z / 2) {

			//if using q2D level, make sure to zero pad from n.z up to N.z / 2 as these values might not be the same
			cu_zeropad(nxRegion * N.y * (N.z / 2 - n.z), cuS_x, cuS_y, cuS_z, cuNc_xy_q2d, cuUpper_z_region_q2d);
		}
	}
}

template void ConvolutionDataCUDA::CopyInputData_AveragedInputs_mGPU(cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cu_arr<cuReal3*>& M_Input_yRegion);
template void ConvolutionDataCUDA::CopyInputData_AveragedInputs_mGPU(cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cu_arr<cuReal3*>& M_Input_yRegion);

template <typename cuVECIn>
void ConvolutionDataCUDA::CopyInputData_AveragedInputs_mGPU(cuVECIn& In1, cuVECIn& In2, cu_arr<cuReal3*>& M_Input_yRegion)
{
	InComponents_to_cuFFTArrays_forOutOfPlace_mGPU<cuVECIn> <<< (n.x * nyRegion * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(In1, In2, M_Input_yRegion, cuIn_x, cuIn_y, cuIn_z, cun, cuN, cuxRegion_R, cuyRegion, cunum_devices);

	//if preserve_zero_padding mode is not enabled, then we need to launch zero padding kernels to zero the parts above n.z and n.y each iteration as zero padding will be polluted in cuS spaces.
	if (!preserve_zero_padding) {

		//for 3D problems we also need to zero pad the upper z region (from n.z up to N.z)
		if (n.z > 1 && !q2D_level) {

			//with pbc enabled n.z will be equal to N.z so no need to launch zero padding kernel
			if (N.z - n.z) {

				cu_zeropad(nxRegion * N.y * (N.z - n.z), cuS_x, cuS_y, cuS_z, cuNc_xy, cuUpper_z_region);
			}
		}
		else if (n.z > 1 && q2D_level && n.z != N.z / 2) {

			//if using q2D level, make sure to zero pad from n.z up to N.z / 2 as these values might not be the same
			cu_zeropad(nxRegion * N.y * (N.z / 2 - n.z), cuS_x, cuS_y, cuS_z, cuNc_xy_q2d, cuUpper_z_region_q2d);
		}
	}
}

template void ConvolutionDataCUDA::CopyInputData_AveragedInputs_mGPU(cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M);
template void ConvolutionDataCUDA::CopyInputData_AveragedInputs_mGPU(cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M);

template <typename cuVECIn>
void ConvolutionDataCUDA::CopyInputData_AveragedInputs_mGPU(cuVECIn& In1, cuVECIn& In2, cu_arr<cuBHalf*>& M_Input_yRegion_half, cu_obj<cuBReal>& normalization_M)
{
	InComponents_to_cuFFTArrays_forOutOfPlace_mGPU<cuVECIn> <<< (n.x * nyRegion * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(In1, In2, M_Input_yRegion_half, normalization_M, cuIn_x, cuIn_y, cuIn_z, cun, cuN, cuxRegion_R, cuyRegion, cunum_devices);

	//if preserve_zero_padding mode is not enabled, then we need to launch zero padding kernels to zero the parts above n.z and n.y each iteration as zero padding will be polluted in cuS spaces.
	if (!preserve_zero_padding) {

		//for 3D problems we also need to zero pad the upper z region (from n.z up to N.z)
		if (n.z > 1 && !q2D_level) {

			//with pbc enabled n.z will be equal to N.z so no need to launch zero padding kernel
			if (N.z - n.z) {

				cu_zeropad(nxRegion * N.y * (N.z - n.z), cuS_x, cuS_y, cuS_z, cuNc_xy, cuUpper_z_region);
			}
		}
		else if (n.z > 1 && q2D_level && n.z != N.z / 2) {

			//if using q2D level, make sure to zero pad from n.z up to N.z / 2 as these values might not be the same
			cu_zeropad(nxRegion * N.y * (N.z / 2 - n.z), cuS_x, cuS_y, cuS_z, cuNc_xy_q2d, cuUpper_z_region_q2d);
		}
	}
}

//perform x step of 2D forward FFTs in current yRegion, after having copied in components from other devices, then transpose and copy data to xIFFT_Data_yRegion so we can transfer to other devices
void ConvolutionDataCUDA::forward_fft_2D_mGPU_xstep(cu_arr<cuBComplex*>& xIFFT_Data_yRegion)
{
	//Forward 3D FFT

	//do not interleave transpose_xy operations in mGPU mode
	//transpose_xy operation in a single call

	cufftR2C(plan2D_fwd_x, cuIn_x, cuSquart_x);
	cufftR2C(plan2D_fwd_x, cuIn_y, cuSquart_y);
	cufftR2C(plan2D_fwd_x, cuIn_z, cuSquart_z);

	//now transpose from cuSquart spaces to cuS spaces in current xRegion and yRegion
	//cuSquart spaces extend for the full N.x/2 + 1 size, and need to send data to other devices from the other x regions
	//thus in xIFFT_Data_yRegion place the other yRegions in order, such that each yRegion block appears in linear memory
	cu_transpose_xy_copycomponents_forward_kernel <<< ((N.x / 2 + 1) * nyRegion * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(*pcuSzpy_x, *pcuSzpy_y, *pcuSzpy_z,
		cuSquart_x, cuSquart_y, cuSquart_z,
		xIFFT_Data_yRegion,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	//with pbc enabled n.y will be equal to N.y so no need to launch zero padding kernel (also not needed if preserve_zero_padding mode is enabled)
	if (N.y - n.y && !preserve_zero_padding) {

		cu_zeropad(nxRegion * (N.y - n.y) * n.z, cuS_x, cuS_y, cuS_z, cuNc_yx, cuUpper_y_transposed_region);
	}
}

void ConvolutionDataCUDA::forward_fft_2D_mGPU_xstep(cu_arr<cuBHalf*>& xIFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization)
{
	//Forward 3D FFT

	//do not interleave transpose_xy operations in mGPU mode
	//transpose_xy operation in a single call

	cufftR2C(plan2D_fwd_x, cuIn_x, cuSquart_x);
	cufftR2C(plan2D_fwd_x, cuIn_y, cuSquart_y);
	cufftR2C(plan2D_fwd_x, cuIn_z, cuSquart_z);

	//now transpose from cuSquart spaces to cuS spaces in current xRegion and yRegion
	//cuSquart spaces extend for the full N.x/2 + 1 size, and need to send data to other devices from the other x regions
	//thus in xIFFT_Data_yRegion place the other yRegions in order, such that each yRegion block appears in linear memory
	cu_transpose_xy_copycomponents_forward_kernel <<< ((N.x / 2 + 1) * nyRegion * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(*pcuSzpy_x, *pcuSzpy_y, *pcuSzpy_z,
		cuSquart_x, cuSquart_y, cuSquart_z,
		xIFFT_Data_yRegion_half, normalization,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	//with pbc enabled n.y will be equal to N.y so no need to launch zero padding kernel (also not needed if preserve_zero_padding mode is enabled)
	if (N.y - n.y && !preserve_zero_padding) {

		cu_zeropad(nxRegion * (N.y - n.y) * n.z, cuS_x, cuS_y, cuS_z, cuNc_yx, cuUpper_y_transposed_region);
	}
}

//copy in data from x FFT obtained from other devices, so we can perform last steps of forward FFT
void ConvolutionDataCUDA::forward_fft_2D_mGPU_ystep(cu_arr<cuBComplex*>& xFFT_Data_xRegion)
{
	//before FFTs copy components from xFFT_Data_xRegion (fixed xRegion) into cuS in missing y regions as they've been transferred in from all other devices
	xFFT_Data_to_cuS_mGPU <<< (nxRegion * (n.y - nyRegion) * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(xFFT_Data_xRegion,
		*pcuSzpy_x, *pcuSzpy_y, *pcuSzpy_z,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	cufftC2C(plan2D_y, *pcuSzpy_x, cuS_x, CUFFT_FORWARD);
	cufftC2C(plan2D_y, *pcuSzpy_y, cuS_y, CUFFT_FORWARD);
	cufftC2C(plan2D_y, *pcuSzpy_z, cuS_z, CUFFT_FORWARD);
}

void ConvolutionDataCUDA::forward_fft_2D_mGPU_ystep(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization)
{
	//before FFTs copy components from xFFT_Data_xRegion (fixed xRegion) into cuS in missing y regions as they've been transferred in from all other devices
	xFFT_Data_to_cuS_mGPU <<< (nxRegion * (n.y - nyRegion) * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(xFFT_Data_xRegion_half, normalization,
		*pcuSzpy_x, *pcuSzpy_y, *pcuSzpy_z,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	cufftC2C(plan2D_y, *pcuSzpy_x, cuS_x, CUFFT_FORWARD);
	cufftC2C(plan2D_y, *pcuSzpy_y, cuS_y, CUFFT_FORWARD);
	cufftC2C(plan2D_y, *pcuSzpy_z, cuS_z, CUFFT_FORWARD);
}

//performy step of IFFT in 2D mode (before x IFFT need to transfer data between devices, which is the data copied to linear spaces xIFFT_Data_Components - this must have dimensions nxRegion * (corresponding nyRegion) * n.z)
void ConvolutionDataCUDA::inverse_fft_2D_mGPU_ystep(cu_arr<cuBComplex*>& xIFFT_Data_xRegion)
{
	//Inverse 2D FFT

	//transpose mode always used for mGPU in 2D
	//do not interleave transpose_xy operations in mGPU mode
	//transpose_xy operation in a single call

	if (!additional_spaces) {

		cufftC2C(plan2D_y, cuS_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan2D_y, cuS_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan2D_y, cuS_z, cuS_z, CUFFT_INVERSE);
	}
	else {

		cufftC2C(plan2D_y, cuS2_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan2D_y, cuS2_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan2D_y, cuS2_z, cuS_z, CUFFT_INVERSE);
	}

	//now transpose from cuS spaces to cuSquart spaces in current xRegion and yRegion
	//cuSquart spaces extend for the full N.x/2 + 1 size, and must receive data from other devices
	//thus in xIFFT_Data_xRegion place the other yRegions in order, such that each yRegion block appears in linear memory
	cu_transpose_xy_copycomponents_inverse_kernel <<< (nxRegion * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(cuS_x, cuS_y, cuS_z,
		cuSquart_x, cuSquart_y, cuSquart_z,
		xIFFT_Data_xRegion,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);
}

void ConvolutionDataCUDA::inverse_fft_2D_mGPU_ystep(cu_arr<cuBHalf*>& xIFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization)
{
	//Inverse 2D FFT

	//transpose mode always used for mGPU in 2D
	//do not interleave transpose_xy operations in mGPU mode
	//transpose_xy operation in a single call

	if (!additional_spaces) {

		cufftC2C(plan2D_y, cuS_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan2D_y, cuS_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan2D_y, cuS_z, cuS_z, CUFFT_INVERSE);
	}
	else {

		cufftC2C(plan2D_y, cuS2_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan2D_y, cuS2_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan2D_y, cuS2_z, cuS_z, CUFFT_INVERSE);
	}

	//now transpose from cuS spaces to cuSquart spaces in current xRegion and yRegion
	//cuSquart spaces extend for the full N.x/2 + 1 size, and must receive data from other devices
	//thus in xIFFT_Data_xRegion place the other yRegions in order, such that each yRegion block appears in linear memory
	cu_transpose_xy_copycomponents_inverse_kernel <<< (nxRegion * n.y * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(cuS_x, cuS_y, cuS_z,
		cuSquart_x, cuSquart_y, cuSquart_z,
		xIFFT_Data_xRegion_half, normalization,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);
}

//perform x IFFT step after copying in xIFFT data from all components not in the current xRegion; xIFFT_Data has size the total number of convolution components, but the one matching the current component doesn't need to be copied over
void ConvolutionDataCUDA::inverse_fft_2D_mGPU_xstep(cu_arr<cuBComplex*>& xIFFT_Data_yRegion, cu_arr<cuReal3*>& Out_Data_yRegion)
{
	//Inverse 2D FFT

	//copy components transferred from other devices (xIFFT_Data_yRegion) to cuSquart before x IFFT in yRegion
	xIFFT_Data_to_cuSquart_mGPU <<< ((N.x / 2 + 1 - nxRegion) * nyRegion * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(xIFFT_Data_yRegion,
		cuSquart_x, cuSquart_y, cuSquart_z,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	cufftC2R(plan2D_inv_x, cuSquart_x, cuOut_x);
	cufftC2R(plan2D_inv_x, cuSquart_y, cuOut_y);
	cufftC2R(plan2D_inv_x, cuSquart_z, cuOut_z);

	//copy data from cuOut to linear spaces Out_Data_yRegion so we can transfer them to respective devices
	cu_copyoutputcomponents_kernel <<< ((n.x - nxRegion_R) * nyRegion * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(cuOut_x, cuOut_y, cuOut_z,
		Out_Data_yRegion,
		cun, cuN, cuxRegion_R, cuyRegion, cunum_devices);
}

void ConvolutionDataCUDA::inverse_fft_2D_mGPU_xstep(cu_arr<cuBHalf*>& xIFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xIFFT, cu_arr<cuBHalf*>& Out_Data_yRegion_half, cu_obj<cuBReal>& normalization_Out)
{
	//Inverse 2D FFT

	//copy components transferred from other devices (xIFFT_Data_yRegion) to cuSquart before x IFFT in yRegion
	xIFFT_Data_to_cuSquart_mGPU <<< ((N.x / 2 + 1 - nxRegion) * nyRegion * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(xIFFT_Data_yRegion_half, normalization_xIFFT,
		cuSquart_x, cuSquart_y, cuSquart_z,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	cufftC2R(plan2D_inv_x, cuSquart_x, cuOut_x);
	cufftC2R(plan2D_inv_x, cuSquart_y, cuOut_y);
	cufftC2R(plan2D_inv_x, cuSquart_z, cuOut_z);

	//copy data from cuOut to linear spaces Out_Data_yRegion so we can transfer them to respective devices
	cu_copyoutputcomponents_kernel <<< ((n.x - nxRegion_R) * nyRegion * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(cuOut_x, cuOut_y, cuOut_z,
		Out_Data_yRegion_half, normalization_Out,
		cun, cuN, cuxRegion_R, cuyRegion, cunum_devices);
}

void ConvolutionDataCUDA::forward_fft_3D_mGPU_xstep(cu_arr<cuBComplex*>& xIFFT_Data_yRegion)
{
	//Forward 3D FFT

	//do not interleave transpose_xy operations in mGPU mode
	//transpose_xy operation in a single call

	cufftR2C(plan3D_fwd_x, cuIn_x, cuSquart_x);
	cufftR2C(plan3D_fwd_x, cuIn_y, cuSquart_y);
	cufftR2C(plan3D_fwd_x, cuIn_z, cuSquart_z);
	
	//now transpose from cuSquart spaces to cuS spaces in current xRegion and yRegion
	//cuSquart spaces extend for the full N.x/2 + 1 size, and need to send data to other devices from the other x regions
	//thus in xIFFT_Data_yRegion place the other yRegions in order, such that each yRegion block appears in linear memory
	cu_transpose_xy_copycomponents_forward_kernel <<< ((N.x / 2 + 1) * nyRegion * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(*pcuSzpy_x, *pcuSzpy_y, *pcuSzpy_z,
		cuSquart_x, cuSquart_y, cuSquart_z,
		xIFFT_Data_yRegion,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);
	
	//with pbc enabled n.y will be equal to N.y so no need to launch zero padding kernel (also not needed if preserve_zero_padding mode is enabled)
	if (N.y - n.y && !preserve_zero_padding) {

		cu_zeropad(nxRegion * (N.y - n.y) * n.z, cuS_x, cuS_y, cuS_z, cuNc_yx, cuUpper_y_transposed_region);
	}
}

void ConvolutionDataCUDA::forward_fft_3D_mGPU_xstep(cu_arr<cuBHalf*>& xIFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization)
{
	//Forward 3D FFT

	//do not interleave transpose_xy operations in mGPU mode
	//transpose_xy operation in a single call

	cufftR2C(plan3D_fwd_x, cuIn_x, cuSquart_x);
	cufftR2C(plan3D_fwd_x, cuIn_y, cuSquart_y);
	cufftR2C(plan3D_fwd_x, cuIn_z, cuSquart_z);

	//now transpose from cuSquart spaces to cuS spaces in current xRegion and yRegion
	//cuSquart spaces extend for the full N.x/2 + 1 size, and need to send data to other devices from the other x regions
	//thus in xIFFT_Data_yRegion place the other yRegions in order, such that each yRegion block appears in linear memory
	cu_transpose_xy_copycomponents_forward_kernel <<< ((N.x / 2 + 1) * nyRegion * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(*pcuSzpy_x, *pcuSzpy_y, *pcuSzpy_z,
		cuSquart_x, cuSquart_y, cuSquart_z,
		xIFFT_Data_yRegion_half, normalization,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	//with pbc enabled n.y will be equal to N.y so no need to launch zero padding kernel (also not needed if preserve_zero_padding mode is enabled)
	if (N.y - n.y && !preserve_zero_padding) {

		cu_zeropad(nxRegion * (N.y - n.y) * n.z, cuS_x, cuS_y, cuS_z, cuNc_yx, cuUpper_y_transposed_region);
	}
}

void ConvolutionDataCUDA::forward_fft_3D_mGPU_yzsteps(cu_arr<cuBComplex*>& xFFT_Data_xRegion)
{
	//before FFTs copy components from xFFT_Data_xRegion (fixed xRegion) into cuS in missing y regions as they've been transferred in from all other devices
	xFFT_Data_to_cuS_mGPU <<< (nxRegion * (n.y - nyRegion) * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(xFFT_Data_xRegion,
		*pcuSzpy_x, *pcuSzpy_y, *pcuSzpy_z,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	cufftC2C(plan3D_y, *pcuSzpy_x, *pcuSzpz_x, CUFFT_FORWARD);
	cufftC2C(plan3D_z, *pcuSzpz_x, cuS_x, CUFFT_FORWARD);

	cufftC2C(plan3D_y, *pcuSzpy_y, *pcuSzpz_y, CUFFT_FORWARD);
	cufftC2C(plan3D_z, *pcuSzpz_y, cuS_y, CUFFT_FORWARD);

	cufftC2C(plan3D_y, *pcuSzpy_z, *pcuSzpz_z, CUFFT_FORWARD);
	cufftC2C(plan3D_z, *pcuSzpz_z, cuS_z, CUFFT_FORWARD);
}

void ConvolutionDataCUDA::forward_fft_3D_mGPU_yzsteps(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization)
{
	//before FFTs copy components from xFFT_Data_xRegion (fixed xRegion) into cuS in missing y regions as they've been transferred in from all other devices
	xFFT_Data_to_cuS_mGPU <<< (nxRegion * (n.y - nyRegion) * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(xFFT_Data_xRegion_half, normalization,
		*pcuSzpy_x, *pcuSzpy_y, *pcuSzpy_z,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	cufftC2C(plan3D_y, *pcuSzpy_x, *pcuSzpz_x, CUFFT_FORWARD);
	cufftC2C(plan3D_z, *pcuSzpz_x, cuS_x, CUFFT_FORWARD);

	cufftC2C(plan3D_y, *pcuSzpy_y, *pcuSzpz_y, CUFFT_FORWARD);
	cufftC2C(plan3D_z, *pcuSzpz_y, cuS_y, CUFFT_FORWARD);

	cufftC2C(plan3D_y, *pcuSzpy_z, *pcuSzpz_z, CUFFT_FORWARD);
	cufftC2C(plan3D_z, *pcuSzpz_z, cuS_z, CUFFT_FORWARD);
}

void ConvolutionDataCUDA::forward_fft_q2D_mGPU_ystep(cu_arr<cuBComplex*>& xFFT_Data_xRegion)
{
	//before FFTs copy components from xFFT_Data_xRegion (fixed xRegion) into cuS in missing y regions as they've been transferred in from all other devices
	xFFT_Data_to_cuS_mGPU <<< (nxRegion * (n.y - nyRegion) * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(xFFT_Data_xRegion,
		*pcuSzpy_x, *pcuSzpy_y, *pcuSzpy_z,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	cufftC2C(plan3D_y, *pcuSzpy_x, cuS_x, CUFFT_FORWARD);
	cufftC2C(plan3D_y, *pcuSzpy_y, cuS_y, CUFFT_FORWARD);
	cufftC2C(plan3D_y, *pcuSzpy_z, cuS_z, CUFFT_FORWARD);
}

void ConvolutionDataCUDA::forward_fft_q2D_mGPU_ystep(cu_arr<cuBHalf*>& xFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization)
{
	//before FFTs copy components from xFFT_Data_xRegion (fixed xRegion) into cuS in missing y regions as they've been transferred in from all other devices
	xFFT_Data_to_cuS_mGPU <<< (nxRegion * (n.y - nyRegion) * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(xFFT_Data_xRegion_half, normalization,
		*pcuSzpy_x, *pcuSzpy_y, *pcuSzpy_z,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	cufftC2C(plan3D_y, *pcuSzpy_x, cuS_x, CUFFT_FORWARD);
	cufftC2C(plan3D_y, *pcuSzpy_y, cuS_y, CUFFT_FORWARD);
	cufftC2C(plan3D_y, *pcuSzpy_z, cuS_z, CUFFT_FORWARD);
}

//perform z and y steps of IFFT (before x IFFT need to transfer data between devices, which is the data copied to linear spaces xIFFT_Data_xRegion - this must have dimensions nxRegion * (corresponding nyRegion) * n.z)
void ConvolutionDataCUDA::inverse_fft_3D_mGPU_zysteps(cu_arr<cuBComplex*>& xIFFT_Data_xRegion)
{
	//do not interleave transpose_xy operations in mGPU mode
	//transpose_xy operation in a single call

	if (!additional_spaces) {

		cufftC2C(plan3D_z, cuS_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_x, cuS_x, CUFFT_INVERSE);

		cufftC2C(plan3D_z, cuS_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_y, cuS_y, CUFFT_INVERSE);

		cufftC2C(plan3D_z, cuS_z, cuS_z, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_z, cuS_z, CUFFT_INVERSE);
	}
	else {

		cufftC2C(plan3D_z, cuS2_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_x, cuS_x, CUFFT_INVERSE);

		cufftC2C(plan3D_z, cuS2_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_y, cuS_y, CUFFT_INVERSE);

		cufftC2C(plan3D_z, cuS2_z, cuS_z, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_z, cuS_z, CUFFT_INVERSE);
	}

	//now transpose from cuS spaces to cuSquart spaces in current xRegion and yRegion
	//cuSquart spaces extend for the full N.x/2 + 1 size, and must receive data from other devices
	//thus in xIFFT_Data_xRegion place the other yRegions in order, such that each yRegion block appears in linear memory
	cu_transpose_xy_copycomponents_inverse_kernel <<< (nxRegion * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(cuS_x, cuS_y, cuS_z, 
		cuSquart_x, cuSquart_y, cuSquart_z, 
		xIFFT_Data_xRegion,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);
}

void ConvolutionDataCUDA::inverse_fft_3D_mGPU_zysteps(cu_arr<cuBHalf*>& xIFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization)
{
	//do not interleave transpose_xy operations in mGPU mode
	//transpose_xy operation in a single call

	if (!additional_spaces) {

		cufftC2C(plan3D_z, cuS_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_x, cuS_x, CUFFT_INVERSE);

		cufftC2C(plan3D_z, cuS_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_y, cuS_y, CUFFT_INVERSE);

		cufftC2C(plan3D_z, cuS_z, cuS_z, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_z, cuS_z, CUFFT_INVERSE);
	}
	else {

		cufftC2C(plan3D_z, cuS2_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_x, cuS_x, CUFFT_INVERSE);

		cufftC2C(plan3D_z, cuS2_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_y, cuS_y, CUFFT_INVERSE);

		cufftC2C(plan3D_z, cuS2_z, cuS_z, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_z, cuS_z, CUFFT_INVERSE);
	}
	
	//now transpose from cuS spaces to cuSquart spaces in current xRegion and yRegion
	//cuSquart spaces extend for the full N.x/2 + 1 size, and must receive data from other devices
	//thus in xIFFT_Data_xRegion place the other yRegions in order, such that each yRegion block appears in linear memory
	cu_transpose_xy_copycomponents_inverse_kernel <<< (nxRegion * n.y * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(cuS_x, cuS_y, cuS_z,
		cuSquart_x, cuSquart_y, cuSquart_z,
		xIFFT_Data_xRegion_half, normalization,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);
}

//performy step of IFFT in q2D mode (before x IFFT need to transfer data between devices, which is the data copied to linear space xIFFT_Data - this must have dimensions nxRegion * (corresponding nyRegion * n.z)
void ConvolutionDataCUDA::inverse_fft_q2D_mGPU_ystep(cu_arr<cuBComplex*>& xIFFT_Data_xRegion)
{
	//Inverse 3D FFT

	//do not interleave transpose_xy operations in mGPU mode
	//transpose_xy operation in a single call

	if (!additional_spaces) {

		cufftC2C(plan3D_y, cuS_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_z, cuS_z, CUFFT_INVERSE);
	}
	else {

		cufftC2C(plan3D_y, cuS2_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS2_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS2_z, cuS_z, CUFFT_INVERSE);
	}

	//now transpose from cuS spaces to cuSquart spaces in current xRegion and yRegion
	//cuSquart spaces extend for the full N.x/2 + 1 size, and must receive data from other devices
	//thus in xIFFT_Data_xRegion place the other yRegions in order, such that each yRegion block appears in linear memory
	cu_transpose_xy_copycomponents_inverse_kernel <<< (nxRegion * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(cuS_x, cuS_y, cuS_z,
		cuSquart_x, cuSquart_y, cuSquart_z,
		xIFFT_Data_xRegion,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	//before x IFFT will need to transfer data between devices
}

void ConvolutionDataCUDA::inverse_fft_q2D_mGPU_ystep(cu_arr<cuBHalf*>& xIFFT_Data_xRegion_half, cu_obj<cuBReal>& normalization)
{
	//Inverse 3D FFT

	//do not interleave transpose_xy operations in mGPU mode
	//transpose_xy operation in a single call

	if (!additional_spaces) {

		cufftC2C(plan3D_y, cuS_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS_z, cuS_z, CUFFT_INVERSE);
	}
	else {

		cufftC2C(plan3D_y, cuS2_x, cuS_x, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS2_y, cuS_y, CUFFT_INVERSE);
		cufftC2C(plan3D_y, cuS2_z, cuS_z, CUFFT_INVERSE);
	}

	//now transpose from cuS spaces to cuSquart spaces in current xRegion and yRegion
	//cuSquart spaces extend for the full N.x/2 + 1 size, and must receive data from other devices
	//thus in xIFFT_Data_xRegion place the other yRegions in order, such that each yRegion block appears in linear memory
	cu_transpose_xy_copycomponents_inverse_kernel <<< (nxRegion * n.y * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(cuS_x, cuS_y, cuS_z,
		cuSquart_x, cuSquart_y, cuSquart_z,
		xIFFT_Data_xRegion_half, normalization,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	//before x IFFT will need to transfer data between devices
}

void ConvolutionDataCUDA::inverse_fft_3D_mGPU_xstep(cu_arr<cuBComplex*>& xIFFT_Data_yRegion, cu_arr<cuReal3*>& Out_Data_yRegion)
{
	//Inverse 3D FFT
	
	//copy components transferred from other devices (xIFFT_Data_yRegion) to cuSquart before x IFFT in yRegion
	xIFFT_Data_to_cuSquart_mGPU <<< ((N.x / 2 + 1 - nxRegion) * nyRegion * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(xIFFT_Data_yRegion, 
		cuSquart_x, cuSquart_y, cuSquart_z, 
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	cufftC2R(plan3D_inv_x, cuSquart_x, cuOut_x);
	cufftC2R(plan3D_inv_x, cuSquart_y, cuOut_y);
	cufftC2R(plan3D_inv_x, cuSquart_z, cuOut_z);

	//copy data from cuOut to linear spaces Out_Data_yRegion so we can transfer them to respective devices
	cu_copyoutputcomponents_kernel <<< ((n.x - nxRegion_R) * nyRegion * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(cuOut_x, cuOut_y, cuOut_z,
		Out_Data_yRegion,
		cun, cuN, cuxRegion_R, cuyRegion, cunum_devices);
}

void ConvolutionDataCUDA::inverse_fft_3D_mGPU_xstep(cu_arr<cuBHalf*>& xIFFT_Data_yRegion_half, cu_obj<cuBReal>& normalization_xIFFT, cu_arr<cuBHalf*>& Out_Data_yRegion_half, cu_obj<cuBReal>& normalization_Out)
{
	//Inverse 3D FFT

	//copy components transferred from other devices (xIFFT_Data_yRegion) to cuSquart before x IFFT in yRegion
	xIFFT_Data_to_cuSquart_mGPU <<< ((N.x / 2 + 1 - nxRegion) * nyRegion * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(xIFFT_Data_yRegion_half, normalization_xIFFT,
		cuSquart_x, cuSquart_y, cuSquart_z,
		cun, cuN, cuxRegion, cuyRegion, cunum_devices);

	cufftC2R(plan3D_inv_x, cuSquart_x, cuOut_x);
	cufftC2R(plan3D_inv_x, cuSquart_y, cuOut_y);
	cufftC2R(plan3D_inv_x, cuSquart_z, cuOut_z);

	//copy data from cuOut to linear spaces Out_Data_yRegion so we can transfer them to respective devices
	cu_copyoutputcomponents_kernel <<< ((n.x - nxRegion_R) * nyRegion * n.z * 3 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
		(cuOut_x, cuOut_y, cuOut_z,
		Out_Data_yRegion_half, normalization_Out,
		cun, cuN, cuxRegion_R, cuyRegion, cunum_devices);
}

//SINGLE INPUT, SINGLE OUTPUT

template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(
	cu_arr<cuReal3*>& Out_Data_xRegion,
	cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_to_Out_Set_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In, Out,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_to_Out_Set_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In, Out,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(
	cu_arr<cuReal3*>& Out_Data_xRegion,
	cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {
		
		cuFFTArrays_to_Out_Add_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In, Out,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_to_Out_Add_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In, Out,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC<cuReal3>& In, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC<cuReal3>& In, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(
	cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
	cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_to_Out_Set_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In, Out,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_to_Out_Set_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In, Out,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC<cuReal3>& In, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC<cuReal3>& In, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(
	cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
	cuVECIn& In, cuVECOut& Out, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_to_Out_Add_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In, Out,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_to_Out_Add_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In, Out,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

//AVERAGED INPUTS, SINGLE OUTPUT

template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

//same as above but for averaged input and duplicated output
template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(
	cu_arr<cuReal3*>& Out_Data_xRegion,
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_Averaged_to_Out_Set_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_Averaged_to_Out_Set_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(
	cu_arr<cuReal3*>& Out_Data_xRegion,
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_Averaged_to_Out_Add_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_Averaged_to_Out_Add_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

//same as above but for averaged input and duplicated output
template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(
	cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_Averaged_to_Out_Set_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_Averaged_to_Out_Set_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(
	cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_Averaged_to_Out_Add_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_Averaged_to_Out_Add_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

//AVERAGED INPUTS, DUPLICATED OUTPUTS

template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVEC<cuReal3>& Out1, cuVEC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out1, cuVEC_VC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC<cuReal3>& Out1, cuVEC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out1, cuVEC_VC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

//same as above but for averaged input and duplicated output
template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(
	cu_arr<cuReal3*>& Out_Data_xRegion,
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_Averaged_to_Out_Duplicated_Set_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out1, Out2,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_Averaged_to_Out_Duplicated_Set_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out1, Out2,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVEC<cuReal3>& Out1, cuVEC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC<cuReal3>& In1, cuVEC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out1, cuVEC_VC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC<cuReal3>& Out1, cuVEC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuReal3*>& Out_Data_xRegion, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out1, cuVEC_VC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(
	cu_arr<cuReal3*>& Out_Data_xRegion,
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_Averaged_to_Out_Duplicated_Add_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out1, Out2,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_Averaged_to_Out_Duplicated_Add_mGPU_forOutOfPlace <<< (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(In1, In2, Out1, Out2,
			Out_Data_xRegion, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC<cuReal3>& Out1, cuVEC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out1, cuVEC_VC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

//same as above but for averaged input and duplicated output
template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Set_mGPU(
	cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_Averaged_to_Out_Duplicated_Set_mGPU_forOutOfPlace << < (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(In1, In2, Out1, Out2,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_Averaged_to_Out_Duplicated_Set_mGPU_forOutOfPlace << < (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(In1, In2, Out1, Out2,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC<cuReal3>& Out1, cuVEC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);
template void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization, cuVEC_VC<cuReal3>& In1, cuVEC_VC<cuReal3>& In2, cuVEC_VC<cuReal3>& Out1, cuVEC_VC<cuReal3>& Out2, cuBReal& energy, bool get_energy, cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy);

template <typename cuVECIn, typename cuVECOut>
void ConvolutionDataCUDA::FinishConvolution_Add_mGPU(
	cu_arr<cuBHalf*>& Out_Data_xRegion_half, cu_obj<cuBReal>& normalization,
	cuVECIn& In1, cuVECIn& In2, cuVECOut& Out1, cuVECOut& Out2, cuBReal& energy, bool get_energy,
	cuVEC<cuReal3>* pH, cuVEC<cuBReal>* penergy)
{
	if (pH && penergy) {

		cuFFTArrays_Averaged_to_Out_Duplicated_Add_mGPU_forOutOfPlace << < (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(In1, In2, Out1, Out2,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy, pH, penergy);
	}
	else {

		cuFFTArrays_Averaged_to_Out_Duplicated_Add_mGPU_forOutOfPlace << < (nxRegion_R * n.y * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(In1, In2, Out1, Out2,
			Out_Data_xRegion_half, normalization, cuOut_x, cuOut_y, cuOut_z,
			cuxRegion_R, cuyRegion, cuN, cunum_devices, energy, get_energy);
	}
}

#endif
