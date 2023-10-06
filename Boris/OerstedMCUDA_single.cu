#include "OerstedMCUDA_single.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_OERSTED

#include "BorisCUDALib.cuh"

#include "OerstedCUDA.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//used to copy data from xRegion_R * ny * nz outside of yRegion to appropriate component
//this gets component_idx, idx_out (output index in component), and idx_in (input index)
__device__ bool from_notyRegion_in_xRegion_R_Oersted(
	int& idx, int& component_idx, int& idx_out, int& idx_in,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices)
{
	int nxRegion_R = xRegion_R.j - xRegion_R.i;
	int nyRegion = yRegion.j - yRegion.i;
	int nCy = n.y - nyRegion;

	if (idx < nxRegion_R * nCy * n.z) {

		//more than 2 components, use the general routine
		if (num_devices > 2) {

			//form ijk in complement space, noting that kernel was launched with (ny - nyRegion) cells along y, since nyRegion data does not need to be transferred
			cuINT3 ijk = cuINT3(idx % nxRegion_R, (idx / nxRegion_R) % nCy, idx / (nxRegion_R * nCy));

			//find which component we need to use
			//if yRegion is not the last region, then yRegion.j - yRegion.i is the n.y value common to all components apart from the last one
			//In this case the last component has n.y value of n.y - nyRegion * (num_devices - 1)
			//If yRegion is the last region, then the common n.y value of components before it is yRegion.i / (num_devices - 1)

			//yRegion is the last region
			if (yRegion.j == n.y) {

				//y dimension of components (they are all the same in this case as only component in last region can be different)
				int component_ny = yRegion.i / (num_devices - 1);
				component_idx = ijk.j / component_ny;
				int component_j = ijk.j % component_ny;

				idx_out = ijk.i + component_j * nxRegion_R + ijk.k * nxRegion_R * component_ny;
				idx_in = ijk.i + ijk.j * nxRegion_R + ijk.k * nxRegion_R * n.y;
			}
			//yRegion is not the last region
			else {

				//component is below yRegion, hence itself not the last region
				if (ijk.j < yRegion.i) {

					int component_ny = nyRegion;
					component_idx = ijk.j / component_ny;
					int component_j = ijk.j % component_ny;

					idx_out = ijk.i + component_j * nxRegion_R + ijk.k * nxRegion_R * component_ny;
					idx_in = ijk.i + ijk.j * nxRegion_R + ijk.k * nxRegion_R * n.y;
				}
				//component is above yRegion, and could be the last region
				else {

					//y index is : ijk.j + yRegion.j - yRegion.i (i.e. skip over yRegion)
					component_idx = (ijk.j + nyRegion) / nyRegion;
					if (component_idx == num_devices) component_idx--;
					int component_j = (ijk.j + nyRegion) - nyRegion * component_idx;
					int component_ny;

					//if this is the last component, it is also the last region, hence find its ny value as :
					if (component_idx == num_devices - 1) component_ny = n.y - nyRegion * (num_devices - 1);
					//otherwise we have its ny value already as:
					else component_ny = nyRegion;

					idx_out = ijk.i + component_j * nxRegion_R + ijk.k * nxRegion_R * component_ny;
					idx_in = ijk.i + (ijk.j + nyRegion) * nxRegion_R + ijk.k * nxRegion_R * n.y;
				}
			}
		}
		//if number of components is 2 use a simpler routine
		else if (num_devices == 2) {

			//ijk in complement space (nxRegion_R, nCy, nz)
			cuINT3 ijk = cuINT3(idx % nxRegion_R, (idx / nxRegion_R) % nCy, idx / (nxRegion_R * nCy));

			if (yRegion.j == n.y) {

				component_idx = 0;
				idx_in = ijk.i + ijk.j * nxRegion_R + ijk.k * nxRegion_R * n.y;
			}
			else {

				component_idx = 1;
				idx_in = ijk.i + (ijk.j + yRegion.j) * nxRegion_R + ijk.k * nxRegion_R * n.y;
			}

			idx_out = idx;
		}

		return true;
	}
	else return false;
}

template <typename VECType>
__global__ void Copy_J_Input_xRegion_Oersted_kernel(
	VECType& J, cuReal3** J_Input_xRegion,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//the component index to use
	int component_idx;
	//form output index in component spaces, which are of dimension nxRegion * (respective y region size) * nz
	int idx_out;
	//form linear index in input data, which is of dimensions (nxRegion * ny * nz)
	int idx_in;

	if (from_notyRegion_in_xRegion_R_Oersted(idx, component_idx, idx_out, idx_in, n, N, xRegion_R, yRegion, num_devices))
		J_Input_xRegion[component_idx][idx_out] = J[idx_in];
}

template <typename VECType>
__global__ void Copy_J_Input_xRegion_Oersted_halfprecision_kernel(
	VECType& J, cuBHalf** J_Input_xRegion_half,
	cuSZ3& n, cuSZ3& N, cuINT2& xRegion_R, cuINT2& yRegion, int& num_devices, cuBReal& normalization)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//the component index to use
	int component_idx;
	//form output index in component spaces, which are of dimension nxRegion * (respective y region size) * nz
	int idx_out;
	//form linear index in input data, which is of dimensions (nxRegion * ny * nz)
	int idx_in;

	if (from_notyRegion_in_xRegion_R_Oersted(idx, component_idx, idx_out, idx_in, n, N, xRegion_R, yRegion, num_devices)) {

		cuReal3 value_in = J[idx_in] / normalization;

#if SINGLEPRECISION
		J_Input_xRegion_half[component_idx][3 * idx_out + 0] = float2half_as_uint16(value_in.x);
		J_Input_xRegion_half[component_idx][3 * idx_out + 1] = float2half_as_uint16(value_in.y);
		J_Input_xRegion_half[component_idx][3 * idx_out + 2] = float2half_as_uint16(value_in.z);
#else
		J_Input_xRegion_half[component_idx][3 * idx_out + 0] = value_in.x;
		J_Input_xRegion_half[component_idx][3 * idx_out + 1] = value_in.y;
		J_Input_xRegion_half[component_idx][3 * idx_out + 2] = value_in.z;
#endif
	}
}

//Copy J data on this device to linear regions so we can transfer
void OerstedMCUDA_single::Copy_J_Input_xRegion(bool half_precision)
{
	if (!half_precision) {

		Copy_J_Input_xRegion_Oersted_kernel <<< (nxRegion_R * (n.y - nyRegion) * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pOerstedCUDA->sm_Vals.get_deviceobject(device_index), Real_xRegion_arr, cun, cuN, cuxRegion_R, cuyRegion, cunum_devices);
	}
	else {

		Copy_J_Input_xRegion_Oersted_halfprecision_kernel <<< (nxRegion_R * (n.y - nyRegion) * n.z + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pOerstedCUDA->sm_Vals.get_deviceobject(device_index), Real_xRegion_half_arr, cun, cuN, cuxRegion_R, cuyRegion, cunum_devices, normalization_J);
	}
}

#endif
#endif