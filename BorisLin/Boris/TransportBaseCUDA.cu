#include "TransportBaseCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "Reduction.cuh"

#include "MeshBaseCUDA.h"

__global__ void ZeroAux_kernel(cuBReal& auxReal)
{
	if (threadIdx.x == 0) auxReal = 0.0;
}

//--------------------------------------------------------------- Electrode Current

__global__ void CalculateElectrodeCurrent_X_Kernel(cuVEC_VC<cuBReal>& elC, cuVEC_VC<cuReal3>& E, cuBReal& current, cuBox electrode_box, int sign)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = electrode_box.e.j - electrode_box.s.j;

	//negative x side of box, parse j and k
	cuINT3 ijk = cuINT3(electrode_box.s.i, (idx % stride) + electrode_box.s.j, (idx / stride) + electrode_box.s.k);

	cuReal3 h_e = E.h;

	cuBReal current_ = 0.0;

	if (idx < stride * (electrode_box.e.k - electrode_box.s.k)) {

		current_ = sign * elC[ijk] * E[ijk].x * h_e.y * h_e.z;
	}

	reduction_sum(0, 1, &current_, current);
}

__global__ void CalculateElectrodeCurrent_Y_Kernel(cuVEC_VC<cuBReal>& elC, cuVEC_VC<cuReal3>& E, cuBReal& current, cuBox electrode_box, int sign)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = electrode_box.e.i - electrode_box.s.i;

	//negative y side of box, parse i and k
	cuINT3 ijk = cuINT3((idx % stride) + electrode_box.s.i, electrode_box.s.j, (idx / stride) + electrode_box.s.k);

	cuReal3 h_e = E.h;

	cuBReal current_ = 0.0;

	if (idx < stride * (electrode_box.e.k - electrode_box.s.k)) {

		current_ = sign * elC[ijk] * E[ijk].y * h_e.x * h_e.z;
	}

	reduction_sum(0, 1, &current_, current);
}

__global__ void CalculateElectrodeCurrent_Z_Kernel(cuVEC_VC<cuBReal>& elC, cuVEC_VC<cuReal3>& E, cuBReal& current, cuBox electrode_box, int sign)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = electrode_box.e.i - electrode_box.s.i;

	//negative z side of box, parse i and j
	cuINT3 ijk = cuINT3((idx % stride) + electrode_box.s.i, (idx / stride) + electrode_box.s.j, electrode_box.s.k);

	cuReal3 h_e = E.h;

	cuBReal current_ = 0.0;

	if (idx < stride * (electrode_box.e.j - electrode_box.s.j)) {

		current_ = sign * elC[ijk] * E[ijk].z * h_e.x * h_e.y;
	}

	reduction_sum(0, 1, &current_, current);
}

cuBReal TransportBaseCUDA::CalculateElectrodeCurrent(cuBox electrode_box, cuINT3 sign)
{
	//calculate current from current density in cells just next to the box
	//Normally there is only one side of the box we can use so it's easier to separate into multiple kernels - one per side.

	//Obtain the current by reduction in the energy value

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		ZeroAux_kernel <<< 1, CUDATHREADS >>>
			(auxReal(mGPU));
	}

	//cells on -x side
	if (sign.x) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			cuBox electrode_sbox = pMeshBaseCUDA->elC.device_sub_box(electrode_box, mGPU);
			size_t ker_size = (electrode_sbox.e.j - electrode_sbox.s.j) * (electrode_sbox.e.k - electrode_sbox.s.k);

			CalculateElectrodeCurrent_X_Kernel <<< (ker_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(pMeshBaseCUDA->elC.get_deviceobject(mGPU), pMeshBaseCUDA->E.get_deviceobject(mGPU), auxReal(mGPU), electrode_sbox, sign.x);
		}
	}

	//cells on -y side
	if (sign.y) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			cuBox electrode_sbox = pMeshBaseCUDA->elC.device_sub_box(electrode_box, mGPU);
			size_t ker_size = (electrode_sbox.e.i - electrode_sbox.s.i) * (electrode_sbox.e.k - electrode_sbox.s.k);

			CalculateElectrodeCurrent_Y_Kernel <<< (ker_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(pMeshBaseCUDA->elC.get_deviceobject(mGPU), pMeshBaseCUDA->E.get_deviceobject(mGPU), auxReal(mGPU), electrode_sbox, sign.y);
		}
	}

	//cells on -z side
	if (sign.z) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			cuBox electrode_sbox = pMeshBaseCUDA->elC.device_sub_box(electrode_box, mGPU);
			size_t ker_size = (electrode_sbox.e.i - electrode_sbox.s.i) * (electrode_sbox.e.j - electrode_sbox.s.j);

			CalculateElectrodeCurrent_Z_Kernel <<< (ker_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(pMeshBaseCUDA->elC.get_deviceobject(mGPU), pMeshBaseCUDA->E.get_deviceobject(mGPU), auxReal(mGPU), electrode_sbox, sign.z);
		}
	}

	//energy has the current value; reset it after as we don't want to count it to the total energy density
	double current = auxReal.to_cpu_sum();

	return current;
}

#endif

#endif