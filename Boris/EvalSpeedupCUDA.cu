#include "EvalSpeedupCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_DEMAG) || defined(MODULE_COMPILATION_SDEMAG) || defined(MODULE_COMPILATION_ATOM_DIPOLEDIPOLE)

#include "BorisCUDALib.cuh"

//----------------------- EVALUATION FINISH

__global__ void EvalSpeedup_SubSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] -= (selfDemagCoeff & M[idx]);
	}
}

//Add newly computed field to Heff, then subtract self demag contribution from it : FM
__global__ void EvalSpeedup_AddField_SubSelf_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& H,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		Heff[idx] += H[idx];

		H[idx] -= (selfDemagCoeff & M[idx]);
	}
}

//Add newly computed field to Heff and Heff2, then subtract self demag contribution from it : AFM
__global__ void EvalSpeedup_AddField_SubSelf_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& H,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		Heff[idx] += H[idx];
		Heff2[idx] += H[idx];

		H[idx] -= (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
	}
}

// ---------------------- - LAUNCHERS EVALUATION FINISH

//subtract self demag contribution from H, using M
//used with EVALSPEEDUP_MODE_ATOM
void EvalSpeedupCUDA::EvalSpeedup_SubSelf(mcu_VEC(cuReal3)&H)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SubSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			pM_cuVEC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//similar to above, but the self contribution to subtract is given in transfer (used for multi-convolution)
void EvalSpeedupCUDA::EvalSpeedup_SubSelf(mcu_VEC(cuReal3)& H, mcu_VEC(cuReal3)& transfer)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SubSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU),
			transfer.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//subtract self demag contribution from H, using M, after adding H to Heff (pH_cuVEC)
//used with EVALSPEEDUP_MODE_FM
void EvalSpeedupCUDA::EvalSpeedup_AddField_SubSelf_FM(mcu_VEC(cuReal3)& H)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddField_SubSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			H.get_deviceobject(mGPU), 
			pM_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//used with EVALSPEEDUP_MODE_AFM
void EvalSpeedupCUDA::EvalSpeedup_AddField_SubSelf_AFM(mcu_VEC(cuReal3)& H)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddField_SubSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), pH2_cuVEC->get_deviceobject(mGPU), 
			H.get_deviceobject(mGPU), 
			pM_cuVEC_VC->get_deviceobject(mGPU), pM2_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//----------------------- EXTRAPOLATION METHODS (SET METHODS)

//QUINTIC
__global__ void EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5, cuVEC<cuReal3>& Hdemag6,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + Hdemag6[idx] * a6 + (selfDemagCoeff & M[idx]);
	}
}

//QUARTIC
__global__ void EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + (selfDemagCoeff & M[idx]);
	}
}

//CUBIC
__global__ void EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + (selfDemagCoeff & M[idx]);
	}
}

//QUADRATIC
__global__ void EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3,
	cuBReal a1, cuBReal a2, cuBReal a3,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + (selfDemagCoeff & M[idx]);
	}
}

//LINEAR
__global__ void EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2,
	cuBReal a1, cuBReal a2,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + (selfDemagCoeff & M[idx]);
	}
}

//----------------------- LAUNCHERS EXTRAPOLATION METHODS (SET METHODS)

//QUINTIC
void EvalSpeedupCUDA::EvalSpeedup_SetExtrapField_AddSelf(
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), Hdemag6.get_deviceobject(mGPU), 
			a1, a2, a3, a4, a5, a6, 
			pM_cuVEC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUARTIC
void EvalSpeedupCUDA::EvalSpeedup_SetExtrapField_AddSelf(
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), 
			a1, a2, a3, a4, a5, 
			pM_cuVEC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//CUBIC
void EvalSpeedupCUDA::EvalSpeedup_SetExtrapField_AddSelf(
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), 
			a1, a2, a3, a4, 
			pM_cuVEC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUADRATIC
void EvalSpeedupCUDA::EvalSpeedup_SetExtrapField_AddSelf(
	cuBReal a1, cuBReal a2, cuBReal a3)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), 
			a1, a2, a3, 
			pM_cuVEC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//LINEAR
void EvalSpeedupCUDA::EvalSpeedup_SetExtrapField_AddSelf(
	cuBReal a1, cuBReal a2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), 
			a1, a2, 
			pM_cuVEC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//----------------------- EXTRAPOLATION METHODS (ADD METHODS FOR FM)

//QUINTIC
__global__ void EvalSpeedup_AddExtrapField_AddSelf_FM_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5, cuVEC<cuReal3>& Hdemag6,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		Heff[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + Hdemag6[idx] * a6 + (selfDemagCoeff & M[idx]);
	}
}

//QUARTIC
__global__ void EvalSpeedup_AddExtrapField_AddSelf_FM_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		Heff[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + (selfDemagCoeff & M[idx]);
	}
}

//CUBIC
__global__ void EvalSpeedup_AddExtrapField_AddSelf_FM_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		Heff[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + (selfDemagCoeff & M[idx]);
	}
}

//QUADRATIC
__global__ void EvalSpeedup_AddExtrapField_AddSelf_FM_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3,
	cuBReal a1, cuBReal a2, cuBReal a3,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		Heff[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + (selfDemagCoeff & M[idx]);
	}
}

//LINEAR
__global__ void EvalSpeedup_AddExtrapField_AddSelf_FM_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2,
	cuBReal a1, cuBReal a2,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		Heff[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + (selfDemagCoeff & M[idx]);
	}
}

//STEP
__global__ void EvalSpeedup_AddField_FM_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Hdemag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		Heff[idx] += Hdemag[idx];
	}
}

//----------------------- LAUNCHERS EXTRAPOLATION METHODS (ADD METHODS FOR FM)

//QUINTIC
void EvalSpeedupCUDA::EvalSpeedup_AddExtrapField_AddSelf_FM(
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddExtrapField_AddSelf_FM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), Hdemag6.get_deviceobject(mGPU), 
			a1, a2, a3, a4, a5, a6, 
			pM_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUARTIC
void EvalSpeedupCUDA::EvalSpeedup_AddExtrapField_AddSelf_FM(
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddExtrapField_AddSelf_FM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), 
			a1, a2, a3, a4, a5, 
			pM_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//CUBIC
void EvalSpeedupCUDA::EvalSpeedup_AddExtrapField_AddSelf_FM(
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddExtrapField_AddSelf_FM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), 
			a1, a2, a3, a4, 
			pM_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUADRATIC
void EvalSpeedupCUDA::EvalSpeedup_AddExtrapField_AddSelf_FM(
	cuBReal a1, cuBReal a2, cuBReal a3)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddExtrapField_AddSelf_FM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), 
			a1, a2, a3, 
			pM_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//LINEAR
void EvalSpeedupCUDA::EvalSpeedup_AddExtrapField_AddSelf_FM(
	cuBReal a1, cuBReal a2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddExtrapField_AddSelf_FM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), 
			a1, a2, 
			pM_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//STEP
void EvalSpeedupCUDA::EvalSpeedup_AddField_FM(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddField_FM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU));
	}
}

//----------------------- EXTRAPOLATION METHODS (ADD METHODS FOR AFM)

//QUINTIC
__global__ void EvalSpeedup_AddExtrapField_AddSelf_AFM_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5, cuVEC<cuReal3>& Hdemag6,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + Hdemag6[idx] * a6 + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//QUARTIC
__global__ void EvalSpeedup_AddExtrapField_AddSelf_AFM_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//CUBIC
__global__ void EvalSpeedup_AddExtrapField_AddSelf_AFM_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//QUADRATIC
__global__ void EvalSpeedup_AddExtrapField_AddSelf_AFM_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3,
	cuBReal a1, cuBReal a2, cuBReal a3,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//LINEAR
__global__ void EvalSpeedup_AddExtrapField_AddSelf_AFM_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2,
	cuBReal a1, cuBReal a2,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//STEP
__global__ void EvalSpeedup_AddField_AFM_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2, cuVEC<cuReal3>& Hdemag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff.linear_size()) {

		Heff[idx] += Hdemag[idx];
		Heff2[idx] += Hdemag[idx];
	}
}

//----------------------- LAUNCHERS EXTRAPOLATION METHODS (ADD METHODS FOR AFM)

//QUINTIC
void EvalSpeedupCUDA::EvalSpeedup_AddExtrapField_AddSelf_AFM(
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddExtrapField_AddSelf_AFM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), pH2_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), Hdemag6.get_deviceobject(mGPU), 
			a1, a2, a3, a4, a5, a6, 
			pM_cuVEC_VC->get_deviceobject(mGPU), pM2_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUARTIC
void EvalSpeedupCUDA::EvalSpeedup_AddExtrapField_AddSelf_AFM(
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddExtrapField_AddSelf_AFM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), pH2_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), 
			a1, a2, a3, a4, a5, 
			pM_cuVEC_VC->get_deviceobject(mGPU), pM2_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//CUBIC
void EvalSpeedupCUDA::EvalSpeedup_AddExtrapField_AddSelf_AFM(
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddExtrapField_AddSelf_AFM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), pH2_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), 
			a1, a2, a3, a4, 
			pM_cuVEC_VC->get_deviceobject(mGPU), pM2_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUADRATIC
void EvalSpeedupCUDA::EvalSpeedup_AddExtrapField_AddSelf_AFM(
	cuBReal a1, cuBReal a2, cuBReal a3)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddExtrapField_AddSelf_AFM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), pH2_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), 
			a1, a2, a3, 
			pM_cuVEC_VC->get_deviceobject(mGPU), pM2_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//LINEAR
void EvalSpeedupCUDA::EvalSpeedup_AddExtrapField_AddSelf_AFM(
	cuBReal a1, cuBReal a2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddExtrapField_AddSelf_AFM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), pH2_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), 
			a1, a2, 
			pM_cuVEC_VC->get_deviceobject(mGPU), pM2_cuVEC_VC->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//STEP
void EvalSpeedupCUDA::EvalSpeedup_AddField_AFM(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_AddField_AFM_Kernel <<< (pH_cuVEC->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(pH_cuVEC->get_deviceobject(mGPU), pH2_cuVEC->get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU));
	}
}

//----------------------- LAUNCHERS EXTRAPOLATION METHODS FOR MULTICONVOLUTION (SET METHODS)

//QUINTIC
void EvalSpeedupCUDA::EvalSpeedup_SetExtrapField_AddSelf_MConv(
	mcu_VEC(cuReal3)* ptransfer,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (ptransfer->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(ptransfer->get_deviceobject(mGPU),
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), Hdemag6.get_deviceobject(mGPU),
			a1, a2, a3, a4, a5, a6,
			ptransfer->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUARTIC
void EvalSpeedupCUDA::EvalSpeedup_SetExtrapField_AddSelf_MConv(
	mcu_VEC(cuReal3)* ptransfer,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (ptransfer->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(ptransfer->get_deviceobject(mGPU),
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU),
			a1, a2, a3, a4, a5,
			ptransfer->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//CUBIC
void EvalSpeedupCUDA::EvalSpeedup_SetExtrapField_AddSelf_MConv(
	mcu_VEC(cuReal3)* ptransfer,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (ptransfer->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(ptransfer->get_deviceobject(mGPU),
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU),
			a1, a2, a3, a4,
			ptransfer->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUADRATIC
void EvalSpeedupCUDA::EvalSpeedup_SetExtrapField_AddSelf_MConv(
	mcu_VEC(cuReal3)* ptransfer,
	cuBReal a1, cuBReal a2, cuBReal a3)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (ptransfer->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(ptransfer->get_deviceobject(mGPU),
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU),
			a1, a2, a3,
			ptransfer->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//LINEAR
void EvalSpeedupCUDA::EvalSpeedup_SetExtrapField_AddSelf_MConv(
	mcu_VEC(cuReal3)* ptransfer,
	cuBReal a1, cuBReal a2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (ptransfer->device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(ptransfer->get_deviceobject(mGPU),
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU),
			a1, a2,
			ptransfer->get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

#endif

#endif