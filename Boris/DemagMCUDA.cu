#include "DemagMCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_DEMAG

#include "BorisCUDALib.cuh"
#include "MeshCUDA.h"

//----------------------- Initialization

__global__ void set_DemagCUDA_pointers_kernel(
	ManagedMeshCUDA& cuMesh, cuVEC<cuReal3>& Module_Heff)
{
	if (threadIdx.x == 0) cuMesh.pDemag_Heff = &Module_Heff;
}

void DemagMCUDA::set_DemagCUDA_pointers(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		set_DemagCUDA_pointers_kernel <<< 1, CUDATHREADS >>>
			(pMeshCUDA->cuMesh.get_deviceobject(mGPU), Module_Heff.get_deviceobject(mGPU));
	}
}

//----------------------- LAUNCHERS

//SUBTRACT SELF DEMAG

//Add newly computed field to Heff and Heff2, then subtract self demag contribution from it : AFM
__global__ void Demag_EvalSpeedup_AddField_SubSelf_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& HField,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < HField.linear_size()) {

		Heff[idx] += HField[idx];
		Heff2[idx] += HField[idx];

		HField[idx] -= (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
	}
}

//Add newly computed field to Heff and Heff2, then subtract self demag contribution from it : FM
__global__ void Demag_EvalSpeedup_AddField_SubSelf_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& HField,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < HField.linear_size()) {

		Heff[idx] += HField[idx];

		HField[idx] -= (selfDemagCoeff & M[idx]);
	}
}

//QUINTIC

//Add extrapolated field together with self demag contribution : AFM, QUINTIC
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5, cuVEC<cuReal3>& Hdemag6,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + Hdemag6[idx] * a6 + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//Add extrapolated field together with self demag contribution : FM, QUINTIC
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5, cuVEC<cuReal3>& Hdemag6,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		Heff[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + Hdemag6[idx] * a6 + (selfDemagCoeff & M[idx]);
	}
}

//QUARTIC

//Add extrapolated field together with self demag contribution : AFM, QUARTIC
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//Add extrapolated field together with self demag contribution : FM, QUARTIC
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		Heff[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + (selfDemagCoeff & M[idx]);
	}
}

//CUBIC

//Add extrapolated field together with self demag contribution : AFM, CUBIC
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//Add extrapolated field together with self demag contribution : FM, CUBIC
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		Heff[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + (selfDemagCoeff & M[idx]);
	}
}

//QUADRATIC

//Add extrapolated field together with self demag contribution : AFM, QUADRATIC
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3,
	cuBReal a1, cuBReal a2, cuBReal a3,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//Add extrapolated field together with self demag contribution : FM, QUADRATIC
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3,
	cuBReal a1, cuBReal a2, cuBReal a3,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		Heff[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + (selfDemagCoeff & M[idx]);
	}
}

//LINEAR

//Add extrapolated field together with self demag contribution : AFM, LINEAR
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2,
	cuBReal a1, cuBReal a2,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//Add extrapolated field together with self demag contribution : FM, LINEAR
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2,
	cuBReal a1, cuBReal a2,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		Heff[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + (selfDemagCoeff & M[idx]);
	}
}

//STEP

//Add extrapolated field together with self demag contribution : AFM, STEP
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff, cuVEC<cuReal3>& Heff2,
	cuVEC<cuReal3>& Hdemag,
	cuVEC_VC<cuReal3>& M, cuVEC_VC<cuReal3>& M2, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		cuReal3 Hdemag_value = Hdemag[idx] + (selfDemagCoeff & (M[idx] + M2[idx]) / 2);
		Heff[idx] += Hdemag_value;
		Heff2[idx] += Hdemag_value;
	}
}

//Add extrapolated field together with self demag contribution : FM, STEP
__global__ void Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& Heff,
	cuVEC<cuReal3>& Hdemag,
	cuVEC_VC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Hdemag.linear_size()) {

		Heff[idx] += Hdemag[idx] + (selfDemagCoeff & M[idx]);
	}
}

//----------------------- LAUNCHERS

//Add newly computed field to Heff and Heff2, then subtract self demag contribution from it : AFM
void DemagMCUDA::Demag_EvalSpeedup_AddField_SubSelf(
	mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
	mcu_VEC(cuReal3)& HField,
	mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddField_SubSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU), Heff2.get_deviceobject(mGPU),
				HField.get_deviceobject(mGPU),
				M.get_deviceobject(mGPU), M2.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//Add newly computed field to Heff and Heff2, then subtract self demag contribution from it : FM
void DemagMCUDA::Demag_EvalSpeedup_AddField_SubSelf(
	mcu_VEC(cuReal3)& Heff,
	mcu_VEC(cuReal3)& HField,
	mcu_VEC_VC(cuReal3)& M)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddField_SubSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU),
				HField.get_deviceobject(mGPU),
				M.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//QUINTIC

//Add extrapolated field together with self demag contribution : AFM, QUINTIC
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
	mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU), Heff2.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU),
				Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), Hdemag6.get_deviceobject(mGPU),
				a1, a2, a3, a4, a5, a6,
				M.get_deviceobject(mGPU), M2.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//Add extrapolated field together with self demag contribution : FM, QUINTIC
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
	mcu_VEC_VC(cuReal3)& M)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), Hdemag6.get_deviceobject(mGPU),
				a1, a2, a3, a4, a5, a6,
				M.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//QUARTIC

//Add extrapolated field together with self demag contribution : AFM, QUARTIC
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
	mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU), Heff2.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU),
				a1, a2, a3, a4, a5,
				M.get_deviceobject(mGPU), M2.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//Add extrapolated field together with self demag contribution : FM, QUARTIC
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
	mcu_VEC_VC(cuReal3)& M)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU),
				a1, a2, a3, a4, a5,
				M.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//CUBIC

//Add extrapolated field together with self demag contribution : AFM, CUBIC
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
	mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU), Heff2.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU),
				a1, a2, a3, a4,
				M.get_deviceobject(mGPU), M2.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//Add extrapolated field together with self demag contribution : FM, CUBIC
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
	mcu_VEC_VC(cuReal3)& M)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU),
				a1, a2, a3, a4,
				M.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//QUADRATIC

//Add extrapolated field together with self demag contribution : AFM, QUADRATIC
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
	cuBReal a1, cuBReal a2, cuBReal a3,
	mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU), Heff2.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU),
				a1, a2, a3,
				M.get_deviceobject(mGPU), M2.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//Add extrapolated field together with self demag contribution : FM, QUADRATIC
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff,
	cuBReal a1, cuBReal a2, cuBReal a3,
	mcu_VEC_VC(cuReal3)& M)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU),
				a1, a2, a3,
				M.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//LINEAR

//Add extrapolated field together with self demag contribution : AFM, LINEAR
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
	cuBReal a1, cuBReal a2,
	mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU), Heff2.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU),
				a1, a2,
				M.get_deviceobject(mGPU), M2.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//Add extrapolated field together with self demag contribution : FM, LINEAR
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff,
	cuBReal a1, cuBReal a2,
	mcu_VEC_VC(cuReal3)& M)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU),
				a1, a2,
				M.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//STEP

//Add extrapolated field together with self demag contribution : AFM, STEP
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
	mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU), Heff2.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU),
				M.get_deviceobject(mGPU), M2.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

//Add extrapolated field together with self demag contribution : FM, STEP
void DemagMCUDA::Demag_EvalSpeedup_AddExtrapField_AddSelf(
	mcu_VEC(cuReal3)& Heff,
	mcu_VEC_VC(cuReal3)& M)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Demag_EvalSpeedup_AddExtrapField_AddSelf_Kernel << < (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >> >
			(Heff.get_deviceobject(mGPU),
				Hdemag.get_deviceobject(mGPU),
				M.get_deviceobject(mGPU),
				selfDemagCoeff(mGPU));
	}
}

#endif

#endif