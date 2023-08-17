#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_DEMAG

#include "ModulesCUDA.h"

class MeshCUDA;
class DemagMCUDA_single;

class DemagMCUDA :
	public ModulesCUDA
{
	friend DemagMCUDA_single;

private:

	////////////////////////////////////////////////////

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	MeshCUDA* pMeshCUDA;

	////////////////////////////////////////////////////

	//one DemagMCUDA_single object per GPU
	std::vector<DemagMCUDA_single*> pDemagMCUDA;

	//transfer data before x-FFTs
	std::vector<std::vector<mGPU_Transfer<cuReal3>*>> M_Input_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> M_Input_transfer_half;

	std::vector<std::vector<mGPU_Transfer<cuBComplex>*>> xFFT_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> xFFT_Data_transfer_half;

	//transfer data before x-IFFTs
	std::vector<std::vector<mGPU_Transfer<cuBComplex>*>> xIFFT_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> xIFFT_Data_transfer_half;
	
	//transfer data after x-IFFTs
	std::vector<std::vector<mGPU_Transfer<cuReal3>*>> Out_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> Out_Data_transfer_half;

	////////////////////////////////////////////////////
	//Evaluation speedup mode data

	//vec for demagnetizing field polynomial extrapolation
	mcu_VEC(cuReal3) Hdemag, Hdemag2, Hdemag3, Hdemag4, Hdemag5, Hdemag6;

	//times at which evaluations were done, used for extrapolation
	double time_demag1 = 0.0, time_demag2 = 0.0, time_demag3 = 0.0, time_demag4 = 0.0, time_demag5 = 0.0, time_demag6 = 0.0;

	int num_Hdemag_saved = 0;

	//-Nxx, -Nyy, -Nzz values at r = r0
	mcu_val<cuReal3> selfDemagCoeff;

private:

	//check if all pDemagMCUDA modules are initialized
	bool Submodules_Initialized(void);

	void set_DemagCUDA_pointers(void);

	//Add newly computed field to Heff and Heff2, then subtract self demag contribution from it : AFM
	void Demag_EvalSpeedup_AddField_SubSelf(
		mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
		mcu_VEC(cuReal3)& HField,
		mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2);

	//Add newly computed field to Heff and Heff2, then subtract self demag contribution from it : FM
	void Demag_EvalSpeedup_AddField_SubSelf(
		mcu_VEC(cuReal3)& Heff,
		mcu_VEC(cuReal3)& HField,
		mcu_VEC_VC(cuReal3)& M);

	//Add extrapolated field together with self demag contribution : AFM, QUINTIC
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
		cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
		mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2);

	//Add extrapolated field together with self demag contribution : FM, QUINTIC
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff,
		cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
		mcu_VEC_VC(cuReal3)& M);

	//Add extrapolated field together with self demag contribution : AFM, QUARTIC
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
		cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
		mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2);

	//Add extrapolated field together with self demag contribution : FM, QUARTIC
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff,
		cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
		mcu_VEC_VC(cuReal3)& M);

	//Add extrapolated field together with self demag contribution : AFM, CUBIC
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
		cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
		mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2);

	//Add extrapolated field together with self demag contribution : FM, CUBIC
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff,
		cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
		mcu_VEC_VC(cuReal3)& M);

	//Add extrapolated field together with self demag contribution : AFM, QUADRATIC
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
		cuBReal a1, cuBReal a2, cuBReal a3,
		mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2);

	//Add extrapolated field together with self demag contribution : FM, QUADRATIC
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff,
		cuBReal a1, cuBReal a2, cuBReal a3,
		mcu_VEC_VC(cuReal3)& M);

	//Add extrapolated field together with self demag contribution : AFM, LINEAR
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
		cuBReal a1, cuBReal a2,
		mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2);

	//Add extrapolated field together with self demag contribution : FM, LINEAR
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff,
		cuBReal a1, cuBReal a2,
		mcu_VEC_VC(cuReal3)& M);

	//Add extrapolated field together with self demag contribution : AFM, STEP
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff, mcu_VEC(cuReal3)& Heff2,
		mcu_VEC_VC(cuReal3)& M, mcu_VEC_VC(cuReal3)& M2);

	//Add extrapolated field together with self demag contribution : FM, STEP
	void Demag_EvalSpeedup_AddExtrapField_AddSelf(
		mcu_VEC(cuReal3)& Heff,
		mcu_VEC_VC(cuReal3)& M);

public:

	DemagMCUDA(MeshCUDA* pMeshCUDA_);
	~DemagMCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);

	//-------------------Configuration

};

#else

class DemagMCUDA
{
};

#endif

#endif


