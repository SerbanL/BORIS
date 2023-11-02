#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_DEMAG) || defined(MODULE_COMPILATION_SDEMAG) || defined(MODULE_COMPILATION_ATOM_DIPOLEDIPOLE)

#include "BorisCUDALib.h"
#include "ErrorHandler.h"
#include "DiffEq_Defs.h"

#include "EvalSpeedupDefs.h"

#include "mGPUConfig.h"

template <typename DBL3> class Transfer;

class EvalSpeedupCUDA
{

private:

	// CONFIGURATION

	//controls which of the pointers below are used and how
	EVALSPEEDUP_MODE_ eval_speedup_mode = EVALSPEEDUP_MODE_FM;

	//the pointers below must be set on initialization every time by derived class (Initialize_Data method)
	
	//cuVEC for input magnetization used for atomistic demag, atomistic macrocell dipole-dipole (EVALSPEEDUP_MODE_ATOM) 
	mcu_VEC(cuReal3)* pM_cuVEC = nullptr;

	//cuVEC for output field used for atomistic demag, atomistic macrocell dipole-dipole (EVALSPEEDUP_MODE_ATOM), and also FM meshes and atomistic dipole-dipole (EVALSPEEDUP_MODE_FM)
	mcu_VEC(cuReal3)* pH_cuVEC = nullptr;
	//additional field cuVEC, used for AFM meshes (EVALSPEEDUP_MODE_AFM)
	mcu_VEC(cuReal3)* pH2_cuVEC = nullptr;

	//cuVEC_VC magnetization used for FM meshes and atomistic dipole-dipole (EVALSPEEDUP_MODE_FM)
	mcu_VEC_VC(cuReal3)* pM_cuVEC_VC = nullptr;
	//additional magnetization cuVEC_VC used for AFM meshes (EVALSPEEDUP_MODE_AFM)
	mcu_VEC_VC(cuReal3)* pM2_cuVEC_VC = nullptr;

	// EVAL SPEEDUP DATA

	//vec for demagnetizing field polynomial extrapolation
	mcu_VEC(cuReal3) Hdemag, Hdemag2, Hdemag3, Hdemag4, Hdemag5, Hdemag6;

	//-Nxx, -Nyy, -Nzz values at r = r0
	mcu_val<cuReal3> selfDemagCoeff;

protected:

	//times at which evaluations were done, used for extrapolation
	double time_demag1 = 0.0, time_demag2 = 0.0, time_demag3 = 0.0, time_demag4 = 0.0, time_demag5 = 0.0, time_demag6 = 0.0;

	int num_Hdemag_saved = 0;

private:

	//EVALUATION FINISH

	//subtract self demag contribution from H, using M
	//used with EVALSPEEDUP_MODE_ATOM
	void EvalSpeedup_SubSelf(mcu_VEC(cuReal3)& H);

	//similar to above, but the self contribution to subtract is given in transfer (used for multi-convolution)
	void EvalSpeedup_SubSelf(mcu_VEC(cuReal3)& H, mcu_VEC(cuReal3)& transfer);

	//subtract self demag contribution from H, using M, after adding H to Heff (pH_cuVEC)
	//used with EVALSPEEDUP_MODE_FM
	void EvalSpeedup_AddField_SubSelf_FM(mcu_VEC(cuReal3)& H);
	//used with EVALSPEEDUP_MODE_AFM
	void EvalSpeedup_AddField_SubSelf_AFM(mcu_VEC(cuReal3)& H);

	//EXTRAPOLATION METHODS

	//Set methods : set extrapolation field in *pH_cuVEC, also adding in self demag using *pM_cuVEC (EVALSPEEDUP_MODE_ATOM)
	void EvalSpeedup_SetExtrapField_AddSelf(cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6);
	void EvalSpeedup_SetExtrapField_AddSelf(cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5);
	void EvalSpeedup_SetExtrapField_AddSelf(cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4);
	void EvalSpeedup_SetExtrapField_AddSelf(cuBReal a1, cuBReal a2, cuBReal a3);
	void EvalSpeedup_SetExtrapField_AddSelf(cuBReal a1, cuBReal a2);

	//Add methods : add extrapolation field to *pH_cuVEC, also adding in self demag using *pM_cuVEC_VC (EVALSPEEDUP_MODE_FM)
	void EvalSpeedup_AddExtrapField_AddSelf_FM(cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6);
	void EvalSpeedup_AddExtrapField_AddSelf_FM(cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5);
	void EvalSpeedup_AddExtrapField_AddSelf_FM(cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4);
	void EvalSpeedup_AddExtrapField_AddSelf_FM(cuBReal a1, cuBReal a2, cuBReal a3);
	void EvalSpeedup_AddExtrapField_AddSelf_FM(cuBReal a1, cuBReal a2);
	void EvalSpeedup_AddField_FM(void);

	//Add methods : add extrapolation field to *pH_cuVEC and *pH2_cuVEC, also adding in self demag using *pM_cuVEC_VC and *pM2_cuVEC_VC (EVALSPEEDUP_MODE_AFM)
	void EvalSpeedup_AddExtrapField_AddSelf_AFM(cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6);
	void EvalSpeedup_AddExtrapField_AddSelf_AFM(cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5);
	void EvalSpeedup_AddExtrapField_AddSelf_AFM(cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4);
	void EvalSpeedup_AddExtrapField_AddSelf_AFM(cuBReal a1, cuBReal a2, cuBReal a3);
	void EvalSpeedup_AddExtrapField_AddSelf_AFM(cuBReal a1, cuBReal a2);
	void EvalSpeedup_AddField_AFM(void);

	//FOR MULTICONVOLUTIUON
	//Set methods : set extrapolation field in *ptransfer, also adding in self demag using also *ptransfer
	void EvalSpeedup_SetExtrapField_AddSelf_MConv(mcu_VEC(cuReal3)* ptransfer, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6);
	void EvalSpeedup_SetExtrapField_AddSelf_MConv(mcu_VEC(cuReal3)* ptransfer, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5);
	void EvalSpeedup_SetExtrapField_AddSelf_MConv(mcu_VEC(cuReal3)* ptransfer, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4);
	void EvalSpeedup_SetExtrapField_AddSelf_MConv(mcu_VEC(cuReal3)* ptransfer, cuBReal a1, cuBReal a2, cuBReal a3);
	void EvalSpeedup_SetExtrapField_AddSelf_MConv(mcu_VEC(cuReal3)* ptransfer, cuBReal a1, cuBReal a2);

protected:

	EvalSpeedupCUDA(void) :
		Hdemag(mGPU), Hdemag2(mGPU), Hdemag3(mGPU), Hdemag4(mGPU), Hdemag5(mGPU), Hdemag6(mGPU),
		selfDemagCoeff(mGPU)
	{}
	virtual ~EvalSpeedupCUDA() {}

	//-------------------Called by respective methods from derived demag modules

	//called by Initialize method, specifically for eval speedup data initialization
	BError Initialize_EvalSpeedup(
		//pre-calculated self demag coefficient
		cuReal3 selfDemagCoeff_cpu,
		//what speedup factor (polynomial order) is being used?
		int evaluation_speedup_factor,
		//cell-size and rect for Hdemag cuVECs
		cuReal3 h, cuRect meshRect,
		//if transfer to other H cuVECs is required from Hdemag cuVECs then set it here, with transfer info pre-calculated
		std::vector<mcu_VEC(cuReal3)*> pVal_to_H = {}, std::vector<mcu_VEC(cuReal3)*> pVal_to_H2 = {}, Transfer<DBL3>* ptransfer_info_cpu = nullptr,
		std::vector<mcu_VEC(cuReal3)*> pVal2_to_H = {}, Transfer<DBL3>* ptransfer2_info_cpu = nullptr);

	void Initialize_EvalSpeedup_Mode_Atom(mcu_VEC(cuReal3)& M_cuVEC, mcu_VEC(cuReal3)& H_cuVEC);
	void Initialize_EvalSpeedup_Mode_FM(mcu_VEC_VC(cuReal3)& M_cuVEC_VC, mcu_VEC(cuReal3)& H_cuVEC);
	void Initialize_EvalSpeedup_Mode_AFM(
		mcu_VEC_VC(cuReal3)& M_cuVEC_VC, mcu_VEC_VC(cuReal3)& M2_cuVEC_VC,
		mcu_VEC(cuReal3)& H_cuVEC, mcu_VEC(cuReal3)& H2_cuVEC);

	//called by UpdateConfiguration method, specifically for eval speedup data configuration update
	void UpdateConfiguration_EvalSpeedup(void);

	//-------------------Runtime

	//check if speedup should be done (true) or not (false)
	//if true, then caller should then run the method below (UpdateField_EvalSpeedup) instead of its no speedup computation
	bool Check_if_EvalSpeedup(int eval_speedup_factor, bool check_step_update);

	//implements eval speedup scheme
	void UpdateField_EvalSpeedup(
		int eval_speedup_factor, bool check_step_update,
		double eval_step_time,
		std::function<void(mcu_VEC(cuReal3)&)>& do_evaluation,
		std::function<void(void)>& do_transfer_in, std::function<void(mcu_VEC(cuReal3)&)>& do_transfer_out);

	//The methods below break down the field update in multiple steps, used for multi-convolution
	mcu_VEC(cuReal3)* UpdateField_EvalSpeedup_MConv_Start(int eval_speedup_factor, bool check_step_update, double eval_step_time);
	void UpdateField_EvalSpeedup_MConv_Finish(int eval_speedup_factor, bool do_transfer, mcu_VEC(cuReal3)* pHdemag, mcu_VEC(cuReal3)& transfer);
	void UpdateField_EvalSpeedup_MConv_Extrap(
		int eval_speedup_factor, double eval_step_time,
		mcu_VEC(cuReal3)* ptransfer);
};

#endif

#endif