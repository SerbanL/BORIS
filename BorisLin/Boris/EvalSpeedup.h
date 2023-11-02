#pragma once

#include "Boris_Enums_Defs.h"
#if defined(MODULE_COMPILATION_DEMAG) || defined(MODULE_COMPILATION_SDEMAG) || defined(MODULE_COMPILATION_ATOM_DIPOLEDIPOLE)

#include "BorisLib.h"
#include "ErrorHandler.h"
#include "DiffEq_Defs.h"

#include "EvalSpeedupDefs.h"

template <typename DBL3> class Transfer;

class EvalSpeedup
{

private:

	// CONFIGURATION

	//controls which of the pointers below are used and how
	EVALSPEEDUP_MODE_ eval_speedup_mode = EVALSPEEDUP_MODE_FM;

	//the pointers below must be set on initialization every time by derived class (Initialize_Data method)

	//VEC for input magnetization used for atomistic demag, atomistic macrocell dipole-dipole (EVALSPEEDUP_MODE_ATOM) 
	VEC<DBL3>* pM_VEC = nullptr;

	//VEC for output field used for atomistic demag, atomistic macrocell dipole-dipole (EVALSPEEDUP_MODE_ATOM), and also FM meshes and atomistic dipole-dipole (EVALSPEEDUP_MODE_FM)
	VEC<DBL3>* pH_VEC = nullptr;
	//additional field VEC, used for AFM meshes (EVALSPEEDUP_MODE_AFM)
	VEC<DBL3>* pH2_VEC = nullptr;

	//VEC_VC magnetization used for FM meshes and atomistic dipole-dipole (EVALSPEEDUP_MODE_FM)
	VEC_VC<DBL3>* pM_VEC_VC = nullptr;
	//additional magnetization VEC_VC used for AFM meshes (EVALSPEEDUP_MODE_AFM)
	VEC_VC<DBL3>* pM2_VEC_VC = nullptr;

	// EVAL SPEEDUP DATA

	//vec for demagnetizing field polynomial extrapolation
	VEC<DBL3> Hdemag, Hdemag2, Hdemag3, Hdemag4, Hdemag5, Hdemag6;

	//-Nxx, -Nyy, -Nzz values at r = r0
	DBL3 selfDemagCoeff = DBL3();

protected:

	//times at which evaluations were done, used for extrapolation
	double time_demag1 = 0.0, time_demag2 = 0.0, time_demag3 = 0.0, time_demag4 = 0.0, time_demag5 = 0.0, time_demag6 = 0.0;

	int num_Hdemag_saved = 0;

private:

protected:

	EvalSpeedup(void) {}
	virtual ~EvalSpeedup() {}

	//-------------------Called by respective methods from derived demag modules
	
	//called by Initialize method, specifically for eval speedup data initialization
	BError Initialize_EvalSpeedup(
		//pre-calculated self demag coefficient
		DBL3 selfDemagCoeff_,
		//what speedup factor (polynomial order) is being used?
		int evaluation_speedup_factor,
		//cell-size and rect for Hdemag cuVECs
		DBL3 h, Rect meshRect,
		std::function<bool(VEC<DBL3>&)>& initialize_mesh_transfer);
	
	void Initialize_EvalSpeedup_Mode_Atom(VEC<DBL3>& M_VEC, VEC<DBL3>& H_VEC);
	void Initialize_EvalSpeedup_Mode_FM(VEC_VC<DBL3>& M_VEC_VC, VEC<DBL3>& H_VEC);
	void Initialize_EvalSpeedup_Mode_AFM(
		VEC_VC<DBL3>& M_VEC_VC, VEC_VC<DBL3>& M2_VEC_VC,
		VEC<DBL3>& H_VEC, VEC<DBL3>& H2_VEC);

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
		std::function<void(VEC<DBL3>&)>& do_evaluation,
		std::function<void(void)>& do_transfer_in, std::function<void(VEC<DBL3>&)>& do_transfer_out);

public:

	//-------------------Getters

	Transfer<DBL3>& get_transfer(void) { return Hdemag.get_transfer(); }

	DBL3& get_selfDemagCoeff(void) { return selfDemagCoeff; }
};

#endif
