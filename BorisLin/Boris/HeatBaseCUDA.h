#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_HEAT

#include "BorisCUDALib.h"
#include "HeatCUDA_PolicyCMBND.h"

#include "Heat_Defs.h"

//class HeatBase;
class SHeatCUDA;

class HeatBaseCUDA {

	//friend HeatBase;
	friend SHeatCUDA;

protected:

	//evaluate heat equation and store result here. After this is done advance time for temperature based on values stored here.
	mcu_VEC(cuBReal) heatEq_RHS;

	//HeatCUDA_CMBND holds methods used in setting cmbnd conditions.
	//Pass the managed HeatCUDA_CMBND object to set_cmbnd_continuous in Temp
	mcu_obj<HeatCUDA_CMBND_Pri, HeatCUDA_PolicyCMBND<HeatCUDA_CMBND_Pri>> temp_cmbnd_funcs_pri;
	mcu_obj<HeatCUDA_CMBND_Sec, HeatCUDA_PolicyCMBND<HeatCUDA_CMBND_Sec>> temp_cmbnd_funcs_sec;

	//Set Q using user equation, thus allowing simultaneous spatial (x, y, z), stage time (t); stage step (Ss) introduced as user constant.
	//A number of constants are always present : mesh dimensions in m (Lx, Ly, Lz)
	mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal> Q_equation;

private:

	//-------------------Calculation Methods (pure virtual)

	//1-temperature model
	virtual void IterateHeatEquation_1TM(cuBReal dT) = 0;

	//2-temperature model
	virtual void IterateHeatEquation_2TM(cuBReal dT) = 0;

	//-------------------Setters

	//transfer values from globalTemp to Temp in this mesh
	virtual void SetFromGlobalTemperature(mcu_VEC_VC(cuBReal)& globalTemp) = 0;

public:

	HeatBaseCUDA(void);

	virtual ~HeatBaseCUDA() {}

	//-------------------Setters

	//set Temp uniformly to base temperature
	virtual void SetBaseTemperature(cuBReal Temperature) = 0;

	//set Temp non-uniformly as specified through the cT mesh parameter
	virtual void SetBaseTemperature_Nonuniform(cuBReal Temperature) = 0;

	//Set Q_equation text equation object
	BError SetQEquation(const std::vector< std::vector<EqComp::FSPEC> >& fspec);
	void ClearQEquation(void);
};

#else

class HeatCUDA
{
};

#endif

#endif
