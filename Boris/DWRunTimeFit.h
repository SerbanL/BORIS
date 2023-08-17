#pragma once

#include "Boris_Enums_Defs.h"

#include "BorisLib.h"

#if COMPILECUDA == 1
#include "BorisCUDALib.h"
#include "DWRunTimeFitCUDA.h"
#endif

#include "DWRunTimeFitDefs.h"

//Class used to analyse domain wall and extract position and width - efficient and robust for runtime, especially atomistic simulations
class DWPosWidth
{

private:

	//auxiliary vector containing smoothed xy data
	std::vector<DBL2> xy_data_smoothed;

#if COMPILECUDA == 1
	//contains CUDA version of this class
	DWPosWidthCUDA* pcu_dwPos = nullptr;
#endif

public:

	DWPosWidth(void) {}
	~DWPosWidth() {}

	//----------------------------------- PUBLIC METHODS

	//Fit the extracted profile for position and width, assuming the magnetization component follows f(x) = [ (As - Ae) * tanh(-PI * (x - x0) / dw) + (As + Ae) ] / 2
	//Here As, Ae are the start and end values - profile must be long enough to include at least DWPOS_ENDSTENCIL (length ratio) flat parts of the tanh profile
	//x0 is the centre position relative to start of profile
	//dw is the domain wall width
	//xy_data contains profile as x coordinates and corresponding y values
	//return x0, dw
	DBL2 FitDomainWall(std::vector<DBL2>& xy_data);

#if COMPILECUDA == 1
	DBL2 FitDomainWallCUDA(double length) { return pcu_dwPos->FitDomainWallCUDA(length); }
	mcu_arr<cuReal2>& getcuda_xy_data_ref(void) { return pcu_dwPos->get_xy_data_ref(); }

	//construct/destruct CUDA objects here as needed
	void SwitchCUDAState(bool cudaState)
	{
		if (cudaState) {

			//switching CUDA on
			pcu_dwPos = new DWPosWidthCUDA();
		}
		else {

			//switching CUDA off
			if (pcu_dwPos) delete pcu_dwPos;
			pcu_dwPos = nullptr;
		}
	}
#endif
};
