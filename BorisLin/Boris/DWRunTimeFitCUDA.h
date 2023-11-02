#pragma once

#include "Boris_Enums_Defs.h"

#if COMPILECUDA == 1

#include "BorisCUDALib.h"

#include "DWRunTimeFitDefs.h"

#include "mGPUConfig.h"

//Class used to analyse domain wall and extract position and width - efficient and robust for runtime, especially atomistic simulations
class DWPosWidthCUDA
{

private:

	//GPU data on base device only

	//auxiliary data
	cu_obj<cuBReal> As, Ae, x0, dw, weight;
	cu_obj<size_t> av_points_x0;

	//auxiliary vector containing smoothed xy data
	cu_arr<cuReal2> xy_data_smoothed;
	size_t xy_data_smoothed_size = 0;

	//profile data on base device
	cu_arr<cuReal2> xy_data_base;

	//profile data as mcu_arr : used to store extracted profile from mesh
	//if multi-GPU then transfer to xy_data_base before processing
	mcu_arr<cuReal2> xy_data;

	//object used to transfer xy_data from multiple device to base xy_data_base cu_arr
	//
	mGPU_Transfer<cuReal2> xy_data_transf;

private:

	void setup_xy_data_transf(void)
	{
		//xy_data_transf needed only if using multiple GPUs
		if (mGPU.get_num_devices() > 1) {

			xy_data_transf.set_transfer_size(xy_data.size());

			//refresh xy_data entries in xy_data_transf
			for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

				xy_data_transf.setup_device_memory_handle(idx, xy_data.get_array(idx));
			}

			//refresh extra entry for device 0 (xy_data_base)
			xy_data_transf.setup_extra_device_memory_handle(0, 1, xy_data_base.get_array());
		}
	}

public:

	DWPosWidthCUDA(void) :
		xy_data(mGPU), xy_data_transf(mGPU)
	{
		//xy_data_transf : entry 0 for each device set to xy_data
		for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

			xy_data_transf.setup_device_memory_handle(idx, xy_data.get_array(idx));
		}

		//extra entry for device 0 set to xy_data_base
		xy_data_transf.add_extra_device_memory_handle(0, xy_data_base.get_array());
	}

	~DWPosWidthCUDA() {}

	//----------------------------------- PUBLIC METHODS

	//Fit the extracted profile (xy_data) for position and width, assuming the magnetization component follows f(x) = [ (As - Ae) * tanh(-PI * (x - x0) / dw) + (As + Ae) ] / 2
	//Here As, Ae are the start and end values - profile must be long enough to include at least DWPOS_ENDSTENCIL (length ratio) flat parts of the tanh profile
	//x0 is the centre position relative to start of profile
	//dw is the domain wall width
	//xy_data contains profile as x coordinates and corresponding y values
	//return x0, dw
	cuReal2 FitDomainWallCUDA(double length);

	//prepare xy_data profile storage with correct dimensions and return through reference so it can be filled in
	mcu_arr<cuReal2>& get_xy_data_ref(void) { return xy_data; }
};

#endif

