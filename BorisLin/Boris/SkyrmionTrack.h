#pragma once

#include "Boris_Enums_Defs.h"

#include "BorisLib.h"
#include "ErrorHandler.h"

#if COMPILECUDA == 1
#include "BorisCUDALib.h"
#include "mGPUConfig.h"
#endif

#include "SimulationData.h"

//Class used to track one or more skyrmions in a given ferromagnetic mesh
class SkyrmionTrack :
	public ProgramState<SkyrmionTrack,
	std::tuple<double, std::vector<DBL2>, std::vector<DBL2>, std::vector<DBL2>, std::vector<Rect>>,
	std::tuple<> >
{

private:

	//set skypos tracking rectangle to skyrmion diameter times dia_mul
	double dia_mul = 2.0;

	//identify skyrmion tracker by bottom-left Rect coordinates
	std::vector<DBL2> skyTrack_Id;

	//the skyrmion total shift - same vector size as skyTrack_Id
	std::vector<DBL2> skyTrack_Shift;

	//the skyrmion total shift from last pass; use it to eliminate shift oscillations - same vector size as skyTrack_Id
	std::vector<DBL2> skyTrack_ShiftLast;

	//when using the skyrmion fitting method we record the updated skyrmion rectangle here
	std::vector<Rect> skyTrack_rect;

	//xy data used for fitting skyrmions - resize as needed and keep size.
	std::vector<DBL2> xy_data;

#if COMPILECUDA == 1
	//in CUDA mode, before fitting extract data from M into y_data_gpu using a kernel call, then transfer it to cpu memory in y_data_cpu; after this you can run the LMA fitting algorithm on the cpu.
	//(not great due to gpu to cpu memory transfer, but the alternative is to write the fitting algorithm in CUDA code which isn't great either : 
	//the xy data size is pretty small so gpu computation here is not ideal, plus it's not straightforward to implement the LMA algorithm in CUDA; the memory transfer is small so current method should be fine)
	mcu_arr<cuBReal>* pdata_gpu = nullptr;
	std::vector<cuBReal> data_cpu;
#endif

private:

	//for given skyrmion identifying rectangle obtain an index in skyTrack_Shift - either an existing entry if skyRect origin found in skyTrack_Id, else make a new entry
	int Get_skyTrack_index(Rect skyRect, Rect maxRectAbsolute);

public:

	SkyrmionTrack(void) :
		ProgramStateNames(this,
			{ VINFO(dia_mul), VINFO(skyTrack_Id), VINFO(skyTrack_Shift), VINFO(skyTrack_ShiftLast), VINFO(skyTrack_rect) }, {})
	{}

	~SkyrmionTrack() {}

	//-----------------------------------

	//implement pure virtual method from ProgramState
	void RepairObjectState(void) 
	{
		if (skyTrack_Shift.size() != skyTrack_Id.size()) skyTrack_Shift.resize(skyTrack_Id.size());
		if (skyTrack_ShiftLast.size() != skyTrack_Id.size()) skyTrack_ShiftLast.resize(skyTrack_Id.size());
	}

	//----------------------------------- PUBLIC METHODS

	//Calculate skyrmion shift for a skyrmion in M initially in the given rectangle (relative coordinates)
	//The bottom-left coordinates of the rectangle are used to uniquely identify the skyrmion (new entry made if this is first call for this Rect), and a shift is calculated
	//so that the shifted rectangle is centered over the skyrmion - the total shift for this entry is returned. This method should be called often enough so tracking is not lost.
	//This method only creates entries, does not clean any.
	DBL2 Get_skyshift(VEC_VC<DBL3>& M, Rect skyRect);

	//additionally return the x and y diameters
	DBL4 Get_skypos_diameters(VEC_VC<DBL3>& M, Rect skyRect);

	double Get_skypos_dmul(void) { return dia_mul; }
	void Set_skypos_dmul(double dia_mul_) { dia_mul = dia_mul_; }

#if COMPILECUDA == 1
	//as for the non-CUDA version but also pass in M.n.dim() and M.h so we don't have to get them from gpu memory
	DBL2 Get_skyshiftCUDA(mcu_VEC_VC(cuReal3)& M, Rect skyRect);

	DBL4 Get_skypos_diametersCUDA(mcu_VEC_VC(cuReal3)& M, Rect skyRect);
#endif

	//Clean any skyrmion tracker entries by comparing the entries held here against external lists (Simulation::dataBoxList and Simulation::saveDataList) - if not found in external lists then delete them here
	//This method is called by FMesh::UpdateConfiguration
	void UpdateConfiguration(vector_lut<DatumConfig>& saveDataList);

#if COMPILECUDA == 1
	//construct/destruct CUDA objects here as needed
	void SwitchCUDAState(bool cudaState);
#endif
};