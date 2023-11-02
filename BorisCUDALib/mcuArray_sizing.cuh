#pragma once

#include "mcuArray.h"

#include "cuArray_sizing.cuh"

//------------------------------------------- STORE ENTRIES : mcuArray_sizing.cuh

template <typename VType>
void mcu_arr<VType>::push_back_deepcopy(int mGPU_idx, VType& new_entry)
{
	pcuarr[mGPU_idx]->push_back_deepcopy(new_entry);
}

template <typename VType>
void mcu_arr<VType>::set_deepcopy(int mGPU_idx, int index, VType& new_entry)
{
	pcuarr[mGPU_idx]->set_deepcopy(index, new_entry);
}