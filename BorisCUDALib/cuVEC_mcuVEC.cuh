#pragma once

#include "mcuVEC_Managed.h"

template <typename VType>
__device__ mcuVEC_Managed<cuVEC<VType>, VType>& cuVEC<VType>::mcuvec(void)
{
	return *pmcuVEC;
}