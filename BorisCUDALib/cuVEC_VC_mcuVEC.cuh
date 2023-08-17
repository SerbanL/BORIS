#pragma once

#include "mcuVEC_Managed.h"

template <typename VType>
__device__ mcuVEC_Managed<cuVEC_VC<VType>, VType>& cuVEC_VC<VType>::mcuvec(void)
{
	return *pmcuVEC;
}
