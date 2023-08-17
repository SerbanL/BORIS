#pragma once

#include "mcuVEC_Managed.h"

//--------------------------------------------MEMORY MANAGEMENT HELPER METHODS

//called by mcuVEC when mcuVEC_Managed objects are constructed, so pointer can be stored here too (cuVEC_VC_mcuVEC.h)
template <typename VType>
__host__ void cuVEC_VC<VType>::set_pmcuVEC(mcuVEC_Managed<cuVEC_VC<VType>, VType>*& pmcuVEC_)
{
	set_gpu_value(pmcuVEC, pmcuVEC_);
}