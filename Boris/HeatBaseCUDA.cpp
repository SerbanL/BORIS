#include "stdafx.h"
#include "HeatBaseCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_HEAT

#include "mGPUConfig.h"

HeatBaseCUDA::HeatBaseCUDA(void) :
	heatEq_RHS(mGPU),
	Q_equation(mGPU),
	temp_cmbnd_funcs_pri(mGPU),
	temp_cmbnd_funcs_sec(mGPU)
{}


#endif

#endif