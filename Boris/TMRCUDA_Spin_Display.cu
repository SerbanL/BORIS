#include "TMRCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TMR

#include "BorisCUDALib.cuh"

#include "MeshCUDA.h"
#include "SuperMeshCUDA.h"
#include "MeshParamsControlCUDA.h"

//-------------------Display Calculation Methods

//Launchers

//return x, y, or z component of spin current (component = 0, 1, or 2)
mcu_VEC(cuReal3)& TMRCUDA::GetSpinCurrent(int component)
{
	return displayVEC;
}

//return spin torque computed from spin accumulation
mcu_VEC(cuReal3)& TMRCUDA::GetSpinTorque(void)
{
	return displayVEC;
}

//Calculate the interface spin accumulation torque for a given contact (in magnetic meshes for NF interfaces with G interface conductance set), accumulating result in displayVEC
void TMRCUDA::CalculateDisplaySAInterfaceTorque(TransportBaseCUDA* ptrans_sec, mCMBNDInfoCUDA& contactCUDA, bool primary_top)
{
}

#endif

#endif