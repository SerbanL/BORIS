#include "Atom_Mesh_CubicCUDA.h"

#if COMPILECUDA == 1

#ifdef MESH_COMPILATION_ATOM_CUBIC

#include "mcuVEC_oper.cuh"

//called by Track_Shift_Algorithm when copy_values_thermalize call is required, since this needs to be implemented in a cu file
void Atom_Mesh_CubicCUDA::Track_Shift_Algorithm_CopyThermalize(mcu_VEC_VC(cuReal3)& M_src, cuBox cells_box_dst, cuBox cells_box_src)
{
	M1.copy_values_thermalize(M_src, thermalize_FM_to_Atom, cells_box_dst, cells_box_src, prng, false);
}

#endif

#endif