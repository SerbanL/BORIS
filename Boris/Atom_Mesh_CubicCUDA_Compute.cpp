#include "stdafx.h"
#include "Atom_Mesh_CubicCUDA.h"

#if COMPILECUDA == 1
#ifdef MESH_COMPILATION_ATOM_CUBIC

#include "Atom_Mesh_Cubic.h"

//return phase transition temperature (K) based on formula Tc = J*e*z/3kB
double Atom_Mesh_CubicCUDA::Show_Transition_Temperature(void)
{
	return paMeshCubic->Show_Transition_Temperature();
}

//return saturation magnetization (A/m) based on formula Ms = mu_s*n/a^3
double Atom_Mesh_CubicCUDA::Show_Ms(void)
{
	return paMeshCubic->Show_Ms();
}

//return exchange stiffness (J/m) based on formula A = J*n/2a
double Atom_Mesh_CubicCUDA::Show_A(void)
{
	return paMeshCubic->Show_A();
}

//return uniaxial anisotropy constant (J/m^3) based on formula K = k*n/a^3
double Atom_Mesh_CubicCUDA::Show_Ku(void)
{
	return paMeshCubic->Show_Ku();
}

#endif
#endif