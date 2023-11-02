#include "MElasticCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MELASTIC

//------------------- Configuration

//set diagonal and shear strain text equations
BError MElasticCUDA::Set_Sd_Equation(std::vector<std::vector< std::vector<EqComp::FSPEC> >> fspec)
{
	BError error(CLASS_STR(MElasticCUDA));

	if (!Sd_equation.make_vector(fspec)) return error(BERROR_OUTOFGPUMEMORY_CRIT);

	return error;
}

BError MElasticCUDA::Set_Sod_Equation(std::vector<std::vector< std::vector<EqComp::FSPEC> >> fspec)
{
	BError error(CLASS_STR(MElasticCUDA));

	if (!Sod_equation.make_vector(fspec)) return error(BERROR_OUTOFGPUMEMORY_CRIT);

	return error;
}

//make Fext_equationCUDA[external_stress_surfaces_index], where external_stress_surfaces_index is an index in external_stress_surfaces
BError MElasticCUDA::Set_Fext_equation(int external_stress_surfaces_index, std::vector<std::vector< std::vector<EqComp::FSPEC> >> fspec)
{
	BError error(CLASS_STR(MElasticCUDA));

	if (Fext_equationCUDA[external_stress_surfaces_index]) delete Fext_equationCUDA[external_stress_surfaces_index];
	Fext_equationCUDA[external_stress_surfaces_index] = nullptr;
	Fext_equationCUDA[external_stress_surfaces_index] = new mTEquationCUDA<cuBReal, cuBReal, cuBReal>(mGPU);
	if (!Fext_equationCUDA[external_stress_surfaces_index]->make_vector(fspec)) return error(BERROR_OUTOFGPUMEMORY_CRIT);

	return error;
}

#endif

#endif