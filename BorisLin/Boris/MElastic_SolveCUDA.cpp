#include "stdafx.h"
#include "MElasticCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MELASTIC

#include "MElastic.h"

//-----MElastic Field

void MElasticCUDA::Calculate_MElastic_Field(void)
{
	switch (pMElastic->crystal) {

	case CRYSTAL_CUBIC:
		Calculate_MElastic_Field_Cubic();
		break;

	case CRYSTAL_TRIGONAL:
		Calculate_MElastic_Field_Trigonal();
		break;

	default:
		Calculate_MElastic_Field_Cubic();
		break;
	}
}

//-----Velocity

void MElasticCUDA::Iterate_Elastic_Solver_Velocity(double dT)
{
	switch (pMElastic->crystal) {

	case CRYSTAL_CUBIC:
		Iterate_Elastic_Solver_Velocity1(dT);
		break;

	case CRYSTAL_TRIGONAL:
		Iterate_Elastic_Solver_Velocity1(dT);
		Iterate_Elastic_Solver_Velocity2(dT);
		break;

	default:
		Iterate_Elastic_Solver_Velocity1(dT);
		break;
	}
}

//-----Stress

void MElasticCUDA::Iterate_Elastic_Solver_Stress(double dT, double magnetic_dT)
{
	switch (pMElastic->crystal) {

	case CRYSTAL_CUBIC:
		Iterate_Elastic_Solver_Stress_Cubic(dT, magnetic_dT);
		break;

	case CRYSTAL_TRIGONAL:
		Iterate_Elastic_Solver_Stress_Trigonal(dT, magnetic_dT);
		break;

	default:
		Iterate_Elastic_Solver_Stress_Cubic(dT, magnetic_dT);
		break;
	}
}

//-----Initial Stress

//if thermoelasticity or magnetostriction is enabled, then initial stress must be set correctly
void MElasticCUDA::Set_Initial_Stress(void)
{
	switch (pMElastic->crystal) {

	case CRYSTAL_CUBIC:
		Set_Initial_Stress_Cubic();
		break;

	case CRYSTAL_TRIGONAL:
		Set_Initial_Stress_Trigonal();
		break;

	default:
		Set_Initial_Stress_Cubic();
		break;
	}
}

#endif

#endif