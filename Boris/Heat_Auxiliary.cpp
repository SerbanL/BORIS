#include "stdafx.h"
#include "Heat.h"

#ifdef MODULE_COMPILATION_HEAT

#include "Mesh.h"
#include "MeshParamsControl.h"

//set Temp uniformly to base temperature, unless a spatial variation is also specified through the cT mesh parameter
void Heat::SetBaseTemperature(double Temperature, bool force_set_uniform)
{
#if COMPILECUDA == 1
	if (pModuleCUDA) {

		if (!pMesh->cT.is_sdep()) {

			if (force_set_uniform) dynamic_cast<HeatCUDA*>(pModuleCUDA)->SetBaseTemperature(Temperature);
		}
		else {

			dynamic_cast<HeatCUDA*>(pModuleCUDA)->SetBaseTemperature_Nonuniform(Temperature);
		}

		return;
	}
#endif

	if (!pMesh->cT.is_sdep()) {

		if (force_set_uniform) {

			//uniform temperature
			pMesh->Temp.setnonempty(Temperature);

			if (pMesh->Temp_l.linear_size()) pMesh->Temp_l.setnonempty(Temperature);
		}
	}
	else {

		//non-uniform temperature setting
#pragma omp parallel for
		for (int idx = 0; idx < pMesh->Temp.linear_size(); idx++) {

			if (pMesh->Temp.is_not_empty(idx)) {

				double cT = pMesh->cT;
				pMesh->update_parameters_tcoarse(idx, pMesh->cT, cT);

				pMesh->Temp[idx] = cT * Temperature;

				if (pMesh->Temp_l.linear_size()) {

					pMesh->Temp_l[idx] = cT * Temperature;
				}
			}
		}
	}
}

//transfer values from globalTemp to Temp in this mesh (cT scaling still applied)
//globalTemp values are scaled by cT, and then added to base temperature
void Heat::SetFromGlobalTemperature(VEC<double>& globalTemp)
{
#pragma omp parallel for
	for (int idx = 0; idx < pMesh->Temp.linear_size(); idx++) {

		if (pMesh->Temp.is_not_empty(idx)) {

			DBL3 abs_pos = pMesh->Temp.cellidx_to_position(idx) + pMesh->Temp.rect.s;

			double cT = pMesh->cT;
			pMesh->update_parameters_tcoarse(idx, pMesh->cT, cT);

			if (globalTemp.rect.contains(abs_pos)) {

				pMesh->Temp[idx] = pMesh->base_temperature + globalTemp[abs_pos - globalTemp.rect.s] * cT;
			}
			else {

				pMesh->Temp[idx] = pMesh->base_temperature;
			}

			if (pMesh->Temp_l.linear_size()) pMesh->Temp_l[idx] = pMesh->Temp[idx];
		}
	}
}

//-------------------Others

//called by MoveMesh method in this mesh - move relevant transport quantities
void Heat::MoveMesh_Heat(double x_shift)
{
	double mesh_end_size = pMesh->meshRect.size().x * MOVEMESH_ENDRATIO;

	Rect shift_rect = Rect(pMesh->meshRect.s + DBL3(mesh_end_size, 0, 0), pMesh->meshRect.e - DBL3(mesh_end_size, 0, 0));

#if COMPILECUDA == 1
	if (pModuleCUDA) {

		pMesh->pMeshCUDA->Temp.shift_x(x_shift, shift_rect);

		if (pMesh->Temp_l.linear_size()) pMesh->pMeshCUDA->Temp_l.shift_x(x_shift, shift_rect);

		return;
	}
#endif

	pMesh->Temp.shift_x(x_shift, shift_rect);

	if (pMesh->Temp_l.linear_size()) pMesh->Temp_l.shift_x(x_shift, shift_rect);
}

#endif