#include "stdafx.h"
#include "Atom_Mesh_BCC.h"

#ifdef MESH_COMPILATION_ATOM_BCC

//----------------------------------- VALUE GETTERS

//get topological charge using formula Q = Integral(m.(dm/dx x dm/dy) dxdy) / 4PI
double Atom_Mesh_BCC::GetTopologicalCharge(Rect rectangle)
{
#if COMPILECUDA == 1
	if (paMeshCUDA) return paMeshCUDA->GetTopologicalCharge(rectangle);
#endif

	if (rectangle.IsNull()) rectangle = meshRect;

	if (M1.linear_size()) {

		double Q = 0.0;

#pragma omp parallel for reduction(+:Q)
		for (int idx = 0; idx < M1.linear_size(); idx++) {

			if (M1.is_not_empty(idx)) {

				DBL3 pos = M1.cellidx_to_position(idx);

				if (!rectangle.contains(pos)) continue;

				double Mnorm = M1[idx].norm();

				DBL33 M_grad = M1.grad_neu(idx);

				DBL3 dm_dx = M_grad.x / Mnorm;
				DBL3 dm_dy = M_grad.y / Mnorm;

				Q += (M1[idx] / Mnorm) * (dm_dx ^ dm_dy) * M1.h.x * M1.h.y;
			}
		}

		return Q / (4 * PI * M1.n.z);
	}
	else return 0.0;
}

//get average magnetization in given rectangle (entire mesh if none specified)
DBL3 Atom_Mesh_BCC::GetAverageMoment(Rect rectangle)
{
#if COMPILECUDA == 1
	if (paMeshCUDA) {

		if (M1.linear_size()) return paMeshCUDA->M1.average(rectangle);
		else return DBL3(0.0);
	}
#endif

	if (M1.linear_size()) return M1.average_omp(rectangle);
	else return DBL3(0.0);
}

//get moment magnitude min-max in given rectangle (entire mesh if none specified)
DBL2 Atom_Mesh_BCC::GetMomentMinMax(Rect rectangle)
{
#if COMPILECUDA == 1
	if (paMeshCUDA) {

		if (M1.linear_size()) return paMeshCUDA->M1.get_minmax(rectangle);
		else return DBL2(0.0);
	}
#endif

	if (M1.linear_size()) return M1.get_minmax(rectangle);
	else return DBL2(0.0);
}

//get moment component min-max in given rectangle (entire mesh if none specified)
DBL2 Atom_Mesh_BCC::GetMomentXMinMax(Rect rectangle)
{
#if COMPILECUDA == 1
	if (paMeshCUDA) {

		if (M1.linear_size()) return paMeshCUDA->M1.get_minmax_component_x(rectangle);
		else return DBL2(0.0);
	}
#endif

	if (M1.linear_size()) return M1.get_minmax_component_x(rectangle);
	else return DBL2(0.0);
}

DBL2 Atom_Mesh_BCC::GetMomentYMinMax(Rect rectangle)
{
#if COMPILECUDA == 1
	if (paMeshCUDA) {

		if (M1.linear_size()) return paMeshCUDA->M1.get_minmax_component_y(rectangle);
		else return DBL2(0.0);
	}
#endif

	if (M1.linear_size()) return M1.get_minmax_component_y(rectangle);
	else return DBL2(0.0);
}

DBL2 Atom_Mesh_BCC::GetMomentZMinMax(Rect rectangle)
{
#if COMPILECUDA == 1
	if (paMeshCUDA) {

		if (M1.linear_size()) return paMeshCUDA->M1.get_minmax_component_z(rectangle);
		else return DBL2(0.0);
	}
#endif

	if (M1.linear_size()) return M1.get_minmax_component_z(rectangle);
	else return DBL2(0.0);
}

#endif