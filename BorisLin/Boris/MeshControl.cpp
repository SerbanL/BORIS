#include "stdafx.h"
#include "Mesh.h"

#include "TransportBase.h"
#include "HeatBase.h"

//----------------------------------- MESH QUANTITIES CONTROL

void Mesh::MoveMesh(double x_shift)
{
	if (Is_Dormant()) return;

	double mesh_end_size = meshRect.size().x * MOVEMESH_ENDRATIO;

	//the rectangle in which to shift mesh values
	Rect shift_rect = Rect(meshRect.s + DBL3(mesh_end_size, 0, 0), meshRect.e - DBL3(mesh_end_size, 0, 0));

#if COMPILECUDA == 1
	if (pMeshCUDA) {

		//moving mesh for gpu memory quantities

		//1. shift M
		if (M.linear_size()) pMeshCUDA->M.shift_x(x_shift, shift_rect);
		if (M2.linear_size()) pMeshCUDA->M2.shift_x(x_shift, shift_rect);

		//2. shift elC
		if (elC.linear_size()) CallModuleMethod(&TransportBase::MoveMesh_Transport, x_shift);

		//3. shift Temp
		if (Temp.linear_size()) CallModuleMethod(&HeatBase::MoveMesh_Heat, x_shift);

		//4. shift mechanical displacement
		//N/A

		return;
	}
#endif

	//moving mesh for cpu memory quantities

	//1. shift M
	if (M.linear_size()) M.shift_x(x_shift, shift_rect);
	if (M2.linear_size()) M2.shift_x(x_shift, shift_rect);

	//2. shift elC
	if (elC.linear_size()) CallModuleMethod(&TransportBase::MoveMesh_Transport, x_shift);

	//3. shift Temp
	if (Temp.linear_size()) CallModuleMethod(&HeatBase::MoveMesh_Heat, x_shift);

	//4. shift mechanical displacement
	//N/A
}

//for dormant meshes this is a fast call useable at runtime: just shift mesh rectangle and primary quantities, no UpdateConfiguration will be issued
//if not a dormant mesh, then UpdateConfiguration must be called
void Mesh::Shift_Mesh_Rectangle(DBL3 shift)
{
	meshRect += shift;

	//1. Magnetization
	if (M.linear_size()) M.shift_rect_start(shift);
	if (M2.linear_size()) M2.shift_rect_start(shift);
	if (Heff.linear_size()) Heff.shift_rect_start(shift);
	if (Heff2.linear_size()) Heff2.shift_rect_start(shift);

	//2. Electrical Conductivity
	if (elC.linear_size()) elC.shift_rect_start(shift);

	//3. Temperature
	if (Temp.linear_size()) Temp.shift_rect_start(shift);

	//4. Mechanical Displacement
	if (u_disp.linear_size()) u_disp.shift_rect_start(shift);

#if COMPILECUDA == 1
	if (pMeshCUDA) {

		//1. Magnetization
		if (M.linear_size()) pMeshCUDA->M.shift_rect_start(shift);
		if (M2.linear_size()) pMeshCUDA->M2.shift_rect_start(shift);
		if (Heff.linear_size()) pMeshCUDA->Heff.shift_rect_start(shift);
		if (Heff2.linear_size()) pMeshCUDA->Heff2.shift_rect_start(shift);

		//2. Electrical Conductivity
		if (elC.linear_size()) pMeshCUDA->elC.shift_rect_start(shift);

		//3. Temperature
		if (Temp.linear_size()) pMeshCUDA->Temp.shift_rect_start(shift);

		//4. Mechanical Displacement
		if (u_disp.linear_size()) pMeshCUDA->u_disp.shift_rect_start(shift);
	}
#endif

	if (!Is_Dormant()) UpdateConfiguration(UPDATECONFIG_MESHCHANGE);
}

//set PBC for required VECs : should only be called from a demag module
BError Mesh::Set_Magnetic_PBC(INT3 pbc_images)
{
	BError error(__FUNCTION__);

	if (M.linear_size()) M.set_pbc(pbc_images.x, pbc_images.y, pbc_images.z);
	if (M2.linear_size()) M2.set_pbc(pbc_images.x, pbc_images.y, pbc_images.z);

#if COMPILECUDA == 1
	if (pMeshCUDA) {

		if (M.linear_size() && !pMeshCUDA->M.copyflags_from_cpuvec(M)) return error(BERROR_GPUERROR_CRIT);
		if (M2.linear_size() && !pMeshCUDA->M2.copyflags_from_cpuvec(M2)) return error(BERROR_GPUERROR_CRIT);
	}
#endif

	return error;
}