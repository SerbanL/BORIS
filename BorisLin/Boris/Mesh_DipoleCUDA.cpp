#include "stdafx.h"
#include "Mesh_DipoleCUDA.h"

#ifdef MESH_COMPILATION_DIPOLE

#include "Mesh_Dipole.h"

#if COMPILECUDA == 1

DipoleMeshCUDA::DipoleMeshCUDA(DipoleMesh* pDipoleMesh_) :
	MeshCUDA(pDipoleMesh_),
	recalculateStrayField(pDipoleMesh_->recalculateStrayField),
	strayField_recalculated(pDipoleMesh_->strayField_recalculated)
{
	pDipoleMesh = pDipoleMesh_;

	if (!M.set_from_cpuvec(pDipoleMesh->M)) error_on_create(BERROR_OUTOFGPUMEMORY_CRIT);

	error_on_create = UpdateConfiguration(UPDATECONFIG_FORCEUPDATE);
}

DipoleMeshCUDA::~DipoleMeshCUDA()
{
	if (Holder_Mesh_Available()) {

		M.copy_to_cpuvec(pDipoleMesh->M);
	}
}

//----------------------------------- IMPORTANT CONTROL METHODS

//call when a configuration change has occurred - some objects might need to be updated accordingly
BError DipoleMeshCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(DipoleMeshCUDA));
	
	if (M.n.dim()) {

		if (!M.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
	}
	else {

		M.assign(h, meshRect, cuReal3(-(Ms.get_current_cpu()), 0, 0));
	}

	//set dipole value from Ms - this could have changed
	Reset_Mdipole();
	
	return error;
}

//----------------------------------- VARIOUS SET METHODS

//set magnitude for Mdipole
void DipoleMeshCUDA::Reset_Mdipole(void)
{
	//set dipole value from Ms - this could have changed
	if (!pDipoleMesh->Temp.linear_size()) M.renormalize((cuBReal)pDipoleMesh->Ms);
	else {

		cuBReal Temperature = Temp.average_nonempty();
		M.renormalize((cuBReal)pDipoleMesh->Ms.get(Temperature));
	}

	recalculateStrayField = true;
	strayField_recalculated = false;
}

void DipoleMeshCUDA::SetMagAngle(cuBReal polar, cuBReal azim)
{
	pDipoleMesh->M[0] = Polar_to_Cartesian(cuReal3(pDipoleMesh->Ms, polar, azim));
	
	M.copy_from_cpuvec(pDipoleMesh->M);

	recalculateStrayField = true;
	strayField_recalculated = false;
}

//shift dipole mesh rectangle by given amount
void DipoleMeshCUDA::Shift_Dipole(void)
{
	//meshRect is already shifted at this point (since this method is called from the CPU version). Thus all we need to do it set the new mesh rect starting point in cuVECs
	
	M.set_rect_start((cuReal3)meshRect.s);

	if (pDipoleMesh->V.linear_size()) V.set_rect_start((cuReal3)meshRect.s);
	if (pDipoleMesh->S.linear_size()) S.set_rect_start((cuReal3)meshRect.s);
	if (pDipoleMesh->elC.linear_size()) elC.set_rect_start((cuReal3)meshRect.s);
	if (pDipoleMesh->Temp.linear_size()) Temp.set_rect_start((cuReal3)meshRect.s);

	recalculateStrayField = true;
	strayField_recalculated = false;
}

//----------------------------------- VARIOUS GET METHODS

bool DipoleMeshCUDA::CheckRecalculateStrayField(void)
{
	//recalculate stray field if flag is set (Mdipole has changed) or non-uniform temperature is enabled and Ms has a temperature dependence (in this case always recalculate)

	if (pDipoleMesh->Temp.linear_size() && pDipoleMesh->Ms.is_tdep()) {

		cuBReal Temperature = Temp.average_nonempty();
		
		M.renormalize((cuBReal)pDipoleMesh->Ms.get(Temperature));

		return true;
	}

	return recalculateStrayField;
}

#endif
#endif