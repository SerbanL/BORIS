#pragma once

#include "Atom_Mesh.h"

#include "SuperMesh.h"

//----------------------------------- RUNTIME UPDATERS

//////////////////////////////////////////////////
////////////////////////////////////HAVE POSITION

//SPATIAL DEPENDENCE ONLY - HAVE POSITION

//update parameters in the list for spatial dependence only
template <typename PType, typename SType, typename ... MeshParam_List>
void Atom_Mesh::update_parameters_spatial(const DBL3& position, MatP<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params)
{
	if (matp.is_sdep()) matp_value = matp.get(position, pSMesh->GetStageTime());

	update_parameters_spatial(position, params...);
}

//update parameters in the list for spatial dependence only - single parameter version
template <typename PType, typename SType>
void Atom_Mesh::update_parameters_spatial(const DBL3& position, MatP<PType, SType>& matp, PType& matp_value)
{
	if (matp.is_sdep()) matp_value = matp.get(position, pSMesh->GetStageTime());
}

//SPATIAL AND TEMPERATURE DEPENDENCE - HAVE POSITION

//update parameters in the list for spatial dependence only
template <typename PType, typename SType, typename ... MeshParam_List>
void Atom_Mesh::update_parameters_full(const DBL3& position, const double& Temperature, MatP<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params)
{
	if (matp.is_sdep()) matp_value = matp.get(position, pSMesh->GetStageTime(), Temperature);
	else if (matp.is_tdep()) matp_value = matp.get(Temperature);

	update_parameters_full(position, Temperature, params...);
}

//update parameters in the list for spatial dependence only - single parameter version
template <typename PType, typename SType>
void Atom_Mesh::update_parameters_full(const DBL3& position, const double& Temperature, MatP<PType, SType>& matp, PType& matp_value)
{
	if (matp.is_sdep()) matp_value = matp.get(position, pSMesh->GetStageTime(), Temperature);
	else if (matp.is_tdep()) matp_value = matp.get(Temperature);
}

//////////////////////////////////////////////////
////////////////////////////////////M COARSENESS

//SPATIAL DEPENDENCE ONLY - NO POSITION YET

//update parameters in the list for spatial dependence only
template <typename PType, typename SType, typename ... MeshParam_List>
void Atom_Mesh::update_parameters_mcoarse_spatial(int mcell_idx, MatP<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params)
{
	if (matp.is_sdep()) {

		DBL3 position = M1.cellidx_to_position(mcell_idx);

		matp_value = matp.get(position, pSMesh->GetStageTime());
		update_parameters_spatial(position, params...);
	}
	else {

		update_parameters_mcoarse_spatial(mcell_idx, params...);
	}
}

//update parameters in the list for spatial dependence only - single parameter version; position not calculated
template <typename PType, typename SType>
void Atom_Mesh::update_parameters_mcoarse_spatial(int mcell_idx, MatP<PType, SType>& matp, PType& matp_value)
{
	if (matp.is_sdep()) matp_value = matp.get(M1.cellidx_to_position(mcell_idx), pSMesh->GetStageTime());
}

//SPATIAL AND TEMPERATURE DEPENDENCE - NO POSITION YET

//update parameters in the list for spatial dependence only
template <typename PType, typename SType, typename ... MeshParam_List>
void Atom_Mesh::update_parameters_mcoarse_full(int mcell_idx, MatP<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params)
{
	if (matp.is_sdep()) {

		DBL3 position = M1.cellidx_to_position(mcell_idx);
		double Temperature = Temp[position];

		matp_value = matp.get(position, pSMesh->GetStageTime(), Temperature);
		update_parameters_full(position, Temperature, params...);
	}
	else if (matp.is_tdep()) {

		DBL3 position = M1.cellidx_to_position(mcell_idx);
		double Temperature = Temp[position];

		matp_value = matp.get(Temperature);
		update_parameters_full(position, Temperature, params...);
	}
	else {

		update_parameters_mcoarse_full(mcell_idx, params...);
	}
}

//update parameters in the list for spatial dependence only - single parameter version
template <typename PType, typename SType>
void Atom_Mesh::update_parameters_mcoarse_full(int mcell_idx, MatP<PType, SType>& matp, PType& matp_value)
{
	if (matp.is_sdep()) {

		DBL3 position = M1.cellidx_to_position(mcell_idx);

		matp_value = matp.get(position, pSMesh->GetStageTime(), Temp[position]);
	}
	else if (matp.is_tdep()) {

		DBL3 position = M1.cellidx_to_position(mcell_idx);

		matp_value = matp.get(Temp[position]);
	}
}

//UPDATER M COARSENESS - PUBLIC

//Update parameter values if temperature dependent at the given cell index - M cell index; position not calculated
template <typename ... MeshParam_List>
void Atom_Mesh::update_parameters_mcoarse(int mcell_idx, MeshParam_List& ... params)
{
	if (Temp.linear_size()) {

		//check both temperature and spatial dependence
		update_parameters_mcoarse_full(mcell_idx, params...);
	}
	else {

		//only spatial dependence (if any)
		update_parameters_mcoarse_spatial(mcell_idx, params...);
	}
}

//////////////////////////////////////////////////
////////////////////////////////////E COARSENESS

//SPATIAL DEPENDENCE ONLY - NO POSITION YET

//update parameters in the list for spatial dependence only
template <typename PType, typename SType, typename ... MeshParam_List>
void Atom_Mesh::update_parameters_ecoarse_spatial(int ecell_idx, MatP<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params)
{
	if (matp.is_sdep()) {

		DBL3 position = V.cellidx_to_position(ecell_idx);

		matp_value = matp.get(position, pSMesh->GetStageTime());
		update_parameters_spatial(position, params...);
	}
	else {

		update_parameters_ecoarse_spatial(ecell_idx, params...);
	}
}

//update parameters in the list for spatial dependence only - single parameter version; position not calculated
template <typename PType, typename SType>
void Atom_Mesh::update_parameters_ecoarse_spatial(int ecell_idx, MatP<PType, SType>& matp, PType& matp_value)
{
	if (matp.is_sdep()) matp_value = matp.get(V.cellidx_to_position(ecell_idx), pSMesh->GetStageTime());
}

//SPATIAL AND TEMPERATURE DEPENDENCE - NO POSITION YET

//update parameters in the list for spatial dependence only
template <typename PType, typename SType, typename ... MeshParam_List>
void Atom_Mesh::update_parameters_ecoarse_full(int ecell_idx, MatP<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params)
{
	if (matp.is_sdep()) {

		DBL3 position = V.cellidx_to_position(ecell_idx);
		double Temperature = Temp[position];

		matp_value = matp.get(position, pSMesh->GetStageTime(), Temperature);
		update_parameters_full(position, Temperature, params...);
	}
	else if (matp.is_tdep()) {

		DBL3 position = V.cellidx_to_position(ecell_idx);
		double Temperature = Temp[position];

		matp_value = matp.get(Temperature);
		update_parameters_full(position, Temperature, params...);
	}
	else {

		update_parameters_ecoarse_full(ecell_idx, params...);
	}
}

//update parameters in the list for spatial dependence only - single parameter version
template <typename PType, typename SType>
void Atom_Mesh::update_parameters_ecoarse_full(int ecell_idx, MatP<PType, SType>& matp, PType& matp_value)
{
	if (matp.is_sdep()) {

		DBL3 position = V.cellidx_to_position(ecell_idx);

		matp_value = matp.get(position, pSMesh->GetStageTime(), Temp[position]);
	}
	else if (matp.is_tdep()) {

		DBL3 position = V.cellidx_to_position(ecell_idx);

		matp_value = matp.get(Temp[position]);
	}
}

//UPDATER E COARSENESS - PUBLIC

//Update parameter values if temperature dependent at the given cell index - M cell index; position not calculated
template <typename ... MeshParam_List>
void Atom_Mesh::update_parameters_ecoarse(int ecell_idx, MeshParam_List& ... params)
{
	if (Temp.linear_size()) {

		//check both temperature and spatial dependence
		update_parameters_ecoarse_full(ecell_idx, params...);
	}
	else {

		//only spatial dependence (if any)
		update_parameters_ecoarse_spatial(ecell_idx, params...);
	}
}

//////////////////////////////////////////////////
////////////////////////////////////T COARSENESS

//SPATIAL DEPENDENCE ONLY - NO POSITION YET

//update parameters in the list for spatial dependence only
template <typename PType, typename SType, typename ... MeshParam_List>
void Atom_Mesh::update_parameters_tcoarse_spatial(int tcell_idx, MatP<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params)
{
	if (matp.is_sdep()) {

		DBL3 position = Temp.cellidx_to_position(tcell_idx);

		matp_value = matp.get(position, pSMesh->GetStageTime());
		update_parameters_spatial(position, params...);
	}
	else {

		update_parameters_tcoarse_spatial(tcell_idx, params...);
	}
}

//update parameters in the list for spatial dependence only - single parameter version; position not calculated
template <typename PType, typename SType>
void Atom_Mesh::update_parameters_tcoarse_spatial(int tcell_idx, MatP<PType, SType>& matp, PType& matp_value)
{
	if (matp.is_sdep()) matp_value = matp.get(Temp.cellidx_to_position(tcell_idx), pSMesh->GetStageTime());
}

//SPATIAL AND TEMPERATURE DEPENDENCE - NO POSITION YET

//update parameters in the list for spatial dependence only
template <typename PType, typename SType, typename ... MeshParam_List>
void Atom_Mesh::update_parameters_tcoarse_full(int tcell_idx, MatP<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params)
{
	if (matp.is_sdep()) {

		DBL3 position = Temp.cellidx_to_position(tcell_idx);
		double Temperature = Temp[tcell_idx];

		matp_value = matp.get(position, pSMesh->GetStageTime(), Temperature);
		update_parameters_full(position, Temperature, params...);
	}
	else if (matp.is_tdep()) {

		DBL3 position = Temp.cellidx_to_position(tcell_idx);
		double Temperature = Temp[tcell_idx];

		matp_value = matp.get(Temperature);
		update_parameters_full(position, Temperature, params...);
	}
	else {

		update_parameters_tcoarse_full(tcell_idx, params...);
	}
}

//update parameters in the list for spatial dependence only - single parameter version
template <typename PType, typename SType>
void Atom_Mesh::update_parameters_tcoarse_full(int tcell_idx, MatP<PType, SType>& matp, PType& matp_value)
{
	if (matp.is_sdep()) {

		DBL3 position = Temp.cellidx_to_position(tcell_idx);

		matp_value = matp.get(position, pSMesh->GetStageTime(), Temp[tcell_idx]);
	}
	else if (matp.is_tdep()) {

		matp_value = matp.get(Temp[tcell_idx]);
	}
}

//UPDATER T COARSENESS - PUBLIC

//Update parameter values if temperature dependent at the given cell index - M cell index; position not calculated
template <typename ... MeshParam_List>
void Atom_Mesh::update_parameters_tcoarse(int tcell_idx, MeshParam_List& ... params)
{
	if (Temp.linear_size()) {

		//check both temperature and spatial dependence
		update_parameters_tcoarse_full(tcell_idx, params...);
	}
	else {

		//only spatial dependence (if any)
		update_parameters_tcoarse_spatial(tcell_idx, params...);
	}
}

//////////////////////////////////////////////////
////////////////////////////////////S COARSENESS

//SPATIAL DEPENDENCE ONLY - NO POSITION YET

//update parameters in the list for spatial dependence only
template <typename PType, typename SType, typename ... MeshParam_List>
void Atom_Mesh::update_parameters_scoarse_spatial(int scell_idx, MatP<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params)
{
	if (matp.is_sdep()) {

		DBL3 position = u_disp.cellidx_to_position(scell_idx);

		matp_value = matp.get(position, pSMesh->GetStageTime());
		update_parameters_spatial(position, params...);
	}
	else {

		update_parameters_scoarse_spatial(scell_idx, params...);
	}
}

//update parameters in the list for spatial dependence only - single parameter version; position not calculated
template <typename PType, typename SType>
void Atom_Mesh::update_parameters_scoarse_spatial(int scell_idx, MatP<PType, SType>& matp, PType& matp_value)
{
	if (matp.is_sdep()) matp_value = matp.get(u_disp.cellidx_to_position(scell_idx), pSMesh->GetStageTime());
}

//SPATIAL AND TEMPERATURE DEPENDENCE - NO POSITION YET

//update parameters in the list for spatial dependence only
template <typename PType, typename SType, typename ... MeshParam_List>
void Atom_Mesh::update_parameters_scoarse_full(int scell_idx, MatP<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params)
{
	if (matp.is_sdep()) {

		DBL3 position = u_disp.cellidx_to_position(scell_idx);
		double Temperature = Temp[position];

		matp_value = matp.get(position, pSMesh->GetStageTime(), Temperature);
		update_parameters_full(position, Temperature, params...);
	}
	else if (matp.is_tdep()) {

		DBL3 position = u_disp.cellidx_to_position(scell_idx);
		double Temperature = Temp[position];

		matp_value = matp.get(Temperature);
		update_parameters_full(position, Temperature, params...);
	}
	else {

		update_parameters_scoarse_full(scell_idx, params...);
	}
}

//update parameters in the list for spatial dependence only - single parameter version
template <typename PType, typename SType>
void Atom_Mesh::update_parameters_scoarse_full(int scell_idx, MatP<PType, SType>& matp, PType& matp_value)
{
	if (matp.is_sdep()) {

		DBL3 position = u_disp.cellidx_to_position(scell_idx);

		matp_value = matp.get(position, pSMesh->GetStageTime(), Temp[position]);
	}
	else if (matp.is_tdep()) {

		DBL3 position = u_disp.cellidx_to_position(scell_idx);

		matp_value = matp.get(Temp[position]);
	}
}

//UPDATER S COARSENESS - PUBLIC

//Update parameter values if temperature and/or spatially dependent at the given cell index - u_disp cell index; position not calculated
template <typename ... MeshParam_List>
void Atom_Mesh::update_parameters_scoarse(int scell_idx, MeshParam_List& ... params)
{
	if (Temp.linear_size()) {

		//check both temperature and spatial dependence
		update_parameters_scoarse_full(scell_idx, params...);
	}
	else {

		//only spatial dependence (if any)
		update_parameters_scoarse_spatial(scell_idx, params...);
	}
}

//////////////////////////////////////////////////
////////////////////////////////////POSITION KNOWN

//UPDATER POSITION KNOWN - PUBLIC

//Update parameter values if temperature dependent at the given cell index - M cell index; position not calculated
template <typename ... MeshParam_List>
void Atom_Mesh::update_parameters_atposition(const DBL3& position, MeshParam_List& ... params)
{
	if (Temp.linear_size()) {

		//check both temperature and spatial dependence
		update_parameters_full(position, Temp[position], params...);
	}
	else {

		//only spatial dependence (if any)
		update_parameters_spatial(position, params...);
	}
}