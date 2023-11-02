#include "stdafx.h"
#include "Atom_Zeeman.h"
#include "OVF2_Handlers.h"

#if defined(MODULE_COMPILATION_ZEEMAN) && ATOMISTIC == 1

#include "Atom_Mesh.h"
#include "Atom_MeshParamsControl.h"
#include "Atom_MeshParamsControl_Multi.h"

#include "SuperMesh.h"

#if COMPILECUDA == 1
#include "Atom_ZeemanCUDA.h"
#endif

Atom_Zeeman::Atom_Zeeman(Atom_Mesh *paMesh_) :
	Modules(),
	ZeemanBase(),
	ProgramStateNames(this, { VINFO(Ha), VINFO(H_equation), VINFO(Havec) }, {})
{
	paMesh = paMesh_;

	pSMesh = paMesh->pSMesh;

	error_on_create = UpdateConfiguration(UPDATECONFIG_FORCEUPDATE);

	//-------------------------- Is CUDA currently enabled?

	//If cuda is enabled we also need to make the cuda module version
	if (paMesh->cudaEnabled) {

		if (!error_on_create) error_on_create = SwitchCUDAState(true);
	}
}

Atom_Zeeman::~Atom_Zeeman()
{
}

//setup globalField transfer
BError Atom_Zeeman::InitializeGlobalField(void)
{
	BError error(__FUNCTION__);

	if (pSMesh->GetGlobalField().linear_size()) {

		//globalField here has to have same mesh rectangle and discretization as Heff.
		//we could initialize mesh transfer directly in Heff, but better not as it could be used for something else in the future
		if (!globalField.resize(paMesh->h, paMesh->meshRect)) return error(BERROR_OUTOFMEMORY_NCRIT);

		std::vector< VEC<DBL3>* > pVal_from;
		std::vector< VEC<DBL3>* > pVal_to;

		pVal_from.push_back(&(pSMesh->GetGlobalField()));

		if (!globalField.Initialize_MeshTransfer(pVal_from, pVal_to, MESHTRANSFERTYPE_WEIGHTED)) return error(BERROR_OUTOFMEMORY_CRIT);

		globalField.transfer_in();
	}
	else globalField.clear();

	return error;
}

//Set globalField from SMesh::globalField (without mesh transfer, simply read values)
void Atom_Zeeman::SetGlobalField(void)
{
#if COMPILECUDA == 1
	if (pModuleCUDA) {

		dynamic_cast<Atom_ZeemanCUDA*>(pModuleCUDA)->SetGlobalField(pSMesh->GetGlobalFieldCUDA());
		return;
	}
#endif

	VEC<DBL3>& SMesh_globalField = pSMesh->GetGlobalField();

#pragma omp parallel for
	for (int idx = 0; idx < globalField.linear_size(); idx++) {

		DBL3 abs_pos = globalField.cellidx_to_position(idx) + globalField.rect.s;

		if (SMesh_globalField.rect.contains(abs_pos)) {

			globalField[idx] = SMesh_globalField[abs_pos - SMesh_globalField.rect.s];
		}
		else {

			globalField[idx] = DBL3();
		}
	}
}

BError Atom_Zeeman::Initialize(void)
{
	BError error(CLASS_STR(Atom_Zeeman));

	non_empty_volume = paMesh->Get_NonEmpty_Magnetic_Volume();

	//If using Havec make sure size and resolution matches M1
	if (Havec.linear_size() && (Havec.size() != paMesh->M1.size())) {
		if (!Havec.resize(paMesh->h, paMesh->meshRect)) {

			Havec.clear();
			error(BERROR_OUTOFMEMORY_NCRIT);
			initialized = false;
		}
	}

	//if using global field, then initialize mesh transfer if needed
	error = InitializeGlobalField();

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		paMesh->h, paMesh->meshRect,
		(MOD_)paMesh->Get_Module_Heff_Display() == MOD_ZEEMAN || paMesh->IsOutputDataSet_withRect(DATA_E_ZEE),
		(MOD_)paMesh->Get_Module_Energy_Display() == MOD_ZEEMAN || paMesh->IsOutputDataSet_withRect(DATA_E_ZEE),
		false, paMesh->M1.get_sublattice_vectors());
	if (!error)	initialized = true;

	return error;
}

BError Atom_Zeeman::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Zeeman));

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHCHANGE, UPDATECONFIG_SMESH_GLOBALFIELD)) {

		Uninitialize();

		//update mesh dimensions in equation constants
		if (H_equation.is_set()) {

			DBL3 meshDim = paMesh->GetMeshDimensions();

			H_equation.set_constant("Lx", meshDim.x, false);
			H_equation.set_constant("Ly", meshDim.y, false);
			H_equation.set_constant("Lz", meshDim.z, false);
			H_equation.remake_equation();
		}

		//if global field not set, then also clear it here
		if (!pSMesh->GetGlobalField().linear_size()) globalField.clear();
	}

	//------------------------ CUDA UpdateConfiguration if set

#if COMPILECUDA == 1
	if (pModuleCUDA) {

		if (!error) error = pModuleCUDA->UpdateConfiguration(cfgMessage);
	}
#endif

	return error;
}

void Atom_Zeeman::UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage)
{
	if (cfgMessage == UPDATECONFIG_TEQUATION_CONSTANTS) {

		UpdateTEquationUserConstants(false);
	}
	else if (cfgMessage == UPDATECONFIG_TEQUATION_CLEAR) {

		if (H_equation.is_set()) H_equation.clear();
	}

#if COMPILECUDA == 1
	if (pModuleCUDA) {

		pModuleCUDA->UpdateConfiguration_Values(cfgMessage);
	}
#endif
}

BError Atom_Zeeman::MakeCUDAModule(void)
{
	BError error(CLASS_STR(Atom_Zeeman));

#if COMPILECUDA == 1

		if (paMesh->paMeshCUDA) {

			//Note : it is posible pMeshCUDA has not been allocated yet, but this module has been created whilst cuda is switched on. This will happen when a new mesh is being made which adds this module by default.
			//In this case, after the mesh has been fully made, it will call SwitchCUDAState on the mesh, which in turn will call this SwitchCUDAState method; then pMeshCUDA will not be nullptr and we can make the cuda module version
			pModuleCUDA = new Atom_ZeemanCUDA(paMesh->paMeshCUDA, this);
			error = pModuleCUDA->Error_On_Create();
		}

#endif

	return error;
}

double Atom_Zeeman::UpdateField(void)
{
	double energy = 0;

	//////////////////////////////////////////////
	//SINGLE LATTICE
	//////////////////////////////////////////////

	if (paMesh->GetMeshType() == MESH_ATOM_CUBIC) {

		if (!H_equation.is_set()) {

			/////////////////////////////////////////
			// No equation set
			/////////////////////////////////////////

#pragma omp parallel for reduction(+:energy)
			for (int idx = 0; idx < paMesh->n.dim(); idx++) {

				double cHA = paMesh->cHA;
				paMesh->update_parameters_mcoarse(idx, paMesh->cHA, cHA);

				DBL3 Hext = cHA * Ha;
				if (Havec.linear_size()) Hext += Havec[idx];
				if (globalField.linear_size()) Hext += globalField[idx] * cHA;

				paMesh->Heff1[idx] = Hext;

				energy += -MUB_MU0 * paMesh->M1[idx] * Hext;

				if (Module_Heff.linear_size()) Module_Heff[idx] = Hext;
				if (Module_energy.linear_size()) Module_energy[idx] = -MUB_MU0 * paMesh->M1[idx] * Hext / paMesh->M1.h.dim();
			}
		}

		/////////////////////////////////////////
		// Field set from user equation
		/////////////////////////////////////////

		else {

			double time = pSMesh->GetStageTime();

#pragma omp parallel for reduction(+:energy)
			for (int j = 0; j < paMesh->n.y; j++) {
				for (int k = 0; k < paMesh->n.z; k++) {
					for (int i = 0; i < paMesh->n.x; i++) {

						int idx = i + j * paMesh->n.x + k * paMesh->n.x * paMesh->n.y;

						//on top of spatial dependence specified through an equation, also allow spatial dependence through the cHA parameter
						double cHA = paMesh->cHA;
						paMesh->update_parameters_mcoarse(idx, paMesh->cHA, cHA);

						DBL3 relpos = DBL3(i + 0.5, j + 0.5, k + 0.5) & paMesh->h;
						DBL3 H = H_equation.evaluate_vector(relpos.x, relpos.y, relpos.z, time);

						DBL3 Hext = cHA * H;
						if (Havec.linear_size()) Hext += Havec[idx];
						if (globalField.linear_size()) Hext += globalField[idx] * cHA;

						paMesh->Heff1[idx] = Hext;

						energy += -MUB_MU0 * paMesh->M1[idx] * Hext;

						if (Module_Heff.linear_size()) Module_Heff[idx] = Hext;
						if (Module_energy.linear_size()) Module_energy[idx] = -MUB_MU0 * paMesh->M1[idx] * Hext / paMesh->M1.h.dim();
					}
				}
			}
		}
	}

	//////////////////////////////////////////////
	//MULTIPLE SUB-LATTICES
	//////////////////////////////////////////////

	else {

		if (!H_equation.is_set()) {

			/////////////////////////////////////////
			// No equation set
			/////////////////////////////////////////

			for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for reduction(+:energy)
				for (int idx = 0; idx < paMesh->n.dim(); idx++) {

					double cHA = paMesh->cHA(sidx);
					paMesh->update_parameters_mcoarse(sidx, idx, paMesh->cHA, cHA);

					DBL3 Hext = cHA * Ha;
					if (Havec.linear_size()) Hext += Havec[idx];
					if (globalField.linear_size()) Hext += globalField[idx] * cHA;

					paMesh->Heff1(sidx, idx) = Hext;

					energy += -MUB_MU0 * paMesh->M1(sidx, idx) * Hext;

					if (Module_Heff.linear_size()) Module_Heff(sidx, idx) = Hext;
					if (Module_energy.linear_size()) Module_energy(sidx, idx) = -MUB_MU0 * paMesh->M1(sidx, idx) * Hext / paMesh->M1.h.dim();
				}
			}
		}

		/////////////////////////////////////////
		// Field set from user equation
		/////////////////////////////////////////

		else {

			double time = pSMesh->GetStageTime();

			for (int sidx = 0; sidx < paMesh->M1.get_num_sublattices(); sidx++) {

#pragma omp parallel for reduction(+:energy)
				for (int j = 0; j < paMesh->n.y; j++) {
					for (int k = 0; k < paMesh->n.z; k++) {
						for (int i = 0; i < paMesh->n.x; i++) {

							int idx = i + j * paMesh->n.x + k * paMesh->n.x * paMesh->n.y;

							//on top of spatial dependence specified through an equation, also allow spatial dependence through the cHA parameter
							double cHA = paMesh->cHA(sidx);
							paMesh->update_parameters_mcoarse(sidx, idx, paMesh->cHA, cHA);

							DBL3 relpos = DBL3(i + 0.5, j + 0.5, k + 0.5) & paMesh->h;
							DBL3 H = H_equation.evaluate_vector(relpos.x, relpos.y, relpos.z, time);

							DBL3 Hext = cHA * H;
							if (Havec.linear_size()) Hext += Havec[idx];
							if (globalField.linear_size()) Hext += globalField[idx] * cHA;

							paMesh->Heff1(sidx, idx) = Hext;

							energy += -MUB_MU0 * paMesh->M1(sidx, idx) * Hext;

							if (Module_Heff.linear_size()) Module_Heff(sidx, idx) = Hext;
							if (Module_energy.linear_size()) Module_energy(sidx, idx) = -MUB_MU0 * paMesh->M1(sidx, idx) * Hext / paMesh->M1.h.dim();
						}
					}
				}
			}
		}
	}

	//convert to energy density and return
	if (non_empty_volume) this->energy = energy / non_empty_volume;
	else this->energy = 0.0;

	return this->energy;
}

//-------------------Energy methods

//For simple cubic mesh spin_index coincides with index in M1
double Atom_Zeeman::Get_EnergyChange(int spin_index, DBL3 Mnew)
{
	//For CUDA there are separate device functions used by CUDA kernels.

	if (paMesh->M1.is_not_empty(spin_index)) {

		double cHA = paMesh->cHA;
		paMesh->update_parameters_mcoarse(spin_index, paMesh->cHA, cHA);

		if (!H_equation.is_set()) {

			/////////////////////////////////////////
			// No equation set
			/////////////////////////////////////////

			DBL3 Hext = cHA * Ha;
			if (Havec.linear_size()) Hext += Havec[spin_index];
			if (globalField.linear_size()) Hext += globalField[spin_index] * cHA;

			if (Mnew != DBL3()) return -MUB_MU0 * (Mnew - paMesh->M1[spin_index]) * Hext;
			else return -MUB_MU0 * paMesh->M1[spin_index] * Hext;
		}
		else {

			/////////////////////////////////////////
			// Field set from user equation
			/////////////////////////////////////////

			DBL3 relpos = paMesh->M1.cellidx_to_position(spin_index);
			DBL3 H = H_equation.evaluate_vector(relpos.x, relpos.y, relpos.z, pSMesh->GetStageTime());

			DBL3 Hext = cHA * H;
			if (Havec.linear_size()) Hext += Havec[spin_index];
			if (globalField.linear_size()) Hext += globalField[spin_index] * cHA;

			if (Mnew != DBL3()) return -MUB_MU0 * (Mnew - paMesh->M1[spin_index]) * Hext;
			else return -MUB_MU0 * paMesh->M1[spin_index] * Hext;
		}
	}
	else return 0.0;
}

//----------------------------------------------- Others

void Atom_Zeeman::SetField(DBL3 Hxyz)
{
	//fixed field is being set - remove any equation settings
	if (H_equation.is_set()) H_equation.clear();
	//also release any memory in field VEC
	if (Havec.linear_size()) Havec.clear();

	Ha = Hxyz;

	//-------------------------- CUDA mirroring

#if COMPILECUDA == 1
	if (pModuleCUDA) dynamic_cast<Atom_ZeemanCUDA*>(pModuleCUDA)->SetField(Ha);
#endif
}

DBL3 Atom_Zeeman::GetField(void)
{
	if (H_equation.is_set()) {

		DBL3 meshDim = paMesh->GetMeshDimensions();

		return H_equation.evaluate_vector(meshDim.x / 2, meshDim.y / 2, meshDim.z / 2, pSMesh->GetStageTime());
	}
	else return Ha;
}

BError Atom_Zeeman::SetFieldEquation(std::string equation_string, int step)
{
	BError error(CLASS_STR(Atom_Zeeman));

	DBL3 meshDim = paMesh->GetMeshDimensions();

	//set equation if not already set, or this is the first step in a stage
	if (!H_equation.is_set() || step == 0) {

		//important to set user constants first : if these are required then the make_from_string call will fail. This may mean the user constants are not set when we expect them to.
		UpdateTEquationUserConstants(false);

		if (!H_equation.make_from_string(equation_string, { {"Lx", meshDim.x}, {"Ly", meshDim.y}, {"Lz", meshDim.z}, {"Tb", paMesh->base_temperature}, {"Ss", step} })) return error(BERROR_INCORRECTSTRING);
	}
	else {

		//equation set and not the first step : adjust Ss constant
		H_equation.set_constant("Ss", step);
		UpdateTEquationUserConstants(false);
	}

	//also release any memory in field VEC
	if (Havec.linear_size()) Havec.clear();

	//-------------------------- CUDA mirroring

#if COMPILECUDA == 1
	if (pModuleCUDA) error = dynamic_cast<Atom_ZeemanCUDA*>(pModuleCUDA)->SetFieldEquation(H_equation.get_vector_fspec());
#endif

	return error;
}

BError Atom_Zeeman::SetFieldVEC_FromOVF2(std::string fileName)
{
	BError error(CLASS_STR(Atom_Zeeman));

	//Load data from file
	OVF2 ovf2;
	error = ovf2.Read_OVF2_VEC(fileName, Havec);
	if (error) {

		Havec.clear();
		return error;
	}

	//Make sure size and resolution matches M
	if (Havec.size() != paMesh->M1.size()) {
		if (!Havec.resize(paMesh->h, paMesh->meshRect)) {

			Havec.clear();
			return error(BERROR_OUTOFMEMORY_NCRIT);
		}
	}

	//all good, clear any equation settings
	if (H_equation.is_set()) H_equation.clear();

	//if displaying module effective field also need to update these
	if (Module_Heff.linear_size()) Module_Heff.copy_values(Havec);

	//-------------------------- CUDA mirroring

#if COMPILECUDA == 1
	if (pModuleCUDA) error = dynamic_cast<Atom_ZeemanCUDA*>(pModuleCUDA)->SetFieldVEC(Havec);
#endif

	return error;
}

BError Atom_Zeeman::SetFieldVEC_FromVEC(VEC<DBL3>& Hext)
{
	BError error(CLASS_STR(Atom_Zeeman));

	//Make sure size and resolution matches M
	if (Havec.size() != paMesh->M1.size()) {
		if (!Havec.resize(paMesh->h, paMesh->meshRect)) {

			Havec.clear();
			return error(BERROR_OUTOFMEMORY_NCRIT);
		}
	}

	//now copy from Hext into Havec
	Havec.copy_values(Hext, Havec.rect - Havec.rect.s, Havec.rect - Hext.rect.s);

	return error;
}

#if COMPILECUDA == 1
BError Atom_Zeeman::SetFieldVEC_FromVEC_CUDA(mcu_VEC(cuReal3)& Hext)
{
	BError error(CLASS_STR(Atom_Zeeman));

	if (pModuleCUDA) error = reinterpret_cast<Atom_ZeemanCUDA*>(pModuleCUDA)->SetFieldVEC_FromcuVEC(Hext);

	return error;
}
#endif

//Update TEquation object with user constants values
void Atom_Zeeman::UpdateTEquationUserConstants(bool makeCuda)
{
	if (paMesh->userConstants.size()) {

		std::vector<std::pair<std::string, double>> constants(paMesh->userConstants.size());
		for (int idx = 0; idx < paMesh->userConstants.size(); idx++) {

			constants[idx] = { paMesh->userConstants.get_key_from_index(idx), paMesh->userConstants[idx] };
		}

		H_equation.set_constants(constants);

		//-------------------------- CUDA mirroring

#if COMPILECUDA == 1
		if (pModuleCUDA && makeCuda) dynamic_cast<Atom_ZeemanCUDA*>(pModuleCUDA)->SetFieldEquation(H_equation.get_vector_fspec());
#endif
	}
}

//if base temperature changes we need to adjust Tb in H_equation if it's used.
void Atom_Zeeman::SetBaseTemperature(double Temperature)
{
	if (H_equation.is_set()) {

		H_equation.set_constant("Tb", Temperature);
	}

	//-------------------------- CUDA mirroring

#if COMPILECUDA == 1
	if (pModuleCUDA) dynamic_cast<Atom_ZeemanCUDA*>(pModuleCUDA)->SetFieldEquation(H_equation.get_vector_fspec());
#endif
}

//-------------------Torque methods

DBL3 Atom_Zeeman::GetTorque(Rect& avRect)
{
#if COMPILECUDA == 1
	if (pModuleCUDA) return pModuleCUDA->GetTorque(avRect);
#endif

	return CalculateTorque(paMesh->M1, avRect);
}

#endif