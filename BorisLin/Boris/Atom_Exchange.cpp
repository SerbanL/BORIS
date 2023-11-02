#include "stdafx.h"
#include "Atom_Exchange.h"

#if defined(MODULE_COMPILATION_EXCHANGE) && ATOMISTIC == 1

#include "Atom_Mesh.h"
#include "Atom_MeshParamsControl.h"

#if COMPILECUDA == 1
#include "Atom_ExchangeCUDA.h"
#endif

Atom_Exchange::Atom_Exchange(Atom_Mesh *paMesh_) :
	Modules(),
	ExchangeBase(paMesh_),
	ProgramStateNames(this, {}, {})
{
	paMesh = paMesh_;

	error_on_create = UpdateConfiguration(UPDATECONFIG_FORCEUPDATE);

	//-------------------------- Is CUDA currently enabled?

	//If cuda is enabled we also need to make the cuda module version
	if (paMesh->cudaEnabled) {

		if (!error_on_create) error_on_create = SwitchCUDAState(true);
	}
}

BError Atom_Exchange::Initialize(void)
{
	BError error(CLASS_STR(Atom_Exchange));

	error = ExchangeBase::Initialize();

	//Make sure display data has memory allocated (or freed) as required
	error = Update_Module_Display_VECs(
		paMesh->h, paMesh->meshRect, 
		(MOD_)paMesh->Get_Module_Heff_Display() == MOD_EXCHANGE || paMesh->IsOutputDataSet_withRect(DATA_E_EXCH),
		(MOD_)paMesh->Get_Module_Energy_Display() == MOD_EXCHANGE || paMesh->IsOutputDataSet_withRect(DATA_E_EXCH));
	if (!error)	initialized = true;

	non_empty_volume = paMesh->Get_NonEmpty_Magnetic_Volume();

	//also reinforce coupled to dipoles status
	if (paMesh->pSMesh->Get_Coupled_To_Dipoles()) paMesh->CoupleToDipoles(true);

	return error;
}

BError Atom_Exchange::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(Atom_Exchange));

	Uninitialize();

	Initialize();

	//------------------------ CUDA UpdateConfiguration if set

#if COMPILECUDA == 1
	if (pModuleCUDA) {

		if (!error) error = pModuleCUDA->UpdateConfiguration(cfgMessage);
	}
#endif

	return error;
}

BError Atom_Exchange::MakeCUDAModule(void)
{
	BError error(CLASS_STR(Atom_Exchange));

#if COMPILECUDA == 1

	if (paMesh->paMeshCUDA) {

		//Note : it is posible pMeshCUDA has not been allocated yet, but this module has been created whilst cuda is switched on. This will happen when a new mesh is being made which adds this module by default.
		//In this case, after the mesh has been fully made, it will call SwitchCUDAState on the mesh, which in turn will call this SwitchCUDAState method; then pMeshCUDA will not be nullptr and we can make the cuda module version
		pModuleCUDA = new Atom_ExchangeCUDA(paMesh->paMeshCUDA, this);
		error = pModuleCUDA->Error_On_Create();
	}

#endif

	return error;
}

double Atom_Exchange::UpdateField(void)
{
	double energy = 0;

#pragma omp parallel for reduction(+:energy)
	for (int idx = 0; idx < paMesh->n.dim(); idx++) {

		if (paMesh->M1.is_not_empty(idx)) {

			double mu_s = paMesh->mu_s;
			double J = paMesh->J;
			paMesh->update_parameters_mcoarse(idx, paMesh->mu_s, mu_s, paMesh->J, J);
		
			//update effective field with the Heisenberg exchange field
			DBL3 Heff_value = (J / (MUB_MU0*mu_s)) * paMesh->M1.ngbr_dirsum(idx);

			paMesh->Heff1[idx] += Heff_value;

			//update energy E = -mu_s * Bex. Will finish off at the end with prefactors.
			energy += paMesh->M1[idx] * Heff_value;

			if (Module_Heff.linear_size()) Module_Heff[idx] = Heff_value;
			if (Module_energy.linear_size()) Module_energy[idx] = -MUB_MU0 * paMesh->M1[idx] * Heff_value / (2  * paMesh->M1.h.dim());
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////// COUPLING ACROSS MULTIPLE MESHES ///////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	//if exchange coupling across multiple meshes, this is calculation method to use
	std::function<double(int, int, DBL3, DBL3, DBL3, MeshBase&, MeshBase&)> calculate_coupling = [&]
	(int cell1_idx, int cell2_idx, DBL3 relpos_m1, DBL3 stencil, DBL3 hshift_primary, MeshBase& Mesh_pri, MeshBase& Mesh_sec) -> double {

		double energy_ = 0.0;

		double hR = hshift_primary.norm();
		double hRsq = hR * hR;

		//this applies for atomistic to atomistic mesh coupling (note, the primary mesh here will be atomistic since Atom_Exchange module is held in Atomistic meshes only)
		if (Mesh_pri.is_atomistic() && Mesh_sec.is_atomistic()) {

			Atom_Mesh& mMesh_pri = *reinterpret_cast<Atom_Mesh*>(&Mesh_pri);
			Atom_Mesh& mMesh_sec = *reinterpret_cast<Atom_Mesh*>(&Mesh_sec);

			double mu_s = mMesh_pri.mu_s;
			double J = mMesh_pri.J;
			mMesh_pri.update_parameters_mcoarse(cell1_idx, mMesh_pri.mu_s, mu_s, mMesh_pri.J, J);

			DBL3 Hexch;

			//direction values at cells -1, 2
			DBL3 m_2;
			if (cell2_idx < mMesh_pri.n.dim() && mMesh_pri.M1.is_not_empty(cell2_idx)) m_2 = mMesh_pri.M1[cell2_idx].normalized();

			DBL3 m_m1 = mMesh_sec.M1.weighted_average(relpos_m1, stencil);
			if (m_m1 != DBL3()) m_m1 = m_m1.normalized();

			Hexch = (J / (MUB_MU0 * mu_s)) * (m_2 + m_m1);

			mMesh_pri.Heff1[cell1_idx] += Hexch;

			energy_ = mMesh_pri.M1[cell1_idx] * Hexch;
		}

		//FM to atomistic coupling
		else if (Mesh_pri.is_atomistic() && Mesh_sec.GetMeshType() == MESH_FERROMAGNETIC) {

			Atom_Mesh& mMesh_pri = *reinterpret_cast<Atom_Mesh*>(&Mesh_pri);
			Mesh& mMesh_sec = *reinterpret_cast<Mesh*>(&Mesh_sec);

			double mu_s = mMesh_pri.mu_s;
			double J = mMesh_pri.J;
			mMesh_pri.update_parameters_mcoarse(cell1_idx, mMesh_pri.mu_s, mu_s, mMesh_pri.J, J);

			DBL3 Hexch;

			//direction values at cells -1, 2
			DBL3 m_2;
			if (cell2_idx < mMesh_pri.n.dim() && mMesh_pri.M1.is_not_empty(cell2_idx)) m_2 = mMesh_pri.M1[cell2_idx].normalized();

			DBL3 m_m1;
			if (!mMesh_sec.M.is_empty(relpos_m1)) m_m1 = mMesh_sec.M[relpos_m1].normalized();

			Hexch = (J / (MUB_MU0 * mu_s)) * (m_2 + m_m1);

			mMesh_pri.Heff1[cell1_idx] += Hexch;

			energy_ = mMesh_pri.M1[cell1_idx] * Hexch;
		}

		return energy_;
	};

	//if exchange coupled to other meshes calculate the exchange field at marked cmbnd cells and accumulate energy density contribution
	if (paMesh->GetMeshExchangeCoupling()) CalculateExchangeCoupling(energy, calculate_coupling);

	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////// FINAL ENERGY DENSITY VALUE //////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

	//convert to energy density and return. Divide by two since in the Hamiltonian the sum is performed only once for every pair of spins, but if you use the M.H expression each sum appears twice.
	//Also note, this energy density is not the same as the micromagnetic one, due to different zero-energy points.
	//To obtain the micromagnetic energy density you also have to subtract the energy density obtained at saturation from the Heisenberg Hamiltonian.
	//Could be done here easily, but decided not to (add J * number of neigbhors to energy).
	if (non_empty_volume) this->energy = -MUB_MU0 * energy / (2*non_empty_volume);
	else this->energy = 0.0;

	return this->energy;
}

//-------------------Energy methods

//For simple cubic mesh spin_index coincides with index in M1
double Atom_Exchange::Get_EnergyChange(int spin_index, DBL3 Mnew)
{
	//For CUDA there are separate device functions used by CUDA kernels.

	if (paMesh->M1.is_not_empty(spin_index)) {

		double J = paMesh->J;
		paMesh->update_parameters_mcoarse(spin_index, paMesh->J, J);

		//local spin energy given -J * Sum_over_neighbors_j (Si . Sj), where Si, Sj are unit vectors
		if (Mnew != DBL3()) return -J * ((Mnew.normalized() - paMesh->M1[spin_index].normalized()) * paMesh->M1.ngbr_dirsum(spin_index));
		else return -J * (paMesh->M1[spin_index].normalized() * paMesh->M1.ngbr_dirsum(spin_index));
	}
	else return 0.0;
}

//-------------------Torque methods

DBL3 Atom_Exchange::GetTorque(Rect& avRect)
{
#if COMPILECUDA == 1
	if (pModuleCUDA) return pModuleCUDA->GetTorque(avRect);
#endif

	return CalculateTorque(paMesh->M1, avRect);
}

#endif