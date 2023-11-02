#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "BorisCUDALib.h"
#include "ModulesCUDA.h"

#include "TransportBaseCUDA.h"
#include "Transport_Defs.h"

class SuperMeshCUDA;
class SuperMesh;

class MeshCUDA;
class Mesh;

class STransportCUDA;
class Atom_TransportCUDA;

class Transport;

class TransportCUDA :
	public ModulesCUDA,
	public TransportBaseCUDA
{
	friend Transport;
	friend Atom_TransportCUDA;
	friend STransportCUDA;

	friend TransportCUDA_Spin_V_Funcs;
	friend TransportCUDA_Spin_S_Funcs;

private:

	//pointer to cpu version of this module
	Transport* pTransport;

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	MeshCUDA* pMeshCUDA;

	//pointer to cpu version of MeshCUDA
	Mesh* pMesh;

private:

	//-------------------Calculation Methods

	//calculate electrical conductivity with AMR present
	void CalculateElectricalConductivity_AMR(void);

	//calculate electrical conductivity with TAMR present
	void CalculateElectricalConductivity_TAMR(void);

	//calculate electrical conductivity with AMR and TAMR present
	void CalculateElectricalConductivity_TAMR_and_AMR(void);

	//calculate electrical conductivity without AMR
	void CalculateElectricalConductivity_NoAMR(void);

	//calculate electric field as the negative gradient of V
	void CalculateElectricField(bool open_potential = false);

	//Charge transport only : V

	//take a single iteration of the Transport solver in this Mesh (cannot solve fully in one go as there may be other meshes so need boundary conditions set externally). Use SOR. Return relaxation value (tends to zero).
	void IterateChargeSolver_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value);

	//Calculation Methods used by Spin Current Solver only

	//before iterating the spin solver (charge part) we need to prime it : pre-compute values which do not change as the spin solver relaxes.
	void PrimeSpinSolver_Charge(void);

	//take a single iteration of the charge transport solver (within the spin current solver) in this Mesh (cannot solve fully in one go as there may be other meshes so need boundary conditions set externally). Use SOR. Return relaxation value (tends to zero).
	//use_NNeu flag indicates if we need to use the homogeneous or non-homogeneous Neumann boundary conditions versions
	void IterateSpinSolver_Charge_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value, bool use_NNeu);

	//before iterating the spin solver (spin part) we need to prime it : pre-compute values which do not change as the spin solver relaxes.
	void PrimeSpinSolver_Spin(void);

	//solve for spin accumulation using Poisson equation for delsq_S. Use SOR.
	//use_NNeu flag indicates if we need to use the homogeneous or non-homogeneous Neumann boundary conditions versions
	void IterateSpinSolver_Spin_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value, bool use_NNeu);

	//------------------Others

	//check if dM_dt Calculation should be enabled
	bool Need_dM_dt_Calculation(void);

	//check if the delsq_V_fixed VEC is needed
	bool Need_delsq_V_fixed_Precalculation(void);

	//check if the delsq_S_fixed VEC is needed
	bool Need_delsq_S_fixed_Precalculation(void);

public:

	TransportCUDA(Transport* pTransport_);
	~TransportCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);

	//-------------------Auxiliary

	//in order to compute differential operators, halos must be exchange on respective quantites
	//these functions exchange all halos
	//charge solver only
	void exchange_all_halos_charge(void);
	//spin solver
	void exchange_all_halos_spin(void);

	//-------------------Public calculation Methods

	//calculate the field resulting from a spin accumulation (spin current solver enabled) so a spin accumulation torque results when using the LLG or LLB equations
	void CalculateSAField(void);

	//Calculate the field resulting from interface spin accumulation torque for a given contact (in magnetic meshes for NF interfaces with G interface conductance set)
	void CalculateSAInterfaceField(TransportBaseCUDA* ptrans_sec, mCMBNDInfoCUDA& contactCUDA, bool primary_top);

	//Calculate the interface spin accumulation torque for a given contact (in magnetic meshes for NF interfaces with G interface conductance set), accumulating result in displayVEC
	void CalculateDisplaySAInterfaceTorque(TransportBaseCUDA* ptrans_sec, mCMBNDInfoCUDA& contactCUDA, bool primary_top);

	//calculate elC VEC using AMR and temperature information
	void CalculateElectricalConductivity(bool force_recalculate = false);

	//-------------------Display Calculation Methods

	//return x, y, or z component of spin current (component = 0, 1, or 2)
	mcu_VEC(cuReal3)& GetSpinCurrent(int component);

	mcu_VEC_VC(cuReal3)& GetChargeCurrent(void);

	//return spin torque computed from spin accumulation
	mcu_VEC(cuReal3)& GetSpinTorque(void);
};

#else

class TransportCUDA
{
};

#endif

#endif