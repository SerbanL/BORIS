#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "BorisCUDALib.h"

#include "TransportCUDA_Poisson_Policy.h"
#include "TransportCUDA_Poisson_Spin_V_Policy.h"
#include "TransportCUDA_Poisson_Spin_S_Policy.h"

#include "TransportCUDA_PolicyCMBND.h"
#include "TransportCUDA_Spin_V_PolicyCMBND.h"
#include "TransportCUDA_Spin_S_PolicyCMBND.h"

class SuperMeshCUDA;
class SuperMesh;

class MeshBaseCUDA;

class TransportBase;

class TransportBaseCUDA
{
	friend class TransportBase;

	friend TransportCUDA_Spin_V_Funcs;
	friend TransportCUDA_Spin_S_Funcs;

	friend TransportCUDA_Spin_V_CMBND_Pri;
	friend TransportCUDA_Spin_V_CMBND_Sec;

	friend TransportCUDA_Spin_S_CMBND_Pri;
	friend TransportCUDA_Spin_S_CMBND_Sec;

	friend class STransportCUDA;
	friend class Atom_TransportCUDA;
	friend class TransportCUDA;

private:

protected:

	//Pointers

	//pointer to cpu version of this Transport Base
	TransportBase* pTransportBase;

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	MeshBaseCUDA* pMeshBaseCUDA;

	//Same for the supermesh
	SuperMesh* pSMesh;
	SuperMeshCUDA* pSMeshCUDA;

	//Configuration and Display

	//spin transport solver type (see Transport_Defs.h)
	int stsolve;

	//auxiliary for reductions
	mcu_val<cuBReal> auxReal;

	//used to compute spin current components and spin torque for display purposes - only calculated and memory allocated when asked to do so.
	mcu_VEC(cuReal3) displayVEC;

	//used to compute charge current for display purposes (and in some cases we need to run vector calculus operations on it)
	mcu_VEC_VC(cuReal3) displayVEC_VC;

	//Poisson objects

	//TransportCUDA_V_Funcs holds the Poisson_RHS method used in solving the Poisson equation.
	//Pass the managed TransportCUDA_V_Funcs object to IteratePoisson_SOR method in V -  see IterateTransportSolver_SOR method here. IteratePoisson_SOR will then use the Poisson_RHS method defined in TransportCUDA_V_Funcs.
	mcu_obj<TransportCUDA_V_Funcs, TransportCUDA_V_FuncsPolicy> poisson_V;

	//similar for the full spin current solver
	mcu_obj<TransportCUDA_Spin_V_Funcs, TransportCUDA_Spin_V_FuncsPolicy> poisson_Spin_V;
	mcu_obj<TransportCUDA_Spin_S_Funcs, TransportCUDA_Spin_S_FuncsPolicy> poisson_Spin_S;

	//CMBND

	//Charge only
	mcu_obj<TransportCUDA_CMBND_Pri, TransportCUDA_PolicyCMBND<TransportCUDA_CMBND_Pri>> charge_V_cmbnd_funcs_pri;
	mcu_obj<TransportCUDA_CMBND_Sec, TransportCUDA_PolicyCMBND<TransportCUDA_CMBND_Sec>> charge_V_cmbnd_funcs_sec;

	//Spin - charge
	mcu_obj<TransportCUDA_Spin_V_CMBND_Pri, TransportCUDA_Spin_V_PolicyCMBND<TransportCUDA_Spin_V_CMBND_Pri>> spin_V_cmbnd_funcs_pri;
	mcu_obj<TransportCUDA_Spin_V_CMBND_Sec, TransportCUDA_Spin_V_PolicyCMBND<TransportCUDA_Spin_V_CMBND_Sec>> spin_V_cmbnd_funcs_sec;
	
	//Spin - spin
	mcu_obj<TransportCUDA_Spin_S_CMBND_Pri, TransportCUDA_Spin_S_PolicyCMBND<TransportCUDA_Spin_S_CMBND_Pri>> spin_S_cmbnd_funcs_pri;
	mcu_obj<TransportCUDA_Spin_S_CMBND_Sec, TransportCUDA_Spin_S_PolicyCMBND<TransportCUDA_Spin_S_CMBND_Sec>> spin_S_cmbnd_funcs_sec;

	//Calculation auxiliaries

	//dM_dt VEC when we need to do vector calculus operations on it
	mcu_VEC_VC(cuReal3) dM_dt;

	//for Poisson equations for V and S some values are fixed during relaxation, so pre-calculate them and store here to re-use.
	mcu_VEC(cuBReal) delsq_V_fixed;
	mcu_VEC(cuReal3) delsq_S_fixed;

	//for meshes with thermoelectric effect enabled count net current (A) coming out of the mesh at cmbnd cells, if open potential enabled.
	mcu_val<cuBReal> mesh_thermoelectric_net_current;

	//does current mesh have thermoelectric effect enabled? (i.e. must have heat module and Sc not zero)
	bool is_thermoelectric_mesh = false;

	//user test equation for TAMR. If not set use default formula.
	mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal> TAMR_conductivity_equation;

protected:

	//-------------------Auxiliary

	//set the stsolve indicator depending on current configuration
	void Set_STSolveType(void);

	//prepare displayVEC ready for calculation of display quantity
	bool PrepareDisplayVEC(DBL3 cellsize);

	//prepare displayVEC_VC ready for calculation of display quantity - used for charge current
	bool PrepareDisplayVEC_VC(DBL3 cellsize);

	//-------------------Calculation Methods

	//calculate electric field as the negative gradient of V
	virtual void CalculateElectricField(bool open_potential = false) = 0;

	//Charge transport only : V

	//take a single iteration of the Transport solver in this Mesh (cannot solve fully in one go as there may be other meshes so need boundary conditions set externally). Use SOR. Return relaxation value (tends to zero).
	virtual void IterateChargeSolver_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value) = 0;

	//Calculation Methods used by Spin Current Solver only

	//before iterating the spin solver (charge part) we need to prime it : pre-compute values which do not change as the spin solver relaxes.
	virtual void PrimeSpinSolver_Charge(void) = 0;

	//take a single iteration of the charge transport solver (within the spin current solver) in this Mesh (cannot solve fully in one go as there may be other meshes so need boundary conditions set externally). Use SOR. Return relaxation value (tends to zero).
	//use_NNeu flag indicates if we need to use the homogeneous or non-homogeneous Neumann boundary conditions versions
	virtual void IterateSpinSolver_Charge_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value, bool use_NNeu) = 0;

	//before iterating the spin solver (spin part) we need to prime it : pre-compute values which do not change as the spin solver relaxes.
	virtual void PrimeSpinSolver_Spin(void) = 0;

	//solve for spin accumulation using Poisson equation for delsq_S. Use SOR.
	//use_NNeu flag indicates if we need to use the homogeneous or non-homogeneous Neumann boundary conditions versions
	virtual void IterateSpinSolver_Spin_SOR(mcu_val<cuBReal>& damping, mcu_val<cuBReal>& max_error, mcu_val<cuBReal>& max_value, bool use_NNeu) = 0;

	//Other Calculation Methods

	//calculate total current flowing into an electrode with given box - return ground_current and net_current values
	cuBReal CalculateElectrodeCurrent(cuBox electrode_box, cuINT3 sign);

	//------------------Others

	//set fixed potential cells in this mesh for given rectangle
	bool SetFixedPotentialCells(cuRect rectangle, cuBReal potential);

	void ClearFixedPotentialCells(void);

	void Set_Linear_PotentialDrop(cuRect contact1, cuBReal potential1, cuRect contact2, cuBReal potential2, cuReal2 degeneracy);

	//check if dM_dt Calculation should be enabled
	virtual bool Need_dM_dt_Calculation(void) = 0;

	//check if the delsq_V_fixed VEC is needed
	virtual bool Need_delsq_V_fixed_Precalculation(void) = 0;

	//check if the delsq_S_fixed VEC is needed
	virtual bool Need_delsq_S_fixed_Precalculation(void) = 0;

public:

	TransportBaseCUDA(TransportBase* pTransportBase_);
	virtual ~TransportBaseCUDA();

	//-------------------Properties

	int Get_STSolveType(void) { return stsolve; }

	bool GInterface_Enabled(void);

	bool iSHA_nonzero(void);
	bool SHA_nonzero(void);

	//-------------------Auxiliary

	//in order to compute differential operators, halos must be exchange on respective quantites
	//these functions exchange all halos
	//charge solver only
	virtual void exchange_all_halos_charge(void) = 0;
	//spin solver
	virtual void exchange_all_halos_spin(void) = 0;

	//-------------------Public calculation Methods

	//calculate the field resulting from a spin accumulation (spin current solver enabled) so a spin accumulation torque results when using the LLG or LLB equations
	virtual void CalculateSAField(void) = 0;

	//Calculate the field resulting from interface spin accumulation torque for a given contact (in magnetic meshes for NF interfaces with G interface conductance set)
	virtual void CalculateSAInterfaceField(TransportBaseCUDA* ptrans_sec, mCMBNDInfoCUDA& contactCUDA, bool primary_top) = 0;

	//Calculate the interface spin accumulation torque for a given contact (in magnetic meshes for NF interfaces with G interface conductance set), accumulating result in displayVEC
	virtual void CalculateDisplaySAInterfaceTorque(TransportBaseCUDA* ptrans_sec, mCMBNDInfoCUDA& contactCUDA, bool primary_top) = 0;

	//calculate elC VEC using AMR and temperature information
	virtual void CalculateElectricalConductivity(bool force_recalculate = false) = 0;

	//-------------------Display Calculation Methods

	//return x, y, or z component of spin current (component = 0, 1, or 2)
	virtual mcu_VEC(cuReal3)& GetSpinCurrent(int component) = 0;

	virtual mcu_VEC_VC(cuReal3)& GetChargeCurrent(void) = 0;

	//return spin torque computed from spin accumulation
	virtual mcu_VEC(cuReal3)& GetSpinTorque(void) = 0;

	//-------------------TAMR

	BError Set_TAMR_Conductivity_Equation(std::vector<std::vector< EqComp::FSPEC >> fspec);
	void Clear_TAMR_Conductivity_Equation(void);
};

#endif

#endif
