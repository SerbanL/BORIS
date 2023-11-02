#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#include "BorisCUDALib.h"

#include "ErrorHandler.h"
#include "ModulesDefs.h"

#include "Atom_MeshParamsCUDA.h"

#include "ManagedAtom_DiffEq_CommonCUDA.h"
#include "ManagedAtom_DiffEqPolicy_CommonCUDA.h"

class Atom_MeshCUDA;
class ManagedMeshCUDA;

//This holds pointers to managed objects in Atom_MeshCUDA (and inherited Atom_MeshParamsCUDA) : set and forget. They are available for use in cuda kernels by passing a cu_obj-managed object ManagedAtom_MeshCUDA

class ManagedAtom_MeshCUDA {

public:

	//Material Parameters

	//-----------SIMPLE CUBIC

	//Relative electron gyromagnetic ratio
	MatPCUDA<cuBReal, cuBReal>* pgrel;

	//Gilbert damping (atomistic: intrinsic)
	MatPCUDA<cuBReal, cuBReal>* palpha;

	//atomic moment (units of muB) - default for bcc Fe
	MatPCUDA<cuBReal, cuBReal>* pmu_s;

	//Exchange constant : (units of J) - default for bcc Fe
	MatPCUDA<cuBReal, cuBReal>* pJ;

	//DMI exchange constant : (units of J)
	MatPCUDA<cuBReal, cuBReal>* pD;

	//Interfacial DMI symmetry axis direction, used by vector interfacial DMI module
	MatPCUDA<cuReal3, cuReal3>* pD_dir;

	//Surface exchange coupling, used by the surfexchange module to couple two spins on different meshes at the surface (units of J)
	MatPCUDA<cuBReal, cuBReal>* pJs;
	//Secondary surface exchange coupling constant, used for coupling atomistic meshes to micromagnetic 2-sublattice meshes.
	MatPCUDA<cuBReal, cuBReal>* pJs2;

	//Magneto-crystalline anisotropy constants (J) and easy axes directions. For uniaxial anisotropy only ea1 is needed.
	MatPCUDA<cuBReal, cuBReal>* pK1;
	MatPCUDA<cuBReal, cuBReal>* pK2;
	MatPCUDA<cuBReal, cuBReal>* pK3;

	//Magneto-crystalline anisotropy easy axes directions
	MatPCUDA<cuReal3, cuReal3>* pmcanis_ea1;
	MatPCUDA<cuReal3, cuReal3>* pmcanis_ea2;
	MatPCUDA<cuReal3, cuReal3>* pmcanis_ea3;

	//Tensorial anisotropy terms
	cuVEC<cuReal4>* pKt;

	//-----------BCC (2 per unit cell)

	//-----------FCC (4 per unit cell)

	//-----------HCP (4 per effective unit cell)

	//-----------Others

	//in-plane demagnetizing factors (used for Atom_Demag_N module)
	MatPCUDA<cuReal2, cuBReal>* pNxy;

	//applied field spatial variation coefficient (unitless)
	MatPCUDA<cuBReal, cuBReal>* pcHA;

	//Magneto-Optical field strength (A/m)
	MatPCUDA<cuBReal, cuBReal>* pcHmo;

	//Stochasticity efficiency parameter
	MatPCUDA<cuBReal, cuBReal>* ps_eff;

	//electrical conductivity (units S/m).
	//this is the value at RT for Ni80Fe20.
	MatPCUDA<cuBReal, cuBReal>* pelecCond;

	//TMR RA products for parallel and antiparallel states (Ohms m^2)
	MatPCUDA<cuBReal, cuBReal>* pRAtmr_p;
	MatPCUDA<cuBReal, cuBReal>* pRAtmr_ap;

	//anisotropic magnetoresistance as a percentage (of base resistance)
	MatPCUDA<cuBReal, cuBReal>* pamrPercentage;

	//tunneling anisotropic magnetoresistance as a percentage
	MatPCUDA<cuBReal, cuBReal>* ptamrPercentage;

	//spin current polarization and non-adiabaticity (for Zhang-Li STT).
	MatPCUDA<cuBReal, cuBReal>* pP;
	MatPCUDA<cuBReal, cuBReal>* pbeta;

	MatPCUDA<cuBReal, cuBReal>* pDe;
	MatPCUDA<cuBReal, cuBReal>* pn_density;
	MatPCUDA<cuBReal, cuBReal>* pbetaD;

	MatPCUDA<cuBReal, cuBReal>* pSHA;
	MatPCUDA<cuBReal, cuBReal>* pflSOT;
	MatPCUDA<cuBReal, cuBReal>* pflSOT2;

	//Slonczewski macrospin torques q+, q- parameters as in PRB 72, 014446 (2005) (unitless)
	MatPCUDA<cuReal2, cuBReal>* pSTq;
	MatPCUDA<cuReal2, cuBReal>* pSTq2;

	//Slonczewski macrospin torques A, B parameters as in PRB 72, 014446 (2005) (unitless)
	MatPCUDA<cuReal2, cuBReal>* pSTa;
	MatPCUDA<cuReal2, cuBReal>* pSTa2;

	//Slonczewski macrospin torques spin polarization unit vector as in PRB 72, 014446 (2005) (unitless)
	MatPCUDA<cuReal3, cuReal3>* pSTp;

	MatPCUDA<cuBReal, cuBReal>* pl_sf;
	MatPCUDA<cuBReal, cuBReal>* pl_ex;
	MatPCUDA<cuBReal, cuBReal>* pl_ph;

	MatPCUDA<cuReal2, cuBReal>* pGi;
	MatPCUDA<cuReal2, cuBReal>* pGmix;

	MatPCUDA<cuBReal, cuBReal>* pts_eff;
	MatPCUDA<cuBReal, cuBReal>* ptsi_eff;

	MatPCUDA<cuBReal, cuBReal>* ppump_eff;
	MatPCUDA<cuBReal, cuBReal>* pcpump_eff;
	MatPCUDA<cuBReal, cuBReal>* pthe_eff;

	//Seebeck coefficient (V/K). Set to zero to disable thermoelectric effect (disabled by default).
	MatPCUDA<cuBReal, cuBReal>* pSc;

	//Joule heating effect efficiency (unitless, varies from 0 : none, up to 1 : full strength)
	//enabled by default
	MatPCUDA<cuBReal, cuBReal>* pjoule_eff;

	//thermal conductivity (W/mK) - default for permalloy
	MatPCUDA<cuBReal, cuBReal>* pthermCond;

	//the mesh base temperature (K)
	cuBReal* pbase_temperature;

	//mass density (kg/m^3) - default for permalloy
	MatPCUDA<cuBReal, cuBReal>* pdensity;

	//specific heat capacity (J/kgK) - default for permalloy
	MatPCUDA<cuBReal, cuBReal>* pshc;

	//electron specific heat capacity at room temperature used in many-temperature models (J/kgK); Note, if used you should assign a temperature dependence to it, e.g. linear with temperature for the free electron approximation; none assigned by default.
	MatPCUDA<cuBReal, cuBReal>* pshc_e;

	//electron-lattice coupling constant (W/m^3K) used in two-temperature model.
	MatPCUDA<cuBReal, cuBReal>* pG_e;

	//set temperature spatial variation coefficient (unitless) - used with temperature settings in a simulation schedule only, not with console command directly
	MatPCUDA<cuBReal, cuBReal>* pcT;

	//Heat source stimulus in heat equation. Ideally used with a spatial variation. (W//m3)
	MatPCUDA<cuBReal, cuBReal>* pQ;

	//-----Magnetic properties

	//Magnetization
	cuVEC_VC<cuReal3>* pM1;

	//effective field - sum total field of all the added modules
	cuVEC<cuReal3>* pHeff1;

	//-----Electric conduction properties (Electron charge and spin Transport)

	//electrical potential - on n_e, h_e mesh
	cuVEC_VC<cuBReal>* pV;

	//electrical conductivity - on n_e, h_e mesh
	cuVEC_VC<cuBReal>* pelC;

	//electrical field - on n_e, h_e mesh
	cuVEC_VC<cuReal3>* pE;

	//spin accumulation - on n_e, h_e mesh
	cuVEC_VC<cuReal3>* pS;

	//-----Thermal conduction properties

	//temperature calculated by Heat module (primary temperature, always used for 1-temperature model; for multi-temperature models in metals this is the itinerant electron temperature)
	cuVEC_VC<cuBReal>* pTemp;

	//lattice temperature used in many-T models
	cuVEC_VC<cuBReal>* pTemp_l;

	//mechanical displacement vectors - on n_m, h_m mesh
	cuVEC_VC<cuReal3>* pu_disp;

	//strain tensor (symmetric):
	//diagonal and off-diagonal components - on n_m, h_m mesh
	//xx, yy, zz
	cuVEC_VC<cuReal3>* pstrain_diag;
	//yz, xz, xy
	cuVEC_VC<cuReal3>* pstrain_odiag;

	//-----

	//Managed cuda mesh pointer so all mesh data can be accessed in device code
	ManagedAtom_DiffEq_CommonCUDA* pcuaDiffEq;

	//----------------------------------- MONTE-CARLO DATA FROM MODULES

	//Atom_SurfExchangeCUDA
	//arrays with pointers to other atomistic meshes in surface exchange coupling with the mesh holding this module, top and bottom (set from Atom_SurfExchangeCUDA)
	ManagedAtom_MeshCUDA** ppaMesh_Top;
	ManagedAtom_MeshCUDA** ppaMesh_Bot;
	size_t paMesh_Top_size, paMesh_Bot_size;

	ManagedMeshCUDA** ppMeshFM_Top;
	ManagedMeshCUDA** ppMeshFM_Bot;
	size_t pMeshFM_Top_size, pMeshFM_Bot_size;

	ManagedMeshCUDA** ppMeshAFM_Top;
	ManagedMeshCUDA** ppMeshAFM_Bot;
	size_t pMeshAFM_Top_size, pMeshAFM_Bot_size;

	ManagedAtom_MeshCUDA** ppaMesh_Bulk;
	ManagedMeshCUDA** ppMeshFM_Bulk;
	ManagedMeshCUDA** ppMeshAFM_Bulk;
	size_t paMesh_Bulk_size, pMeshFM_Bulk_size, pMeshAFM_Bulk_size;
	cuVEC<cuINT3>* pbulk_coupling_mask;

	//Atom_DipoleDipoleCUDA / Atom_DemagMCUDA / SDemag_DemagCUDA
	cuVEC<cuReal3>*  pAtom_Demag_Heff;

	//Atom_ZeemanCUDA
	cuVEC<cuReal3>* pHavec;
	cuVEC<cuReal3>* pglobalField;
	//apled field to be used in MC routine (set before running MC routines)
	cuReal3 Ha_MC;

	//StrayField_AtomMeshCUDA
	cuVEC<cuReal3>* pstrayField;

private:

	//----------------------------------- RUNTIME PARAMETER UPDATERS (AUXILIARY) (Atom_MeshParamsControlCUDA.h)

	//UPDATER M COARSENESS - PRIVATE

	//SPATIAL DEPENDENCE ONLY - NO POSITION YET

	//update parameters in the list for spatial dependence only
	template <typename PType, typename SType, typename ... MeshParam_List>
	__device__ void update_parameters_mcoarse_spatial(int mcell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params);

	//update parameters in the list for spatial dependence only - single parameter version; position not calculated
	template <typename PType, typename SType>
	__device__ void update_parameters_mcoarse_spatial(int mcell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value);

	//SPATIAL AND TEMPERATURE DEPENDENCE - NO POSITION YET

	//update parameters in the list for spatial dependence only
	template <typename PType, typename SType, typename ... MeshParam_List>
	__device__ void update_parameters_mcoarse_full(int mcell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params);

	//update parameters in the list for spatial dependence only - single parameter version
	template <typename PType, typename SType>
	__device__ void update_parameters_mcoarse_full(int mcell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value);

	//UPDATER E COARSENESS - PRIVATE

	//SPATIAL DEPENDENCE ONLY - NO POSITION YET

	//update parameters in the list for spatial dependence only
	template <typename PType, typename SType, typename ... MeshParam_List>
	__device__ void update_parameters_ecoarse_spatial(int ecell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params);

	//update parameters in the list for spatial dependence only - single parameter version; position not calculated
	template <typename PType, typename SType>
	__device__ void update_parameters_ecoarse_spatial(int ecell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value);

	//SPATIAL AND TEMPERATURE DEPENDENCE - NO POSITION YET

	//update parameters in the list for spatial dependence only
	template <typename PType, typename SType, typename ... MeshParam_List>
	__device__ void update_parameters_ecoarse_full(int ecell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params);

	//update parameters in the list for spatial dependence only - single parameter version
	template <typename PType, typename SType>
	__device__ void update_parameters_ecoarse_full(int ecell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value);

	//UPDATER T COARSENESS - PRIVATE

	//SPATIAL DEPENDENCE ONLY - NO POSITION YET

	//update parameters in the list for spatial dependence only
	template <typename PType, typename SType, typename ... MeshParam_List>
	__device__ void update_parameters_tcoarse_spatial(int tcell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params);

	//update parameters in the list for spatial dependence only - single parameter version; position not calculated
	template <typename PType, typename SType>
	__device__ void update_parameters_tcoarse_spatial(int tcell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value);

	//SPATIAL AND TEMPERATURE DEPENDENCE - NO POSITION YET

	//update parameters in the list for spatial dependence only
	template <typename PType, typename SType, typename ... MeshParam_List>
	__device__ void update_parameters_tcoarse_full(int tcell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params);

	//update parameters in the list for spatial dependence only - single parameter version
	template <typename PType, typename SType>
	__device__ void update_parameters_tcoarse_full(int tcell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value);

	//UPDATER S COARSENESS - PRIVATE

	//SPATIAL DEPENDENCE ONLY - NO POSITION YET

	//update parameters in the list for spatial dependence only
	template <typename PType, typename SType, typename ... MeshParam_List>
	__device__ void update_parameters_scoarse_spatial(int scell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params);

	//update parameters in the list for spatial dependence only - single parameter version; position not calculated
	template <typename PType, typename SType>
	__device__ void update_parameters_scoarse_spatial(int scell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value);

	//SPATIAL AND TEMPERATURE DEPENDENCE - NO POSITION YET

	//update parameters in the list for spatial dependence only
	template <typename PType, typename SType, typename ... MeshParam_List>
	__device__ void update_parameters_scoarse_full(int scell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params);

	//update parameters in the list for spatial dependence only - single parameter version
	template <typename PType, typename SType>
	__device__ void update_parameters_scoarse_full(int scell_idx, MatPCUDA<PType, SType>& matp, PType& matp_value);

public:

	void construct_cu_obj(void) {}

	void destruct_cu_obj(void) {}

	BError set_pointers(Atom_MeshCUDA* paMeshCUDA, int idx_device);

	//----------------------------------- RUNTIME PARAMETER UPDATERS (Atom_MeshParamsControlCUDA.h)

	//SPATIAL DEPENDENCE ONLY - HAVE POSITION

	//update parameters in the list for spatial dependence only
	template <typename PType, typename SType, typename ... MeshParam_List>
	__device__ void update_parameters_spatial(const cuReal3& position, MatPCUDA<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params);

	//update parameters in the list for spatial dependence only - single parameter version
	template <typename PType, typename SType>
	__device__ void update_parameters_spatial(const cuReal3& position, MatPCUDA<PType, SType>& matp, PType& matp_value);

	//SPATIAL AND TEMPERATURE DEPENDENCE - HAVE POSITION

	//update parameters in the list for spatial dependence only
	template <typename PType, typename SType, typename ... MeshParam_List>
	__device__ void update_parameters_full(const cuReal3& position, const cuBReal& Temperature, MatPCUDA<PType, SType>& matp, PType& matp_value, MeshParam_List& ... params);

	//update parameters in the list for spatial dependence only - single parameter version
	template <typename PType, typename SType>
	__device__ void update_parameters_full(const cuReal3& position, const cuBReal& Temperature, MatPCUDA<PType, SType>& matp, PType& matp_value);

	//UPDATER M COARSENESS - PUBLIC

	//Update parameter values if temperature dependent at the given cell index - M cell index; position not calculated
	template <typename ... MeshParam_List>
	__device__ void update_parameters_mcoarse(int mcell_idx, MeshParam_List& ... params);

	//UPDATER E COARSENESS - PUBLIC

	//Update parameter values if temperature dependent at the given cell index - M cell index; position not calculated
	template <typename ... MeshParam_List>
	__device__ void update_parameters_ecoarse(int ecell_idx, MeshParam_List& ... params);

	//UPDATER T COARSENESS - PUBLIC

	//Update parameter values if temperature dependent at the given cell index - M cell index; position not calculated
	template <typename ... MeshParam_List>
	__device__ void update_parameters_tcoarse(int tcell_idx, MeshParam_List& ... params);

	//UPDATER S COARSENESS - PUBLIC

	//Update parameter values if temperature and/or spatially dependent at the given cell index - u_disp cell index; position not calculated
	template <typename ... MeshParam_List>
	__device__ void update_parameters_scoarse(int scell_idx, MeshParam_List& ... params);

	//UPDATER POSITION KNOWN - PUBLIC

	//Update parameter values if temperature dependent at the given cell index - M cell index; position not calculated
	template <typename ... MeshParam_List>
	__device__ void update_parameters_atposition(const cuReal3& position, MeshParam_List& ... params);

	//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

	typedef cuBReal(ManagedAtom_MeshCUDA::* pSC_MCFunc)(int, cuReal3);

	//only entries 0, ..., num_SC_MCFuncs - 1 are valid in pSC_MCFuncs
	int num_SC_MCFuncs;

	//array of configured SC energy change functions for Monte Carlo algorithm
	pSC_MCFunc pSC_MCFuncs[MOD_NUM_MODULES];

#if MONTE_CARLO == 1 && ATOMISTIC == 1
	
	//Energy Deltas

	////////////////////////////////////
	//
	// Simple Cubic

	//setup function pointers in pSC_MCFuncs depending on configured modules
	//for each new module, declare an energy function below, define it in its respective module, and also add it as a case in Set_SC_MCFuncs
	void Set_SC_MCFuncs(cu_arr<int>& cuModules);

	//-----MONTE-CARLO DATA - SIMPLE CUBIC

	//Atom_Demag_N
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_DemagNCUDA(int spin_index, cuReal3 Mnew);

	//Atom_Demag
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_DemagCUDA(int spin_index, cuReal3 Mnew);

	//Atom_DipoleDipole
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_DipoleDipoleCUDA(int spin_index, cuReal3 Mnew);

	//StrayField_AtomMesh
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_StrayField_AtomMeshCUDA(int spin_index, cuReal3 Mnew);

	//Atom_ExchangeCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_ExchangeCUDA(int spin_index, cuReal3 Mnew);

	//Atom_DMExchangeCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_DMExchangeCUDA(int spin_index, cuReal3 Mnew);

	//Atom_iDMExchangeCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_iDMExchangeCUDA(int spin_index, cuReal3 Mnew);

	//Atom_viDMExchangeCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_viDMExchangeCUDA(int spin_index, cuReal3 Mnew);

	//Atom_SurfExchangeCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_SurfExchangeCUDA(int spin_index, cuReal3 Mnew);

	//Atom_ZeemanCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_ZeemanCUDA(int spin_index, cuReal3 Mnew);

	//Atom_MOpticalCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_MOpticalCUDA(int spin_index, cuReal3 Mnew);

	//Atom_AnisotropyCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_AnisotropyCUDA(int spin_index, cuReal3 Mnew);

	//Atom_AnisotropyCubiCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_AnisotropyCubiCUDA(int spin_index, cuReal3 Mnew);

	//Atom_AnisotropyBiaxialCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_AnisotropyBiaxialCUDA(int spin_index, cuReal3 Mnew);

	//Atom_AnisotropyTensorialCUDA
	__device__ cuBReal Get_Atomistic_EnergyChange_SC_AnisotropyTensorialCUDA(int spin_index, cuReal3 Mnew);

#else
	void Set_SC_MCFuncs(cu_arr<int>& cuModules) {}
#endif
};

#endif