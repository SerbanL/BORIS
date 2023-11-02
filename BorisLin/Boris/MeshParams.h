#pragma once

#include "BorisLib.h"

#include "MaterialParameter.h"
#include "ErrorHandler.h"
#include "Boris_Enums_Defs.h"

#include "MeshParamsBase.h"

#if COMPILECUDA == 1
#include "MeshParamsCUDA.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	The Mesh material parameters : micromagnetic meshes only
//

class MeshParams :
	virtual public MeshParamsBase			//need virtual inheritance : see MeshParamsBase.h comments
{
	friend MeshParamsBase;					//so we can access run_on_param from it

#if COMPILECUDA == 1
	friend MeshParamsCUDA;
#endif

private:

protected:

	//Special functions to be set in material parameters text equations when needed

	//resolution of 10000 means e.g. for Tc = 1000 the Curie-Weiss function will be available with a resolution of 0.1 K
	std::shared_ptr<Funcs_Special> pCurieWeiss = nullptr;
	std::shared_ptr<Funcs_Special> pLongRelSus = nullptr;
	std::shared_ptr<Funcs_Special> pCurieWeiss1 = nullptr;
	std::shared_ptr<Funcs_Special> pCurieWeiss2 = nullptr;
	std::shared_ptr<Funcs_Special> pLongRelSus1 = nullptr;
	std::shared_ptr<Funcs_Special> pLongRelSus2 = nullptr;
	std::shared_ptr<Funcs_Special> pAlpha1 = nullptr;
	std::shared_ptr<Funcs_Special> pAlpha2 = nullptr;

public:

	//-------------------------------- LIST ALL MESH PARAMETERS HERE

	//A number of parameters have _AFM termination. These are used for antiferromagnetic meshes with 2-sublattice local approximation and are doubled-up for sub-lattices A, B

	//Relative electron gyromagnetic ratio
	MatP<double, double> grel = 1.0;
	MatP<DBL2, double> grel_AFM = DBL2(1.0);

	//Gilbert damping
	MatP<double, double> alpha = 0.02;
	MatP<DBL2, double> alpha_AFM = DBL2(0.02);

	//Saturation magnetization (A/m)
	MatP<double, double> Ms = 8e5;
	MatP<DBL2, double> Ms_AFM = DBL2(8e5);

	//in-plane demagnetizing factors (used for Demag_N module)
	MatP<DBL2, double> Nxy = DBL2(0);

	//Exchange stiffness (J/m)
	MatP<double, double> A = 1.3e-11;
	MatP<DBL2, double> A_AFM = DBL2(1.3e-11);

	//Homogeneous AFM coupling between sub-lattices A and B, defined as A / a*a (J/m^3), where A is the homogeneous antiferromagnetic exchange stifness (negative), and a is the lattice constant.
	//e.g. a = 0.3nm, A = -1pJ/m gives Ah as -1e7 J/m^3 to order of magnitude.
	MatP<DBL2, double> Ah = DBL2(-1e+7);

	//Nonhomogeneous AFM coupling between sub-lattices A and B (J/m)
	MatP<DBL2, double> Anh = DBL2(-10e-12);

	//Dzyaloshinskii-Moriya exchange constant (J/m^2)
	MatP<double, double> D = 3e-3;
	MatP<DBL2, double> D_AFM = DBL2(3e-3);

	//Interfacial DMI symmetry axis direction, used by vector interfacial DMI module
	MatP<DBL3, DBL3> D_dir = DBL3(0, 0, 1);

	//Homogeneous DMI constant for 2-sublattice models (J/m^3)
	MatP<double, double> Dh = 0.0;
	//Homogeneous DMI term orientation
	MatP<DBL3, DBL3> dh_dir = DBL3(1, 0, 0);

	//Coupling between exchange integral and critical temperature (Neel or Curie temperature) for 2-sublattice model : intra-lattice term, 0.5 for ideal antiferromagnet
	//J = 3 * tau * kB * Tc
	MatP<DBL2, double> tau_ii = DBL2(0.5);

	//Coupling between exchange integral and critical temperature (Neel or Curie temperature) for 2-sublattice model : inter-lattice, or cross-lattice term, 0.5 for ideal antiferromagnet.
	//J = 3 * tau * kB * Tc
	MatP<DBL2, double> tau_ij = DBL2(0.5);

	//bilinear surface exchange coupling (J/m^2) : J1
	//biquadratic surface exchange coupling (J/m^2) : J2
	//For coupled meshes it is the top mesh that sets the J values.
	MatP<double, double> J1 = -1e-3;
	MatP<double, double> J2 = 0;

	//Magneto-crystalline anisotropy K1 and K2 constants (J/m^3) and easy axes directions. For uniaxial anisotropy only ea1 is needed, for cubic ea1 and ea2 should be orthogonal.
	MatP<double, double> K1 = 1e4;
	MatP<double, double> K2 = 0;
	MatP<double, double> K3 = 0;
	MatP<DBL3, DBL3> mcanis_ea1 = DBL3(1, 0, 0);
	MatP<DBL3, DBL3> mcanis_ea2 = DBL3(0, 1, 0);
	MatP<DBL3, DBL3> mcanis_ea3 = DBL3(0, 0, 1);

	//tensorial anisotropy. each term is a contribution to the anisotropy energy density as d*a^n1 b^n2 c^n3. Here a = m.mcanis_ea1, b = m.mcanis_ea2, c = m.mcanis_ea3.
	//For 2nd order we aditionally multiply by K1, 4th order K2, 6th order K3. Any other orders d coefficient contains anisotropy energy density.
	//each DBL4 stores (d, n1, n2, n3), where d != 0, n1, n2, n3 >= 0, n1+n2+n3>0. Odd order terms allowed.
	std::vector<DBL4> Kt, Kt2;

	//Anisotropy values for 2-sublattice model
	MatP<DBL2, double> K1_AFM = DBL2(1e5);
	MatP<DBL2, double> K2_AFM = DBL2(0.0);
	MatP<DBL2, double> K3_AFM = DBL2(0.0);

	//longitudinal (parallel) susceptibility relative to mu0*Ms0, i.e. divided by mu0*Ms0, Ms0 is the 0K Ms value - for use with LLB equation. Units As^2/kg
	MatP<double, double> susrel = 1.0;

	//longitudinal (parallel) susceptibility relative to mu0*Ms0, i.e. divided by mu0*Ms0, Ms0 is the 0K Ms value - for use with LLB equation 2-sublattice model. Units As^2/kg
	MatP<DBL2, double> susrel_AFM = DBL2(1.0);

	//perpendicular (transverse) susceptibility relative to mu0*Ms0, i.e. divided by mu0*Ms0, Ms0 is the 0K Ms value - for use with LLB equation. Units As^2/kg
	MatP<double, double> susprel = 1.0;

	//applied field spatial variation coefficient (unitless)
	MatP<double, double> cHA = 1.0;

	//Magneto-Optical field strength (A/m)
	MatP<double, double> cHmo = 0.0;

	//Stochasticity efficiency parameter
	MatP<double, double> s_eff = 1.0;

	//electrical conductivity (units S/m).
	//this is the value at RT for Ni80Fe20.
	MatP<double, double> elecCond = 7e6;

	//TMR RA products for parallel and antiparallel states (Ohms m^2)
	MatP<double, double> RAtmr_p = 0.075e-12;
	MatP<double, double> RAtmr_ap = 0.225e-12;

	//anisotropic magnetoresistance as a percentage (of base resistance)
	MatP<double, double> amrPercentage = 0.0;

	//tunneling anisotropic magnetoresistance as a percentage
	MatP<double, double> tamrPercentage = 0.0;

	//spin current polarization (also the charge current spin polarization for the spin current solver) and non-adiabaticity (for Zhang-Li STT). (unitless)
	MatP<double, double> P = 0.4;
	MatP<double, double> beta = 0.04;

	//parameters for spin current solver

	//electron diffusion constant (m^2/s)
	MatP<double, double> De = 1e-2;

	//electron carrier density (1/m^3)
	MatP<double, double> n_density = 1.8e29;

	//diffusion spin polarization (unitless)
	MatP<double, double> betaD = 0.5;

	//spin Hall angle (unitless)
	MatP<double, double> SHA = 0.1;

	//"inverse" spin Hall angle (unitless) -> should normally be the same as SHA but e.g. can be set to zero to turn off the inverse SHE in the spin transport equation
	MatP<double, double> iSHA = 0.1;

	//field-like spin torque coefficient (unitless)
	MatP<double, double> flSOT = 0.0;
	//second one if STT must be included from top and bottom layers
	MatP<double, double> flSOT2 = 0.0;

	//Slonczewski macrospin torques q+, q- parameters as in PRB 72, 014446 (2005) (unitless)
	MatP<DBL2, double> STq = DBL2(1.0, 0.0);
	//second one if STT must be included from top and bottom layers
	MatP<DBL2, double> STq2 = DBL2(0.0, 0.0);

	//Slonczewski macrospin torques A, B parameters as in PRB 72, 014446 (2005) (unitless)
	MatP<DBL2, double> STa = DBL2(0.6, 0.4);
	//second one if STT must be included from top and bottom layers
	MatP<DBL2, double> STa2 = DBL2(0.0, 0.0);

	//Slonczewski macrospin torques spin polarization unit vector as in PRB 72, 014446 (2005) (unitless); or SOT symmetry axis direction (e.g. z direction for HM/FM bilayer).
	//Set to zero to disable fixed polarization, and instead look at top and bottom layers directly
	MatP<DBL3, DBL3> STp = DBL3(0, 0, 1);

	//spin-flip length (m)
	MatP<double, double> l_sf = 10e-9;

	//spin exchange rotation length (m)
	MatP<double, double> l_ex = 2e-9;

	//spin dephasing length (m)
	MatP<double, double> l_ph = 4e-9;

	//interface spin-dependent conductivity (spin-up and spin-down) (S/m^2)
	MatP<DBL2, double> Gi = DBL2(1e15, 1e14);

	//interface spin-mixing conductivity (real and imaginary parts) (S/m^2)
	MatP<DBL2, double> Gmix = DBL2(1e15, 1e14);

	//spin accumulation torque efficiency in the bulk (unitless, varies from 0 : no torque, up to 1 : full torque)
	MatP<double, double> ts_eff = 1;

	//spin accumulation torque efficiency at interfaces (unitless, varies from 0 : no torque, up to 1 : full torque)
	MatP<double, double> tsi_eff = 1;

	//spin pumping efficiency (unitless, varies from 0 : no spin pumping, up to 1 : full strength)
	//disabled by default
	MatP<double, double> pump_eff = 0;

	//charge pumping efficiency (unitless, varies from 0 : no charge pumping, up to 1 : full strength)
	//disabled by default
	MatP<double, double> cpump_eff = 0;

	//topological Hall effect efficiency (unitless, varies from 0 : none, up to 1 : full strength)
	//disabled by default
	MatP<double, double> the_eff = 0;

	//Curie temperature - 870K for permalloy but turn it off by default. If LLG is the default equation we don't want temperature dependencies to be updated every time the applied field changes.
	//This is the actually set value
	//Can also be used for anti-ferromagnetic meshes (as the Neel temperature), but still calling it T_Curie (I suppose the variable name should be changed to T_critical, but too late now as need to maintain backward compatibility with older simulation files)
	double T_Curie = 0.0;

	//This is the indicative Curie temperature of the material, but not used in any calculations.
	//If you want to turn default temperature dependeces on, this is the value you should set in T_Curie.
	MatP<double, double> T_Curie_material = 870;

	//The atomic magnetic moment as a multiple of the Bohr magneton - default 1 ub for permalloy.
	MatP<double, double> atomic_moment = 1.0;

	//atomic moments for 2-sublattice model (again multiples of the Bohr magneton)
	MatP<DBL2, double> atomic_moment_AFM = DBL2(1.0);

	//Seebeck coefficient (V/K). Set to zero to disable thermoelectric effect (disabled by default).
	MatP<double, double> Sc = 0.0;

	//thermal conductivity (W/mK) - default for permalloy
	MatP<double, double> thermCond = 46.4;

	//Joule heating effect efficiency (unitless, varies from 0 : none, up to 1 : full strength)
	//enabled by default
	MatP<double, double> joule_eff = 1;

	//mass density (kg/m^3) - default for permalloy
	MatP<double, double> density = 8740;

	//specific heat capacity (J/kgK) - default for permalloy
	MatP<double, double> shc = 430;

	//electron specific heat capacity at room temperature used in many-temperature models (J/kgK); Note, if used you should assign a temperature dependence to it, e.g. linear with temperature for the free electron approximation; none assigned by default.
	MatP<double, double> shc_e = 40;

	//electron-lattice coupling constant (W/m^3K) used in two-temperature model.
	MatP<double, double> G_e = 1e18;

	//Magneto-elastic coefficients (J/m^3) - default for Ni
	//B1, B2 for cubic
	//B21, B22 for trigonal
	MatP<DBL2, double> MEc = DBL2(8e6);

	//Magnetostriction coefficients (J/m^3) - default for Ni (should be same as MEc, but can be set independently, e.g. to disable one or the other effect)
	MatP<DBL2, double> mMEc = DBL2(8e6);

	//Magneto-elastic coefficients (J/m^3)
	//B3, B4 for trigonal
	MatP<DBL2, double> MEc2 = DBL2();

	//Magnetostriction coefficients (J/m^3). Should be same as MEc2, but can be set independently, e.g. to disable one or the other effect.
	MatP<DBL2, double> mMEc2 = DBL2();

	//Magneto-elastic coefficients (J/m^3)
	//B14, B34 for trigonal
	MatP<DBL2, double> MEc3 = DBL2();

	//Magnetostriction coefficients (J/m^3). Should be same as MEc2, but can be set independently, e.g. to disable one or the other effect.
	MatP<DBL2, double> mMEc3 = DBL2();

	//Young's modulus (Pa) - default for permalloy
	MatP<double, double> Ym = 113e9;
	
	//Poisson's ratio (unitless) - default for permalloy
	MatP<double, double> Pr = 0.3;

	//Stiffness constants for a cubic system as c11, c12, c44 (N/m^2)
	MatP<DBL3, double> cC = DBL3(3e11, 2e11, 5e10);

	//Stiffness constants as c22, c23, c55 (N/m^2) - needed for Orthorhombic system
	MatP<DBL3, double> cC2 = DBL3(3e11, 2e11, 5e10);

	//Stiffness constants as c33, c13, c66 (N/m^2) - needed for Hexagonal, Tetragonal, Trigonal systems
	MatP<DBL3, double> cC3 = DBL3(3e11, 2e11, 5e10);

	//Stiffness constants as c14, c15, c16 (N/m^2) - needed for Tetragonal (c16), Trigonal (c14, c15) systems
	MatP<DBL3, double> cCs = DBL3(1e10, 1e10, 1e10);

	//mechanical damping value
	MatP<double, double> mdamping = 1e14;

	//coefficient of thermal expansion (thermoelastic constant) - disabled by default; typical value e.g. 12x10^-6 / K for Fe.
	MatP<double, double> thalpha = 0.0;

private:

	//-------------------------Parameter control

	//run set code (run_this) using set parameters (run_this_args) on a MatP object selected through the paramID selector (majorID in meshParams identifying the required material parameter).
	template <typename RType, typename Lambda, typename ... PType>
	RType run_on_param(PARAM_ paramID, Lambda& run_this, PType& ... run_this_args);

protected:

	//set pre-calculated Funcs_Special objects in material parameters
	void set_special_functions(PARAM_ paramID = PARAM_ALL);

public:

	//------------------------CTOR/DTOR

	//inherited by Mesh implementations
	MeshParams(std::vector<PARAM_>& enabledParams);
	
	virtual ~MeshParams() {}

	//-------------------------Setters

	//copy all parameters from another Mesh
	void copy_parameters(MeshParams& copy_this);

	//-------------------------Getters

	std::string get_tensorial_anisotropy_string(void);

	//-------------------------Setters/Updaters : text equations

	//set the mesh parameter temperature equation with given user constants
	void set_meshparam_t_equation(PARAM_ paramID, std::string& equationText, vector_key<double>& userConstants);

	//update text equations for mesh parameters with user constants, mesh dimensions, Curie temperature, base temperature
	bool update_meshparam_equations(PARAM_ paramID, vector_key<double>& userConstants, DBL3 meshDimensions);
};

//-------------------------Parameter control

template <typename RType, typename Lambda, typename ... PType>
RType MeshParams::run_on_param(PARAM_ paramID, Lambda& run_this, PType& ... run_this_args)
{
	switch (paramID) {

	case PARAM_GREL:
		return run_this(grel, run_this_args...);
		break;

	case PARAM_GREL_AFM:
		return run_this(grel_AFM, run_this_args...);
		break;

	case PARAM_GDAMPING:
		return run_this(alpha, run_this_args...);
		break;

	case PARAM_GDAMPING_AFM:
		return run_this(alpha_AFM, run_this_args...);
		break;

	case PARAM_MS:
		return run_this(Ms, run_this_args...);
		break;

	case PARAM_MS_AFM:
		return run_this(Ms_AFM, run_this_args...);
		break;

	case PARAM_DEMAGXY:
		return run_this(Nxy, run_this_args...);
		break;

	case PARAM_A:
		return run_this(A, run_this_args...);
		break;

	case PARAM_A_AFM:
		return run_this(A_AFM, run_this_args...);
		break;

	case PARAM_A_AFH:
		return run_this(Ah, run_this_args...);
		break;

	case PARAM_AFTAU:
		return run_this(tau_ii, run_this_args...);
		break;

	case PARAM_AFTAUCROSS:
		return run_this(tau_ij, run_this_args...);
		break;

	case PARAM_A_AFNH:
		return run_this(Anh, run_this_args...);
		break;

	case PARAM_D:
		return run_this(D, run_this_args...);
		break;

	case PARAM_D_AFM:
		return run_this(D_AFM, run_this_args...);
		break;

	case PARAM_DMI_DH:
		return run_this(Dh, run_this_args...);
		break;

	case PARAM_DMI_DH_DIR:
		return run_this(dh_dir, run_this_args...);
		break;

	case PARAM_DMI_DIR:
		return run_this(D_dir, run_this_args...);
		break;

	case PARAM_J1:
		return run_this(J1, run_this_args...);
		break;

	case PARAM_J2:
		return run_this(J2, run_this_args...);
		break;

	case PARAM_K1:
		return run_this(K1, run_this_args...);
		break;

	case PARAM_K2:
		return run_this(K2, run_this_args...);
		break;

	case PARAM_K3:
		return run_this(K3, run_this_args...);
		break;

	case PARAM_K1_AFM:
		return run_this(K1_AFM, run_this_args...);
		break;

	case PARAM_K2_AFM:
		return run_this(K2_AFM, run_this_args...);
		break;

	case PARAM_K3_AFM:
		return run_this(K3_AFM, run_this_args...);
		break;

	case PARAM_EA1:
		return run_this(mcanis_ea1, run_this_args...);
		break;

	case PARAM_EA2:
		return run_this(mcanis_ea2, run_this_args...);
		break;

	case PARAM_EA3:
		return run_this(mcanis_ea3, run_this_args...);
		break;

	case PARAM_TC:
		return run_this(T_Curie_material, run_this_args...);
		break;

	case PARAM_MUB:
		return run_this(atomic_moment, run_this_args...);
		break;

	case PARAM_MUB_AFM:
		return run_this(atomic_moment_AFM, run_this_args...);
		break;

	case PARAM_SUSREL:
		return run_this(susrel, run_this_args...);
		break;

	case PARAM_SUSREL_AFM:
		return run_this(susrel_AFM, run_this_args...);
		break;

	case PARAM_SUSPREL:
		return run_this(susprel, run_this_args...);
		break;

	case PARAM_HA:
		return run_this(cHA, run_this_args...);
		break;

	case PARAM_HMO:
		return run_this(cHmo, run_this_args...);
		break;

	case PARAM_S_EFF:
		return run_this(s_eff, run_this_args...);
		break;

	case PARAM_ELC:
		return run_this(elecCond, run_this_args...);
		break;

	case PARAM_RATMR_P:
		return run_this(RAtmr_p, run_this_args...);
		break;

	case PARAM_RATMR_AP:
		return run_this(RAtmr_ap, run_this_args...);
		break;

	case PARAM_AMR:
		return run_this(amrPercentage, run_this_args...);
		break;

	case PARAM_TAMR:
		return run_this(tamrPercentage, run_this_args...);
		break;

	case PARAM_P:
		return run_this(P, run_this_args...);
		break;

	case PARAM_BETA:
		return run_this(beta, run_this_args...);
		break;

	case PARAM_DE:
		return run_this(De, run_this_args...);
		break;

	case PARAM_BETAD:
		return run_this(betaD, run_this_args...);
		break;

	case PARAM_SHA:
		return run_this(SHA, run_this_args...);
		break;

	case PARAM_FLSOT:
		return run_this(flSOT, run_this_args...);
		break;

	case PARAM_FLSOT2:
		return run_this(flSOT2, run_this_args...);
		break;

	case PARAM_STQ:
		return run_this(STq, run_this_args...);
		break;

	case PARAM_STQ2:
		return run_this(STq2, run_this_args...);
		break;

	case PARAM_STA:
		return run_this(STa, run_this_args...);
		break;

	case PARAM_STA2:
		return run_this(STa2, run_this_args...);
		break;

	case PARAM_STP:
		return run_this(STp, run_this_args...);
		break;

	case PARAM_ISHA:
		return run_this(iSHA, run_this_args...);
		break;

	case PARAM_LSF:
		return run_this(l_sf, run_this_args...);
		break;

	case PARAM_LEX:
		return run_this(l_ex, run_this_args...);
		break;

	case PARAM_LPH:
		return run_this(l_ph, run_this_args...);
		break;

	case PARAM_GI:
		return run_this(Gi, run_this_args...);
		break;

	case PARAM_GMIX:
		return run_this(Gmix, run_this_args...);
		break;

	case PARAM_TSEFF:
		return run_this(ts_eff, run_this_args...);
		break;

	case PARAM_TSIEFF:
		return run_this(tsi_eff, run_this_args...);
		break;

	case PARAM_PUMPEFF:
		return run_this(pump_eff, run_this_args...);
		break;

	case PARAM_CPUMP_EFF:
		return run_this(cpump_eff, run_this_args...);
		break;

	case PARAM_THE_EFF:
		return run_this(the_eff, run_this_args...);
		break;

	case PARAM_NDENSITY:
		return run_this(n_density, run_this_args...);
		break;

	case PARAM_SEEBECK:
		return run_this(Sc, run_this_args...);
		break;

	case PARAM_JOULE_EFF:
		return run_this(joule_eff, run_this_args...);
		break;

	case PARAM_THERMCOND:
		return run_this(thermCond, run_this_args...);
		break;

	case PARAM_DENSITY:
		return run_this(density, run_this_args...);
		break;

	case PARAM_MECOEFF:
		return run_this(MEc, run_this_args...);
		break;

	case PARAM_MMECOEFF:
		return run_this(mMEc, run_this_args...);
		break;

	case PARAM_MECOEFF2:
		return run_this(MEc2, run_this_args...);
		break;

	case PARAM_MMECOEFF2:
		return run_this(mMEc2, run_this_args...);
		break;

	case PARAM_MECOEFF3:
		return run_this(MEc3, run_this_args...);
		break;

	case PARAM_MMECOEFF3:
		return run_this(mMEc3, run_this_args...);
		break;

	case PARAM_YOUNGSMOD:
		return run_this(Ym, run_this_args...);
		break;

	case PARAM_POISSONRATIO:
		return run_this(Pr, run_this_args...);
		break;

	case PARAM_STIFFC_CUBIC:
		return run_this(cC, run_this_args...);
		break;

	case PARAM_STIFFC_2:
		return run_this(cC2, run_this_args...);
		break;

	case PARAM_STIFFC_3:
		return run_this(cC3, run_this_args...);
		break;

	case PARAM_STIFFC_S:
		return run_this(cCs, run_this_args...);
		break;

	case PARAM_MDAMPING:
		return run_this(mdamping, run_this_args...);
		break;

	case PARAM_THERMEL:
		return run_this(thalpha, run_this_args...);
		break;

	case PARAM_SHC:
		return run_this(shc, run_this_args...);
		break;

	case PARAM_SHC_E:
		return run_this(shc_e, run_this_args...);
		break;

	case PARAM_G_E:
		return run_this(G_e, run_this_args...);
		break;

	case PARAM_T:
		return run_this(cT, run_this_args...);
		break;

	case PARAM_Q:
		return run_this(Q, run_this_args...);
		break;

	default:
		//this is needed to stop the "not all control paths return a value" error, but should never get here
		return run_this(grel, run_this_args...);
		break;
	}
}