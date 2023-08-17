#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#include "BorisCUDALib.h"

#include "MaterialParameterPolicyCUDA.h"
#include "ParametersDefs.h"

class MeshParams;

class MeshParamsCUDA {

private:

	MeshParams *pmeshParams;

protected:

	//Special functions to be set in material parameters text equations when needed

	//resolution of 10000 means e.g. for Tc = 1000 the Curie-Weiss function will be available with a resolution of 0.1 K
	mcu_SpecialFunc CurieWeiss_CUDA;
	mcu_SpecialFunc LongRelSus_CUDA;
	mcu_SpecialFunc CurieWeiss1_CUDA;
	mcu_SpecialFunc CurieWeiss2_CUDA;
	mcu_SpecialFunc LongRelSus1_CUDA;
	mcu_SpecialFunc LongRelSus2_CUDA;
	mcu_SpecialFunc Alpha1_CUDA;
	mcu_SpecialFunc Alpha2_CUDA;

public:

	//A number of parameters have _AFM termination. These are used for antiferromagnetic meshes with 2-sublattice local approximation and are doubled-up for sub-lattices A, B

	//Relative electron gyromagnetic ratio
	mcu_MatPCUDA(cuBReal, cuBReal) grel;
	mcu_MatPCUDA(cuReal2, cuBReal) grel_AFM;

	//Gilbert damping
	mcu_MatPCUDA(cuBReal, cuBReal) alpha;
	mcu_MatPCUDA(cuReal2, cuBReal) alpha_AFM;

	//Saturation magnetization (A/m)
	mcu_MatPCUDA(cuBReal, cuBReal) Ms;
	mcu_MatPCUDA(cuReal2, cuBReal) Ms_AFM;

	//in-plane demagnetizing factors (used for Demag_N module)
	mcu_MatPCUDA(cuReal2, cuBReal) Nxy;

	//Exchange stiffness (J/m)
	mcu_MatPCUDA(cuBReal, cuBReal) A;
	mcu_MatPCUDA(cuReal2, cuBReal) A_AFM;

	//Homogeneous AFM coupling between sub-lattices A and B, defined as A / a*a (J/m^3), where A is the homogeneous antiferromagnetic exchange stifness (negative), and a is the lattice constant.
	//e.g. a = 0.3nm, A = -1pJ/m gives Ah as -1e7 J/m^3 to order of magnitude.
	mcu_MatPCUDA(cuReal2, cuBReal) Ah;

	//Nonhomogeneous AFM coupling between sub-lattices A and B (J/m)
	mcu_MatPCUDA(cuReal2, cuBReal) Anh;

	//Dzyaloshinskii-Moriya exchange constant (J/m^2)
	mcu_MatPCUDA(cuBReal, cuBReal) D;
	mcu_MatPCUDA(cuReal2, cuBReal) D_AFM;

	//Homogeneous DMI constant for 2-sublattice models (J/m^3)
	mcu_MatPCUDA(cuBReal, cuBReal) Dh;
	//Homogeneous DMI term orientation
	mcu_MatPCUDA(cuReal3, cuReal3) dh_dir;

	//Interfacial DMI symmetry axis direction, used by vector interfacial DMI module
	mcu_MatPCUDA(cuReal3, cuReal3) D_dir;

	//Coupling between exchange integral and critical temperature (Neel or Curie temperature) for 2-sublattice model : intra-lattice term, 0.5 for ideal antiferromagnet
	//J = 3 * tau * kB * Tc
	mcu_MatPCUDA(cuReal2, cuBReal) tau_ii;

	//Coupling between exchange integral and critical temperature (Neel or Curie temperature) for 2-sublattice model : inter-lattice, or cross-lattice term, 0.5 for ideal antiferromagnet.
	//J = 3 * tau * kB * Tc
	mcu_MatPCUDA(cuReal2, cuBReal) tau_ij;

	//bilinear surface exchange coupling (J/m^2) : J1, bottom and top layer values
	//biquadratic surface exchange coupling (J/m^2) : J2, bottom and top layer values
	mcu_MatPCUDA(cuBReal, cuBReal) J1;
	mcu_MatPCUDA(cuBReal, cuBReal) J2;

	//Magneto-crystalline anisotropy K1 and K2 constants (J/m^3) and easy axes directions. For uniaxial anisotropy only ea1 is needed, for cubic ea1 and ea2 should be orthogonal.
	mcu_MatPCUDA(cuBReal, cuBReal) K1;
	mcu_MatPCUDA(cuBReal, cuBReal) K2;
	mcu_MatPCUDA(cuBReal, cuBReal) K3;
	mcu_MatPCUDA(cuReal3, cuReal3) mcanis_ea1;
	mcu_MatPCUDA(cuReal3, cuReal3) mcanis_ea2;
	mcu_MatPCUDA(cuReal3, cuReal3) mcanis_ea3;

	//Anisotropy values for 2-sublattice model
	mcu_MatPCUDA(cuReal2, cuBReal) K1_AFM;
	mcu_MatPCUDA(cuReal2, cuBReal) K2_AFM;
	mcu_MatPCUDA(cuReal2, cuBReal) K3_AFM;

	//tensorial anisotropy. each term is a contribution to the anisotropy energy density as d*a^n1 b^n2 c^n3. Here a = m.mcanis_ea1, b = m.mcanis_ea2, c = m.mcanis_ea3.
	//For 2nd order we aditionally multiply by K1, 4th order K2, 6th order K3. Any other orders d coefficient contains anisotropy energy density.
	//each DBL4 stores (d, n1, n2, n3), where d != 0, n1, n2, n3 >= 0, n1+n2+n3>0. Odd order terms allowed.
	mcu_VEC(cuReal4) Kt, Kt2;

	//longitudinal (parallel) susceptibility relative to mu0*Ms0, i.e. divided by mu0*Ms0, Ms0 is the 0K Ms value - for use with LLB equation. Units As^2/kg
	mcu_MatPCUDA(cuBReal, cuBReal) susrel;

	//longitudinal (parallel) susceptibility relative to mu0*Ms0, i.e. divided by mu0*Ms0, Ms0 is the 0K Ms value - for use with LLB equation 2-sublattice model. Units As^2/kg
	mcu_MatPCUDA(cuReal2, cuBReal) susrel_AFM;

	//perpendicular (transverse) susceptibility relative to mu0*Ms0, i.e. divided by mu0*Ms0, Ms0 is the 0K Ms value - for use with LLB equation. Units As^2/kg
	mcu_MatPCUDA(cuBReal, cuBReal) susprel;

	//applied field spatial variation coefficient (unitless)
	mcu_MatPCUDA(cuBReal, cuBReal) cHA;

	//Magneto-Optical field strength (A/m)
	mcu_MatPCUDA(cuBReal, cuBReal) cHmo;

	//Stochasticity efficiency parameter
	mcu_MatPCUDA(cuBReal, cuBReal) s_eff;

	//electrical conductivity (units S/m).
	//this is the value at 0K for Ni80Fe20. Temperature dependence typically scaled by 1 / (1 + alpha*(T-T0)), where alpha = 0.003, T0 = 293K with sigma = 1.7e6 S/m and 293K.
	//Using scaling 1 / (1 + alpha0 * T) on the zero-temperature conductivity gives sigma0 = sigmaT0 / (1 - alpha*T0), alpha0 = alpha / (1 - alpha*T0), so alpha0 = 0.025.
	mcu_MatPCUDA(cuBReal, cuBReal) elecCond;

	//TMR RA products for parallel and antiparallel states (Ohms m^2)
	mcu_MatPCUDA(cuBReal, cuBReal) RAtmr_p;
	mcu_MatPCUDA(cuBReal, cuBReal) RAtmr_ap;

	//anisotropic magnetoresistance as a percentage (of base resistance)
	mcu_MatPCUDA(cuBReal, cuBReal) amrPercentage;
	//tunneling anisotropic magnetoresistance as a percentage
	mcu_MatPCUDA(cuBReal, cuBReal) tamrPercentage;

	//spin current polarization and non-adiabaticity (for Zhang-Li STT).
	mcu_MatPCUDA(cuBReal, cuBReal) P;
	mcu_MatPCUDA(cuBReal, cuBReal) beta;

	//parameters for spin current solver

	//electron diffusion constant (m^2/s)
	mcu_MatPCUDA(cuBReal, cuBReal) De;

	//electron carrier density (1/m^3)
	mcu_MatPCUDA(cuBReal, cuBReal) n_density;

	//diffusion spin polarization (unitless)
	mcu_MatPCUDA(cuBReal, cuBReal) betaD;

	//spin Hall angle (unitless)
	mcu_MatPCUDA(cuBReal, cuBReal) SHA;

	//"inverse" spin Hall angle (unitless) -> should normally be the same as SHA but e.g. can be set to zero to turn off the inverse SHE in the spin transport equation
	mcu_MatPCUDA(cuBReal, cuBReal) iSHA;

	//field-like spin torque coefficient (unitless)
	mcu_MatPCUDA(cuBReal, cuBReal) flSOT;
	mcu_MatPCUDA(cuBReal, cuBReal) flSOT2;

	//Slonczewski macrospin torques q+, q- parameters as in PRB 72, 014446 (2005) (unitless)
	mcu_MatPCUDA(cuReal2, cuBReal) STq;
	mcu_MatPCUDA(cuReal2, cuBReal) STq2;

	//Slonczewski macrospin torques A, B parameters as in PRB 72, 014446 (2005) (unitless)
	mcu_MatPCUDA(cuReal2, cuBReal) STa;
	mcu_MatPCUDA(cuReal2, cuBReal) STa2;

	//Slonczewski macrospin torques spin polarization unit vector as in PRB 72, 014446 (2005) (unitless)
	mcu_MatPCUDA(cuReal3, cuReal3) STp;

	//spin-flip length (m)
	mcu_MatPCUDA(cuBReal, cuBReal) l_sf;

	//spin exchange rotation length (m)
	mcu_MatPCUDA(cuBReal, cuBReal) l_ex;

	//spin dephasing length (m)
	mcu_MatPCUDA(cuBReal, cuBReal) l_ph;

	//interface spin-dependent conductivity (spin-up and spin-down) (S/m^2)
	mcu_MatPCUDA(cuReal2, cuBReal) Gi;

	//interface spin-mixing conductivity (real and imaginary parts) (S/m^2)
	mcu_MatPCUDA(cuReal2, cuBReal) Gmix;

	//spin accumulation torque efficiency in the bulk (unitless, varies from 0 : no torque, up to 1 : full torque)
	mcu_MatPCUDA(cuBReal, cuBReal) ts_eff;

	//spin accumulation torque efficiency at interfaces (unitless, varies from 0 : no torque, up to 1 : full torque)
	mcu_MatPCUDA(cuBReal, cuBReal) tsi_eff;

	//spin pumping efficiency (unitless, varies from 0 : no spin pumping, up to 1 : full strength)
	mcu_MatPCUDA(cuBReal, cuBReal) pump_eff;

	//charge pumping efficiency (unitless, varies from 0 : no charge pumping, up to 1 : full strength)
	mcu_MatPCUDA(cuBReal, cuBReal) cpump_eff;

	//topological Hall effect efficiency (unitless, varies from 0 : none, up to 1 : full strength)
	mcu_MatPCUDA(cuBReal, cuBReal) the_eff;

	//the mesh base temperature (K)
	mcu_val<cuBReal> base_temperature;

	//Curie temperture (K)
	mcu_val<cuBReal> T_Curie;

	//The atomic magnetic moment as a multiple of the Bohr magneton - default 1 ub for permalloy.
	mcu_MatPCUDA(cuBReal, cuBReal) atomic_moment;

	//atomic moments for 2-sublattice model (again multiples of the Bohr magneton)
	mcu_MatPCUDA(cuReal2, cuBReal) atomic_moment_AFM;

	//Seebeck coefficient (V/K). Set to zero to disable thermoelectric effect (disabled by default).
	mcu_MatPCUDA(cuBReal, cuBReal) Sc;

	//Joule heating effect efficiency (unitless, varies from 0 : none, up to 1 : full strength)
	//enabled by default
	mcu_MatPCUDA(cuBReal, cuBReal) joule_eff;

	//thermal conductivity (W/mK)
	mcu_MatPCUDA(cuBReal, cuBReal) thermCond;

	//mass density (kg/m^3)
	mcu_MatPCUDA(cuBReal, cuBReal) density;

	//Magneto-elastic coefficients (J/m^3)
	mcu_MatPCUDA(cuReal2, cuBReal) MEc;

	//Magnetostriction coefficients (J/m^3) - default for Ni (should be same as MEc, but can be set independently, e.g. to disable one or the other effect)
	mcu_MatPCUDA(cuReal2, cuBReal) mMEc;

	//Magneto-elastic coefficients (J/m^3)
	//B3, B4 for trigonal
	mcu_MatPCUDA(cuReal2, cuBReal) MEc2;

	//Magnetostriction coefficients (J/m^3). Should be same as MEc2, but can be set independently, e.g. to disable one or the other effect.
	mcu_MatPCUDA(cuReal2, cuBReal) mMEc2;

	//Magneto-elastic coefficients (J/m^3)
	//B14, B34 for trigonal
	mcu_MatPCUDA(cuReal2, cuBReal) MEc3;

	//Magnetostriction coefficients (J/m^3). Should be same as MEc2, but can be set independently, e.g. to disable one or the other effect.
	mcu_MatPCUDA(cuReal2, cuBReal) mMEc3;

	//Young's modulus (Pa)
	mcu_MatPCUDA(cuBReal, cuBReal) Ym;

	//Poisson's ratio (unitless)
	mcu_MatPCUDA(cuBReal, cuBReal) Pr;

	//Stiffness constants for a cubic system as c11, c12, c44 (N/m^2)
	mcu_MatPCUDA(cuReal3, cuBReal) cC;

	//Stiffness constants as c22, c23, c55 (N/m^2) - needed for Orthorhombic system
	mcu_MatPCUDA(cuReal3, cuBReal) cC2;

	//Stiffness constants as c33, c13, c66 (N/m^2) - needed for Hexagonal, Tetragonal, Trigonal systems
	mcu_MatPCUDA(cuReal3, cuBReal) cC3;

	//Stiffness constants as c14, c15, c16 (N/m^2) - needed for Tetragonal (c16), Trigonal (c14, c15) systems
	mcu_MatPCUDA(cuReal3, cuBReal) cCs;

	//mechanical damping value
	mcu_MatPCUDA(cuBReal, cuBReal) mdamping;

	//coefficient of thermal expansion (thermoelastic constant) - disabled by default; typical value e.g. 12x10^-6 / K for Fe.
	mcu_MatPCUDA(cuBReal, cuBReal) thalpha;

	//specific heat capacity (J/kgK)
	mcu_MatPCUDA(cuBReal, cuBReal) shc;

	//electron specific heat capacity at room temperature used in many-temperature models (J/kgK); Note, if used you should assign a temperature dependence to it, e.g. linear with temperature for the free electron approximation; none assigned by default.
	mcu_MatPCUDA(cuBReal, cuBReal) shc_e;

	//electron-lattice coupling constant (W/m^3K) used in two-temperature model.
	mcu_MatPCUDA(cuBReal, cuBReal) G_e;

	//set temperature spatial variation coefficient (unitless) - used with temperature settings in a simulation schedule only, not with console command directly
	mcu_MatPCUDA(cuBReal, cuBReal) cT;

	//Heat source stimulus in heat equation. Ideally used with a spatial variation. (W//m3)
	mcu_MatPCUDA(cuBReal, cuBReal) Q;

private:

	//set pre-calculated Funcs_Special objects in enabled material parameters
	void set_special_functions(PARAM_ paramID = PARAM_ALL);

public:

	MeshParamsCUDA(MeshParams *pmeshParams);
	virtual ~MeshParamsCUDA();

	void set_special_functions_data(void);
};

#endif
