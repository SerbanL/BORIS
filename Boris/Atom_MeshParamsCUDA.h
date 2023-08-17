#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#include "BorisCUDALib.h"

#include "MaterialParameterPolicyCUDA.h"
#include "ParametersDefs.h"

class Atom_MeshParams;

class Atom_MeshParamsCUDA {

private:

	Atom_MeshParams *pameshParams;

protected:

public:

	//-----------SIMPLE CUBIC

	//Relative electron gyromagnetic ratio
	mcu_MatPCUDA(cuBReal, cuBReal) grel;

	//Gilbert damping (atomistic: intrinsic)
	mcu_MatPCUDA(cuBReal, cuBReal) alpha;

	//atomic moment (units of muB) - default for bcc Fe
	mcu_MatPCUDA(cuBReal, cuBReal) mu_s;

	//Exchange constant (units of J) - default for bcc Fe
	mcu_MatPCUDA(cuBReal, cuBReal) J;

	//DMI exchange constant : (units of J)
	mcu_MatPCUDA(cuBReal, cuBReal) D;

	//Interfacial DMI symmetry axis direction, used by vector interfacial DMI module
	mcu_MatPCUDA(cuReal3, cuReal3) D_dir;

	//Surface exchange coupling, used by the surfexchange module to couple two spins on different meshes at the surface (units of J)
	mcu_MatPCUDA(cuBReal, cuBReal) Js;
	//Secondary surface exchange coupling constant, used for coupling atomistic meshes to micromagnetic 2-sublattice meshes.
	mcu_MatPCUDA(cuBReal, cuBReal) Js2;

	//Magneto-crystalline anisotropy constants (J) and easy axes directions. For uniaxial anisotropy only ea1 is needed.
	mcu_MatPCUDA(cuBReal, cuBReal) K1;
	mcu_MatPCUDA(cuBReal, cuBReal) K2;
	mcu_MatPCUDA(cuBReal, cuBReal) K3;

	//Magneto-crystalline anisotropy easy axes directions
	mcu_MatPCUDA(cuReal3, cuReal3) mcanis_ea1;
	mcu_MatPCUDA(cuReal3, cuReal3) mcanis_ea2;
	mcu_MatPCUDA(cuReal3, cuReal3) mcanis_ea3;

	//tensorial anisotropy. each term is a contribution to the anisotropy energy density as d*a^n1 b^n2 c^n3. Here a = m.mcanis_ea1, b = m.mcanis_ea2, c = m.mcanis_ea3.
	//For 2nd order we aditionally multiply by K1, 4th order K2, 6th order K3. Any other orders d coefficient contains anisotropy energy density.
	//each DBL4 stores (d, n1, n2, n3), where d != 0, n1, n2, n3 >= 0, n1+n2+n3>0. Odd order terms allowed.
	mcu_VEC(cuReal4) Kt;

	//-----------BCC (2 per unit cell)

	//-----------FCC (4 per unit cell)

	//-----------HCP (4 per effective unit cell)

	//-----------Others

	//in-plane demagnetizing factors (used for Atom_Demag_N module)
	mcu_MatPCUDA(cuReal2, cuBReal) Nxy;

	//applied field spatial variation coefficient (unitless)
	mcu_MatPCUDA(cuBReal, cuBReal) cHA;

	//Magneto-Optical field strength (A/m)
	mcu_MatPCUDA(cuBReal, cuBReal) cHmo;

	//Stochasticity efficiency parameter
	mcu_MatPCUDA(cuBReal, cuBReal) s_eff;

	//electrical conductivity (units S/m).
	//this is the value at RT for Ni80Fe20.
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

	//Seebeck coefficient (V/K). Set to zero to disable thermoelectric effect (disabled by default).
	mcu_MatPCUDA(cuBReal, cuBReal) Sc;

	//Joule heating effect efficiency (unitless, varies from 0 : none, up to 1 : full strength)
	//enabled by default
	mcu_MatPCUDA(cuBReal, cuBReal) joule_eff;

	//thermal conductivity (W/mK) - default for permalloy
	mcu_MatPCUDA(cuBReal, cuBReal) thermCond;

	//mass density (kg/m^3) - default for permalloy
	mcu_MatPCUDA(cuBReal, cuBReal) density;

	//specific heat capacity (J/kgK) - default for permalloy
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

public:

	Atom_MeshParamsCUDA(Atom_MeshParams *pameshParams);
	virtual ~Atom_MeshParamsCUDA();
};

#endif

