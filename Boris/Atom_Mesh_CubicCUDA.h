#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#include "ErrorHandler.h"
#include "Atom_MeshCUDA.h"

#include "Atom_Mesh_CubicCUDA_ThermalizePolicy.h"

#include "BorisCUDALib.h"

#ifdef MESH_COMPILATION_ATOM_CUBIC

class Atom_Mesh_Cubic;
class ManagedAtom_DiffEqCubicCUDA;
class ManagedAtom_DiffEqPolicyCubicCUDA;

//Store Mesh quantities as cu_obj managed cuda VECs
class Atom_Mesh_CubicCUDA :
	public Atom_MeshCUDA
{

private:

	//pointer to cpu version of this mesh
	Atom_Mesh_Cubic *paMeshCubic;

	// MONTE-CARLO DATA

	// Constrained MONTE-CARLO DATA

	//mc indices and shuffling auxiliary array : same as for the cpu version, but generate unsigned random numbers, not doubles, for most efficient sort-based shuffle
	mcu_arr<unsigned> mc_indices_red, mc_indices_black;
	mcu_arr<unsigned> mc_shuf_red, mc_shuf_black;

	// THERMALIZATION CLASS

	mcu_obj<Thermalize_FM_to_Atom, Thermalize_FM_to_AtomPolicy> thermalize_FM_to_Atom;

protected:

public:

	//make this object by copying data from the Mesh holding this object
	Atom_Mesh_CubicCUDA(Atom_Mesh_Cubic* paMesh);

	~Atom_Mesh_CubicCUDA();

	//----------------------------------- IMPORTANT CONTROL METHODS

	//call when a configuration change has occurred - some objects might need to be updated accordingly
	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	//Take a Monte Carlo Metropolis step in this atomistic mesh
	cuBReal Iterate_MonteCarloCUDA_Classic(cuBReal mc_cone_angledeg, double target_acceptance_rate);

	//Take a constrained Monte Carlo Metropolis step in this atomistic mesh
	cuBReal Iterate_MonteCarloCUDA_Constrained(cuBReal mc_cone_angledeg, double target_acceptance_rate);

	//----------------------------------- OTHER IMPORTANT CONTROL METHODS

	//Check if mesh needs to be moved (using the MoveMesh method) - return amount of movement required (i.e. parameter to use when calling MoveMesh).
	cuBReal CheckMoveMesh(bool antisymmetric, double threshold);

	//----------------------------------- OTHER CONTROL METHODS : implement pure virtual Atom_Mesh methods

	//----------------------------------- ALGORITHMS

	//called by Track_Shift_Algorithm when copy_values_thermalize call is required, since this needs to be implemented in a cu file
	void Track_Shift_Algorithm_CopyThermalize(mcu_VEC_VC(cuReal3)& M_src, cuBox cells_box_dst, cuBox cells_box_src);

	//----------------------------------- ENABLED MESH PROPERTIES CHECKERS

	//get exchange_couple_to_meshes status flag from the cpu version
	bool GetMeshExchangeCoupling(void);

	//----------------------------------- CALCULATION METHODS

	//calculate thermodynamic average of magnetization
	cuReal3 GetThermodynamicAverageMagnetization(cuRect rectangle);

	//As for Get_Histogram, but use thermal averaging in each macrocell
	bool Get_ThAvHistogram(std::vector<double>& histogram_x, std::vector<double>& histogram_p, int num_bins, double& min, double& max, cuINT3 macrocell_dims);

	//As for Get_AngHistogram, but use thermal averaging in each macrocell
	bool Get_ThAvAngHistogram(std::vector<double>& histogram_x, std::vector<double>& histogram_p, int num_bins, double& min, double& max, cuINT3 macrocell_dims, cuReal3 ndir);

	//----------------------------------- OTHER CALCULATION METHODS : Atom_Mesh_CubicCUDA_Compute.cpp

	//get topological charge using formula Q = Integral(m.(dm/dx x dm/dy) dxdy) / 4PI
	cuBReal GetTopologicalCharge(cuRect rectangle);

	//compute topological charge density spatial dependence and have it available in auxVEC_cuBReal
	//Use formula Qdensity = m.(dm/dx x dm/dy) / 4PI
	void Compute_TopoChargeDensity(void);

	//return phase transition temperature (K) based on formula Tc = J*e*z/3kB
	double Show_Transition_Temperature(void);

	//return saturation magnetization (A/m) based on formula Ms = mu_s*n/a^3
	double Show_Ms(void);

	//return exchange stiffness (J/m) based on formula A = J*n/2a
	double Show_A(void);

	//return uniaxial anisotropy constant (J/m^3) based on formula K = k*n/a^3
	double Show_Ku(void);

	//----------------------------------- ODE METHODS IN MAGNETIC MESH : Atom_Mesh_CubicCUDA.cu

	//return average dm/dt in the given avRect (relative rect). Here m is the direction vector.
	cuReal3 Average_dmdt(cuBox avBox);

	//return average m x dm/dt in the given avRect (relative rect). Here m is the direction vector.
	cuReal3 Average_mxdmdt(cuBox avBox);

	//-----------------------------------OBJECT GETTERS

	mcu_obj<ManagedAtom_DiffEqCubicCUDA, ManagedAtom_DiffEqPolicyCubicCUDA>& Get_ManagedAtom_DiffEqCUDA(void);
};

#else

class Atom_Mesh_CubicCUDA :
	public Atom_MeshCUDA
{

private:

public:

	//make this object by copying data from the Mesh holding this object
	Atom_Mesh_CubicCUDA(Atom_Mesh* paMesh) :
		Atom_MeshCUDA(paMesh)
	{}

	~Atom_Mesh_CubicCUDA() {}

	//----------------------------------- IMPORTANT CONTROL METHODS

	//call when a configuration change has occurred - some objects might need to be updated accordingly
	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage) { return BError(); }
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	//Take a Monte Carlo Metropolis step in this atomistic mesh
	cuBReal Iterate_MonteCarloCUDA_Classic(cuBReal mc_cone_angledeg, double target_acceptance_rate) { return 0.0; }

	//Take a constrained Monte Carlo Metropolis step in this atomistic mesh
	cuBReal Iterate_MonteCarloCUDA_Constrained(cuBReal mc_cone_angledeg, double target_acceptance_rate) { return 0.0; }

	//----------------------------------- OTHER IMPORTANT CONTROL METHODS

	//----------------------------------- VALUE GETTERS

	//get topological charge using formula Q = Integral(m.(dm/dx x dm/dy) dxdy) / 4PI
	cuBReal GetTopologicalCharge(cuRect rectangle) { return 0.0; }

	//compute topological charge density spatial dependence and have it available in auxVEC_cuBReal
	//Use formula Qdensity = m.(dm/dx x dm/dy) / 4PI
	void Compute_TopoChargeDensity(void) {}

	//----------------------------------- ODE METHODS IN MAGNETIC MESH : Atom_Mesh_CubicCUDA.cu

	//return average dm/dt in the given avRect (relative rect). Here m is the direction vector.
	cuReal3 Average_dmdt(cuBox avBox) { return DBL3(); }

	//return average m x dm/dt in the given avRect (relative rect). Here m is the direction vector.
	cuReal3 Average_mxdmdt(cuBox avBox) { return DBL3(); }
};

#endif
#endif