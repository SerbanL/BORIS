#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#include "MeshBaseCUDA.h"

#include "Atom_MeshParamsCUDA.h"
#include "ManagedAtom_MeshPolicyCUDA.h"

class Atom_Mesh;

class ManagedAtom_DiffEqCubicCUDA;
class ManagedAtom_DiffEqPolicyCubicCUDA;

//Store Atom_Mesh quantities as cu_obj managed cuda VECs
class Atom_MeshCUDA :
	public MeshBaseCUDA,
	public Atom_MeshParamsCUDA,
	public MeshDisplayCUDA
{
private:

	//pointer to cpu version of this mesh (base) - need it in the destructor. 
	//Still keep references to some Mesh data members here as we cannot use pMesh in .cu files (cannot have BorisLib.h in those compilation units - real headache, will need to fix this at some point somehow: problem is the nvcc compiler throws errors due to C++14 code in BorisLib)
	Atom_Mesh *paMesh;

protected:

public:

	//This points to data in Atom_ZeemanCUDA if set (else nullptr) - managed by Atom_ZeemanCUDA
	//pHa is nullptr if Atom_ZeemanCUDA not active
	mcu_val<cuReal3>* pHa = nullptr;

	//Managed Mesh
	mcu_obj<ManagedAtom_MeshCUDA, ManagedAtom_MeshPolicyCUDA> cuaMesh;

	//-----Magnetic properties

	//Atomic moments in units of Bohr magneton using double floating point precision (first sub-lattice : used in cubic, bcc, fcc, hcp)
	mcu_VEC_VC(cuReal3) M1;

	//effective field (units of A/m : easier to integrate with micromagnetic meshes in a multiscale simulation this way) - sum total field of all the added modules (first sub-lattice : used in cubic, bcc, fcc, hcp)
	mcu_VEC(cuReal3) Heff1;

	//-----Demagnetizing field macrocell : compute demagnetizing field or macroscopic dipole-dipole interaction using this cellsize

	//number of cells (n.x, n.y, n.z)
	SZ3& n_dm;

	//cellsize; doesn't have to be an integer number of atomistic cells
	DBL3& h_dm;

	//-----Electric conduction properties (Electron charge and spin Transport)

	//In MeshBaseCUDA

	//-----Thermal conduction properties

	//In MeshBaseCUDA

	//-----Elastic properties

	//In MeshBaseCUDA

public:

	//------------------------CTOR/DTOR

	//make this object by copying data from the Mesh holding this object
	Atom_MeshCUDA(Atom_Mesh* paMesh);

	virtual ~Atom_MeshCUDA();

	//----------------------------------- OTHER IMPORTANT CONTROL METHODS

	//Check if mesh needs to be moved (using the MoveMesh method) - return amount of movement required (i.e. parameter to use when calling MoveMesh).
	cuBReal CheckMoveMesh(bool antisymmetric, double threshold) { return 0.0; }

	//----------------------------------- ALGORITHMS

	//called by Track_Shift_Algorithm when copy_values_thermalize call is required, since this needs to be implemented in a cu file
	virtual void Track_Shift_Algorithm_CopyThermalize(mcu_VEC_VC(cuReal3)& M_src, cuBox cells_box_dst, cuBox cells_box_src) = 0;

	//----------------------------------- DISPLAY-ASSOCIATED GET/SET METHODS

	PhysQ FetchOnScreenPhysicalQuantity(double detail_level, bool getBackground);

	//save the quantity currently displayed on screen in an ovf2 file using the specified format
	BError SaveOnScreenPhysicalQuantity(std::string fileName, std::string ovf2_dataType, MESHDISPLAY_ quantity);

	//extract profile from focused mesh, from currently display mesh quantity, but reading directly from the quantity
	//Displayed	mesh quantity can be scalar or a vector; pass in std::vector pointers, then check for nullptr to determine what type is displayed
	//if do_average = true then build average and don't return anything, else return just a single-shot profile. If read_average = true then simply read out the internally stored averaged profile by assigning to pointer.
	void GetPhysicalQuantityProfile(
		DBL3 start, DBL3 end, double step, DBL3 stencil, 
		std::vector<DBL3>*& pprofile_dbl3, std::vector<double>*& pprofile_dbl, 
		bool do_average, bool read_average, MESHDISPLAY_ quantity);

	//return average value for currently displayed mesh quantity in the given relative rectangle
	Any GetAverageDisplayedMeshValue(Rect rel_rect, MESHDISPLAY_ quantity);

	//copy auxVEC_cuBReal in GPU memory to displayVEC in CPU memory
	void copy_auxVEC_cuBReal(VEC<double>& displayVEC);

	//----------------------------------- ENABLED MESH PROPERTIES CHECKERS

	//magnetization dynamics computation enabled
	bool MComputation_Enabled(void);

	bool Magnetism_Enabled(void);

	//electrical conduction computation enabled
	bool EComputation_Enabled(void);

	//thermal conduction computation enabled
	bool TComputation_Enabled(void);

	//mechanical computation enabled
	bool MechComputation_Enabled(void);

	//check if interface conductance is enabled (for spin transport solver)
	bool GInterface_Enabled(void);

	virtual bool GetMeshExchangeCoupling(void) { return false; }

	bool Get_Kernel_Initialize_on_GPU(void);

	//----------------------------------- OTHER CALCULATION METHODS

	//get topological charge using formula Q = Integral(m.(dm/dx x dm/dy) dxdy) / 4PI
	virtual cuBReal GetTopologicalCharge(cuRect rectangle) = 0;

	//compute topological charge density spatial dependence and have it available in auxVEC_cuBReal
	//Use formula Qdensity = m.(dm/dx x dm/dy) / 4PI
	virtual void Compute_TopoChargeDensity(void) = 0;

	//return phase transition temperature (K) based on formula Tc = J*e*z/3kB
	virtual double Show_Transition_Temperature(void) = 0;

	//return saturation magnetization (A/m) based on formula Ms = mu_s*n/a^3
	virtual double Show_Ms(void) = 0;

	//return exchange stiffness (J/m) based on formula A = J*n/2a
	virtual double Show_A(void) = 0;

	//return uniaxial anisotropy constant (J/m^3) based on formula K = k*n/a^3
	virtual double Show_Ku(void) = 0;

	//----------------------------------- MESH SHAPE CONTROL

	//copy all meshes controlled using change_mesh_shape from cpu to gpu versions
	BError copy_shapes_from_cpu(void);

	//copy all meshes controlled using change_mesh_shape from gpu to cpu versions
	BError copy_shapes_to_cpu(void);

	//----------------------------------- OTHER CONTROL METHODS

	//-----------------------------------OBJECT GETTERS

	mcu_obj<ManagedAtom_DiffEq_CommonCUDA, ManagedAtom_DiffEqPolicy_CommonCUDA>& Get_ManagedAtom_DiffEq_CommonCUDA(void);

	mcu_obj<ManagedAtom_DiffEqCubicCUDA, ManagedAtom_DiffEqPolicyCubicCUDA>& Get_ManagedAtom_DiffEqCubicCUDA(void);

	std::vector<DBL4>& get_tensorial_anisotropy(void);
};

#endif
