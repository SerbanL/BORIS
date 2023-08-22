#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_MELASTIC

#include "BorisCUDALib.h"
#include "ModulesCUDA.h"

class SMElasticCUDA;
class MElastic;
class MeshCUDA;
class Mesh;

class MElastic_PolicyBoundaryCUDA;
class MElastic_BoundaryCUDA;

class MElasticCUDA :
	public ModulesCUDA
{
	friend SMElasticCUDA;
	friend MElastic;

private:

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	MeshCUDA* pMeshCUDA;

	//pointer to cpu version of MeshCUDA
	Mesh *pMesh;

	//pointer to MElastic holder module
	MElastic* pMElastic;

private:

	//---------------------- FDTD scheme velocity-stress representation. Need separate components as they are staggered.

	//---- Main scheme

	//velocity x : mid x edges, size nx * (ny + 1) * (nz + 1)
	mcu_VEC_VC(cuBReal) vx;

	//velocity y : mid y edges, size (nx + 1) * ny * (nz + 1)
	mcu_VEC_VC(cuBReal) vy;

	//velocity z : mid z edges, size (nx + 1) * (ny + 1) * nz
	mcu_VEC_VC(cuBReal) vz;

	//diagonal stress components : vertices, size (nx + 1)*(ny + 1)*(nz + 1)
	mcu_VEC_VC(cuReal3) sdd;

	//off-diagonal stress sigma_xy : mid xy faces, size nx*ny*(nz + 1)
	mcu_VEC_VC(cuBReal) sxy;

	//off-diagonal stress sigma_xz : mid xz faces, size nx*(ny + 1)*nz
	mcu_VEC_VC(cuBReal) sxz;

	//off-diagonal stress sigma_yz : mid yz faces, size (nx + 1)*ny*nz
	mcu_VEC_VC(cuBReal) syz;

	//---- Additional discretization scheme needed for trigonal crystal system. These are velocity and stress values on rectangular frame at +hy/2, -hz/2 staggering w.r.t. main one.
	//positions below indicated for 1) main discretization frame, 2) additional staggered discretization frame

	//velocity x : 1) centers of cells, 2) mid x edges, size nx * ny * nz
	mcu_VEC_VC(cuBReal) vx2;

	//velocity y : 1) mid z edges 2) mid y edges, size (nx + 1) * (ny + 1) * nz
	mcu_VEC_VC(cuBReal) vy2;

	//velocity z : 1) mid y edges 2) mid z edges, size (nx + 1) * ny * (nz + 1)
	mcu_VEC_VC(cuBReal) vz2;

	//diagonal stress components : 1) mid yz faces 2) vertices, size (nx + 1) * ny * nz
	mcu_VEC_VC(cuReal3) sdd2;

	//off-diagonal stress sigma_xy : 1) mid xz faces  2) mid xy faces, size nx * (ny + 1) * nz
	mcu_VEC_VC(cuBReal) sxy2;

	//off-diagonal stress sigma_xz : 1) mid xy faces 2) mid xz faces, size nx * ny * (nz + 1)
	mcu_VEC_VC(cuBReal) sxz2;

	//off-diagonal stress sigma_yz : 1) vertices 2) mid yz faces, size (nx + 1) * (ny + 1) * (nz + 1)
	mcu_VEC_VC(cuBReal) syz2;

	//----------------------
	
	//corresponds to MElastic::external_stress_surfaces
	std::vector<mcu_obj<MElastic_BoundaryCUDA, MElastic_PolicyBoundaryCUDA>*> external_stress_surfaces;
	
	//same information as in external_stress_surfaces, but stored in a cu_arr so we can pass it whole into a CUDA kernel
	mcu_arr<MElastic_BoundaryCUDA> external_stress_surfaces_arr;
	
	//with cuda switched on this will hold the text equation object (TEquationCUDA manages GPU memory but is held in CPU memory itself, so cannot place it directly in MElastic_BoundaryCUDA)
	//vector size same as MElastic(CUDA)::external_stress_surfaces
	std::vector<mTEquationCUDA<cuBReal, cuBReal, cuBReal>*> Fext_equationCUDA;

	//----------------------
	
	//Strain using user equation, thus allowing simultaneous spatial (x, y, z), and stage time (t) dependence.
	//A number of constants are always present : mesh dimensions in m (Lx, Ly, Lz)
	//When text equation set, then elastodynamics solver is disabled.
	//diagonal
	mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal> Sd_equation;
	//off-diagonal
	mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal> Sod_equation;

	//----------------------

	//include thermal expansion? Include if thalpha (thermoelastic constant) is non-zero and heat module is enabled
	bool thermoelasticity_enabled = false;

	//save previous temperature values so we can compute dT / dt when including thermoelasticity
	mcu_VEC(cuBReal) Temp_previous;

	//if thermoelasticity is enabled, T_ambient will take same value as in the Heat module; refreshed at initialization.
	//NOTE : any changes to T_ambient in Heat module cause simulation to stop (CMD_AMBIENTTEMPERATURE or CMD_TEMPERATURE), so modules will go through Initialize() before simulation restart
	mcu_val<cuBReal> T_ambient;

	//----------------------

	//include magnetostriction? (mMEc constant not zero, and magnetic mesh)
	bool magnetostriction_enabled = false;

	//----------------------

	//disabled by setting magnetoelastic coefficient to zero (also disabled in non-magnetic meshes)
	bool melastic_field_disabled = false;

private:

	//----------------------------------------------- Auxiliary

	//Run-time auxiliary to set strain directly from user supplied text formulas
	void Set_Strain_From_Formula(void);

	void clear_Fext_equationCUDA(void);
	void make_Fext_equationCUDA(size_t size);

	void clear_external_stress_surfaces(void);
	void make_external_stress_surfaces(size_t size);

	//----------------------------------------------- Computational Helpers

	//-----Velocity

	//update velocity for dT time increment (also updating displacement)
	void Iterate_Elastic_Solver_Velocity(double dT);

	//update velocity on main discretization scheme
	void Iterate_Elastic_Solver_Velocity1(double dT);

	//update velocity on additional discretization scheme
	void Iterate_Elastic_Solver_Velocity2(double dT);

	//-----Stress

	//update stress for dT time increment
	void Iterate_Elastic_Solver_Stress(double dT, double magnetic_dT);

	//CRYSTAL_CUBIC
	void Iterate_Elastic_Solver_Stress_Cubic(double dT, double magnetic_dT);

	//CRYSTAL_TRIGONAL
	void Iterate_Elastic_Solver_Stress_Trigonal(double dT, double magnetic_dT);

	//-----Initial Stress

	//if thermoelasticity or magnetostriction is enabled, then initial stress must be set correctly
	void Set_Initial_Stress(void);

	//CRYSTAL_CUBIC
	void Set_Initial_Stress_Cubic(void);

	//CRYSTAL_TRIGONAL
	void Set_Initial_Stress_Trigonal(void);

	//-----MElastic Field

	//compute magnetoelastic effective field to use in magnetization equation. Accumulate energy.
	void Calculate_MElastic_Field(void);

	//CRYSTAL_CUBIC
	void Calculate_MElastic_Field_Cubic(void);

	//CRYSTAL_TRIGONAL
	void Calculate_MElastic_Field_Trigonal(void);

	//-----Strain

	//update strain from stress
	void Calculate_Strain(void);

	//-----Others

	//if thermoelasticity is enabled then save current temperature values in Temp_previous (called after elastic solver fully incremented by magnetic_dT)
	void Save_Current_Temperature(void);

	//---------------------------------------------- CMBND

	//-----Velocity

	void make_velocity_continuous(int axis, mCMBNDInfoCUDA& contact, MElasticCUDA* pMElastic_sec);

	//-----Stress

	void make_stress_continuous(int axis, mCMBNDInfoCUDA& contact, MElasticCUDA* pMElastic_sec);

public:

	MElasticCUDA(Mesh* pMesh_, MElastic* pMElastic_);
	~MElasticCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage);

	void UpdateField(void);

	//------------------- Configuration

	//reset stress-strain solver to initial values (zero velocity, displacement and stress)
	void Reset_ElSolver(void);

	//set diagonal and shear strain text equations
	BError Set_Sd_Equation(std::vector<std::vector< std::vector<EqComp::FSPEC> >> fspec);
	BError Set_Sod_Equation(std::vector<std::vector< std::vector<EqComp::FSPEC> >> fspec);
	//clear text equations
	void Clear_Sd_Sod_Equations(void);

	//make Fext_equationCUDA[external_stress_surfaces_index], where external_stress_surfaces_index is an index in external_stress_surfaces
	BError Set_Fext_equation(int external_stress_surfaces_index, std::vector<std::vector< std::vector<EqComp::FSPEC> >> fspec);

	//-------------------
	
	//copy all required mechanical VECs from their cpu versions
	BError copy_VECs_to_GPU(void);
};

#else

class MElasticCUDA
{
};

#endif

#endif