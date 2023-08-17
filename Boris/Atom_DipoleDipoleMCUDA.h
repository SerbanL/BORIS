#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_ATOM_DIPOLEDIPOLE) && ATOMISTIC == 1

#include "ModulesCUDA.h"

class Atom_MeshCUDA;
class Atom_DipoleDipoleMCUDA_single;

class Atom_DipoleDipoleMCUDA :
	public ModulesCUDA
{
	friend Atom_DipoleDipoleMCUDA_single;

private:

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	Atom_MeshCUDA* paMeshCUDA;

	////////////////////////////////////////////////////

	//one Atom_DipoleDipoleMCUDA_single object per GPU
	std::vector<Atom_DipoleDipoleMCUDA_single*> pDipoleDipoleMCUDA;

	//transfer data before x-FFTs
	std::vector<std::vector<mGPU_Transfer<cuReal3>*>> M_Input_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> M_Input_transfer_half;

	std::vector<std::vector<mGPU_Transfer<cuBComplex>*>> xFFT_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> xFFT_Data_transfer_half;

	//transfer data before x-IFFTs
	std::vector<std::vector<mGPU_Transfer<cuBComplex>*>> xIFFT_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> xIFFT_Data_transfer_half;

	//transfer data after x-IFFTs
	std::vector<std::vector<mGPU_Transfer<cuReal3>*>> Out_Data_transfer;
	std::vector<std::vector<mGPU_Transfer<cuBHalf>*>> Out_Data_transfer_half;

	////////////////////////////////////////////////////

	//The dipole-dipole field and magnetization computed separately at the demag macrocell size.
	//Hd has cellsize h_dm (but can be cleared so need to keep this info separate, above).
	mcu_VEC(cuReal3) M, Hd;

	//use macrocell method, or compute dipole-dipole interaction at the atomic unit cell level (i.e. when h_dm == h)?
	bool using_macrocell = true;

	////////////////////////////////////////////////////
	//Evaluation speedup mode data

	//vec for demagnetizing field polynomial extrapolation
	mcu_VEC(cuReal3) Hdemag, Hdemag2, Hdemag3, Hdemag4, Hdemag5, Hdemag6;

	//times at which evaluations were done, used for extrapolation
	double time_demag1 = 0.0, time_demag2 = 0.0, time_demag3 = 0.0, time_demag4 = 0.0, time_demag5 = 0.0, time_demag6 = 0.0;

	int num_Hdemag_saved = 0;

	//-Nxx, -Nyy, -Nzz values at r = r0
	mcu_val<cuReal3> selfDemagCoeff;

private:

	//check if all pDipoleDipoleMCUDA modules are initialized
	bool Submodules_Initialized(void);

	//Set pointers in ManagedAtom_MeshCUDA so we can access them in device code. This is used by MonteCarlo algorithm.
	void set_Atom_DipoleDipoleCUDA_pointers(void);

	//convert value in energy to energy density by dividing by cellsize volume of V
	void Energy_to_EnergyDensity(mcu_VEC(cuReal3)& V);

	//from M1 transfer to M macrocell as a total spin
	void Transfer_Moments_to_Macrocell(void);

	//from Hd transfer calculated field by adding into Heff1
	void Transfer_DipoleDipole_Field(mcu_VEC(cuReal3)& H);
	
	void Atom_DipoleDipole_EvalSpeedup_SubSelf(mcu_VEC(cuReal3)& H);

	//Set Hd (macrocell mode)
	void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6);
	void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5);
	void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4);
	void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3);
	void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2);
	void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H);

	//Add into Heff1 directly
	void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6);
	void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5);
	void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4);
	void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3);
	void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2);
	void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H);

public:

	Atom_DipoleDipoleMCUDA(Atom_MeshCUDA* paMeshCUDA_);
	~Atom_DipoleDipoleMCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);
};

#else

class Atom_DipoleDipoleCUDA
{
};

#endif

#endif
