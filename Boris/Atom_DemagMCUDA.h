#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_DEMAG) && ATOMISTIC == 1

#include "ModulesCUDA.h"

#include "EvalSpeedupCUDA.h"

class Atom_MeshCUDA;
class Atom_DemagMCUDA_single;

class Atom_DemagMCUDA :
	public ModulesCUDA,
	public EvalSpeedupCUDA
{
	friend Atom_DemagMCUDA_single;

private:

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module
	Atom_MeshCUDA* paMeshCUDA;

	////////////////////////////////////////////////////

	//one Atom_DemagMCUDA_single object per GPU
	std::vector<Atom_DemagMCUDA_single*> pDemagMCUDA;

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

	//The demag field and magnetization computed separately at the demag macrocell size.
	//Hd has cellsize h_dm (but can be cleared so need to keep this info separate, above).
	mcu_VEC(cuReal3) M, Hd;

private:

	//check if all pDemagMCUDA modules are initialized
	bool Submodules_Initialized(void);

	//Set pointers in ManagedAtom_MeshCUDA so we can access them in device code. This is used by MonteCarlo algorithm.
	void set_Atom_DemagCUDA_pointers(void);

	//from M1 transfer to M, converting magnetic moments to magnetization
	void Transfer_Moments_to_Magnetization(void);

	//from Hd transfer calculated field by adding into Heff1
	void Transfer_Demag_Field(mcu_VEC(cuReal3)& H);

public:

	Atom_DemagMCUDA(Atom_MeshCUDA* paMeshCUDA_);
	~Atom_DemagMCUDA();

	//-------------------Abstract base class method implementations

	void Uninitialize(void) { initialized = false; }

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);
};

#else

class Atom_DemagMCUDA
{
};

#endif

#endif