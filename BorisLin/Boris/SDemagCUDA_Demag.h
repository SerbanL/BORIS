#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SDEMAG

#include "ModulesCUDA.h"
#include "BorisCUDALib.h"
#include "EvalSpeedupCUDA.h"

class MeshBaseCUDA;
class MeshCUDA;
class Atom_MeshCUDA;
class ManagedMeshCUDA;
class SDemag_Demag;
class SDemagCUDA;

class SDemagMCUDA_Demag_single;

class SDemagCUDA_Demag :
	public ModulesCUDA,
	public EvalSpeedupCUDA
{
	friend SDemagCUDA;
	friend SDemagMCUDA_Demag_single;

private:

	MeshBaseCUDA *pMeshBaseCUDA = nullptr;

	//pointer to CUDA version of mesh object holding the effective field module holding this CUDA module (either micromagnetic or atomistic - only one will be not nullptr, so check)
	MeshCUDA * pMeshCUDA = nullptr;
	Atom_MeshCUDA * paMeshCUDA = nullptr;

	//pointer to cpu version of this module
	SDemag_Demag *pSDemag_Demag = nullptr;

	////////////////////////////////////////////////////

	//transfer values from M of this mesh to a mcuVEC with fixed number of cells -> use same meshing for all layers.
	mcu_VEC(cuReal3) transfer;

	//if displaying module Heff or energy, and mesh transfer is required, then use these (capture output field and energy, then do transfer)
	mcu_VEC(cuReal3) transfer_Module_Heff;
	mcu_VEC(cuBReal) transfer_Module_energy;

	//do transfer as M -> transfer -> convolution -> transfer -> Heff if true
	//if false then don't use the transfer VEC but can do M -> convolution -> Heff directly
	//this flag will be set false only if the convolution rect matches that of M and n_common matches the discretisation of M (i.e. in this case a mesh transfer would be pointless).
	bool do_transfer = true;

	//different meshes have different weights when contributing to the total energy density -> ratio of their non-empty volume to total non-empty volume
	mcu_val<cuBReal> energy_density_weight;

	////////////////////////////////////////////////////

	//one SDemagMCUDA_Demag_single object per GPU
	std::vector<SDemagMCUDA_Demag_single*> pDemagMCUDA;

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

private:

	//check if all pDemagMCUDA modules are initialized
	bool Submodules_Initialized(void);

	void set_SDemag_DemagCUDA_pointers(void);

	//this is called from SDemagCUDA so this mesh module can set convolution sizes
	BError Set_Convolution_Dimensions(cuBReal h_max, cuSZ3 n_common, cuINT3 pbc, std::vector<cuRect>& Rect_collection, int mesh_idx);

	//Get Ms0 value for mesh holding this module
	double Get_Ms0(void);

public:

	SDemagCUDA_Demag(MeshBaseCUDA* pMeshBaseCUDA_, SDemag_Demag *pSDemag_Demag_);
	~SDemagCUDA_Demag();

	//-------------------Abstract base class method implementations

	void Uninitialize(void);

	BError Initialize(void);

	BError UpdateConfiguration(UPDATECONFIG_ cfgMessage);
	void UpdateConfiguration_Values(UPDATECONFIG_ cfgMessage) {}

	void UpdateField(void);
	
	//-------------------Getters

	//add energy in this module to a running total
	void Add_Energy(mcu_val<cuBReal>& total_energy);

	//-------------------Setters
};

#else

class SDemagCUDA_Demag
{
};

#endif

#endif

