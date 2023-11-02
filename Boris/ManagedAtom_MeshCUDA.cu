#include "ManagedAtom_MeshCUDA.h"

#if COMPILECUDA == 1 && ATOMISTIC == 1 && MONTE_CARLO == 1

////////////////////////////////////
//
// Simple Cubic

__global__ void Set_SC_MCFuncs_kernel(int* cuModules, int numModules, ManagedAtom_MeshCUDA& cuaMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0) {

		cuaMesh.num_SC_MCFuncs = numModules;
	}

	if (idx < numModules) {

		switch (cuModules[idx]) {

		case MOD_DEMAG_N:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_DemagNCUDA;
			break;

		case MOD_DEMAG:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_DemagCUDA;
			break;

		case MOD_ATOM_DIPOLEDIPOLE:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_DipoleDipoleCUDA;
			break;

		case MOD_SDEMAG_DEMAG:
			//same method as for MOD_DEMAG, but the effective field pointer now points to SDemag_DemagCUDA module effective field
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_DemagCUDA;
			break;

		case MOD_STRAYFIELD_MESH:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_StrayField_AtomMeshCUDA;
			break;

		case MOD_EXCHANGE:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_ExchangeCUDA;
			break;

		case MOD_DMEXCHANGE:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_DMExchangeCUDA;
			break;

		case MOD_IDMEXCHANGE:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_iDMExchangeCUDA;
			break;

		case MOD_VIDMEXCHANGE:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_viDMExchangeCUDA;
			break;

		case MOD_SURFEXCHANGE:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_SurfExchangeCUDA;
			break;

		case MOD_ZEEMAN:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_ZeemanCUDA;
			break;

		case MOD_MOPTICAL:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_MOpticalCUDA;
			break;

		case MOD_ANIUNI:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_AnisotropyCUDA;
			break;

		case MOD_ANICUBI:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_AnisotropyCubiCUDA;
			break;

		case MOD_ANIBI:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_AnisotropyBiaxialCUDA;
			break;

		case MOD_ANITENS:
			cuaMesh.pSC_MCFuncs[idx] = &ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_AnisotropyTensorialCUDA;
			break;

		case MOD_ROUGHNESS:
			//not applicable for atomistic meshes
			cuaMesh.pSC_MCFuncs[idx] = nullptr;
			break;

		case MOD_MELASTIC:
			//not defined in atomistic meshes
			cuaMesh.pSC_MCFuncs[idx] = nullptr;
			break;

		default:
			//before calling function check for nullptr
			cuaMesh.pSC_MCFuncs[idx] = nullptr;
			break;
		}
	}
}

//setup function pointers in pSC_MCFuncs depending on configured modules
void ManagedAtom_MeshCUDA::Set_SC_MCFuncs(cu_arr<int>& cuModules)
{
	if (cuModules.size()) {

		Set_SC_MCFuncs_kernel <<< 1, cuModules.size() >>> (cuModules, cuModules.size(), *this);
	}
}

#endif