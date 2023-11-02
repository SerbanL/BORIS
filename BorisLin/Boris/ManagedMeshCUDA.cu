#include "ManagedMeshCUDA.h"

#if COMPILECUDA == 1 && MONTE_CARLO == 1

////////////////////////////////////
//
// Ferromagnetic

__global__ void Set_FM_MCFuncs_kernel(int* cuModules, int numModules, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0) {

		cuMesh.num_FM_MCFuncs = numModules;
		cuMesh.num_AFM_MCFuncs = 0;
	}

	if (idx < numModules) {

		switch (cuModules[idx]) {

		case MOD_DEMAG_N:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_DemagNCUDA;
			break;

		case MOD_DEMAG:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_DemagCUDA;
			break;

		case MOD_SDEMAG_DEMAG:
			//same method as for MOD_DEMAG, but the effective field pointer now points to SDemag_DemagCUDA module effective field
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_DemagCUDA;
			break;

		case MOD_STRAYFIELD_MESH:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_StrayField_MeshCUDA;
			break;

		case MOD_EXCHANGE:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_ExchangeCUDA;
			break;

		case MOD_DMEXCHANGE:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_DMExchangeCUDA;
			break;

		case MOD_IDMEXCHANGE:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_iDMExchangeCUDA;
			break;

		case MOD_VIDMEXCHANGE:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_viDMExchangeCUDA;
			break;

		case MOD_SURFEXCHANGE:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_SurfExchangeCUDA;
			break;

		case MOD_ZEEMAN:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_ZeemanCUDA;
			break;

		case MOD_MOPTICAL:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_MOpticalCUDA;
			break;

		case MOD_ANIUNI:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_AnisotropyCUDA;
			break;

		case MOD_ANICUBI:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_AnisotropyCubiCUDA;
			break;

		case MOD_ANIBI:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_AnisotropyBiaxialCUDA;
			break;

		case MOD_ANITENS:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_AnisotropyTensorialCUDA;
			break;

		case MOD_ROUGHNESS:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_RoughnessCUDA;
			break;

		case MOD_MELASTIC:
			cuMesh.pFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_FM_MElasticCUDA;
			break;

		default:
			//before calling function check for nullptr
			cuMesh.pFM_MCFuncs[idx] = nullptr;
			break;
		}
	}
}

//setup function pointers in pFM_MCFuncs depending on configured modules
void ManagedMeshCUDA::Set_FM_MCFuncs(cu_arr<int>& cuModules)
{
	if (cuModules.size()) {

		Set_FM_MCFuncs_kernel <<< 1, cuModules.size() >>> (cuModules, cuModules.size(), *this);
	}
}

////////////////////////////////////
//
// Antiferromagnetic

__global__ void Set_AFM_MCFuncs_kernel(int* cuModules, int numModules, ManagedMeshCUDA& cuMesh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0) {

		cuMesh.num_AFM_MCFuncs = numModules;
		cuMesh.num_FM_MCFuncs = 0;
	}

	if (idx < numModules) {

		switch (cuModules[idx]) {

		case MOD_DEMAG_N:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_DemagNCUDA;
			break;

		case MOD_DEMAG:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_DemagCUDA;
			break;

		case MOD_SDEMAG_DEMAG:
			//same method as for MOD_DEMAG, but the effective field pointer now points to SDemag_DemagCUDA module effective field
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_DemagCUDA;
			break;

		case MOD_STRAYFIELD_MESH:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_StrayField_MeshCUDA;
			break;

		case MOD_EXCHANGE:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_ExchangeCUDA;
			break;

		case MOD_DMEXCHANGE:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_DMExchangeCUDA;
			break;

		case MOD_IDMEXCHANGE:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_iDMExchangeCUDA;
			break;

		case MOD_VIDMEXCHANGE:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_viDMExchangeCUDA;
			break;

		case MOD_SURFEXCHANGE:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_SurfExchangeCUDA;
			break;

		case MOD_ZEEMAN:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_ZeemanCUDA;
			break;

		case MOD_MOPTICAL:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_MOpticalCUDA;
			break;

		case MOD_ANIUNI:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_AnisotropyCUDA;
			break;

		case MOD_ANICUBI:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_AnisotropyCubiCUDA;
			break;

		case MOD_ANIBI:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_AnisotropyBiaxialCUDA;
			break;

		case MOD_ANITENS:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_AnisotropyTensorialCUDA;
			break;

		case MOD_ROUGHNESS:
			cuMesh.pAFM_MCFuncs[idx] = &ManagedMeshCUDA::Get_EnergyChange_AFM_RoughnessCUDA;
			break;

		case MOD_MELASTIC:
			//Not available for AFM
			cuMesh.pAFM_MCFuncs[idx] = nullptr;
			break;

		default:
			//before calling function check for nullptr
			cuMesh.pAFM_MCFuncs[idx] = nullptr;
			break;
		}
	}
}

//setup function pointers in pAFM_MCFuncs depending on configured modules
void ManagedMeshCUDA::Set_AFM_MCFuncs(cu_arr<int>& cuModules)
{
	if (cuModules.size()) {

		Set_AFM_MCFuncs_kernel << < 1, cuModules.size() >> > (cuModules, cuModules.size(), *this);
	}
}

#endif