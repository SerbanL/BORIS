#include "stdafx.h"
#include "TransportCUDA_Poisson_Spin_S.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "MeshCUDA.h"
#include "Atom_MeshCUDA.h"
#include "TransportBaseCUDA.h"

//for modules held in micromagnetic meshes
BError TransportCUDA_Spin_S_Funcs::set_pointers(MeshCUDA* pMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int device_idx)
{
	BError error(__FUNCTION__);

	if (set_gpu_value(pcuMesh, pMeshCUDA->cuMesh.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(pPoisson_Spin_V, pTransportBaseCUDA->poisson_Spin_V.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(pdelsq_S_fixed, pTransportBaseCUDA->delsq_S_fixed.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(stsolve, pTransportBaseCUDA->Get_STSolveType()) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	return error;
}

//for modules held in atomistic meshes
BError TransportCUDA_Spin_S_Funcs::set_pointers(Atom_MeshCUDA* paMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int device_idx)
{
	BError error(__FUNCTION__);

	if (set_gpu_value(pcuaMesh, paMeshCUDA->cuaMesh.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(pPoisson_Spin_V, pTransportBaseCUDA->poisson_Spin_V.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(pdelsq_S_fixed, pTransportBaseCUDA->delsq_S_fixed.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(stsolve, pTransportBaseCUDA->Get_STSolveType()) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	return error;
}

#endif

#endif