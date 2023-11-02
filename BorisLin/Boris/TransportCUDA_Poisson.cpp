#include "stdafx.h"
#include "TransportCUDA_Poisson.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "MeshCUDA.h"
#include "Atom_MeshCUDA.h"

//for modules held in micromagnetic meshes
BError TransportCUDA_V_Funcs::set_pointers(MeshCUDA* pMeshCUDA, int device_idx)
{
	BError error(__FUNCTION__);

	if (set_gpu_value(pcuMesh, pMeshCUDA->cuMesh.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	return error;
}

//for modules held in atomistic meshes
BError TransportCUDA_V_Funcs::set_pointers(Atom_MeshCUDA* paMeshCUDA, int device_idx)
{
	BError error(__FUNCTION__);

	if (set_gpu_value(pcuaMesh, paMeshCUDA->cuaMesh.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	return error;
}

#endif

#endif