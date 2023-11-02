#include "stdafx.h"
#include "Atom_Mesh_CubicCUDA_Thermalize.h"

#if COMPILECUDA == 1

#ifdef MESH_COMPILATION_ATOM_CUBIC

#include "Atom_MeshCUDA.h"

BError Thermalize_FM_to_Atom::set_pointers(Atom_MeshCUDA* paMeshCUDA, int device_idx)
{
	BError error(__FUNCTION__);

	if (set_gpu_value(pcuaMesh, paMeshCUDA->cuaMesh.get_managed_object(device_idx)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	return error;
}

#endif

#endif