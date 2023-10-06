#pragma once

#include "Boris_Enums_Defs.h"
#if COMPILECUDA == 1

#ifdef MESH_COMPILATION_ATOM_CUBIC

#include "BorisCUDALib.h"

#include "ErrorHandler.h"

#include "ManagedAtom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"

class Thermalize_FM_to_Atom
{
private:

	//the atomistic mesh
	ManagedAtom_MeshCUDA* pcuaMesh;

public:

	__host__ void construct_cu_obj(void) {}
	__host__ void destruct_cu_obj(void) {}

	BError set_pointers(Atom_MeshCUDA* pMeshCUDA, int device_idx);

	__device__ cuReal3 thermalize_func(cuReal3 Mval, int idx_src, int idx_dst, cuBorisRand<>& prng)
	{
		cuBReal Mval_norm = Mval.norm();
		if (cuIsZ(Mval_norm)) return cuReal3();

		cuReal3 Mval_dir = Mval / Mval_norm;

		cuBReal Ms0 = pcuaMesh->pmu_s->get0() * (MUB / pcuaMesh->pM1->h.dim());

		//need the normalized average over spins in this cell to be mz (the resultant Ms0 value in this atomistic mesh should match that in the micromagnetic mesh)
		cuBReal mz = Mval_norm / Ms0;
		//enforce min and max values possible
		if (mz < 1.0 / 6) mz = 1.0 / 6;
		if (mz > 1.0) mz = 1.0;
		//maximum polar angle around Mval direction
		cuBReal theta_max = sqrt(10 - sqrt(100 + 120 * (mz - 1)));
		if (theta_max > PI) theta_max = PI;

		cuBReal theta = prng.rand() * theta_max;
		cuBReal phi = prng.rand() * 2 * PI;

		cuBReal mu_s_val = *pcuaMesh->pmu_s;
		pcuaMesh->update_parameters_mcoarse(idx_dst, *pcuaMesh->pmu_s, mu_s_val);

		cuReal3 spin_val = mu_s_val * relrotate_polar(Mval_dir, theta, phi);

		return spin_val;
	}
};

#endif

#endif