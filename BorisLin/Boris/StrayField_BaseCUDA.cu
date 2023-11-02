#include "StrayField_BaseCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_STRAYFIELD

#include "DipoleTFuncCUDA.h"
#include "Mesh_DipoleCUDA.h"

__global__ void CalculateStrayField_kernel(cuVEC<cuReal3>& strayField, ManagedMeshCUDA** pMdipoles, size_t num_dipoles)
{
	int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (cell_idx < strayField.linear_size()) {

		//origin of super-mesh ferromagnetic rectangle - strayField rectangle is the supermesh rectangle
		cuReal3 meshOrigin = strayField.rect.get_s();

		//sizes and cellsize for super-mesh
		cuINT3 n = strayField.n;
		cuReal3 h = strayField.h;

		//reset so we can add contributions into this cell
		strayField[cell_idx] = cuReal3();

		//go through all dipole meshes to add contributions from each at cell_idx
		for (int idx = 0; idx < num_dipoles; idx++) {

			//rectangle of dipole mesh - Mdipoles[idx] is the M cuVEC in this dipole mesh
			cuRect dipoleRect = pMdipoles[idx]->pM->rect;

			//dipole rectangular prism dimensions
			cuReal3 abc = dipoleRect.size();

			//centre position of dipole mesh
			cuReal3 dipole_centre = (dipoleRect.get_s() + dipoleRect.get_e()) / 2;

			//calculate stray field contribution from current dipole this super-mesh cell
			cuINT3 ijk = cuINT3(cell_idx % n.x, (cell_idx / n.x) % n.y, cell_idx / (n.x * n.y));

			//distance from dipole_centre to current supermesh-cell
			cuReal3 xyz = dipole_centre - cuReal3((ijk.i + 0.5) * h.x + meshOrigin.x, (ijk.j + 0.5) * h.y + meshOrigin.y, (ijk.k + 0.5) * h.z + meshOrigin.z);

			cuBReal p11, p22, p33, p12, p13, p23;

			p11 = Nxx(xyz, abc);
			p22 = Nyy(xyz, abc);
			p33 = Nzz(xyz, abc);
			p12 = Nxy(xyz, abc);
			p13 = Nxz(xyz, abc);
			p23 = Nyz(xyz, abc);

			//the Mdipole for this mesh (already correctly set for the mesh temperature)
			cuReal3 Mdipole = (*pMdipoles[idx]->pM)[0];

			strayField[cell_idx].x += -(p11 * Mdipole.x + p12 * Mdipole.y + p13 * Mdipole.z);
			strayField[cell_idx].y += -(p12 * Mdipole.x + p22 * Mdipole.y - p23 * Mdipole.z);
			strayField[cell_idx].z += -(p13 * Mdipole.x - p23 * Mdipole.y + p33 * Mdipole.z);
		}
	}
}

void StrayField_BaseCUDA::CalculateStrayField(void)
{
	if (!dipoles.size()) strayField.set(cuReal3(0.0));
	else {
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			CalculateStrayField_kernel <<< (strayField.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
				(strayField.get_deviceobject(mGPU), pmanaged_dipoles.get_array(mGPU), pmanaged_dipoles.size(mGPU));
		}
	}
}

#endif

#endif