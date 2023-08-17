#include "Atom_DipoleDipoleMCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_ATOM_DIPOLEDIPOLE) && ATOMISTIC == 1

#include "BorisCUDALib.cuh"
#include "Atom_MeshCUDA.h"

//----------------------- Initialization

__global__ void set_Atom_DipoleDipoleCUDA_pointers_kernel(
	ManagedAtom_MeshCUDA& cuaMesh, cuVEC<cuReal3>& Module_Heff)
{
	if (threadIdx.x == 0) cuaMesh.pAtom_Demag_Heff = &Module_Heff;
}

void Atom_DipoleDipoleMCUDA::set_Atom_DipoleDipoleCUDA_pointers(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		set_Atom_DipoleDipoleCUDA_pointers_kernel <<< 1, CUDATHREADS >>>
			(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), Module_Heff.get_deviceobject(mGPU));
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Inefficient method (don't use)
/*
__global__ void Transfer_Moments_to_Magnetization_Kernel(
	cuVEC<cuReal3>& M, cuVEC_VC<cuReal3>& M1)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < M1.linear_size()) {

		cuReal3 relpos = M1.cellidx_to_position(idx);
		int idxM = M.position_to_cellidx(relpos);

		atomicAdd(&M[idxM], M1[idx] * MUB / M.h.dim());
	}
}
*/

//Block reduction routine in a block of given BLOCKSIZE (number of threads), from M1 to M. Start at BLOCKOFFSET for reading data in.
__device__ void Transfer_Moments_to_Macrocell_BlockReduction(
	cuVEC<cuReal3>& M, cuVEC_VC<cuReal3>& M1, cuReal3* pshared_memory, unsigned BLOCKSIZE, unsigned int BLOCKOFFSET)
{
	cuSZ3& n = M1.n;
	cuSZ3& n_m = M.n;
	//number of atomistic spins inside micromagnetic cell in each dimension
	int rx = n.x / n_m.x;
	int ry = n.y / n_m.y;
	int rz = n.z / n_m.z;

	//block index : also the micromagnetic cell index (number of blocks equals number of micromagnetic cells)
	int bidx = blockIdx.x;
	//ijk in micromagnetic space
	cuINT3 ijk_m = cuINT3(bidx % n_m.x, (bidx / n_m.x) % n_m.y, bidx / (n_m.x * n_m.y));

	//thread index : index in atomistic spins inside a micromagnetic cell (the current block)
	int tidx = threadIdx.x;
	int tidx_offset = tidx + BLOCKOFFSET;
	//ijk in atomistic space (blocksize is greater or equal to number of atomistic spins in this block, so ijk index may exceed dimensions - check done below when loading into shared memory)
	cuINT3 ijk = cuINT3(tidx_offset % rx + ijk_m.i * rx, (tidx_offset / rx) % ry + ijk_m.j * ry, tidx_offset / (rx * ry) + ijk_m.k * rz);

	//load atomistic spin into shared memory in this block
	if (tidx_offset < rx* ry* rz) pshared_memory[tidx] = M1[ijk.i + ijk.j * n.x + ijk.k * n.x * n.y];
	else pshared_memory[tidx] = 0;

	//synchronize before starting reduction in block
	__syncthreads();

	for (unsigned s = BLOCKSIZE / 2; s > 0; s >>= 1) {

		if (tidx < s) {

			//summing reduction
			pshared_memory[tidx] += pshared_memory[tidx + s];
		}

		__syncthreads();
	}

	if (tidx == 0) {

		//only thread 0 in this block sets value in micromagnetic cell (block), so no data race conditions will occur
		//BLOCKOFFSET 0 means this is the first reduction
		if (BLOCKOFFSET == 0) M[ijk_m.i + ijk_m.j * n_m.x + ijk_m.k * n_m.x * n_m.y] = pshared_memory[0];
		//BLOCKOFFSET > 0 means the block reduction could not be completed in one pass due to reaching maximum allowed number of threads per block
		//thus this is an additional pass starting at BLOCKOFFSET, and we must add into M instead
		else M[ijk_m.i + ijk_m.j * n_m.x + ijk_m.k * n_m.x * n_m.y] += pshared_memory[0];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//these will be compiled depending on which template N values are given in Atom_DipoleDipoleMCUDA::Transfer_Moments_to_Macrocell, but do not exceed a maximum value (see below), and keep to powers of 2 
template <unsigned int N> __global__ void Transfer_Moments_to_Magnetization_Kernel_BlockSize_N(cuVEC<cuReal3>& M, cuVEC_VC<cuReal3>& M1, unsigned int BLOCKOFFSET);

template <unsigned int N>
__global__ void Transfer_Moments_to_Macrocell_Kernel_BlockSize_N(
	cuVEC<cuReal3>& M, cuVEC_VC<cuReal3>& M1, unsigned int BLOCKOFFSET)
{
	__shared__ cuReal3 shared_memory[N];
	Transfer_Moments_to_Macrocell_BlockReduction(M, M1, shared_memory, N, BLOCKOFFSET);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//from M1 transfer to M macrocell as a total spin
void Atom_DipoleDipoleMCUDA::Transfer_Moments_to_Macrocell(void)
{
	//number of threads in each block : number of atomistic cells in each micromagnetic cell
	int num_atomspins_per_cell = (paMeshCUDA->n / paMeshCUDA->n_dm).dim();

	auto find_pow2 = [](int remaining_spins) -> unsigned int
	{
		//above 512 threads it's more efficient to launch reduction multiple times
		constexpr unsigned int MAX_THREADS = 512;

		//find smallest power of 2 greater or equal to num_threads (this will be the block size)
		unsigned int pow2 = 1;
		unsigned int n = remaining_spins;
		while (n != 0 && pow2 != remaining_spins && pow2 < MAX_THREADS) {

			n >>= 1;
			pow2 <<= 1;
		}

		return pow2;
	};

	unsigned int BLOCKOFFSET = 0;
	do {

		unsigned int pow2 = find_pow2(num_atomspins_per_cell - BLOCKOFFSET);

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			//number of blocks : number of micromagnetic cells (on each device)
			//number of threads : number of spins in a micromagnetic cell, if this is smaller or equal to maximum number of threads allowed
			//if number of spins exceeds maximum number of allowed cells, then we need another reduction, but starting at next offset (multiple of maximum number of threads)
			//repeat until fully reduced

			switch (pow2) {

			case 1:
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<1> <<< M.device_size(mGPU), 1 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), BLOCKOFFSET);
				break;
			case 2:
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<2> <<< M.device_size(mGPU), 2 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), BLOCKOFFSET);
				break;
			case 4:
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<4> <<< M.device_size(mGPU), 4 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), BLOCKOFFSET);
				break;
			case 8:
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<8> <<< M.device_size(mGPU), 8 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), BLOCKOFFSET);
				break;
			case 16:
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<16> <<< M.device_size(mGPU), 16 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), BLOCKOFFSET);
				break;
			case 32:
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<32> <<< M.device_size(mGPU), 32 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), BLOCKOFFSET);
				break;
			case 64:
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<64> <<< M.device_size(mGPU), 64 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), BLOCKOFFSET);
				break;
			case 128:
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<128> <<< M.device_size(mGPU), 128 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), BLOCKOFFSET);
				break;
			case 256:
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<256> <<< M.device_size(mGPU), 256 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), BLOCKOFFSET);
				break;
			case 512:
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<512> <<< M.device_size(mGPU), 512 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), BLOCKOFFSET);
				break;
			}
		}

		BLOCKOFFSET += pow2;
	} while (BLOCKOFFSET < num_atomspins_per_cell);
}

__global__ void Transfer_DipoleDipole_Field_Kernel(
	cuVEC<cuReal3>& Hd, cuVEC<cuReal3>& Heff1)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Heff1.linear_size()) {

		cuSZ3& n = Heff1.n;
		cuSZ3& n_dm = Hd.n;
		//it must be checked before starting computations that the dm cell has an integer number of atomistic cells in each dimension
		cuINT3 ijk = cuINT3((idx % n.x) / (n.x / n_dm.x), ((idx / n.x) % n.y) / (n.y / n_dm.y), (idx / (n.x * n.y)) / (n.z / n_dm.z));

		Heff1[idx] += Hd[ijk.i + ijk.j * n_dm.x + ijk.k * n_dm.x * n_dm.y];
	}
}

//from Hd transfer calculated field by adding into Heff1
void Atom_DipoleDipoleMCUDA::Transfer_DipoleDipole_Field(mcu_VEC(cuReal3)& H)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Transfer_DipoleDipole_Field_Kernel <<< (paMeshCUDA->Heff1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), paMeshCUDA->Heff1.get_deviceobject(mGPU));
	}
}

//----------------------- Auxiliary

__global__ void Energy_to_EnergyDensity_Kernel(cuBReal& energy, cuVEC<cuReal3>& V)
{
	if (threadIdx.x == 0) energy *= (cuBReal)MUB / V.h.dim();
}

//convert value in energy to energy density by dividing by cellsize volume of V
void Atom_DipoleDipoleMCUDA::Energy_to_EnergyDensity(mcu_VEC(cuReal3)& V)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Energy_to_EnergyDensity_Kernel <<< 1, CUDATHREADS >>> (energy(mGPU), V.get_deviceobject(mGPU));
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//----------------------- KERNELS

__global__ void Atom_DipoleDipole_EvalSpeedup_SubSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] -= (selfDemagCoeff & M[idx]);
	}
}

//QUINTIC
__global__ void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5, cuVEC<cuReal3>& Hdemag6,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + Hdemag6[idx] * a6 + (selfDemagCoeff & M[idx]);
	}
}

__global__ void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5, cuVEC<cuReal3>& Hdemag6,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + Hdemag6[idx] * a6 + (selfDemagCoeff & M[idx]);
	}
}

//QUARTIC
__global__ void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + (selfDemagCoeff & M[idx]);
	}
}

__global__ void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4, cuVEC<cuReal3>& Hdemag5,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + Hdemag5[idx] * a5 + (selfDemagCoeff & M[idx]);
	}
}

//CUBIC
__global__ void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + (selfDemagCoeff & M[idx]);
	}
}

__global__ void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3, cuVEC<cuReal3>& Hdemag4,
	cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + Hdemag4[idx] * a4 + (selfDemagCoeff & M[idx]);
	}
}

//QUADRATIC
__global__ void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3,
	cuBReal a1, cuBReal a2, cuBReal a3,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + (selfDemagCoeff & M[idx]);
	}
}

__global__ void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2, cuVEC<cuReal3>& Hdemag3,
	cuBReal a1, cuBReal a2, cuBReal a3,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + Hdemag3[idx] * a3 + (selfDemagCoeff & M[idx]);
	}
}

//LINEAR
__global__ void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2,
	cuBReal a1, cuBReal a2,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] * a1 + Hdemag2[idx] * a2 + (selfDemagCoeff & M[idx]);
	}
}

__global__ void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag, cuVEC<cuReal3>& Hdemag2,
	cuBReal a1, cuBReal a2,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] += Hdemag[idx] * a1 + Hdemag2[idx] * a2 + (selfDemagCoeff & M[idx]);
	}
}

//STEP
__global__ void Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] = Hdemag[idx] + (selfDemagCoeff & M[idx]);
	}
}

__global__ void Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel(
	cuVEC<cuReal3>& H,
	cuVEC<cuReal3>& Hdemag,
	cuVEC<cuReal3>& M, cuReal3& selfDemagCoeff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < H.linear_size()) {

		H[idx] += Hdemag[idx] + (selfDemagCoeff & M[idx]);
	}
}

//----------------------- LAUNCHERS

//Macrocell mode: subtract self contribution from calculated field
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_SubSelf(mcu_VEC(cuReal3)& H)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_SubSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), (using_macrocell ? M.get_deviceobject(mGPU) : paMeshCUDA->M1.get_deviceobject(mGPU)), selfDemagCoeff(mGPU));
	}
}

//Macrocell mode, QUINTIC: extrapolate field and add self contribution
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), Hdemag6.get_deviceobject(mGPU), 
			a1, a2, a3, a4, a5, a6, 
			M.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}


//Macrocell mode, QUARTIC: extrapolate field and add self contribution
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), 
			a1, a2, a3, a4, a5, 
			M.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//Macrocell mode, CUBIC: extrapolate field and add self contribution
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), 
			a1, a2, a3, a4, 
			M.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//Macrocell mode, QUADRATIC: extrapolate field and add self contribution
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), 
			a1, a2, a3, 
			M.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//Macrocell mode, LINEAR: extrapolate field and add self contribution
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), 
			a1, a2, 
			M.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//Macrocell mode, STEP: extrapolate field and add self contribution
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf(mcu_VEC(cuReal3)& H)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_SetExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), 
			M.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUINTIC: extrapolate field (Add into Heff1 directly)
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5, cuBReal a6)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), Hdemag6.get_deviceobject(mGPU), 
			a1, a2, a3, a4, a5, a6,
			paMeshCUDA->M1.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUARTIC: extrapolate field (Add into Heff1 directly)
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4, cuBReal a5)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), Hdemag5.get_deviceobject(mGPU), 
			a1, a2, a3, a4, a5,
			paMeshCUDA->M1.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//CUBIC: extrapolate field (Add into Heff1 directly)
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3, cuBReal a4)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), Hdemag4.get_deviceobject(mGPU), 
			a1, a2, a3, a4,
			paMeshCUDA->M1.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//QUADRATIC: extrapolate field (Add into Heff1 directly)
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2, cuBReal a3)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), Hdemag3.get_deviceobject(mGPU), 
			a1, a2, a3,
			paMeshCUDA->M1.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//LINEAR: extrapolate field (Add into Heff1 directly)
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H, cuBReal a1, cuBReal a2)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU), Hdemag2.get_deviceobject(mGPU), 
			a1, a2,
			paMeshCUDA->M1.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

//STEP: extrapolate field (Add into Heff1 directly)
void Atom_DipoleDipoleMCUDA::Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf(mcu_VEC(cuReal3)& H)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Atom_DipoleDipole_EvalSpeedup_AddExtrapField_AddSelf_Kernel <<< (H.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(H.get_deviceobject(mGPU), 
			Hdemag.get_deviceobject(mGPU),
			paMeshCUDA->M1.get_deviceobject(mGPU), selfDemagCoeff(mGPU));
	}
}

#endif

#endif