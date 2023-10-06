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
	cuVEC<cuReal3>& M, cuVEC_VC<cuReal3>& M1, 
	cuReal3* pshared_memory, 
	int device_index,
	unsigned int NUMREDUCTIONS, unsigned int BLOCKSIZE, unsigned int BLOCKOFFSET)
{
	cuSZ3& n = M1.n;
	cuSZ3& n_m = M.n;
	//number of atomistic spins inside micromagnetic cell in each dimension
	int rx = n.x / n_m.x;
	int ry = n.y / n_m.y;
	int rz = n.z / n_m.z;

	//block index : also the micromagnetic cell index (number of blocks equals number of micromagnetic cells) 
	//(for NUMREDUCTIONS = 1, otherwise we have multiple reductions packed in same block)
	int bidx = blockIdx.x * NUMREDUCTIONS + threadIdx.x % NUMREDUCTIONS;
	//ijk in micromagnetic space
	cuINT3 ijk_m = cuINT3(bidx % n_m.x, (bidx / n_m.x) % n_m.y, bidx / (n_m.x * n_m.y));

	//thread index : index in atomistic spins inside a micromagnetic cell (the current block)
	int tidx = threadIdx.x / NUMREDUCTIONS;
	int tidx_offset = tidx + BLOCKOFFSET;
	//ijk in atomistic space (blocksize is greater or equal to number of atomistic spins in this block, so ijk index may exceed dimensions - check done below when loading into shared memory)
	cuINT3 ijk = cuINT3(tidx_offset % rx + ijk_m.i * rx, (tidx_offset / rx) % ry + ijk_m.j * ry, tidx_offset / (rx * ry) + ijk_m.k * rz);

	//load atomistic spin into shared memory in this block
	if (tidx_offset < rx * ry * rz) {

		//macrocell is contained in current cuda device, but not all spins inside macrocell may be on same device
		if (ijk.i < n.x) pshared_memory[threadIdx.x] = M1[ijk.i + ijk.j * n.x + ijk.k * n.x * n.y];
		else {

			//must obtain spin from next device (x partitioning used)
			mcuVEC_Managed<cuVEC_VC<cuReal3>, cuReal3>& M1_man = M1.mcuvec();
			if (device_index + 1 < M1_man.get_num_devices()) {

				cuSZ3 dn = M1_man.device_n(device_index + 1);
				pshared_memory[threadIdx.x] = M1_man[cuINT2(device_index + 1, ijk.i - n.x + ijk.j * dn.x + ijk.k * dn.x * dn.y)];
			}
			else pshared_memory[threadIdx.x] = cuReal3();
		}
	}
	else pshared_memory[threadIdx.x] = cuReal3();

	//synchronize before starting reduction in block
	__syncthreads();

	for (unsigned s = BLOCKSIZE / 2; s >= NUMREDUCTIONS; s >>= 1) {

		if (tidx < s) {

			//summing reduction
			pshared_memory[threadIdx.x] += pshared_memory[threadIdx.x + s];
		}

		__syncthreads();
	}

	if (tidx == 0) {

		//only thread 0 in this block sets value in micromagnetic cell (block), so no data race conditions will occur
		//BLOCKOFFSET 0 means this is the first reduction
		if (BLOCKOFFSET == 0) M[ijk_m.i + ijk_m.j * n_m.x + ijk_m.k * n_m.x * n_m.y] = pshared_memory[threadIdx.x];
		//BLOCKOFFSET > 0 means the block reduction could not be completed in one pass due to reaching maximum allowed number of threads per block
		//thus this is an additional pass starting at BLOCKOFFSET, and we must add into M instead
		else M[ijk_m.i + ijk_m.j * n_m.x + ijk_m.k * n_m.x * n_m.y] += pshared_memory[threadIdx.x];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//these will be compiled depending on which template N values are given in Atom_DipoleDipoleMCUDA::Transfer_Moments_to_Macrocell, but do not exceed a maximum value (see below), and keep to powers of 2 
template <unsigned int N, unsigned int NUMREDUCTIONS>
__global__ void Transfer_Moments_to_Macrocell_Kernel_BlockSize_N(
	cuVEC<cuReal3>& M, cuVEC_VC<cuReal3>& M1, 
	int device_index,
	unsigned int BLOCKOFFSET)
{
	__shared__ cuReal3 shared_memory[N * NUMREDUCTIONS];

	Transfer_Moments_to_Macrocell_BlockReduction(
		M, M1, 
		shared_memory, 
		device_index,
		NUMREDUCTIONS, N, BLOCKOFFSET);
}

__global__ void Transfer_Moments_to_Macrocell_Kernel_BlockSize_1(cuVEC<cuReal3>& M, cuVEC_VC<cuReal3>& M1)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < M.linear_size()) M[idx] = M1[idx];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//from M1 transfer to M macrocell as a total spin
void Atom_DipoleDipoleMCUDA::Transfer_Moments_to_Macrocell(void)
{
	//number of threads in each block : number of atomistic cells in each micromagnetic cell
	int num_atomspins_per_cell = (paMeshCUDA->n / paMeshCUDA->n_dm).dim();

	auto find_pow2 = [](int remaining_spins) -> unsigned int
	{
		//find smallest power of 2 greater or equal to num_threads (this will be the block size)
		unsigned int pow2 = 1;
		unsigned int n = remaining_spins;
		while (n != 0 && pow2 != remaining_spins) {

			n >>= 1;
			pow2 <<= 1;
		}

		return pow2;
	};

	unsigned int blocksize = find_pow2(num_atomspins_per_cell);
	unsigned int BLOCKOFFSET = 0;
	do {

		//number of blocks : number of micromagnetic cells (on each device)
		//number of threads : number of spins in a micromagnetic cell, if this is smaller or equal to maximum number of threads allowed
		//if number of spins exceeds maximum number of allowed cells, then we need another reduction, but starting at next offset (multiple of maximum number of threads)
		//repeat until fully reduced

		switch (blocksize) {

		case 1:
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_1 <<< (M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU));
			}
			break;
		//it is very inefficient to use a blocksize less than MIN_CUDATHREADS (warp size), which is 32
		//for this reason block sizes less than 32 should be handled together in the same cuda block
		//e.g. 2 requires 16 separate reductions within a block size of 32, etc.
		case 2:
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 2> <<< (M.device_size(mGPU) * 2 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), mGPU, 0);
			}
			BLOCKOFFSET += 2;
			break;
		case 4:
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 4> <<< (M.device_size(mGPU) * 4 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), mGPU, 0);
			}
			BLOCKOFFSET += 4;
			break;
		case 8:
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 8> <<< (M.device_size(mGPU) * 8 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), mGPU, 0);
			}
			BLOCKOFFSET += 8;
			break;
		case 16:
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 16> <<< (M.device_size(mGPU) * 16 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), mGPU, 0);
			}
			BLOCKOFFSET += 16;
			break;
		case 32:
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<32, 1> <<< M.device_size(mGPU), 32 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), mGPU, 0);
			}
			BLOCKOFFSET += 32;
			break;
		case 64:
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<64, 1> <<< M.device_size(mGPU), 64 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), mGPU, 0);
			}
			BLOCKOFFSET += 64;
			break;
		case 128:
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<128, 1> <<< M.device_size(mGPU), 128 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), mGPU, 0);
			}
			BLOCKOFFSET += 128;
			break;
		case 256:
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<256, 1> <<< M.device_size(mGPU), 256 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), mGPU, 0);
			}
			BLOCKOFFSET += 256;
			break;
		default: //default case : blocksize exceeds maximum number of allowed threads per block
		case 512:
			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
				Transfer_Moments_to_Macrocell_Kernel_BlockSize_N<512, 1> <<< M.device_size(mGPU), 512 >>>
					(M.get_deviceobject(mGPU), paMeshCUDA->M1.get_deviceobject(mGPU), mGPU, BLOCKOFFSET);
			}
			BLOCKOFFSET += 512;
			break;
		}

	} while (BLOCKOFFSET < blocksize);
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

#endif

#endif