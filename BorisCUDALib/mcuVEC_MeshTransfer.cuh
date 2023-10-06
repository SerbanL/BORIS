#pragma once

#include "mcuVEC_MeshTransfer.h"
#include "launchers.h"

//------------------------------------------------------------------- AUXILIARY

template <typename VType>
__global__ void mcuTransfer_zero_mesh_out_kernel(int& mesh_out_num, cuVEC<VType>*& mesh_out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int idxMesh = 0; idxMesh < mesh_out_num; idxMesh++) {

		cuVEC<VType>& cuvec_out = mesh_out[idxMesh];

		if (idx < cuvec_out.linear_size()) {

			cuvec_out[idx] = VType();
		}
	}
}

template void mcuTransfer<float>::zero_output_meshes(mcuTransfer_Info& transfer_info);
template void mcuTransfer<double>::zero_output_meshes(mcuTransfer_Info& transfer_info);

template void mcuTransfer<cuFLT3>::zero_output_meshes(mcuTransfer_Info& transfer_info);
template void mcuTransfer<cuDBL3>::zero_output_meshes(mcuTransfer_Info& transfer_info);

//clear values in mesh_out
template <typename VType>
__host__ void mcuTransfer<VType>::zero_output_meshes(mcuTransfer_Info& transfer_info)
{
	mcuTransfer_zero_mesh_out_kernel <<< (transfer_info.max_mesh_out_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (mesh_out_num, mesh_out);
}

template <typename VType>
__global__ void mcuTransfer_zero_mesh_out_duplicated_kernel(int& mesh_out_num, cuVEC<VType>*& mesh_out1, cuVEC<VType>*& mesh_out2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int idxMesh = 0; idxMesh < mesh_out_num; idxMesh++) {

		cuVEC<VType>& cuvec_out1 = mesh_out1[idxMesh];

		if (idx < cuvec_out1.linear_size()) {

			cuvec_out1[idx] = VType();

			cuVEC<VType>& cuvec_out2 = mesh_out2[idxMesh];

			if (cuvec_out2.linear_size()) {

				cuvec_out2[idx] = VType();
			}
		}
	}
}

template void mcuTransfer<float>::zero_output_duplicated_meshes(mcuTransfer_Info& transfer_info);
template void mcuTransfer<double>::zero_output_duplicated_meshes(mcuTransfer_Info& transfer_info);

template void mcuTransfer<cuFLT3>::zero_output_duplicated_meshes(mcuTransfer_Info& transfer_info);
template void mcuTransfer<cuDBL3>::zero_output_duplicated_meshes(mcuTransfer_Info& transfer_info);

//clear values in mesh_out and mesh_out2
template <typename VType>
__host__ void mcuTransfer<VType>::zero_output_duplicated_meshes(mcuTransfer_Info& transfer_info)
{
	mcuTransfer_zero_mesh_out_duplicated_kernel <<< (transfer_info.max_mesh_out_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> (mesh_out_num, mesh_out, mesh_out2);
}

//------------------------------------------------------------------- MESH TRANSFER IN

//transfer_in_type as follows:
//0:
//SINGLE INPUT, SINGLE OUTPUT
//1:
//AVERAGED INPUTS
//2:
//MULTIPLIED INPUTS

template <typename VType, typename VECType>
__device__ void transfer_in_kernel_BlockReduction(
	size_t transfer_in_info_index_offset, size_t transfer_in_info_index_size,
	VECType& sMesh_subVEC, cuVEC<VType>*& mesh_in, cuVEC<VType>*& mesh_in2, cuVEC<cuBReal>*& mesh_in2_real,
	cuINT2*& transfer_in_info_index, cuPair<cuINT4, cuBReal>*& transfer_in_info, 
	VType* pshared_memory, 
	int transfer_in_type,
	unsigned int NUMREDUCTIONS, unsigned int BLOCKSIZE, unsigned int BLOCKOFFSET)
{
	//block index : number of blocks equals number of unique destination cells (for NUMREDUCTIONS = 1, otherwise we have multiple reductions packed in same block)
	int bidx = blockIdx.x * NUMREDUCTIONS + threadIdx.x % NUMREDUCTIONS;
	//thread index : index in the current block for number of contributions to destination cell
	int tidx = threadIdx.x / NUMREDUCTIONS;
	int tidx_offset = tidx + BLOCKOFFSET;

	if (bidx < transfer_in_info_index_size) {

		//index in transfer_in_info for first destination cell contribution
		int start_idx = transfer_in_info_index[transfer_in_info_index_offset + bidx].i;
		//...and number of contributions to it
		int num_contributions = transfer_in_info_index[transfer_in_info_index_offset + bidx].j;

		//load contribution into shared memory in this block
		if (tidx_offset < num_contributions) {

			//i - mesh index, j - mesh cell index, k - supermesh (this mesh) cell index, l - device index for mesh
			cuINT4 full_index = transfer_in_info[start_idx + tidx_offset].first;

			//weight to apply to external mesh values
			cuBReal weight = transfer_in_info[start_idx + tidx_offset].second;

			//obtain weighted value from external mesh

			//first get the external mesh
			mcuVEC_Managed<cuVEC<VType>, VType>& cuvec_in = mesh_in[full_index.i].mcuvec();

			//now get its weighted value
			VType weighted_value = VType();

			if (transfer_in_type == 0) {

				//SINGLE INPUT, SINGLE OUTPUT
				weighted_value = cuvec_in[cuINT2(full_index.l, full_index.j)] * weight;
			}
			else if (transfer_in_type == 1) {

				//AVERAGED INPUTS
				mcuVEC_Managed<cuVEC<VType>, VType>& cuvec_in2 = mesh_in2[full_index.i].mcuvec();
				if (cuvec_in2.linear_size()) weighted_value = (cuvec_in[cuINT2(full_index.l, full_index.j)] + cuvec_in2[cuINT2(full_index.l, full_index.j)]) * weight / 2;
				else weighted_value = cuvec_in[cuINT2(full_index.l, full_index.j)] * weight;
			}
			else if (transfer_in_type == 2) {

				//MULTIPLIED INPUTS
				mcuVEC_Managed<cuVEC<cuBReal>, cuBReal>& cuvec_in2_real = mesh_in2_real[full_index.i].mcuvec();
				if (cuvec_in2_real.linear_size()) weighted_value = cuvec_in[cuINT2(full_index.l, full_index.j)] * cuvec_in2_real[cuINT2(full_index.l, full_index.j)] * weight;
				else weighted_value = cuvec_in[cuINT2(full_index.l, full_index.j)] * weight;
			}

			pshared_memory[threadIdx.x] = weighted_value;
		}
		else pshared_memory[threadIdx.x] = VType();

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

			//only thread 0 in this block sets value in destination cell (block), so no data race conditions will occur
			sMesh_subVEC[transfer_in_info[start_idx].first.k] += pshared_memory[threadIdx.x];
		}
	}
}

template <unsigned int N, unsigned int NUMREDUCTIONS, typename VType, typename VECType>
__global__ void transfer_in_kernel_BlockSize_N(
	size_t transfer_in_info_index_offset, size_t transfer_in_info_index_size,
	VECType& sMesh_subVEC, cuVEC<VType>*& mesh_in, cuVEC<VType>*& mesh_in2, cuVEC<cuBReal>*& mesh_in2_real,
	cuINT2*& transfer_in_info_index, cuPair<cuINT4, cuBReal>*& transfer_in_info, 
	int transfer_in_type,
	unsigned int BLOCKOFFSET)
{
	__shared__ VType shared_memory[N * NUMREDUCTIONS];

	transfer_in_kernel_BlockReduction(
		transfer_in_info_index_offset, transfer_in_info_index_size,
		sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real,
		transfer_in_info_index, transfer_in_info, 
		shared_memory, 
		transfer_in_type,
		NUMREDUCTIONS, N, BLOCKOFFSET);
}

template <typename VType, typename VECType>
__global__ void transfer_in_kernel_BlockSize_1(
	size_t transfer_in_info_index_offset, size_t transfer_in_info_index_size,
	VECType& sMesh_subVEC, cuVEC<VType>*& mesh_in, cuVEC<VType>*& mesh_in2, cuVEC<cuBReal>*& mesh_in2_real,
	cuINT2*& transfer_in_info_index, cuPair<cuINT4, cuBReal>*& transfer_in_info, 
	int transfer_in_type)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < transfer_in_info_index_size) {

		//i - mesh index, j - mesh cell index, k - supermesh (this mesh) cell index, l - device index for mesh
		cuINT4 full_index = transfer_in_info[transfer_in_info_index[transfer_in_info_index_offset + idx].i].first;

		//weight to apply to external mesh values
		cuBReal weight = transfer_in_info[transfer_in_info_index[transfer_in_info_index_offset + idx].i].second;

		//obtain weighted value from external mesh

		//first get the external mesh
		mcuVEC_Managed<cuVEC<VType>, VType>& cuvec_in = mesh_in[full_index.i].mcuvec();

		//now get its weighted value
		VType weighted_value = VType();

		if (transfer_in_type == 0) {

			//SINGLE INPUT, SINGLE OUTPUT
			weighted_value = cuvec_in[cuINT2(full_index.l, full_index.j)] * weight;
		}
		else if (transfer_in_type == 1) {

			//AVERAGED INPUTS
			mcuVEC_Managed<cuVEC<VType>, VType>& cuvec_in2 = mesh_in2[full_index.i].mcuvec();
			if (cuvec_in2.linear_size()) weighted_value = (cuvec_in[cuINT2(full_index.l, full_index.j)] + cuvec_in2[cuINT2(full_index.l, full_index.j)]) * weight / 2;
			else weighted_value = cuvec_in[cuINT2(full_index.l, full_index.j)] * weight;
		}
		else if (transfer_in_type == 2) {

			//MULTIPLIED INPUTS
			mcuVEC_Managed<cuVEC<cuBReal>, cuBReal>& cuvec_in2_real = mesh_in2_real[full_index.i].mcuvec();
			if (cuvec_in2_real.linear_size()) weighted_value = cuvec_in[cuINT2(full_index.l, full_index.j)] * cuvec_in2_real[cuINT2(full_index.l, full_index.j)] * weight;
			else weighted_value = cuvec_in[cuINT2(full_index.l, full_index.j)] * weight;
		}

		//now add reduced value
		sMesh_subVEC[full_index.k] += weighted_value;
	}
}

template void mcuTransfer<float>::transfer_in(cuVEC<float>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_in_type);
template void mcuTransfer<double>::transfer_in(cuVEC<double>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_in_type);

template void mcuTransfer<cuFLT3>::transfer_in(cuVEC<cuFLT3>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_in_type);
template void mcuTransfer<cuDBL3>::transfer_in(cuVEC<cuDBL3>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_in_type);

template void mcuTransfer<float>::transfer_in(cuVEC_VC<float>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_in_type);
template void mcuTransfer<double>::transfer_in(cuVEC_VC<double>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_in_type);

template void mcuTransfer<cuFLT3>::transfer_in(cuVEC_VC<cuFLT3>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_in_type);
template void mcuTransfer<cuDBL3>::transfer_in(cuVEC_VC<cuDBL3>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_in_type);

//transfer values from mesh_in meshes using transfer_info into sMesh_quantity which has given size
template <typename VType>
template <typename VECType>
__host__ void mcuTransfer<VType>::transfer_in(VECType& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_in_type)
{
	//different block sizes must be handled with separate cuda launches
	for (int idx = 0; idx < transfer_info.transfer_in_info_index_blocksizes.size(); idx++) {
		
		size_t transfer_in_info_index_offset = transfer_info.transfer_in_info_index_blocksizes[idx].i;
		size_t transfer_in_info_index_size = transfer_info.transfer_in_info_index_blocksizes[idx].k;

		int blocksize = transfer_info.transfer_in_info_index_blocksizes[idx].j;

		unsigned int BLOCKOFFSET = 0;
		do {

			//number of blocks : number of unique destination cells
			//number of threads : maximum number of contributions to each destination cell, if this is smaller or equal to maximum number of threads allowed
			//if number of contributions exceeds maximum number of allowed threads, then we need another reduction, but starting at next offset (multiple of maximum number of threads)
			//repeat until fully reduced

			switch (blocksize) {

			case 0:
				break;

			case 1:
				//special case no reduction required
				transfer_in_kernel_BlockSize_1 <<< (transfer_in_info_index_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(transfer_in_info_index_offset, transfer_in_info_index_size, 
					sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real, 
					transfer_in_info_index, transfer_in_info, transfer_in_type);
				BLOCKOFFSET += 1;
				break;
			case 2:
				//it is very inefficient to use a blocksize less than MIN_CUDATHREADS (warp size), which is 32
				//for this reason block sizes less than 32 should be handled together in the same cuda block
				//e.g. 2 requires 16 separate reductions within a block size of 32, etc.
				transfer_in_kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 2, VType, VECType> <<< (transfer_in_info_index_size * 2 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(transfer_in_info_index_offset, transfer_in_info_index_size, 
					sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real, 
					transfer_in_info_index, transfer_in_info, transfer_in_type, 0);
				BLOCKOFFSET += 2;
				break;
			case 4:
				transfer_in_kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 4, VType, VECType> <<< (transfer_in_info_index_size * 4 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(transfer_in_info_index_offset, transfer_in_info_index_size, 
					sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real, 
					transfer_in_info_index, transfer_in_info, transfer_in_type, 0);
				BLOCKOFFSET += 4;
				break;
			case 8:
				transfer_in_kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 8, VType, VECType> <<< (transfer_in_info_index_size * 8 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(transfer_in_info_index_offset, transfer_in_info_index_size, 
					sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real, 
					transfer_in_info_index, transfer_in_info, transfer_in_type, 0);
				BLOCKOFFSET += 8;
				break;
			case 16:
				transfer_in_kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 16, VType, VECType> <<< (transfer_in_info_index_size * 16 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(transfer_in_info_index_offset, transfer_in_info_index_size, 
					sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real, 
					transfer_in_info_index, transfer_in_info, transfer_in_type, 0);
				BLOCKOFFSET += 16;
				break;
			case 32:
				transfer_in_kernel_BlockSize_N<32, 1, VType, VECType> <<< transfer_in_info_index_size, 32 >>>
					(transfer_in_info_index_offset, transfer_in_info_index_size, 
					sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real, 
					transfer_in_info_index, transfer_in_info, transfer_in_type, 0);
				BLOCKOFFSET += 32;
				break;
			case 64:
				transfer_in_kernel_BlockSize_N<64, 1, VType, VECType> <<< transfer_in_info_index_size, 64 >>>
					(transfer_in_info_index_offset, transfer_in_info_index_size, 
					sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real, 
					transfer_in_info_index, transfer_in_info, transfer_in_type, 0);
				BLOCKOFFSET += 64;
				break;
			case 128:
				transfer_in_kernel_BlockSize_N<128, 1, VType, VECType> <<< transfer_in_info_index_size, 128 >>>
					(transfer_in_info_index_offset, transfer_in_info_index_size, 
					sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real, 
					transfer_in_info_index, transfer_in_info, transfer_in_type, 0);
				BLOCKOFFSET += 128;
				break;
			case 256:
				transfer_in_kernel_BlockSize_N<256, 1, VType, VECType> <<< transfer_in_info_index_size, 256 >>>
					(transfer_in_info_index_offset, transfer_in_info_index_size, 
					sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real, 
					transfer_in_info_index, transfer_in_info, transfer_in_type, 0);
				BLOCKOFFSET += 256;
				break;
			default: //default case : blocksize exceeds maximum number of allowed threads per block
			case 512:
				transfer_in_kernel_BlockSize_N<512, 1, VType, VECType> <<< transfer_in_info_index_size, 512 >>>
					(transfer_in_info_index_offset, transfer_in_info_index_size, 
					sMesh_subVEC, mesh_in, mesh_in2, mesh_in2_real, 
					transfer_in_info_index, transfer_in_info, transfer_in_type, BLOCKOFFSET);
				BLOCKOFFSET += 512;
				break;
			}

		} while (BLOCKOFFSET < blocksize);
	}
}

//------------------------------------------------------------------- MESH TRANSFER OUT

template <typename VType, typename VECType>
__device__ void transfer_out_kernel_BlockReduction(
	size_t transfer_out_info_index_offset, size_t transfer_out_info_index_size,
	VECType& sMesh_subVEC, cuVEC<VType>*& mesh_out, cuVEC<VType>*& mesh_out2,
	cuINT2*& transfer_out_info_index, cuPair<cuINT4, cuBReal>*& transfer_out_info,
	VType* pshared_memory,
	int transfer_out_type,
	unsigned int NUMREDUCTIONS, unsigned int BLOCKSIZE, unsigned int BLOCKOFFSET)
{
	//block index : number of blocks equals number of unique destination cells (for NUMREDUCTIONS = 1, otherwise we have multiple reductions packed in same block)
	int bidx = blockIdx.x * NUMREDUCTIONS + threadIdx.x % NUMREDUCTIONS;
	//thread index : index in the current block for number of contributions to destination cell
	int tidx = threadIdx.x / NUMREDUCTIONS;
	int tidx_offset = tidx + BLOCKOFFSET;

	if (bidx < transfer_out_info_index_size) {

		//index in transfer_in_info for first destination cell contribution
		int start_idx = transfer_out_info_index[transfer_out_info_index_offset + bidx].i;
		//...and number of contributions to it
		int num_contributions = transfer_out_info_index[transfer_out_info_index_offset + bidx].j;

		//load contribution into shared memory in this block
		if (tidx_offset < num_contributions) {

			//i - mesh index, j - mesh cell index, k - supermesh (this mesh) cell index, l - device index for mesh
			cuINT4 full_index = transfer_out_info[start_idx + tidx_offset].first;

			//weight to apply to supermesh values
			cuBReal weight = transfer_out_info[start_idx + tidx_offset].second;

			//obtain weighted value from supermesh
			VType weighted_value = sMesh_subVEC[full_index.k] * weight;

			pshared_memory[threadIdx.x] = weighted_value;
		}
		else pshared_memory[threadIdx.x] = VType();

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

			//only thread 0 in this block sets value in destination cell (block), so no data race conditions will occur

			//first get the external mesh
			cuINT4 full_index = transfer_out_info[start_idx].first;
			mcuVEC_Managed<cuVEC<VType>, VType>& cuvec_out = mesh_out[full_index.i].mcuvec();

			//now add reduced value
			
			//SINGLE INPUT, SINGLE OUTPUT
			cuvec_out[cuINT2(full_index.l, full_index.j)] += pshared_memory[threadIdx.x];

			//DUPLICATED OUTPUT
			if (transfer_out_type == 1) {

				mcuVEC_Managed<cuVEC<VType>, VType>& cuvec_out2 = mesh_out2[full_index.i].mcuvec();
				if (cuvec_out2.linear_size()) cuvec_out2[cuINT2(full_index.l, full_index.j)] += pshared_memory[threadIdx.x];
			}
		}
	}
}

template <unsigned int N, unsigned int NUMREDUCTIONS, typename VType, typename VECType>
__global__ void transfer_out_kernel_BlockSize_N(
	size_t transfer_out_info_index_offset, size_t transfer_out_info_index_size,
	VECType& sMesh_subVEC, cuVEC<VType>*& mesh_out, cuVEC<VType>*& mesh_out2,
	cuINT2*& transfer_out_info_index, cuPair<cuINT4, cuBReal>*& transfer_out_info,
	int transfer_out_type,
	unsigned int BLOCKOFFSET)
{
	__shared__ VType shared_memory[N * NUMREDUCTIONS];

	transfer_out_kernel_BlockReduction(
		transfer_out_info_index_offset, transfer_out_info_index_size,
		sMesh_subVEC, mesh_out, mesh_out2,
		transfer_out_info_index, transfer_out_info, 
		shared_memory, 
		transfer_out_type,
		NUMREDUCTIONS, N, BLOCKOFFSET);
}

template <typename VType, typename VECType>
__global__ void transfer_out_kernel_BlockSize_1(
	size_t transfer_out_info_index_offset, size_t transfer_out_info_index_size,
	VECType& sMesh_subVEC, cuVEC<VType>*& mesh_out, cuVEC<VType>*& mesh_out2,
	cuINT2*& transfer_out_info_index, cuPair<cuINT4, cuBReal>*& transfer_out_info,
	int transfer_out_type)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < transfer_out_info_index_size) {

		//i - mesh index, j - mesh cell index, k - supermesh (this mesh) cell index, l - device index for mesh
		cuINT4 full_index = transfer_out_info[transfer_out_info_index[transfer_out_info_index_offset + idx].i].first;

		//weight to apply to supermesh values
		cuBReal weight = transfer_out_info[transfer_out_info_index[transfer_out_info_index_offset + idx].i].second;

		//obtain weighted value from supermesh
		VType weighted_value = sMesh_subVEC[full_index.k] * weight;

		//get the external mesh
		mcuVEC_Managed<cuVEC<VType>, VType>& cuvec_out = mesh_out[full_index.i].mcuvec();

		//now add reduced value
		//SINGLE INPUT, SINGLE OUTPUT
		cuvec_out[cuINT2(full_index.l, full_index.j)] += weighted_value;

		//DUPLICATED OUTPUT
		if (transfer_out_type == 1) {

			mcuVEC_Managed<cuVEC<VType>, VType>& cuvec_out2 = mesh_out2[full_index.i].mcuvec();
			if (cuvec_out2.linear_size()) cuvec_out2[cuINT2(full_index.l, full_index.j)] += weighted_value;
		}
	}
}

template void mcuTransfer<float>::transfer_out(cuVEC<float>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_out_type);
template void mcuTransfer<double>::transfer_out(cuVEC<double>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_out_type);

template void mcuTransfer<cuFLT3>::transfer_out(cuVEC<cuFLT3>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_out_type);
template void mcuTransfer<cuDBL3>::transfer_out(cuVEC<cuDBL3>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_out_type);

template void mcuTransfer<float>::transfer_out(cuVEC_VC<float>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_out_type);
template void mcuTransfer<double>::transfer_out(cuVEC_VC<double>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_out_type);

template void mcuTransfer<cuFLT3>::transfer_out(cuVEC_VC<cuFLT3>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_out_type);
template void mcuTransfer<cuDBL3>::transfer_out(cuVEC_VC<cuDBL3>& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_out_type);

//transfer values from mesh_in meshes using transfer_info into sMesh_quantity which has given size
template <typename VType>
template <typename VECType>
__host__ void mcuTransfer<VType>::transfer_out(VECType& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_out_type)
{
	//different block sizes must be handled with separate cuda launches
	for (int idx = 0; idx < transfer_info.transfer_out_info_index_blocksizes.size(); idx++) {

		size_t transfer_out_info_index_offset = transfer_info.transfer_out_info_index_blocksizes[idx].i;
		size_t transfer_out_info_index_size = transfer_info.transfer_out_info_index_blocksizes[idx].k;

		int blocksize = transfer_info.transfer_out_info_index_blocksizes[idx].j;

		//different block sizes must be handled with separate cuda launches
		unsigned int BLOCKOFFSET = 0;
		do {

			//number of blocks : number of unique destination cells
			//number of threads : maximum number of contributions to each destination cell, if this is smaller or equal to maximum number of threads allowed
			//if number of contributions exceeds maximum number of allowed threads, then we need another reduction, but starting at next offset (multiple of maximum number of threads)
			//repeat until fully reduced

			switch (blocksize) {

			case 0:
				break;

			case 1:
				//special case no reduction required
				transfer_out_kernel_BlockSize_1 <<< (transfer_out_info_index_size + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
					(transfer_out_info_index_offset, transfer_out_info_index_size, sMesh_subVEC, mesh_out, mesh_out2, transfer_out_info_index, transfer_out_info, transfer_out_type);
				BLOCKOFFSET += 1;
				break;
			case 2:
				//it is very inefficient to use a blocksize less than MIN_CUDATHREADS (warp size), which is 32
				//for this reason block sizes less than 32 should be handled together in the same cuda block
				//e.g. 2 requires 16 separate reductions within a block size of 32, etc.
				transfer_out_kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 2, VType, VECType> <<< (transfer_out_info_index_size * 2 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(transfer_out_info_index_offset, transfer_out_info_index_size, sMesh_subVEC, mesh_out, mesh_out2, transfer_out_info_index, transfer_out_info, transfer_out_type, 0);
				BLOCKOFFSET += 2;
				break;
			case 4:
				transfer_out_kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 4, VType, VECType> <<< (transfer_out_info_index_size * 4 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(transfer_out_info_index_offset, transfer_out_info_index_size, sMesh_subVEC, mesh_out, mesh_out2, transfer_out_info_index, transfer_out_info, transfer_out_type, 0);
				BLOCKOFFSET += 4;
				break;
			case 8:
				transfer_out_kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 8, VType, VECType> <<< (transfer_out_info_index_size * 8 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(transfer_out_info_index_offset, transfer_out_info_index_size, sMesh_subVEC, mesh_out, mesh_out2, transfer_out_info_index, transfer_out_info, transfer_out_type, 0);
				BLOCKOFFSET += 8;
				break;
			case 16:
				transfer_out_kernel_BlockSize_N<MIN_CUDATHREADS, MIN_CUDATHREADS / 16, VType, VECType> <<< (transfer_out_info_index_size * 16 + MIN_CUDATHREADS) / MIN_CUDATHREADS, MIN_CUDATHREADS >>>
					(transfer_out_info_index_offset, transfer_out_info_index_size, sMesh_subVEC, mesh_out, mesh_out2, transfer_out_info_index, transfer_out_info, transfer_out_type, 0);
				BLOCKOFFSET += 16;
				break;
			case 32:
				transfer_out_kernel_BlockSize_N<32, 1, VType, VECType> <<< transfer_out_info_index_size, 32 >>>
					(transfer_out_info_index_offset, transfer_out_info_index_size, sMesh_subVEC, mesh_out, mesh_out2, transfer_out_info_index, transfer_out_info, transfer_out_type, 0);
				BLOCKOFFSET += 32;
				break;
			case 64:
				transfer_out_kernel_BlockSize_N<64, 1, VType, VECType> <<< transfer_out_info_index_size, 64 >>>
					(transfer_out_info_index_offset, transfer_out_info_index_size, sMesh_subVEC, mesh_out, mesh_out2, transfer_out_info_index, transfer_out_info, transfer_out_type, 0);
				BLOCKOFFSET += 64;
				break;
			case 128:
				transfer_out_kernel_BlockSize_N<128, 1, VType, VECType> <<< transfer_out_info_index_size, 128 >>>
					(transfer_out_info_index_offset, transfer_out_info_index_size, sMesh_subVEC, mesh_out, mesh_out2, transfer_out_info_index, transfer_out_info, transfer_out_type, 0);
				BLOCKOFFSET += 128;
				break;
			case 256:
				transfer_out_kernel_BlockSize_N<256, 1, VType, VECType> <<< transfer_out_info_index_size, 256 >>>
					(transfer_out_info_index_offset, transfer_out_info_index_size, sMesh_subVEC, mesh_out, mesh_out2, transfer_out_info_index, transfer_out_info, transfer_out_type, 0);
				BLOCKOFFSET += 256;
				break;
			default: //default case : block_size_cpu exceeds maximum number of allowed threads per block
			case 512:
				transfer_out_kernel_BlockSize_N<512, 1, VType, VECType> <<< transfer_out_info_index_size, 512 >>>
					(transfer_out_info_index_offset, transfer_out_info_index_size, sMesh_subVEC, mesh_out, mesh_out2, transfer_out_info_index, transfer_out_info, transfer_out_type, BLOCKOFFSET);
				BLOCKOFFSET += 512;
				break;
			}

		} while (BLOCKOFFSET < blocksize);
	}
}