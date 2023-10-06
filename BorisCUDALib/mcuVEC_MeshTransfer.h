#pragma once

#include "mcuVEC.h"

#include <algorithm>
#include <omp.h>

struct mcuTransfer_Info
{
	//vector of different block sizes and respective starting index in mcuTransfer::transfer_in_info_index for each different block size
	//NOTE mcuTransfer::transfer_in_info_index is sorted by blocksize
	//i : starting index in mcuTransfer::transfer_in_info_index for a set of transfers with same block size, j : block size (adjusted to power of 2)
	//k : number of indexes with same block size
	std::vector<cuINT3> transfer_in_info_index_blocksizes;

	//as above but for mcuTransfer::transfer_out_info_index
	std::vector<cuINT3> transfer_out_info_index_blocksizes;

	//maximum number of cells from mcuTransfer::mesh_out cuVECs
	size_t max_mesh_out_size;

	void clear(void)
	{
		transfer_in_info_index_blocksizes.clear();
		transfer_out_info_index_blocksizes.clear();
		max_mesh_out_size = 0;
	}
};

template <typename VType>
class mcuTransfer {

	//-------

	//array with transfer information from mesh_in : full cell indexes for transfer : i - mesh index, j - mesh cell index, k - supermesh (this mesh) cell index, l - device index for mesh
	cuPair<cuINT4, cuBReal>* transfer_in_info;

	//indexing object for the sorted transfer_in_info : i - starting index in transfer_in_info for a set of contributions to same supermesh cell, j - number of contributions to it
	cuINT2* transfer_in_info_index;

	//-------

	//array with transfer information to mesh_out : full cell indexes for transfer : i - mesh index, j - mesh cell index, k - supermesh (this mesh) cell index, l - device index for mesh
	cuPair<cuINT4, cuBReal>* transfer_out_info;

	//indexing object for the sorted transfer_out_info : i - starting index in transfer_out_info for a set of contributions to same output mesh cell, j - number of contributions to it
	cuINT2* transfer_out_info_index;

	//-------

	//array of meshes for transfer in and out from / to
	//mesh_in and mesh_out VECs have exactly the same rectangle and cellsize for each index, but may differ in value stored (e.g. magnetization and effective field) - they could also be exactly the same VEC
	//mesh_in2 can be used if we require multiple inputs, e.g. averaging inputs or multiplying inputs
	//mesh_out2 can be used if require duplicating outputs
	//For both mesh_in2 and mesh_out2, the input averaging and output duplicating is done if the respective VECs are not empty
	//Thus when using these modes, the secondary VECs should either be empty or have exactly same size as the primary VECs.
	//In any case, if using these modes the vectors below have to have exactly the same dimensions
	//NOTE : the cuVECs below will be set from same device as the one for which this mcuTransfer applies. To access data on other devices used UVA through cuVEC::pmcuVEC
	cuVEC<VType>* mesh_in;
	cuVEC<VType>* mesh_in2;
	cuVEC<cuBReal>* mesh_in2_real;
	cuVEC<VType>* mesh_out;
	cuVEC<VType>* mesh_out2;
	int mesh_out_num;	//number of entries in mesh_out

private:

	//auxiliary : does the actual transfer info copy after in and out cuVEC have been set
	//here n and pbox_d (array size number of devices) are the number of cells and subvec cells boxes for the mcuVEC holding this transfer object
	//for the input meshes (mesh_in) and output meshes (mesh_out) we also have the same information stored in vectors (which have size of mesh_in and mesh_out arrays respectively)
	template <typename cpuTransfer>
	__host__ bool copy_transfer_info(
		cpuTransfer& vec_transfer,
		std::vector<cuSZ3>& n_in, std::vector<cuBox*>& pbox_d_in,
		std::vector<cuSZ3>& n_out, std::vector<cuBox*>& pbox_d_out,
		cuSZ3 n, cuBox* pbox_d, int device_idx, int num_devices,
		mcuTransfer_Info& transfer_info);

	//set and allocate memory for transfer_in_info array and its indexing array
	__host__ bool set_transfer_in_info_size(size_t size, size_t size_index);

	//set and allocate memory for transfer_out_info array and its indexing array
	__host__ bool set_transfer_out_info_size(size_t size, size_t size_index);

public:

	__host__ void construct_cu_obj(void)
	{
		nullgpuptr(transfer_in_info);
		nullgpuptr(transfer_out_info);
		nullgpuptr(transfer_in_info_index);
		nullgpuptr(transfer_out_info_index);

		nullgpuptr(mesh_in);
		nullgpuptr(mesh_out);

		nullgpuptr(mesh_in2);
		nullgpuptr(mesh_out2);

		nullgpuptr(mesh_in2_real);

		set_transfer_in_info_size(0, 0);
		set_transfer_out_info_size(0, 0);

		set_gpu_value(mesh_out_num, (int)0);
	}

	__host__ void destruct_cu_obj(void)
	{
		gpu_free_managed(transfer_in_info);
		gpu_free_managed(transfer_out_info);
		gpu_free_managed(transfer_in_info_index);
		gpu_free_managed(transfer_out_info_index);

		gpu_free_managed(mesh_in);
		gpu_free_managed(mesh_out);

		gpu_free_managed(mesh_in2);
		gpu_free_managed(mesh_out2);

		gpu_free_managed(mesh_in2_real);
	}

	//reset to zero all transfer info data
	__host__ void clear_transfer_data(mcuTransfer_Info& transfer_info);

	//--------------------------------------------AUXILIARY

	//clear values in mesh_out
	__host__ void zero_output_meshes(mcuTransfer_Info& transfer_info);
	//clear values in mesh_out and mesh_out2
	__host__ void zero_output_duplicated_meshes(mcuTransfer_Info& transfer_info);

	//--------------------------------------------MESH TRANSFER COPY

	//NOTES : n and pbox_d (array size number of devices) are the number of cells and subvec cells boxes for the mcuVEC holding this transfer object

	//SINGLE INPUT, SINGLE OUTPUT

	template <typename cpuTransfer, typename MTypeIn, typename MTypeOut>
	__host__ bool copy_transfer_info(
		const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in,
		const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out,
		cpuTransfer& vec_transfer,
		cuSZ3 n, cuBox* pbox_d, int device_idx, int num_devices,
		mcuTransfer_Info& transfer_info);

	//MULTIPLE INPUTS, SINGLE OUTPUT

	template <typename cpuTransfer, typename MTypeIn, typename MTypeOut>
	__host__ bool copy_transfer_info_averagedinputs(
		const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
		const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in2,
		const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out,
		cpuTransfer& vec_transfer,
		cuSZ3 n, cuBox* pbox_d, int device_idx, int num_devices,
		mcuTransfer_Info& transfer_info);

	template <typename cpuTransfer, typename MTypeIn, typename MTypeInR, typename MTypeOut>
	__host__ bool copy_transfer_info_multipliedinputs(
		const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
		const std::vector<mcu_obj<MTypeInR, mcuVEC<cuBReal, MTypeInR>>*>& meshes_in2,
		const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out,
		cpuTransfer& vec_transfer,
		cuSZ3 n, cuBox* pbox_d, int device_idx, int num_devices,
		mcuTransfer_Info& transfer_info);

	//MULTIPLE INPUT, MULTIPLE OUTPUT

	template <typename cpuTransfer, typename MTypeIn, typename MTypeOut>
	__host__ bool copy_transfer_info_averagedinputs_duplicatedoutputs(
		const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
		const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in2,
		const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out1,
		const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out2,
		cpuTransfer& vec_transfer,
		cuSZ3 n, cuBox* pbox_d, int device_idx, int num_devices,
		mcuTransfer_Info& transfer_info);

	//--------------------------------------------MESH TRANSFER

	//do the actual transfer of values to and from this mesh using these

	//transfer_in_type as follows:
	//0:
	//SINGLE INPUT, SINGLE OUTPUT
	//1:
	//AVERAGED INPUTS
	//2:
	//MULTIPLIED INPUTS

	//transfer from input meshes
	template <typename VECType>
	void transfer_in(VECType& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_in_type);

	//transfer to output meshes.

	//transfer_out_type as follows:
	//0:
	//SINGLE INPUT, SINGLE OUTPUT
	//1:
	//DUPLICATED OUTPUT

	template <typename VECType>
	void transfer_out(VECType& sMesh_subVEC, mcuTransfer_Info& transfer_info, int transfer_out_type);
};

//--------------------------------------------HELPERS

//reset to zero all transfer info data
template <typename VType>
__host__ void mcuTransfer<VType>::clear_transfer_data(mcuTransfer_Info& transfer_info)
{
	set_transfer_in_info_size(0, 0);
	set_transfer_out_info_size(0, 0);

	gpu_free_managed(mesh_in);
	gpu_free_managed(mesh_out);
	set_gpu_value(mesh_out_num, (int)0);

	gpu_free_managed(mesh_in2);
	gpu_free_managed(mesh_out2);

	gpu_free_managed(mesh_in2_real);

	transfer_info.clear();
}

//set and allocate memory for transfer_in_info array and its indexing array
template <typename VType>
__host__ bool mcuTransfer<VType>::set_transfer_in_info_size(size_t size, size_t size_index)
{
	if (size == 0) {

		gpu_free_managed(transfer_in_info);
		gpu_free_managed(transfer_in_info_index);
		return true;
	}

	//new size value set : must also adjust memory allocation
	cudaError_t error1 = gpu_alloc_managed(transfer_in_info, size);
	cudaError_t error2 = gpu_alloc_managed(transfer_in_info_index, size_index);
	if (error1 != cudaSuccess || error2 != cudaSuccess) {

		gpu_free_managed(transfer_in_info);
		gpu_free_managed(transfer_in_info_index);

		return false;
	}

	return true;
}

//set and allocate memory for transfer_out_info array and its indexing array
template <typename VType>
__host__ bool mcuTransfer<VType>::set_transfer_out_info_size(size_t size, size_t size_index)
{
	if (size == 0) {

		gpu_free_managed(transfer_out_info);
		gpu_free_managed(transfer_out_info_index);
		return true;
	}

	//new size value set : must also adjust memory allocation
	cudaError_t error1 = gpu_alloc_managed(transfer_out_info, size);
	cudaError_t error2 = gpu_alloc_managed(transfer_out_info_index, size_index);
	if (error1 != cudaSuccess || error2 != cudaSuccess) {

		gpu_free_managed(transfer_out_info);
		gpu_free_managed(transfer_out_info_index);

		return false;
	}

	return true;
}

//--------------------------------------------MESH TRANSFER COPY

//auxiliary : does the actual transfer info copy after in and out cuVEC have been set
//here n and pbox_d (array size number of devices) are the number of cells and subvec cells boxes for the mcuVEC holding this transfer object
//for the input meshes (mesh_in) and output meshes (mesh_out) we also have the same information stored in vectors (which have size of mesh_in and mesh_out arrays respectively)
template <typename VType>
template <typename cpuTransfer>
__host__ bool mcuTransfer<VType>::copy_transfer_info(
	cpuTransfer& vec_transfer,
	std::vector<cuSZ3>& n_in, std::vector<cuBox*>& pbox_d_in,
	std::vector<cuSZ3>& n_out, std::vector<cuBox*>& pbox_d_out,
	cuSZ3 n, cuBox* pbox_d, int device_idx, int num_devices,
	mcuTransfer_Info& transfer_info)
{
	//when the final transfer info object is used, we need to avoid data race conditions where a supermesh cell can receive multiple contributions
	//for single-GPU usage atomicAdd is mostly good enough if on average there are not a large number of contributions for each cell
	//for multi-GPU usage atomicAdd is not practical (extremely poor performance)
	//block reduction can be used though which works as:
	//1. sort the transfer info by destination cell index, then we can identify multiple contributions to same cell
	//2. construct a separate indexing object which contains a) first cell contribution index in transfer info, b) number of contributions to that cell
	//3. the cuda kernel is launched with number of blocks equal to size of indexing object from 2., and block size equal to smallest power of 2 greater or equal to maximum number of contributions to any cell
	//Special case of only 1 contribution to all cells can be handled separately. Then each block will first perform a reduction to obtain the total contribution, and finally this is added to destination cell.
	//This avoids use of atomicAdd and data race conditions. 
	//Multiple block reductions can be performed when number of contributions exceeds maximum possible block size.
	//It is possible the transfer info contains large disparities between number of contributions to different destination cells.
	//This is inefficient to handle in a single kernel launch since same (largest) block size must be used.
	//For this reason the indexing object must be sorted by number of contributions, and for each a block size determined (powers of 2).
	//Then multiple kernel launches can be used so that each kernel launch handles only one block size, i.e. only a subset of the ordered indexing object.
	//This requires an additional array be constructed, with size equal to different number of block sizes.
	//This array contains offsets in indexing object and block size to use. The number of blocks for each kernel can then be determined easily.
		
	//convert global indexes into device-local indexes
	//for supermesh only extract and convert indexes which are contained in current device (device_idx_dst)
	//for in/out meshes convert global cell indexes into a device index and corresponding device-local cell index
	auto convert_indexes = [](
		std::vector<std::pair<cuINT3, cuBReal>>& transfer_info_src,
		std::vector<std::pair<cuINT4, cuBReal>>& transfer_info_dst,
		std::vector<cuSZ3>& n_src, std::vector<cuBox*>& pbox_d_src,
		cuSZ3 n_dst, cuBox* pbox_d_dst, int device_idx_dst, int num_devices) -> void {

#pragma omp parallel for
		for (int idx = 0; idx < transfer_info_src.size(); idx++) {

			// i - mesh index, j - mesh cell index, k - supermesh (this mesh) cell index
			cuINT3 global_idx = transfer_info_src[idx].first;

			// i - mesh index, j - mesh cell index, k - supermesh (this mesh) cell index, l - device index for mesh
			cuINT4 local_idx;
			//mark entry as inactive initially - if this device (device_idx_dst) does not contain this supermesh cell, it will remain marked as inactive and after index conversion we can prune these entries
			local_idx = cuINT4(-1);

			//supermesh index in full space with i, j, k indexes
			int s_idx = global_idx.k;
			int s_i = s_idx % n_dst.x;
			int s_j = (s_idx / n_dst.x) % n_dst.y;
			int s_k = s_idx / (n_dst.x * n_dst.y);

			//box size for this device
			cuSZ3 n_d = pbox_d_dst[device_idx_dst].size();
			//if supermesh index contained then adjust it for this device
			if (pbox_d_dst[device_idx_dst].contains(s_i, s_j, s_k)) {

				local_idx.k = (s_i - pbox_d_dst[device_idx_dst].s.i) + (s_j - pbox_d_dst[device_idx_dst].s.j) * n_d.x + (s_k - pbox_d_dst[device_idx_dst].s.k) * n_d.x * n_d.y;

				//now that entry is active on this device, also adjust the contributing mesh index

				//input mesh
				local_idx.i = global_idx.i;
				//input mesh index in full space with i, j, k indexes
				int m_idx = global_idx.j;
				int m_i = m_idx % n_src[local_idx.i].x;
				int m_j = (m_idx / n_src[local_idx.i].x) % n_src[local_idx.i].y;
				int m_k = m_idx / (n_src[local_idx.i].x * n_src[local_idx.i].y);

				//for mesh with index local_idx.i, find which box contains mesh cell index m_idx
				for (int src_device_idx = 0; src_device_idx < num_devices; src_device_idx++) {

					if (pbox_d_src[local_idx.i][src_device_idx].contains(m_i, m_j, m_k)) {

						//set device index
						local_idx.l = src_device_idx;

						//box size for this device
						n_d = pbox_d_src[local_idx.i][src_device_idx].size();
						//adjust index for this device
						local_idx.j = (m_i - pbox_d_src[local_idx.i][src_device_idx].s.i) + (m_j - pbox_d_src[local_idx.i][src_device_idx].s.j) * n_d.x + (m_k - pbox_d_src[local_idx.i][src_device_idx].s.k) * n_d.x * n_d.y;
					}
				}
			}

			transfer_info_dst[idx].first = local_idx;
			transfer_info_dst[idx].second = transfer_info_src[idx].second;
		}
	};

	//get maximum block size to use
	auto find_pow2 = [](int num_contributions) -> unsigned int
	{
		//find smallest power of 2 greater or equal to num_contributions (this will be the block size)
		unsigned int pow2 = 1;
		unsigned int n = num_contributions;
		while (n != 0 && pow2 != num_contributions) {

			n >>= 1;
			pow2 <<= 1;
		}

		return pow2;
	};

	//--------------------------------------- Input Meshes

	if (vec_transfer.size_transfer_in() && n_in.size()) {

		//get the transfer info from vec_transfer into flattened and converted form in cpuVEC_transfer_in_info
		std::vector<std::pair<cuINT3, cuBReal>> cpuVEC_transfer_in_info;

		//set size
		if (!malloc_vector(cpuVEC_transfer_in_info, vec_transfer.size_transfer_in())) return false;

		//flatten
		std::vector<std::pair<INT3, double>> cpuVEC_transfer_in_info_flat = vec_transfer.get_flattened_transfer_in_info();
		if (cpuVEC_transfer_in_info_flat.size() != vec_transfer.size_transfer_in()) return false;

		//now copy and convert
#pragma omp parallel for
		for (int idx = 0; idx < vec_transfer.size_transfer_in(); idx++) {

			cpuVEC_transfer_in_info[idx].first = cpuVEC_transfer_in_info_flat[idx].first;
			cpuVEC_transfer_in_info[idx].second = cpuVEC_transfer_in_info_flat[idx].second;
		}

		//-----

		//cpuVEC_transfer_in_info can now be used to calculate adjusted indexes and devices to be stored in transfer_in_info_cpu
		//initially make transfer_in_info_cpu size same as cpuVEC_transfer_in_info
		//however transfer_in_info_cpu should typically be smaller since not all supermesh cell indexes will be contained in this device
		//after index conversions then prune 'inactive' entires
		std::vector<std::pair<cuINT4, cuBReal>> transfer_in_info_cpu(cpuVEC_transfer_in_info.size());

		convert_indexes(cpuVEC_transfer_in_info, transfer_in_info_cpu, n_in, pbox_d_in, n, pbox_d, device_idx, num_devices);

		//-----

		//1. -> sort transfer_in_info_cpu first by supermesh cell index (destination cell)
		std::sort(transfer_in_info_cpu.begin(), transfer_in_info_cpu.end(),
			[](const std::pair<cuINT4, cuBReal>& a, const std::pair<cuINT4, cuBReal>& b) { return a.first.k < b.first.k; });

		//with transfer_in_info_cpu sorted, all destination cells with -1 index (i.e. the ones we need to prune) will appear at the beginning
		//thus find out how much of transfer_in_info_cpu we need to erase by finding starting cell index in transfer_in_info_cpu with valid destination cells
		std::vector<int> prune_index(omp_get_num_procs(), transfer_in_info_cpu.size());

#pragma omp parallel for
		for (int idx = 0; idx < transfer_in_info_cpu.size(); idx++) {

			//if already found on this thread continue
			if (prune_index[omp_get_thread_num()] < transfer_in_info_cpu.size()) continue;

			//found one a possible start cell on this thread
			if (transfer_in_info_cpu[idx].first.k >= 0) prune_index[omp_get_thread_num()] = idx;
		}

		//find smallest from all threads
		int prune_end_index = transfer_in_info_cpu.size();
		for (int idx = 0; idx < prune_index.size(); idx++)
			prune_end_index = (prune_end_index < prune_index[idx] ? prune_end_index : prune_index[idx]);

		transfer_in_info_cpu.erase(transfer_in_info_cpu.begin(), transfer_in_info_cpu.begin() + prune_end_index);

		if (transfer_in_info_cpu.size()) {

			//2. -> make indexing object for transfer_in_info_cpu

			//divide work between cpu threads by identifying starting indexes in transfer_in_info_cpu roughly equally spaced
			int OmpThreads = (omp_get_num_procs() < transfer_in_info_cpu.size() ? omp_get_num_procs() : 1);
			std::vector<int> thread_start_index(OmpThreads);

			thread_start_index[0] = 0;
			for (int idx = 1; idx < OmpThreads; idx++) {

				int ts_idx = idx * (int)(transfer_in_info_cpu.size() / OmpThreads);
				//adjust index so complete sets of same destination cells are handled on each thread
				int dst_cell = transfer_in_info_cpu[ts_idx].first.k;
				while (ts_idx < transfer_in_info_cpu.size())
				{
					if (transfer_in_info_cpu[ts_idx].first.k != dst_cell) {	//start of new set detected

						if (ts_idx > thread_start_index[idx - 1]) break;	//must make sure this thread will handle a different portion than other threads
						else dst_cell = transfer_in_info_cpu[ts_idx].first.k;
					}
					++ts_idx;
				}
				thread_start_index[idx] = ts_idx;	//set adjusted index : this could actually be transfer_in_info_cpu.size(), so check before using
			}

			//count required indexing vector size
			//for each thread need to determine number of elements it will set in the indexing vector, since we need to know at which index in this it needs to start storing entries
			std::vector<int> thread_index_size(OmpThreads, 1);
			std::vector<int> thread_index_start(OmpThreads, 0);

#pragma omp parallel for
			for (int idx = 0; idx < OmpThreads; idx++) {
				size_t index_size_thread = 0;
				int dst_cell = (thread_start_index[idx] < transfer_in_info_cpu.size() ? transfer_in_info_cpu[thread_start_index[idx]].first.k : 0);
				for (int tidx = thread_start_index[idx] + 1; tidx < (idx == OmpThreads - 1 ? transfer_in_info_cpu.size() : thread_start_index[idx + 1]); tidx++) {
					//count unique destination cells
					if (transfer_in_info_cpu[tidx].first.k != dst_cell) { thread_index_size[idx]++; dst_cell = transfer_in_info_cpu[tidx].first.k; }
				}

				//if mesh space too small some threads may have nothing to do so set their size to zero
				if (thread_start_index[idx] == transfer_in_info_cpu.size()) thread_index_size[idx] = 0;
			}

			//now make the actual indexing vector with required size (needed to determine size first, then populate it, since cannot/shouldn't use push_back)
			size_t index_size = 0;
			for (int idx = 0; idx < OmpThreads; idx++) {

				thread_index_start[idx] = index_size;
				index_size += thread_index_size[idx];
			}
			std::vector<cuINT2> transfer_in_info_index_cpu(index_size);

#pragma omp parallel for
			for (int idx = 0; idx < OmpThreads; idx++) {

				//index in indexing vector - this is the start value for this thread
				int iv_idx = thread_index_start[idx];
				//first destination cell index and its value
				int dst_cell_idx = thread_start_index[idx];
				int dst_cell = (thread_start_index[idx] < transfer_in_info_cpu.size() ? transfer_in_info_cpu[thread_start_index[idx]].first.k : 0);
				//number of contributions to this destination cell
				int num_contributions = 0;

				for (int tidx = thread_start_index[idx]; tidx < (idx == OmpThreads - 1 ? transfer_in_info_cpu.size() : thread_start_index[idx + 1]); tidx++) {

					if (transfer_in_info_cpu[tidx].first.k != dst_cell) {

						//finished counting a set of contributions, so make a new entry in indexing vector
						//the checks above mean either we detected start of a new set, or else this was the last contribution to count on this thread
						transfer_in_info_index_cpu[iv_idx++] = cuINT2(dst_cell_idx, num_contributions);

						//update values for next set to count
						dst_cell_idx = tidx;
						dst_cell = transfer_in_info_cpu[tidx].first.k;
						num_contributions = 1;
					}
					else num_contributions++;

					//if last index in the loop we need to add this contribution also
					if (tidx + 1 == (idx == OmpThreads - 1 ? transfer_in_info_cpu.size() : thread_start_index[idx + 1])) {

						transfer_in_info_index_cpu[iv_idx] = cuINT2(dst_cell_idx, num_contributions);
					}
				}
			}

			//-----

			//sort the indexing vector by number of contributions
			std::sort(transfer_in_info_index_cpu.begin(), transfer_in_info_index_cpu.end(),
				[](const cuINT2& a, const cuINT2& b) { return a.j < b.j; });

			//store information in transfer_info object: vector of different block sizes for transfer and start indexes in transfer_in_info_index for each
			
			std::vector<std::vector<cuINT2>> block_sizes(omp_get_num_procs());
#pragma omp parallel for
			for (int idx = 0; idx < transfer_in_info_index_cpu.size(); idx++) {

				int tn = omp_get_thread_num();

				//on this thread identify different number of contributions of starting index in transfer_in_info_index for each
				//remember num_contributions in transfer_in_info_index_cpu is not adjusted to nearest containing power of 2 so this will need to be done after
				if (!block_sizes[tn].size() || block_sizes[tn].back().j != transfer_in_info_index_cpu[idx].j) {

					//push_back is fine since in typical use cases there won't be a large number of different contributions
					block_sizes[tn].push_back(cuINT2(idx, transfer_in_info_index_cpu[idx].j));
				}
			}

			//block_sizes now needs to be reduced to a single object by collecting information stored by different threads above, also finding power-of-2 blocksizes
			std::vector<cuINT2> block_sizes_collapsed;
			for (int tn = 0; tn < block_sizes.size(); tn++) {
				for (int idx = 0; idx < block_sizes[tn].size(); idx++) {
					block_sizes_collapsed.push_back(cuINT2(block_sizes[tn][idx].i, find_pow2(block_sizes[tn][idx].j)));
				}
			}

			//sort block_sizes_collapsed by index values (i) - since these are index values in transfer_in_info_index_cpu, which is sorted by block sizes, then block sizes will also appear in order
			std::sort(block_sizes_collapsed.begin(), block_sizes_collapsed.end(),
				[](const cuINT2& a, const cuINT2& b) { return a.i < b.i; });

			//this is not quite the finished object though, since some block sizes will possibly appear multiple times, so only need to keep unique ones with smallest index values
			transfer_info.transfer_in_info_index_blocksizes.clear();
			for (int idx = 0; idx < block_sizes_collapsed.size(); idx++) {

				if (idx == 0 || transfer_info.transfer_in_info_index_blocksizes.back().j != block_sizes_collapsed[idx].j)
					transfer_info.transfer_in_info_index_blocksizes.push_back(cuINT3(block_sizes_collapsed[idx].i, block_sizes_collapsed[idx].j, 0));
			}

			//also determine number of indexes with same block size
			for (int idx = 0; idx < transfer_info.transfer_in_info_index_blocksizes.size(); idx++) {

				if (idx < transfer_info.transfer_in_info_index_blocksizes.size() - 1)
					transfer_info.transfer_in_info_index_blocksizes[idx].k = transfer_info.transfer_in_info_index_blocksizes[idx + 1].i - transfer_info.transfer_in_info_index_blocksizes[idx].i;
				else
					transfer_info.transfer_in_info_index_blocksizes[idx].k = transfer_in_info_index_cpu.size() - transfer_info.transfer_in_info_index_blocksizes[idx].i;
			}

			//-----
			//finally copy transfer_in_info_cpu and transfer_in_info_index_cpu to gpu memory

			//allocate gpu memory for transfer_in_info
			if (!set_transfer_in_info_size(transfer_in_info_cpu.size(), transfer_in_info_index_cpu.size())) return false;

			//copy the indexing vector to gpu memory
			cpu_to_gpu_managed(transfer_in_info_index, transfer_in_info_index_cpu.data(), transfer_in_info_index_cpu.size());

			//copy to transfer_in_info
			cpu_to_gpu_managed(transfer_in_info, transfer_in_info_cpu.data(), transfer_in_info_cpu.size());
		}
		else set_transfer_in_info_size(0, 0);
	}
	else set_transfer_in_info_size(0, 0);
	
	//--------------------------------------- Output Meshes
	
	//Same as above but for output meshes

	if (vec_transfer.size_transfer_out() && n_out.size()) {

		//get the transfer info from vec_transfer into flattened and converted form in cpuVEC_transfer_out_info
		std::vector<std::pair<cuINT3, cuBReal>> cpuVEC_transfer_out_info;

		//set size
		if (!malloc_vector(cpuVEC_transfer_out_info, vec_transfer.size_transfer_out())) return false;

		//flatten
		std::vector<std::pair<INT3, double>> cpuVEC_transfer_out_info_flat = vec_transfer.get_flattened_transfer_out_info();
		if (cpuVEC_transfer_out_info_flat.size() != vec_transfer.size_transfer_out()) return false;

		//now copy and convert
#pragma omp parallel for
		for (int idx = 0; idx < vec_transfer.size_transfer_out(); idx++) {

			cpuVEC_transfer_out_info[idx].first = cpuVEC_transfer_out_info_flat[idx].first;
			cpuVEC_transfer_out_info[idx].second = cpuVEC_transfer_out_info_flat[idx].second;
		}

		//-----

		//cpuVEC_transfer_out_info can now be used to calculate adjusted indexes and devices to be stored in transfer_out_info_cpu
		//initially make transfer_out_info_cpu size same as cpuVEC_transfer_out_info
		//however transfer_out_info_cpu should typically be smaller since not all required supermesh cell indexes will be contained in this device
		//after index conversions then prune 'inactive' entires
		std::vector<std::pair<cuINT4, cuBReal>> transfer_out_info_cpu(cpuVEC_transfer_out_info.size());

		convert_indexes(cpuVEC_transfer_out_info, transfer_out_info_cpu, n_out, pbox_d_out, n, pbox_d, device_idx, num_devices);

		//-----

		//1. -> sort transfer_out_info_cpu first by destination mesh index, device and cell index
		std::sort(transfer_out_info_cpu.begin(), transfer_out_info_cpu.end(),
			[](const std::pair<cuINT4, cuBReal>& a, const std::pair<cuINT4, cuBReal>& b) 
		{ return (a.first.i < b.first.i) || (a.first.i == b.first.i && a.first.l < b.first.l) || (a.first.i == b.first.i && a.first.l == b.first.l && a.first.j < b.first.j); });

		//with transfer_out_info_cpu sorted, all destination cells with -1 index (i.e. the ones we need to prune) will appear at the beginning
		//thus find out how much of transfer_out_info_cpu we need to erase by finding starting cell index in transfer_out_info_cpu with valid destination cells
		std::vector<int> prune_index(omp_get_num_procs(), transfer_out_info_cpu.size());

#pragma omp parallel for
		for (int idx = 0; idx < transfer_out_info_cpu.size(); idx++) {

			//if already found on this thread continue
			if (prune_index[omp_get_thread_num()] < transfer_out_info_cpu.size()) continue;

			//found one a possible start cell on this thread
			if (transfer_out_info_cpu[idx].first.j >= 0) prune_index[omp_get_thread_num()] = idx;
		}

		//find smallest from all threads
		int prune_end_index = transfer_out_info_cpu.size();
		for (int idx = 0; idx < prune_index.size(); idx++)
			prune_end_index = (prune_end_index < prune_index[idx] ? prune_end_index : prune_index[idx]);

		transfer_out_info_cpu.erase(transfer_out_info_cpu.begin(), transfer_out_info_cpu.begin() + prune_end_index);

		if (transfer_out_info_cpu.size()) {

			//2. -> make indexing object for transfer_out_info_cpu

			//divide work between cpu threads by identifying starting indexes in transfer_out_info_cpu roughly equally spaced
			int OmpThreads = (omp_get_num_procs() < transfer_out_info_cpu.size() ? omp_get_num_procs() : 1);
			std::vector<int> thread_start_index(OmpThreads);

			thread_start_index[0] = 0;
			for (int idx = 1; idx < OmpThreads; idx++) {

				int ts_idx = idx * (int)(transfer_out_info_cpu.size() / OmpThreads);
				//adjust index so complete sets of same destination cells are handled on each thread (mesh index, device number, device-local cell index)
				cuINT3 dst_cell = cuINT3(transfer_out_info_cpu[ts_idx].first.i, transfer_out_info_cpu[ts_idx].first.l, transfer_out_info_cpu[ts_idx].first.j);
				while (ts_idx < transfer_out_info_cpu.size())
				{
					cuINT3 trial_cell = cuINT3(transfer_out_info_cpu[ts_idx].first.i, transfer_out_info_cpu[ts_idx].first.l, transfer_out_info_cpu[ts_idx].first.j);
					if (trial_cell != dst_cell) {	//start of new set detected

						if (ts_idx > thread_start_index[idx - 1]) break;	//must make sure this thread will handle a different portion than other threads
						else dst_cell = trial_cell;
					}
					++ts_idx;
				}
				thread_start_index[idx] = ts_idx;	//set adjusted index : this could actually be transfer_out_info_cpu.size(), so check before using
			}

			//count required indexing vector size
			//for each thread need to determine number of elements it will set in the indexing vector, since we need to know at which index in this it needs to start storing entries
			std::vector<int> thread_index_size(OmpThreads, 1);
			std::vector<int> thread_index_start(OmpThreads, 0);

#pragma omp parallel for
			for (int idx = 0; idx < OmpThreads; idx++) {
				size_t index_size_thread = 0;
				cuINT3 dst_cell = (thread_start_index[idx] < transfer_out_info_cpu.size() ?
					cuINT3(transfer_out_info_cpu[thread_start_index[idx]].first.i, transfer_out_info_cpu[thread_start_index[idx]].first.l, transfer_out_info_cpu[thread_start_index[idx]].first.j) : cuINT3());
				for (int tidx = thread_start_index[idx] + 1; tidx < (idx == OmpThreads - 1 ? transfer_out_info_cpu.size() : thread_start_index[idx + 1]); tidx++) {
					//count unique destination cells
					cuINT3 trial_cell = cuINT3(transfer_out_info_cpu[tidx].first.i, transfer_out_info_cpu[tidx].first.l, transfer_out_info_cpu[tidx].first.j);
					if (trial_cell != dst_cell) { thread_index_size[idx]++; dst_cell = trial_cell; }
				}

				//if mesh space too small some threads may have nothing to do so set their size to zero
				if (thread_start_index[idx] == transfer_out_info_cpu.size()) thread_index_size[idx] = 0;
			}

			//now make the actual indexing vector with required size (needed to determine size first, then populate it, since cannot/shouldn't use push_back)
			size_t index_size = 0;
			for (int idx = 0; idx < OmpThreads; idx++) {

				thread_index_start[idx] = index_size;
				index_size += thread_index_size[idx];
			}
			std::vector<cuINT2> transfer_out_info_index_cpu(index_size);

#pragma omp parallel for
			for (int idx = 0; idx < OmpThreads; idx++) {

				//index in indexing vector - this is the start value for this thread
				int iv_idx = thread_index_start[idx];
				//first destination cell index and its value
				int dst_cell_idx = thread_start_index[idx];
				cuINT3 dst_cell = (thread_start_index[idx] < transfer_out_info_cpu.size() ?
					cuINT3(transfer_out_info_cpu[thread_start_index[idx]].first.i, transfer_out_info_cpu[thread_start_index[idx]].first.l, transfer_out_info_cpu[thread_start_index[idx]].first.j) : cuINT3());
				//number of contributions to this destination cell
				int num_contributions = 0;

				for (int tidx = thread_start_index[idx]; tidx < (idx == OmpThreads - 1 ? transfer_out_info_cpu.size() : thread_start_index[idx + 1]); tidx++) {

					cuINT3 trial_cell = cuINT3(transfer_out_info_cpu[tidx].first.i, transfer_out_info_cpu[tidx].first.l, transfer_out_info_cpu[tidx].first.j);
					if (trial_cell != dst_cell) {

						//finished counting a set of contributions, so make a new entry in indexing vector
						//the checks above mean either we detected start of a new set, or else this was the last contribution to count on this thread
						transfer_out_info_index_cpu[iv_idx++] = cuINT2(dst_cell_idx, num_contributions);

						//update values for next set to count
						dst_cell_idx = tidx;
						dst_cell = trial_cell;
						num_contributions = 1;
					}
					else num_contributions++;

					//if last index in the loop we need to add this contribution also
					if (tidx + 1 == (idx == OmpThreads - 1 ? transfer_out_info_cpu.size() : thread_start_index[idx + 1])) {

						transfer_out_info_index_cpu[iv_idx] = cuINT2(dst_cell_idx, num_contributions);
					}
				}
			}

			//-----

			//sort the indexing vector by number of contributions
			std::sort(transfer_out_info_index_cpu.begin(), transfer_out_info_index_cpu.end(),
				[](const cuINT2& a, const cuINT2& b) { return a.j < b.j; });

			//store information in transfer_info object: vector of different block sizes for transfer and start indexes in transfer_out_info_index for each

			std::vector<std::vector<cuINT2>> block_sizes(omp_get_num_procs());
#pragma omp parallel for
			for (int idx = 0; idx < transfer_out_info_index_cpu.size(); idx++) {

				int tn = omp_get_thread_num();

				//on this thread identify different number of contributions of starting index in transfer_out_info_index for each
				//remember num_contributions in transfer_out_info_index_cpu is not adjusted to nearest containing power of 2 so this will need to be done after
				if (!block_sizes[tn].size() || block_sizes[tn].back().j != transfer_out_info_index_cpu[idx].j) {

					//push_back is fine since in typical use cases there won't be a large number of different contributions
					block_sizes[tn].push_back(cuINT2(idx, transfer_out_info_index_cpu[idx].j));
				}
			}

			//block_sizes now needs to be reduced to a single object by collecting information stored by different threads above, also finding power-of-2 blocksizes
			std::vector<cuINT2> block_sizes_collapsed;
			for (int tn = 0; tn < block_sizes.size(); tn++) {
				for (int idx = 0; idx < block_sizes[tn].size(); idx++) {
					block_sizes_collapsed.push_back(cuINT2(block_sizes[tn][idx].i, find_pow2(block_sizes[tn][idx].j)));
				}
			}

			//sort block_sizes_collapsed by index values (i) - since these are index values in transfer_out_info_index_cpu, which is sorted by block sizes, then block sizes will also appear in order
			std::sort(block_sizes_collapsed.begin(), block_sizes_collapsed.end(),
				[](const cuINT2& a, const cuINT2& b) { return a.i < b.i; });

			//this is not quite the finished object though, since some block sizes will possibly appear multiple times, so only need to keep unique ones with smallest index values
			transfer_info.transfer_out_info_index_blocksizes.clear();
			for (int idx = 0; idx < block_sizes_collapsed.size(); idx++) {

				if (idx == 0 || transfer_info.transfer_out_info_index_blocksizes.back().j != block_sizes_collapsed[idx].j)
					transfer_info.transfer_out_info_index_blocksizes.push_back(cuINT3(block_sizes_collapsed[idx].i, block_sizes_collapsed[idx].j, 0));
			}

			//also determine number of indexes with same block size
			for (int idx = 0; idx < transfer_info.transfer_out_info_index_blocksizes.size(); idx++) {

				if (idx < transfer_info.transfer_out_info_index_blocksizes.size() - 1)
					transfer_info.transfer_out_info_index_blocksizes[idx].k = transfer_info.transfer_out_info_index_blocksizes[idx + 1].i - transfer_info.transfer_out_info_index_blocksizes[idx].i;
				else
					transfer_info.transfer_out_info_index_blocksizes[idx].k = transfer_out_info_index_cpu.size() - transfer_info.transfer_out_info_index_blocksizes[idx].i;
			}

			//-----

			//finally copy transfer_out_info_cpu and transfer_out_info_index_cpu to gpu memory

			//allocate gpu memory for transfer_out_info
			if (!set_transfer_out_info_size(transfer_out_info_cpu.size(), transfer_out_info_index_cpu.size())) return false;

			//copy the indexing vector to gpu memory
			cpu_to_gpu_managed(transfer_out_info_index, transfer_out_info_index_cpu.data(), transfer_out_info_index_cpu.size());

			//copy to transfer_out_info
			cpu_to_gpu_managed(transfer_out_info, transfer_out_info_cpu.data(), transfer_out_info_cpu.size());
		}
		else set_transfer_out_info_size(0, 0);
	}
	else set_transfer_out_info_size(0, 0);

	return true;
}

//SINGLE INPUT, SINGLE OUTPUT

//copy precalculated transfer info from cpu memory
template <typename VType>
template <typename cpuTransfer, typename MTypeIn, typename MTypeOut>
__host__ bool mcuTransfer<VType>::copy_transfer_info(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out,
	cpuTransfer& vec_transfer,
	cuSZ3 n, cuBox* pbox_d, int device_idx, int num_devices,
	mcuTransfer_Info& transfer_info)
{
	clear_transfer_data(transfer_info);

	size_t size_in = meshes_in.size();
	size_t size_out = meshes_out.size();

	//------

	cu_arr<cuVEC<VType>> mesh_in_arr;
	cu_arr<cuVEC<VType>> mesh_out_arr;

	//copy cuVECs on device with device_idx from meshes_in into mesh_in_arr
	for (int idx = 0; idx < size_in; idx++) {

		mesh_in_arr.push_back((cuVEC<VType>*&)meshes_in[idx]->get_managed_object(device_idx));
	}

	//copy cuVECs on device with device_idx from meshes_out into mesh_out_arr
	for (int idx = 0; idx < size_out; idx++) {

		mesh_out_arr.push_back((cuVEC<VType>*&)meshes_out[idx]->get_managed_object(device_idx));
	}

	//------

	//copy cuVEC pointers from mesh_in_arr to the managed mesh_in array for later usage
	if (size_in > 0) {

		gpu_alloc_managed(mesh_in, size_in);
		gpu_to_gpu_managed1st(mesh_in, (cuVEC<VType>*)mesh_in_arr, size_in);
	}

	//copy cuVEC pointers from mesh_out_arr to the managed mesh_out array for later usage
	if (size_out > 0) {

		gpu_alloc_managed(mesh_out, size_out);
		gpu_to_gpu_managed1st(mesh_out, (cuVEC<VType>*)mesh_out_arr, size_out);
		set_gpu_value(mesh_out_num, (int)size_out);
	}

	//------

	//extract sizes and cells boxes from meshes_in and meshes_out

	std::vector<cuSZ3> n_in;
	std::vector<cuBox*> pbox_d_in;
	std::vector<cuSZ3> n_out;
	std::vector<cuBox*> pbox_d_out;

	for (int idx = 0; idx < size_in; idx++) {

		n_in.push_back(meshes_in[idx]->n);
		pbox_d_in.push_back(meshes_in[idx]->devices_boxes());
	}

	//calculate maximum output mesh size on this device as this will be needed if we need to zero output meshes
	transfer_info.max_mesh_out_size = 0;

	for (int idx = 0; idx < size_out; idx++) {

		n_out.push_back(meshes_out[idx]->n);
		pbox_d_out.push_back(meshes_out[idx]->devices_boxes());
		transfer_info.max_mesh_out_size = (meshes_out[idx]->device_size(device_idx) > transfer_info.max_mesh_out_size ? meshes_out[idx]->device_size(device_idx) : transfer_info.max_mesh_out_size);
	}

	//------

	return copy_transfer_info(vec_transfer, n_in, pbox_d_in, n_out, pbox_d_out, n, pbox_d, device_idx, num_devices, transfer_info);
}

//MULTIPLE INPUTS, SINGLE OUTPUT

//copy precalculated transfer info from cpu memory
template <typename VType>
template <typename cpuTransfer, typename MTypeIn, typename MTypeOut>
__host__ bool mcuTransfer<VType>::copy_transfer_info_averagedinputs(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in2,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out,
	cpuTransfer& vec_transfer,
	cuSZ3 n, cuBox* pbox_d, int device_idx, int num_devices,
	mcuTransfer_Info& transfer_info)
{
	clear_transfer_data(transfer_info);

	size_t size_in = meshes_in1.size();
	size_t size_in2 = meshes_in2.size();
	if (size_in != size_in2) return false;

	size_t size_out = meshes_out.size();

	//------

	cu_arr<cuVEC<VType>> mesh_in_arr1, mesh_in_arr2;
	cu_arr<cuVEC<VType>> mesh_out_arr;

	//copy cuVECs on device with device_idx from meshes_in into mesh_in_arr
	for (int idx = 0; idx < size_in; idx++) {

		mesh_in_arr1.push_back((cuVEC<VType>*&)meshes_in1[idx]->get_managed_object(device_idx));
		mesh_in_arr2.push_back((cuVEC<VType>*&)meshes_in2[idx]->get_managed_object(device_idx));
	}

	//copy cuVECs on device with device_idx from meshes_out into mesh_out_arr
	for (int idx = 0; idx < size_out; idx++) {

		mesh_out_arr.push_back((cuVEC<VType>*&)meshes_out[idx]->get_managed_object(device_idx));
	}

	//------

	//copy cuVEC pointers from mesh_in_arr to the managed mesh_in array for later usage
	if (size_in > 0) {

		gpu_alloc_managed(mesh_in, size_in);
		gpu_to_gpu_managed1st(mesh_in, (cuVEC<VType>*)mesh_in_arr1, size_in);

		gpu_alloc_managed(mesh_in2, size_in);
		gpu_to_gpu_managed1st(mesh_in2, (cuVEC<VType>*)mesh_in_arr2, size_in);
	}

	//copy cuVEC pointers from mesh_out_arr to the managed mesh_out array for later usage
	if (size_out > 0) {

		gpu_alloc_managed(mesh_out, size_out);
		gpu_to_gpu_managed1st(mesh_out, (cuVEC<VType>*)mesh_out_arr, size_out);
		set_gpu_value(mesh_out_num, (int)size_out);
	}

	//------

	//extract sizes and cells boxes from meshes_in and meshes_out

	std::vector<cuSZ3> n_in;
	std::vector<cuBox*> pbox_d_in;
	std::vector<cuSZ3> n_out;
	std::vector<cuBox*> pbox_d_out;

	for (int idx = 0; idx < size_in; idx++) {

		n_in.push_back(meshes_in1[idx]->n);
		pbox_d_in.push_back(meshes_in1[idx]->devices_boxes());
	}

	//calculate maximum output mesh size on this device as this will be needed if we need to zero output meshes
	transfer_info.max_mesh_out_size = 0;

	for (int idx = 0; idx < size_out; idx++) {

		n_out.push_back(meshes_out[idx]->n);
		pbox_d_out.push_back(meshes_out[idx]->devices_boxes());
		transfer_info.max_mesh_out_size = (meshes_out[idx]->device_size(device_idx) > transfer_info.max_mesh_out_size ? meshes_out[idx]->device_size(device_idx) : transfer_info.max_mesh_out_size);
	}

	//------

	return copy_transfer_info(vec_transfer, n_in, pbox_d_in, n_out, pbox_d_out, n, pbox_d, device_idx, num_devices, transfer_info);
}

//copy precalculated transfer info from cpu memory
template <typename VType>
template <typename cpuTransfer, typename MTypeIn, typename MTypeInR, typename MTypeOut>
__host__ bool mcuTransfer<VType>::copy_transfer_info_multipliedinputs(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
	const std::vector<mcu_obj<MTypeInR, mcuVEC<cuBReal, MTypeInR>>*>& meshes_in2,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out,
	cpuTransfer& vec_transfer,
	cuSZ3 n, cuBox* pbox_d, int device_idx, int num_devices,
	mcuTransfer_Info& transfer_info)
{
	clear_transfer_data(transfer_info);

	size_t size_in = meshes_in1.size();
	size_t size_in2 = meshes_in2.size();
	if (size_in != size_in2) return false;

	size_t size_out = meshes_out.size();

	//------

	cu_arr<cuVEC<VType>> mesh_in_arr1;
	cu_arr<cuVEC<cuBReal>> mesh_in_arr2;
	cu_arr<cuVEC<VType>> mesh_out_arr;

	//copy cuVECs on device with device_idx from meshes_in into mesh_in_arr
	for (int idx = 0; idx < size_in; idx++) {

		mesh_in_arr1.push_back((cuVEC<VType>*&)meshes_in1[idx]->get_managed_object(device_idx));
		mesh_in_arr2.push_back((cuVEC<cuBReal>*&)meshes_in2[idx]->get_managed_object(device_idx));
	}

	//copy cuVECs on device with device_idx from meshes_out into mesh_out_arr
	for (int idx = 0; idx < size_out; idx++) {

		mesh_out_arr.push_back((cuVEC<VType>*&)meshes_out[idx]->get_managed_object(device_idx));
	}

	//------

	//copy cuVEC pointers from mesh_in_arr to the managed mesh_in array for later usage
	if (size_in > 0) {

		gpu_alloc_managed(mesh_in, size_in);
		gpu_to_gpu_managed1st(mesh_in, (cuVEC<VType>*)mesh_in_arr1, size_in);

		gpu_alloc_managed(mesh_in2_real, size_in);
		gpu_to_gpu_managed1st(mesh_in2_real, (cuVEC<cuBReal>*)mesh_in_arr2, size_in);
	}

	//copy cuVEC pointers from mesh_out_arr to the managed mesh_out array for later usage
	if (size_out > 0) {

		gpu_alloc_managed(mesh_out, size_out);
		gpu_to_gpu_managed1st(mesh_out, (cuVEC<VType>*)mesh_out_arr, size_out);
		set_gpu_value(mesh_out_num, (int)size_out);
	}

	//------

	//extract sizes and cells boxes from meshes_in and meshes_out

	std::vector<cuSZ3> n_in;
	std::vector<cuBox*> pbox_d_in;
	std::vector<cuSZ3> n_out;
	std::vector<cuBox*> pbox_d_out;

	for (int idx = 0; idx < size_in; idx++) {

		n_in.push_back(meshes_in1[idx]->n);
		pbox_d_in.push_back(meshes_in1[idx]->devices_boxes());
	}

	//calculate maximum output mesh size on this device as this will be needed if we need to zero output meshes
	transfer_info.max_mesh_out_size = 0;

	for (int idx = 0; idx < size_out; idx++) {

		n_out.push_back(meshes_out[idx]->n);
		pbox_d_out.push_back(meshes_out[idx]->devices_boxes());
		transfer_info.max_mesh_out_size = (meshes_out[idx]->device_size(device_idx) > transfer_info.max_mesh_out_size ? meshes_out[idx]->device_size(device_idx) : transfer_info.max_mesh_out_size);
	}

	//------

	return copy_transfer_info(vec_transfer, n_in, pbox_d_in, n_out, pbox_d_out, n, pbox_d, device_idx, num_devices, transfer_info);
}

//MULTIPLE INPUTS, MULTIPLE OUTPUTS

//copy precalculated transfer info from cpu memory
template <typename VType>
template <typename cpuTransfer, typename MTypeIn, typename MTypeOut>
__host__ bool mcuTransfer<VType>::copy_transfer_info_averagedinputs_duplicatedoutputs(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in2,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out1,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out2,
	cpuTransfer& vec_transfer,
	cuSZ3 n, cuBox* pbox_d, int device_idx, int num_devices,
	mcuTransfer_Info& transfer_info)
{
	clear_transfer_data(transfer_info);

	size_t size_in = meshes_in1.size();
	size_t size_in2 = meshes_in2.size();

	size_t size_out = meshes_out1.size();
	size_t size_out2 = meshes_out2.size();

	if (size_in != size_in2 || size_out != size_out2) return false;

	//------

	cu_arr<cuVEC<VType>> mesh_in_arr1, mesh_in_arr2;
	cu_arr<cuVEC<VType>> mesh_out_arr1, mesh_out_arr2;

	//copy cuVECs on device with device_idx from meshes_in into mesh_in_arr
	for (int idx = 0; idx < size_in; idx++) {

		mesh_in_arr1.push_back((cuVEC<VType>*&)meshes_in1[idx]->get_managed_object(device_idx));
		mesh_in_arr2.push_back((cuVEC<VType>*&)meshes_in2[idx]->get_managed_object(device_idx));
	}

	//copy cuVECs on device with device_idx from meshes_out into mesh_out_arr
	for (int idx = 0; idx < size_out; idx++) {

		mesh_out_arr1.push_back((cuVEC<VType>*&)meshes_out1[idx]->get_managed_object(device_idx));
		mesh_out_arr2.push_back((cuVEC<VType>*&)meshes_out2[idx]->get_managed_object(device_idx));
	}

	//------

	//copy cuVEC pointers from mesh_in_arr to the managed mesh_in array for later usage
	if (size_in > 0) {

		gpu_alloc_managed(mesh_in, size_in);
		gpu_to_gpu_managed1st(mesh_in, (cuVEC<VType>*)mesh_in_arr1, size_in);

		gpu_alloc_managed(mesh_in2, size_in);
		gpu_to_gpu_managed1st(mesh_in2, (cuVEC<VType>*)mesh_in_arr2, size_in);
	}

	//copy cuVEC pointers from mesh_out_arr to the managed mesh_out array for later usage
	if (size_out > 0) {

		gpu_alloc_managed(mesh_out, size_out);
		gpu_to_gpu_managed1st(mesh_out, (cuVEC<VType>*)mesh_out_arr1, size_out);
		set_gpu_value(mesh_out_num, (int)size_out);

		gpu_alloc_managed(mesh_out2, size_out);
		gpu_to_gpu_managed1st(mesh_out2, (cuVEC<VType>*)mesh_out_arr2, size_out);
	}

	//------

	//extract sizes and cells boxes from meshes_in and meshes_out

	std::vector<cuSZ3> n_in;
	std::vector<cuBox*> pbox_d_in;
	std::vector<cuSZ3> n_out;
	std::vector<cuBox*> pbox_d_out;

	for (int idx = 0; idx < size_in; idx++) {

		n_in.push_back(meshes_in1[idx]->n);
		pbox_d_in.push_back(meshes_in1[idx]->devices_boxes());
	}

	//calculate maximum output mesh size on this device as this will be needed if we need to zero output meshes
	transfer_info.max_mesh_out_size = 0;

	for (int idx = 0; idx < size_out; idx++) {

		n_out.push_back(meshes_out1[idx]->n);
		pbox_d_out.push_back(meshes_out1[idx]->devices_boxes());
		transfer_info.max_mesh_out_size = (meshes_out1[idx]->device_size(device_idx) > transfer_info.max_mesh_out_size ? meshes_out1[idx]->device_size(device_idx) : transfer_info.max_mesh_out_size);
	}

	//------

	return copy_transfer_info(vec_transfer, n_in, pbox_d_in, n_out, pbox_d_out, n, pbox_d, device_idx, num_devices, transfer_info);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//AUXILIARY

template <typename VType, typename MType>
void mcuVEC<VType, MType>::clear_transfer(void)
{
	if (transfer.size()) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			if (transfer[mGPU].first) delete transfer[mGPU].first;
			transfer[mGPU].first = nullptr;
		}

		transfer.clear();
	}
}

template <typename VType, typename MType>
void mcuVEC<VType, MType>::clear_transfer2(void)
{
	if (transfer2.size()) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			if (transfer2[mGPU].first) delete transfer2[mGPU].first;
			transfer2[mGPU].first = nullptr;
		}

		transfer2.clear();
	}
}

template <typename VType, typename MType>
void mcuVEC<VType, MType>::make_transfer(void)
{
	if (transfer.size()) clear_transfer();

	transfer.resize(mGPU.get_num_devices(), std::pair<cu_obj<mcuTransfer<VType>>*, mcuTransfer_Info>(nullptr, mcuTransfer_Info()));
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		transfer[mGPU].first = new cu_obj<mcuTransfer<VType>>();
	}
}

template <typename VType, typename MType>
void mcuVEC<VType, MType>::make_transfer2(void)
{
	if (transfer2.size()) clear_transfer2();

	transfer2.resize(mGPU.get_num_devices(), std::pair<cu_obj<mcuTransfer<VType>>*, mcuTransfer_Info>(nullptr, mcuTransfer_Info()));
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		transfer2[mGPU].first = new cu_obj<mcuTransfer<VType>>();
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Copy transfer info

//SINGLE INPUT, SINGLE OUTPUT

template <typename VType, typename MType>
template <typename MTypeIn, typename MTypeOut, typename cpuTransfer>
bool mcuVEC<VType, MType>::copy_transfer_info(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out,
	cpuTransfer& vec_transfer)
{
	make_transfer();

	bool success = true;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		success &= (*transfer[mGPU].first)()->copy_transfer_info(
			meshes_in, 
			meshes_out, 
			vec_transfer, 
			n, pbox_d, mGPU, mGPU.get_num_devices(), 
			transfer[mGPU].second);
	}

	return success;
}

//same but for secondary mesh transfer
template <typename VType, typename MType>
template <typename MTypeIn, typename MTypeOut, typename cpuTransfer>
bool mcuVEC<VType, MType>::copy_transfer2_info(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out,
	cpuTransfer& vec_transfer)
{
	make_transfer2();

	bool success = true;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		success &= (*transfer2[mGPU].first)()->copy_transfer_info(
			meshes_in, 
			meshes_out, 
			vec_transfer, 
			n, pbox_d, mGPU, mGPU.get_num_devices(), 
			transfer2[mGPU].second);
	}

	return success;
}

//MULTIPLE INPUTS, SINGLE OUTPUT

//copy pre-calculated transfer info from cpu memory. return false if not enough memory to copy
//meshes_in1 and meshes_in2 vectors must have same sizes
//All mcuVECs in meshes_in1 should be non-empty
//Some mcuVECs in meshes_in2 allowed to be empty (in this case single input is used), but otherwise should have exactly same dimensions as the corresponding mcuVECs in meshes_in1
template <typename VType, typename MType>
template <typename MTypeIn, typename MTypeOut, typename cpuTransfer>
bool mcuVEC<VType, MType>::copy_transfer_info_averagedinputs(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in2,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out, cpuTransfer& vec_transfer)
{
	make_transfer();

	bool success = true;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		success &= (*transfer[mGPU].first)()->copy_transfer_info_averagedinputs(
			meshes_in1, meshes_in2, 
			meshes_out, 
			vec_transfer,
			n, pbox_d, mGPU, mGPU.get_num_devices(), 
			transfer[mGPU].second);
	}

	return success;
}

//same but for secondary mesh transfer
template <typename VType, typename MType>
template <typename MTypeIn, typename MTypeOut, typename cpuTransfer>
bool mcuVEC<VType, MType>::copy_transfer2_info_averagedinputs(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in2,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out, cpuTransfer& vec_transfer)
{
	make_transfer2();

	bool success = true;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		success &= (*transfer2[mGPU].first)()->copy_transfer_info_averagedinputs(
			meshes_in1, meshes_in2, 
			meshes_out, 
			vec_transfer, 
			n, pbox_d, mGPU, mGPU.get_num_devices(), 
			transfer2[mGPU].second);
	}

	return success;
}

template <typename VType, typename MType>
template <typename MTypeIn, typename MTypeInR, typename MTypeOut, typename cpuTransfer>
bool mcuVEC<VType, MType>::copy_transfer_info_multipliedinputs(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
	const std::vector<mcu_obj<MTypeInR, mcuVEC<cuBReal, MTypeInR>>*>& meshes_in2,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out, cpuTransfer& vec_transfer)
{
	make_transfer();

	bool success = true;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		success &= (*transfer[mGPU].first)()->copy_transfer_info_multipliedinputs(
			meshes_in1, meshes_in2, 
			meshes_out, 
			vec_transfer, 
			n, pbox_d, mGPU, mGPU.get_num_devices(), 
			transfer[mGPU].second);
	}

	return success;
}

//same but for secondary mesh transfer
template <typename VType, typename MType>
template <typename MTypeIn, typename MTypeInR, typename MTypeOut, typename cpuTransfer>
bool mcuVEC<VType, MType>::copy_transfer2_info_multipliedinputs(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
	const std::vector<mcu_obj<MTypeInR, mcuVEC<cuBReal, MTypeInR>>*>& meshes_in2,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out, cpuTransfer& vec_transfer)
{
	make_transfer2();

	bool success = true;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		success &= (*transfer2[mGPU].first)()->copy_transfer_info_multipliedinputs(
			meshes_in1, meshes_in2, 
			meshes_out, 
			vec_transfer, 
			n, pbox_d, mGPU, mGPU.get_num_devices(), 
			transfer2[mGPU].second);
	}

	return success;
}

//MULTIPLE INPUTS, MULTIPLE OUTPUT

//copy pre-calculated transfer info from cpu memory. return false if not enough memory to copy
//meshes_in1 and meshes_in2 vectors must have same sizes; same for meshes_out1, meshes_out2
//All mcuVECs in meshes_in1 and meshes_out1 should be non-empty
//Some mcuVECs in meshes_in2 and meshes_out2 allowed to be empty (in this single input/output is used), but otherwise should have exactly same dimensions as the corresponding mcuVECs in meshes_in1, meshes_out1
//Also if a mcuVEC in meshes_in2 is non-empty the corresponding VEC in meshes_out2 should also be non-empty.
template <typename VType, typename MType>
template <typename MTypeIn, typename MTypeOut, typename cpuTransfer>
bool mcuVEC<VType, MType>::copy_transfer_info_averagedinputs_duplicatedoutputs(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in2,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out1,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out2,
	cpuTransfer& vec_transfer)
{
	make_transfer();

	bool success = true;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		success &= (*transfer[mGPU].first)()->copy_transfer_info_averagedinputs_duplicatedoutputs(
			meshes_in1, meshes_in2, 
			meshes_out1, meshes_out2, 
			vec_transfer, 
			n, pbox_d, mGPU, mGPU.get_num_devices(),
			transfer[mGPU].second);
	}

	return success;
}

//same but for secondary mesh transfer
template <typename VType, typename MType>
template <typename MTypeIn, typename MTypeOut, typename cpuTransfer>
bool mcuVEC<VType, MType>::copy_transfer2_info_averagedinputs_duplicatedoutputs(
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in1,
	const std::vector<mcu_obj<MTypeIn, mcuVEC<VType, MTypeIn>>*>& meshes_in2,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out1,
	const std::vector<mcu_obj<MTypeOut, mcuVEC<VType, MTypeOut>>*>& meshes_out2,
	cpuTransfer& vec_transfer)
{
	make_transfer2();

	bool success = true;

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		success &= (*transfer2[mGPU].first)()->copy_transfer_info_averagedinputs_duplicatedoutputs(
			meshes_in1, meshes_in2, 
			meshes_out1, meshes_out2, 
			vec_transfer, 
			n, pbox_d, mGPU, mGPU.get_num_devices(),
			transfer2[mGPU].second);
	}

	return success;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Do transfers

//SINGLE INPUT, SINGLE OUTPUT

//do the actual transfer of values to and from this mesh using these. with clear_input true then set values in this mesh, otherwise add to it.
template <typename VType, typename MType>
void mcuVEC<VType, MType>::transfer_in(bool clear_input)
{
	//first zero smesh quantity as we'll be adding in values from meshes
	if (clear_input) set(VType());

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*transfer[mGPU].first)()->transfer_in(mng.get_deviceobject(mGPU), transfer[mGPU].second, 0);
	}
}

template <typename VType, typename MType>
void mcuVEC<VType, MType>::transfer2_in(bool clear_input)
{
	//first zero smesh quantity as we'll be adding in values from meshes
	if (clear_input) set(VType());

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*transfer2[mGPU].first)()->transfer_in(mng.get_deviceobject(mGPU), transfer2[mGPU].second, 0);
	}
}

//transfer to output meshes. with clear_output true then set values in output meshes, otherwise add to them
template <typename VType, typename MType>
void mcuVEC<VType, MType>::transfer_out(bool clear_output)
{
	if (clear_output) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			(*transfer[mGPU].first)()->zero_output_meshes(transfer[mGPU].second);
		}
	}

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*transfer[mGPU].first)()->transfer_out(mng.get_deviceobject(mGPU), transfer[mGPU].second, 0);
	}
}

template <typename VType, typename MType>
void mcuVEC<VType, MType>::transfer2_out(bool clear_output)
{
	if (clear_output) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			(*transfer2[mGPU].first)()->zero_output_meshes(transfer2[mGPU].second);
		}
	}

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*transfer2[mGPU].first)()->transfer_out(mng.get_deviceobject(mGPU), transfer2[mGPU].second, 0);
	}
}

//AVERAGED INPUT

template <typename VType, typename MType>
void mcuVEC<VType, MType>::transfer_in_averaged(bool clear_input)
{
	//first zero smesh quantity as we'll be adding in values from meshes
	if (clear_input) set(VType());

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*transfer[mGPU].first)()->transfer_in(mng.get_deviceobject(mGPU), transfer[mGPU].second, 1);
	}
}

template <typename VType, typename MType>
void mcuVEC<VType, MType>::transfer2_in_averaged(bool clear_input)
{
	//first zero smesh quantity as we'll be adding in values from meshes
	if (clear_input) set(VType());

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*transfer2[mGPU].first)()->transfer_in(mng.get_deviceobject(mGPU), transfer2[mGPU].second, 1);
	}
}

//MULTIPLIED INPUTS

template <typename VType, typename MType>
void mcuVEC<VType, MType>::transfer_in_multiplied(bool clear_input)
{
	//first zero smesh quantity as we'll be adding in values from meshes
	if (clear_input) set(VType());

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*transfer[mGPU].first)()->transfer_in(mng.get_deviceobject(mGPU), transfer[mGPU].second, 2);
	}
}

template <typename VType, typename MType>
void mcuVEC<VType, MType>::transfer2_in_multiplied(bool clear_input)
{
	//first zero smesh quantity as we'll be adding in values from meshes
	if (clear_input) set(VType());

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*transfer2[mGPU].first)()->transfer_in(mng.get_deviceobject(mGPU), transfer2[mGPU].second, 2);
	}
}

//DUPLICATED OUTPUT

template <typename VType, typename MType>
void mcuVEC<VType, MType>::transfer_out_duplicated(bool clear_output)
{
	if (clear_output) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			(*transfer[mGPU].first)()->zero_output_duplicated_meshes(transfer[mGPU].second);
		}
	}

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*transfer[mGPU].first)()->transfer_out(mng.get_deviceobject(mGPU), transfer[mGPU].second, 1);
	}
}

template <typename VType, typename MType>
void mcuVEC<VType, MType>::transfer2_out_duplicated(bool clear_output)
{
	if (clear_output) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			(*transfer2[mGPU].first)()->zero_output_duplicated_meshes(transfer2[mGPU].second);
		}
	}

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		(*transfer2[mGPU].first)()->transfer_out(mng.get_deviceobject(mGPU), transfer2[mGPU].second, 1);
	}
}