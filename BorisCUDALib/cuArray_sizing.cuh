#pragma once

#include "cuArray.h"

#include "alloc_cpy.cuh"

//------------------------------------------- STORE NEW ENTRIES : cuArray_sizing.cuh

//new_entry is a pointer in cpu memory to an object in gpu memory
template <typename VType>
__host__ void cu_arr<VType>::push_back_deepcopy(VType& new_entry)
{
	//allocate new memory size in a temporary array
	VType* new_array = nullptr;
	cudaError_t error = gpu_alloc(new_array, arr_size + 1);

	if (error != cudaSuccess) {

		gpu_free(new_array);
		return;
	}

	//copy data currently in array to temporary array (if any)
	if (arr_size > 0) {

		gpu_to_gpu(new_array, cu_array, arr_size);
	}

	//add new entry to end of temporary array
	gpu_to_gpu_deepcopy(new_array, new_entry, arr_size);

	//swap pointers so array now points to newly constructed memory
	gpu_swap(cu_array, new_array);

	//free old memory
	gpu_free(new_array);

	//set new size
	arr_size++;

	if (!pcu_array) gpu_alloc(pcu_array);
	cpu_to_gpu(pcu_array, &cu_array);
}

template <typename VType>
__host__ void cu_arr<VType>::set_deepcopy(int index, VType& new_entry)
{
	gpu_to_gpu_deepcopy(cu_array, new_entry, index);
}