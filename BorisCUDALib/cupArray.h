#pragma once

#include "alloc_cpy.h"

//cu_parr is used to manage an array of pointers in gpu memory
//should not use cu_arr<PType*> for this purpose as it will not work as intended, instead use cu_parr<PType>.

template <typename PType>
class cu_parr
{

private:

	//array in gpu memory : cu_parray itself is stored in cpu memory, but contains pointers in gpu memory
	PType** cu_parray;

	//size value stored in cpu memory
	size_t arr_size;

private:

public:

	//------------------------------------------- CONSTRUCTOR

	//void constructor
	__host__ cu_parr(void)
	{
		cu_parray = nullptr;
		arr_size = 0;
	}

	//size constructor
	__host__ cu_parr(size_t size_)
	{
		cu_parray = nullptr;
		arr_size = 0;

		resize(size_);
	}

	//------------------------------------------- DESTRUCTOR

	//destructor
	__host__ ~cu_parr()
	{
		gpu_free(cu_parray);
	}

	//------------------------------------------- GET ARRAY

	//cu_parr<PType> parr;
	//if we have __global__ void cuda_kernel(size_t size, PType** parr); then launch kernel as:
	//cuda_kernel<<<...>>>(parr.size(), parr); 
	__host__ operator PType**&() { return cu_parray; }

	//as above but with function call
	__host__ PType**& get_array(void)
	{
		return cu_parray;
	}

	//------------------------------------------- INDEXING

	__device__ PType* operator[](int idx) { return cu_parray[idx]; }

	//------------------------------------------- RESIZING : cupArray_sizing.h

	__host__ bool resize(size_t size_)
	{
		if (size_ == size()) return true;

		if (!size_) clear();
		else {

			cudaError_t error = gpu_alloc(cu_parray, size_);
			if (error == cudaSuccess) {

				arr_size = size_;
				return true;
			}
			else {

				clear();
				return false;
			}
		}

		return true;
	}

	__host__ void clear(void)
	{
		gpu_free(cu_parray);
		cu_parray = nullptr;

		arr_size = 0;
	}

	//------------------------------------------- STORE ENTRIES

	//new_entry is a pointer in cpu memory to an object in gpu memory
	__host__ void push_back(PType*& new_entry)
	{
		//allocate new memory size in a temporary array
		PType** new_array = nullptr;
		cudaError_t error = gpu_alloc(new_array, arr_size + 1);

		if (error != cudaSuccess) {

			gpu_free(new_array);
			return;
		}

		//copy data currently in array to temporary array (if any)
		if (arr_size > 0) {

			gpu_to_gpu(new_array, cu_parray, arr_size);
		}

		//add new entry to end of temporary array
		cpu_to_gpu(new_array + arr_size, &new_entry);

		//swap pointers so array now points to newly constructed memory
		gpu_swap(cu_parray, new_array);

		//free old memory
		gpu_free(new_array);

		//set new size
		arr_size++;
	}

	//new_entry at given index is a pointer in cpu memory to an object in gpu memory. correct size must be allocated already.
	__host__ void set(int index, PType*& new_entry) { cpu_to_gpu(cu_parray + index, &new_entry); }

	//------------------------------------------- GET SIZE

	__host__ size_t size(void) { return arr_size; }

	//--------------------
};