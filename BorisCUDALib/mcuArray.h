#pragma once

#include "cuArray.h"
#include "mGPU.h"

//mcu_arr is used to manage multiple cu_arr objects, one for each gpu available

//Some special cases:
// 
//See EXAMPLE 6 usage in cuArray.h. The same can be achieved with a mcu_arr as follows:
//
//If we have (in addition to cuObj we require the usual Policy class for it):
//mcu_obj<cuObj, cuObj_Policy> mcuObj1(mGPU);
//mcu_obj<cuObj, cuObj_Policy> mcuObj2(mGPU);

//mcu_arr<cuObj> mcuObj_arr(mGPU);
//
//for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
//
//	mcuObj_arr.push_back(mGPU, mcuObj1.get_managed_object(mGPU));
//	mcuObj_arr.push_back(mGPU, mcuObj2.get_managed_object(mGPU));
//}
//If we have __global__ void cuda_kernel(cuObj* pcuObj_arr, size_t arr_size);
//Then pass it into the kernel as follows:
//
//for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
//
//	func_kernel <<<1, 1 >>> (mcuObj_arr(mGPU), mcuObj_arr.size(mGPU));
//}
//
//Now each cuda_kernel function on each respective device will have access to an array of cuObj objects for each respective device

template <typename VType>
class mcu_arr
{

private:

	//used to iterate over configured devices, so user doesn't have to worry about selecting them, or which device numbers have been set.
	mGPUConfig& mGPU;

	//contained cu_arr objects, one for each gpu configured
	cu_arr<VType> ** pcuarr = nullptr;

public:


	//------------------------------------------- CONSTRUCTOR

	//void constructor
	mcu_arr(mGPUConfig& mGPU_) :
		mGPU(mGPU_)
	{
		//array of pointers set to number of required gpus
		pcuarr = new cu_arr<VType>*[mGPU.get_num_devices()];

		//construct each cu_arr on the respective device
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			pcuarr[mGPU] = nullptr;
			pcuarr[mGPU] = new cu_arr<VType>();
		}
	}

	//size constructor (each cu_arr gets same size)
	mcu_arr(mGPUConfig& mGPU_, size_t size) :
		mGPU(mGPU_)
	{
		//array of pointers set to number of required gpus
		pcuarr = new cu_arr<VType>*[mGPU.get_num_devices()];

		//construct each cu_arr on the respective device
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			pcuarr[mGPU] = nullptr;
			pcuarr[mGPU] = new cu_arr<VType>(size);
		}
	}

	//------------------------------------------- DESTRUCTOR

	//destructor
	~mcu_arr()
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			delete pcuarr[mGPU];
			pcuarr[mGPU] = nullptr;
		}

		delete [] pcuarr;
		pcuarr = nullptr;
	}

	//------------------------------------------- GET ARRAY

	//mcu_arr<float> arr;
	//if we have __global__ void cuda_kernel(size_t size, float* arr); then launch kernel as:
	//cuda_kernel<<<...>>>(arr.size(mGPU), arr(mGPU)); 
	VType*& operator()(int mGPU_idx) { return *pcuarr[mGPU_idx]; }

	//get gpu address of stored array in cpu memory, i.e. get_managed_array() returns a pointer stored in cpu memory, which contains an address to gpu memory, such that *get_managed_array() is the stored array in gpu memory
	//This is useful to build a cu_arr of arrays.
	//Example if we have cu_arr<float> arr1, cu_arr<float*> arr_col, then you can use arr_col.push_back(arr1.get_managed_array()); arr_col can now be passed to a __global__, where arr_col[0][0] accesses first element in arr1, etc.
	VType**& get_managed_array(int mGPU_idx)
	{
		return pcuarr[mGPU_idx]->get_managed_array();
	}

	//get gpu address of stored array in cpu memory directly; this can be used with thrust device pointers, e.g. thrust::device_ptr<VType> dev_ptr(arr.get_array());
	VType*& get_array(int mGPU_idx)
	{
		return pcuarr[mGPU_idx]->get_array();
	}

	//get contained cu_arr for indexed device
	cu_arr<VType>& get_cu_arr(int mGPU_idx)
	{
		return *pcuarr[mGPU_idx];
	}

	//------------------------------------------- INDEXING

	//done on each cu_arr separately after passing to kernel

	//------------------------------------------- RESIZING : mcuArray_sizing.h

	//resize all devices to given size
	bool resize(size_t size);

	//resize indexed device to given size
	//must be used as part of the usual mGPU for loop construct to select device before
	bool resize(int mGPU_idx, size_t size);

	//clear for all devices
	void clear(void);

	//clear for indexed device only
	//must be used as part of the usual mGPU for loop construct to select device before
	void clear(int mGPU_idx);

	//------------------------------------------- STORE ENTRIES : mcuArray_sizing.h, mcuArray_sizing.cuh

	//new_entry is a pointer in cpu memory to an object in gpu memory. add it to indexed cu_arr
	//must be used as part of the usual mGPU for loop construct to select device before
	void push_back(int mGPU_idx, VType*& new_entry);

	//new_entry at given index is a pointer in cpu memory to an object in gpu memory. add it to indexed cu_arr (mGPU_idx). correct size must be allocated already.
	//must be used as part of the usual mGPU for loop construct to select device before
	void set(int mGPU_idx, int index, VType*& new_entry) { pcuarr[mGPU_idx]->set(index, new_entry); }

	//deep copy version of push_back and set, which must be called in cu files
	//these replace the gpu_to_gpu calls when storing entries with gpu_to_gpu_deep calls (see comments for gpu_to_gpu_deep)
	//new_entry is stored in gpu memory
	void push_back_deepcopy(int mGPU_idx, VType& new_entry);
	void set_deepcopy(int mGPU_idx, int index, VType& new_entry);

	//------------------------------------------- GPU <-> CPU TRANSFER : mcuArray_transfer.h

	//copy values from a std::vector into gpu memory. Doesn't set size, but copies up to currently allocated size. Copy to all configured devices.
	template <typename Type>
	void copy_from_vector(std::vector<Type>& cpuvec);

	//copy values to a std::vector into cpu memory. Doesn't set size, but copies up to currently allocated size.
	//Copy by putting all data from devices head to head in order of devices.
	template <typename Type>
	void copy_to_vector(std::vector<Type>& cpuvec);

	//------------------------------------------- SET VALUE : mcuArray_aux.h

	//set all entries to given value, for all devices
	void set(VType value);

	//set all entries to given value, for indexed device
	//must be used as part of the usual mGPU for loop construct to select device before
	void set(int idx, VType value);

	//set single value from cpu memory at given index for all devices configured
	void setvalue(int idx, VType value);

	//------------------------------------------- GET SIZE : mcuArray_aux.h

	//get size of array on indexed device
	size_t size(int mGPU_idx);

	//get total size (i.e. sum of all allocated device sizes)
	size_t size(void);

	//--------------------
};