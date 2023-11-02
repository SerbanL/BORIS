#pragma once

#include "cupArray.h"
#include "mGPU.h"

//mcu_parr is used to manage multiple cu_parr objects, one for each gpu available

template <typename PType>
class mcu_parr
{

private:

	//used to iterate over configured devices, so user doesn't have to worry about selecting them, or which device numbers have been set.
	mGPUConfig& mGPU;

	//contained cu_parr objects, one for each gpu configured
	cu_parr<PType> ** pcuparr = nullptr;

public:


	//------------------------------------------- CONSTRUCTOR

	//void constructor
	mcu_parr(mGPUConfig& mGPU_) :
		mGPU(mGPU_)
	{
		//array of pointers set to number of required gpus
		pcuparr = new cu_parr<PType>*[mGPU.get_num_devices()];

		//construct each cu_arr on the respective device
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			pcuparr[mGPU] = nullptr;
			pcuparr[mGPU] = new cu_parr<PType>();
		}
	}

	//size constructor (each cu_arr gets same size)
	mcu_parr(mGPUConfig& mGPU_, size_t size) :
		mGPU(mGPU_)
	{
		//array of pointers set to number of required gpus
		pcuparr = new cu_parr<PType>*[mGPU.get_num_devices()];

		//construct each cu_arr on the respective device
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			pcuparr[mGPU] = nullptr;
			pcuparr[mGPU] = new cu_parr<PType>(size);
		}
	}

	//------------------------------------------- DESTRUCTOR

	//destructor
	~mcu_parr()
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			delete pcuparr[mGPU];
			pcuparr[mGPU] = nullptr;
		}

		delete [] pcuparr;
		pcuparr = nullptr;
	}

	//------------------------------------------- GET ARRAY

	//mcu_parr<PType> parr;
	//if we have __global__ void cuda_kernel(size_t size, PType** parr); then launch kernel as:
	//cuda_kernel<<<...>>>(parr.size(), parr(mGPU)); 
	PType**& operator()(int mGPU_idx) { return *pcuparr[mGPU_idx]; }

	//as above but with function call
	PType**& get_array(int mGPU_idx)
	{
		return pcuparr[mGPU_idx]->get_array();
	}

	//get contained cu_parr for indexed device
	cu_parr<PType>& get_cu_parr(int mGPU_idx)
	{
		return *pcuparr[mGPU_idx];
	}

	//------------------------------------------- INDEXING

	//done on each cu_parr separately after passing to kernel

	//------------------------------------------- RESIZING : mcupArray_sizing.h

	//resize all devices to given size
	bool resize(size_t size)
	{
		bool success = true;

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			success &= pcuparr[mGPU]->resize(size);
		}

		return success;
	}

	//clear for all devices
	void clear(void) 
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			pcuparr[mGPU]->clear();
		}
	}

	//------------------------------------------- STORE ENTRIES

	//new_entry is a pointer in cpu memory to an object in gpu memory. add it to indexed cu_parr
	//must be used as part of the usual mGPU for loop construct to select device before
	void push_back(int mGPU_idx, PType*& new_entry) { pcuparr[mGPU_idx]->push_back(new_entry); }

	//new_entry at given index is a pointer in cpu memory to an object in gpu memory. add it to indexed cu_parr (mGPU_idx). correct size must be allocated already.
	//must be used as part of the usual mGPU for loop construct to select device before
	void set(int mGPU_idx, int index, PType*& new_entry) { pcuparr[mGPU_idx]->set(index, new_entry); }

	//------------------------------------------- GET SIZE

	//get size of array on indexed device
	size_t size(int mGPU_idx) { return pcuparr[mGPU_idx]->size(); }

	//get total size (i.e. sum of all allocated device sizes)
	size_t size(void)
	{
		size_t total_size = 0;
		for (int idx = 0; idx < mGPU.get_num_devices(); idx++) total_size += pcuparr[idx]->size();
		return total_size;
	}

	//--------------------
};