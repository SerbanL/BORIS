#pragma once

#include <cuda_runtime.h>
#include <vector>

//EXAMPLE :

//To iterate over devices:

//for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {
//
// ... //mGPU is convertible to a linear device index
//}

//multi-GPU memory transfer configuration (p2p or via host)
//also used to iterate over devices
class mGPUConfig
{
private:

	//----------------- Memory transfer data

	//set to true if all memory transfers are p2p (if all P2P then UVA can also be used) - query with is_all_p2p()
	bool all_p2p = false;

	//special mode for testing : disable p2p access between all devices
	bool disable_p2p = false;

	//special mode : disable UVA. set to false to allow UVA if all_p2p - query with is_uva_enabled()
	bool disable_uva = false;

	//matrix to indicate type of transfer between any 2 devices. 0 : indirect (through host), 1 : p2p
	//NOTE : if all transfers are p2p, then clear this vector and set all_p2p = true
	//NOTE : first index is for device to transfer TO, second index is for device to transfer FROM
	int** transfer_type = nullptr;

	//----------------- Special configuration

	//mcuVEC configuration
	// 
	//use this to force allocation of sub-vec rectangles along an indicated axis:
	//0 : x, 1 : y, 2 : z, -1 : automatic (along longest axis - see get_devices_n_values)
	int subvec_axis = 0;

	//when transferring data between devices can choose to convert to half-precision first in order to reduce transfer size
	bool halfprecision_transfer = false;

	//----------------- Device iterator data

	int iterator_idx = 0;

	//configured device numbers this object manages - must ensure devices are actually available before configuring this object
	//index this to get the device number to set with cudaSetDevice, and nothing else: DO NOT USE pdevices[idx] AS AN INDEX!
	int* pdevices = nullptr;
	int num_devices = 0;

private:

	//cleanup all memory allocation, bringing object back to empty state
	void clear(void)
	{
		if (pdevices) {

			delete[] pdevices;
			pdevices = nullptr;
		}

		if (transfer_type) {

			for (int idx = 0; idx < num_devices; idx++) {

				if (transfer_type[idx]) delete[] transfer_type[idx];
				transfer_type[idx] = nullptr;
			}

			delete[] transfer_type;
			transfer_type = nullptr;
		}
	}

	void configure_devices_aux(std::vector<int> devices)
	{
		num_devices = devices.size();

		//store devices configured
		pdevices = new int[num_devices];
		std::copy(devices.begin(), devices.end(), pdevices);
		
		//transfer_type size : number of devices squared
		transfer_type = new int*[num_devices];
		for (int idx = 0; idx < num_devices; idx++) transfer_type[idx] = new int[num_devices];

		all_p2p = !disable_p2p;

		for (int idx = 0; idx < num_devices; idx++) {

			cudaSetDevice(pdevices[idx]);
			cudaDeviceReset();
		}

		for (int idx_device_to = 0; idx_device_to < num_devices; idx_device_to++) {

			//the current device
			cudaSetDevice(pdevices[idx_device_to]);

			//check all other devices, which this current device can access in p2p mode
			for (int idx_device_from = 0; idx_device_from < num_devices; idx_device_from++) {

				transfer_type[idx_device_to][idx_device_from] = 1;
				if (pdevices[idx_device_to] == pdevices[idx_device_from]) continue;

				if (disable_p2p) transfer_type[idx_device_to][idx_device_from] = 0;
				{
					int can_access_peer;
					cudaDeviceCanAccessPeer(&can_access_peer, pdevices[idx_device_to], pdevices[idx_device_from]);

					if (can_access_peer == 0) {

						all_p2p = false;
						transfer_type[idx_device_to][idx_device_from] = 0;
					}
					//enable p2p from current device to device_from
					else cudaDeviceEnablePeerAccess(pdevices[idx_device_from], 0);
				}
			}
		}

		//if all p2p (ideal case) then transfer_type matrix not needed
		if (all_p2p) {

			for (int idx = 0; idx < num_devices; idx++) {

				if (transfer_type[idx]) delete[] transfer_type[idx];
				transfer_type[idx] = nullptr;
			}

			delete[] transfer_type;
			transfer_type = nullptr;
		}
	}

public:

	/////////////CONSTRUCTORS

	mGPUConfig(void) {}

	mGPUConfig(std::vector<int> devices) { configure_devices_aux(devices); }

	~mGPUConfig() { clear(); }

	/////////////CONFIGURATION CHANGE

	bool configuration_already_set(std::vector<int> devices)
	{
		//if no devices configured, then don't change current configuration
		if (!devices.size()) return true;

		//first make sure configuration doesn't exist already
		if (num_devices == devices.size()) {

			for (int device_idx = 0; device_idx < devices.size(); device_idx++) {

				if (pdevices[device_idx] != devices[device_idx]) return false;
			}

			return true;
		}
		else return false;
	}

	//NOTE : if changing devices configuration, first delete all objects which use this configuration, then remake them after configuring new devices
	void configure_devices(std::vector<int> devices)
	{
		if (!configuration_already_set(devices)) {

			clear();
			configure_devices_aux(devices);
		}
	}

	//change modes : P2P and UVA
	void set_disable_p2p(bool status)
	{
		if (status != disable_p2p) {

			disable_p2p = status;

			//must recalculate value of all_p2p and transfer_type matrix
			if (num_devices) {

				std::vector<int> devices(pdevices, pdevices + num_devices);

				clear();
				configure_devices_aux(devices);
			}
		}
	}

	void set_disable_uva(bool status) { disable_uva = status; }

	//force allocation of subvec rectangles along a given axis.
	//when this is changed CUDA should be reset, i.e. turn off, set subvec axis, turn back on again (requires clean memory re-allocation for all mcuVECs)
	void set_subvec_axis(int subvec_axis_) { subvec_axis = subvec_axis_; }
	int get_subvec_axis(void) { return subvec_axis; }

	void set_halfprecision_transfer(bool halfprecision_transfer_) { halfprecision_transfer = halfprecision_transfer_; }
	bool get_halfprecision_transfer(void) { return halfprecision_transfer; }

	/////////////ITERATOR

	//start iterator
	mGPUConfig& device_begin(void)
	{
		iterator_idx = 0;
		//if only one device configured, don't need to keep selecting it
		if (num_devices > 1) cudaSetDevice(pdevices[iterator_idx]);
		return *this;
	}

	//check for last device
	int device_end(void) const { return num_devices; }

	//check iterator index
	bool operator!=(int value) { return iterator_idx != value; }

	//increment iterator
	mGPUConfig& operator++(int)
	{
		iterator_idx++;
		if (iterator_idx < num_devices) cudaSetDevice(pdevices[iterator_idx]);
		return *this;
	}

	/////////////INFO ITERATOR

	//number of devices available
	int get_num_devices(void) const { return num_devices; }

	//conversion operator to get linear device index (iterator value)
	operator int() const { return iterator_idx; }

	/////////////INFO MEMORY TRANSFER

	//check for a particular device combination if p2p is enabled
	bool is_p2p(int device_to, int device_from)
	{
		if (all_p2p) return true;
		else if (disable_p2p) return false;
		else return transfer_type[device_to][device_from];
	}

	bool is_all_p2p(void) { return all_p2p; }

	//UVA can be used if all devices are P2P, and UVA not explicitly disabled (disabled by default)
	bool is_uva_enabled(void) { return all_p2p && !disable_uva; }

	/////////////MANUAL DEVICE SELECTION

	//select a particular device with an index from 0 to num_devices
	void select_device(int idx) const { cudaSetDevice(pdevices[idx]); }

	bool select_previous_device(void) const
	{
		if (iterator_idx > 0) cudaSetDevice(pdevices[iterator_idx - 1]);
		else return false;
		return true;
	}

	bool select_next_device(void) const
	{
		if (iterator_idx < num_devices - 1) cudaSetDevice(pdevices[iterator_idx + 1]);
		else return false;
		return true;
	}

	void select_current_device(void) const { cudaSetDevice(pdevices[iterator_idx]); }

	/////////////AUXILIARY
	
	//synchronize all configured devices even if there's just one 
	//this will typically be used before and after a block of code involving asynchronous memory transfers or kernels accessing memory on other devices through UVA
	void synchronize(void)
	{
		for (int idx = 0; idx < num_devices; idx++) {

			cudaSetDevice(pdevices[idx]);
			cudaDeviceSynchronize();
		}
	}

	//similar to synchronize, but only done if more than one device present
	void synchronize_if_multi(void)
	{
		if (num_devices > 1) synchronize();
	}

	//similar to synchronize, but only done if there's more than 1 device and UVA being used  (e.g. as opposed to halo exchanges)
	//typically used before and after kernels accessing memory on other devices through UVA
	void synchronize_if_uva(void)
	{
		if (num_devices > 1 && disable_uva == false) synchronize();
	}
};