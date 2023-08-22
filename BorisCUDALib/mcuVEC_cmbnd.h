#pragma once

#include "cuVEC_VC_cmbnd.h"

//mCMBNDInfo describes a composite media boundary contact between 2 meshes of same type, used to calculate values at CMBND cells using boundary conditions
//Differs from CMBNDInfoCUDA (which is used for single GPU computations) as:
//mCMBNDInfoCUDA holds a CMBNDInfoCUDA cuda for each device, which describes the contact between the sub-cuVEC on each device (primary one managed here), and the entire logical cuVEC to which it is in contact (the secondary one, which will be spread across multiple devices)
struct mCMBNDInfoCUDA {

	//------------------------------------- DATA

	//multi-GPU configuration (list of physical devices configured with memory transfer type configuration set)
	mGPUConfig& mGPU;

	//cu-obj managed CMBNDInfoCUDA in this contact, one for each device
	cu_obj<CMBNDInfoCUDA>** ppCMBNDInfoCUDA = nullptr;

	//cells box for each device, but held in cpu memory
	cuBox* pcells_box = nullptr;

	//contact size for each device, i.e. this is cells_box.size().dim() but held in cpu memory
	size_t* pcontact_size = nullptr;

	//------------------------------------- CTOR/DTOR

	mCMBNDInfoCUDA(mGPUConfig& mGPU_) :
		mGPU(mGPU_)
	{
		ppCMBNDInfoCUDA = new cu_obj<CMBNDInfoCUDA>*[mGPU_.get_num_devices()];
		pcontact_size = new size_t[mGPU_.get_num_devices()];
		pcells_box = new cuBox[mGPU_.get_num_devices()];

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			ppCMBNDInfoCUDA[mGPU] = new cu_obj<CMBNDInfoCUDA>();
			pcontact_size[mGPU] = 0;
			pcells_box[mGPU] = cuBox();
		}
	}

	mCMBNDInfoCUDA(const mCMBNDInfoCUDA& copyThis) :
		mGPU(copyThis.mGPU)
	{
		ppCMBNDInfoCUDA = new cu_obj<CMBNDInfoCUDA>*[mGPU.get_num_devices()];
		pcontact_size = new size_t[mGPU.get_num_devices()];
		pcells_box = new cuBox[mGPU.get_num_devices()];

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			ppCMBNDInfoCUDA[mGPU] = new cu_obj<CMBNDInfoCUDA>();
			pcontact_size[mGPU] = 0;
			pcells_box[mGPU] = cuBox();
		}

		*this = copyThis;
	}

	mCMBNDInfoCUDA& operator=(const mCMBNDInfoCUDA& copyThis)
	{
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			*ppCMBNDInfoCUDA[mGPU] = *copyThis.ppCMBNDInfoCUDA[mGPU];
			pcontact_size[mGPU] = copyThis.pcontact_size[mGPU];
			pcells_box[mGPU] = copyThis.pcells_box[mGPU];
		}

		return *this;
	}

	~mCMBNDInfoCUDA()
	{
		if (ppCMBNDInfoCUDA) {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				delete ppCMBNDInfoCUDA[mGPU];
				ppCMBNDInfoCUDA[mGPU] = nullptr;
			}

			delete[] ppCMBNDInfoCUDA;
			ppCMBNDInfoCUDA = nullptr;
		}

		if (pcontact_size) delete[] pcontact_size;
		pcontact_size = nullptr;

		if (pcells_box) delete[] pcells_box;
		pcells_box = nullptr;
	}

	//------------------------------------- Make CMBND

	//copy from CPU version of calculated CMBND contact
	//We also need box for each device (box relative to entire cuVEC, i.e. cell start and end box coordinates make sense in cuBox(n), where n dimensions of entire cuVEC)
	template <typename CMBNDInfo>
	void copy_from_CMBNDInfo(CMBNDInfo& cmbndInfo, cuBox*& pbox_d)
	{
		//for most parameters in each CMBNDInfoCUDA it's just a straight copy on each device
		//the only one that differs is cells_box (box containing all cells on primary side of contact)
		//this is why we need to know info in pbox_d, so we can recompute the correct cells_box on each device
		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			(*ppCMBNDInfoCUDA[mGPU])()->copy_from_CMBNDInfo(cmbndInfo);

			//find cells_box for each device, but still relative to entire cuVEC
			cuBox cells_box = pbox_d[mGPU].get_intersection(cmbndInfo.cells_box);

			//now make relative to device cuVEC (if not an empty box, in which case set it as empty)
			if (!cells_box.IsNull()) cells_box -= pbox_d[mGPU].s;

			//finally set it in respective CMBNDInfoCUDA
			(*ppCMBNDInfoCUDA[mGPU])()->set_cells_box(cells_box);

			//set cells box for each device in cpu memory
			pcells_box[mGPU] = cells_box;

			//set contact size in cpu memory so we can access it quickly at run-time (this will set size of cuda kernels to launch)
			pcontact_size[mGPU] = cells_box.size().dim();
		}
	}

	//------------------------------------- INFO

	size_t contact_size(int device_idx) { return pcontact_size[device_idx]; }

	cuBox& cells_box(int device_idx) { return pcells_box[device_idx]; }

	//------------------------------------- GET MANAGED CMBND

	//get contained CMBNDInfoCUDA contact descriptor for given device
	CMBNDInfoCUDA& get_deviceobject(int device_idx) { return *ppCMBNDInfoCUDA[device_idx]; }
};