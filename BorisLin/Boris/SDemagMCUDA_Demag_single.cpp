#include "stdafx.h"
#include "SDemagMCUDA_Demag_single.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_SDEMAG

#include "SDemagCUDA_Demag.h"

SDemagMCUDA_Demag_single::SDemagMCUDA_Demag_single(SDemagCUDA_Demag* pSDemagCUDA_Demag_, int device_index_) :
	ConvolutionCUDA<SDemagMCUDA_Demag_single, DemagKernelCollectionCUDA>(),
	device_index(device_index_)
{
	pSDemagCUDA_Demag = pSDemagCUDA_Demag_;

	////////////////////////////////////////////////////

	Real_xRegion.resize(mGPU.get_num_devices());
	Real_yRegion.resize(mGPU.get_num_devices());
	Real_xRegion_half.resize(mGPU.get_num_devices());
	Real_yRegion_half.resize(mGPU.get_num_devices());

	Complex_xRegion.resize(mGPU.get_num_devices());
	Complex_yRegion.resize(mGPU.get_num_devices());
	Complex_xRegion_half.resize(mGPU.get_num_devices());
	Complex_yRegion_half.resize(mGPU.get_num_devices());

	for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

		Real_xRegion[idx] = nullptr;
		Real_yRegion[idx] = nullptr;
		Real_xRegion_half[idx] = nullptr;
		Real_yRegion_half[idx] = nullptr;

		Complex_xRegion[idx] = nullptr;
		Complex_yRegion[idx] = nullptr;
		Complex_xRegion_half[idx] = nullptr;
		Complex_yRegion_half[idx] = nullptr;
	}
}

SDemagMCUDA_Demag_single::~SDemagMCUDA_Demag_single()
{
	for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

		if (Real_xRegion[idx]) delete Real_xRegion[idx];
		Real_xRegion[idx] = nullptr;
		if (Real_yRegion[idx]) delete Real_yRegion[idx];
		Real_yRegion[idx] = nullptr;
		if (Real_xRegion_half[idx]) delete Real_xRegion_half[idx];
		Real_xRegion_half[idx] = nullptr;
		if (Real_yRegion_half[idx]) delete Real_yRegion_half[idx];
		Real_yRegion_half[idx] = nullptr;

		if (Complex_xRegion[idx]) delete Complex_xRegion[idx];
		Complex_xRegion[idx] = nullptr;
		if (Complex_yRegion[idx]) delete Complex_yRegion[idx];
		Complex_yRegion[idx] = nullptr;
		if (Complex_xRegion_half[idx]) delete Complex_xRegion_half[idx];
		Complex_xRegion_half[idx] = nullptr;
		if (Complex_yRegion_half[idx]) delete Complex_yRegion_half[idx];
		Complex_yRegion_half[idx] = nullptr;
	}
}

BError SDemagMCUDA_Demag_single::Initialize(std::vector<DemagKernelCollectionCUDA*>& kernelCollection, int n_z, bool initialize_on_gpu)
{
	BError error(CLASS_STR(SDemagMCUDA_Demag_single));

	if (!initialized) {
		
		//calculate kernels
		if (!error) error = Calculate_Demag_Kernels(kernelCollection, initialize_on_gpu, device_index);
		if (error) {

			//attemp to calculate kernels again, but with reduced convolution memory usage
			error.reset();
			error = Set_Preserve_Zero_Padding(false);
			if (!error) error = Calculate_Demag_Kernels(kernelCollection, initialize_on_gpu, device_index);
			if (error) return error(BERROR_OUTOFMEMORY_CRIT);
		}
		
		bool success = true;

		Real_xRegion_arr.clear();
		Real_yRegion_arr.clear();
		Real_xRegion_half_arr.clear();
		Real_yRegion_half_arr.clear();

		Complex_xRegion_arr.clear();
		Complex_yRegion_arr.clear();
		Complex_xRegion_half_arr.clear();
		Complex_yRegion_half_arr.clear();

		//setup input magnetization transfer spaces
		for (int idx = 0; idx < mGPU.get_num_devices(); idx++) {

			if (mGPU.get_num_devices() > 1) {

				if (!mGPU.get_halfprecision_transfer()) {

					if (!Real_xRegion[idx]) Real_xRegion[idx] = new cu_arr<cuReal3>();
					if (!Real_yRegion[idx]) Real_yRegion[idx] = new cu_arr<cuReal3>();

					if (!Complex_xRegion[idx]) Complex_xRegion[idx] = new cu_arr<cuBComplex>();
					if (!Complex_yRegion[idx]) Complex_yRegion[idx] = new cu_arr<cuBComplex>();

					//don't need the half spaces if half precision is not set
					if (Real_xRegion_half[idx]) Real_xRegion_half[idx]->clear();
					if (Real_yRegion_half[idx]) Real_yRegion_half[idx]->clear();

					if (Complex_xRegion_half[idx]) Complex_xRegion_half[idx]->clear();
					if (Complex_yRegion_half[idx]) Complex_yRegion_half[idx]->clear();
				}
				else {

					if (!Real_xRegion_half[idx]) Real_xRegion_half[idx] = new cu_arr<cuBHalf>();
					if (!Real_yRegion_half[idx]) Real_yRegion_half[idx] = new cu_arr<cuBHalf>();

					if (!Complex_xRegion_half[idx]) Complex_xRegion_half[idx] = new cu_arr<cuBHalf>();
					if (!Complex_yRegion_half[idx]) Complex_yRegion_half[idx] = new cu_arr<cuBHalf>();

					//don't need the full spaces if half precision is set
					if (Real_xRegion[idx]) Real_xRegion[idx]->clear();
					if (Real_yRegion[idx]) Real_yRegion[idx]->clear();

					if (Complex_xRegion[idx]) Complex_xRegion[idx]->clear();
					if (Complex_yRegion[idx]) Complex_yRegion[idx]->clear();
				}

				if (idx != device_index) {

					int region_xdim = pSDemagCUDA_Demag->pDemagMCUDA[idx]->nxRegion;
					int region_xdim_R = pSDemagCUDA_Demag->pDemagMCUDA[idx]->nxRegion_R;
					int region_ydim = pSDemagCUDA_Demag->pDemagMCUDA[idx]->nyRegion;

					if (!mGPU.get_halfprecision_transfer()) {

						if (success) success = Real_xRegion[idx]->resize(nxRegion_R * region_ydim * n_z);
						if (success) success = Real_yRegion[idx]->resize(region_xdim_R * nyRegion * n_z);

						if (success) success = Complex_xRegion[idx]->resize(nxRegion * region_ydim * n_z * 3);
						if (success) success = Complex_yRegion[idx]->resize(region_xdim * nyRegion * n_z * 3);
					}
					else {

						if (success) success = Real_xRegion_half[idx]->resize(nxRegion_R * region_ydim * n_z * 3);
						if (success) success = Real_yRegion_half[idx]->resize(region_xdim_R * nyRegion * n_z * 3);

						//x2 since packing complex into reals
						if (success) success = Complex_xRegion_half[idx]->resize(nxRegion * region_ydim * n_z * 3 * 2);
						//x2 since packing complex into reals
						if (success) success = Complex_yRegion_half[idx]->resize(region_xdim * nyRegion * n_z * 3 * 2);
					}
				}

				if (!mGPU.get_halfprecision_transfer()) {

					Real_xRegion_arr.push_back(Real_xRegion[idx]->get_managed_array());
					Real_yRegion_arr.push_back(Real_yRegion[idx]->get_managed_array());

					Complex_xRegion_arr.push_back(Complex_xRegion[idx]->get_managed_array());
					Complex_yRegion_arr.push_back(Complex_yRegion[idx]->get_managed_array());
				}
				else {

					Real_xRegion_half_arr.push_back(Real_xRegion_half[idx]->get_managed_array());
					Real_yRegion_half_arr.push_back(Real_yRegion_half[idx]->get_managed_array());

					Complex_xRegion_half_arr.push_back(Complex_xRegion_half[idx]->get_managed_array());
					Complex_yRegion_half_arr.push_back(Complex_yRegion_half[idx]->get_managed_array());
				}
			}
		}

		if (success) initialized = true;
		else error(BERROR_OUTOFMEMORY_CRIT);
	}

	//set normalization constant : exponent of FFT result numbers is proportional to Ms and mesh dimensions.
	normalization.from_cpu(pSDemagCUDA_Demag->Get_Ms0() * N.dim());
	normalization_M.from_cpu(pSDemagCUDA_Demag->Get_Ms0());

	return error;
}

BError SDemagMCUDA_Demag_single::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(SDemagMCUDA_Demag_single));

	if (cfgMessage == UPDATECONFIG_DEMAG_CONVCHANGE || cfgMessage == UPDATECONFIG_MESHCHANGE) {

		initialized = false;
	}

	return error;
}

#endif

#endif