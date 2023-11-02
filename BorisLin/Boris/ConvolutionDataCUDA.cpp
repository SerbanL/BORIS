#include "stdafx.h"
#include "ConvolutionDataCUDA.h"

#if COMPILECUDA == 1

ConvolutionDataCUDA::~ConvolutionDataCUDA()
{
	//if n.z == 0 then no plan is currently allocated
	if (n.z == 1) {

		cufftDestroy(plan2D_fwd_x);
		cufftDestroy(plan2D_y);
		cufftDestroy(plan2D_inv_x);
	}
	else if (n.z > 1) {

		cufftDestroy(plan3D_fwd_x);
		cufftDestroy(plan3D_y);
		cufftDestroy(plan3D_z);
		cufftDestroy(plan3D_inv_x);
	}

	DeallocateConvolutionSpaces();
}

//for given n value and pbc images recommend N value to use
cuSZ3 ConvolutionDataCUDA::Recommend_N_from_n(cuSZ3 n_, cuINT3 pbc_images_)
{
	//detemine N value to set for given n_ and pbc_images

	cuSZ3 N_ = cuSZ3(1, 1, 1);

	//set N values for FFT dimensions
	if (pbc_images_.x) {

		//pbc : can use wrap-around
		N_.x = n_.x;
	}
	else {

		//no wrap-around thus double the size, with input to be zero-padded
		N_.x = 2 * n_.x;
	}

	if (pbc_images_.y) {

		N_.y = n_.y;
	}
	else {

		N_.y = 2 * n_.y;
	}

	if (n_.z > 1) {

		if (pbc_images_.z) {

			N_.z = n_.z;

			//in z pbc mode we'll want to disable q2d mode as that only works for powers of 2 with N.z at least 2 times larger than n.z
			//if N.z is a power of 2 could adapt a q2D type mode for n.z = N.z but not worth the effort currently - is z pbc mode really that useful?
		}
		else {

			//if not in pbc mode we want to stick to powers of 2 here to take advantage of q2D mode
			while (N_.z < 2 * n_.z) { N_.z = N_.z * 2; };
		}
	}

	return N_;
}

//Similar to SetConvolutionDimensions but use a pre-determined N value
BError ConvolutionDataCUDA::SetConvolutionDimensions(cuSZ3 n_, cuReal3 h_, cuSZ3 N_, CONV_ convtype, cuINT3 pbc_images_, cuINT2 xRegion_, std::pair<int, int> devices_cfg)
{
	BError error(__FUNCTION__);

	//-------------------------

	embed_multiplication = (convtype == CONV_SINGLEMESH || convtype == CONV_MULTIMESH);
	additional_spaces = (convtype == CONV_MULTIMESH || convtype == CONV_MULTIMESH_NOTEMBEDDED);

	//-------------------------

	//if n.z == 0 then no plan is currently allocated (since n has not been set yet)
	if (n.z == 1) {

		//currently using 2D plan : free resources as new ones will be remade
		cufftDestroy(plan2D_fwd_x);
		cufftDestroy(plan2D_y);
		cufftDestroy(plan2D_inv_x);
	}
	else if (n.z > 1) {

		//currently using 3D plan : free resources as new ones will be remade
		cufftDestroy(plan3D_fwd_x);
		cufftDestroy(plan3D_y);
		cufftDestroy(plan3D_z);
		cufftDestroy(plan3D_inv_x);
	}

	//-------------------------

	n = n_;
	h = h_;
	N = N_;
	pbc_images = pbc_images_;

	//n in gpu memory
	cun.from_cpu(n);
	//N in gpu memory
	cuN.from_cpu(N);

	//-------------------------

	//setup xRegion
	xRegion = xRegion_;
	xRegion_R = xRegion;
	//in xRegion mode this value is not null, and input value xRegion_ ranges from 0 to n.x
	//since we must cover the range 0 to N.x/2 + 1, for xRegion[1] == n.x, adjust it by setting it to N.x/2 + 1. Now we can use xRegion to perform y and z FFTs in the appropriate region after x FFT.
	if (xRegion.j == n.x) xRegion.j = N.x / 2 + 1;

	if (pbc_images.x) {

		//with pbc along x, need to reduce xRegion values so they cover the range 0 to n.x / 2 + 1 (complex space)
		//it's not enough to simply divide by 2 since we must guarantee that all xRegions, except the last one, have the same number of cells

		int curr_device = devices_cfg.first;
		num_devices = devices_cfg.second;

		int num_x_cells_first = ((N.x / 2 + 1) - (N.x / 2 + 1) % num_devices) / num_devices;
		if (xRegion.j != N.x / 2 + 1) xRegion = cuINT2(num_x_cells_first * curr_device, num_x_cells_first * (curr_device + 1));
		else xRegion = cuINT2(num_x_cells_first * curr_device, N.x / 2 + 1);
	}

	cuxRegion.from_cpu(xRegion);
	cuxRegion_R.from_cpu(xRegion_R);

	//setup yRegion depending on devices_cfg
	if (!xRegion.IsNull()) {

		int curr_device = devices_cfg.first;
		num_devices = devices_cfg.second;
		int ycells_per_device = (n.y / num_devices);

		yRegion = cuINT2(ycells_per_device * curr_device, (curr_device == num_devices - 1 ? n.y : ycells_per_device * (curr_device + 1)));
	}
	else yRegion = cuINT2(0, n.y);
	cuyRegion.from_cpu(yRegion);

	nxRegion = (xRegion.IsNull() ? (int)(N.x / 2 + 1) : xRegion.j - xRegion.i);
	nxRegion_R = (xRegion.IsNull() ? n.x : xRegion_R.j - xRegion_R.i);
	nyRegion = yRegion.j - yRegion.i;

	cunum_devices.from_cpu(num_devices);

	cuN_xRegion.from_cpu(cuSZ3(nxRegion, N.y, N.z));

	//-------------------------

	//set size of complex arrays in gpu memory
	cuNc_xy.from_cpu(cuSZ3(nxRegion, N.y, N.z));
	cuNc_yx.from_cpu(cuSZ3(N.y, nxRegion, N.z));

	cuNc_xy_q2d.from_cpu(cuSZ3(nxRegion, N.y, N.z / 2));
	cuNc_yx_q2d.from_cpu(cuSZ3(N.y, nxRegion, N.z / 2));

	cuNcquart_xy.from_cpu(cuSZ3(N.x / 2 + 1, n.y, n.z));
	cuNcquart_yx.from_cpu(cuSZ3(n.y, N.x / 2 + 1, n.z));

	cuUpper_y_region.from_cpu(cuRect(cuReal3(0, n.y, 0), cuReal3(nxRegion, N.y, n.z)));
	cuUpper_y_transposed_region.from_cpu(cuRect(cuReal3(n.y, 0, 0), cuReal3(N.y, nxRegion, n.z)));
	cuUpper_z_region.from_cpu(cuRect(cuReal3(0, 0, n.z), cuReal3(nxRegion, N.y, N.z)));
	cuUpper_z_region_q2d.from_cpu(cuRect(cuReal3(0, 0, n.z), cuReal3(nxRegion, N.y, N.z / 2)));

	//-------------------------

	//Make cuda FFT plans
	cufftResult cuffterr = CUFFT_SUCCESS;

	if (n.z == 1) {

		//q2D mode not applicable for true 2D problems
		q2D_level = 0;

		//transpose xy before y fft / ifft? Use if N.y exceeds threshold, and always use in mGPU mode (xRegions) in 2D
		if (N.y >= TRANSPOSE_XY_YTHRESHOLD || !xRegion.IsNull()) transpose_xy = true;
		else transpose_xy = false;

#if SINGLEPRECISION == 1

		int embed[1] = { 0 };
		int ndims_x[1] = { (int)N.x };
		int ndims_y[1] = { (int)N.y };

		//Forward fft along x direction (out-of-place):
		if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan2D_fwd_x, 1, ndims_x,
			embed, 1, N.x,
			embed, 1, (N.x / 2 + 1),
			CUFFT_R2C, (int)nyRegion);

		if (!transpose_xy) {

			//Forward fft along y direction
			if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan2D_y, 1, ndims_y,
				embed, nxRegion, 1,
				embed, nxRegion, 1,
				CUFFT_C2C, nxRegion);
		}
		else {

			if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan2D_y, 1, ndims_y,
				embed, 1, N.y,
				embed, 1, N.y,
				CUFFT_C2C, nxRegion);
		}

		//Inverse fft along x direction
		if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan2D_inv_x, 1, ndims_x,
			embed, 1, (N.x / 2 + 1),
			embed, 1, N.x,
			CUFFT_C2R, (int)nyRegion);

#else

		int embed[1] = { 0 };
		int ndims_x[1] = { (int)N.x };
		int ndims_y[1] = { (int)N.y };

		//Forward fft along x direction (out-of-place):
		if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan2D_fwd_x, 1, ndims_x,
			embed, 1, N.x,
			embed, 1, (N.x / 2 + 1),
			CUFFT_D2Z, (int)nyRegion);

		if (!transpose_xy) {

			//Forward fft along y direction
			if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan2D_y, 1, ndims_y,
				embed, nxRegion, 1,
				embed, nxRegion, 1,
				CUFFT_Z2Z, nxRegion);
		}
		else {

			if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan2D_y, 1, ndims_y,
				embed, 1, N.y,
				embed, 1, N.y,
				CUFFT_Z2Z, nxRegion);
		}

		//Inverse fft along x direction
		if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan2D_inv_x, 1, ndims_x,
			embed, 1, (N.x / 2 + 1),
			embed, 1, N.x,
			CUFFT_Z2D, (int)nyRegion);

#endif
	}
	else {

		//3D problem

		if (pbc_images.z) {

			//disable q2D mode if using z pbc
			q2D_level = 0;
		}
		else {

			//quasi 2D mode? (not applicable if convolution not embedded
			if (embed_multiplication && n.z == 2) {

				//N.z = 4
				q2D_level = 4;
			}
			else if (embed_multiplication && n.z <= 4) {

				//N.z = 8
				q2D_level = 8;
			}

			else if (embed_multiplication && n.z <= 8) {

				//N.z = 16
				q2D_level = 16;
			}

			else if (embed_multiplication && n.z <= 16) {

				//N.z = 32
				q2D_level = 32;
			}
			//above this level q2D mode is slower than full 3D mode due to inefficient use of gpu bandwidth
			else {

				//disable q2D mode
				q2D_level = 0;
			}
		}

		//always use transpose_xy in 3D
		transpose_xy = true;

#if SINGLEPRECISION == 1

		int embed[1] = { 0 };
		int ndims_x[1] = { (int)N.x };
		int ndims_y[1] = { (int)N.y };
		int ndims_z[1] = { (int)N.z };

		//batched x fft from cuIn to cuSquart
		if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan3D_fwd_x, 1, ndims_x,
			embed, 1, N.x,
			embed, 1, (N.x / 2 + 1),
			CUFFT_R2C, (int)nyRegion * (int)n.z);

		//batched y fft from cuS (transposed) to cuS
		if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan3D_y, 1, ndims_y,
			embed, 1, N.y,
			embed, 1, N.y,
			CUFFT_C2C, nxRegion * n.z);

		//batched inverse x fft from cuSquart to cuOut
		if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan3D_inv_x, 1, ndims_x,
			embed, 1, (N.x / 2 + 1),
			embed, 1, N.x,
			CUFFT_C2R, (int)nyRegion * (int)n.z);

		if (!q2D_level) {

			//Forward and inverse fft along z direction all batched - not applicable for q2D mode

			if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan3D_z, 1, ndims_z,
				embed, nxRegion * N.y, 1,
				embed, nxRegion * N.y, 1,
				CUFFT_C2C, nxRegion * N.y);
		}

#else

		int embed[1] = { 0 };
		int ndims_x[1] = { (int)N.x };
		int ndims_y[1] = { (int)N.y };
		int ndims_z[1] = { (int)N.z };

		//batched x fft from cuIn to cuSquart
		if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan3D_fwd_x, 1, ndims_x,
			embed, 1, N.x,
			embed, 1, (N.x / 2 + 1),
			CUFFT_D2Z, (int)nyRegion * (int)n.z);

		//batched y fft from cuS (transposed) to cuS
		if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan3D_y, 1, ndims_y,
			embed, 1, N.y,
			embed, 1, N.y,
			CUFFT_Z2Z, nxRegion * n.z);

		//batched inverse x fft from cuSquart to cuOut
		if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan3D_inv_x, 1, ndims_x,
			embed, 1, (N.x / 2 + 1),
			embed, 1, N.x,
			CUFFT_Z2D, (int)nyRegion * (int)n.z);

		if (!q2D_level) {

			//Forward and inverse fft along z direction all batched - not applicable for q2D mode

			if (cuffterr == CUFFT_SUCCESS) cuffterr = cufftPlanMany(&plan3D_z, 1, ndims_z,
				embed, nxRegion * N.y, 1,
				embed, nxRegion * N.y, 1,
				CUFFT_Z2Z, nxRegion * N.y);
		}

#endif
	}

	if (cuffterr != CUFFT_SUCCESS) return error(BERROR_OUTOFGPUMEMORY_CRIT);

	//-------------------------
	
	//always use higher performance mode
	//if you want to use lower performance mode (e.g. due to out of memory), then call Set_Preserve_Zero_Padding(false) separately after
	//this way the convoultion algorithm always reverts back to higher performance mode when it is changed
	preserve_zero_padding = true;

	//FFT spaces memory allocation
	error = AllocateConvolutionSpaces();
	//if this fails, try again with lower memory usage mode
	if (error) {

		error.reset();
		error = Set_Preserve_Zero_Padding(false);
	}

	return error;
}

void ConvolutionDataCUDA::DeallocateConvolutionSpaces(void)
{
	cuIn_x.clear();
	cuIn_y.clear();
	cuIn_z.clear();
	
	cuOut_x.clear();
	cuOut_y.clear();
	cuOut_z.clear();

	cuS_x.clear();
	cuS_y.clear();
	cuS_z.clear();

	cuS2_x.clear();
	cuS2_y.clear();
	cuS2_z.clear();

	cuSquart_x.clear();
	cuSquart_y.clear();
	cuSquart_z.clear();

	//free zpz spaces only if separate memory was previously allocated (i.e. not nullptr and not simple pointers to cuS spaces)
	if (pcuSzpz_x && pcuSzpz_x != &cuS_x) delete pcuSzpz_x;
	pcuSzpz_x = nullptr;
	if (pcuSzpz_y && pcuSzpz_y != &cuS_y) delete pcuSzpz_y;
	pcuSzpz_y = nullptr;
	if (pcuSzpz_z && pcuSzpz_z != &cuS_z) delete pcuSzpz_z;
	pcuSzpz_z = nullptr;

	//free zpy spaces only if separate memory was previously allocated (i.e. not nullptr and not simple pointers to cuS spaces)
	if (pcuSzpy_x && pcuSzpy_x != &cuS_x) delete pcuSzpy_x;
	pcuSzpy_x = nullptr;
	if (pcuSzpy_y && pcuSzpy_y != &cuS_y) delete pcuSzpy_y;
	pcuSzpy_y = nullptr;
	if (pcuSzpy_z && pcuSzpy_z != &cuS_z) delete pcuSzpy_z;
	pcuSzpy_z = nullptr;
}

//FFT spaces memory allocation
BError ConvolutionDataCUDA::AllocateConvolutionSpaces(void)
{
	BError error(__FUNCTION__);

	DeallocateConvolutionSpaces();

	//Input/Output space
	if (!cuIn_x.resize(N.x * nyRegion * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	if (!cuIn_y.resize(N.x * nyRegion * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	if (!cuIn_z.resize(N.x * nyRegion * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	
	cuIn_x.set(cuBReal(0.0));
	cuIn_y.set(cuBReal(0.0));
	cuIn_z.set(cuBReal(0.0));

	if (!cuOut_x.resize(N.x * nyRegion * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	if (!cuOut_y.resize(N.x * nyRegion * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	if (!cuOut_z.resize(N.x * nyRegion * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
	cuOut_x.set(cuBReal(0.0));
	cuOut_y.set(cuBReal(0.0));
	cuOut_z.set(cuBReal(0.0));

	//full-size scratch space
	if (!q2D_level) {

		//applies for 3D problems as well as 2D problems (n.z = 1 and thus N.z = 1)

		if (!cuS_x.resize(nxRegion * N.y * N.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!cuS_y.resize(nxRegion * N.y * N.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!cuS_z.resize(nxRegion * N.y * N.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		cuS_x.set(cuBComplex());
		cuS_y.set(cuBComplex());
		cuS_z.set(cuBComplex());
	}
	else {

		//in q2D mode we can halve the space required for cuS scratch spaces (note, this is disabled for z pbc, so N.z = 2 * n.z here)

		if (!cuS_x.resize(nxRegion * N.y * N.z / 2)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!cuS_y.resize(nxRegion * N.y * N.z / 2)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!cuS_z.resize(nxRegion * N.y * N.z / 2)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }

		cuS_x.set(cuBComplex());
		cuS_y.set(cuBComplex());
		cuS_z.set(cuBComplex());
	}

	//Scratch spaces with zero padding above ny for forward FFT : not needed if y pbc is on.
	if (!pbc_images.y && preserve_zero_padding) {

		//allocate memory if nullptr or if zpy scratch spaces point to cuS (i.e. they don't have separate memory allocated)
		if (!pcuSzpy_x || pcuSzpy_x == &cuS_x) pcuSzpy_x = new cu_arr<cuBComplex>();
		if (!pcuSzpy_y || pcuSzpy_y == &cuS_y) pcuSzpy_y = new cu_arr<cuBComplex>();
		if (!pcuSzpy_z || pcuSzpy_z == &cuS_z) pcuSzpy_z = new cu_arr<cuBComplex>();

		if (!pcuSzpy_x->resize(nxRegion * N.y * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!pcuSzpy_y->resize(nxRegion * N.y * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!pcuSzpy_z->resize(nxRegion * N.y * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		pcuSzpy_x->set(cuBComplex());
		pcuSzpy_y->set(cuBComplex());
		pcuSzpy_z->set(cuBComplex());
	}
	else {

		//free zpy spaces only if separate memory was previously allocated (i.e. not nullptr and not simple pointers to cuS spaces)
		if (pcuSzpy_x && pcuSzpy_x != &cuS_x) delete pcuSzpy_x;
		pcuSzpy_x = &cuS_x;
		if (pcuSzpy_y && pcuSzpy_y != &cuS_y) delete pcuSzpy_y;
		pcuSzpy_y = &cuS_y;
		if (pcuSzpy_z && pcuSzpy_z != &cuS_z) delete pcuSzpy_z;
		pcuSzpy_z = &cuS_z;
	}

	//Scratch spaces with zero padding above nz for forward FFT : not needed if z pbc is on, or in q2D or 2D modes.
	if (!pbc_images.z && !q2D_level && n.z > 1 && preserve_zero_padding) {

		//allocate memory if nullptr or if zpz scratch spaces point to cuS (i.e. they don't have separate memory allocated)
		if (!pcuSzpz_x || pcuSzpz_x == &cuS_x) pcuSzpz_x = new cu_arr<cuBComplex>();
		if (!pcuSzpz_y || pcuSzpz_y == &cuS_y) pcuSzpz_y = new cu_arr<cuBComplex>();
		if (!pcuSzpz_z || pcuSzpz_z == &cuS_z) pcuSzpz_z = new cu_arr<cuBComplex>();

		if (!pcuSzpz_x->resize(nxRegion * N.y * N.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!pcuSzpz_y->resize(nxRegion * N.y * N.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!pcuSzpz_z->resize(nxRegion * N.y * N.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		pcuSzpz_x->set(cuBComplex());
		pcuSzpz_y->set(cuBComplex());
		pcuSzpz_z->set(cuBComplex());
	}
	else {

		//free zpz spaces only if separate memory was previously allocated (i.e. not nullptr and not simple pointers to cuS spaces)
		if (pcuSzpz_x && pcuSzpz_x != &cuS_x) delete pcuSzpz_x;
		pcuSzpz_x = &cuS_x;
		if (pcuSzpz_y && pcuSzpz_y != &cuS_y) delete pcuSzpz_y;
		pcuSzpz_y = &cuS_y;
		if (pcuSzpz_z && pcuSzpz_z != &cuS_z) delete pcuSzpz_z;
		pcuSzpz_z = &cuS_z;
	}

	if (additional_spaces) {

		//allocate additional cuS2 scratch spaces : always full size
		if (!cuS2_x.resize(nxRegion * N.y * N.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!cuS2_y.resize(nxRegion * N.y * N.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!cuS2_z.resize(nxRegion * N.y * N.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		cuS2_x.set(cuBComplex());
		cuS2_y.set(cuBComplex());
		cuS2_z.set(cuBComplex());
	}
	else {

		cuS2_x.clear();
		cuS2_y.clear();
		cuS2_z.clear();
	}

	//quarter scratch space used only if transpose mode is on
	if (transpose_xy) {

		if (!cuSquart_x.resize((N.x / 2 + 1) * nyRegion * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!cuSquart_y.resize((N.x / 2 + 1) * nyRegion * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		if (!cuSquart_z.resize((N.x / 2 + 1) * nyRegion * n.z)) { DeallocateConvolutionSpaces();  return error(BERROR_OUTOFGPUMEMORY_CRIT); }
		cuSquart_x.set(cuBComplex());
		cuSquart_y.set(cuBComplex());
		cuSquart_z.set(cuBComplex());
	}
	else {

		cuSquart_x.clear();
		cuSquart_y.clear();
		cuSquart_z.clear();
	}

	return error;
}

BError ConvolutionDataCUDA::SetConvolutionDimensions(cuSZ3 n_, cuReal3 h_, CONV_ convtype, cuINT3 pbc_images_, cuINT2 xRegion_, std::pair<int, int> devices_cfg)
{
	//detemine N value to set for given n_ and pbc_images
	cuSZ3 N_ = Recommend_N_from_n(n_, pbc_images_);

	//now set everything
	return SetConvolutionDimensions(n_, h_, N_, convtype, pbc_images_, xRegion_, devices_cfg);
}

#endif