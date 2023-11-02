#include "stdafx.h"
#include "DemagKernelCollection.h"

#ifdef MODULE_COMPILATION_SDEMAG

//-------------------------- KERNEL CALCULATION

//this initializes all the convolution kernels for the given mesh dimensions. 2D is for n.z == 1.
BError DemagKernelCollection::Calculate_Demag_Kernels(std::vector<DemagKernelCollection*>& kernelCollection)
{
	BError error(__FUNCTION__);

	for (int index = 0; index < Rect_collection.size(); index++) {

		if (!error) {

			//before allocating and computing kernel, check to see if any other DemagKernelCollection module has not already calculated one like it

			//Rect_collection is in one-to-one correspondence with kernelCollection
			//For a demag kernel to be identical to the one we need, it must have the same shift, same h source and same h destination

			//shift for source as Rect_collection[index] and destination as this_rect
			DBL3 shift = (this_rect.s - Rect_collection[index].s);

			DBL3 h_src = kernelCollection[index]->h;
			DBL3 h_dst = h;
			
			for (int idx = 0; idx < kernelCollection.size(); idx++) {

				std::shared_ptr<KerType> existing_kernel = kernelCollection[idx]->KernelAlreadyComputed(shift, h_src, h_dst);

				if (existing_kernel != nullptr) {

					//found one : just increase ref count
					kernels[index] = existing_kernel;

					//is it inverse z-shifted?
					if (IsZ(shift.x) && IsZ(shift.y)) {

						if (IsNZ(shift.z) && IsZ(shift.z + kernels[index]->shift.z)) {

							//yes it is. mark it here so we can adjust the kernel multiplications
							inverse_shifted[index] = true;
						}
					}

					break;
				}
			}
			
			if (kernels[index] == nullptr) {

				//no -> allocate then compute it
				kernels[index] = std::shared_ptr<KerType>(new KerType());
				error = kernels[index]->AllocateKernels(Rect_collection[index], this_rect, N);
				if (error) return error;

				//now compute it
				if (Rect_collection[index] == this_rect) {

					kernels[index]->shift = DBL3();
					kernels[index]->h_dst = h;
					kernels[index]->h_src = h;

					//use self versions
					if (n.z == 1) error = Calculate_Demag_Kernels_2D_Self(index);
					else error = Calculate_Demag_Kernels_3D_Self(index);
				}
				else {
					
					//shift for source as Rect_collection[index] and destination as this_rect
					kernels[index]->shift = (this_rect.s - Rect_collection[index].s);
					kernels[index]->h_dst = h;
					kernels[index]->h_src = kernelCollection[index]->h;
					
					if (n.z == 1) {

						if (kernels[index]->zshifted) {

							//z-shifted kernels for 2D
							error = Calculate_Demag_Kernels_2D_zShifted(index);
						}
						else if (kernels[index]->xshifted) {

							//x-shifted kernels for 2D
							error = Calculate_Demag_Kernels_2D_xShifted(index);
						}
						else {

							//general 2D kernels
							error = Calculate_Demag_Kernels_2D_Complex_Full(index);
						}
					}
					else {
					
						if (kernels[index]->zshifted) {

							//z-shifted kernels for 3D
							error = Calculate_Demag_Kernels_3D_zShifted(index);
						}
						else if (kernels[index]->xshifted) {

							//x-shifted kernels for 3D
							error = Calculate_Demag_Kernels_3D_xShifted(index);
						}
						else {
						
							//general 3D kernels
							error = Calculate_Demag_Kernels_3D_Complex_Full(index);
						}
					}
				}

				//set flag to say it's been computed so it could be reused if needed
				kernels[index]->kernel_calculated = true;
			}
		}
	}

	return error;
}

//search to find a matching kernel that has already been computed and return pointer to it -> kernel can be identified from shift, source and destination discretisation
std::shared_ptr<KerType> DemagKernelCollection::KernelAlreadyComputed(DBL3 shift, DBL3 h_src, DBL3 h_dst)
{
	//kernels[index] must not be nullptr, must have kernel_calculated = true and shift, h_src, h_dst must match the corresponding values in kernels[index] 

	for (int idx = 0; idx < kernels.size(); idx++) {

		if (kernels[idx] && kernels[idx]->kernel_calculated) {

			//match in source and destination cellsizes?
			if (kernels[idx]->h_src == h_src && kernels[idx]->h_dst == h_dst) {

				//do the shifts match?
				if (kernels[idx]->shift == shift) {

					return kernels[idx];
				}

				//are the shifts z shifts that differ only in sign? Only applies in 2D mode.
				if (N.z == 1 && IsZ(shift.x) && IsZ(shift.y) && IsZ(kernels[idx]->shift.x) && IsZ(kernels[idx]->shift.y) && IsZ(shift.z + kernels[idx]->shift.z)) {

					return kernels[idx];
				}
			}
		}
	}

	return nullptr;
}

//2D kernels (Kdiag_real, and K2D_odiag, with full use of kernel symmetries)
BError DemagKernelCollection::Calculate_Demag_Kernels_2D_Self(int index)
{
	BError error(__FUNCTION__);

	//-------------- CALCULATE DEMAG TENSOR

	//Demag tensor components
	//
	// D11 D12 0
	// D12 D22 0
	// 0   0   D33

	//Ddiag : D11, D22, D33 are the diagonal tensor elements
	VEC<DBL3> Ddiag;
	if (!Ddiag.resize(N)) return error(BERROR_OUTOFMEMORY_NCRIT);

	//off-diagonal tensor elements
	std::vector<double> Dodiag;
	if (!malloc_vector(Dodiag, N.x*N.y)) return error(BERROR_OUTOFMEMORY_NCRIT);

	//use ratios instead of cellsizes directly - same result but better in terms of floating point errors
	DemagTFunc dtf;

	if (pbc_images.IsNull()) {

		//no pbcs in any dimension -> use these
		if (!dtf.CalcDiagTens2D(Ddiag, n, N, h / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
		if (!dtf.CalcOffDiagTens2D(Dodiag, n, N, h / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		//pbcs used in at least one dimension
		if (!dtf.CalcDiagTens2D_PBC(
			Ddiag, N, h / h_max, 
			true, ASYMPTOTIC_DISTANCE, 
			pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);

		if (!dtf.CalcOffDiagTens2D_PBC(
			Dodiag, N, h / h_max, 
			true, ASYMPTOTIC_DISTANCE, 
			pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	
	//-------------- SETUP FFT

	double* pline_real = fftw_alloc_real(maximum(N.x, N.y, N.z) * 3);
	double* pline_real_odiag = fftw_alloc_real(maximum(N.x, N.y, N.z));
	fftw_complex* pline_odiag = fftw_alloc_complex(maximum(N.x / 2 + 1, N.y, N.z));
	fftw_complex* pline = fftw_alloc_complex(maximum(N.x / 2 + 1, N.y, N.z) * 3);

	//make fft plans
	int dims_x[1] = { (int)N.x };
	int dims_y[1] = { (int)N.y };

	fftw_plan plan_fwd_x = fftw_plan_many_dft_r2c(1, dims_x, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_x_odiag = fftw_plan_many_dft_r2c(1, dims_x, 1,
		pline_real_odiag, nullptr, 1, 1,
		pline_odiag, nullptr, 1, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_y = fftw_plan_many_dft_r2c(1, dims_y, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_y_odiag = fftw_plan_many_dft_r2c(1, dims_y, 1,
		pline_real_odiag, nullptr, 1, 1,
		pline_odiag, nullptr, 1, 1,
		FFTW_PATIENT);

	//-------------- FFT REAL TENSOR INTO REAL KERNELS

	//1. FFTs along x
	for (int j = 0; j < N.y; j++) {

		//write input into fft line
		for (int i = 0; i < N.x; i++) {

			int idx_in = i + j * N.x;

			*reinterpret_cast<DBL3*>(pline_real + i * 3) = Ddiag[idx_in];
			*reinterpret_cast<double*>(pline_real_odiag + i) = Dodiag[idx_in];
		}

		//fft on line
		fftw_execute(plan_fwd_x);
		fftw_execute(plan_fwd_x_odiag);

		//pack into tensor for next step
		for (int i = 0; i < N.x / 2 + 1; i++) {

			ReIm3 value = *reinterpret_cast<ReIm3*>(pline + i * 3);
			ReIm value_odiag = *reinterpret_cast<ReIm*>(pline_odiag + i);

			Ddiag[i + j * (N.x / 2 + 1)] = DBL3(value.x.Re, value.y.Re, value.z.Re);
			Dodiag[i + j * (N.x / 2 + 1)] = value_odiag.Im;
		}
	}

	//2. FFTs along y
	for (int i = 0; i < (N.x / 2 + 1); i++) {

		//fetch line from array
		for (int j = 0; j < N.y; j++) {

			*reinterpret_cast<DBL3*>(pline_real + j * 3) = Ddiag[i + j * (N.x / 2 + 1)];
			*reinterpret_cast<double*>(pline_real_odiag + j) = Dodiag[i + j * (N.x / 2 + 1)];
		}

		//fft on line
		fftw_execute(plan_fwd_y);
		fftw_execute(plan_fwd_y_odiag);

		//pack into output real kernels with reduced strides
		for (int j = 0; j < N.y / 2 + 1; j++) {

			ReIm3 value = *reinterpret_cast<ReIm3*>(pline + j * 3);
			ReIm value_odiag = *reinterpret_cast<ReIm*>(pline_odiag + j);

			//even w.r.t. y so output is purely real
			kernels[index]->Kdiag_real[i + j * (N.x / 2 + 1)] = DBL3(value.x.Re, value.y.Re, value.z.Re);

			//odd w.r.t. y so the purely imaginary input becomes purely real
			//however since we used CopyRealShuffle and RealfromComplexFFT, i.e. treating the input as purely real rather than purely imaginary we need to account for the i * i = -1 term, hence the - sign below
			kernels[index]->K2D_odiag[i + j * (N.x / 2 + 1)] = -value_odiag.Im;
		}
	}

	//-------------- CLEANUP

	fftw_destroy_plan(plan_fwd_x);
	fftw_destroy_plan(plan_fwd_x_odiag);
	fftw_destroy_plan(plan_fwd_y);
	fftw_destroy_plan(plan_fwd_y_odiag);

	fftw_free((double*)pline_real);
	fftw_free((double*)pline_real_odiag);
	fftw_free((fftw_complex*)pline);

	return error;
}

//2D layers, z shift only : Kernels can be stored as real with use of kernel symmetries. Kxx, Kyy, Kzz, Kxy real, Kxz, Kyz imaginary
BError DemagKernelCollection::Calculate_Demag_Kernels_2D_zShifted(int index)
{
	BError error(__FUNCTION__);

	//-------------- DEMAG TENSOR

	//Demag tensor components
	//
	// D11 D12 D13
	// D12 D22 D23
	// D13 D23 D33

	//D11, D22, D33 are the diagonal tensor elements
	//D12, D13, D23 are the off-diagonal tensor elements
	VEC<DBL3> D;

	if (!D.resize(N)) return error(BERROR_OUTOFMEMORY_NCRIT);

	//object used to compute tensor elements
	DemagTFunc dtf;

	//-------------- SETUP FFT

	double* pline_real = fftw_alloc_real(maximum(N.x, N.y, N.z) * 3);
	fftw_complex* pline = fftw_alloc_complex(maximum(N.x / 2 + 1, N.y, N.z) * 3);

	//make fft plans
	int dims_x[1] = { (int)N.x };
	int dims_y[1] = { (int)N.y };

	fftw_plan plan_fwd_x = fftw_plan_many_dft_r2c(1, dims_x, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_y = fftw_plan_many_dft_r2c(1, dims_y, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	//-------------- FFT REAL TENSOR INTO REAL KERNELS

	//lambda used to transform an input tensor into an output kernel
	auto tensor_to_kernel = [&](VEC<DBL3>& tensor, VEC<DBL3>& kernel, bool off_diagonal) -> void {

		//1. FFTs along x
		for (int j = 0; j < N.y; j++) {

			//write input into fft line (zero padding kept)
			for (int i = 0; i < N.x; i++) {

				int idx_in = i + j * N.x;

				*reinterpret_cast<DBL3*>(pline_real + i * 3) = tensor[idx_in];
			}

			//fft on line
			fftw_execute(plan_fwd_x);

			//pack into tensor for next step
			for (int i = 0; i < N.x / 2 + 1; i++) {

				ReIm3 value = *reinterpret_cast<ReIm3*>(pline + i * 3);

				if (!off_diagonal) {

					//even w.r.t. to x so output is purely real
					tensor[i + j * (N.x / 2 + 1)] = DBL3(value.x.Re, value.y.Re, value.z.Re);
				}
				else {

					//Dxy : odd x, Dxz : odd x, Dyz : even x
					tensor[i + j * (N.x / 2 + 1)] = DBL3(value.x.Im, value.y.Im, value.z.Re);
				}
			}
		}

		//2. FFTs along y
		for (int i = 0; i < (N.x / 2 + 1); i++) {

			//fetch line from fft array (zero padding kept)
			for (int j = 0; j < N.y; j++) {

				int idx_in = i + j * (N.x / 2 + 1);

				*reinterpret_cast<DBL3*>(pline_real + j * 3) = tensor[idx_in];
			}

			//fft on line
			fftw_execute(plan_fwd_y);

			//pack into kernel
			for (int j = 0; j < N.y / 2 + 1; j++) {

				ReIm3 value = *reinterpret_cast<ReIm3*>(pline + j * 3);

				if (!off_diagonal) {

					//even w.r.t. to y so output is purely real
					kernel[i + j * (N.x / 2 + 1)] = DBL3(value.x.Re, value.y.Re, value.z.Re);
				}
				else {

					//adjust for i * i = -1 in Nxy element
					kernel[i + j * (N.x / 2 + 1)] = DBL3(-value.x.Im, value.y.Re, value.z.Im);
				}
			}
		}
	};

	//-------------- CALCULATE DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	//no need to pass the actual cellsize values, just normalized values will do

	if (pbc_images.IsNull()) {

		if (!dtf.CalcDiagTens2D_Shifted_Irregular(D, n, N, kernels[index]->h_src / h_max, kernels[index]->h_dst / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		if (!dtf.CalcDiagTens2D_Shifted_Irregular_PBC(
			D, N, kernels[index]->h_src / h_max, kernels[index]->h_dst / h_max, kernels[index]->shift / h_max, 
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kdiag_real, false);

	//-------------- CALCULATE OFF-DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	if (pbc_images.IsNull()) {

		if (!dtf.CalcOffDiagTens2D_Shifted_Irregular(D, n, N, kernels[index]->h_src / h_max, kernels[index]->h_dst / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		if (!dtf.CalcOffDiagTens2D_Shifted_Irregular_PBC(
			D, N, kernels[index]->h_src / h_max, kernels[index]->h_dst / h_max, kernels[index]->shift / h_max,
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kodiag_real, true);

	//-------------- CLEANUP

	fftw_destroy_plan(plan_fwd_x);
	fftw_destroy_plan(plan_fwd_y);

	fftw_free((double*)pline_real);
	fftw_free((fftw_complex*)pline);

	//Done
	return error;
}

//2D layers, x shift only : Kernels are complex but can be stored using symmetries
BError DemagKernelCollection::Calculate_Demag_Kernels_2D_xShifted(int index)
{
	BError error(__FUNCTION__);

	//-------------- DEMAG TENSOR

	//Demag tensor components
	//
	// D11 D12 D13
	// D12 D22 D23
	// D13 D23 D33

	//D11, D22, D33 are the diagonal tensor elements
	//D12, D13, D23 are the off-diagonal tensor elements
	VEC<DBL3> D;
	VEC<ReIm3> Scratch;

	if (!D.resize(N)) return error(BERROR_OUTOFMEMORY_NCRIT);
	if (!Scratch.resize(SZ3(N.x / 2 + 1, N.y, N.z))) return error(BERROR_OUTOFMEMORY_NCRIT);

	//object used to compute tensor elements
	DemagTFunc dtf;

	//-------------- SETUP FFT

	double* pline_real = fftw_alloc_real(maximum(N.x, N.y, N.z) * 3);
	fftw_complex* pline = fftw_alloc_complex(maximum(N.x / 2 + 1, N.y, N.z) * 3);

	//make fft plans
	int dims_x[1] = { (int)N.x };
	int dims_y[1] = { (int)N.y };

	fftw_plan plan_fwd_x = fftw_plan_many_dft_r2c(1, dims_x, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_y = fftw_plan_many_dft(1, dims_y, 3,
		pline, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_FORWARD, FFTW_PATIENT);

	//-------------- FFT REAL TENSOR INTO REAL KERNELS

	//lambda used to transform an input tensor into an output kernel
	auto tensor_to_kernel = [&](VEC<DBL3>& tensor, VEC<ReIm3>& kernel) -> void {

		//1. FFTs along x
		for (int j = 0; j < N.y; j++) {

			//write input into fft line (zero padding kept)
			for (int i = 0; i < N.x; i++) {

				int idx_in = i + j * N.x;

				*reinterpret_cast<DBL3*>(pline_real + i * 3) = tensor[idx_in];
			}

			//fft on line
			fftw_execute(plan_fwd_x);

			//pack into scratch space
			for (int i = 0; i < N.x / 2 + 1; i++) {

				ReIm3 value = *reinterpret_cast<ReIm3*>(pline + i * 3);

				Scratch[i + j * (N.x / 2 + 1)] = value;
			}
		}

		//2. FFTs along y
		for (int i = 0; i < (N.x / 2 + 1); i++) {

			//fetch line from fft array (zero padding kept)
			for (int j = 0; j < N.y; j++) {

				int idx_in = i + j * (N.x / 2 + 1);

				*reinterpret_cast<ReIm3*>(pline + j * 3) = Scratch[idx_in];
			}

			//fft on line
			fftw_execute(plan_fwd_y);

			//pack into scratch space
			for (int j = 0; j < N.y / 2 + 1; j++) {

				ReIm3 value = *reinterpret_cast<ReIm3*>(pline + j * 3);

				kernel[i + j * (N.x / 2 + 1)] = value;
			}
		}
	};

	//-------------- CALCULATE DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	//no need to pass the actual cellsize values, just normalized values will do
	if (pbc_images.IsNull()) {

		//no pbcs in any dimension -> use these
		if (!dtf.CalcDiagTens2D_Shifted(D, n, N, h / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		//pbcs used in at least one dimension
		if (!dtf.CalcDiagTens2D_Shifted_PBC(
			D, N, h / h_max, kernels[index]->shift / h_max,
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kdiag_cmpl);

	//-------------- CALCULATE OFF-DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	if (pbc_images.IsNull()) {

		//no pbcs in any dimension -> use these
		if (!dtf.CalcOffDiagTens2D_Shifted(D, n, N, h / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		//pbcs used in at least one dimension
		if (!dtf.CalcOffDiagTens2D_Shifted_PBC(
			D, N, h / h_max, kernels[index]->shift / h_max,
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kodiag_cmpl);

	//-------------- CLEANUP

	fftw_destroy_plan(plan_fwd_x);
	fftw_destroy_plan(plan_fwd_y);

	fftw_free((double*)pline_real);
	fftw_free((fftw_complex*)pline);

	//Done
	return error;
}

//2D layers, complex kernels most general case (Kdiag_cmpl, and Kodiag_cmpl, without any kernel symmetries)
BError DemagKernelCollection::Calculate_Demag_Kernels_2D_Complex_Full(int index)
{
	BError error(__FUNCTION__);

	//-------------- DEMAG TENSOR

	//Demag tensor components
	//
	// D11 D12 D13
	// D12 D22 D23
	// D13 D23 D33

	//D11, D22, D33 are the diagonal tensor elements
	//D12, D13, D23 are the off-diagonal tensor elements
	VEC<DBL3> D;

	if (!D.resize(N)) return error(BERROR_OUTOFMEMORY_NCRIT);

	//object used to compute tensor elements
	DemagTFunc dtf;

	//-------------- SETUP FFT

	double* pline_real = fftw_alloc_real(maximum(N.x, N.y, N.z) * 3);
	fftw_complex* pline = fftw_alloc_complex(maximum(N.x / 2 + 1, N.y, N.z) * 3);

	//make fft plans
	int dims_x[1] = { (int)N.x };
	int dims_y[1] = { (int)N.y };

	fftw_plan plan_fwd_x = fftw_plan_many_dft_r2c(1, dims_x, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_y = fftw_plan_many_dft(1, dims_y, 3,
		pline, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_FORWARD, FFTW_PATIENT);

	//-------------- FFT REAL TENSOR INTO REAL KERNELS

	//lambda used to transform an input tensor into an output kernel
	auto tensor_to_kernel = [&](VEC<DBL3>& tensor, VEC<ReIm3>& kernel) -> void {

		//1. FFTs along x
		for (int j = 0; j < N.y; j++) {

			//write input into fft line (zero padding kept)
			for (int i = 0; i < N.x; i++) {

				int idx_in = i + j * N.x;

				*reinterpret_cast<DBL3*>(pline_real + i * 3) = tensor[idx_in];
			}

			//fft on line
			fftw_execute(plan_fwd_x);

			//pack into kernel for next step
			for (int i = 0; i < N.x / 2 + 1; i++) {

				ReIm3 value = *reinterpret_cast<ReIm3*>(pline + i * 3);

				kernel[i + j * (N.x / 2 + 1)] = value;
			}
		}

		//2. FFTs along y
		for (int i = 0; i < (N.x / 2 + 1); i++) {

			//fetch line from fft array (zero padding kept)
			for (int j = 0; j < N.y; j++) {

				int idx_in = i + j * (N.x / 2 + 1);

				*reinterpret_cast<ReIm3*>(pline + j * 3) = kernel[idx_in];
			}

			//fft on line
			fftw_execute(plan_fwd_y);

			for (int j = 0; j < N.y; j++) {

				ReIm3 value = *reinterpret_cast<ReIm3*>(pline + j * 3);

				kernel[i + j * (N.x / 2 + 1)] = value;
			}
		}
	};
	
	//-------------- CALCULATE DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	//no need to pass the actual cellsize values, just normalized values will do

	if (pbc_images.IsNull()) {

		if (!dtf.CalcDiagTens2D_Shifted_Irregular(D, n, N, kernels[index]->h_src / h_max, kernels[index]->h_dst / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		if (!dtf.CalcDiagTens2D_Shifted_Irregular_PBC(
			D, N, kernels[index]->h_src / h_max, kernels[index]->h_dst / h_max, kernels[index]->shift / h_max, 
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kdiag_cmpl);

	//-------------- CALCULATE OFF-DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	if (pbc_images.IsNull()) {

		if (!dtf.CalcOffDiagTens2D_Shifted_Irregular(D, n, N, kernels[index]->h_src / h_max, kernels[index]->h_dst / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		if (!dtf.CalcOffDiagTens2D_Shifted_Irregular_PBC(
			D, N, kernels[index]->h_src / h_max, kernels[index]->h_dst / h_max, kernels[index]->shift / h_max, 
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kodiag_cmpl);

	//-------------- CLEANUP
	
	fftw_destroy_plan(plan_fwd_x);
	fftw_destroy_plan(plan_fwd_y);

	fftw_free((double*)pline_real);
	fftw_free((fftw_complex*)pline);
	
	//Done
	return error;
}

//3D real kernels (Kdiag_real, and Kodiag_real, with full use of kernel symmetries)
BError DemagKernelCollection::Calculate_Demag_Kernels_3D_Self(int index)
{
	BError error(__FUNCTION__);

	//-------------- DEMAG TENSOR

	//Demag tensor components
	//
	// D11 D12 D13
	// D12 D22 D23
	// D13 D23 D33

	//D11, D22, D33 are the diagonal tensor elements
	//D12, D13, D23 are the off-diagonal tensor elements
	VEC<DBL3> D;

	if (!D.resize(N)) return error(BERROR_OUTOFMEMORY_NCRIT);

	//object used to compute tensor elements
	DemagTFunc dtf;

	//-------------- SETUP FFT

	double* pline_real = fftw_alloc_real(maximum(N.x, N.y, N.z) * 3);
	fftw_complex* pline = fftw_alloc_complex(maximum(N.x / 2 + 1, N.y, N.z) * 3);

	//make fft plans
	int dims_x[1] = { (int)N.x };
	int dims_y[1] = { (int)N.y };
	int dims_z[1] = { (int)N.z };

	fftw_plan plan_fwd_x = fftw_plan_many_dft_r2c(1, dims_x, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_y = fftw_plan_many_dft_r2c(1, dims_y, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_z = fftw_plan_many_dft_r2c(1, dims_z, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	//-------------- FFT REAL TENSOR INTO REAL KERNELS

	//lambda used to transform an input tensor into an output kernel
	auto tensor_to_kernel = [&](VEC<DBL3>& tensor, VEC<DBL3>& kernel, bool off_diagonal) -> void {

		//1. FFTs along x
		for (int k = 0; k < N.z; k++) {
			for (int j = 0; j < N.y; j++) {

				//write input into fft line (zero padding kept)
				for (int i = 0; i < N.x; i++) {

					int idx_in = i + j * N.x + k * N.x * N.y;

					*reinterpret_cast<DBL3*>(pline_real + i * 3) = tensor[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_x);

				//pack into tensor for next step
				for (int i = 0; i < N.x / 2 + 1; i++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + i * 3);

					if (!off_diagonal) {

						//even w.r.t. to x so output is purely real
						tensor[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y] = DBL3(value.x.Re, value.y.Re, value.z.Re);
					}
					else {

						//Dxy : odd x, Dxz : odd x, Dyz : even x
						tensor[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y] = DBL3(value.x.Im, value.y.Im, value.z.Re);
					}
				}
			}
		}

		//2. FFTs along y
		for (int k = 0; k < N.z; k++) {
			for (int i = 0; i < (N.x / 2 + 1); i++) {

				//fetch line from fft array (zero padding kept)
				for (int j = 0; j < N.y; j++) {

					int idx_in = i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y;

					*reinterpret_cast<DBL3*>(pline_real + j * 3) = tensor[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_y);

				//pack into lower half of tensor column for next step (keep same row and plane strides)
				for (int j = 0; j < N.y / 2 + 1; j++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + j * 3);

					if (!off_diagonal) {

						//even w.r.t. to y so output is purely real
						tensor[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1)] = DBL3(value.x.Re, value.y.Re, value.z.Re);
					}
					else {

						//Dxy : odd y, Dxz : even y, Dyz : odd y
						tensor[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1)] = DBL3(value.x.Im, value.y.Re, value.z.Im);
					}
				}
			}
		}

		//3. FFTs along z
		for (int j = 0; j < N.y / 2 + 1; j++) {
			for (int i = 0; i < N.x / 2 + 1; i++) {

				//fetch line from fft array (zero padding kept)
				for (int k = 0; k < N.z; k++) {

					int idx_in = i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);

					*reinterpret_cast<DBL3*>(pline_real + k * 3) = tensor[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_z);

				//pack into output kernels with reduced strides
				for (int k = 0; k < N.z / 2 + 1; k++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + k * 3);

					if (!off_diagonal) {

						//even w.r.t. to z so output is purely real
						kernel[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1)] = DBL3(value.x.Re, value.y.Re, value.z.Re);
					}
					else {

						//Dxy : even z, Dxz : odd z, Dyz : odd z
						//Also multiply by -1 since all off-diagonal tensor elements have been odd twice
						//The final output is thus purely real but we always treated the input as purely real even when it should have been purely imaginary
						//This means we need to account for i * i = -1 at the end
						kernel[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1)] = DBL3(-value.x.Re, -value.y.Im, -value.z.Im);
					}
				}
			}
		}
	};

	//-------------- CALCULATE DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	//no need to pass the actual cellsize values, just normalized values will do
	if (pbc_images.IsNull()) {

		//no pbcs in any dimension -> use these
		if (!dtf.CalcDiagTens3D(D, n, N, h / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		//pbcs used in at least one dimension
		if (!dtf.CalcDiagTens3D_PBC(
			D, N, h / h_max, 
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kdiag_real, false);

	//-------------- CALCULATE OFF-DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	if (pbc_images.IsNull()) {

		//no pbcs in any dimension -> use these
		if (!dtf.CalcOffDiagTens3D(D, n, N, h / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		//pbcs used in at least one dimension
		if (!dtf.CalcOffDiagTens3D_PBC(
			D, N, h / h_max, 
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kodiag_real, true);

	//-------------- CLEANUP

	fftw_destroy_plan(plan_fwd_x);
	fftw_destroy_plan(plan_fwd_y);
	fftw_destroy_plan(plan_fwd_z);

	fftw_free((double*)pline_real);
	fftw_free((fftw_complex*)pline);

	//Done
	return error;
}

//3D layers, z shift only : Kernels can be stored with use of kernel symmetries (but still complex).
BError DemagKernelCollection::Calculate_Demag_Kernels_3D_zShifted(int index)
{
	BError error(__FUNCTION__);

	//-------------- DEMAG TENSOR

	//Demag tensor components
	//
	// D11 D12 D13
	// D12 D22 D23
	// D13 D23 D33

	//D11, D22, D33 are the diagonal tensor elements
	//D12, D13, D23 are the off-diagonal tensor elements
	VEC<DBL3> D;

	if (!D.resize(N)) return error(BERROR_OUTOFMEMORY_NCRIT);

	//object used to compute tensor elements
	DemagTFunc dtf;

	//-------------- SETUP FFT

	double* pline_real = fftw_alloc_real(maximum(N.x, N.y, N.z) * 3);
	fftw_complex* pline = fftw_alloc_complex(maximum(N.x / 2 + 1, N.y, N.z) * 3);

	//make fft plans
	int dims_x[1] = { (int)N.x };
	int dims_y[1] = { (int)N.y };
	int dims_z[1] = { (int)N.z };

	fftw_plan plan_fwd_x = fftw_plan_many_dft_r2c(1, dims_x, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_y = fftw_plan_many_dft(1, dims_y, 3,
		pline, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_FORWARD, FFTW_PATIENT);

	fftw_plan plan_fwd_z = fftw_plan_many_dft(1, dims_z, 3,
		pline, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_FORWARD, FFTW_PATIENT);

	//-------------- FFT REAL TENSOR INTO REAL KERNELS

	//lambda used to transform an input tensor into an output kernel
	auto tensor_to_kernel = [&](VEC<DBL3>& tensor, VEC<ReIm3>& kernel) -> void {

		//1. FFTs along x
		for (int k = 0; k < N.z; k++) {
			for (int j = 0; j < N.y; j++) {

				//write input into fft line (zero padding kept)
				for (int i = 0; i < N.x; i++) {

					int idx_in = i + j * N.x + k * N.x * N.y;

					*reinterpret_cast<DBL3*>(pline_real + i * 3) = tensor[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_x);

				//pack into scratch space
				for (int i = 0; i < N.x / 2 + 1; i++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + i * 3);

					F[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y] = value;
				}
			}
		}

		//2. FFTs along y
		for (int k = 0; k < N.z; k++) {
			for (int i = 0; i < (N.x / 2 + 1); i++) {

				//fetch line from fft array (zero padding kept)
				for (int j = 0; j < N.y; j++) {

					int idx_in = i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y;

					*reinterpret_cast<ReIm3*>(pline + j * 3) = F[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_y);

				//pack into scratch space
				for (int j = 0; j < N.y / 2 + 1; j++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + j * 3);

					F[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1)] = value;
				}
			}
		}

		//3. FFTs along z
		for (int j = 0; j < N.y / 2 + 1; j++) {
			for (int i = 0; i < N.x / 2 + 1; i++) {

				//fetch line from fft array (zero padding kept)
				for (int k = 0; k < N.z; k++) {

					int idx_in = i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);

					*reinterpret_cast<ReIm3*>(pline + k * 3) = F[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_z);

				//pack into output kernels with reduced strides
				for (int k = 0; k < N.z / 2 + 1; k++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + k * 3);

					kernel[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1)] = value;
				}
			}
		}
	};

	//-------------- CALCULATE DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	//no need to pass the actual cellsize values, just normalized values will do
	if (pbc_images.IsNull()) {

		//no pbcs in any dimension -> use these
		if (!dtf.CalcDiagTens3D_Shifted(D, n, N, h / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		//pbcs used in at least one dimension
		if (!dtf.CalcDiagTens3D_Shifted_PBC(
			D, N, h / h_max, kernels[index]->shift / h_max, 
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kdiag_cmpl);

	//-------------- CALCULATE OFF-DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	if (pbc_images.IsNull()) {

		//no pbcs in any dimension -> use these
		if (!dtf.CalcOffDiagTens3D_Shifted(D, n, N, h / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		//pbcs used in at least one dimension
		if (!dtf.CalcOffDiagTens3D_Shifted_PBC(
			D, N, h / h_max, kernels[index]->shift / h_max, 
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kodiag_cmpl);

	//-------------- CLEANUP

	fftw_destroy_plan(plan_fwd_x);
	fftw_destroy_plan(plan_fwd_y);
	fftw_destroy_plan(plan_fwd_z);

	fftw_free((double*)pline_real);
	fftw_free((fftw_complex*)pline);

	//Done
	return error;
}

//3D layers, z shift only : Kernels can be stored with use of kernel symmetries (but still complex).
BError DemagKernelCollection::Calculate_Demag_Kernels_3D_xShifted(int index)
{
	BError error(__FUNCTION__);

	//-------------- DEMAG TENSOR

	//Demag tensor components
	//
	// D11 D12 D13
	// D12 D22 D23
	// D13 D23 D33

	//D11, D22, D33 are the diagonal tensor elements
	//D12, D13, D23 are the off-diagonal tensor elements
	VEC<DBL3> D;

	if (!D.resize(N)) return error(BERROR_OUTOFMEMORY_NCRIT);

	//object used to compute tensor elements
	DemagTFunc dtf;

	//-------------- SETUP FFT

	double* pline_real = fftw_alloc_real(maximum(N.x, N.y, N.z) * 3);
	fftw_complex* pline = fftw_alloc_complex(maximum(N.x / 2 + 1, N.y, N.z) * 3);

	//make fft plans
	int dims_x[1] = { (int)N.x };
	int dims_y[1] = { (int)N.y };
	int dims_z[1] = { (int)N.z };

	fftw_plan plan_fwd_x = fftw_plan_many_dft_r2c(1, dims_x, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_y = fftw_plan_many_dft(1, dims_y, 3,
		pline, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_FORWARD, FFTW_PATIENT);

	fftw_plan plan_fwd_z = fftw_plan_many_dft(1, dims_z, 3,
		pline, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_FORWARD, FFTW_PATIENT);

	//-------------- FFT REAL TENSOR INTO REAL KERNELS

	//lambda used to transform an input tensor into an output kernel
	auto tensor_to_kernel = [&](VEC<DBL3>& tensor, VEC<ReIm3>& kernel) -> void {

		//1. FFTs along x
		for (int k = 0; k < N.z; k++) {
			for (int j = 0; j < N.y; j++) {

				//write input into fft line (zero padding kept)
				for (int i = 0; i < N.x; i++) {

					int idx_in = i + j * N.x + k * N.x * N.y;

					*reinterpret_cast<DBL3*>(pline_real + i * 3) = tensor[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_x);

				//pack into scratch space
				for (int i = 0; i < N.x / 2 + 1; i++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + i * 3);

					F[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y] = value;
				}
			}
		}

		//2. FFTs along y
		for (int k = 0; k < N.z; k++) {
			for (int i = 0; i < (N.x / 2 + 1); i++) {

				//fetch line from fft array (zero padding kept)
				for (int j = 0; j < N.y; j++) {

					int idx_in = i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y;

					*reinterpret_cast<ReIm3*>(pline + j * 3) = F[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_y);

				//pack into scratch space
				for (int j = 0; j < N.y / 2 + 1; j++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + j * 3);

					F[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1)] = value;
				}
			}
		}

		//3. FFTs along z
		for (int j = 0; j < N.y / 2 + 1; j++) {
			for (int i = 0; i < N.x / 2 + 1; i++) {

				//fetch line from fft array (zero padding kept)
				for (int k = 0; k < N.z; k++) {

					int idx_in = i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1);

					*reinterpret_cast<ReIm3*>(pline + k * 3) = F[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_z);

				//pack into output kernels with reduced strides
				for (int k = 0; k < N.z / 2 + 1; k++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + k * 3);

					kernel[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * (N.y / 2 + 1)] = value;
				}
			}
		}
	};

	//-------------- CALCULATE DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	//no need to pass the actual cellsize values, just normalized values will do
	if (pbc_images.IsNull()) {

		//no pbcs in any dimension -> use these
		if (!dtf.CalcDiagTens3D_Shifted(D, n, N, h / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		//pbcs used in at least one dimension
		if (!dtf.CalcDiagTens3D_Shifted_PBC(
			D, N, h / h_max, kernels[index]->shift / h_max,
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kdiag_cmpl);

	//-------------- CALCULATE OFF-DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	if (pbc_images.IsNull()) {

		//no pbcs in any dimension -> use these
		if (!dtf.CalcOffDiagTens3D_Shifted(D, n, N, h / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		//pbcs used in at least one dimension
		if (!dtf.CalcOffDiagTens3D_Shifted_PBC(
			D, N, h / h_max, kernels[index]->shift / h_max,
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kodiag_cmpl);

	//-------------- CLEANUP

	fftw_destroy_plan(plan_fwd_x);
	fftw_destroy_plan(plan_fwd_y);
	fftw_destroy_plan(plan_fwd_z);

	fftw_free((double*)pline_real);
	fftw_free((fftw_complex*)pline);

	//Done
	return error;
}

//3D complex kernels (Kdiag_cmpl, and Kodiag_cmpl, without any kernel symmetries)
BError DemagKernelCollection::Calculate_Demag_Kernels_3D_Complex_Full(int index)
{
	BError error(__FUNCTION__);
	
	//-------------- DEMAG TENSOR

	//Demag tensor components
	//
	// D11 D12 D13
	// D12 D22 D23
	// D13 D23 D33

	//D11, D22, D33 are the diagonal tensor elements
	//D12, D13, D23 are the off-diagonal tensor elements
	VEC<DBL3> D;

	if (!D.resize(N)) return error(BERROR_OUTOFMEMORY_NCRIT);

	//object used to compute tensor elements
	DemagTFunc dtf;

	//-------------- SETUP FFT

	double* pline_real = fftw_alloc_real(maximum(N.x, N.y, N.z) * 3);
	fftw_complex* pline = fftw_alloc_complex(maximum(N.x / 2 + 1, N.y, N.z) * 3);

	//make fft plans
	int dims_x[1] = { (int)N.x };
	int dims_y[1] = { (int)N.y };
	int dims_z[1] = { (int)N.z };

	fftw_plan plan_fwd_x = fftw_plan_many_dft_r2c(1, dims_x, 3,
		pline_real, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_PATIENT);

	fftw_plan plan_fwd_y = fftw_plan_many_dft(1, dims_y, 3,
		pline, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_FORWARD, FFTW_PATIENT);

	fftw_plan plan_fwd_z = fftw_plan_many_dft(1, dims_z, 3,
		pline, nullptr, 3, 1,
		pline, nullptr, 3, 1,
		FFTW_FORWARD, FFTW_PATIENT);

	//-------------- FFT REAL TENSOR INTO REAL KERNELS

	//lambda used to transform an input tensor into an output kernel
	auto tensor_to_kernel = [&](VEC<DBL3>& tensor, VEC<ReIm3>& kernel) -> void {

		//1. FFTs along x
		for (int k = 0; k < N.z; k++) {
			for (int j = 0; j < N.y; j++) {

				//write input into fft line (zero padding kept)
				for (int i = 0; i < N.x; i++) {

					int idx_in = i + j * N.x + k * N.x * N.y;

					*reinterpret_cast<DBL3*>(pline_real + i * 3) = tensor[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_x);

				//pack into scratch space
				for (int i = 0; i < N.x / 2 + 1; i++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + i * 3);

					F[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y] = value;
				}
			}
		}

		//2. FFTs along y
		for (int k = 0; k < N.z; k++) {
			for (int i = 0; i < (N.x / 2 + 1); i++) {

				//fetch line from fft array (zero padding kept)
				for (int j = 0; j < N.y; j++) {

					int idx_in = i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y;

					*reinterpret_cast<ReIm3*>(pline + j * 3) = F[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_y);

				//pack into scratch space
				for (int j = 0; j < N.y; j++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + j * 3);

					F[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y] = value;
				}
			}
		}

		//3. FFTs along z
		for (int j = 0; j < N.y; j++) {
			for (int i = 0; i < N.x / 2 + 1; i++) {

				//fetch line from fft array (zero padding kept)
				for (int k = 0; k < N.z; k++) {

					int idx_in = i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y;

					*reinterpret_cast<ReIm3*>(pline + k * 3) = F[idx_in];
				}

				//fft on line
				fftw_execute(plan_fwd_z);

				//pack into output kernels with reduced strides
				for (int k = 0; k < N.z; k++) {

					ReIm3 value = *reinterpret_cast<ReIm3*>(pline + k * 3);

					kernel[i + j * (N.x / 2 + 1) + k * (N.x / 2 + 1) * N.y] = value;
				}
			}
		}
	};

	//-------------- CALCULATE DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	if (pbc_images.IsNull()) {

		//no need to pass the actual cellsize values, just normalized values will do
		if (!dtf.CalcDiagTens3D_Shifted(D, n, N, h / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		if (!dtf.CalcDiagTens3D_Shifted_PBC(
			D, N, h / h_max, kernels[index]->shift / h_max,
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kdiag_cmpl);

	//-------------- CALCULATE OFF-DIAGONAL TENSOR ELEMENTS THEN TRANSFORM INTO KERNEL

	if (pbc_images.IsNull()) {

		if (!dtf.CalcOffDiagTens3D_Shifted(D, n, N, h / h_max, kernels[index]->shift / h_max)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}
	else {

		if (!dtf.CalcOffDiagTens3D_Shifted_PBC(
			D, N, h / h_max, kernels[index]->shift / h_max,
			true, ASYMPTOTIC_DISTANCE, pbc_images.x, pbc_images.y, pbc_images.z)) return error(BERROR_OUTOFMEMORY_NCRIT);
	}

	tensor_to_kernel(D, kernels[index]->Kodiag_cmpl);

	//-------------- CLEANUP

	fftw_destroy_plan(plan_fwd_x);
	fftw_destroy_plan(plan_fwd_y);
	fftw_destroy_plan(plan_fwd_z);

	fftw_free((double*)pline_real);
	fftw_free((fftw_complex*)pline);

	//Done
	return error;
}

#endif