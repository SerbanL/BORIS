#include "Atom_Mesh_CubicCUDA.h"

#if COMPILECUDA == 1

#ifdef MESH_COMPILATION_ATOM_CUBIC

#include "Reduction.cuh"

#include "Atom_MeshParamsControlCUDA.h"

__global__ void SetField_SCHisto_CUDA(ManagedAtom_MeshCUDA& cuaMesh, cuReal3& Ha)
{
	if (threadIdx.x == 0) {

		//set applied field for MC
		cuaMesh.Ha_MC = Ha;
	}
}

__global__ void ZeroField_SCHisto_CUDA(ManagedAtom_MeshCUDA& cuaMesh)
{
	if (threadIdx.x == 0) {

		//set applied field for MC
		cuaMesh.Ha_MC = cuReal3();
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void GetThermodynamicAverageMagnetization_kernel(ManagedAtom_MeshCUDA& cuaMesh, cuRect rectangle, cuBReal& Z, cuReal3& Mthav)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC_VC<cuBReal>& Temp = *cuaMesh.pTemp;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal Z_value = 0.0;
	cuReal3 Mthav_value = cuReal3();
	bool include_in_reduction = false;

	cuReal3 pos;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			pos = M1.cellidx_to_position(idx);

			cuBReal J = *cuaMesh.pJ;
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pJ, J);
			cuBReal E0 = -6 * J;

			cuBReal Ei = 0.0;
			for (int midx = 0; midx < cuaMesh.num_SC_MCFuncs; midx++)
				if (cuaMesh.pSC_MCFuncs[midx]) Ei += (cuaMesh.*(cuaMesh.pSC_MCFuncs[midx]))(idx, cuReal3());

			cuBReal Temperature;
			if (Temp.linear_size()) Temperature = Temp[M1.cellidx_to_position(idx)];
			else Temperature = *cuaMesh.pbase_temperature;

			cuBReal w = exp(-(Ei - E0) / ((cuBReal)BOLTZMANN * Temperature));
			Z_value = w;

			Mthav_value = w * M1[idx];

			include_in_reduction = true;
		}
	}

	reduction_sum(0, 1, &Z_value, Z, include_in_reduction && rectangle.contains(pos));
	reduction_sum(0, 1, &Mthav_value, Mthav, include_in_reduction && rectangle.contains(pos));
}

//calculate thermodynamic average of magnetization
cuReal3 Atom_Mesh_CubicCUDA::GetThermodynamicAverageMagnetization(cuRect rectangle)
{
	if (rectangle.IsNull()) rectangle = meshRect;

	Zero_aux_values();

	if (pHa) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			SetField_SCHisto_CUDA <<< 1, CUDATHREADS >>> (cuaMesh.get_deviceobject(mGPU), (*pHa)(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			ZeroField_SCHisto_CUDA <<< 1, CUDATHREADS >>> (cuaMesh.get_deviceobject(mGPU));
		}
	}

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		GetThermodynamicAverageMagnetization_kernel <<< (M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(cuaMesh.get_deviceobject(mGPU), rectangle, aux_real(mGPU), aux_real3(mGPU));
	}

	return aux_real3.to_cpu() * (MUB / h.dim()) / aux_real.to_cpu();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Get_ThAvHistogram_preaverage_kernel(
	ManagedAtom_MeshCUDA& cuaMesh, cuVEC<cuReal3>& auxVEC_w, cuVEC<cuBReal>& auxVEC_Z, cuINT3 num_av_cells, cuINT3 av_cell_dims)
{
	//launched with num_av_cells.dim(), (1024 or CUDATHREADS) kernel dimensions
	//i.e there are num_av_cells.dim() segments, each of size of av_cell_dims.dim()

	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC_VC<cuBReal>& Temp = *cuaMesh.pTemp;

	//segment size
	size_t K = av_cell_dims.dim();

	//linear index in this segment, starting at threadIdx.x value
	int linear_idx = threadIdx.x;

	//partial segment sum in this thread
	cuReal3 sum_w = cuReal3();
	cuBReal sum_Z = 0.0;

	//each segment receives up to 1024 worker threads. first use them to load all input data in current segment.
	while (linear_idx < K) {

		//segment ijk values
		int i_seg = blockIdx.x % num_av_cells.x;
		int j_seg = (blockIdx.x / num_av_cells.x) % num_av_cells.y;
		int k_seg = blockIdx.x / (num_av_cells.x * num_av_cells.y);

		//convert linear segment index to cuvec ijk index for this segment
		int i = linear_idx % av_cell_dims.x;
		int j = (linear_idx / av_cell_dims.x) % av_cell_dims.y;
		int k = linear_idx / (av_cell_dims.x * av_cell_dims.y);

		//finally required ijk index in cuvec
		cuINT3 ijk = cuINT3(i_seg * av_cell_dims.i + i, j_seg * av_cell_dims.j + j, k_seg * av_cell_dims.k + k);
		int idx = ijk.i + ijk.j * M1.n.x + ijk.k * M1.n.x*M1.n.y;

		if (idx < M1.linear_size() && M1.is_not_empty(idx)) {
			
			cuBReal J = *cuaMesh.pJ;
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pJ, J);
			cuBReal E0 = -6 * J;

			cuBReal Ei = 0.0;
			for (int midx = 0; midx < cuaMesh.num_SC_MCFuncs; midx++)
				if (cuaMesh.pSC_MCFuncs[midx]) Ei += (cuaMesh.*(cuaMesh.pSC_MCFuncs[midx]))(idx, cuReal3());

			cuBReal Temperature;
			if (Temp.linear_size()) Temperature = Temp[M1.cellidx_to_position(idx)];
			else Temperature = *cuaMesh.pbase_temperature;

			cuBReal w = exp(-(Ei - E0) / ((cuBReal)BOLTZMANN * Temperature));
			
			sum_Z += w;
			
			sum_w += w * M1[idx];
		}

		linear_idx += blockDim.x;
	}

	//now reduced all partial segment sums in this block
	reduction_sum(0, 1, &sum_w, auxVEC_w[blockIdx.x]);
	reduction_sum(0, 1, &sum_Z, auxVEC_Z[blockIdx.x]);
}

__global__ void Get_ThAvHistogram_preaverage_finish_kernel(cuVEC<cuReal3>& auxVEC_w, cuVEC<cuBReal>& auxVEC_Z)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < auxVEC_w.linear_size()) {

		if (auxVEC_Z[idx]) auxVEC_w[idx] /= auxVEC_Z[idx];
	}
}

bool Atom_Mesh_CubicCUDA::Get_ThAvHistogram(std::vector<double>& histogram_x, std::vector<double>& histogram_p, int num_bins, double& min, double& max, cuINT3 macrocell_dims)
{
	//First do thermal cell-wise pre-averaging

	//allocate required memory for auxVEC
	cuINT3 num_av_cells = round((cuReal3)n / macrocell_dims);
	auxVEC_cuReal3.assign(num_av_cells, cuReal3());
	auxVEC_cuBReal.assign(num_av_cells, 0.0);

	if (pHa) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			SetField_SCHisto_CUDA <<< 1, CUDATHREADS >>> (cuaMesh.get_deviceobject(mGPU), (*pHa)(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			ZeroField_SCHisto_CUDA <<< 1, CUDATHREADS >>> (cuaMesh.get_deviceobject(mGPU));
		}
	}

	//cell-wise pre-averaging
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Get_ThAvHistogram_preaverage_kernel <<< auxVEC_cuReal3.device_size(mGPU), CUDATHREADS >>>
			(cuaMesh.get_deviceobject(mGPU), auxVEC_cuReal3.get_deviceobject(mGPU), auxVEC_cuBReal.get_deviceobject(mGPU), num_av_cells, macrocell_dims);
	}

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Get_ThAvHistogram_preaverage_finish_kernel <<< (auxVEC_cuReal3.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(auxVEC_cuReal3.get_deviceobject(mGPU), auxVEC_cuBReal.get_deviceobject(mGPU));
	}

	//get histogram from auxVEC
	return auxVEC_cuReal3.get_mag_histogram(histogram_x, histogram_p, num_bins, min, max, num_av_cells.dim());
}

//As for Get_AngHistogram, but use thermal averaging in each macrocell
bool Atom_Mesh_CubicCUDA::Get_ThAvAngHistogram(std::vector<double>& histogram_x, std::vector<double>& histogram_p, int num_bins, double& min, double& max, cuINT3 macrocell_dims, cuReal3 ndir)
{
	//First do thermal cell-wise pre-averaging

	//allocate required memory for auxVEC
	cuINT3 num_av_cells = round((cuReal3)n / macrocell_dims);
	auxVEC_cuReal3.assign(num_av_cells, cuReal3());
	auxVEC_cuBReal.assign(num_av_cells, 0.0);

	//cell-wise pre-averaging
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Get_ThAvHistogram_preaverage_kernel <<< auxVEC_cuReal3.device_size(mGPU), CUDATHREADS >>>
			(cuaMesh.get_deviceobject(mGPU), auxVEC_cuReal3.get_deviceobject(mGPU), auxVEC_cuBReal.get_deviceobject(mGPU), num_av_cells, macrocell_dims);
	}

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Get_ThAvHistogram_preaverage_finish_kernel <<< (auxVEC_cuReal3.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(auxVEC_cuReal3.get_deviceobject(mGPU), auxVEC_cuBReal.get_deviceobject(mGPU));
	}

	if (ndir.IsNull()) ndir = GetThermodynamicAverageMagnetization(cuRect()).normalized();

	//get histogram from auxVEC
	return auxVEC_cuReal3.get_ang_histogram(histogram_x, histogram_p, num_bins, min, max, num_av_cells.dim(), cuINT3(1), ndir);
}

#endif

#endif