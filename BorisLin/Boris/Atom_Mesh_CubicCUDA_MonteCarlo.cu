//Use thrust library for access to sorting algorithm
//NOTE: the thrust headers have to be included before anything else, otherwise c++ compiler gets tripped up and starts spewing errors
//Ideally you'd have a sorting algorithm incorporated in cuVEC, together with a sort-by-key variant - nice to have, but thrust will do for now.
//Not difficult, most promising is radix sort (https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2009.01542.x) - sounds like a 1 or 2 day project to implement so do it when time allows.
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "Atom_Mesh_CubicCUDA.h"

#if COMPILECUDA == 1

#ifdef MESH_COMPILATION_ATOM_CUBIC && ATOMISTIC == 1

#include "Reduction.cuh"
#include "cu_prng.cuh"
#include "mcuVEC_halo.cuh"

#include "Atom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"

__global__ void Zero_MCAux_SetField_SCCUDA(cuBReal& aux_real, cuBReal& aux_real2, ManagedAtom_MeshCUDA& cuaMesh, cuReal3& Ha)
{
	if (threadIdx.x == 0) aux_real = 0.0;
	else if (threadIdx.x == 1) aux_real2 = 0.0;
	else if (threadIdx.x == 2) cuaMesh.Ha_MC = Ha;
}

__global__ void Zero_MCAux_ZeroField_SCCUDA(cuBReal& aux_real, cuBReal& aux_real2, ManagedAtom_MeshCUDA& cuaMesh)
{
	if (threadIdx.x == 0) aux_real = 0.0;
	else if (threadIdx.x == 1) aux_real2 = 0.0;
	else if (threadIdx.x == 2) cuaMesh.Ha_MC = cuReal3();
}

///////////////////////////////////////////////////////////////
// PARALLEL MONTE-CARLO METROPOLIS

__global__ void Iterate_MonteCarloCUDA_Classic_Cubic_red_kernel(
	ManagedAtom_MeshCUDA& cuaMesh, 
	cuBorisRand<>& prng, 
	cuBReal mc_cone_angledeg, cuBReal& mc_acceptance_rate, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;

	cuSZ3& n = M1.n;

	cuBReal base_temperature = *cuaMesh.pbase_temperature;

	int num_moves = M1.get_nonempty_cells();

	cuBReal acceptance_rate = 0.0;

	//this method must be called with half-size : n.dim() / 2, i.e. <<< (n.dim() / 2 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
	//double idx : idx values will now take on even values
	int spin_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

	//ijk coordinates corresponding to idx
	cuINT3 ijk = cuINT3(spin_idx % n.x, (spin_idx / n.x) % n.y, spin_idx / (n.x*n.y));

	if (n.x % 2 == 1) {
		if (n.y % 2 == 0) {

			//nx is odd and ny is even : for red squares nudge on odd planes only
			spin_idx += (int)(n.z % 2);
		}
		//else : nx is odd and ny is odd, no nudge is needed
	}
	else {

		//nx is even : for red squares nudge on a) odd rows and even planes, and b) even rows and odd planes

		//red_nudge = true for odd rows and even planes or for even rows and odd planes - have to keep index on the checkerboard pattern
		bool red_nudge = (((ijk.j % 2) == 1 && (ijk.k % 2) == 0) || (((ijk.j % 2) == 0 && (ijk.k % 2) == 1)));

		spin_idx += (int)red_nudge;
	}
	
	//calculate only in non-empty and non-frozen cells
	if (spin_idx < n.dim() && M1.is_not_empty(spin_idx) && !M1.is_skipcell(spin_idx)) {

		//Picked spin is M1[spin_idx]
		cuReal3 M1_old = M1[spin_idx];

		//obtain rotated spin in a cone around the picked spin
		cuBReal theta_rot = prng.rand() * mc_cone_angledeg * (cuBReal)PI / 180.0;
		cuBReal phi_rot = prng.rand() * 2 * (cuBReal)PI;
		//Move spin in cone with uniform random probability distribution. This approach only requires 2 random numbers to be generated. 
		//Also using a Gaussian distribution to move spin around the initial spin is less efficient, requiring more steps to thermalize.
		cuReal3 M1_new = relrotate_polar(M1_old, theta_rot, phi_rot);

		//find energy change : new - old
		cuBReal energy_delta = 0.0;
		for (int midx = 0; midx < cuaMesh.num_SC_MCFuncs; midx++)
			if (cuaMesh.pSC_MCFuncs[midx]) energy_delta += (cuaMesh.*(cuaMesh.pSC_MCFuncs[midx]))(spin_idx, M1_new);

		//Compute acceptance probability
		cuBReal P_accept = 0.0, P = 1.0;
		if (base_temperature > 0.0) {

			P_accept = exp(-energy_delta / ((cuBReal)BOLTZMANN * base_temperature));
			//uniform random number between 0 and 1
			P = prng.rand();
		}
		else if (energy_delta < 0) P_accept = 1.0;

		if (P <= P_accept) {

			acceptance_rate = 1.0 / num_moves;

			//renormalize spin to mu_s to avoid floating point error creep
			cuBReal mu_s = *cuaMesh.pmu_s;
			cuaMesh.update_parameters_mcoarse(spin_idx, *cuaMesh.pmu_s, mu_s);
			M1_new.renormalize(mu_s);

			//set new spin
			M1[spin_idx] = M1_new;
		}
	}

	if (do_reduction) reduction_sum(0, 1, &acceptance_rate, mc_acceptance_rate);
}

__global__ void Iterate_MonteCarloCUDA_Classic_Cubic_black_kernel(
	ManagedAtom_MeshCUDA& cuaMesh,
	cuBorisRand<>& prng, 
	cuBReal mc_cone_angledeg, cuBReal& mc_acceptance_rate, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;

	cuSZ3& n = M1.n;

	cuBReal base_temperature = *cuaMesh.pbase_temperature;

	int num_moves = M1.get_nonempty_cells();

	cuBReal acceptance_rate = 0.0;

	//this method must be called with half-size : n.dim() / 2, i.e. <<< (n.dim() / 2 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
	//double idx : idx values will now take on even values
	int spin_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

	//ijk coordinates corresponding to idx
	cuINT3 ijk = cuINT3(spin_idx % n.x, (spin_idx / n.x) % n.y, spin_idx / (n.x*n.y));

	if (n.x % 2 == 1) {
		if (n.y % 2 == 0) {

			//nx is odd and ny is even : for black squares nudge on even planes only
			spin_idx += (int)(n.z % 2 == 0);
		}
		else {

			//nx is odd and ny is odd, nudge everything by 1 for black squares
			spin_idx++;
		}
	}
	else {

		//nx is even : for black squares nudge on a) even rows and even planes, and b) odd rows and odd planes

		//red_nudge = true for odd rows and even planes or for even rows and odd planes - have to keep index on the checkerboard pattern
		bool black_nudge = (((ijk.j % 2) == 0 && (ijk.k % 2) == 0) || (((ijk.j % 2) == 1 && (ijk.k % 2) == 1)));

		spin_idx += (int)black_nudge;
	}

	//calculate only in non-empty and non-frozen cells
	if (spin_idx < n.dim() && M1.is_not_empty(spin_idx) && !M1.is_skipcell(spin_idx)) {

		//Picked spin is M1[spin_idx]
		cuReal3 M1_old = M1[spin_idx];

		//obtain rotated spin in a cone around the picked spin
		cuBReal theta_rot = prng.rand() * mc_cone_angledeg * (cuBReal)PI / 180.0;
		cuBReal phi_rot = prng.rand() * 2 * (cuBReal)PI;
		//Move spin in cone with uniform random probability distribution. This approach only requires 2 random numbers to be generated. 
		//Also using a Gaussian distribution to move spin around the initial spin is less efficient, requiring more steps to thermalize.
		cuReal3 M1_new = relrotate_polar(M1_old, theta_rot, phi_rot);

		//find energy change : new - old
		cuBReal energy_delta = 0.0;
		for (int midx = 0; midx < cuaMesh.num_SC_MCFuncs; midx++)
			if (cuaMesh.pSC_MCFuncs[midx]) energy_delta += (cuaMesh.*(cuaMesh.pSC_MCFuncs[midx]))(spin_idx, M1_new);

		//Compute acceptance probability
		cuBReal P_accept = 0.0, P = 1.0;
		if (base_temperature > 0.0) {

			P_accept = exp(-energy_delta / ((cuBReal)BOLTZMANN * base_temperature));
			//uniform random number between 0 and 1
			P = prng.rand();
		}
		else if (energy_delta < 0) P_accept = 1.0;

		if (P <= P_accept) {

			acceptance_rate = 1.0 / num_moves;

			//renormalize spin to mu_s to avoid floating point error creep
			cuBReal mu_s = *cuaMesh.pmu_s;
			cuaMesh.update_parameters_mcoarse(spin_idx, *cuaMesh.pmu_s, mu_s);
			M1_new.renormalize(mu_s);

			//set new spin
			M1[spin_idx] = M1_new;
		}
	}

	if (do_reduction) reduction_sum(0, 1, &acceptance_rate, mc_acceptance_rate);
}

cuBReal Atom_Mesh_CubicCUDA::Iterate_MonteCarloCUDA_Classic(cuBReal mc_cone_angledeg, double target_acceptance_rate)
{
	if (pHa) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Zero_MCAux_SetField_SCCUDA <<< 1, CUDATHREADS >>> (mc_acceptance_rate(mGPU), cmc_M(mGPU), cuaMesh.get_deviceobject(mGPU), (*pHa)(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Zero_MCAux_ZeroField_SCCUDA <<< 1, CUDATHREADS >>> (mc_acceptance_rate(mGPU), cmc_M(mGPU), cuaMesh.get_deviceobject(mGPU));
		}
	}

	//must refresh halos before both red and black passes
	M1.exchange_halos();

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		//with acceptance rate reduction if mc_acceptance_reduction_counter is zero
		Iterate_MonteCarloCUDA_Classic_Cubic_red_kernel <<< (M1.device_size(mGPU) / 2 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(cuaMesh.get_deviceobject(mGPU), 
			prng.get_deviceobject(mGPU), 
			mc_cone_angledeg, mc_acceptance_rate(mGPU), mc_acceptance_reduction_counter == 0);
	}

	//must refresh halos before both red and black passes
	M1.exchange_halos();

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		Iterate_MonteCarloCUDA_Classic_Cubic_black_kernel <<< (M1.device_size(mGPU) / 2 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(cuaMesh.get_deviceobject(mGPU), 
			prng.get_deviceobject(mGPU), 
			mc_cone_angledeg, mc_acceptance_rate(mGPU), mc_acceptance_reduction_counter == 0);
	}

	if (mc_acceptance_reduction_counter == 0) {

		mc_acceptance_rate_last = mc_acceptance_rate.to_cpu();

		//is acceptance rate close enough to target acceptance? If yes don't do reduction next time.
		if (abs(target_acceptance_rate - mc_acceptance_rate_last) < MONTECARLO_ACCEPTANCETOLERANCE) mc_acceptance_reduction_counter = 1;

		return mc_acceptance_rate_last;
	}
	else {

		//increase counter until we come full circle : when mc_acceptance_reduction_counter becomes zero again we'll have to do reduction just to check.
		mc_acceptance_reduction_counter = (mc_acceptance_reduction_counter + 1) % MONTECARLO_REDUCTIONITERS;

		//return exact terget acceptance rate means cone angle will not be adjusted
		return target_acceptance_rate;
	}
}

///////////////////////////////////////////////////////////////
// PARALLEL CONSTRAINED MONTE-CARLO METROPOLIS - WITH REDUCTION

//Step 1
__global__ void MonteCarloCUDA_Constrained_Cubic_CalculateProjection(ManagedAtom_MeshCUDA& cuaMesh, cuBReal& cmc_M, cuReal3& cmc_n)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal cmc_M_ = 0.0;

	if (idx < M1.linear_size()) {

		if (M1.is_not_empty(idx)) {

			cmc_M_ = M1[idx] * cmc_n;
		}
	}

	reduction_sum(0, 1, &cmc_M_, cmc_M);
}

//Step 2
__global__ void MonteCarloCUDA_Constrained_Cubic_SetIndices(
	ManagedAtom_MeshCUDA& cuaMesh, 
	size_t num_reds, unsigned* mc_indices_red, unsigned* mc_shuf_red,
	size_t num_blacks, unsigned* mc_indices_black, unsigned* mc_shuf_black,
	cuBorisRand<>& prng)
{
	cuSZ3& n = cuaMesh.pM1->n;

	int spin_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (spin_idx < n.dim()) {

		//ijk coordinates corresponding to idx
		int i = spin_idx % n.x;
		int j = (spin_idx / n.x) % n.y;
		int k = spin_idx / (n.x*n.y);

		//red_nudge = true for odd rows and even planes or for even rows and odd planes - have to keep index on the checkerboard pattern
		bool red_nudge = (((j % 2) == 1 && (k % 2) == 0) || (((j % 2) == 0 && (k % 2) == 1)));

		//true for black square, false for red square
		bool black_square = (i - red_nudge) % 2;

		if (!black_square) {

			//Form red squares index (0, 1, 2 ...) from total index
			int even_rows = 0, odd_rows;

			if (k % 2 == 0) {

				even_rows = (j / 2) * (n.x / 2);
				odd_rows = (j - (j / 2)) * (n.x / 2 + n.x % 2);
			}
			else {

				even_rows = (j / 2) * (n.x / 2 + n.x % 2);
				odd_rows = (j - (j / 2)) * (n.x / 2);
			}

			int even_planes = (k / 2) * (n.x * n.y / 2);
			int odd_planes = (k - (k / 2)) * (n.x * n.y / 2 + (n.x * n.y) % 2);

			int idx_red = (i / 2) + (even_rows + odd_rows) + (even_planes + odd_planes);

			mc_indices_red[idx_red] = spin_idx;
			mc_shuf_red[idx_red] = prng.randi();
		}
		else {

			//Form black squares index (0, 1, 2 ...) from total index
			int even_rows = 0, odd_rows;

			if (k % 2 == 0) {

				even_rows = (j / 2) * (n.x / 2);
				odd_rows = (j - (j / 2)) * (n.x / 2 + n.x % 2);
			}
			else {

				even_rows = (j / 2) * (n.x / 2 + n.x % 2);
				odd_rows = (j - (j / 2)) * (n.x / 2);
			}

			int even_planes = (k / 2) * (n.x * n.y / 2);
			int odd_planes = (k - (k / 2)) * (n.x * n.y / 2 + (n.x * n.y) % 2);

			int idx_black = (i / 2) + (even_rows + odd_rows) + (even_planes + odd_planes);

			mc_indices_black[idx_black] = spin_idx;
			mc_shuf_black[idx_black] = prng.randi();
		}
	}
}

//Step 3
//Done with thrust::sort_by_key

//Step 4 (same kernel for red and black, just pass in the right indices!
__global__ void Iterate_MonteCarloCUDA_Constrained_Cubic_redblack_kernel(
	ManagedAtom_MeshCUDA& cuaMesh, 
	size_t num_points, unsigned* mc_indices, 
	cuBorisRand<>& prng,
	cuBReal& cmc_M, cuReal3& cmc_n,
	cuBReal mc_cone_angledeg, cuBReal& mc_acceptance_rate, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;

	cuSZ3& n = M1.n;

	cuBReal base_temperature = *cuaMesh.pbase_temperature;

	int num_moves = M1.get_nonempty_cells();

	cuBReal acceptance_rate = 0.0;

	//this kernel is launched with at least (num_points / 2 + 1) threads
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

	if (idx < num_points - 1) {

		//use pre-shuffled spins : each will be picked exactly once if even number; if odd, then one spin (random) will be left untouched.
		int spin_idx1 = mc_indices[idx];
		int spin_idx2 = mc_indices[idx + 1];

		//If there are empty cells then make sure to only pair non-empty ones
		if (M1.is_not_empty(spin_idx1) && M1.is_not_empty(spin_idx2) && !M1.is_skipcell(spin_idx1) && !M1.is_skipcell(spin_idx2)) {

			//Picked spins are M1[spin_idx1], M1[spin_idx2]
			cuReal3 M_old1 = M1[spin_idx1];
			cuReal3 M_old2 = M1[spin_idx2];

			//rotate to a system with x axis along cmc_n : easier to apply algorithm formulas this way
			//NOTE : don't rotate x axis towards cmc_n, but rotate cmc_n towards x axis, i.e. use invrotate_polar, not rotate_polar
			cuReal3 Mrot_old1 = invrotate_polar(M_old1, cmc_n);
			cuReal3 Mrot_old2 = invrotate_polar(M_old2, cmc_n);

			//obtain rotated spin in a cone around the first picked spin
			cuBReal theta_rot = prng.rand() * mc_cone_angledeg * (cuBReal)PI / 180.0;
			cuBReal phi_rot = prng.rand() * 2 * (cuBReal)PI;
			cuReal3 Mrot_new1 = relrotate_polar(Mrot_old1, theta_rot, phi_rot);

			//adjust second spin to keep required total moment direction
			cuReal3 Mrot_new2 = cuReal3(0.0, Mrot_old2.y + Mrot_old1.y - Mrot_new1.y, Mrot_old2.z + Mrot_old1.z - Mrot_new1.z);
			cuBReal sq2 = Mrot_new2.y * Mrot_new2.y + Mrot_new2.z * Mrot_new2.z;
			cuBReal sqnorm = M_old2.norm()*M_old2.norm();

			if (sq2 < sqnorm) {

				Mrot_new2.x = cu_get_sign(Mrot_old2.x) * sqrt(sqnorm - sq2);

				//Obtain new spins in original coordinate system, i.e. rotate back
				cuReal3 M_new1 = rotate_polar(Mrot_new1, cmc_n);
				cuReal3 M_new2 = rotate_polar(Mrot_new2, cmc_n);

				//find energy change : new - old
				cuBReal energy_delta = 0.0;
				for (int midx = 0; midx < cuaMesh.num_SC_MCFuncs; midx++)
					if (cuaMesh.pSC_MCFuncs[midx]) {
						energy_delta += (cuaMesh.*(cuaMesh.pSC_MCFuncs[midx]))(spin_idx1, M_new1);
						energy_delta += (cuaMesh.*(cuaMesh.pSC_MCFuncs[midx]))(spin_idx2, M_new2);
					}

				//use abs: since we're not updating cmc_M after every spin it can become negative above the Curie temperature. with the cmc_M_new > 0.0 check this will result in solver getting stuck
				cuBReal cmc_M_new = abs(cmc_M) + Mrot_new1.x + Mrot_new2.x - Mrot_old1.x - Mrot_old2.x;

				if (cmc_M_new > 0.0) {

					//Compute acceptance probability; make sure cmc_M is not zero otherwise we'll stop accepting anything and solver gets stuck
					cuBReal P_accept = 0.0, P = 1.0;
					if (base_temperature > 0.0) {

						if (cmc_M) P_accept = (cmc_M_new / cmc_M) * (cmc_M_new / cmc_M) * (abs(Mrot_old2.x) / abs(Mrot_new2.x)) * exp(-energy_delta / ((cuBReal)BOLTZMANN * base_temperature));
						else P_accept = (abs(Mrot_old2.x) / abs(Mrot_new2.x)) * exp(-energy_delta / ((cuBReal)BOLTZMANN * base_temperature));
						//uniform random number between 0 and 1
						P = prng.rand();
					}
					else if (energy_delta < 0) P_accept = 1.0;

					if (P <= P_accept) {

						//move accepted (x2 since we moved 2 spins)
						acceptance_rate += 2.0 / num_moves;

						//renormalize spins to mu_s to avoid floating point error creep
						cuBReal mu_s = *cuaMesh.pmu_s;
						cuaMesh.update_parameters_mcoarse(spin_idx1, *cuaMesh.pmu_s, mu_s);
						M_new1.renormalize(mu_s);
						
						cuaMesh.update_parameters_mcoarse(spin_idx2, *cuaMesh.pmu_s, mu_s);
						M_new2.renormalize(mu_s);

						//Set new spins
						M1[spin_idx1] = M_new1;
						M1[spin_idx2] = M_new2;
					}
				}
			}
		}
	}

	if (do_reduction) reduction_sum(0, 1, &acceptance_rate, mc_acceptance_rate);
}

//Take a constrained Monte Carlo Metropolis step in this atomistic mesh
cuBReal Atom_Mesh_CubicCUDA::Iterate_MonteCarloCUDA_Constrained(cuBReal mc_cone_angledeg, double target_acceptance_rate)
{
	if (pHa) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Zero_MCAux_SetField_SCCUDA <<< 1, CUDATHREADS >>> (mc_acceptance_rate(mGPU), cmc_M(mGPU), cuaMesh.get_deviceobject(mGPU), (*pHa)(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Zero_MCAux_ZeroField_SCCUDA <<< 1, CUDATHREADS >>> (mc_acceptance_rate(mGPU), cmc_M(mGPU), cuaMesh.get_deviceobject(mGPU));
		}
	}
	
	//1. Calculate cmc_M using reduction over the mesh (total moment along constrained direction)

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		MonteCarloCUDA_Constrained_Cubic_CalculateProjection <<< (M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
			(cuaMesh.get_deviceobject(mGPU), cmc_M(mGPU), cmc_n(mGPU));
	}

	//2. Set red and black indices and generate random numbers for shuffling (all in one kernel launch)

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		cuSZ3 nd = M1.device_n(mGPU);

		//number of red and black squares
		int num_reds = (nd.z / 2) * (nd.x * nd.y / 2) + (nd.z - (nd.z / 2)) * (nd.x * nd.y / 2 + (nd.x * nd.y) % 2);
		int num_blacks = (nd.z / 2) * (nd.x * nd.y / 2 + (nd.x * nd.y) % 2) + (nd.z - (nd.z / 2)) * (nd.x * nd.y / 2);

		//make sure indices array has correct memory allocated
		if (mc_indices_red.size(mGPU) != num_reds) {

			mc_indices_red.resize(mGPU, num_reds);
			mc_shuf_red.resize(mGPU, num_reds);
		}

		if (mc_indices_black.size(mGPU) != num_blacks) {

			mc_indices_black.resize(mGPU, num_blacks);
			mc_shuf_black.resize(mGPU, num_blacks);
		}

		MonteCarloCUDA_Constrained_Cubic_SetIndices <<< (M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(cuaMesh.get_deviceobject(mGPU),
			mc_indices_red.size(mGPU), mc_indices_red(mGPU), mc_shuf_red(mGPU),
			mc_indices_black.size(mGPU), mc_indices_black(mGPU), mc_shuf_black(mGPU),
			prng.get_deviceobject(mGPU));
	}

	//3. Sort-based shuffle of indices (red and black separately - two kernel launches)

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		//get thrust device pointers from cu_arr so we can use thrust sort
		thrust::device_ptr<unsigned> dev_ptr_mc_indices_red(mc_indices_red.get_array(mGPU));
		thrust::device_ptr<unsigned> dev_ptr_mc_shuf_red(mc_shuf_red.get_array(mGPU));
		thrust::device_ptr<unsigned> dev_ptr_mc_indices_black(mc_indices_black.get_array(mGPU));
		thrust::device_ptr<unsigned> dev_ptr_mc_shuf_black(mc_shuf_black.get_array(mGPU));

		thrust::sort_by_key(dev_ptr_mc_shuf_red, dev_ptr_mc_shuf_red + mc_indices_red.size(mGPU), dev_ptr_mc_indices_red);
		thrust::sort_by_key(dev_ptr_mc_shuf_black, dev_ptr_mc_shuf_black + mc_indices_black.size(mGPU), dev_ptr_mc_indices_black);
	}

	//4. Red and black CMC passes - two kernel launches

	//must refresh halos before both red and black passes
	M1.exchange_halos();

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		//red
		Iterate_MonteCarloCUDA_Constrained_Cubic_redblack_kernel <<< (mc_indices_red.size(mGPU) / 2 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(cuaMesh.get_deviceobject(mGPU),
			mc_indices_red.size(mGPU), mc_indices_red(mGPU),
			prng.get_deviceobject(mGPU),
			cmc_M(mGPU), cmc_n(mGPU),
			mc_cone_angledeg, mc_acceptance_rate(mGPU), mc_acceptance_reduction_counter == 0);
	}

	//must refresh halos before both red and black passes
	M1.exchange_halos();

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		//black
		Iterate_MonteCarloCUDA_Constrained_Cubic_redblack_kernel <<< (mc_indices_black.size(mGPU) / 2 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(cuaMesh.get_deviceobject(mGPU),
			mc_indices_black.size(mGPU), mc_indices_black(mGPU),
			prng.get_deviceobject(mGPU),
			cmc_M(mGPU), cmc_n(mGPU),
			mc_cone_angledeg, mc_acceptance_rate(mGPU), mc_acceptance_reduction_counter == 0);
	}

	if (mc_acceptance_reduction_counter == 0) {

		mc_acceptance_rate_last = mc_acceptance_rate.to_cpu_sum() / mGPU.get_num_devices();

		//is acceptance rate close enough to target acceptance? If yes don't do reduction next time.
		if (abs(target_acceptance_rate - mc_acceptance_rate_last) < MONTECARLO_ACCEPTANCETOLERANCE) mc_acceptance_reduction_counter = 1;

		return mc_acceptance_rate_last;
	}
	else {

		//increase counter until we come full circle : when mc_acceptance_reduction_counter becomes zero again we'll have to do reduction just to check.
		mc_acceptance_reduction_counter = (mc_acceptance_reduction_counter + 1) % MONTECARLO_REDUCTIONITERS;

		//return exact terget acceptance rate means cone angle will not be adjusted
		return target_acceptance_rate;
	}
}

#endif

#endif