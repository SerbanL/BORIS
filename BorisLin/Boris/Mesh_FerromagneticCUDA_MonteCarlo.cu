#include "Mesh_FerromagneticCUDA.h"

#if COMPILECUDA == 1

#ifdef MESH_COMPILATION_FERROMAGNETIC

#include "MeshCUDA.h"
#include "MeshParamsControlCUDA.h"

#include "Reduction.cuh"
#include "mcuVEC_halo.cuh"

__global__ void Zero_MCAux_SetField_FMCUDA(cuBReal& aux_real, ManagedMeshCUDA& cuMesh, cuReal3& Ha)
{
	if (threadIdx.x == 0) aux_real = 0.0;
	//set applied field for MC
	else if (threadIdx.x == 1) cuMesh.Ha_MC = Ha;
}

__global__ void Zero_MCAux_ZeroField_FMCUDA(cuBReal& aux_real, ManagedMeshCUDA& cuMesh)
{
	if (threadIdx.x == 0) aux_real = 0.0;
	//set applied field for MC
	else if (threadIdx.x == 1) cuMesh.Ha_MC = cuReal3();
}

///////////////////////////////////////////////////////////////
// PARALLEL MONTE-CARLO METROPOLIS

__global__ void Iterate_MonteCarloCUDA_Classic_FM_red_kernel(
	ManagedMeshCUDA& cuMesh, 
	cuBorisRand<>& prng, 
	cuBReal mc_cone_angledeg, cuBReal& mc_acceptance_rate, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuBReal>& Temp = *cuMesh.pTemp;

	cuSZ3& n = M.n;

	int num_moves = M.get_nonempty_cells();

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
	if (spin_idx < n.dim() && M.is_not_empty(spin_idx) && !M.is_skipcell(spin_idx)) {

		cuBReal Ms_val = *cuMesh.pMs;
		cuBReal susrel_val = *cuMesh.psusrel;
		cuMesh.update_parameters_mcoarse(spin_idx, *cuMesh.pMs, Ms_val, *cuMesh.psusrel, susrel_val);

		cuBReal Temperature;
		if (Temp.linear_size()) Temperature = Temp[M.cellidx_to_position(spin_idx)];
		else Temperature = *cuMesh.pbase_temperature;

		cuBReal Ms0 = cuMesh.pMs->get0();
		cuBReal me = Ms_val / Ms0;

		//Picked spin is M[spin_idx]
		cuReal3 M_old = M[spin_idx];

		//obtain rotated spin in a cone around the picked spin
		cuBReal theta_rot = prng.rand() * mc_cone_angledeg * PI / 180.0;
		cuBReal phi_rot = prng.rand() * 2 * PI;
		//Move spin in cone with uniform random probability distribution.
		cuReal3 M_new = relrotate_polar(M_old, theta_rot, phi_rot);

		//now allow magnetization length to change slightly with a Gaussian pdf around current value with sigma value from the normal distribution of P(m^2).
		if (Temperature > 0.0) {

			cuBReal sigma = 2 * me*sqrt(susrel_val*(cuBReal)BOLTZMANN*Temperature / (M.h.dim() * Ms0));
			if (Temperature >= *cuMesh.pT_Curie || sigma > 0.03) sigma = 0.03;
			M_new *= 1 + (prng.rand() * 2 * sigma - sigma);
		}

		//1. Find energy change
		cuBReal energy_delta = 0.0;
		for (int midx = 0; midx < cuMesh.num_FM_MCFuncs; midx++)
			if (cuMesh.pFM_MCFuncs[midx]) energy_delta += (cuMesh.*(cuMesh.pFM_MCFuncs[midx]))(spin_idx, M_new);

		//2. Find contribution to free energy change from longitudinal susceptibility
		cuReal3 m = M_old / Ms0;
		cuReal3 m_new = M_new / Ms0;

		if (Temperature > 0.0 && Temperature <= *cuMesh.pT_Curie) {

			cuBReal diff = m * m - me * me;
			cuBReal diff_new = m_new * m_new - me * me;

			energy_delta += M.h.dim() * (Ms0 / (8 * susrel_val * me*me)) * (diff_new * diff_new - diff * diff);
		}
		else if (Temperature > 0.0) {

			cuBReal r = 3 * *cuMesh.pT_Curie / (10 * (Temperature - *cuMesh.pT_Curie));
			cuBReal m_new_sq = m_new * m_new;
			cuBReal m_sq = m * m;
			energy_delta += M.h.dim() * (Ms0 / (2 * susrel_val)) * (m_new_sq * (1 + r * m_new_sq) - m_sq * (1 + r * m_sq));
		}

		//Compute acceptance probability.
		//Target pdf is proportional to M^2 * exp(-E/kBT), however spin picking probability is not uniform, but proportional to M^2. Thus acceptance probability required to satisfy detailed balance is min{1, (M_new^4 / M_old^4) * exp(-dE/kBT)}
		cuBReal P_accept = 0.0, P = 1.0;
		if (Temperature > 0.0) {

			//Target pdf is proportional to M^2 * exp(-E/kBT), however spin picking probability is not uniform, but proportional to M^2. Thus acceptance probability required to satisfy detailed balance is min{1, (M_new^4 / M_old^4) * exp(-dE/kBT)}
			cuBReal Mratio = (M_new*M_new) / (M_old*M_old);
			P_accept = Mratio * Mratio * exp(-energy_delta / ((cuBReal)BOLTZMANN * Temperature));
			//uniform random number between 0 and 1
			P = prng.rand();
		}
		else if (energy_delta < 0) P_accept = 1.0;
		
		if (P <= P_accept) {

			acceptance_rate = 1.0 / num_moves;

			//set new spin
			M[spin_idx] = M_new;
		}
	}

	if (do_reduction) reduction_sum(0, 1, &acceptance_rate, mc_acceptance_rate);
}

__global__ void Iterate_MonteCarloCUDA_Classic_FM_black_kernel(
	ManagedMeshCUDA& cuMesh, 
	cuBorisRand<>& prng, 
	cuBReal mc_cone_angledeg, cuBReal& mc_acceptance_rate, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC_VC<cuBReal>& Temp = *cuMesh.pTemp;

	cuSZ3& n = M.n;

	int num_moves = M.get_nonempty_cells();

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
	if (spin_idx < n.dim() && M.is_not_empty(spin_idx) && !M.is_skipcell(spin_idx)) {

		cuBReal Ms_val = *cuMesh.pMs;
		cuBReal susrel_val = *cuMesh.psusrel;
		cuMesh.update_parameters_mcoarse(spin_idx, *cuMesh.pMs, Ms_val, *cuMesh.psusrel, susrel_val);

		cuBReal Temperature;
		if (Temp.linear_size()) Temperature = Temp[M.cellidx_to_position(spin_idx)];
		else Temperature = *cuMesh.pbase_temperature;

		cuBReal Ms0 = cuMesh.pMs->get0();
		cuBReal me = Ms_val / Ms0;

		//Picked spin is M[spin_idx]
		cuReal3 M_old = M[spin_idx];

		//obtain rotated spin in a cone around the picked spin
		cuBReal theta_rot = prng.rand() * mc_cone_angledeg * PI / 180.0;
		cuBReal phi_rot = prng.rand() * 2 * PI;
		//Move spin in cone with uniform random probability distribution.
		cuReal3 M_new = relrotate_polar(M_old, theta_rot, phi_rot);

		//now allow magnetization length to change slightly with a Gaussian pdf around current value with sigma value from the normal distribution of P(m^2).
		if (Temperature > 0.0) {

			cuBReal sigma = 2 * me*sqrt(susrel_val*(cuBReal)BOLTZMANN*Temperature / (M.h.dim() * Ms0));
			if (Temperature >= *cuMesh.pT_Curie || sigma > 0.03) sigma = 0.03;
			M_new *= 1 + (prng.rand() * 2 * sigma - sigma);
		}

		//1. Find energy change
		cuBReal energy_delta = 0.0;
		for (int midx = 0; midx < cuMesh.num_FM_MCFuncs; midx++)
			if (cuMesh.pFM_MCFuncs[midx]) energy_delta += (cuMesh.*(cuMesh.pFM_MCFuncs[midx]))(spin_idx, M_new);

		//2. Find contribution to free energy change from longitudinal susceptibility
		cuReal3 m = M_old / Ms0;
		cuReal3 m_new = M_new / Ms0;

		if (Temperature > 0.0 && Temperature <= *cuMesh.pT_Curie) {

			cuBReal diff = m * m - me * me;
			cuBReal diff_new = m_new * m_new - me * me;

			energy_delta += M.h.dim() * (Ms0 / (8 * susrel_val * me*me)) * (diff_new * diff_new - diff * diff);
		}
		else if (Temperature > 0.0) {

			cuBReal r = 3 * *cuMesh.pT_Curie / (10 * (Temperature - *cuMesh.pT_Curie));
			cuBReal m_new_sq = m_new * m_new;
			cuBReal m_sq = m * m;
			energy_delta += M.h.dim() * (Ms0 / (2 * susrel_val)) * (m_new_sq * (1 + r * m_new_sq) - m_sq * (1 + r * m_sq));
		}

		//Compute acceptance probability.
		//Target pdf is proportional to M^2 * exp(-E/kBT), however spin picking probability is not uniform, but proportional to M^2. Thus acceptance probability required to satisfy detailed balance is min{1, (M_new^4 / M_old^4) * exp(-dE/kBT)}
		cuBReal P_accept = 0.0, P = 1.0;
		if (Temperature > 0.0) {

			//Target pdf is proportional to M^2 * exp(-E/kBT), however spin picking probability is not uniform, but proportional to M^2. Thus acceptance probability required to satisfy detailed balance is min{1, (M_new^4 / M_old^4) * exp(-dE/kBT)}
			cuBReal Mratio = (M_new*M_new) / (M_old*M_old);
			P_accept = Mratio * Mratio * exp(-energy_delta / ((cuBReal)BOLTZMANN * Temperature));
			//uniform random number between 0 and 1
			P = prng.rand();
		}
		else if (energy_delta < 0) P_accept = 1.0;

		if (P <= P_accept) {

			acceptance_rate = 1.0 / num_moves;

			//set new spin
			M[spin_idx] = M_new;
		}
	}

	if (do_reduction) reduction_sum(0, 1, &acceptance_rate, mc_acceptance_rate);
}

cuBReal FMeshCUDA::Iterate_MonteCarloCUDA_Classic(cuBReal mc_cone_angledeg, double target_acceptance_rate)
{
	if (pHa) {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Zero_MCAux_SetField_FMCUDA <<< 1, CUDATHREADS >>> (mc_acceptance_rate(mGPU), cuMesh.get_deviceobject(mGPU), (*pHa)(mGPU));
		}
	}
	else {

		for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

			Zero_MCAux_ZeroField_FMCUDA <<< 1, CUDATHREADS >>> (mc_acceptance_rate(mGPU), cuMesh.get_deviceobject(mGPU));
		}
	}

	//must refresh halos before both red and black passes
	M.exchange_halos();

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		//with acceptance rate reduction if mc_acceptance_reduction_counter is zero
		Iterate_MonteCarloCUDA_Classic_FM_red_kernel <<< (M.device_size(mGPU) / 2 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(cuMesh.get_deviceobject(mGPU), 
			prng.get_deviceobject(mGPU), 
			mc_cone_angledeg, mc_acceptance_rate(mGPU), mc_acceptance_reduction_counter == 0);
	}

	//must refresh halos before both red and black passes
	M.exchange_halos();

	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		//with acceptance rate reduction if mc_acceptance_reduction_counter is zero
		Iterate_MonteCarloCUDA_Classic_FM_black_kernel <<< (M.device_size(mGPU) / 2 + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(cuMesh.get_deviceobject(mGPU), 
			prng.get_deviceobject(mGPU), 
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