#include "Atom_AnisotropyTensorialCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_ANITENS) && ATOMISTIC == 1

#include "Reduction.cuh"

#include "Atom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"
#include "MeshDefs.h"

__global__ void Atom_Anisotropy_TensorialCUDA_Cubic_UpdateField(ManagedAtom_MeshCUDA& cuaMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuaMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuaMesh.pHeff1;
	cuVEC<cuReal4>& Kt = *cuaMesh.pKt;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff1.linear_size()) {

		cuReal3 Heff_value = cuReal3();

		if (M1.is_not_empty(idx)) {

			cuBReal mu_s = *cuaMesh.pmu_s;
			cuBReal K1 = *cuaMesh.pK1;
			cuBReal K2 = *cuaMesh.pK2;
			cuBReal K3 = *cuaMesh.pK3;
			cuReal3 mcanis_ea1 = *cuaMesh.pmcanis_ea1;
			cuReal3 mcanis_ea2 = *cuaMesh.pmcanis_ea2;
			cuReal3 mcanis_ea3 = *cuaMesh.pmcanis_ea3;
			cuaMesh.update_parameters_mcoarse(idx, *cuaMesh.pmu_s, mu_s, *cuaMesh.pK1, K1, *cuaMesh.pK2, K2, *cuaMesh.pK3, K3, *cuaMesh.pmcanis_ea1, mcanis_ea1, *cuaMesh.pmcanis_ea2, mcanis_ea2, *cuaMesh.pmcanis_ea3, mcanis_ea3);

			//calculate dot products
			cuBReal a = (M1[idx] * mcanis_ea1) / mu_s;
			cuBReal b = (M1[idx] * mcanis_ea2) / mu_s;
			cuBReal c = (M1[idx] * mcanis_ea3) / mu_s;

			for (int tidx = 0; tidx < Kt.linear_size(); tidx++) {

				//for each energy density term d*a^n1 b^n2 c^n3 we have an effective field contribution as:
				//(-d / mu0 mu_s) * [n1 * a^(n1-1) * b^n2 * c^n3 * mcanis_ea1 + n2 * a^n1) * b^(n2-1) * c^n3 * mcanis_ea2 + n3 * a^n1 * b^n2 * c^(n3-1) * mcanis_ea3] - for each n1, n2, n3 > 0

				cuBReal ap1 = 0.0, bp1 = 0.0, cp1 = 0.0;
				cuBReal ap = 0.0, bp = 0.0, cp = 0.0;
				if (Kt[tidx].j > 0) { ap1 = pow(a, Kt[tidx].j - 1); ap = ap1 * a; }
				else ap = pow(a, Kt[tidx].j);
				if (Kt[tidx].k > 0) { bp1 = pow(b, Kt[tidx].k - 1); bp = bp1 * b; }
				else bp = pow(b, Kt[tidx].k);
				if (Kt[tidx].l > 0) { cp1 = pow(c, Kt[tidx].l - 1); cp = cp1 * c; }
				else cp = pow(c, Kt[tidx].l);

				cuBReal coeff;
				int order = Kt[tidx].j + Kt[tidx].k + Kt[tidx].l;
				if (order == 2) coeff = -K1 * Kt[tidx].i / ((cuBReal)MUB_MU0*mu_s);
				else if (order == 4) coeff = -K2 * Kt[tidx].i / ((cuBReal)MUB_MU0*mu_s);
				else if (order == 6) coeff = -K3 * Kt[tidx].i / ((cuBReal)MUB_MU0*mu_s);
				else coeff = -Kt[tidx].i / ((cuBReal)MUB_MU0*mu_s);

				Heff_value += coeff * (Kt[tidx].j * ap1*bp*cp * mcanis_ea1 + Kt[tidx].k * ap*bp1*cp * mcanis_ea2 + Kt[tidx].l * ap*bp*cp1 * mcanis_ea3);

				energy_ += -coeff * (cuBReal)MUB_MU0*mu_s * ap*bp*cp;
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = energy_ / M1.h.dim();

			if (do_reduction) {

				//update energy density
				cuBReal non_empty_volume = M1.get_nonempty_cells() * M1.h.dim();
				if (non_empty_volume) energy_ /= non_empty_volume;
			}
		}

		Heff1[idx] += Heff_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Atom_Anisotropy_TensorialCUDA::UpdateField(void)
{
	if (paMeshCUDA->GetMeshType() == MESH_ATOM_CUBIC) {

		//atomistic simple-cubic mesh

		if (paMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Atom_Anisotropy_TensorialCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Atom_Anisotropy_TensorialCUDA_Cubic_UpdateField <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && ATOMISTIC == 1 && MONTE_CARLO == 1

__device__ cuBReal ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_AnisotropyTensorialCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M1 = *pM1;

	cuBReal K1 = *pK1;
	cuBReal K2 = *pK2;
	cuBReal K3 = *pK3;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	cuReal3 mcanis_ea2 = *pmcanis_ea2;
	cuReal3 mcanis_ea3 = *pmcanis_ea3;
	update_parameters_mcoarse(spin_index, *pK1, K1, *pK2, K2, *pK3, K3, *pmcanis_ea1, mcanis_ea1, *pmcanis_ea2, mcanis_ea2, *pmcanis_ea3, mcanis_ea3);

	auto Get_Energy = [&](cuBReal a, cuBReal b, cuBReal c) -> cuBReal {

		cuBReal energy_ = 0.0;

		for (int tidx = 0; tidx < pKt->linear_size(); tidx++) {

			//for each energy density term d*a^n1 b^n2 c^n3 we have an effective field contribution as:
			//(-d / mu0 mu_s) * [n1 * a^(n1-1) * b^n2 * c^n3 * mcanis_ea1 + n2 * a^n1) * b^(n2-1) * c^n3 * mcanis_ea2 + n3 * a^n1 * b^n2 * c^(n3-1) * mcanis_ea3] - for each n1, n2, n3 > 0

			cuBReal coeff;
			int order = (*pKt)[tidx].j + (*pKt)[tidx].k + (*pKt)[tidx].l;
			if (order == 2) coeff = K1 * (*pKt)[tidx].i;
			else if (order == 4) coeff = K2 * (*pKt)[tidx].i;
			else if (order == 6) coeff = K3 * (*pKt)[tidx].i;
			else coeff = (*pKt)[tidx].i;

			energy_ += coeff * pow(a, (*pKt)[tidx].j) * pow(b, (*pKt)[tidx].k) * pow(c, (*pKt)[tidx].l);
		}

		return energy_;
	};

	//calculate dot products
	cuBReal a = M1[spin_index].normalized() * mcanis_ea1;
	cuBReal b = M1[spin_index].normalized() * mcanis_ea2;
	cuBReal c = M1[spin_index].normalized() * mcanis_ea3;

	cuBReal energy_ = Get_Energy(a, b, c);

	if (Mnew != cuReal3()) {

		cuBReal anew = Mnew.normalized() * mcanis_ea1;
		cuBReal bnew = Mnew.normalized() * mcanis_ea2;
		cuBReal cnew = Mnew.normalized() * mcanis_ea3;

		cuBReal energynew_ = Get_Energy(anew, bnew, cnew);

		return energynew_ - energy_;
	}
	else return energy_;
}

#endif