#include "AnisotropyTensorialCUDA.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_ANITENS

#include "Reduction.cuh"

#include "MeshCUDA.h"
#include "MeshParamsControlCUDA.h"
#include "MeshDefs.h"

__global__ void Anisotropy_TensorialCUDA_FM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal4>& Kt = *cuMesh.pKt;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuReal3 Heff_value = cuReal3();

		if (M.is_not_empty(idx)) {

			cuBReal Ms = *cuMesh.pMs;
			cuBReal K1 = *cuMesh.pK1;
			cuBReal K2 = *cuMesh.pK2;
			cuBReal K3 = *cuMesh.pK3;
			cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
			cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
			cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs, Ms, *cuMesh.pK1, K1, *cuMesh.pK2, K2, *cuMesh.pK3, K3, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);
			
			//calculate dot products
			cuBReal a = (M[idx] * mcanis_ea1) / Ms;
			cuBReal b = (M[idx] * mcanis_ea2) / Ms;
			cuBReal c = (M[idx] * mcanis_ea3) / Ms;

			for (int tidx = 0; tidx < Kt.linear_size(); tidx++) {

				//for each energy density term d*a^n1 b^n2 c^n3 we have an effective field contribution as:
				//(-d / mu0Ms) * [n1 * a^(n1-1) * b^n2 * c^n3 * mcanis_ea1 + n2 * a^n1) * b^(n2-1) * c^n3 * mcanis_ea2 + n3 * a^n1 * b^n2 * c^(n3-1) * mcanis_ea3] - for each n1, n2, n3 > 0

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
				if (order == 2) coeff = -K1 * Kt[tidx].i / ((cuBReal)MU0*Ms);
				else if (order == 4) coeff = -K2 * Kt[tidx].i / ((cuBReal)MU0*Ms);
				else if (order == 6) coeff = -K3 * Kt[tidx].i / ((cuBReal)MU0*Ms);
				else coeff = -Kt[tidx].i / ((cuBReal)MU0*Ms);

				Heff_value += coeff * (Kt[tidx].j * ap1*bp*cp * mcanis_ea1 + Kt[tidx].k * ap*bp1*cp * mcanis_ea2 + Kt[tidx].l * ap*bp*cp1 * mcanis_ea3);

				energy_ += -coeff * (cuBReal)MU0*Ms * ap*bp*cp;
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = energy_;

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ /= non_empty_cells;
			}
		}

		Heff[idx] += Heff_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void Anisotropy_TensorialCUDA_AFM_UpdateField(ManagedMeshCUDA& cuMesh, ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M = *cuMesh.pM;
	cuVEC<cuReal3>& Heff = *cuMesh.pHeff;
	cuVEC<cuReal4>& Kt = *cuMesh.pKt;

	cuVEC_VC<cuReal3>& M2 = *cuMesh.pM2;
	cuVEC<cuReal3>& Heff2 = *cuMesh.pHeff2;
	cuVEC<cuReal4>& Kt2 = *cuMesh.pKt2;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff.linear_size()) {

		cuReal3 Heff_value = cuReal3();
		cuReal3 Heff2_value = cuReal3();

		if (M.is_not_empty(idx)) {

			cuReal2 Ms_AFM = *cuMesh.pMs_AFM;
			cuReal2 K1_AFM = *cuMesh.pK1_AFM;
			cuReal2 K2_AFM = *cuMesh.pK2_AFM;
			cuReal2 K3_AFM = *cuMesh.pK3_AFM;
			cuReal3 mcanis_ea1 = *cuMesh.pmcanis_ea1;
			cuReal3 mcanis_ea2 = *cuMesh.pmcanis_ea2;
			cuReal3 mcanis_ea3 = *cuMesh.pmcanis_ea3;
			cuMesh.update_parameters_mcoarse(idx, *cuMesh.pMs_AFM, Ms_AFM, *cuMesh.pK1_AFM, K1_AFM, *cuMesh.pK2_AFM, K2_AFM, *cuMesh.pK3_AFM, K3_AFM, *cuMesh.pmcanis_ea1, mcanis_ea1, *cuMesh.pmcanis_ea2, mcanis_ea2, *cuMesh.pmcanis_ea3, mcanis_ea3);

			//calculate dot products
			cuBReal a = (M[idx] * mcanis_ea1) / Ms_AFM.i;
			cuBReal b = (M[idx] * mcanis_ea2) / Ms_AFM.i;
			cuBReal c = (M[idx] * mcanis_ea3) / Ms_AFM.i;

			cuBReal a2 = (M2[idx] * mcanis_ea1) / Ms_AFM.j;
			cuBReal b2 = (M2[idx] * mcanis_ea2) / Ms_AFM.j;
			cuBReal c2 = (M2[idx] * mcanis_ea3) / Ms_AFM.j;

			cuBReal energy1_ = 0.0, energy2_ = 0.0;

			for (int tidx = 0; tidx < Kt.linear_size(); tidx++) {

				//for each energy density term d*a^n1 b^n2 c^n3 we have an effective field contribution as:
				//(-d / mu0Ms) * [n1 * a^(n1-1) * b^n2 * c^n3 * mcanis_ea1 + n2 * a^n1) * b^(n2-1) * c^n3 * mcanis_ea2 + n3 * a^n1 * b^n2 * c^(n3-1) * mcanis_ea3] - for each n1, n2, n3 > 0

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
				if (order == 2) coeff = -K1_AFM.i * Kt[tidx].i / ((cuBReal)MU0*Ms_AFM.i);
				else if (order == 4) coeff = -K2_AFM.i * Kt[tidx].i / ((cuBReal)MU0*Ms_AFM.i);
				else if (order == 6) coeff = -K3_AFM.i * Kt[tidx].i / ((cuBReal)MU0*Ms_AFM.i);
				else coeff = -Kt[tidx].i / ((cuBReal)MU0*Ms_AFM.i);

				Heff_value += coeff * (Kt[tidx].j * ap1*bp*cp * mcanis_ea1 + Kt[tidx].k * ap*bp1*cp * mcanis_ea2 + Kt[tidx].l * ap*bp*cp1 * mcanis_ea3);

				energy1_ += -coeff * (cuBReal)MU0*Ms_AFM.i * ap*bp*cp;
			}

			for (int tidx = 0; tidx < Kt2.linear_size(); tidx++) {

				//for each energy density term d*a^n1 b^n2 c^n3 we have an effective field contribution as:
				//(-d / mu0Ms) * [n1 * a^(n1-1) * b^n2 * c^n3 * mcanis_ea1 + n2 * a^n1) * b^(n2-1) * c^n3 * mcanis_ea2 + n3 * a^n1 * b^n2 * c^(n3-1) * mcanis_ea3] - for each n1, n2, n3 > 0

				cuBReal ap1 = 0.0, bp1 = 0.0, cp1 = 0.0;
				cuBReal ap = 0.0, bp = 0.0, cp = 0.0;
				if (Kt2[tidx].j > 0) { ap1 = pow(a2, Kt2[tidx].j - 1); ap = ap1 * a2; }
				else ap = pow(a2, Kt2[tidx].j);
				if (Kt2[tidx].k > 0) { bp1 = pow(b2, Kt2[tidx].k - 1); bp = bp1 * b2; }
				else bp = pow(b2, Kt2[tidx].k);
				if (Kt2[tidx].l > 0) { cp1 = pow(c2, Kt2[tidx].l - 1); cp = cp1 * c2; }
				else cp = pow(c2, Kt2[tidx].l);

				cuBReal coeff;
				int order = Kt2[tidx].j + Kt2[tidx].k + Kt2[tidx].l;
				if (order == 2) coeff = -K1_AFM.j * Kt2[tidx].i / ((cuBReal)MU0*Ms_AFM.j);
				else if (order == 4) coeff = -K2_AFM.j * Kt2[tidx].i / ((cuBReal)MU0*Ms_AFM.j);
				else if (order == 6) coeff = -K3_AFM.j * Kt2[tidx].i / ((cuBReal)MU0*Ms_AFM.j);
				else coeff = -Kt2[tidx].i / ((cuBReal)MU0*Ms_AFM.j);

				Heff2_value += coeff * (Kt2[tidx].j * ap1*bp*cp * mcanis_ea1 + Kt2[tidx].k * ap*bp1*cp * mcanis_ea2 + Kt2[tidx].l * ap*bp*cp1 * mcanis_ea3);

				energy2_ += -coeff * (cuBReal)MU0*Ms_AFM.j * ap*bp*cp;
			}

			if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Heff_value;
			if (do_reduction && cuModule.pModule_Heff2->linear_size()) (*cuModule.pModule_Heff2)[idx] = Heff2_value;
			if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = energy1_;
			if (do_reduction && cuModule.pModule_energy2->linear_size()) (*cuModule.pModule_energy2)[idx] = energy2_;

			if (do_reduction) {

				int non_empty_cells = M.get_nonempty_cells();
				if (non_empty_cells) energy_ = (energy1_ + energy2_) / (2 * non_empty_cells);
			}
		}

		Heff[idx] += Heff_value;
		Heff2[idx] += Heff2_value;
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Anisotropy_TensorialCUDA::UpdateField(void)
{
	if (pMeshCUDA->GetMeshType() == MESH_ANTIFERROMAGNETIC) {

		//anti-ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_TensorialCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			Anisotropy_TensorialCUDA_AFM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
				(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
		}
	}
	else {

		//ferromagnetic mesh

		if (pMeshCUDA->CurrentTimeStepSolved()) {

			ZeroEnergy();

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_TensorialCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), true);
			}
		}
		else {

			for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

				Anisotropy_TensorialCUDA_FM_UpdateField <<< (pMeshCUDA->M.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(pMeshCUDA->cuMesh.get_deviceobject(mGPU), cuModule.get_deviceobject(mGPU), false);
			}
		}
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && MONTE_CARLO == 1

//Ferromagnetic
__device__ cuBReal ManagedMeshCUDA::Get_EnergyChange_FM_AnisotropyTensorialCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M = *pM;

	cuBReal K1 = *pK1;
	cuBReal K2 = *pK2;
	cuBReal K3 = *pK3;
	cuBReal Ms = *pMs;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	cuReal3 mcanis_ea2 = *pmcanis_ea2;
	cuReal3 mcanis_ea3 = *pmcanis_ea3;
	update_parameters_mcoarse(spin_index, *pMs, Ms, *pK1, K1, *pK2, K2, *pK3, K3, *pmcanis_ea1, mcanis_ea1, *pmcanis_ea2, mcanis_ea2, *pmcanis_ea3, mcanis_ea3);

	auto Get_Energy = [&](cuBReal a, cuBReal b, cuBReal c) -> cuBReal {

		cuBReal energy_ = 0.0;

		for (int tidx = 0; tidx < pKt->linear_size(); tidx++) {

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
	cuBReal a = M[spin_index] * mcanis_ea1 / Ms;
	cuBReal b = M[spin_index] * mcanis_ea2 / Ms;
	cuBReal c = M[spin_index] * mcanis_ea3 / Ms;

	cuBReal energy_ = Get_Energy(a, b, c);

	if (Mnew != cuReal3()) {

		cuBReal anew = Mnew * mcanis_ea1 / Ms;
		cuBReal bnew = Mnew * mcanis_ea2 / Ms;
		cuBReal cnew = Mnew * mcanis_ea3 / Ms;

		cuBReal energynew_ = Get_Energy(anew, bnew, cnew);

		return M.h.dim() * (energynew_ - energy_);
	}
	else return M.h.dim() * energy_;
}

//Antiferromagnetic
__device__ cuReal2 ManagedMeshCUDA::Get_EnergyChange_AFM_AnisotropyTensorialCUDA(int spin_index, cuReal3 Mnew_A, cuReal3 Mnew_B)
{
	cuVEC_VC<cuReal3>& M = *pM;
	cuVEC_VC<cuReal3>& M2 = *pM2;

	cuReal2 Ms_AFM = *pMs_AFM;
	cuReal2 K1_AFM = *pK1_AFM;
	cuReal2 K2_AFM = *pK2_AFM;
	cuReal2 K3_AFM = *pK3_AFM;
	cuReal3 mcanis_ea1 = *pmcanis_ea1;
	cuReal3 mcanis_ea2 = *pmcanis_ea2;
	cuReal3 mcanis_ea3 = *pmcanis_ea3;
	update_parameters_mcoarse(spin_index, *pMs_AFM, Ms_AFM, *pK1_AFM, K1_AFM, *pK2_AFM, K2_AFM, *pK3_AFM, K3_AFM, *pmcanis_ea1, mcanis_ea1, *pmcanis_ea2, mcanis_ea2, *pmcanis_ea3, mcanis_ea3);

	auto Get_Energy = [&](cuBReal a, cuBReal b, cuBReal c, cuBReal a2, cuBReal b2, cuBReal c2) -> cuReal2 {

		cuBReal energyA = 0.0, energyB = 0.0;

		for (int tidx = 0; tidx < pKt->linear_size(); tidx++) {

			cuBReal coeff;
			int order = (*pKt)[tidx].j + (*pKt)[tidx].k + (*pKt)[tidx].l;
			if (order == 2) coeff = K1_AFM.i * (*pKt)[tidx].i;
			else if (order == 4) coeff = K2_AFM.i * (*pKt)[tidx].i;
			else if (order == 6) coeff = K3_AFM.i * (*pKt)[tidx].i;
			else coeff = (*pKt)[tidx].i;

			energyA += coeff * pow(a, (*pKt)[tidx].j) * pow(b, (*pKt)[tidx].k) * pow(c, (*pKt)[tidx].l);
		}

		for (int tidx = 0; tidx < pKt2->linear_size(); tidx++) {

			cuBReal coeff;
			int order = (*pKt2)[tidx].j + (*pKt2)[tidx].k + (*pKt2)[tidx].l;
			if (order == 2) coeff = K1_AFM.j * (*pKt2)[tidx].i;
			else if (order == 4) coeff = K2_AFM.j * (*pKt2)[tidx].i;
			else if (order == 6) coeff = K3_AFM.j * (*pKt2)[tidx].i;
			else coeff = (*pKt2)[tidx].i;

			energyB += coeff * pow(a2, (*pKt2)[tidx].j) * pow(b2, (*pKt2)[tidx].k) * pow(c2, (*pKt2)[tidx].l);
		}

		return cuReal2(energyA, energyB);
	};

	//calculate dot products
	cuBReal a = (M[spin_index] * mcanis_ea1) / Ms_AFM.i;
	cuBReal b = (M[spin_index] * mcanis_ea2) / Ms_AFM.i;
	cuBReal c = (M[spin_index] * mcanis_ea3) / Ms_AFM.i;

	cuBReal a2 = (M2[spin_index] * mcanis_ea1) / Ms_AFM.j;
	cuBReal b2 = (M2[spin_index] * mcanis_ea2) / Ms_AFM.j;
	cuBReal c2 = (M2[spin_index] * mcanis_ea3) / Ms_AFM.j;

	cuReal2 energy_ = Get_Energy(a, b, c, a2, b2, c2);

	if (Mnew_A != cuReal3() && Mnew_B != cuReal3()) {

		cuBReal anew = Mnew_A * mcanis_ea1 / Ms_AFM.j;
		cuBReal bnew = Mnew_A * mcanis_ea2 / Ms_AFM.j;
		cuBReal cnew = Mnew_A * mcanis_ea3 / Ms_AFM.j;

		cuBReal a2new = Mnew_B * mcanis_ea1 / Ms_AFM.j;
		cuBReal b2new = Mnew_B * mcanis_ea2 / Ms_AFM.j;
		cuBReal c2new = Mnew_B * mcanis_ea3 / Ms_AFM.j;

		cuReal2 energynew_ = Get_Energy(anew, bnew, cnew, a2new, b2new, c2new);

		return M.h.dim() * (energynew_ - energy_);
	}
	else return M.h.dim() * energy_;
}

#endif
