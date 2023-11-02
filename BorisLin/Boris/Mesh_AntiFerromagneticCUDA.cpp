#include "stdafx.h"
#include "Mesh_AntiFerromagneticCUDA.h"

#if COMPILECUDA == 1
#ifdef MESH_COMPILATION_ANTIFERROMAGNETIC

#include "Mesh_AntiFerromagnetic.h"
#include "ManagedDiffEqAFMCUDA.h"
#include "DiffEqAFMCUDA.h"

#include "SimScheduleDefs.h"

AFMeshCUDA::AFMeshCUDA(AFMesh* pMesh) :
	MeshCUDA(pMesh)
{
	pAFMesh = pMesh;
}

AFMeshCUDA::~AFMeshCUDA()
{

}

//----------------------------------- IMPORTANT CONTROL METHODS

//call when a configuration change has occurred - some objects might need to be updated accordingly
BError AFMeshCUDA::UpdateConfiguration(UPDATECONFIG_ cfgMessage)
{
	BError error(CLASS_STR(AFMeshCUDA));

	if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MESHCHANGE)) {

		//resize arrays held in Mesh - if changing mesh to new size then use resize : this maps values from old mesh to new mesh size (keeping magnitudes). Otherwise assign magnetization along x in whole mesh.
		if (M.n.dim()) {

			if (!M.resize((cuReal3)h, (cuRect)meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (!M2.resize((cuReal3)h, (cuRect)meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		}
		else {

			if (!M.assign((cuReal3)h, (cuRect)meshRect, cuReal3(-Ms_AFM.get_current_cpu().i, 0, 0))) return error(BERROR_OUTOFGPUMEMORY_CRIT);
			if (!M2.assign((cuReal3)h, (cuRect)meshRect, cuReal3(Ms_AFM.get_current_cpu().j, 0, 0))) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		}

		if (!Heff.assign((cuReal3)h, (cuRect)meshRect, cuReal3())) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		if (!Heff2.assign((cuReal3)h, (cuRect)meshRect, cuReal3())) return error(BERROR_OUTOFGPUMEMORY_CRIT);

		//setup prng for MC methods
		if (IsStageSet(SS_MONTECARLO)) {

			//setup prng for MC methods
			if (prng.initialize((pAFMesh->prng_seed == 0 ? GetSystemTickCount() : pAFMesh->prng_seed), n.dim(), (pAFMesh->prng_seed == 0 ? 0 : 1)) != cudaSuccess) error(BERROR_OUTOFGPUMEMORY_NCRIT);
		}
		else prng.clear();

		//setup function pointers for Monte Carlo methods from list of configured module ids
		if (ucfg::check_cfgflags(cfgMessage, UPDATECONFIG_MODULEADDED, UPDATECONFIG_MODULEDELETED)) {

			std::vector<int> modules_ids(pAFMesh->pMod.size());
			for (int idx = 0; idx < modules_ids.size(); idx++) modules_ids[idx] = pAFMesh->pMod.get_ID_from_index(idx);
			cuMesh.Set_AFM_MCFuncs(modules_ids);
		}
	}

	return error;
}

cuBReal AFMeshCUDA::CheckMoveMesh(bool antisymmetric, double threshold)
{
	//move mesh algorithm applied to systems containing domain walls in order to simulate domain wall movement.
	//In this case the magnetization at the ends of the mesh must be fixed (magnetization evolver not applied to the ends).
	//Find the average magnetization projected along the absolute direction of the magnetization at the ends.
	//For a domain wall at the centre this value should be close to zero. If it exceeds a threshold then move mesh by a cellsize along +x or -x.

	if (!pAFMesh->move_mesh_trigger) return 0;

	//the ends should not be completely empty, and must have a constant magnetization direction
	int cells_fixed = (int)ceil_epsilon((double)n.x * MOVEMESH_ENDRATIO);

	cuReal3 M_end = M.average(cuBox(0, 0, 0, cells_fixed, n.y, n.z));
	cuReal3 M_av_left = M.average(cuBox(cells_fixed, 0, 0, n.x / 2, n.y, n.z));
	cuReal3 M_av_right = M.average(cuBox(n.x / 2, 0, 0, n.x - cells_fixed, n.y, n.z));

	cuBReal direction = (2 * cuBReal(antisymmetric) - 1);

	cuReal3 M_av = M_av_right + direction * M_av_left;

	if (GetMagnitude(M_end) * GetMagnitude(M_av)) {

		double Mratio = (M_end * M_av) / (GetMagnitude(M_end) * GetMagnitude(M_av));

		if (Mratio > threshold) return -h.x * direction;
		if (Mratio < -threshold) return h.x * direction;
	}

	return 0;
}

//----------------------------------- ENABLED MESH PROPERTIES CHECKERS

//get exchange_couple_to_meshes status flag from the cpu version
bool AFMeshCUDA::GetMeshExchangeCoupling(void)
{
	return pAFMesh->GetMeshExchangeCoupling();
}

//-----------------------------------OBJECT GETTERS

mcu_obj<ManagedDiffEqAFMCUDA, ManagedDiffEqPolicyAFMCUDA>& AFMeshCUDA::Get_ManagedDiffEqCUDA(void)
{
	return dynamic_cast<DifferentialEquationAFMCUDA*>(pAFMesh->Get_DifferentialEquation().Get_DifferentialEquationCUDA_ptr())->Get_ManagedDiffEqCUDA();
}

//----------------------------------- ODE METHODS

//Save current magnetization in sM VECs (e.g. useful to reset dM / dt calculation)
void AFMeshCUDA::SaveMagnetization(void)
{
	pAFMesh->SaveMagnetization();
}

#endif
#endif