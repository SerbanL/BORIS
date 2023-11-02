#include "Atom_ZeemanCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_ZEEMAN) && ATOMISTIC == 1

#include "Reduction.cuh"
#include "TEquationCUDA_Function.cuh"

#include "MeshDefs.h"

#include "Atom_MeshCUDA.h"
#include "Atom_MeshParamsControlCUDA.h"

//----------------------- Initialization

__global__ void set_Atom_ZeemanCUDA_pointers_kernel(
	ManagedAtom_MeshCUDA& cuaMesh, cuVEC<cuReal3>& Havec, cuVEC<cuReal3>& globalField)
{
	if (threadIdx.x == 0) cuaMesh.pHavec = &Havec;
	if (threadIdx.x == 1) cuaMesh.pglobalField = &globalField;
}

void Atom_ZeemanCUDA::set_Atom_ZeemanCUDA_pointers(void)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		set_Atom_ZeemanCUDA_pointers_kernel <<< 1, CUDATHREADS >>>
			(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), Havec.get_deviceobject(mGPU), globalField.get_deviceobject(mGPU));
	}
}

//----------------------- Computation

__global__ void Atom_ZeemanCUDA_UpdateField_Cubic(
	ManagedAtom_MeshCUDA& cuMesh, 
	cuReal3& Ha, cuVEC<cuReal3>& Havec, cuVEC<cuReal3>& globalField, 
	ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff1.linear_size()) {

		cuReal3 Hext = cuReal3();

		cuBReal cHA = *cuMesh.pcHA;
		cuMesh.update_parameters_mcoarse(idx, *cuMesh.pcHA, cHA);

		Hext = cHA * Ha;
		if (Havec.linear_size()) Hext += Havec[idx];
		if (globalField.linear_size()) Hext += globalField[idx] * cHA;

		Heff1[idx] = Hext;

		if (do_reduction) {

			//energy density
			int non_empty_cells = M1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MUB_MU0 * M1[idx] * Hext / (non_empty_cells * M1.h.dim());
		}

		if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hext;
		if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MUB_MU0 * M1[idx] * Hext / M1.h.dim();
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

__global__ void Atom_ZeemanCUDA_UpdateField_Equation_Cubic(
	ManagedAtom_MeshCUDA& cuMesh,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& H_equation_x,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& H_equation_y,
	ManagedFunctionCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& H_equation_z,
	cuBReal time,
	cuVEC<cuReal3>& Havec, cuVEC<cuReal3>& globalField,
	ManagedModulesCUDA& cuModule, bool do_reduction)
{
	cuVEC_VC<cuReal3>& M1 = *cuMesh.pM1;
	cuVEC<cuReal3>& Heff1 = *cuMesh.pHeff1;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cuBReal energy_ = 0.0;

	if (idx < Heff1.linear_size()) {

		cuReal3 Hext = cuReal3();

		cuBReal cHA = *cuMesh.pcHA;
		cuMesh.update_parameters_mcoarse(idx, *cuMesh.pcHA, cHA);

		cuReal3 relpos = M1.cellidx_to_position(idx);
		cuReal3 H = cuReal3(
			H_equation_x.evaluate(relpos.x, relpos.y, relpos.z, time),
			H_equation_y.evaluate(relpos.x, relpos.y, relpos.z, time),
			H_equation_z.evaluate(relpos.x, relpos.y, relpos.z, time));

		Hext = cHA * H;
		if (Havec.linear_size()) Hext += Havec[idx];
		if (globalField.linear_size()) Hext += globalField[idx] * cHA;

		Heff1[idx] = Hext;

		if (do_reduction) {

			//energy density
			int non_empty_cells = M1.get_nonempty_cells();
			if (non_empty_cells) energy_ = -(cuBReal)MUB_MU0 * M1[idx] * Hext / (non_empty_cells * M1.h.dim());
		}

		if (do_reduction && cuModule.pModule_Heff->linear_size()) (*cuModule.pModule_Heff)[idx] = Hext;
		if (do_reduction && cuModule.pModule_energy->linear_size()) (*cuModule.pModule_energy)[idx] = -(cuBReal)MUB_MU0 * M1[idx] * Hext / M1.h.dim();
	}

	if (do_reduction) reduction_sum(0, 1, &energy_, *cuModule.penergy);
}

//----------------------- UpdateField LAUNCHER

void Atom_ZeemanCUDA::UpdateField(void)
{
	/////////////////////////////////////////
	// Fixed set field
	/////////////////////////////////////////

	if (!H_equation.is_set()) {

		if (paMeshCUDA->GetMeshType() == MESH_ATOM_CUBIC) {

			if (paMeshCUDA->CurrentTimeStepSolved()) {

				ZeroEnergy();

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					Atom_ZeemanCUDA_UpdateField_Cubic <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
						(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), 
						Ha(mGPU), Havec.get_deviceobject(mGPU), globalField.get_deviceobject(mGPU), 
						cuModule.get_deviceobject(mGPU), true);
				}
			}
			else {

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					Atom_ZeemanCUDA_UpdateField_Cubic <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
						(paMeshCUDA->cuaMesh.get_deviceobject(mGPU), 
						Ha(mGPU), Havec.get_deviceobject(mGPU), globalField.get_deviceobject(mGPU), 
						cuModule.get_deviceobject(mGPU), false);
				}
			}
		}
	}

	/////////////////////////////////////////
	// Field set from user equation
	/////////////////////////////////////////

	else {

		if (paMeshCUDA->GetMeshType() == MESH_ATOM_CUBIC) {

			if (paMeshCUDA->CurrentTimeStepSolved()) {

				ZeroEnergy();

				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					Atom_ZeemanCUDA_UpdateField_Equation_Cubic <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
						(paMeshCUDA->cuaMesh.get_deviceobject(mGPU),
						H_equation.get_x(mGPU), H_equation.get_y(mGPU), H_equation.get_z(mGPU),
						paMeshCUDA->GetStageTime(),
						Havec.get_deviceobject(mGPU), globalField.get_deviceobject(mGPU),
						cuModule.get_deviceobject(mGPU), true);
				}
			}
			else {
				
				for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

					Atom_ZeemanCUDA_UpdateField_Equation_Cubic <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>> 
					(paMeshCUDA->cuaMesh.get_deviceobject(mGPU),
					H_equation.get_x(mGPU), H_equation.get_y(mGPU), H_equation.get_z(mGPU),
					paMeshCUDA->GetStageTime(),
					Havec.get_deviceobject(mGPU), globalField.get_deviceobject(mGPU),
					cuModule.get_deviceobject(mGPU), false);
				}
			}
		}
	}
}

//-------------------Others

BError Atom_ZeemanCUDA::SetFieldEquation(const std::vector<std::vector< std::vector<EqComp::FSPEC> >>& fspec)
{
	BError error(CLASS_STR(Atom_ZeemanCUDA));

	if (!H_equation.make_vector(fspec)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
	if (Havec.size_cpu().dim()) Havec.clear();

	return error;
}

__global__ void SetFromGlobalField_Atom_Zeeman_Kernel(cuVEC<cuReal3>& globalField, cuVEC<cuReal3>& SMesh_globalField)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < globalField.linear_size()) {

		cuReal3 abs_pos = globalField.cellidx_to_position(idx) + globalField.rect.s;

		if (SMesh_globalField.rect.contains(abs_pos)) {

			globalField[idx] = SMesh_globalField[abs_pos - SMesh_globalField.rect.s];
		}
		else {

			globalField[idx] = cuReal3();
		}
	}
}

void Atom_ZeemanCUDA::SetGlobalField(mcu_VEC(cuReal3)& SMesh_globalField)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		SetFromGlobalField_Atom_Zeeman_Kernel <<< (paMeshCUDA->M1.device_size(mGPU) + CUDATHREADS) / CUDATHREADS, CUDATHREADS >>>
			(globalField.get_deviceobject(mGPU), SMesh_globalField.get_deviceobject(mGPU));
	}
}

#endif

#endif

//----------------------------------- MONTE-CARLO METHODS FOR ENERGY COMPUTATION

#if COMPILECUDA == 1 && ATOMISTIC == 1 && MONTE_CARLO == 1

__device__ cuBReal ManagedAtom_MeshCUDA::Get_Atomistic_EnergyChange_SC_ZeemanCUDA(int spin_index, cuReal3 Mnew)
{
	cuVEC_VC<cuReal3>& M1 = *pM1;

	cuReal3 Hext = cuReal3();

	cuBReal cHA = *pcHA;
	update_parameters_mcoarse(spin_index, *pcHA, cHA);

	Hext = cHA * Ha_MC;
	if (pHavec && pHavec->linear_size()) Hext += (*pHavec)[spin_index];
	if (pglobalField->linear_size()) Hext += (*pglobalField)[spin_index] * cHA;

	if (Mnew != cuReal3()) return -(cuBReal)MUB_MU0 * (Mnew - M1[spin_index]) * Hext;
	else return -(cuBReal)MUB_MU0 * M1[spin_index] * Hext;
}

#endif