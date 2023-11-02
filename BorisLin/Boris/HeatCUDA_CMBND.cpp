#include "stdafx.h"
#include "HeatCUDA_CMBND.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_HEAT

#include "MeshCUDA.h"
#include "Atom_MeshCUDA.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

BError HeatCUDA_CMBND_Pri::set_pointers(MeshCUDA* pMeshCUDA, int idx_device)
{
	BError error(__FUNCTION__);

	if (set_gpu_value(pcuMesh, pMeshCUDA->cuMesh.get_managed_object(idx_device)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	
	return error;
}

BError HeatCUDA_CMBND_Pri::set_pointers(Atom_MeshCUDA* paMeshCUDA, int idx_device)
{
	BError error(__FUNCTION__);

	if (set_gpu_value(pcuaMesh, paMeshCUDA->cuaMesh.get_managed_object(idx_device)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	return error;
}

//set pQ_equation as needed
BError HeatCUDA_CMBND_Pri::set_Q_equation(mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Q_equation, int idx_device)
{
	BError error(__FUNCTION__);
	
	TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>&  Q_equation_ref = Q_equation.get_managed_object(idx_device);

	if (Q_equation_ref.is_set()) {
		
		if (set_gpu_value(pQ_equation, Q_equation_ref.get_pcu_obj_x()->get_managed_object()) != cudaSuccess) return error(BERROR_GPUERROR_CRIT);
	}
	else {
	
		nullgpuptr(pQ_equation);
	}

	return error;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

BError HeatCUDA_CMBND_Sec::set_pointers(MeshCUDA* pMeshCUDA, int idx_device)
{
	BError error(__FUNCTION__);
	
	set_gpu_value(curr_device, idx_device);

	//HeatCUDA_CMBND_Sec objects are held in a mcu_obj, one for each device
	//However, each HeatCUDA_CMBND_Sec must hold pointers for all devices to allow access through UVA - thus loop here through allk devices and set pointers
	int num_devices_cpu = get_gpu_value(num_devices);
	for (int idx = 0; idx < num_devices_cpu; idx++) {

		if (cpu_to_gpu_managed(ppcuMesh, &pMeshCUDA->cuMesh.get_managed_object(idx), 1, idx) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

		//since we are setting ppcuMesh, must null all entries for ppcuaMesh
		ManagedAtom_MeshCUDA* cu_pointer_handle = nullptr;
		cpu_to_gpu_managed(ppcuaMesh, &cu_pointer_handle, 1, idx);
	}

	return error;
}

BError HeatCUDA_CMBND_Sec::set_pointers(Atom_MeshCUDA* paMeshCUDA, int idx_device)
{
	BError error(__FUNCTION__);

	set_gpu_value(curr_device, idx_device);

	//HeatCUDA_CMBND_Sec objects are held in a mcu_obj, one for each device
	//However, each HeatCUDA_CMBND_Sec must hold pointers for all devices to allow access through UVA - thus loop here through allk devices and set pointers
	int num_devices_cpu = get_gpu_value(num_devices);
	for (int idx = 0; idx < num_devices_cpu; idx++) {

		if (cpu_to_gpu_managed(ppcuaMesh, &paMeshCUDA->cuaMesh.get_managed_object(idx), 1, idx) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

		//since we are setting ppcuaMesh, must null all entries for ppcuMesh
		ManagedMeshCUDA* cu_pointer_handle = nullptr;
		cpu_to_gpu_managed(ppcuMesh, &cu_pointer_handle, 1, idx);
	}
	
	return error;
}

//set pQ_equation as needed
BError HeatCUDA_CMBND_Sec::set_Q_equation(mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Q_equation, int idx_device)
{
	BError error(__FUNCTION__);
	
	set_gpu_value(curr_device, idx_device);

	TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Q_equation_ref = Q_equation.get_managed_object(idx_device);

	if (Q_equation_ref.is_set()) {

		if (set_gpu_value(pQ_equation, Q_equation_ref.get_pcu_obj_x()->get_managed_object()) != cudaSuccess) return error(BERROR_GPUERROR_CRIT);
	}
	else {

		nullgpuptr(pQ_equation);
	}
	
	return error;
}

#endif

#endif