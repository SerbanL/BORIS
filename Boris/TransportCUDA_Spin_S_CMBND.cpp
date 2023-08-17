#include "stdafx.h"
#include "TransportCUDA_Spin_S_CMBND.h"

#if COMPILECUDA == 1

#ifdef MODULE_COMPILATION_TRANSPORT

#include "MeshCUDA.h"
#include "Atom_MeshCUDA.h"
#include "TransportBaseCUDA.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

BError TransportCUDA_Spin_S_CMBND_Pri::set_pointers(MeshCUDA* pMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int idx_device)
{
	BError error(__FUNCTION__);

	if (set_gpu_value(pcuMesh, pMeshCUDA->cuMesh.get_managed_object(idx_device)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pspin_V_cmbnd_funcs_pri, pTransportBaseCUDA->spin_V_cmbnd_funcs_pri.get_managed_object(idx_device)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pdM_dt, pTransportBaseCUDA->dM_dt.get_managed_object(idx_device)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(stsolve, pTransportBaseCUDA->Get_STSolveType()) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	return error;
}

BError TransportCUDA_Spin_S_CMBND_Pri::set_pointers(Atom_MeshCUDA* paMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int idx_device)
{
	BError error(__FUNCTION__);

	if (set_gpu_value(pcuaMesh, paMeshCUDA->cuaMesh.get_managed_object(idx_device)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pspin_V_cmbnd_funcs_pri, pTransportBaseCUDA->spin_V_cmbnd_funcs_pri.get_managed_object(idx_device)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	if (set_gpu_value(pdM_dt, pTransportBaseCUDA->dM_dt.get_managed_object(idx_device)) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	if (set_gpu_value(stsolve, pTransportBaseCUDA->Get_STSolveType()) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	return error;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////
//
// CTOR/DTOR

__host__ void TransportCUDA_Spin_S_CMBND_Sec::construct_cu_obj(void)
{
	nullgpuptr(ppcuMesh);
	nullgpuptr(ppcuaMesh);
	nullgpuptr(ppspin_V_cmbnd_funcs_pri);
	nullgpuptr(ppdM_dt);

	set_gpu_value(num_devices, (int)0);
	set_gpu_value(curr_device, (int)0);
}

__host__ void TransportCUDA_Spin_S_CMBND_Sec::destruct_cu_obj(void)
{
	if (get_gpu_value(num_devices)) {

		if (!isnullgpuptr(ppcuMesh)) {

			gpu_free_managed(ppcuMesh);
			nullgpuptr(ppcuMesh);
		}

		if (!isnullgpuptr(ppcuaMesh)) {

			gpu_free_managed(ppcuaMesh);
			nullgpuptr(ppcuaMesh);
		}

		if (!isnullgpuptr(ppspin_V_cmbnd_funcs_pri)) {

			gpu_free_managed(ppspin_V_cmbnd_funcs_pri);
			nullgpuptr(ppspin_V_cmbnd_funcs_pri);
		}

		if (!isnullgpuptr(ppdM_dt)) {

			gpu_free_managed(ppdM_dt);
			nullgpuptr(ppdM_dt);
		}
	}
}

////////////////////////////////////////////////////
//
// Configuration

void TransportCUDA_Spin_S_CMBND_Sec::set_number_of_devices_mesh(int num_devices_cpu)
{
	if (!get_gpu_value(num_devices)) {

		gpu_alloc_managed(ppcuMesh, num_devices_cpu);

		gpu_alloc_managed(ppdM_dt, num_devices_cpu);

		gpu_alloc_managed(ppspin_V_cmbnd_funcs_pri, num_devices_cpu);

		set_gpu_value(num_devices, num_devices_cpu);

		//null all entries
		for (int idx = 0; idx < num_devices_cpu; idx++) {

			ManagedMeshCUDA* cu_pointer_handle_pcuMesh = nullptr;
			cpu_to_gpu_managed(ppcuMesh, &cu_pointer_handle_pcuMesh, 1, idx);

			TransportCUDA_Spin_V_CMBND_Pri* cu_pointer_handle_ppspin_V_cmbnd_funcs_pri = nullptr;
			cpu_to_gpu_managed(ppspin_V_cmbnd_funcs_pri, &cu_pointer_handle_ppspin_V_cmbnd_funcs_pri, 1, idx);

			cuVEC_VC<cuReal3>* cu_pointer_handle_pdM_dt = nullptr;
			cpu_to_gpu_managed(ppdM_dt, &cu_pointer_handle_pdM_dt, 1, idx);
		}
	}
}

void TransportCUDA_Spin_S_CMBND_Sec::set_number_of_devices_amesh(int num_devices_cpu)
{
	if (!get_gpu_value(num_devices)) {

		gpu_alloc_managed(ppcuaMesh, num_devices_cpu);

		gpu_alloc_managed(ppdM_dt, num_devices_cpu);

		gpu_alloc_managed(ppspin_V_cmbnd_funcs_pri, num_devices_cpu);

		set_gpu_value(num_devices, num_devices_cpu);

		//null all entries
		for (int idx = 0; idx < num_devices_cpu; idx++) {

			ManagedAtom_MeshCUDA* cu_pointer_handle_pcuaMesh = nullptr;
			cpu_to_gpu_managed(ppcuaMesh, &cu_pointer_handle_pcuaMesh, 1, idx);

			TransportCUDA_Spin_V_CMBND_Pri* cu_pointer_handle_ppspin_V_cmbnd_funcs_pri = nullptr;
			cpu_to_gpu_managed(ppspin_V_cmbnd_funcs_pri, &cu_pointer_handle_ppspin_V_cmbnd_funcs_pri, 1, idx);

			cuVEC_VC<cuReal3>* cu_pointer_handle_pdM_dt = nullptr;
			cpu_to_gpu_managed(ppdM_dt, &cu_pointer_handle_pdM_dt, 1, idx);
		}
	}
}

BError TransportCUDA_Spin_S_CMBND_Sec::set_pointers(MeshCUDA* pMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int idx_device)
{
	BError error(__FUNCTION__);

	set_gpu_value(curr_device, idx_device);

	//TransportCUDA_Spin_S_CMBND_Sec objects are held in a mcu_obj, one for each device
	//However, each TransportCUDA_Spin_S_CMBND_Sec must hold pointers for all devices to allow access through UVA - thus loop here through allk devices and set pointers
	int num_devices_cpu = get_gpu_value(num_devices);
	for (int idx = 0; idx < num_devices_cpu; idx++) {

		if (cpu_to_gpu_managed(ppcuMesh, &pMeshCUDA->cuMesh.get_managed_object(idx), 1, idx) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
		if (cpu_to_gpu_managed(ppspin_V_cmbnd_funcs_pri, &pTransportBaseCUDA->spin_V_cmbnd_funcs_pri.get_managed_object(idx), 1, idx) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
		if (cpu_to_gpu_managed(ppdM_dt, &pTransportBaseCUDA->dM_dt.get_managed_object(idx), 1, idx) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	}

	if (set_gpu_value(stsolve, pTransportBaseCUDA->Get_STSolveType()) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	return error;
}

BError TransportCUDA_Spin_S_CMBND_Sec::set_pointers(Atom_MeshCUDA* paMeshCUDA, TransportBaseCUDA* pTransportBaseCUDA, int idx_device)
{
	BError error(__FUNCTION__);

	set_gpu_value(curr_device, idx_device);

	//TransportCUDA_Spin_S_CMBND_Sec objects are held in a mcu_obj, one for each device
	//However, each TransportCUDA_Spin_S_CMBND_Sec must hold pointers for all devices to allow access through UVA - thus loop here through allk devices and set pointers
	int num_devices_cpu = get_gpu_value(num_devices);
	for (int idx = 0; idx < num_devices_cpu; idx++) {

		if (cpu_to_gpu_managed(ppcuaMesh, &paMeshCUDA->cuaMesh.get_managed_object(idx), 1, idx) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
		if (cpu_to_gpu_managed(ppspin_V_cmbnd_funcs_pri, &pTransportBaseCUDA->spin_V_cmbnd_funcs_pri.get_managed_object(idx), 1, idx) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
		if (cpu_to_gpu_managed(ppdM_dt, &pTransportBaseCUDA->dM_dt.get_managed_object(idx), 1, idx) != cudaSuccess) error(BERROR_GPUERROR_CRIT);
	}

	if (set_gpu_value(stsolve, pTransportBaseCUDA->Get_STSolveType()) != cudaSuccess) error(BERROR_GPUERROR_CRIT);

	return error;
}

#endif

#endif