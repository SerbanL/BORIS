#pragma once

#include "mcuVEC.h"
#include "mcuVEC_cmbnd.h"

template <typename VType, typename MType>
template <typename Class_CMBNDs, typename Class_PolicyCMBNDs, typename Class_CMBNDp, typename Class_PolicyCMBNDp, typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::set_cmbnd_continuous(
	std::vector<cu_obj<mcuVEC_Managed<MType, VType>>*>& man_mcuVEC_sec, 
	mcu_obj<Class_CMBNDs, Class_PolicyCMBNDs>& mcmbndFuncs_sec, 
	mcu_obj<Class_CMBNDp, Class_PolicyCMBNDp>& mcmbndFuncs_pri, 
	mCMBNDInfoCUDA& contact)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		size_t size = contact.contact_size(mGPU);
		if (!size) continue;
		
		mng(mGPU)->set_cmbnd_continuous(size, man_mcuVEC_sec[mGPU]->get_dereferenced(), mcmbndFuncs_sec.get_deviceobject(mGPU), mcmbndFuncs_pri.get_deviceobject(mGPU), contact.get_deviceobject(mGPU));
	}
}

template <typename VType, typename MType>
template <typename Class_CMBNDs, typename Class_PolicyCMBNDs, typename Class_CMBNDp, typename Class_PolicyCMBNDp, typename Class_CMBND_S, typename Class_PolicyCMBND_S, typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::set_cmbnd_continuousflux(
	std::vector<cu_obj<mcuVEC_Managed<MType, VType>>*>& man_mcuVEC_sec,
	mcu_obj<Class_CMBNDs, Class_PolicyCMBNDs>& mcmbndFuncs_sec,
	mcu_obj<Class_CMBNDp, Class_PolicyCMBNDp>& mcmbndFuncs_pri,
	mcu_obj<Class_CMBND_S, Class_PolicyCMBND_S>& mcmbndFuncs_s,
	mCMBNDInfoCUDA& contact)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		size_t size = contact.contact_size(mGPU);
		if (!size) continue;

		mng(mGPU)->set_cmbnd_continuousflux(size, man_mcuVEC_sec[mGPU]->get_dereferenced(), mcmbndFuncs_sec.get_deviceobject(mGPU), mcmbndFuncs_pri.get_deviceobject(mGPU), mcmbndFuncs_s.get_deviceobject(mGPU), contact.get_deviceobject(mGPU));
	}
}

template <typename VType, typename MType>
template <typename Class_CMBNDs, typename Class_PolicyCMBNDs, typename Class_CMBNDp, typename Class_PolicyCMBNDp, typename Class_CMBND_S, typename Class_PolicyCMBND_S, typename VType_, typename MType_, std::enable_if_t<std::is_same<MType_, cuVEC_VC<VType_>>::value>*>
void mcuVEC<VType, MType>::set_cmbnd_discontinuous(
	std::vector<cu_obj<mcuVEC_Managed<MType, VType>>*>& man_mcuVEC_sec,
	mcu_obj<Class_CMBNDs, Class_PolicyCMBNDs>& mcmbndFuncs_sec,
	mcu_obj<Class_CMBNDp, Class_PolicyCMBNDp>& mcmbndFuncs_pri,
	mcu_obj<Class_CMBND_S, Class_PolicyCMBND_S>& mcmbndFuncs_s,
	mCMBNDInfoCUDA& contact)
{
	for (mGPU.device_begin(); mGPU != mGPU.device_end(); mGPU++) {

		size_t size = contact.contact_size(mGPU);
		if (!size) continue;
		
		mng(mGPU)->set_cmbnd_discontinuous(size, man_mcuVEC_sec[mGPU]->get_dereferenced(), mcmbndFuncs_sec.get_deviceobject(mGPU), mcmbndFuncs_pri.get_deviceobject(mGPU), mcmbndFuncs_s.get_deviceobject(mGPU), contact.get_deviceobject(mGPU));
	}
}