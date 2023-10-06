#include "stdafx.h"
#include "ExchangeBaseCUDA.h"
#include "ExchangeBase.h"

#include "Mesh.h"
#include "Atom_Mesh.h"

#include "MeshCUDA.h"
#include "ManagedMeshPolicyCUDA.h"
#include "Atom_MeshCUDA.h"
#include "ManagedAtom_MeshPolicyCUDA.h"

#if COMPILECUDA == 1

#include "MeshCUDA.h"

ExchangeBaseCUDA::ExchangeBaseCUDA(MeshBaseCUDA* pMeshBaseCUDA_, ExchangeBase* pExchBase_)
{
	pMeshBaseCUDA = pMeshBaseCUDA_;
	pExchBase = pExchBase_;
}

ExchangeBaseCUDA::~ExchangeBaseCUDA() {}

BError ExchangeBaseCUDA::Initialize(void)
{
	BError error(CLASS_STR(ExchangeBaseCUDA));

	//initialize the cpu version of ExchangeBase
	//doing this will set cmbnd flags in M VECs which we can copy to gpu version here
	pExchBase->Initialize();

	//clear everything then rebuild
	CMBNDcontactsCUDA.clear();
	CMBNDcontacts.clear();
	pContactingManagedMeshes.clear();
	pContactingManagedAtomMeshes.clear();
	pContactingMeshes.clear();

	//copy managed cuda meshes to pContactingManagedMeshes and pContactingManagedAtomMeshes using the pMeshes initialized in ExchangeBase
	for (int idx = 0; idx < pExchBase->pMeshes.size(); idx++) {

		pContactingMeshes.push_back(pExchBase->pMeshes[idx]->pMeshBaseCUDA);

		if (!pExchBase->pMeshes[idx]->is_atomistic()) {

			pContactingManagedMeshes.push_back(&(dynamic_cast<Mesh*>(pExchBase->pMeshes[idx])->pMeshCUDA->cuMesh));
			pContactingManagedAtomMeshes.push_back(nullptr);
		}
		else {

			pContactingManagedAtomMeshes.push_back(&(dynamic_cast<Atom_Mesh*>(pExchBase->pMeshes[idx])->paMeshCUDA->cuaMesh));
			pContactingManagedMeshes.push_back(nullptr);
		}
	}

	//set cmbnd flags
	if (!pMeshBaseCUDA->is_atomistic()) {

		if (!dynamic_cast<MeshCUDA*>(pMeshBaseCUDA)->M.copyflags_from_cpuvec(dynamic_cast<Mesh*>(pExchBase->pMeshBase)->M)) error(BERROR_GPUERROR_CRIT);
	}
	else {

		if (!dynamic_cast<Atom_MeshCUDA*>(pMeshBaseCUDA)->M1.copyflags_from_cpuvec(dynamic_cast<Atom_Mesh*>(pExchBase->pMeshBase)->M1)) error(BERROR_GPUERROR_CRIT);
	}

	//copy CMBNDInfo for contacts
	for (int idx_contact = 0; idx_contact < pExchBase->CMBNDcontacts.size(); idx_contact++) {

		mCMBNDInfoCUDA contact(mGPU);
		if (!pMeshBaseCUDA->is_atomistic()) {

			contact.copy_from_CMBNDInfo<CMBNDInfo>(pExchBase->CMBNDcontacts[idx_contact], dynamic_cast<MeshCUDA*>(pMeshBaseCUDA)->M.get_pbox_d_ref());
		}
		else {

			contact.copy_from_CMBNDInfo<CMBNDInfo>(pExchBase->CMBNDcontacts[idx_contact], dynamic_cast<Atom_MeshCUDA*>(pMeshBaseCUDA)->M1.get_pbox_d_ref());
		}

		CMBNDcontactsCUDA.push_back(contact);
		CMBNDcontacts.push_back(pExchBase->CMBNDcontacts[idx_contact]);
	}

	return error;
}

#endif
