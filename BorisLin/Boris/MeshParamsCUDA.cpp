#include "stdafx.h"
#include "MeshParamsCUDA.h"
#include "MeshParams.h"
#include "mGPUConfig.h"

#if COMPILECUDA == 1

MeshParamsCUDA::MeshParamsCUDA(MeshParams *pmeshParams) :
	CurieWeiss_CUDA(mGPU),
	LongRelSus_CUDA(mGPU),
	CurieWeiss1_CUDA(mGPU),
	CurieWeiss2_CUDA(mGPU),
	LongRelSus1_CUDA(mGPU),
	LongRelSus2_CUDA(mGPU),
	Alpha1_CUDA(mGPU),
	Alpha2_CUDA(mGPU),
	grel(mGPU), grel_AFM(mGPU),
	alpha(mGPU), alpha_AFM(mGPU),
	Ms(mGPU), Ms_AFM(mGPU),
	Nxy(mGPU),
	A(mGPU), A_AFM(mGPU),
	Ah(mGPU), Anh(mGPU),
	D(mGPU), D_AFM(mGPU), Dh(mGPU), dh_dir(mGPU), D_dir(mGPU),
	tau_ii(mGPU), tau_ij(mGPU),
	J1(mGPU), J2(mGPU),
	K1(mGPU), K2(mGPU), K3(mGPU), mcanis_ea1(mGPU), mcanis_ea2(mGPU), mcanis_ea3(mGPU),
	K1_AFM(mGPU), K2_AFM(mGPU), K3_AFM(mGPU),
	Kt(mGPU), Kt2(mGPU),
	susrel(mGPU), susrel_AFM(mGPU), susprel(mGPU), cHA(mGPU), cHmo(mGPU),
	s_eff(mGPU),
	elecCond(mGPU),
	RAtmr_p(mGPU), RAtmr_ap(mGPU),
	amrPercentage(mGPU), tamrPercentage(mGPU),
	P(mGPU), beta(mGPU),
	De(mGPU), n_density(mGPU), betaD(mGPU), SHA(mGPU), iSHA(mGPU), 
	flSOT(mGPU), flSOT2(mGPU), STq(mGPU), STq2(mGPU), STa(mGPU), STa2(mGPU), STp(mGPU),
	l_sf(mGPU), l_ex(mGPU), l_ph(mGPU),
	Gi(mGPU), Gmix(mGPU),
	ts_eff(mGPU), tsi_eff(mGPU), pump_eff(mGPU), cpump_eff(mGPU), the_eff(mGPU),
	base_temperature(mGPU), T_Curie(mGPU),
	atomic_moment(mGPU), atomic_moment_AFM(mGPU),
	Sc(mGPU), joule_eff(mGPU),
	thermCond(mGPU),
	density(mGPU),
	MEc(mGPU), mMEc(mGPU), MEc2(mGPU), mMEc2(mGPU), MEc3(mGPU), mMEc3(mGPU), Ym(mGPU), Pr(mGPU), cC(mGPU), cC2(mGPU), cC3(mGPU), cCs(mGPU), mdamping(mGPU), thalpha(mGPU),
	shc(mGPU), shc_e(mGPU), G_e(mGPU), cT(mGPU), Q(mGPU)
{
	this->pmeshParams = pmeshParams;
	
	pmeshParams->grel.set_p_cu_obj_mpcuda(&grel);
	grel.set_from_cpu(pmeshParams->grel);
	pmeshParams->grel_AFM.set_p_cu_obj_mpcuda(&grel_AFM);
	grel_AFM.set_from_cpu(pmeshParams->grel_AFM);

	pmeshParams->alpha.set_p_cu_obj_mpcuda(&alpha);
	alpha.set_from_cpu(pmeshParams->alpha);
	pmeshParams->alpha_AFM.set_p_cu_obj_mpcuda(&alpha_AFM);
	alpha_AFM.set_from_cpu(pmeshParams->alpha_AFM);

	pmeshParams->Ms.set_p_cu_obj_mpcuda(&Ms);
	Ms.set_from_cpu(pmeshParams->Ms);
	pmeshParams->Ms_AFM.set_p_cu_obj_mpcuda(&Ms_AFM);
	Ms_AFM.set_from_cpu(pmeshParams->Ms_AFM);

	pmeshParams->Nxy.set_p_cu_obj_mpcuda(&Nxy);
	Nxy.set_from_cpu(pmeshParams->Nxy);
	
	pmeshParams->A.set_p_cu_obj_mpcuda(&A);
	A.set_from_cpu(pmeshParams->A);
	pmeshParams->A_AFM.set_p_cu_obj_mpcuda(&A_AFM);
	A_AFM.set_from_cpu(pmeshParams->A_AFM);
	pmeshParams->Ah.set_p_cu_obj_mpcuda(&Ah);
	Ah.set_from_cpu(pmeshParams->Ah);
	pmeshParams->Anh.set_p_cu_obj_mpcuda(&Anh);
	Anh.set_from_cpu(pmeshParams->Anh);
	
	pmeshParams->D.set_p_cu_obj_mpcuda(&D);
	D.set_from_cpu(pmeshParams->D);
	pmeshParams->D_AFM.set_p_cu_obj_mpcuda(&D_AFM);
	D_AFM.set_from_cpu(pmeshParams->D_AFM);
	pmeshParams->Dh.set_p_cu_obj_mpcuda(&Dh);
	Dh.set_from_cpu(pmeshParams->Dh);
	pmeshParams->dh_dir.set_p_cu_obj_mpcuda(&dh_dir);
	dh_dir.set_from_cpu(pmeshParams->dh_dir);
	pmeshParams->D_dir.set_p_cu_obj_mpcuda(&D_dir);
	D_dir.set_from_cpu(pmeshParams->D_dir);
	
	pmeshParams->tau_ii.set_p_cu_obj_mpcuda(&tau_ii);
	tau_ii.set_from_cpu(pmeshParams->tau_ii);
	pmeshParams->tau_ij.set_p_cu_obj_mpcuda(&tau_ij);
	tau_ij.set_from_cpu(pmeshParams->tau_ij);
	
	pmeshParams->J1.set_p_cu_obj_mpcuda(&J1);
	J1.set_from_cpu(pmeshParams->J1);
	pmeshParams->J2.set_p_cu_obj_mpcuda(&J2);
	J2.set_from_cpu(pmeshParams->J2);
	
	pmeshParams->K1.set_p_cu_obj_mpcuda(&K1);
	K1.set_from_cpu(pmeshParams->K1);
	pmeshParams->K2.set_p_cu_obj_mpcuda(&K2);
	K2.set_from_cpu(pmeshParams->K2);
	pmeshParams->K3.set_p_cu_obj_mpcuda(&K3);
	K3.set_from_cpu(pmeshParams->K3);
	pmeshParams->mcanis_ea1.set_p_cu_obj_mpcuda(&mcanis_ea1);
	mcanis_ea1.set_from_cpu(pmeshParams->mcanis_ea1);
	pmeshParams->mcanis_ea2.set_p_cu_obj_mpcuda(&mcanis_ea2);
	mcanis_ea2.set_from_cpu(pmeshParams->mcanis_ea2);
	pmeshParams->mcanis_ea3.set_p_cu_obj_mpcuda(&mcanis_ea3);
	mcanis_ea3.set_from_cpu(pmeshParams->mcanis_ea3);
	pmeshParams->K1_AFM.set_p_cu_obj_mpcuda(&K1_AFM);
	K1_AFM.set_from_cpu(pmeshParams->K1_AFM);
	pmeshParams->K2_AFM.set_p_cu_obj_mpcuda(&K2_AFM);
	K2_AFM.set_from_cpu(pmeshParams->K2_AFM);
	pmeshParams->K3_AFM.set_p_cu_obj_mpcuda(&K3_AFM);
	K3_AFM.set_from_cpu(pmeshParams->K3_AFM);
	
	pmeshParams->susrel.set_p_cu_obj_mpcuda(&susrel);
	susrel.set_from_cpu(pmeshParams->susrel);
	pmeshParams->susrel_AFM.set_p_cu_obj_mpcuda(&susrel_AFM);
	susrel_AFM.set_from_cpu(pmeshParams->susrel_AFM);
	pmeshParams->susprel.set_p_cu_obj_mpcuda(&susprel);
	susprel.set_from_cpu(pmeshParams->susprel);
	
	pmeshParams->cHA.set_p_cu_obj_mpcuda(&cHA);
	cHA.set_from_cpu(pmeshParams->cHA);
	pmeshParams->cHmo.set_p_cu_obj_mpcuda(&cHmo);
	cHmo.set_from_cpu(pmeshParams->cHmo);
	
	pmeshParams->s_eff.set_p_cu_obj_mpcuda(&s_eff);
	s_eff.set_from_cpu(pmeshParams->s_eff);
	
	pmeshParams->elecCond.set_p_cu_obj_mpcuda(&elecCond);
	elecCond.set_from_cpu(pmeshParams->elecCond);
	pmeshParams->amrPercentage.set_p_cu_obj_mpcuda(&amrPercentage);
	amrPercentage.set_from_cpu(pmeshParams->amrPercentage);
	pmeshParams->tamrPercentage.set_p_cu_obj_mpcuda(&tamrPercentage);
	tamrPercentage.set_from_cpu(pmeshParams->tamrPercentage);
	pmeshParams->RAtmr_p.set_p_cu_obj_mpcuda(&RAtmr_p);
	RAtmr_p.set_from_cpu(pmeshParams->RAtmr_p);
	pmeshParams->RAtmr_ap.set_p_cu_obj_mpcuda(&RAtmr_ap);
	RAtmr_ap.set_from_cpu(pmeshParams->RAtmr_ap);
	
	pmeshParams->P.set_p_cu_obj_mpcuda(&P);
	P.set_from_cpu(pmeshParams->P);
	pmeshParams->beta.set_p_cu_obj_mpcuda(&beta);
	beta.set_from_cpu(pmeshParams->beta);
	
	pmeshParams->De.set_p_cu_obj_mpcuda(&De);
	De.set_from_cpu(pmeshParams->De);
	
	pmeshParams->n_density.set_p_cu_obj_mpcuda(&n_density);
	n_density.set_from_cpu(pmeshParams->n_density);
	
	pmeshParams->betaD.set_p_cu_obj_mpcuda(&betaD);
	betaD.set_from_cpu(pmeshParams->betaD);
	
	pmeshParams->SHA.set_p_cu_obj_mpcuda(&SHA);
	SHA.set_from_cpu(pmeshParams->SHA);
	pmeshParams->iSHA.set_p_cu_obj_mpcuda(&iSHA);
	iSHA.set_from_cpu(pmeshParams->iSHA);
	pmeshParams->flSOT.set_p_cu_obj_mpcuda(&flSOT);
	flSOT.set_from_cpu(pmeshParams->flSOT);
	pmeshParams->flSOT2.set_p_cu_obj_mpcuda(&flSOT2);
	flSOT2.set_from_cpu(pmeshParams->flSOT2);
	
	pmeshParams->STq.set_p_cu_obj_mpcuda(&STq);
	STq.set_from_cpu(pmeshParams->STq);
	pmeshParams->STq2.set_p_cu_obj_mpcuda(&STq2);
	STq2.set_from_cpu(pmeshParams->STq2);
	pmeshParams->STa.set_p_cu_obj_mpcuda(&STa);
	STa.set_from_cpu(pmeshParams->STa);
	pmeshParams->STa2.set_p_cu_obj_mpcuda(&STa2);
	STa2.set_from_cpu(pmeshParams->STa2);
	pmeshParams->STp.set_p_cu_obj_mpcuda(&STp);
	STp.set_from_cpu(pmeshParams->STp);
	
	pmeshParams->l_sf.set_p_cu_obj_mpcuda(&l_sf);
	l_sf.set_from_cpu(pmeshParams->l_sf);
	pmeshParams->l_ex.set_p_cu_obj_mpcuda(&l_ex);
	l_ex.set_from_cpu(pmeshParams->l_ex);
	pmeshParams->l_ph.set_p_cu_obj_mpcuda(&l_ph);
	l_ph.set_from_cpu(pmeshParams->l_ph);
	
	pmeshParams->Gi.set_p_cu_obj_mpcuda(&Gi);
	Gi.set_from_cpu(pmeshParams->Gi);
	pmeshParams->Gmix.set_p_cu_obj_mpcuda(&Gmix);
	Gmix.set_from_cpu(pmeshParams->Gmix);
	
	pmeshParams->ts_eff.set_p_cu_obj_mpcuda(&ts_eff);
	ts_eff.set_from_cpu(pmeshParams->ts_eff);
	pmeshParams->tsi_eff.set_p_cu_obj_mpcuda(&tsi_eff);
	tsi_eff.set_from_cpu(pmeshParams->tsi_eff);
	pmeshParams->pump_eff.set_p_cu_obj_mpcuda(&pump_eff);
	pump_eff.set_from_cpu(pmeshParams->pump_eff);
	pmeshParams->cpump_eff.set_p_cu_obj_mpcuda(&cpump_eff);
	cpump_eff.set_from_cpu(pmeshParams->cpump_eff);
	pmeshParams->the_eff.set_p_cu_obj_mpcuda(&the_eff);
	the_eff.set_from_cpu(pmeshParams->the_eff);
	
	base_temperature.from_cpu(pmeshParams->base_temperature);
	T_Curie.from_cpu(pmeshParams->T_Curie);

	pmeshParams->atomic_moment.set_p_cu_obj_mpcuda(&atomic_moment);
	atomic_moment.set_from_cpu(pmeshParams->atomic_moment);
	pmeshParams->atomic_moment_AFM.set_p_cu_obj_mpcuda(&atomic_moment_AFM);
	atomic_moment_AFM.set_from_cpu(pmeshParams->atomic_moment_AFM);

	pmeshParams->Sc.set_p_cu_obj_mpcuda(&Sc);
	Sc.set_from_cpu(pmeshParams->Sc);
	pmeshParams->joule_eff.set_p_cu_obj_mpcuda(&joule_eff);
	joule_eff.set_from_cpu(pmeshParams->joule_eff);

	pmeshParams->thermCond.set_p_cu_obj_mpcuda(&thermCond);
	thermCond.set_from_cpu(pmeshParams->thermCond);
	pmeshParams->density.set_p_cu_obj_mpcuda(&density);
	density.set_from_cpu(pmeshParams->density);
	
	pmeshParams->shc.set_p_cu_obj_mpcuda(&shc);
	shc.set_from_cpu(pmeshParams->shc);
	pmeshParams->shc_e.set_p_cu_obj_mpcuda(&shc_e);
	shc_e.set_from_cpu(pmeshParams->shc_e);
	pmeshParams->G_e.set_p_cu_obj_mpcuda(&G_e);
	G_e.set_from_cpu(pmeshParams->G_e);
	
	pmeshParams->cT.set_p_cu_obj_mpcuda(&cT);
	cT.set_from_cpu(pmeshParams->cT);
	
	pmeshParams->Q.set_p_cu_obj_mpcuda(&Q);
	Q.set_from_cpu(pmeshParams->Q);
	
	pmeshParams->MEc.set_p_cu_obj_mpcuda(&MEc);
	MEc.set_from_cpu(pmeshParams->MEc);
	pmeshParams->mMEc.set_p_cu_obj_mpcuda(&mMEc);
	mMEc.set_from_cpu(pmeshParams->mMEc);
	pmeshParams->MEc2.set_p_cu_obj_mpcuda(&MEc2);
	MEc2.set_from_cpu(pmeshParams->MEc2);
	pmeshParams->mMEc2.set_p_cu_obj_mpcuda(&mMEc2);
	mMEc2.set_from_cpu(pmeshParams->mMEc2);
	pmeshParams->MEc3.set_p_cu_obj_mpcuda(&MEc3);
	MEc3.set_from_cpu(pmeshParams->MEc3);
	pmeshParams->mMEc3.set_p_cu_obj_mpcuda(&mMEc3);
	mMEc3.set_from_cpu(pmeshParams->mMEc3);
	pmeshParams->Ym.set_p_cu_obj_mpcuda(&Ym);
	Ym.set_from_cpu(pmeshParams->Ym);
	pmeshParams->Pr.set_p_cu_obj_mpcuda(&Pr);
	Pr.set_from_cpu(pmeshParams->Pr);
	pmeshParams->cC.set_p_cu_obj_mpcuda(&cC);
	cC.set_from_cpu(pmeshParams->cC);
	pmeshParams->cC2.set_p_cu_obj_mpcuda(&cC2);
	cC2.set_from_cpu(pmeshParams->cC2);
	pmeshParams->cC3.set_p_cu_obj_mpcuda(&cC3);
	cC3.set_from_cpu(pmeshParams->cC3);
	pmeshParams->cCs.set_p_cu_obj_mpcuda(&cCs);
	cCs.set_from_cpu(pmeshParams->cCs);
	pmeshParams->mdamping.set_p_cu_obj_mpcuda(&mdamping);
	mdamping.set_from_cpu(pmeshParams->mdamping);

	pmeshParams->thalpha.set_p_cu_obj_mpcuda(&thalpha);
	thalpha.set_from_cpu(pmeshParams->thalpha);

	//setup CUDA special functions to corresponding data held in the cpu objects
	set_special_functions_data();

	//make sure special functions are set by default for all material parameters text equations
	set_special_functions();
}

//set pre-calculated Funcs_Special objects in enabled material parameters
void MeshParamsCUDA::set_special_functions(PARAM_ paramID)
{
	auto set_param_special_functions = [&](PARAM_ update_paramID) {

		auto code = [&](auto& MatP_object) -> void {

			MatP_object.set_t_scaling_special_functions_CUDA(&CurieWeiss_CUDA, &LongRelSus_CUDA, &CurieWeiss1_CUDA, &CurieWeiss2_CUDA, &LongRelSus1_CUDA, &LongRelSus2_CUDA, &Alpha1_CUDA, &Alpha2_CUDA);
		};

		pmeshParams->run_on_param<void>(update_paramID, code);
	};

	if (paramID == PARAM_ALL) {

		for (int index = 0; index < pmeshParams->meshParams.size(); index++) {

			set_param_special_functions((PARAM_)pmeshParams->meshParams.get_ID_from_index(index));
		}
	}
	else set_param_special_functions(paramID);
}

void MeshParamsCUDA::set_special_functions_data(void)
{
	//setup CUDA special functions to corresponding data held in the cpu objects
	CurieWeiss_CUDA.set_data(pmeshParams->pCurieWeiss->get_data(), pmeshParams->pCurieWeiss->get_start(), pmeshParams->pCurieWeiss->get_resolution());
	LongRelSus_CUDA.set_data(pmeshParams->pLongRelSus->get_data(), pmeshParams->pLongRelSus->get_start(), pmeshParams->pLongRelSus->get_resolution());

	CurieWeiss1_CUDA.set_data(pmeshParams->pCurieWeiss1->get_data(), pmeshParams->pCurieWeiss1->get_start(), pmeshParams->pCurieWeiss1->get_resolution());
	CurieWeiss2_CUDA.set_data(pmeshParams->pCurieWeiss2->get_data(), pmeshParams->pCurieWeiss2->get_start(), pmeshParams->pCurieWeiss2->get_resolution());
	LongRelSus1_CUDA.set_data(pmeshParams->pLongRelSus1->get_data(), pmeshParams->pLongRelSus1->get_start(), pmeshParams->pLongRelSus1->get_resolution());
	LongRelSus2_CUDA.set_data(pmeshParams->pLongRelSus2->get_data(), pmeshParams->pLongRelSus2->get_start(), pmeshParams->pLongRelSus2->get_resolution());

	Alpha1_CUDA.set_data(pmeshParams->pAlpha1->get_data(), pmeshParams->pAlpha1->get_start(), pmeshParams->pAlpha1->get_resolution());
	Alpha2_CUDA.set_data(pmeshParams->pAlpha2->get_data(), pmeshParams->pAlpha2->get_start(), pmeshParams->pAlpha2->get_resolution());
}

MeshParamsCUDA::~MeshParamsCUDA()
{
	//fine to access data in MeshParams here : Mesh inherits from MeshParams, so in the destruction process Mesh gets destroyed first. 
	//Mesh destructor then calls for MeshCUDA implementation to be destroyed, then we get here since MeshCUDA inherits from MeshParamsCUDA. After this we return back to Mesh destructor to continue destruction down the list.
	
	pmeshParams->grel.null_p_cu_obj_mpcuda();
	pmeshParams->grel_AFM.null_p_cu_obj_mpcuda();

	pmeshParams->alpha.null_p_cu_obj_mpcuda();
	pmeshParams->alpha_AFM.null_p_cu_obj_mpcuda();

	pmeshParams->Ms.null_p_cu_obj_mpcuda();
	pmeshParams->Ms_AFM.null_p_cu_obj_mpcuda();

	pmeshParams->Nxy.null_p_cu_obj_mpcuda();

	pmeshParams->A.null_p_cu_obj_mpcuda();
	pmeshParams->A_AFM.null_p_cu_obj_mpcuda();
	
	pmeshParams->Ah.null_p_cu_obj_mpcuda();
	pmeshParams->Anh.null_p_cu_obj_mpcuda();

	pmeshParams->D.null_p_cu_obj_mpcuda();
	pmeshParams->D_AFM.null_p_cu_obj_mpcuda();
	pmeshParams->Dh.null_p_cu_obj_mpcuda();
	pmeshParams->dh_dir.null_p_cu_obj_mpcuda();

	pmeshParams->D_dir.null_p_cu_obj_mpcuda();

	pmeshParams->tau_ii.null_p_cu_obj_mpcuda();
	pmeshParams->tau_ij.null_p_cu_obj_mpcuda();

	pmeshParams->J1.null_p_cu_obj_mpcuda();
	pmeshParams->J2.null_p_cu_obj_mpcuda();

	pmeshParams->K1.null_p_cu_obj_mpcuda();
	pmeshParams->K2.null_p_cu_obj_mpcuda();
	pmeshParams->K3.null_p_cu_obj_mpcuda();

	pmeshParams->mcanis_ea1.null_p_cu_obj_mpcuda();
	pmeshParams->mcanis_ea2.null_p_cu_obj_mpcuda();
	pmeshParams->mcanis_ea3.null_p_cu_obj_mpcuda();

	pmeshParams->K1_AFM.null_p_cu_obj_mpcuda();
	pmeshParams->K2_AFM.null_p_cu_obj_mpcuda();
	pmeshParams->K3_AFM.null_p_cu_obj_mpcuda();

	pmeshParams->susrel.null_p_cu_obj_mpcuda();
	pmeshParams->susrel_AFM.null_p_cu_obj_mpcuda();
	pmeshParams->susprel.null_p_cu_obj_mpcuda();

	pmeshParams->cHA.null_p_cu_obj_mpcuda();
	pmeshParams->cHmo.null_p_cu_obj_mpcuda();

	pmeshParams->s_eff.null_p_cu_obj_mpcuda();

	pmeshParams->elecCond.null_p_cu_obj_mpcuda();
	pmeshParams->amrPercentage.null_p_cu_obj_mpcuda();
	pmeshParams->tamrPercentage.null_p_cu_obj_mpcuda();
	pmeshParams->RAtmr_p.null_p_cu_obj_mpcuda();
	pmeshParams->RAtmr_ap.null_p_cu_obj_mpcuda();

	pmeshParams->P.null_p_cu_obj_mpcuda();
	pmeshParams->beta.null_p_cu_obj_mpcuda();

	pmeshParams->De.null_p_cu_obj_mpcuda();
	pmeshParams->n_density.null_p_cu_obj_mpcuda();
	pmeshParams->betaD.null_p_cu_obj_mpcuda();
	
	pmeshParams->SHA.null_p_cu_obj_mpcuda();
	pmeshParams->iSHA.null_p_cu_obj_mpcuda();
	pmeshParams->flSOT.null_p_cu_obj_mpcuda();
	pmeshParams->flSOT2.null_p_cu_obj_mpcuda();

	pmeshParams->STq.null_p_cu_obj_mpcuda();
	pmeshParams->STq2.null_p_cu_obj_mpcuda();
	pmeshParams->STa.null_p_cu_obj_mpcuda();
	pmeshParams->STa2.null_p_cu_obj_mpcuda();
	pmeshParams->STp.null_p_cu_obj_mpcuda();
	
	pmeshParams->l_sf.null_p_cu_obj_mpcuda();
	pmeshParams->l_ex.null_p_cu_obj_mpcuda();
	pmeshParams->l_ph.null_p_cu_obj_mpcuda();
	
	pmeshParams->Gi.null_p_cu_obj_mpcuda();
	pmeshParams->Gmix.null_p_cu_obj_mpcuda();
	
	pmeshParams->ts_eff.null_p_cu_obj_mpcuda();
	pmeshParams->tsi_eff.null_p_cu_obj_mpcuda();
	
	pmeshParams->pump_eff.null_p_cu_obj_mpcuda();
	pmeshParams->cpump_eff.null_p_cu_obj_mpcuda();
	pmeshParams->the_eff.null_p_cu_obj_mpcuda();

	pmeshParams->atomic_moment.null_p_cu_obj_mpcuda();
	pmeshParams->atomic_moment_AFM.null_p_cu_obj_mpcuda();

	pmeshParams->Sc.null_p_cu_obj_mpcuda();
	pmeshParams->joule_eff.null_p_cu_obj_mpcuda();

	pmeshParams->thermCond.null_p_cu_obj_mpcuda();
	pmeshParams->density.null_p_cu_obj_mpcuda();
	
	pmeshParams->shc.null_p_cu_obj_mpcuda();
	pmeshParams->shc_e.null_p_cu_obj_mpcuda();
	pmeshParams->G_e.null_p_cu_obj_mpcuda();

	pmeshParams->cT.null_p_cu_obj_mpcuda();
	pmeshParams->Q.null_p_cu_obj_mpcuda();

	pmeshParams->MEc.null_p_cu_obj_mpcuda();
	pmeshParams->mMEc.null_p_cu_obj_mpcuda();
	pmeshParams->MEc2.null_p_cu_obj_mpcuda();
	pmeshParams->mMEc2.null_p_cu_obj_mpcuda();
	pmeshParams->MEc3.null_p_cu_obj_mpcuda();
	pmeshParams->mMEc3.null_p_cu_obj_mpcuda();
	pmeshParams->Ym.null_p_cu_obj_mpcuda();
	pmeshParams->Pr.null_p_cu_obj_mpcuda();
	pmeshParams->cC.null_p_cu_obj_mpcuda();
	pmeshParams->cC2.null_p_cu_obj_mpcuda();
	pmeshParams->cC3.null_p_cu_obj_mpcuda();
	pmeshParams->cCs.null_p_cu_obj_mpcuda();
	pmeshParams->mdamping.null_p_cu_obj_mpcuda();

	pmeshParams->thalpha.null_p_cu_obj_mpcuda();
}

#endif