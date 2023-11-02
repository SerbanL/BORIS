#include "stdafx.h"
#include "Atom_MeshParamsCUDA.h"
#include "Atom_MeshParams.h"
#include "mGPUConfig.h"

#if COMPILECUDA == 1

Atom_MeshParamsCUDA::Atom_MeshParamsCUDA(Atom_MeshParams *pameshParams) :
	grel(mGPU),
	alpha(mGPU),
	mu_s(mGPU),
	J(mGPU),
	D(mGPU), D_dir(mGPU),
	Js(mGPU), Js2(mGPU),
	K1(mGPU), K2(mGPU), K3(mGPU), mcanis_ea1(mGPU), mcanis_ea2(mGPU), mcanis_ea3(mGPU), Kt(mGPU),
	Nxy(mGPU), cHA(mGPU), cHmo(mGPU),
	s_eff(mGPU),
	elecCond(mGPU),
	RAtmr_p(mGPU), RAtmr_ap(mGPU),
	amrPercentage(mGPU), tamrPercentage(mGPU),
	P(mGPU), beta(mGPU),
	De(mGPU), n_density(mGPU), 
	betaD(mGPU), 
	SHA(mGPU), flSOT(mGPU), flSOT2(mGPU), STq(mGPU), STq2(mGPU), STa(mGPU), STa2(mGPU), STp(mGPU),
	l_sf(mGPU), l_ex(mGPU), l_ph(mGPU),
	Gi(mGPU), Gmix(mGPU),
	ts_eff(mGPU), tsi_eff(mGPU), pump_eff(mGPU), cpump_eff(mGPU), the_eff(mGPU),
	base_temperature(mGPU),
	Sc(mGPU), joule_eff(mGPU),
	thermCond(mGPU), density(mGPU),
	shc(mGPU), shc_e(mGPU), G_e(mGPU), cT(mGPU), Q(mGPU)
{
	this->pameshParams = pameshParams;

	//-----------SIMPLE CUBIC

	pameshParams->grel.set_p_cu_obj_mpcuda(&grel);
	grel.set_from_cpu(pameshParams->grel);
	
	pameshParams->alpha.set_p_cu_obj_mpcuda(&alpha);
	alpha.set_from_cpu(pameshParams->alpha);
	
	pameshParams->mu_s.set_p_cu_obj_mpcuda(&mu_s);
	mu_s.set_from_cpu(pameshParams->mu_s);
	
	pameshParams->J.set_p_cu_obj_mpcuda(&J);
	J.set_from_cpu(pameshParams->J);
	
	pameshParams->D.set_p_cu_obj_mpcuda(&D);
	D.set_from_cpu(pameshParams->D);
	pameshParams->D_dir.set_p_cu_obj_mpcuda(&D_dir);
	D_dir.set_from_cpu(pameshParams->D_dir);
	
	pameshParams->Js.set_p_cu_obj_mpcuda(&Js);
	Js.set_from_cpu(pameshParams->Js);
	pameshParams->Js2.set_p_cu_obj_mpcuda(&Js2);
	Js2.set_from_cpu(pameshParams->Js2);
	
	pameshParams->K1.set_p_cu_obj_mpcuda(&K1);
	K1.set_from_cpu(pameshParams->K1);
	pameshParams->K2.set_p_cu_obj_mpcuda(&K2);
	K2.set_from_cpu(pameshParams->K2);
	pameshParams->K3.set_p_cu_obj_mpcuda(&K3);
	K3.set_from_cpu(pameshParams->K3);
	pameshParams->mcanis_ea1.set_p_cu_obj_mpcuda(&mcanis_ea1);
	mcanis_ea1.set_from_cpu(pameshParams->mcanis_ea1);
	pameshParams->mcanis_ea2.set_p_cu_obj_mpcuda(&mcanis_ea2);
	mcanis_ea2.set_from_cpu(pameshParams->mcanis_ea2);
	pameshParams->mcanis_ea3.set_p_cu_obj_mpcuda(&mcanis_ea3);
	mcanis_ea3.set_from_cpu(pameshParams->mcanis_ea3);

	//-----------Others

	pameshParams->Nxy.set_p_cu_obj_mpcuda(&Nxy);
	Nxy.set_from_cpu(pameshParams->Nxy);
	
	pameshParams->cHA.set_p_cu_obj_mpcuda(&cHA);
	cHA.set_from_cpu(pameshParams->cHA);
	
	pameshParams->cHmo.set_p_cu_obj_mpcuda(&cHmo);
	cHmo.set_from_cpu(pameshParams->cHmo);
	
	pameshParams->s_eff.set_p_cu_obj_mpcuda(&s_eff);
	s_eff.set_from_cpu(pameshParams->s_eff);
	
	pameshParams->elecCond.set_p_cu_obj_mpcuda(&elecCond);
	elecCond.set_from_cpu(pameshParams->elecCond);
	pameshParams->amrPercentage.set_p_cu_obj_mpcuda(&amrPercentage);
	amrPercentage.set_from_cpu(pameshParams->amrPercentage);
	pameshParams->tamrPercentage.set_p_cu_obj_mpcuda(&tamrPercentage);
	tamrPercentage.set_from_cpu(pameshParams->tamrPercentage);
	pameshParams->RAtmr_p.set_p_cu_obj_mpcuda(&RAtmr_p);
	RAtmr_p.set_from_cpu(pameshParams->RAtmr_p);
	pameshParams->RAtmr_ap.set_p_cu_obj_mpcuda(&RAtmr_ap);
	RAtmr_ap.set_from_cpu(pameshParams->RAtmr_ap);
	
	pameshParams->P.set_p_cu_obj_mpcuda(&P);
	P.set_from_cpu(pameshParams->P);
	pameshParams->beta.set_p_cu_obj_mpcuda(&beta);
	beta.set_from_cpu(pameshParams->beta);
	
	pameshParams->De.set_p_cu_obj_mpcuda(&De);
	De.set_from_cpu(pameshParams->De);
	
	pameshParams->n_density.set_p_cu_obj_mpcuda(&n_density);
	n_density.set_from_cpu(pameshParams->n_density);
	
	pameshParams->betaD.set_p_cu_obj_mpcuda(&betaD);
	betaD.set_from_cpu(pameshParams->betaD);
	
	pameshParams->SHA.set_p_cu_obj_mpcuda(&SHA);
	SHA.set_from_cpu(pameshParams->SHA);
	pameshParams->flSOT.set_p_cu_obj_mpcuda(&flSOT);
	flSOT.set_from_cpu(pameshParams->flSOT);
	pameshParams->flSOT2.set_p_cu_obj_mpcuda(&flSOT2);
	flSOT2.set_from_cpu(pameshParams->flSOT2);
	
	pameshParams->STq.set_p_cu_obj_mpcuda(&STq);
	STq.set_from_cpu(pameshParams->STq);
	pameshParams->STq2.set_p_cu_obj_mpcuda(&STq2);
	STq2.set_from_cpu(pameshParams->STq2);
	pameshParams->STa.set_p_cu_obj_mpcuda(&STa);
	STa.set_from_cpu(pameshParams->STa);
	pameshParams->STa2.set_p_cu_obj_mpcuda(&STa2);
	STa2.set_from_cpu(pameshParams->STa2);
	pameshParams->STp.set_p_cu_obj_mpcuda(&STp);
	STp.set_from_cpu(pameshParams->STp);
	
	pameshParams->l_sf.set_p_cu_obj_mpcuda(&l_sf);
	l_sf.set_from_cpu(pameshParams->l_sf);
	pameshParams->l_ex.set_p_cu_obj_mpcuda(&l_ex);
	l_ex.set_from_cpu(pameshParams->l_ex);
	pameshParams->l_ph.set_p_cu_obj_mpcuda(&l_ph);
	l_ph.set_from_cpu(pameshParams->l_ph);
	
	pameshParams->Gi.set_p_cu_obj_mpcuda(&Gi);
	Gi.set_from_cpu(pameshParams->Gi);
	pameshParams->Gmix.set_p_cu_obj_mpcuda(&Gmix);
	Gmix.set_from_cpu(pameshParams->Gmix);
	
	pameshParams->ts_eff.set_p_cu_obj_mpcuda(&ts_eff);
	ts_eff.set_from_cpu(pameshParams->ts_eff);
	pameshParams->tsi_eff.set_p_cu_obj_mpcuda(&tsi_eff);
	tsi_eff.set_from_cpu(pameshParams->tsi_eff);
	pameshParams->pump_eff.set_p_cu_obj_mpcuda(&pump_eff);
	pump_eff.set_from_cpu(pameshParams->pump_eff);
	pameshParams->cpump_eff.set_p_cu_obj_mpcuda(&cpump_eff);
	cpump_eff.set_from_cpu(pameshParams->cpump_eff);
	pameshParams->the_eff.set_p_cu_obj_mpcuda(&the_eff);
	the_eff.set_from_cpu(pameshParams->the_eff);
	
	base_temperature.from_cpu(pameshParams->base_temperature);

	pameshParams->Sc.set_p_cu_obj_mpcuda(&Sc);
	Sc.set_from_cpu(pameshParams->Sc);
	pameshParams->joule_eff.set_p_cu_obj_mpcuda(&joule_eff);
	joule_eff.set_from_cpu(pameshParams->joule_eff);

	pameshParams->thermCond.set_p_cu_obj_mpcuda(&thermCond);
	thermCond.set_from_cpu(pameshParams->thermCond);
	pameshParams->density.set_p_cu_obj_mpcuda(&density);
	density.set_from_cpu(pameshParams->density);
	
	pameshParams->shc.set_p_cu_obj_mpcuda(&shc);
	shc.set_from_cpu(pameshParams->shc);
	pameshParams->shc_e.set_p_cu_obj_mpcuda(&shc_e);
	shc_e.set_from_cpu(pameshParams->shc_e);
	pameshParams->G_e.set_p_cu_obj_mpcuda(&G_e);
	G_e.set_from_cpu(pameshParams->G_e);
	
	pameshParams->cT.set_p_cu_obj_mpcuda(&cT);
	cT.set_from_cpu(pameshParams->cT);
	
	pameshParams->Q.set_p_cu_obj_mpcuda(&Q);
	Q.set_from_cpu(pameshParams->Q);
}

Atom_MeshParamsCUDA::~Atom_MeshParamsCUDA()
{
	//fine to access data in MeshParams here : Mesh inherits from MeshParams, so in the destruction process Mesh gets destroyed first. 
	//Mesh destructor then calls for MeshCUDA implementation to be destroyed, then we get here since MeshCUDA inherits from MeshParamsCUDA. After this we return back to Mesh destructor to continue destruction down the list.

	//-----------SIMPLE CUBIC

	pameshParams->grel.null_p_cu_obj_mpcuda();

	pameshParams->alpha.null_p_cu_obj_mpcuda();

	pameshParams->mu_s.null_p_cu_obj_mpcuda();

	pameshParams->J.null_p_cu_obj_mpcuda();
	
	pameshParams->D.null_p_cu_obj_mpcuda();
	pameshParams->D_dir.null_p_cu_obj_mpcuda();

	pameshParams->Js.null_p_cu_obj_mpcuda();
	pameshParams->Js2.null_p_cu_obj_mpcuda();

	pameshParams->K1.null_p_cu_obj_mpcuda();
	pameshParams->K2.null_p_cu_obj_mpcuda();
	pameshParams->K3.null_p_cu_obj_mpcuda();

	pameshParams->mcanis_ea1.null_p_cu_obj_mpcuda();
	pameshParams->mcanis_ea2.null_p_cu_obj_mpcuda();
	pameshParams->mcanis_ea3.null_p_cu_obj_mpcuda();

	//-----------Others

	pameshParams->Nxy.null_p_cu_obj_mpcuda();

	pameshParams->cHA.null_p_cu_obj_mpcuda();
	pameshParams->cHmo.null_p_cu_obj_mpcuda();

	pameshParams->s_eff.null_p_cu_obj_mpcuda();

	pameshParams->elecCond.null_p_cu_obj_mpcuda();
	pameshParams->amrPercentage.null_p_cu_obj_mpcuda();
	pameshParams->tamrPercentage.null_p_cu_obj_mpcuda();
	pameshParams->RAtmr_p.null_p_cu_obj_mpcuda();
	pameshParams->RAtmr_ap.null_p_cu_obj_mpcuda();

	pameshParams->P.null_p_cu_obj_mpcuda();
	pameshParams->beta.null_p_cu_obj_mpcuda();

	pameshParams->De.null_p_cu_obj_mpcuda();
	pameshParams->n_density.null_p_cu_obj_mpcuda();
	pameshParams->betaD.null_p_cu_obj_mpcuda();

	pameshParams->SHA.null_p_cu_obj_mpcuda();
	pameshParams->flSOT.null_p_cu_obj_mpcuda();
	pameshParams->flSOT2.null_p_cu_obj_mpcuda();

	pameshParams->STq.null_p_cu_obj_mpcuda();
	pameshParams->STq2.null_p_cu_obj_mpcuda();
	pameshParams->STa.null_p_cu_obj_mpcuda();
	pameshParams->STa2.null_p_cu_obj_mpcuda();
	pameshParams->STp.null_p_cu_obj_mpcuda();

	pameshParams->l_sf.null_p_cu_obj_mpcuda();
	pameshParams->l_ex.null_p_cu_obj_mpcuda();
	pameshParams->l_ph.null_p_cu_obj_mpcuda();

	pameshParams->Gi.null_p_cu_obj_mpcuda();
	pameshParams->Gmix.null_p_cu_obj_mpcuda();

	pameshParams->ts_eff.null_p_cu_obj_mpcuda();
	pameshParams->tsi_eff.null_p_cu_obj_mpcuda();

	pameshParams->pump_eff.null_p_cu_obj_mpcuda();
	pameshParams->cpump_eff.null_p_cu_obj_mpcuda();
	pameshParams->the_eff.null_p_cu_obj_mpcuda();

	pameshParams->Sc.null_p_cu_obj_mpcuda();
	pameshParams->joule_eff.null_p_cu_obj_mpcuda();

	pameshParams->thermCond.null_p_cu_obj_mpcuda();
	pameshParams->density.null_p_cu_obj_mpcuda();

	pameshParams->shc.null_p_cu_obj_mpcuda();
	pameshParams->shc_e.null_p_cu_obj_mpcuda();
	pameshParams->G_e.null_p_cu_obj_mpcuda();

	pameshParams->cT.null_p_cu_obj_mpcuda();
	
	pameshParams->Q.null_p_cu_obj_mpcuda();
}

#endif