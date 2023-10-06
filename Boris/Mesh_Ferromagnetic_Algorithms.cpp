#include "stdafx.h"
#include "Mesh_Ferromagnetic.h"

#ifdef MESH_COMPILATION_FERROMAGNETIC

#include "SuperMesh.h"
#include "MeshParamsControl.h"
#include "Atom_MeshParamsControl.h"

//----------------------------------- IMPORTANT CONTROL METHODS

//called at the start of each iteration
void FMesh::PrepareNewIteration(void) 
{ 
	if (Is_Dormant()) Track_Shift_Algorithm();
	else {

		if (!IsModuleSet(MOD_ZEEMAN)) Heff.set(DBL3(0));
	}
}

#if COMPILECUDA == 1
void FMesh::PrepareNewIterationCUDA(void) 
{ 
	if (Is_Dormant()) Track_Shift_Algorithm();
	else {

		if (pMeshCUDA && !IsModuleSet(MOD_ZEEMAN)) pMeshCUDA->Heff.set(cuReal3());
	}
}
#endif

//----------------------------------- ALGORITHMS

//setup track shifting algoithm for the holder mesh, with simulation window mesh, to be moved at set velocity and clipping during a simulation
BError FMesh::Setup_Track_Shifting(std::vector<int> sim_meshIds, DBL3 velocity, DBL3 clipping)
{
	BError error(__FUNCTION__);

	trackWindow_velocity = velocity;
	trackWindow_shift_clip = clipping;

	//if velocity is zero then turn off track shifting algorithm and write magnetization from sim_meshId back to this holder mesh
	if (velocity == DBL3()) {

		trackWindow_shift_debt = DBL3();
		trackWindow_last_time = 0.0;

		for (int sidx = 0; sidx < sim_meshIds.size(); sidx++) {

			int tidx = search_vector(idTrackShiftMesh, sim_meshIds[sidx]);
			if (tidx >= 0) {

				idTrackShiftMesh.erase(idTrackShiftMesh.begin() + tidx);

				int midx = pSMesh->contains_id(sim_meshIds[sidx]);
				if (midx < 0) continue;

				Rect rect_intersection = (*pSMesh)[midx]->meshRect.get_intersection(meshRect);
				if (!rect_intersection.IsNull()) {

					//write magnetization from simulation mesh to holder mesh in the overlap region
					error = copy_mesh_data(*(*pSMesh)[midx], rect_intersection - meshRect.s, rect_intersection - (*pSMesh)[midx]->meshRect.s);
				}
			}
		}
	}
	else {

		idTrackShiftMesh.clear();

		for (int sidx = 0; sidx < sim_meshIds.size(); sidx++) {

			int sim_meshId = sim_meshIds[sidx];

			int midx = pSMesh->contains_id(sim_meshId);
			if (midx >= 0) {

				//store simulation meshId if not already stored and if so also copy magnetization to simulation mesh and empy holder mesh in overlap region
				if (!vector_contains(idTrackShiftMesh, sim_meshId)) {

					trackWindow_shift_debt = DBL3();
					trackWindow_last_time = 0.0;

					idTrackShiftMesh.push_back(sim_meshId);

					Rect rect_intersection = (*pSMesh)[midx]->meshRect.get_intersection(meshRect);

					if (!rect_intersection.IsNull()) {

						error = (*pSMesh)[midx]->copy_mesh_data(*this, rect_intersection - (*pSMesh)[midx]->meshRect.s, rect_intersection - meshRect.s);
						error = delrect(rect_intersection - meshRect.s);
					}
				}
			}
		}
	}

	return error;
}

//implement track shifting - called during PrepareNewIteration if this is a dormant mesh with track shifting configured (non-zero trackWindow_velocity and idTrackShiftMesh vector not empty)
void FMesh::Track_Shift_Algorithm(void)
{
	if (!idTrackShiftMesh.size() || trackWindow_velocity == DBL3() || !pSMesh->CurrentTimeStepSolved()) return;

	//current time so we can calculate required shift
	double trackWindow_current_time = pSMesh->GetTime();

	//if current time less than stored previous time then something is wrong (e.g. ode was reset - reset shifting debt as well)
	if (trackWindow_current_time < trackWindow_last_time) {

		trackWindow_last_time = trackWindow_current_time;
		trackWindow_shift_debt = DBL3();
	}

	//add to total amount of shifting which hasn't yet been executed (the shift debt)
	trackWindow_shift_debt += (trackWindow_current_time - trackWindow_last_time) * trackWindow_velocity;

	//clip the shift to execute if required
	DBL3 shift = DBL3(
		trackWindow_shift_clip.x > 0.0 ? 0.0 : trackWindow_shift_debt.x,
		trackWindow_shift_clip.y > 0.0 ? 0.0 : trackWindow_shift_debt.y,
		trackWindow_shift_clip.z > 0.0 ? 0.0 : trackWindow_shift_debt.z);

	if (trackWindow_shift_clip.x > 0.0 && fabs(trackWindow_shift_debt.x) > trackWindow_shift_clip.x)
		shift.x = floor(fabs(trackWindow_shift_debt.x) / trackWindow_shift_clip.x) * trackWindow_shift_clip.x * get_sign(trackWindow_shift_debt.x);

	if (trackWindow_shift_clip.y > 0.0 && fabs(trackWindow_shift_debt.y) > trackWindow_shift_clip.y)
		shift.y = floor(fabs(trackWindow_shift_debt.y) / trackWindow_shift_clip.y) * trackWindow_shift_clip.y * get_sign(trackWindow_shift_debt.y);

	if (trackWindow_shift_clip.z > 0.0 && fabs(trackWindow_shift_debt.z) > trackWindow_shift_clip.z)
		shift.z = floor(fabs(trackWindow_shift_debt.z) / trackWindow_shift_clip.z) * trackWindow_shift_clip.z * get_sign(trackWindow_shift_debt.z);

	//execute shift if needed
	if (shift != DBL3()) {

		for (int tidx = 0; tidx < idTrackShiftMesh.size(); tidx++) {

			//get simulation window mesh index using idTrackShiftMesh stored meshId
			int midx = pSMesh->contains_id(idTrackShiftMesh[tidx]);
			if (midx < 0) continue;

			Rect dstRect, srcRect;

			if (shift.x > 0) {

				dstRect = (*pSMesh)[midx]->meshRect - M.rect.s;
				dstRect.e.x = dstRect.s.x + shift.x;
				srcRect = dstRect + M.rect.s - (*pSMesh)[midx]->meshRect.s;
			}
			else {

				dstRect = (*pSMesh)[midx]->meshRect - M.rect.s;
				dstRect.s.x = dstRect.e.x + shift.x;
				srcRect = dstRect + M.rect.s - (*pSMesh)[midx]->meshRect.s;
			}

			//////////////////////////////////////////
			// FERROMAGNETIC WINDOW
			//////////////////////////////////////////

			if (!(*pSMesh)[midx]->is_atomistic()) {

				Mesh* pTrackShiftMesh = reinterpret_cast<Mesh*>((*pSMesh)[midx]);

				//1. copy magnetization data from simulation window mesh trailing end into this dormant mesh
#if COMPILECUDA == 1
				if (pMeshCUDA) {

					Box cells_box_dst = M.box_from_rect_max(dstRect + M.rect.s);
					Box cells_box_src = pTrackShiftMesh->M.box_from_rect_max(srcRect + (*pSMesh)[midx]->meshRect.s);
					pMeshCUDA->M.copy_values(pTrackShiftMesh->pMeshCUDA->M, cells_box_dst, cells_box_src, 1.0, false);
				}
				else M.copy_values(pTrackShiftMesh->M, dstRect, srcRect, 1.0, false);
#else
				M.copy_values(pTrackShiftMesh->M, dstRect, srcRect, 1.0, false);
#endif
				
				//2. perform magnetization data shift in simulation window mesh
				if (pTrackShiftMesh->M.linear_size()) {

#if COMPILECUDA == 1
					if (pMeshCUDA) {

						pTrackShiftMesh->pMeshCUDA->M.shift_x(-shift.x, (*pSMesh)[midx]->meshRect, false);
					}
					else pTrackShiftMesh->M.shift_x(-shift.x, (*pSMesh)[midx]->meshRect, false);
#else
					pTrackShiftMesh->M.shift_x(-shift.x, (*pSMesh)[midx]->meshRect, false);
#endif
				}
			}

			//////////////////////////////////////////
			// ATOMISTIC WINDOW
			//////////////////////////////////////////

			else {

				Atom_Mesh* paTrackShiftMesh = reinterpret_cast<Atom_Mesh*>((*pSMesh)[midx]);

				//used to convert moment to magnetization in each atomistic unit cell
				double conversion = MUB / paTrackShiftMesh->M1.h.dim();

				//1. copy magnetization data from simulation window mesh trailing end into this dormant mesh
#if COMPILECUDA == 1
				if (pMeshCUDA) {

					Box cells_box_dst = M.box_from_rect_max(dstRect + M.rect.s);
					Box cells_box_src = paTrackShiftMesh->M1.box_from_rect_max(srcRect + (*pSMesh)[midx]->meshRect.s);
					pMeshCUDA->M.copy_values(paTrackShiftMesh->paMeshCUDA->M1, cells_box_dst, cells_box_src, conversion, false);
				}
				else M.copy_values(paTrackShiftMesh->M1, dstRect, srcRect, conversion, false);
#else
				M.copy_values(paTrackShiftMesh->M1, dstRect, srcRect, conversion, false);
#endif

				//2. perform magnetization data shift in simulation window mesh
				if (paTrackShiftMesh->M1.linear_size()) {

#if COMPILECUDA == 1
					if (pMeshCUDA) {

						paTrackShiftMesh->paMeshCUDA->M1.shift_x(-shift.x, (*pSMesh)[midx]->meshRect, false);
					}
					else paTrackShiftMesh->M1.shift_x(-shift.x, (*pSMesh)[midx]->meshRect, false);
#else
					paTrackShiftMesh->M1.shift_x(-shift.x, (*pSMesh)[midx]->meshRect, false);
#endif
				}
			}
		}

		//3. shift dormant mesh rectangle (in the opposite direction of velocity vector, since it's the simulation window mesh which is shifting relative to dormant track mesh)
		Shift_Mesh_Rectangle(-1 * shift);

		for (int tidx = 0; tidx < idTrackShiftMesh.size(); tidx++) {

			//get simulation window mesh index using idTrackShiftMesh stored meshId
			int midx = pSMesh->contains_id(idTrackShiftMesh[tidx]);
			if (midx < 0) continue;

			Rect dstRect, srcRect;

			if (shift.x > 0) {

				dstRect = (*pSMesh)[midx]->meshRect - M.rect.s;
				dstRect.e.x = dstRect.s.x + shift.x;
				srcRect = dstRect + M.rect.s - (*pSMesh)[midx]->meshRect.s;
			}
			else {

				dstRect = (*pSMesh)[midx]->meshRect - M.rect.s;
				dstRect.s.x = dstRect.e.x + shift.x;
				srcRect = dstRect + M.rect.s - (*pSMesh)[midx]->meshRect.s;
			}

			//////////////////////////////////////////
			// FERROMAGNETIC WINDOW
			//////////////////////////////////////////

			if (!(*pSMesh)[midx]->is_atomistic()) {

				Mesh* pTrackShiftMesh = reinterpret_cast<Mesh*>((*pSMesh)[midx]);

				//4. copy magnetization data from this dormant mesh into simulation window leading end
				if (shift.x > 0) {

					dstRect = (*pSMesh)[midx]->meshRect - (*pSMesh)[midx]->meshRect.s;
					dstRect.s.x = dstRect.e.x - shift.x;
					srcRect = dstRect + (*pSMesh)[midx]->meshRect.s - M.rect.s;
				}
				else {

					dstRect = (*pSMesh)[midx]->meshRect - (*pSMesh)[midx]->meshRect.s;
					dstRect.e.x = dstRect.s.x - shift.x;
					srcRect = dstRect + (*pSMesh)[midx]->meshRect.s - M.rect.s;
				}

#if COMPILECUDA == 1
				if (pMeshCUDA) {

					Box cells_box_dst = pTrackShiftMesh->M.box_from_rect_max(dstRect + (*pSMesh)[midx]->meshRect.s);
					Box cells_box_src = M.box_from_rect_max(srcRect + M.rect.s);
					pTrackShiftMesh->pMeshCUDA->M.copy_values(pMeshCUDA->M, cells_box_dst, cells_box_src, 1.0, false);
					pMeshCUDA->M.delrect(srcRect, false);
				}
				else {

					pTrackShiftMesh->M.copy_values(M, dstRect, srcRect, 1.0, false);
					M.delrect(srcRect, false);
				}
#else
				pTrackShiftMesh->M.copy_values(M, dstRect, srcRect, 1.0, false);
				M.delrect(srcRect, false);
#endif

				//5a. recalculate flags in simulation mesh
#if COMPILECUDA == 1
				if (pMeshCUDA) {

					pTrackShiftMesh->pMeshCUDA->M.set_ngbrFlags_shapeonly();

					//mesh shape has changed, so when restarting simulation must re-synchronize shapes between GPU and CPU versions
					pTrackShiftMesh->Set_Shape_Synchronization_Lost();
				}
				else pTrackShiftMesh->M.set_ngbrFlags_shapeonly();
#else
				pTrackShiftMesh->M.set_ngbrFlags_shapeonly();
#endif
			}

			//////////////////////////////////////////
			// ATOMISTIC WINDOW
			//////////////////////////////////////////

			else {

				Atom_Mesh* paTrackShiftMesh = reinterpret_cast<Atom_Mesh*>((*pSMesh)[midx]);

				//4. copy magnetization data from this dormant mesh into simulation window leading end

				double Ms0 = paTrackShiftMesh->Show_Ms();
				std::function<DBL3(DBL3, int, int)> thermalize_func = [&](DBL3 Mval, int idx_src, int idx_dst) -> DBL3 {

					double Mval_norm = Mval.norm();
					if (IsZ(Mval_norm)) return DBL3();

					DBL3 Mval_dir = Mval / Mval_norm;

					//need the normalized average over spins in this cell to be mz (the resultant Ms0 value in this atomistic mesh should match that in the micromagnetic mesh)
					double mz = Mval_norm / Ms0;
					//enforce min and max values possible
					if (mz < 1.0 / 6) mz = 1.0 / 6;
					if (mz > 1.0) mz = 1.0;
					//maximum polar angle around Mval direction
					double theta_max = sqrt(10 - sqrt(100 + 120 * (mz - 1)));
					if (theta_max > PI) theta_max = PI;

					double theta = prng.rand() * theta_max;
					double phi = prng.rand() * 2 * PI;

					double mu_s_val = paTrackShiftMesh->mu_s;
					paTrackShiftMesh->update_parameters_mcoarse(idx_dst, paTrackShiftMesh->mu_s, mu_s_val);

					DBL3 spin_val = mu_s_val * relrotate_polar(Mval_dir, theta, phi);

					return spin_val;
				};

				if (shift.x > 0) {

					dstRect = (*pSMesh)[midx]->meshRect - (*pSMesh)[midx]->meshRect.s;
					dstRect.s.x = dstRect.e.x - shift.x;
					srcRect = dstRect + (*pSMesh)[midx]->meshRect.s - M.rect.s;
				}
				else {

					dstRect = (*pSMesh)[midx]->meshRect - (*pSMesh)[midx]->meshRect.s;
					dstRect.e.x = dstRect.s.x - shift.x;
					srcRect = dstRect + (*pSMesh)[midx]->meshRect.s - M.rect.s;
				}

#if COMPILECUDA == 1
				if (pMeshCUDA) {

					Box cells_box_dst = paTrackShiftMesh->M1.box_from_rect_max(dstRect + (*pSMesh)[midx]->meshRect.s);
					Box cells_box_src = M.box_from_rect_max(srcRect + M.rect.s);
					paTrackShiftMesh->paMeshCUDA->Track_Shift_Algorithm_CopyThermalize(pMeshCUDA->M, cells_box_dst, cells_box_src);
					pMeshCUDA->M.delrect(srcRect, false);
				}
				else {

					paTrackShiftMesh->M1.copy_values_thermalize(M, thermalize_func, dstRect, srcRect, false);
					M.delrect(srcRect, false);
				}
#else
				paTrackShiftMesh->M1.copy_values_thermalize(M, thermalize_func, dstRect, srcRect, false);
				M.delrect(srcRect, false);
#endif

				//5a. recalculate flags in simulation mesh
#if COMPILECUDA == 1
				if (pMeshCUDA) {

					paTrackShiftMesh->paMeshCUDA->M1.set_ngbrFlags_shapeonly();

					//mesh shape has changed, so when restarting simulation must re-synchronize shapes between GPU and CPU versions
					paTrackShiftMesh->Set_Shape_Synchronization_Lost();
				}
				else paTrackShiftMesh->M1.set_ngbrFlags_shapeonly();
#else
				paTrackShiftMesh->M1.set_ngbrFlags_shapeonly();
#endif
			}
		}
		
		//5b. recalculate flags in dormant mesh
#if COMPILECUDA == 1
		if (pMeshCUDA) {

			pMeshCUDA->M.set_ngbrFlags_shapeonly();

			//mesh shape has changed, so when restarting simulation must re-synchronize shapes between GPU and CPU versions
			Set_Shape_Synchronization_Lost();
		}
		else M.set_ngbrFlags_shapeonly();
#else
		M.set_ngbrFlags_shapeonly();
#endif
		
		//6. Update demag field in dormant mesh if demag module enabled
		if (IsModuleSet(MOD_DEMAG)) {

#if COMPILECUDA == 1
			if (pMeshCUDA) {

				pMeshCUDA->Heff.set(cuReal3(0));
				pMod[INT2(MOD_DEMAG, 0)]->UpdateFieldCUDA();
			}
			else {

				Heff.set(DBL3(0));
				pMod[INT2(MOD_DEMAG, 0)]->UpdateField();
			}
#else
			//update field
			Heff.set(DBL3(0));
			pMod[INT2(MOD_DEMAG, 0)]->UpdateField();
#endif

			//now extract from Heff the stray field and store it in simulation meshes using their Zeeman module Havec
			for (int tidx = 0; tidx < idTrackShiftMesh.size(); tidx++) {

				//get simulation window mesh index using idTrackShiftMesh stored meshId
				int midx = pSMesh->contains_id(idTrackShiftMesh[tidx]);
				if (midx < 0) continue;

				//////////////////////////////////////////
				// FERROMAGNETIC WINDOW
				//////////////////////////////////////////

				if (!(*pSMesh)[midx]->is_atomistic()) {

					Mesh* pTrackShiftMesh = reinterpret_cast<Mesh*>((*pSMesh)[midx]);

					if (pTrackShiftMesh->IsModuleSet(MOD_ZEEMAN)) {
#if COMPILECUDA == 1
						if (pMeshCUDA) {

							reinterpret_cast<Zeeman*>(pTrackShiftMesh->GetModule(MOD_ZEEMAN))->SetFieldVEC_FromVEC_CUDA(pMeshCUDA->Heff);
						}
						else reinterpret_cast<Zeeman*>(pTrackShiftMesh->GetModule(MOD_ZEEMAN))->SetFieldVEC_FromVEC(Heff);
#else
						reinterpret_cast<Zeeman*>(pTrackShiftMesh->GetModule(MOD_ZEEMAN))->SetFieldVEC_FromVEC(Heff);
#endif
					}
				}

				//////////////////////////////////////////
				// ATOMISTIC WINDOW
				//////////////////////////////////////////

				else {

					Atom_Mesh* paTrackShiftMesh = reinterpret_cast<Atom_Mesh*>((*pSMesh)[midx]);

					if (paTrackShiftMesh->IsModuleSet(MOD_ZEEMAN)) {
#if COMPILECUDA == 1
						if (pMeshCUDA) {

							reinterpret_cast<Atom_Zeeman*>(paTrackShiftMesh->GetModule(MOD_ZEEMAN))->SetFieldVEC_FromVEC_CUDA(pMeshCUDA->Heff);
						}
						else reinterpret_cast<Atom_Zeeman*>(paTrackShiftMesh->GetModule(MOD_ZEEMAN))->SetFieldVEC_FromVEC(Heff);
#else
						reinterpret_cast<Atom_Zeeman*>(paTrackShiftMesh->GetModule(MOD_ZEEMAN))->SetFieldVEC_FromVEC(Heff);
#endif
					}
				}
			}
		}

		trackWindow_shift_debt -= shift;
	}

	trackWindow_last_time = trackWindow_current_time;
}

#endif