#include "stdafx.h"
#include "PhysQRep.h"

#if GRAPHICS == 1

#include "BorisGraphics.h"

//calculate settings for default view (these are the settings set when the mesh focus changes)
PhysQRepSettings::PhysQRepSettings(std::vector<PhysQ>& physQ, UINT wndWidth, D2D1_RECT_F spaceRect)
{
	//quantities being auto-adjusted are:
	//1. focusRect
	//2. m_to_l
	//3. detail_level

	//4. Eye position (camX, camY, camZ) - use D3D::SetCameraPosition
	//5. Up position (camUx, camUy, camUz) - use D3D::SetCameraUp
	//6. View shift (view_shiftX, view_shiftY, view_shiftZ) - use Set3DOriginPixelPosition
	//7. FOV

	D2D1_SIZE_F spaceRectDims;

	spaceRectDims.width = spaceRect.right - spaceRect.left;
	spaceRectDims.height = spaceRect.bottom - spaceRect.top;

	//1. rectangle of mesh in focus - the 3D coordinate system origin is placed at the center of this rect (when converted to logical units)

	//get rectangle of focused mesh
	for (int idx = 0; idx < physQ.size(); idx++) {

		if (physQ[idx].is_focused()) {

			focusRect = physQ[idx].rectangle();
			break;
		}

		//must have at least one mesh in focus
		if (idx == physQ.size() - 1) return;
	}

	//2. conversion factor from meters to logical units : multiply mesh dimensions by this to obtain logical dimensions
	
	//units to fit along viewing width and height (remember DISPLAYUNITS is the number of units that fit along screen width)
	float units_x = (DISPLAYUNITS * spaceRectDims.width / wndWidth);
	float units_y = (DISPLAYUNITS * spaceRectDims.height / wndWidth);
	if (units_x < 1) units_x = 1;
	if (units_y < 1) units_y = 1;

	//calculate conversion factor from mesh length (metric) to logical units, such that the longest mesh rectangle fits exactly within the viewing rectangle
	double m_to_l_x = units_x / focusRect.length();
	double m_to_l_y = units_y / focusRect.width();

	//Let c be the conversion factor, W and H the mesh width and height, uw and uh the logical units in the viewing rectangle along width and height.
	//Then c * W <= uw and c * H <= uh must both be satisfied. We adjust this slightly to make a looser fit (multiply by UNITSSCALEBACK < 1)
	m_to_l = (m_to_l_x < m_to_l_y ? m_to_l_x : m_to_l_y) * UNITSSCALEBACK;

	//3. level of displayed detail(sets the number of display elements per logical unit)
	
	//set default detail level in order to fit a maximum number of cells along maximum mesh dimensions
	detail_level = focusRect.maxDimension() / DETAILDEFAULTCELLS;

	//4. Eye position (camX, camY, camZ) - use D3D::SetCameraPosition
	
	camX = 0;
	camY = 0;
	camZ = CAMERADISTANCE;

	//5. Up position (camUx, camUy, camUz) - use D3D::SetCameraUp
	
	camUx = 0.0;
	camUy = 1.0;
	camUz = 0.0;

	//6. View shift (view_shiftX, view_shiftY, view_shiftZ) - use D3D::Set3DOriginPixelPosition
	//this shift sets the logical units coordinate system center in the middle of the spaceRect
	//This is because Set3DOriginPixelPosition further adds -screenwidth/2 shift to x and +screenheight/2 shift to y 
	view_shiftX = spaceRectDims.width / 2 + spaceRect.left;
	view_shiftY = spaceRectDims.height / 2 + spaceRect.top;
	view_shiftZ = 0;

	//7. FOV

	fov = (2 * atan(units_x*wndWidth / (2 * camZ*spaceRectDims.width))) * 180 / XM_PI;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//get current display settings
PhysQRepSettings PhysQRep::get_current_settings(void)
{
	return PhysQRepSettings(focusRect, m_to_l, detail_level, pBG->GetCameraPosition(), pBG->GetCameraUp(), pBG->GetViewShift(), pBG->GetCameraFOV());
}

//From physQ calculate physQRep, and auto-set display to fit the mesh in focus on screen at default camera settings
void PhysQRep::CalculateRepresentation_AutoSettings(std::vector<PhysQ> physQ, D2D1_RECT_F spaceRect)
{
	PhysQRepSettings settings(physQ, pBG->wndWidth, spaceRect);
	CalculateRepresentation_NewSettings(physQ, settings);
}

void PhysQRep::CalculateRepresentation_NewSettings(std::vector<PhysQ> physQ, PhysQRepSettings newSettings)
{
	//set PhysQRep settings
	focusRect = newSettings.focusRect;
	m_to_l = newSettings.m_to_l;
	detail_level = newSettings.detail_level;

	highlightCell = false;

	//set camera settings
	pBG->SetCameraPosition(newSettings.camX, newSettings.camY, newSettings.camZ);
	pBG->SetCameraUp(newSettings.camUx, newSettings.camUy, newSettings.camUz);
	pBG->SetCameraFOV(newSettings.fov);
	pBG->Set3DOriginPixelPosition(newSettings.view_shiftX, newSettings.view_shiftY, newSettings.view_shiftZ);

	//now do the actual calculations : calculate transformBatch for each mesh
	CalculateRepresentation(physQ);
}

//Calculate physical quantity representation were display option settings have already been set (either by default or through the CalculateRepresentation_AutoSettings method above)
void PhysQRep::CalculateRepresentation(std::vector<PhysQ> physQ)
{
	//physQ and physQRep have to have same dimensions
	if (physQRep.size() != physQ.size())
		physQRep.resize(physQ.size());

	meshName_focused = "none";
	typeName_focused = "";
	unit = "";
	minmax_focusedmeshValues = DBL2();

	int focused_mesh_index = 0;

	//find maximum number of displayed types (actually find value of maximum type)
	unsigned maxtypeval = 0;
	for (int idx = 0; idx < (int)physQ.size(); idx++) {

		unsigned type = physQ[idx].get_type();
		maxtypeval = (maxtypeval > type ? maxtypeval : type);

		if (physQ[idx].is_focused()) {

			focused_mesh_index = idx;
			unit = physQ[idx].get_unit();
			meshName_focused = physQ[idx].get_name();
			typeName_focused = physQ[idx].get_type_name();
		}

		//unit of this PhysQRep component
		physQRep[idx].unit = physQ[idx].get_unit();
		//the name of displayed quantity for this PhysQRep component
		physQRep[idx].typeName = physQ[idx].get_type_name();
		//the mesh name holding the displayed quantity for this PhysQRep component
		physQRep[idx].meshName = physQ[idx].get_name();
		//the type of representation to use for vectorial quantites
		physQRep[idx].vec3rep = physQ[idx].get_vec3rep();
	}
	
	std::vector<double> minimum, maximum, maximum2;
	std::vector<bool>  min_not_set, max_not_set, max2_not_set;

	minimum.assign(maxtypeval + 1, 0.0);
	maximum.assign(maxtypeval + 1, 0.0);
	min_not_set.assign(maxtypeval + 1, true);
	max_not_set.assign(maxtypeval + 1, true);

	//calculate all representations and find minimum, maximum for each type of physical quantity
	for (int idx = 0; idx < (int)physQ.size(); idx++) {
		
		//set mesh frame - do this irrespective of mesh display being empty or not
		Rect meshRect = physQ[idx].rectangle();
		DBL3 frameScaling = meshRect.size() * m_to_l;
		physQRep[idx].meshFrame.Scale = XMMatrixIdentity() * XMMatrixScaling(frameScaling.x, frameScaling.y, frameScaling.z);
		DBL3 frameTranslation = (meshRect.s - focusRect.s) * m_to_l * 2 - (focusRect.size() - meshRect.size()) * m_to_l;
		physQRep[idx].meshFrame.Trans = XMMatrixIdentity() * XMMatrixTranslation(frameTranslation.x, frameTranslation.y, frameTranslation.z);

		if (physQ[idx].is_empty()) {

			physQRep[idx].transformBatch.clear();
			physQRep[idx].transformBatch_render.clear();
			continue;
		}

		DBL2 minmax;

		//Separate computations for vectorial and scalar quantities
		if (physQ[idx].is_vectorial()) {
			
			//vectorial
			if (physQ[idx].is_vec_vc()) {

				//VEC_VC. Single or double precision used?
				if (physQ[idx].is_double_precision()) {
					
					if (physQ[idx].is_dual()) minmax = CalculateRepresentation_VEC(physQ[idx].get_vec_vc_dbl3(), physQ[idx].get2_vec_vc_dbl3(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds(), physQ[idx].get_thresholdtrigger());
					else minmax = CalculateRepresentation_VEC(physQ[idx].get_vec_vc_dbl3(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds(), physQ[idx].get_thresholdtrigger());
				}
				else {

					if (physQ[idx].is_dual()) minmax = CalculateRepresentation_VEC(physQ[idx].get_vec_vc_flt3(), physQ[idx].get2_vec_vc_flt3(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds(), physQ[idx].get_thresholdtrigger());
					else minmax = CalculateRepresentation_VEC(physQ[idx].get_vec_vc_flt3(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds(), physQ[idx].get_thresholdtrigger());
				}
			}
			else {

				//just a VEC
				if (physQ[idx].is_double_precision()) {

					if (physQ[idx].is_dual()) minmax = CalculateRepresentation_VEC(physQ[idx].get_vec_dbl3(), physQ[idx].get2_vec_dbl3(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds(), physQ[idx].get_thresholdtrigger());
					else minmax = CalculateRepresentation_VEC(physQ[idx].get_vec_dbl3(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds(), physQ[idx].get_thresholdtrigger());
				}
				else {

					if (physQ[idx].is_dual()) minmax = CalculateRepresentation_VEC(physQ[idx].get_vec_flt3(), physQ[idx].get2_vec_flt3(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds(), physQ[idx].get_thresholdtrigger());
					else minmax = CalculateRepresentation_VEC(physQ[idx].get_vec_flt3(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds(), physQ[idx].get_thresholdtrigger());
				}
			}
		}
		else {

			//scalar
			if (physQ[idx].is_vec_vc()) {

				//VEC_VC. Single or double precision used?
				if (physQ[idx].is_double_precision()) {

					minmax = CalculateRepresentation_SCA(physQ[idx].get_vec_vc_double(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds());
				}
				else {

					minmax = CalculateRepresentation_SCA(physQ[idx].get_vec_vc_float(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds());
				}
			}
			else {

				//just a VEC
				if (physQ[idx].is_double_precision()) {

					minmax = CalculateRepresentation_SCA(physQ[idx].get_vec_double(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds());
				}
				else {

					minmax = CalculateRepresentation_SCA(physQ[idx].get_vec_float(), physQRep[idx], physQ[idx].get_transparency(), physQ[idx].get_thresholds());
				}
			}
		}

		if (idx == focused_mesh_index) minmax_focusedmeshValues = minmax;

		unsigned type = physQ[idx].get_type();

		if (min_not_set[type]) { minimum[type] = minmax.i; min_not_set[type] = false; }
		else minimum[type] = (minimum[type] < minmax.i ? minimum[type] : minmax.i);

		if (max_not_set[type]) { maximum[type] = minmax.j; max_not_set[type] = false; }
		else maximum[type] = (maximum[type] > minmax.j ? maximum[type] : minmax.j);
	}
	
	//now that we have minimum and maximum values for each type adjust physical representations accordingly
	for (int idx = 0; idx < (int)physQ.size(); idx++) {

		//do we need to apply relative scaling to arrows ? (relative to largest magnitude)
		if (physQRep[idx].scale_to_magnitude && physQRep[idx].transformBatch.linear_size()) {

			unsigned type = physQ[idx].get_type();

			if (physQ[idx].is_vectorial()) {

				AdjustMagnitude_VEC(physQRep[idx], DBL2(minimum[type], maximum[type]));
			}
			else {

				AdjustMagnitude_SCA(physQRep[idx], DBL2(minimum[type], maximum[type]));
			}
		}
		
		//are we going to use transformBatch_render instead of transformBatch for rendering? If so copy elements over from transformBatch to transformBatch_render
		//Note the calculation routines determine which one to use, and clears transformBatch_render if not needed, or sets required size for it
		if (physQRep[idx].transformBatch_render.size()) {

			int num_rendered_cells = 0;
			for (int cell_idx = 0; cell_idx < physQRep[idx].transformBatch.linear_size(); cell_idx++) {

				if (!physQRep[idx].transformBatch[cell_idx].skip_render)
					physQRep[idx].transformBatch_render[num_rendered_cells++] = physQRep[idx].transformBatch[cell_idx];
			}
		}
	}
}

//------------------------------------------------------------------------------ VECTOR QUANTITIES ------------------------------------------------------------------------------//

//////////////////// SINGLE VEC DISPLAY

template DBL2 PhysQRep::CalculateRepresentation_VEC<VEC<DBL3>>(VEC<DBL3>* pQ, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds, int displayThresholdTrigger);
template DBL2 PhysQRep::CalculateRepresentation_VEC<VEC_VC<DBL3>>(VEC_VC<DBL3>* pQ, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds, int displayThresholdTrigger);
template DBL2 PhysQRep::CalculateRepresentation_VEC<VEC<FLT3>>(VEC<FLT3>* pQ, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds, int displayThresholdTrigger);
template DBL2 PhysQRep::CalculateRepresentation_VEC<VEC_VC<FLT3>>(VEC_VC<FLT3>* pQ, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds, int displayThresholdTrigger);

template <typename VECType>
DBL2 PhysQRep::CalculateRepresentation_VEC(VECType* pQ, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds, int displayThresholdTrigger)
{	
	Rect meshRect = pQ->rect;
	DBL3 h = pQ->h;
	INT3 n = pQ->n;

	//display cell size - this sets the level of displayed detail
	DBL3 hdisp(detail_level);

	//number of cells for display, and adjust display cell size (must have an integer number of cells to display)
	INT3 ndisp = round(meshRect / hdisp);

	if (ndisp.x < 1) ndisp.x = 1;
	if (ndisp.y < 1) ndisp.y = 1;
	if (ndisp.z < 1) ndisp.z = 1;
	if (ndisp.x > n.x)  ndisp.x = n.x;
	if (ndisp.y > n.y)  ndisp.y = n.y;
	if (ndisp.z > n.z)  ndisp.z = n.z;
	hdisp = meshRect / ndisp;

	physQRepComponent.transformBatch.resize(hdisp, meshRect);
	if(physQRepComponent.emptyCell.size() != ndisp.dim()) physQRepComponent.emptyCell.assign(physQRepComponent.transformBatch.linear_size(), false);

	if (physQRepComponent.vec3rep == VEC3REP_FULL) {

		if (renderspeedup1_cells && ndisp.dim() >= renderspeedup1_cells) physQRepComponent.obSelector = CDO_CONE;
		else physQRepComponent.obSelector = CDO_ARROW;
	}
	else {

		//if not a full representation then we'll be representing a scalar-type quantity extracted from the VEC
		physQRepComponent.obSelector = CDO_CUBE;
	}

	//Vector quantity - we need to compute rotations, scaling and color coding
	
	omp_reduction.new_minmax_reduction();

	int num_rendered_cells = 0;

	for (int k = 0; k < ndisp.z; k++) {
#pragma omp parallel for reduction(+:num_rendered_cells)
		for (int j = 0; j < ndisp.y; j++) {
			for (int i = 0; i < ndisp.x; i++) {

				DBL3 value;

				DBL3 rel_pos = hdisp & (DBL3(i, j, k) + DBL3(0.5));

				int idx = i + j * ndisp.x + k * ndisp.x*ndisp.y;

				physQRepComponent.transformBatch[idx].skip_render = false;

				if (pQ->is_empty(rel_pos)) {

					//most cells are not empty, so the check above should be kept since it's fast
					//there is the possibility that for a coarse display cell the check above returns empty but actually the coarse cell is not empty
					//thus use the complete check here with the coarse cell rectangle
					Rect cellRect = Rect(DBL3(i, j, k) & hdisp, DBL3(i + 1, j + 1, k + 1) & hdisp);

					if (pQ->is_empty(cellRect)) {

						//set this to empty
						physQRepComponent.transformBatch[idx] = CBObjectTransform();
						physQRepComponent.transformBatch[idx].skip_render = true;
						physQRepComponent.emptyCell[idx] = true;
						continue;
					}
					else {

						value = (DBL3)(*pQ).average(cellRect);
						physQRepComponent.emptyCell[idx] = false;
					}
				}
				else {

					value = (DBL3)(*pQ).cell_sum(rel_pos);
					physQRepComponent.emptyCell[idx] = false;
				}
							
				//the magnitude of VEC3 value
				double value_mag = (double)GetMagnitude(value);

				XMMATRIX Rotation = XMMatrixIdentity();

				DBL3 scaling = hdisp * m_to_l;
				XMMATRIX Scale = XMMatrixIdentity() * XMMatrixScaling(scaling.x, scaling.y, scaling.z);

				if (physQRepComponent.vec3rep == VEC3REP_FULL) {

					//Only need rotation for full representations
					if (IsNZ(value.x) || IsNZ(value.y)) Rotation = XMMatrixRotationAxis(XMLoadFloat3(&XMFLOAT3((float)-value.y, (float)value.x, 0)), (float)acos(value.z / value_mag));
					else {

						if (value.z > 0) Rotation = XMMatrixIdentity();
						else Rotation = XMMatrixRotationX(XM_PI);
					}
				}

				DBL3 translation = (DBL3(i + 0.5, j + 0.5, k + 0.5) & hdisp) * m_to_l * 2 + (meshRect.s - focusRect.s) * m_to_l * 2 - (focusRect.size() * m_to_l);
				XMMATRIX Translation = XMMatrixTranslation(translation.x, translation.y, translation.z);

				XMFLOAT4 Color;

				float alpha = minimum(get_alpha_value(translation), (float)maxTransparency);

				if (physQRepComponent.vec3rep == VEC3REP_FULL || physQRepComponent.vec3rep == VEC3REP_DIRECTION) {

					//use color wheel coding for full representations and direction-only representations

					//(R,G,B) channels. Linear transition between the 6 points defined below.
					//
					//+x : (1,0,0) : RED
					//+y : (1,1,0) : YELLOW
					//-x : (0,0,1) : BLUE
					//-y : (0,1,1) : CYAN
					//+z : (0,1,0) : GREEN
					//-z : (1,0,1) : MAGENTA

					//b is the normalized polar angle, ranges from 0 to 1 for +z and 1 to 2 for -z
					//a1, a2, a3, a4 are in-plane angles in the 4 quadrants: a1 measured from +x axis, a2 from +y axis, a3 from -x axis, a4 from -y axis

					if (value_mag > 0) {

						if (value.x > 0 && value.y >= 0) {

							float a1 = (float)asin(value.y / sqrt(value.x*value.x + value.y*value.y)) * 2 / XM_PI;
							float b = (float)acos(value.z / value_mag) * 2 / XM_PI;

							if (value.z >= 0) Color = XMFLOAT4(b, 1 - b + b * a1, 0.0f, alpha);
							else Color = XMFLOAT4(1, (2 - b)*a1, b - 1, alpha);
						}
						if (value.x <= 0 && value.y > 0) {

							float a2 = (float)asin(-value.x / sqrt(value.x*value.x + value.y*value.y)) * 2 / XM_PI;
							float b = (float)acos(value.z / value_mag) * 2 / XM_PI;

							if (value.z >= 0) Color = XMFLOAT4(b - a2 * b, 1 - a2 * b, a2*b, alpha);
							else Color = XMFLOAT4(1 - a2 * (2 - b), (2 - b)*(1 - a2), 1 - (2 - b)*(1 - a2), alpha);
						}
						if (value.x < 0 && value.y <= 0) {

							float a3 = (float)asin(-value.y / sqrt(value.x*value.x + value.y*value.y)) * 2 / XM_PI;
							float b = (float)acos(value.z / value_mag) * 2 / XM_PI;

							if (value.z >= 0) Color = XMFLOAT4(0, 1 - b + a3 * b, b, alpha);
							else Color = XMFLOAT4(b - 1, a3*(2 - b), 1, alpha);
						}
						if (value.x >= 0 && value.y < 0) {

							float a4 = (float)asin(value.x / sqrt(value.x*value.x + value.y*value.y)) * 2 / XM_PI;
							float b = (float)acos(value.z / value_mag) * 2 / XM_PI;

							if (value.z >= 0) Color = XMFLOAT4(a4*b, 1 - a4 * b, b - a4 * b, alpha);
							else Color = XMFLOAT4(1 - (2 - b)*(1 - a4), (2 - b)*(1 - a4), 1 - a4 * (2 - b), alpha);
						}
						if (IsZ(value.x) && IsZ(value.y)) {

							if (value.z > 0) Color = XMFLOAT4(0.0f, 1.0f, 0.0f, alpha);
							else Color = XMFLOAT4(1.0f, 0.0f, 1.0f, alpha);
						}
					}
					else Color = XMFLOAT4(1.0f, 1.0f, 1.0f, alpha);
				}
				else {

					//for VEC3REP_X, VEC3REP_Y and VEC3REP_Z, VEC3REP_MAGNITUDE, use color coding based on min-max values, i.e. calculated the same as for a scalar quantity : done in AdjustMagnitude_VEC
					Color = XMFLOAT4(0.0f, 0.0f, 1.0f, minimum(get_alpha_value(translation), (float)maxTransparency));
				}

				//if thresholds enabled, then check if we should skip adding cell to rendering queue.
				bool skip = false;
				if (!displayThresholds.IsNull()) {

					switch (displayThresholdTrigger) {

					case (int)VEC3REP_MAGNITUDE:
						skip = (value_mag <= displayThresholds.i || value_mag >= displayThresholds.j);
						break;
					case (int)VEC3REP_X:
						skip = (value.x <= displayThresholds.i || value.x >= displayThresholds.j);
						break;
					case (int)VEC3REP_Y:
						skip = (value.y <= displayThresholds.i || value.y >= displayThresholds.j);
						break;
					case (int)VEC3REP_Z:
						skip = (value.z <= displayThresholds.i || value.z >= displayThresholds.j);
						break;
					}
				}

				//skip this cell
				if (skip) {

					physQRepComponent.transformBatch[idx] = CBObjectTransform();
					physQRepComponent.emptyCell[idx] = true;
					physQRepComponent.transformBatch[idx].skip_render = true;
				}

				switch (physQRepComponent.vec3rep)
				{
				case VEC3REP_FULL:
				case VEC3REP_DIRECTION:
					//reduce magnitude values
					omp_reduction.reduce_minmax(value_mag);
					if (!skip) physQRepComponent.transformBatch[idx] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value_mag);
					break;

				case VEC3REP_MAGNITUDE:
					//reduce magnitude values
					omp_reduction.reduce_minmax(value_mag);
					if (!skip) physQRepComponent.transformBatch[idx] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value_mag);
					break;

				case VEC3REP_X:
					//reduce x values
					omp_reduction.reduce_minmax(value.x);
					if (!skip) physQRepComponent.transformBatch[idx] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value.x);
					break;

				case VEC3REP_Y:
					//reduce y values
					omp_reduction.reduce_minmax(value.y);
					if (!skip) physQRepComponent.transformBatch[idx] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value.y);
					break;

				case VEC3REP_Z:
					//reduce z values
					omp_reduction.reduce_minmax(value.z);
					if (!skip) physQRepComponent.transformBatch[idx] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value.z);
					break;
				}

				physQRepComponent.transformBatch[idx].maxTransparency = maxTransparency;
				
				//skip rendering obscured cells?
				if (!skip && ((renderspeedup2_cells && ndisp.dim() >= renderspeedup2_cells) || (renderspeedup3_cells && ndisp.dim() >= renderspeedup3_cells))) {

					bool force_skip = false;

					if (physQRepComponent.vec3rep == VEC3REP_FULL && renderspeedup3_cells && ndisp.dim() >= renderspeedup3_cells) {

						//red_nudge = true for odd rows and even planes or for even rows and odd planes - have to keep index on the checkerboard pattern
						bool red_nudge = (((j % 2) == 1 && (k % 2) == 0) || (((j % 2) == 0 && (k % 2) == 1)));

						//true for black square, false for red square
						force_skip = (i - red_nudge) % 2;
						if (force_skip) physQRepComponent.transformBatch[idx].skip_render = true;
					}
					
					//enable faster rendering by skipping cells which are surrounded (thus obscured).
					if (!force_skip && renderspeedup2_cells && i > 0 && j > 0 && k > 0 && i < ndisp.x - 1 && j < ndisp.y - 1 && k < ndisp.z - 1) {

						Rect cellRect_1 = Rect(DBL3(i - 1, j - 1, k - 1) & hdisp, DBL3(i, j + 2, k + 2) & hdisp);
						Rect cellRect_2 = Rect(DBL3(i + 1, j - 1, k - 1) & hdisp, DBL3(i + 2, j + 2, k + 2) & hdisp);
						Rect cellRect_3 = Rect(DBL3(i, j - 1, k - 1) & hdisp, DBL3(i + 1, j, k + 2) & hdisp);
						Rect cellRect_4 = Rect(DBL3(i, j + 1, k - 1) & hdisp, DBL3(i + 1, j + 2, k + 2) & hdisp);
						Rect cellRect_5 = Rect(DBL3(i, j, k + 1) & hdisp, DBL3(i + 1, j + 1, k + 2) & hdisp);
						Rect cellRect_6 = Rect(DBL3(i, j, k - 1) & hdisp, DBL3(i + 1, j + 1, k) & hdisp);

						if (pQ->is_not_empty(cellRect_1) && pQ->is_not_empty(cellRect_2) && 
							pQ->is_not_empty(cellRect_3) && pQ->is_not_empty(cellRect_4) &&
							pQ->is_not_empty(cellRect_5) && pQ->is_not_empty(cellRect_6))
							physQRepComponent.transformBatch[idx].skip_render = true;
					}
				}

				if (!physQRepComponent.transformBatch[idx].skip_render) num_rendered_cells++;
			}
		}
	}
	
	//make sure transformBatch_render has correct size as this will detemine if it's used for rendering or not; elements will be copied from transformBatch to transformBatch_render after this function completes if needed
	if (num_rendered_cells < physQRepComponent.transformBatch.linear_size()) {

		if (physQRepComponent.transformBatch_render.size() != num_rendered_cells) physQRepComponent.transformBatch_render.resize(num_rendered_cells);
	}
	else {

		physQRepComponent.transformBatch_render.clear();
	}

	//return minimum, maximum
	return omp_reduction.minmax();
}

//////////////////// DUAL VEC DISPLAY

template DBL2 PhysQRep::CalculateRepresentation_VEC<VEC<DBL3>>(VEC<DBL3>* pQ, VEC<DBL3>* pQ2, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds, int displayThresholdTrigger);
template DBL2 PhysQRep::CalculateRepresentation_VEC<VEC_VC<DBL3>>(VEC_VC<DBL3>* pQ, VEC_VC<DBL3>* pQ2, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds, int displayThresholdTrigger);
template DBL2 PhysQRep::CalculateRepresentation_VEC<VEC<FLT3>>(VEC<FLT3>* pQ, VEC<FLT3>* pQ2, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds, int displayThresholdTrigger);
template DBL2 PhysQRep::CalculateRepresentation_VEC<VEC_VC<FLT3>>(VEC_VC<FLT3>* pQ, VEC_VC<FLT3>* pQ2, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds, int displayThresholdTrigger);

template <typename VECType>
DBL2 PhysQRep::CalculateRepresentation_VEC(VECType* pQ, VECType* pQ2, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds, int displayThresholdTrigger)
{
	Rect meshRect = pQ->rect;
	DBL3 h = pQ->h;
	INT3 n = pQ->n;

	//display cell size - this sets the level of displayed detail
	DBL3 hdisp(detail_level);

	//number of cells for display, and adjust display cell size (must have an integer number of cells to display)
	INT3 ndisp = round(meshRect / hdisp);
	
	if (ndisp.x < 1) ndisp.x = 1;
	if (ndisp.y < 1) ndisp.y = 1;
	if (ndisp.z < 1) ndisp.z = 1;
	if (ndisp.x > n.x)  ndisp.x = n.x;
	if (ndisp.y > n.y)  ndisp.y = n.y;
	if (ndisp.z > n.z)  ndisp.z = n.z;

	//double the number of cells along z to fit in the dual representation : this will also halve the z cellsize
	ndisp.z *= 2;
	hdisp = meshRect / ndisp;

	physQRepComponent.transformBatch.resize(hdisp, meshRect);
	if (physQRepComponent.emptyCell.size() != ndisp.dim()) physQRepComponent.emptyCell.assign(physQRepComponent.transformBatch.linear_size(), false);

	//when calculating representation to go back to single representation values
	ndisp.z /= 2;
	hdisp.z *= 2;

	if (physQRepComponent.vec3rep == VEC3REP_FULL) {

		physQRepComponent.obSelector = CDO_HALFARROW;
	}
	else {

		//if not a full representation then we'll be representing a scalar-type quantity extracted from the VEC
		physQRepComponent.obSelector = CDO_HALFCUBE;
	}

	//Vector quantity - we need to compute rotations, scaling and color coding

	omp_reduction.new_minmax_reduction();

	int num_rendered_cells = 0;

	for (int k = 0; k < ndisp.z; k++) {
#pragma omp parallel for reduction(+:num_rendered_cells)
		for (int j = 0; j < ndisp.y; j++) {
			for (int i = 0; i < ndisp.x; i++) {

				DBL3 value, value2;

				DBL3 rel_pos = hdisp & (DBL3(i, j, k) + DBL3(0.5));

				int idx1 = i + j * ndisp.x + 2 * k * ndisp.x*ndisp.y;
				int idx2 = i + j * ndisp.x + (2 * k + 1) * ndisp.x*ndisp.y;

				physQRepComponent.transformBatch[idx1].skip_render = false;
				physQRepComponent.transformBatch[idx2].skip_render = false;

				if (pQ->is_empty(rel_pos)) {

					//most cells are not empty, so the check above should be kept since it's fast
					//there is the possibility that for a coarse display cell the check above returns empty but actually the coarse cell is not empty
					//thus use the complete check here with the coarse cell rectangle
					Rect cellRect = Rect(DBL3(i, j, k) & hdisp, DBL3(i + 1, j + 1, k + 1) & hdisp);
					if (pQ->is_empty(cellRect)) {

						//set this to empty
						physQRepComponent.transformBatch[idx1] = CBObjectTransform();
						physQRepComponent.transformBatch[idx2] = CBObjectTransform();
						physQRepComponent.transformBatch[idx1].skip_render = true;
						physQRepComponent.transformBatch[idx2].skip_render = true;

						physQRepComponent.emptyCell[idx1] = true;
						physQRepComponent.emptyCell[idx2] = true;
						continue;
					}
					else {

						value = (DBL3)(*pQ).average(cellRect);
						physQRepComponent.emptyCell[idx1] = false;

						value2 = (DBL3)(*pQ2).average(cellRect);
						physQRepComponent.emptyCell[idx2] = false;
					}
				}
				else {

					value = (DBL3)(*pQ).cell_sum(rel_pos);
					physQRepComponent.emptyCell[idx1] = false;

					value2 = (DBL3)(*pQ2).cell_sum(rel_pos);
					physQRepComponent.emptyCell[idx2] = false;
				}

				//the magnitude of VEC3 value
				double value_mag = (double)GetMagnitude(value);
				double value_mag2 = (double)GetMagnitude(value2);

				XMMATRIX Rotation = XMMatrixIdentity();
				XMMATRIX Rotation2 = XMMatrixIdentity();

				DBL3 scaling = hdisp * m_to_l;
				XMMATRIX Scale = XMMatrixIdentity() * XMMatrixScaling(scaling.x, scaling.y, scaling.z);

				switch (physQRepComponent.vec3rep) {

				case VEC3REP_FULL:
					//Only need rotation for full representations
					if (IsNZ(value.x) || IsNZ(value.y)) Rotation = XMMatrixRotationAxis(XMLoadFloat3(&XMFLOAT3((float)-value.y, (float)value.x, 0)), (float)acos(value.z / value_mag));
					else {

						if (value.z > 0) Rotation = XMMatrixIdentity();
						else Rotation = XMMatrixRotationX(XM_PI);
					}

					if (IsNZ(value2.x) || IsNZ(value2.y)) Rotation2 = XMMatrixRotationAxis(XMLoadFloat3(&XMFLOAT3((float)-value2.y, (float)value2.x, 0)), (float)acos(value2.z / value_mag2));
					else {

						if (value2.z > 0) Rotation2 = XMMatrixIdentity();
						else Rotation2 = XMMatrixRotationX(XM_PI);
					}
					break;

				case VEC3REP_X:
					Rotation = XMMatrixRotationAxis(XMLoadFloat3(&XMFLOAT3(0, 1, 0)), PI / 2);
					Rotation2 = XMMatrixRotationAxis(XMLoadFloat3(&XMFLOAT3(0, -1, 0)), PI / 2);
					break;

				case VEC3REP_Y:
					Rotation = XMMatrixRotationAxis(XMLoadFloat3(&XMFLOAT3(1, 0, 0)), PI / 2);
					Rotation2 = XMMatrixRotationAxis(XMLoadFloat3(&XMFLOAT3(-1, 0, 0)), PI / 2);
					break;

				case VEC3REP_DIRECTION:
				case VEC3REP_MAGNITUDE:
				case VEC3REP_Z:
					Rotation2 = XMMatrixRotationAxis(XMLoadFloat3(&XMFLOAT3(0, 0, 0)), (float)XM_PI);
					break;
				}

				DBL3 translation = (DBL3(i + 0.5, j + 0.5, k + 0.5) & hdisp) * m_to_l * 2 + (meshRect.s - focusRect.s) * m_to_l * 2 - (focusRect.size() * m_to_l);
				XMMATRIX Translation = XMMatrixTranslation(translation.x, translation.y, translation.z);
				
				XMFLOAT4 Color, Color2;

				float alpha = minimum(get_alpha_value(translation), (float)maxTransparency);

				auto get_color = [](DBL3 value, double value_mag, float alpha) ->XMFLOAT4 {

					XMFLOAT4 Color;

					//use color wheel coding for full representations and direction-only representations

					//(R,G,B) channels. Linear transition between the 6 points defined below.
					//
					//+x : (1,0,0) : RED
					//+y : (1,1,0) : YELLOW
					//-x : (0,0,1) : BLUE
					//-y : (0,1,1) : CYAN
					//+z : (0,1,0) : GREEN
					//-z : (1,0,1) : MAGENTA

					//b is the normalized polar angle, ranges from 0 to 1 for +z and 1 to 2 for -z
					//a1, a2, a3, a4 are in-plane angles in the 4 quadrants: a1 measured from +x axis, a2 from +y axis, a3 from -x axis, a4 from -y axis

					if (value_mag > 0) {

						if (value.x > 0 && value.y >= 0) {

							float a1 = (float)asin(value.y / sqrt(value.x*value.x + value.y*value.y)) * 2 / XM_PI;
							float b = (float)acos(value.z / value_mag) * 2 / XM_PI;

							if (value.z >= 0) Color = XMFLOAT4(b, 1 - b + b * a1, 0.0f, alpha);
							else Color = XMFLOAT4(1, (2 - b)*a1, b - 1, alpha);
						}
						if (value.x <= 0 && value.y > 0) {

							float a2 = (float)asin(-value.x / sqrt(value.x*value.x + value.y*value.y)) * 2 / XM_PI;
							float b = (float)acos(value.z / value_mag) * 2 / XM_PI;

							if (value.z >= 0) Color = XMFLOAT4(b - a2 * b, 1 - a2 * b, a2*b, alpha);
							else Color = XMFLOAT4(1 - a2 * (2 - b), (2 - b)*(1 - a2), 1 - (2 - b)*(1 - a2), alpha);
						}
						if (value.x < 0 && value.y <= 0) {

							float a3 = (float)asin(-value.y / sqrt(value.x*value.x + value.y*value.y)) * 2 / XM_PI;
							float b = (float)acos(value.z / value_mag) * 2 / XM_PI;

							if (value.z >= 0) Color = XMFLOAT4(0, 1 - b + a3 * b, b, alpha);
							else Color = XMFLOAT4(b - 1, a3*(2 - b), 1, alpha);
						}
						if (value.x >= 0 && value.y < 0) {

							float a4 = (float)asin(value.x / sqrt(value.x*value.x + value.y*value.y)) * 2 / XM_PI;
							float b = (float)acos(value.z / value_mag) * 2 / XM_PI;

							if (value.z >= 0) Color = XMFLOAT4(a4*b, 1 - a4 * b, b - a4 * b, alpha);
							else Color = XMFLOAT4(1 - (2 - b)*(1 - a4), (2 - b)*(1 - a4), 1 - a4 * (2 - b), alpha);
						}
						if (IsZ(value.x) && IsZ(value.y)) {

							if (value.z > 0) Color = XMFLOAT4(0.0f, 1.0f, 0.0f, alpha);
							else Color = XMFLOAT4(1.0f, 0.0f, 1.0f, alpha);
						}
					}
					else Color = XMFLOAT4(1.0f, 1.0f, 1.0f, alpha);

					return Color;
				};
				
				if (physQRepComponent.vec3rep == VEC3REP_FULL || physQRepComponent.vec3rep == VEC3REP_DIRECTION) {

					Color = get_color(value, value_mag, alpha);
					Color2 = get_color(value2, value_mag2, alpha);
				}
				else {

					//for VEC3REP_X, VEC3REP_Y, VEC3REP_Z, VEC3REP_MAGNITUDE use color coding based on min-max values, i.e. calculated the same as for a scalar quantity : done in AdjustMagnitude_VEC
					Color = XMFLOAT4(0.0f, 0.0f, 1.0f, minimum(get_alpha_value(translation), (float)maxTransparency));
					Color2 = XMFLOAT4(0.0f, 0.0f, 1.0f, minimum(get_alpha_value(translation), (float)maxTransparency));
				}

				//if thresholds enabled, then check if we should skip adding cell to rendering queue.
				bool skip = false;
				if (!displayThresholds.IsNull()) {

					switch (displayThresholdTrigger) {

					case (int)VEC3REP_MAGNITUDE:
						skip = (value_mag <= displayThresholds.i || value_mag >= displayThresholds.j);
						break;
					case (int)VEC3REP_X:
						skip = (value.x <= displayThresholds.i || value.x >= displayThresholds.j);
						break;
					case (int)VEC3REP_Y:
						skip = (value.y <= displayThresholds.i || value.y >= displayThresholds.j);
						break;
					case (int)VEC3REP_Z:
						skip = (value.z <= displayThresholds.i || value.z >= displayThresholds.j);
						break;
					}
				}

				//skip this cell
				if (skip) {

					physQRepComponent.transformBatch[idx1] = CBObjectTransform();
					physQRepComponent.transformBatch[idx2] = CBObjectTransform();
					physQRepComponent.transformBatch[idx1].skip_render = true;
					physQRepComponent.transformBatch[idx2].skip_render = true;

					physQRepComponent.emptyCell[idx1] = true;
					physQRepComponent.emptyCell[idx2] = true;
				}

				switch (physQRepComponent.vec3rep)
				{
				case VEC3REP_FULL:
				case VEC3REP_DIRECTION:
					//reduce magnitude values
					omp_reduction.reduce_minmax(value_mag);
					omp_reduction.reduce_minmax(value_mag2);
					if (!skip) {

						physQRepComponent.transformBatch[idx1] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value_mag);
						physQRepComponent.transformBatch[idx2] = CBObjectTransform(Rotation2, Scale, Translation, Color2, maxTransparency, translation, value_mag2);
					}
					break;

				case VEC3REP_MAGNITUDE:
					//reduce magnitude values
					omp_reduction.reduce_minmax(value_mag);
					omp_reduction.reduce_minmax(value_mag2);
					if (!skip) {

						physQRepComponent.transformBatch[idx1] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value_mag);
						physQRepComponent.transformBatch[idx2] = CBObjectTransform(Rotation2, Scale, Translation, Color2, maxTransparency, translation, value_mag2);
					}
					break;

				case VEC3REP_X:
					//reduce x values
					omp_reduction.reduce_minmax(value.x);
					omp_reduction.reduce_minmax(value2.x);
					if (!skip) {

						physQRepComponent.transformBatch[idx1] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value.x);
						physQRepComponent.transformBatch[idx2] = CBObjectTransform(Rotation2, Scale, Translation, Color2, maxTransparency, translation, value2.x);
					}
					break;

				case VEC3REP_Y:
					//reduce y values
					omp_reduction.reduce_minmax(value.y);
					omp_reduction.reduce_minmax(value2.y);
					if (!skip) {

						physQRepComponent.transformBatch[idx1] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value.y);
						physQRepComponent.transformBatch[idx2] = CBObjectTransform(Rotation2, Scale, Translation, Color2, maxTransparency, translation, value2.y);
					}
					break;

				case VEC3REP_Z:
					//reduce z values
					omp_reduction.reduce_minmax(value.z);
					omp_reduction.reduce_minmax(value2.z);
					if (!skip) {

						physQRepComponent.transformBatch[idx1] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value.z);
						physQRepComponent.transformBatch[idx2] = CBObjectTransform(Rotation2, Scale, Translation, Color2, maxTransparency, translation, value2.z);
					}
					break;
				}

				physQRepComponent.transformBatch[idx1].maxTransparency = maxTransparency;
				physQRepComponent.transformBatch[idx2].maxTransparency = maxTransparency;

				//skip rendering obscured cells?
				if (!skip && ((renderspeedup2_cells && ndisp.dim() >= renderspeedup2_cells) || (renderspeedup3_cells && ndisp.dim() >= renderspeedup3_cells))) {

					bool force_skip = false;

					if (physQRepComponent.vec3rep == VEC3REP_FULL && renderspeedup3_cells && ndisp.dim() >= renderspeedup3_cells) {

						//red_nudge = true for odd rows and even planes or for even rows and odd planes - have to keep index on the checkerboard pattern
						bool red_nudge = (((j % 2) == 1 && (k % 2) == 0) || (((j % 2) == 0 && (k % 2) == 1)));

						//true for black square, false for red square
						force_skip = (i - red_nudge) % 2;
						if (force_skip) {

							physQRepComponent.transformBatch[idx1].skip_render = true;
							physQRepComponent.transformBatch[idx2].skip_render = true;
						}
					}

					//enable faster rendering by skipping cells which are surrounded (thus obscured).
					if (!force_skip && renderspeedup2_cells && i > 0 && j > 0 && k > 0 && i < ndisp.x - 1 && j < ndisp.y - 1 && k < ndisp.z - 1) {

						Rect cellRect_1 = Rect(DBL3(i - 1, j - 1, k - 1) & hdisp, DBL3(i, j + 2, k + 2) & hdisp);
						Rect cellRect_2 = Rect(DBL3(i + 1, j - 1, k - 1) & hdisp, DBL3(i + 2, j + 2, k + 2) & hdisp);
						Rect cellRect_3 = Rect(DBL3(i, j - 1, k - 1) & hdisp, DBL3(i + 1, j, k + 2) & hdisp);
						Rect cellRect_4 = Rect(DBL3(i, j + 1, k - 1) & hdisp, DBL3(i + 1, j + 2, k + 2) & hdisp);
						Rect cellRect_5 = Rect(DBL3(i, j, k + 1) & hdisp, DBL3(i + 1, j + 1, k + 2) & hdisp);
						Rect cellRect_6 = Rect(DBL3(i, j, k - 1) & hdisp, DBL3(i + 1, j + 1, k) & hdisp);

						if (pQ->is_not_empty(cellRect_1) && pQ->is_not_empty(cellRect_2) &&
							pQ->is_not_empty(cellRect_3) && pQ->is_not_empty(cellRect_4) &&
							pQ->is_not_empty(cellRect_5) && pQ->is_not_empty(cellRect_6)) {

							physQRepComponent.transformBatch[idx1].skip_render = true;
							physQRepComponent.transformBatch[idx2].skip_render = true;
						}
					}
				}

				if (!physQRepComponent.transformBatch[idx1].skip_render) num_rendered_cells++;
				if (!physQRepComponent.transformBatch[idx2].skip_render) num_rendered_cells++;
			}
		}
	}

	//make sure transformBatch_render has correct size as this will detemine if it's used for rendering or not; elements will be copied from transformBatch to transformBatch_render after this function completes if needed
	if (num_rendered_cells < physQRepComponent.transformBatch.linear_size()) {

		if (physQRepComponent.transformBatch_render.size() != num_rendered_cells) physQRepComponent.transformBatch_render.resize(num_rendered_cells);
	}
	else {

		physQRepComponent.transformBatch_render.clear();
	}

	//return minimum, maximum
	return omp_reduction.minmax();
}

void PhysQRep::AdjustMagnitude_VEC(PhysQRepComponent& physQRepComponent, DBL2 minmax)
{
	if (physQRepComponent.vec3rep == VEC3REP_FULL) {

		//only adjust scaling for full representations

		double delta = minmax.j - minmax.i;

		if (delta > 0 && minmax.j) {

#pragma omp parallel for
			for (int tbidx = 0; tbidx < physQRepComponent.transformBatch.linear_size(); tbidx++) {

				//do not use a linear scaling as it doesn't look good. Use a gentle decrease in the upper range of 0 to 1 : power 0.2 works nicely
				float magRel = (float)pow(physQRepComponent.transformBatch[tbidx].value / minmax.j, physQRepComponent.exponent);
				physQRepComponent.transformBatch[tbidx].Scale *= XMMatrixScaling(magRel, magRel, magRel);
			}
		}
	}
	else if (physQRepComponent.vec3rep != VEC3REP_DIRECTION) {

		AdjustMagnitude_SCA(physQRepComponent, minmax);
	}
}

//------------------------------------------------------------------------------ SCALAR QUANTITIES ------------------------------------------------------------------------------//

template DBL2 PhysQRep::CalculateRepresentation_SCA<VEC<double>>(VEC<double>* pQ, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds);
template DBL2 PhysQRep::CalculateRepresentation_SCA<VEC_VC<double>>(VEC_VC<double>* pQ, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds);
template DBL2 PhysQRep::CalculateRepresentation_SCA<VEC<float>>(VEC<float>* pQ, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds);
template DBL2 PhysQRep::CalculateRepresentation_SCA<VEC_VC<float>>(VEC_VC<float>* pQ, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds);

template <typename VECType>
DBL2 PhysQRep::CalculateRepresentation_SCA(VECType* pQ, PhysQRepComponent& physQRepComponent, double maxTransparency, DBL2 displayThresholds)
{
	Rect meshRect = pQ->rect;
	DBL3 h = pQ->h;
	INT3 n = pQ->n;

	//display cell size - this sets the level of displayed detail
	DBL3 hdisp(detail_level);

	//number of cells for display, and adjust display cell size (must have an integer number of cells to display)
	INT3 ndisp = round(meshRect / hdisp);

	if (ndisp.x < 1) ndisp.x = 1;
	if (ndisp.y < 1) ndisp.y = 1;
	if (ndisp.z < 1) ndisp.z = 1;
	if (ndisp.x > n.x)  ndisp.x = n.x;
	if (ndisp.y > n.y)  ndisp.y = n.y;
	if (ndisp.z > n.z)  ndisp.z = n.z;
	hdisp = meshRect / ndisp;

	if (physQRepComponent.transformBatch.size() != ndisp) physQRepComponent.transformBatch.assign(hdisp, meshRect, CBObjectTransform());
	if (physQRepComponent.emptyCell.size() != ndisp.dim()) physQRepComponent.emptyCell.assign(physQRepComponent.transformBatch.linear_size(), false);
	
	physQRepComponent.obSelector = CDO_CUBE;

	//Scalar quantity - we need to compute color coding

	omp_reduction.new_minmax_reduction();

	int num_rendered_cells = 0;

	#pragma omp parallel for reduction(+:num_rendered_cells)
	for (int j = 0; j < ndisp.y; j++) {
		for (int k = 0; k < ndisp.z; k++) {
			for (int i = 0; i < ndisp.x; i++) {

				double value;

				DBL3 rel_pos = hdisp & (DBL3(i, j, k) + DBL3(0.5));

				int idx = i + j * ndisp.x + k * ndisp.x*ndisp.y;

				physQRepComponent.transformBatch[idx].skip_render = false;

				if (pQ->is_empty(rel_pos)) {

					//most cells are not empty, so the check above should be kept since it's fast
					//there is the possibility that for a coarse display cell the check above returns empty but actually the coarse cell is not empty
					//thus use the complete check here with the coarse cell rectangle
					Rect cellRect = Rect(DBL3(i, j, k) & hdisp, DBL3(i + 1, j + 1, k + 1) & hdisp);
					if (pQ->is_empty(cellRect)) {

						//set this to empty
						physQRepComponent.transformBatch[idx] = CBObjectTransform();
						physQRepComponent.transformBatch[idx].skip_render = true;
						physQRepComponent.emptyCell[idx] = true;
						continue;
					}
					else {

						value = (double)(*pQ).average(cellRect);
						physQRepComponent.emptyCell[idx] = false;
					}
				}
				else {

					value = (double)(*pQ).cell_sum(rel_pos);
					physQRepComponent.emptyCell[idx] = false;
				}

				//reduce values
				omp_reduction.reduce_minmax(value);

				XMMATRIX Rotation = XMMatrixIdentity();

				DBL3 scaling = hdisp * m_to_l;
				XMMATRIX Scale = XMMatrixIdentity() * XMMatrixScaling(scaling.x, scaling.y, scaling.z);

				DBL3 translation = (DBL3(i + 0.5, j + 0.5, k + 0.5) & hdisp) * m_to_l * 2 + (meshRect.s - focusRect.s) * m_to_l * 2 - (focusRect.size() * m_to_l);
				XMMATRIX Translation = XMMatrixTranslation(translation.x, translation.y, translation.z);

				XMFLOAT4 Color = XMFLOAT4(0.0f, 0.0f, 1.0f, minimum(get_alpha_value(translation), (float)maxTransparency));

				bool skip = false;

				if (!displayThresholds.IsNull() && (value <= displayThresholds.i || value >= displayThresholds.j)) {

					physQRepComponent.transformBatch[idx] = CBObjectTransform();
					physQRepComponent.transformBatch[idx].skip_render = true;
					physQRepComponent.emptyCell[idx] = true;
					skip = true;
				}
				else physQRepComponent.transformBatch[idx] = CBObjectTransform(Rotation, Scale, Translation, Color, maxTransparency, translation, value);

				physQRepComponent.transformBatch[idx].maxTransparency = maxTransparency;

				//skip rendering obscured cells?
				if (!skip && renderspeedup2_cells && ndisp.dim() >= renderspeedup2_cells) {

					//enable faster rendering by skipping cells which are surrounded (thus obscured).
					if (renderspeedup2_cells && i > 0 && j > 0 && k > 0 && i < ndisp.x - 1 && j < ndisp.y - 1 && k < ndisp.z - 1) {

						Rect cellRect_1 = Rect(DBL3(i - 1, j - 1, k - 1) & hdisp, DBL3(i, j + 2, k + 2) & hdisp);
						Rect cellRect_2 = Rect(DBL3(i + 1, j - 1, k - 1) & hdisp, DBL3(i + 2, j + 2, k + 2) & hdisp);
						Rect cellRect_3 = Rect(DBL3(i, j - 1, k - 1) & hdisp, DBL3(i + 1, j, k + 2) & hdisp);
						Rect cellRect_4 = Rect(DBL3(i, j + 1, k - 1) & hdisp, DBL3(i + 1, j + 2, k + 2) & hdisp);
						Rect cellRect_5 = Rect(DBL3(i, j, k + 1) & hdisp, DBL3(i + 1, j + 1, k + 2) & hdisp);
						Rect cellRect_6 = Rect(DBL3(i, j, k - 1) & hdisp, DBL3(i + 1, j + 1, k) & hdisp);

						if (pQ->is_not_empty(cellRect_1) && pQ->is_not_empty(cellRect_2) &&
							pQ->is_not_empty(cellRect_3) && pQ->is_not_empty(cellRect_4) &&
							pQ->is_not_empty(cellRect_5) && pQ->is_not_empty(cellRect_6))
							physQRepComponent.transformBatch[idx].skip_render = true;
					}
				}

				if (!physQRepComponent.transformBatch[idx].skip_render) num_rendered_cells++;
			}
		}
	}

	//make sure transformBatch_render has correct size as this will detemine if it's used for rendering or not; elements will be copied from transformBatch to transformBatch_render after this function completes if needed
	if (num_rendered_cells < physQRepComponent.transformBatch.linear_size()) {

		if (physQRepComponent.transformBatch_render.size() != num_rendered_cells) physQRepComponent.transformBatch_render.resize(num_rendered_cells);
	}
	else {

		physQRepComponent.transformBatch_render.clear();
	}

	//return minimum, maximum
	return omp_reduction.minmax();
}

void PhysQRep::AdjustMagnitude_SCA(PhysQRepComponent& physQRepComponent, DBL2 minmax)
{
	double delta = minmax.j - minmax.i;

	//color the objects acording to their position between minimum and maximum
	//(R,G,B) channels. Smallest value is blue (0, 0, 1) -> halfway is black (0, 0, 0) -> largest value is red (1, 0, 0)
	if (delta > 0) {

		#pragma omp parallel for
		for (int tbidx = 0; tbidx < physQRepComponent.transformBatch.linear_size(); tbidx++) {

			//magNorm ranges from 0 to 1
			float magNorm = float((physQRepComponent.transformBatch[tbidx].value - minmax.i) / delta);

			if (magNorm < 0.5) {

				float alpha = physQRepComponent.transformBatch[tbidx].Color.w;

				//blue color range
				physQRepComponent.transformBatch[tbidx].Color = XMFLOAT4(0.0f, 0.0f, 1 - 2 * magNorm, alpha);
			}
			else {

				float alpha = physQRepComponent.transformBatch[tbidx].Color.w;

				//red color range
				physQRepComponent.transformBatch[tbidx].Color = XMFLOAT4(2 * magNorm - 1, 0.0f, 0.0, alpha);
			}
		}
	}
}

//------------------------------------------------------------------------------ DRAWING ------------------------------------------------------------------------------//

float PhysQRep::get_alpha_value(DBL3 position)
{
	float alpha = pow(get_distance(pBG->GetCameraPosition(), (FLT3)position) / CAMERACUTOFFDISTANCE, ALPHABLENDPOW);
	if (alpha > ALPHAMAXIMUM) alpha = ALPHAMAXIMUM;

	return alpha;
}

//calculate transparency (objects closer to camera become more transparent)
void PhysQRep::SetAlphaBlend(void)
{
	for (int idx = 0; idx < (int)physQRep.size(); idx++) {

		if (physQRep[idx].transformBatch_render.size()) {

#pragma omp parallel for
			for (int cidx = 0; cidx < physQRep[idx].transformBatch_render.size(); cidx++) {

				physQRep[idx].transformBatch_render[cidx].Color.w = minimum(get_alpha_value(physQRep[idx].transformBatch_render[cidx].position), (float)physQRep[idx].transformBatch_render[cidx].maxTransparency);
			}
		}
		else {

#pragma omp parallel for
			for (int cidx = 0; cidx < physQRep[idx].transformBatch.linear_size(); cidx++) {

				physQRep[idx].transformBatch[cidx].Color.w = minimum(get_alpha_value(physQRep[idx].transformBatch[cidx].position), (float)physQRep[idx].transformBatch[cidx].maxTransparency);
			}
		}
	}
}

void PhysQRep::DrawPhysQRep(void)
{
	for (int idx = 0; idx < (int)physQRep.size(); idx++) {

		//draw mesh values representations
		if (physQRep[idx].transformBatch_render.size()) pBG->DrawCBObjectBatch(physQRep[idx].transformBatch_render, (CDO_)physQRep[idx].obSelector);
		else pBG->DrawCBObjectBatch(physQRep[idx].transformBatch.get_vector(), (CDO_)physQRep[idx].obSelector);
		
		//draw mesh frame
		if(physQRep[idx].drawFrame) 
			pBG->DrawFrameObject(CDO_CUBEFRAME, XMMatrixIdentity(), physQRep[idx].meshFrame.Scale, physQRep[idx].meshFrame.Trans, MESHFRAMECOLOR);
	}

	//draw highlighted cell if any
	if(highlightCell)
		pBG->DrawCBObject(CDO_CUBE, XMMatrixIdentity(), highlightedCell.Scale * XMMatrixScaling(HIGHLIGHTCELLSCALE, HIGHLIGHTCELLSCALE, HIGHLIGHTCELLSCALE), highlightedCell.Trans, HIGHLIGHTCELLCOLOR);
}

//------------------------------------------- Value reading

bool PhysQRep::GetMouseInfo(INT2 mouse, std::string* pmeshName, std::string* ptypeName, DBL3* pMeshPosition, double* pMeshValue, std::string* pValueUnit)
{
	highlightCell = false;

	//pick far clipping plane point corresponding to mouse position, and get camera point.
	//convert to mesh coordinates, remembering the world coordinates center is located at the center of the focused mesh
	DBL3 farPoint = pBG->Pick_FarPlane_Point(mouse) / (2 * m_to_l) + focusRect.size() / 2 + focusRect.s;
	DBL3 camPoint = pBG->GetCameraPosition() / (2 * m_to_l) + focusRect.size() / 2 + focusRect.s;

	//these are used to store multiple intersections and will sort them by distance. The closest mesh is not necessarily the final output since it may have empty cells.
	mouseInfo.assign(physQRep.size(), { get_distance(camPoint, farPoint), 0, DBL3() });

	bool intersection_found = false;

	//First just check all mesh rectangles for closest intersections with the line from camPoint to farPoint
	for (int idx = 0; idx < physQRep.size(); idx++) {

		if (!physQRep[idx].transformBatch.linear_size()) continue;

		Rect meshRect = physQRep[idx].transformBatch.rect;
		DBL3 intersectionPoint;

		if (meshRect.intersection_test(camPoint, farPoint, &intersectionPoint)) {

			mouseInfo[idx] = { get_distance(camPoint, intersectionPoint), idx, intersectionPoint };
			intersection_found = true;
		}
	}

	//have we found a mesh?
	if (intersection_found) {

		//sort all obtained intersections by distance - closest to camera first
		std::sort(mouseInfo.begin(), mouseInfo.end());

		//now check all of them - cannot just read from first mesh intersection as there may be empty cells
		for (int idx = 0; idx < mouseInfo.size(); idx++) {

			int picked_mesh_idx = std::get<1>(mouseInfo[idx]);
			DBL3 meshPoint = std::get<2>(mouseInfo[idx]);

			Rect meshRect = physQRep[picked_mesh_idx].transformBatch.rect;
			DBL3 cellSize = physQRep[picked_mesh_idx].transformBatch.h;
			INT3 cells = physQRep[picked_mesh_idx].transformBatch.n;

			if (meshPoint == camPoint) return false;
			DBL3 direction = (meshPoint - camPoint).normalized();
			DBL3 displacement = direction * DBL3(cellSize.x * direction.x, cellSize.y * direction.y, cellSize.z * direction.z).norm();
			if (displacement == DBL3()) return false;

			//advance by a small fraction of displacement to start off with - if we read value just from edge of mesh on the positive sides, cellidx_from_position will return the outside bounding cell index values which we cannot use to index transformBatch
			meshPoint += displacement / 1000;

			int num_cells_checked = 0;
			while (meshRect.contains(meshPoint) && num_cells_checked < 1000) {

				//get cell index in current mesh being checked
				INT3 cellIndex = physQRep[picked_mesh_idx].transformBatch.cellidx_from_position(meshPoint);
				int linear_cellIndex = cellIndex.i + cellIndex.j * cells.x + cellIndex.k * cells.x * cells.y;

				//check for empty cell - if not empty then we have our value : set output and return
				//it is important you check the upper bound for linear_cellIndex - this used to be the cause of a bug
				//cellidx_from_position above can return a cellIndex which cannot be used for indexing e.g. INT3(n.x, n.y, n.z)
				//this can happen even though we used the contains(meshPoint) check above - this check uses <= comparison at the upper end
				//can cause a bad memory access if used without further checks, thus when you convert to linear cell index check the upper bound
				//Note, cellidx_from_position is meant to do this.
				if (linear_cellIndex < physQRep[picked_mesh_idx].emptyCell.size() && !physQRep[picked_mesh_idx].emptyCell[linear_cellIndex]) {

					//return mesh position of cell center
					if (pMeshPosition)
						*pMeshPosition = (cellSize & cellIndex) + meshRect.s;

					if (pMeshValue)
						*pMeshValue = physQRep[picked_mesh_idx].transformBatch[cellIndex].value;

					if (pValueUnit)
						*pValueUnit = physQRep[picked_mesh_idx].unit;

					//mesh name - display type name
					*pmeshName = physQRep[picked_mesh_idx].meshName;
					*ptypeName = physQRep[picked_mesh_idx].typeName;

					highlightedCell = physQRep[picked_mesh_idx].transformBatch[cellIndex];
					highlightCell = true;

					return true;
				}
				//if empty then advance along line direction to check next cell
				else meshPoint += displacement;

				num_cells_checked++;
			}

			//if we are here then the mesh was empty along line : try next mesh intersection along line
		}
	}

	return false;
}

#endif