#include "stdafx.h"
#include "SkyrmionTrack.h"
#include "SimulationData.h"

//-------------------------------------------------- Get_skyshift : CUDA 

#if COMPILECUDA == 1

DBL2 SkyrmionTrack::Get_skyshiftCUDA(mcu_VEC_VC(cuReal3)& M, Rect skyRect)
{
	//must have a set rectangle
	if (skyRect.IsNull()) return DBL2();

	int skyTrack_idx = Get_skyTrack_index(skyRect, M.rect);
	if (skyTrack_idx < 0) return DBL2();

	//shift the skyrmion rectangle to current tracking position
	skyRect += DBL3(skyTrack_ShiftLast[skyTrack_idx].x, skyTrack_ShiftLast[skyTrack_idx].y, 0.0);

	//Find M averages in the 4 skyRect xy-plane quadrants
	DBL3 bottom_left = (DBL3)M.average_nonempty(skyRect.get_quadrant_bl());
	DBL3 bottom_right = (DBL3)M.average_nonempty(skyRect.get_quadrant_br());
	DBL3 top_left = (DBL3)M.average_nonempty(skyRect.get_quadrant_tl());
	DBL3 top_right = (DBL3)M.average_nonempty(skyRect.get_quadrant_tr());

	//the new shift value
	DBL2 skyTrack_newShift = DBL2();

	//if left half z-component value modulus has increased compared to the right half then it contains less of the skyrmion ring -> skyrmion must have shifted along +x so follow it
	if (mod(bottom_left.z + top_left.z) > mod(bottom_right.z + top_right.z)) {

		skyTrack_newShift.x += M.h.x;
	}
	else {

		skyTrack_newShift.x -= M.h.x;
	}

	//if bottom half z-component value modulus has increased compared to the top half then it contains less of the skyrmion ring -> skyrmion must have shifted along +y so follow it
	if (mod(bottom_left.z + bottom_right.z) > mod(top_left.z + top_right.z)) {

		skyTrack_newShift.y += M.h.y;
	}
	else {

		skyTrack_newShift.y -= M.h.y;
	}

	//set actual total shift as average of new and last total shift values - this eliminates tracking oscillations
	skyTrack_Shift[skyTrack_idx] = skyTrack_ShiftLast[skyTrack_idx] + skyTrack_newShift / 2;

	//save current total shift for next time
	skyTrack_ShiftLast[skyTrack_idx] += skyTrack_newShift;

	return skyTrack_Shift[skyTrack_idx];
}

DBL4 SkyrmionTrack::Get_skypos_diametersCUDA(mcu_VEC_VC(cuReal3)& M, Rect skyRect)
{
	//must have a set rectangle
	if (skyRect.IsNull()) return DBL4();

	int skyTrack_idx = Get_skyTrack_index(skyRect, M.rect);
	if (skyTrack_idx < 0) return DBL4();

	CurveFitting fit;

	std::vector<double> params;

	//skyRect was the original skyrmion rectangle, used to identify it. We now want to work with the updated skyrmion rectangle.
	skyRect = skyTrack_rect[skyTrack_idx];

	//in-plane mesh dimensions
	DBL2 meshDim = DBL2(M.rect.e.x - M.rect.s.x, M.rect.e.y - M.rect.s.y);

	//maximum number of points along any dimension in the skyrmion tracking window; multiply by 2 since the skyrmion window is adjusted to be 2 times the skyrmion diameter so in the extreme it could be two times the mesh size.
	int max_points = maximum((M.rect.e.x - M.rect.s.x) / M.h.x, (M.rect.e.y - M.rect.s.y) / M.h.y) * 2;
	if (max_points <= 0) return DBL4();

	auto fit_x_axis_zerocrossing = [&](void) -> DBL2 {

		auto search_line = [&](double pos_y) -> DBL2 {

			int points_x = (skyRect.e.x - skyRect.s.x) / M.h.x;

			//skyrmion too large for current mesh size so fail the fitting
			if (points_x > max_points) return DBL2(-1);

			//keep working arrays to maximum possible size (memory use is not significant and this avoids allocating memory often).
			if (pdata_gpu->size() < max_points) pdata_gpu->resize(max_points);
			if (pdata_gpu->size() != data_cpu.size()) data_cpu.resize(pdata_gpu->size());

			//Extract profile from M (gpu to gpu)
			double position = skyRect.s.x + M.h.x/2;
			position -= floor_epsilon(position / meshDim.x) * meshDim.x;

			if (pdata_gpu->size() >= points_x) M.extract_profilevalues_component_z(points_x, *pdata_gpu, cuReal3(position, pos_y, M.h.z / 2), cuReal3(M.h.x, 0, 0));
			//just in case
			else return DBL2(-1);

			//Transfer extracted profile from gpu to cpu
			pdata_gpu->copy_to_vector(data_cpu);

			bool plus_sign = data_cpu[0] > 0;
			double first_crossing = 0.0, second_crossing = 0.0;

			for (int idx = 0; idx < points_x; idx++) {

				double value = data_cpu[idx];

				if ((plus_sign && value < 0) || (!plus_sign && value > 0)) {

					plus_sign = !plus_sign;

					if (!first_crossing) {

						first_crossing = skyRect.s.x + ((double)idx + 0.5) * M.h.x;
					}
					else {

						second_crossing = skyRect.s.x + ((double)idx + 0.5) * M.h.x;
						break;
					}
				}
			}

			if (first_crossing && second_crossing) return DBL2(second_crossing - first_crossing, (first_crossing + second_crossing) / 2);
			else return DBL2(-1);
		};

		//initially search through the center of the tracker rectangle
		double pos_y = (skyRect.e.y + skyRect.s.y) / 2;
		pos_y -= floor_epsilon(pos_y / meshDim.y) * meshDim.y;

		DBL2 dia_pos = search_line(pos_y);

		if (dia_pos.i > 0) return dia_pos;
		else {

			DBL2 max_dia_pos = DBL2(-1);

			//bounds couldn't be found, so search line by line for the largest bounds distance
			for (int idx_y = 0; idx_y < (skyRect.e.y - skyRect.s.y) / M.h.y; idx_y++) {

				double pos_y = skyRect.s.y + idx_y * M.h.y;
				pos_y -= floor_epsilon(pos_y / meshDim.y) * meshDim.y;
				dia_pos = search_line(pos_y);
				if (dia_pos >= 0) {

					if (dia_pos.i > max_dia_pos.i) max_dia_pos = dia_pos;
				}
			}

			if (max_dia_pos >= 0) return max_dia_pos;

			//searched everything and still couldn't find bounds : no skyrmion present in current rectangle, or skyrmion too small for current cellsize
			//finally try to set the tracker rectangle to the entire mesh and search again

			//set tracker rectangle to entire mesh, remembering we need a relative rect
			skyRect = M.rect - M.rect.s;

			for (int idx_y = 0; idx_y < (skyRect.e.y - skyRect.s.y) / M.h.y; idx_y++) {

				dia_pos = search_line(skyRect.s.y + idx_y * M.h.y);
				if (dia_pos >= 0) {

					if (dia_pos.i > max_dia_pos.i) max_dia_pos = dia_pos;
				}
			}

			return max_dia_pos;
		}
	};

	auto fit_y_axis_zerocrossing = [&](void) -> DBL2 {

		auto search_line = [&](double pos_x) -> DBL2 {

			int points_y = (skyRect.e.y - skyRect.s.y) / M.h.y;

			//skyrmion too large for current mesh size so fail the fitting
			if (points_y > max_points) return DBL2(-1);

			//keep working arrays to maximum possible size (memory use is not significant and this avoids allocating memory often).
			if (pdata_gpu->size() < max_points) pdata_gpu->resize(max_points);
			if (pdata_gpu->size() != data_cpu.size()) data_cpu.resize(pdata_gpu->size());

			//Extract profile from M (gpu to gpu)
			double position = skyRect.s.y + M.h.y / 2;
			position -= floor_epsilon(position / meshDim.y) * meshDim.y;

			if (pdata_gpu->size() >= points_y) M.extract_profilevalues_component_z(points_y, *pdata_gpu, cuReal3(pos_x, position, M.h.z / 2), cuReal3(0, M.h.y, 0));
			else return DBL2(-1);

			//Transfer extracted profile from gpu to cpu
			pdata_gpu->copy_to_vector(data_cpu);

			bool plus_sign = data_cpu[0] > 0;
			double first_crossing = 0.0, second_crossing = 0.0;

			for (int idx = 0; idx < points_y; idx++) {

				double value = data_cpu[idx];

				if ((plus_sign && value < 0) || (!plus_sign && value > 0)) {

					plus_sign = !plus_sign;

					if (!first_crossing) {

						first_crossing = skyRect.s.y + ((double)idx + 0.5) * M.h.y;
					}
					else {

						second_crossing = skyRect.s.y + ((double)idx + 0.5) * M.h.y;
						break;
					}
				}
			}

			if (first_crossing && second_crossing) return DBL2(second_crossing - first_crossing, (first_crossing + second_crossing) / 2);
			else return DBL2(-1);
		};

		//initially search through the center of the tracker rectangle
		double pos_x = (skyRect.e.x + skyRect.s.x) / 2;
		//wrap around if needed
		pos_x -= floor_epsilon(pos_x / meshDim.x) * meshDim.x;

		DBL2 dia_pos = search_line(pos_x);

		if (dia_pos.i > 0) return dia_pos;
		else {

			DBL2 max_dia_pos = DBL2(-1);

			//bounds couldn't be found, so search line by line for the largest bounds distance
			for (int idx_x = 0; idx_x < (skyRect.e.x - skyRect.s.x) / M.h.x; idx_x++) {

				double pos_x = skyRect.s.x + idx_x * M.h.x;
				//wrap around if needed
				pos_x -= floor_epsilon(pos_x / meshDim.x) * meshDim.x;
				dia_pos = search_line(pos_x);
				if (dia_pos >= 0) {

					if (dia_pos.i > max_dia_pos.i) max_dia_pos = dia_pos;
				}
			}

			if (max_dia_pos >= 0) return max_dia_pos;

			//searched everything and still couldn't find bounds : no skyrmion present in current rectangle, or skyrmion too small for current cellsize
			//finally try to set the tracker rectangle to the entire mesh and search again

			//set tracker rectangle to entire mesh, remembering we need a relative rect
			skyRect = M.rect - M.rect.s;

			for (int idx_x = 0; idx_x < (skyRect.e.x - skyRect.s.x) / M.h.x; idx_x++) {

				dia_pos = search_line(skyRect.s.x + idx_x * M.h.x);
				if (dia_pos >= 0) {

					if (dia_pos.i > max_dia_pos.i) max_dia_pos = dia_pos;
				}
			}

			return max_dia_pos;
		}
	};

	//1. Fitting along x direction

	DBL2 dia_pos = fit_x_axis_zerocrossing();

	//need these checks just in case the fitting fails
	if (dia_pos.i < 0) return DBL4();

	double diameter_x = dia_pos.i;
	double position_x = dia_pos.j;

	//center rectangle along x
	skyRect += DBL3(position_x - (skyRect.e.x + skyRect.s.x) / 2, 0.0, 0.0);

	//2. Fit along y direction - this gives us the correct y axis diameter and y center position, and also allows us to center the rectangle along y

	dia_pos = fit_y_axis_zerocrossing();

	//need these checks just in case the fitting fails
	if (dia_pos.i < 0) return DBL4();

	double diameter_y = dia_pos.i;
	double position_y = dia_pos.j;

	//center rectangle along y
	skyRect += DBL3(0.0, position_y - (skyRect.e.y + skyRect.s.y) / 2, 0.0);
	
	//3. Fitting along x direction again

	dia_pos = fit_x_axis_zerocrossing();

	//need these checks just in case the fitting fails
	if (dia_pos.i < 0) return DBL4();

	diameter_x = dia_pos.i;
	position_x = dia_pos.j;

	//center rectangle along x
	skyRect += DBL3(position_x - (skyRect.e.x + skyRect.s.x) / 2, 0.0, 0.0);

	//Update the skyrmion rectangle for next time - center it on the skyrmion with dimensions dia_mul times larger than the diameter.
	double start_x = position_x - diameter_x * dia_mul / 2;
	double start_y = position_y - diameter_y * dia_mul / 2;
	double end_x = position_x + diameter_x * dia_mul / 2;
	double end_y = position_y + diameter_y * dia_mul / 2;

	//Update the skyrmion rectangle for next time - center it on the skyrmion with dimensions 2 times larger than the diameter.
	skyTrack_rect[skyTrack_idx] = Rect(DBL3(start_x, start_y, 0.0), DBL3(end_x, end_y, M.h.z));

	return DBL4(position_x, position_y, diameter_x, diameter_y);
}

#endif