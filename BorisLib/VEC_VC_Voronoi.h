#pragma once

#include "VEC_VC.h"
#include "BLib_prng.h"

//--------------------------------------------MULTIPLE ENTRIES SETTERS - SHAPE GENERATORS : VEC_VC_genshape.h, VEC_VC_Voronoi.h

//simple brute-force Voronoi diagram generation.
//when you have time use a propoer algorithm:
//1. Steven Fortune's algorithm
//2. Bowyer-Watson algorithm for Delauney triangulation, then obtain Voronoi diagram for it
//Need to  look into a better algorithm for 3D also

//Generate 2D Voronoi cells with boundaries between cells set to empty
template <typename VType>
bool VEC_VC<VType>::generate_Voronoi2D_Grains(double spacing, unsigned seed)
{
	BorisRand prng(seed);

	//spacing determines number of voronoi cells
	int num_cells = round((VEC<VType>::rect.e.x - VEC<VType>::rect.s.x) * (VEC<VType>::rect.e.y - VEC<VType>::rect.s.y) / (spacing * spacing));

	//generate voronoi cell sites and coefficients for each cell
	std::vector<DBL2> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return false;

	//cell markers
	std::vector<int> markers;
	if (!malloc_vector(markers, VEC<VType>::n.x * VEC<VType>::n.y)) return false;

	//generate sites first
	for (int idx = 0; idx < sites.size(); idx++) {

		double x_pos = prng.rand() * VEC<VType>::rect.size().x;
		double y_pos = prng.rand() * VEC<VType>::rect.size().y;

		sites[idx] = DBL2(x_pos, y_pos);
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL2 meshcell_pos) -> int {

		double distance = GetMagnitude(VEC<VType>::rect.size().x, VEC<VType>::rect.size().y);
		int cellidx = 0;

		for (int idx = 0; idx < sites.size(); idx++) {

			double new_distance = get_distance(sites[idx], meshcell_pos);

			if (new_distance < distance) {

				distance = new_distance;
				cellidx = idx;
			}
		}

		return cellidx;
	};

	//mark cells in markers
#pragma omp parallel for
	for (int i = 0; i < VEC<VType>::n.x; i++) {
		for (int j = 0; j < VEC<VType>::n.y; j++) {

			DBL2 meshcell_pos = DBL2(i + 0.5, j + 0.5) & DBL2(VEC<VType>::h.x, VEC<VType>::h.y);

			int cellidx = find_voronoi_cellidx(meshcell_pos);

			markers[i + j * VEC<VType>::n.x] = cellidx;
		}
	}

	//now pass over cells in mesh and detect Voronoi cell boundaries - mark them empty
	for (int i = 0; i < VEC<VType>::n.x; i++) {
		for (int j = 0; j < VEC<VType>::n.y; j++) {

			int idx = i + j * VEC<VType>::n.x;

			bool cell_empty = false;
			int cell_marker = markers[idx];

			if ((ngbrFlags[idx] & NF_NPX) && !cell_empty) { if (markers[(i + 1) + j * VEC<VType>::n.x] != cell_marker && ngbrFlags[(i + 1) + j * VEC<VType>::n.x] & NF_NOTEMPTY) cell_empty = true; }
			if ((ngbrFlags[idx] & NF_NNX) && !cell_empty) { if (markers[(i - 1) + j * VEC<VType>::n.x] != cell_marker && ngbrFlags[(i - 1) + j * VEC<VType>::n.x] & NF_NOTEMPTY) cell_empty = true; }

			if ((ngbrFlags[idx] & NF_NPY) && !cell_empty) { if (markers[i + (j + 1) * VEC<VType>::n.x] != cell_marker && ngbrFlags[i + (j + 1) * VEC<VType>::n.x] & NF_NOTEMPTY) cell_empty = true; }
			if ((ngbrFlags[idx] & NF_NNY) && !cell_empty) { if (markers[i + (j - 1) * VEC<VType>::n.x] != cell_marker && ngbrFlags[i + (j - 1) * VEC<VType>::n.x] & NF_NOTEMPTY) cell_empty = true; }

			//detected boundary : mark empty cells; it's 2D so mark all z direction cells
			if (cell_empty) {

				for (int k = 0; k < VEC<VType>::n.z; k++) {

					mark_empty(i + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y);
				}
			}
		}
	}

	set_ngbrFlags();

	return true;
}

//Generate 3D Voronoi cells with boundaries between cells set to empty
template <typename VType>
bool VEC_VC<VType>::generate_Voronoi3D_Grains(double spacing, unsigned seed)
{
	BorisRand prng(seed);

	//spacing determines number of voronoi cells
	int num_cells = round((VEC<VType>::rect.e.x - VEC<VType>::rect.s.x) * (VEC<VType>::rect.e.y - VEC<VType>::rect.s.y) * (VEC<VType>::rect.e.z - VEC<VType>::rect.s.z) / (spacing * spacing * spacing));

	//generate voronoi cell sites and coefficients for each cell
	std::vector<DBL3> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return false;

	//cell markers
	std::vector<int> markers;
	if (!malloc_vector(markers, VEC<VType>::n.dim())) return false;

	//generate sites first
	for (int idx = 0; idx < sites.size(); idx++) {

		double x_pos = prng.rand() * VEC<VType>::rect.size().x;
		double y_pos = prng.rand() * VEC<VType>::rect.size().y;
		double z_pos = prng.rand() * VEC<VType>::rect.size().z;

		sites[idx] = DBL3(x_pos, y_pos, z_pos);
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL3 meshcell_pos) -> int {

		double distance = GetMagnitude(VEC<VType>::rect.size().x, VEC<VType>::rect.size().y);
		int cellidx = 0;

		for (int idx = 0; idx < sites.size(); idx++) {

			double new_distance = get_distance(sites[idx], meshcell_pos);

			if (new_distance < distance) {

				distance = new_distance;
				cellidx = idx;
			}
		}

		return cellidx;
	};

	//mark cells in markers
#pragma omp parallel for
	for (int i = 0; i < VEC<VType>::n.x; i++) {
		for (int j = 0; j < VEC<VType>::n.y; j++) {
			for (int k = 0; k < VEC<VType>::n.z; k++) {

				DBL3 meshcell_pos = DBL3(i + 0.5, j + 0.5, k + 0.5) & VEC<VType>::h;

				int cellidx = find_voronoi_cellidx(meshcell_pos);

				markers[i + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] = cellidx;
			}
		}
	}

	//now pass over cells in mesh and detect Voronoi cell boundaries - mark them empty
	for (int i = 0; i < VEC<VType>::n.x; i++) {
		for (int j = 0; j < VEC<VType>::n.y; j++) {
			for (int k = 0; k < VEC<VType>::n.z; k++) {

				int idx = i + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y;

				bool cell_empty = false;
				int cell_marker = markers[idx];

				if ((ngbrFlags[idx] & NF_NPX) && !cell_empty) {

					if (markers[(i + 1) + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[(i + 1) + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				if ((ngbrFlags[idx] & NF_NNX) && !cell_empty) {

					if (markers[(i - 1) + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[(i - 1) + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				if ((ngbrFlags[idx] & NF_NPY) && !cell_empty) {

					if (markers[i + (j + 1) * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[i + (j + 1) * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				if ((ngbrFlags[idx] & NF_NNY) && !cell_empty) {

					if (markers[i + (j - 1) * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[i + (j - 1) * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				if ((ngbrFlags[idx] & NF_NPZ) && !cell_empty) {

					if (markers[i + j * VEC<VType>::n.x + (k + 1) * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[i + j * VEC<VType>::n.x + (k + 1) * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				if ((ngbrFlags[idx] & NF_NNZ) && !cell_empty) {

					if (markers[i + j * VEC<VType>::n.x + (k - 1) * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[i + j * VEC<VType>::n.x + (k - 1) * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				//detected boundary : mark empty cell
				if (cell_empty) {

					mark_empty(idx);
				}
			}
		}
	}

	set_ngbrFlags();

	return true;
}

//Generate uniform 2D Voronoi cells with boundaries between cells set to empty
//There's an additional variation parameter (same units as spacing) which controls grain uniformity (e.g. if zero they are all square)
template <typename VType>
bool VEC_VC<VType>::generate_uVoronoi2D_Grains(double spacing, double variation, unsigned seed)
{
	BorisRand prng(seed);

	//calculate number of cells along x and y, and add extra outside of the mesh (so we can get a granular edge effect)
	int num_cells_x = floor((VEC<VType>::rect.e.x - VEC<VType>::rect.s.x) / spacing) + 3;
	int num_cells_y = floor((VEC<VType>::rect.e.y - VEC<VType>::rect.s.y) / spacing) + 3;
	int num_cells = num_cells_x * num_cells_y;

	//generate voronoi cell sites and coefficients for each cell
	std::vector<DBL2> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return false;

	//cell markers
	std::vector<int> markers;
	if (!malloc_vector(markers, VEC<VType>::n.x * VEC<VType>::n.y)) return false;

	//generate sites first on regular grid, but with some random variation controlled by variation parameter
	for (int i = 0; i < num_cells_x; i++) {
		for (int j = 0; j < num_cells_y; j++) {

			double start_x = spacing / 2 - (num_cells_x * spacing - (VEC<VType>::rect.e.x - VEC<VType>::rect.s.x)) / 2;
			double start_y = spacing / 2 - (num_cells_y * spacing - (VEC<VType>::rect.e.y - VEC<VType>::rect.s.y)) / 2;

			double x_pos = start_x + spacing * i + (-0.5 + prng.rand()) * variation;
			double y_pos = start_y + spacing * j + (-0.5 + prng.rand()) * variation;

			sites[i + j * num_cells_x] = DBL2(x_pos, y_pos);
		}
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL2 meshcell_pos) -> int {

		double distance = GetMagnitude(VEC<VType>::rect.size().x, VEC<VType>::rect.size().y);
		int cellidx = 0;

		for (int idx = 0; idx < sites.size(); idx++) {

			double new_distance = get_distance(sites[idx], meshcell_pos);

			if (new_distance < distance) {

				distance = new_distance;
				cellidx = idx;
			}
		}

		return cellidx;
	};

	//mark cells in markers
#pragma omp parallel for
	for (int i = 0; i < VEC<VType>::n.x; i++) {
		for (int j = 0; j < VEC<VType>::n.y; j++) {

			DBL2 meshcell_pos = DBL2(i + 0.5, j + 0.5) & DBL2(VEC<VType>::h.x, VEC<VType>::h.y);

			int cellidx = find_voronoi_cellidx(meshcell_pos);

			markers[i + j * VEC<VType>::n.x] = cellidx;
		}
	}

	//now pass over cells in mesh and detect Voronoi cell boundaries - mark them empty
	for (int i = 0; i < VEC<VType>::n.x; i++) {
		for (int j = 0; j < VEC<VType>::n.y; j++) {

			int idx = i + j * VEC<VType>::n.x;

			bool cell_empty = false;
			int cell_marker = markers[idx];

			//cells which belong to voronoi sites outside of the mesh will be marked as empty (so we get granular edges effect)
			DBL2 site_xy_pos = sites[cell_marker];
			DBL3 site_abs_pos = DBL3(site_xy_pos.x, site_xy_pos.y, (VEC<VType>::rect.e.z - VEC<VType>::rect.s.z) / 2) + VEC<VType>::rect.s;
			if (!VEC<VType>::rect.contains(site_abs_pos)) cell_empty = true;

			if ((ngbrFlags[idx] & NF_NPX) && !cell_empty) { if (markers[(i + 1) + j * VEC<VType>::n.x] != cell_marker && ngbrFlags[(i + 1) + j * VEC<VType>::n.x] & NF_NOTEMPTY) cell_empty = true; }
			if ((ngbrFlags[idx] & NF_NNX) && !cell_empty) { if (markers[(i - 1) + j * VEC<VType>::n.x] != cell_marker && ngbrFlags[(i - 1) + j * VEC<VType>::n.x] & NF_NOTEMPTY) cell_empty = true; }

			if ((ngbrFlags[idx] & NF_NPY) && !cell_empty) { if (markers[i + (j + 1) * VEC<VType>::n.x] != cell_marker && ngbrFlags[i + (j + 1) * VEC<VType>::n.x] & NF_NOTEMPTY) cell_empty = true; }
			if ((ngbrFlags[idx] & NF_NNY) && !cell_empty) { if (markers[i + (j - 1) * VEC<VType>::n.x] != cell_marker && ngbrFlags[i + (j - 1) * VEC<VType>::n.x] & NF_NOTEMPTY) cell_empty = true; }

			//detected boundary : mark empty cells; it's 2D so mark all z direction cells
			if (cell_empty) {

				for (int k = 0; k < VEC<VType>::n.z; k++) {

					mark_empty(i + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y);
				}
			}
		}
	}

	set_ngbrFlags();

	return true;
}

//Generate uniform 3D Voronoi cells with boundaries between cells set to empty
//There's an additional variation parameter (same units as spacing) which controls grain uniformity (e.g. if zero they are all square)
template <typename VType>
bool VEC_VC<VType>::generate_uVoronoi3D_Grains(double spacing, double variation, unsigned seed)
{
	BorisRand prng(seed);

	//calculate number of cells along x and y, and add extra outside of the mesh (so we can get a granular edge effect)
	int num_cells_x = floor((VEC<VType>::rect.e.x - VEC<VType>::rect.s.x) / spacing) + 3;
	int num_cells_y = floor((VEC<VType>::rect.e.y - VEC<VType>::rect.s.y) / spacing) + 3;
	int num_cells_z = floor((VEC<VType>::rect.e.z - VEC<VType>::rect.s.z) / spacing) + 3;
	int num_cells = num_cells_x * num_cells_y * num_cells_z;

	//generate voronoi cell sites and coefficients for each cell
	std::vector<DBL3> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return false;

	//cell markers
	std::vector<int> markers;
	if (!malloc_vector(markers, VEC<VType>::n.dim())) return false;

	//generate sites first on regular grid, but with some random variation controlled by variation parameter
	for (int i = 0; i < num_cells_x; i++) {
		for (int j = 0; j < num_cells_y; j++) {
			for (int k = 0; k < num_cells_z; k++) {

				double start_x = spacing / 2 - (num_cells_x * spacing - (VEC<VType>::rect.e.x - VEC<VType>::rect.s.x)) / 2;
				double start_y = spacing / 2 - (num_cells_y * spacing - (VEC<VType>::rect.e.y - VEC<VType>::rect.s.y)) / 2;
				double start_z = spacing / 2 - (num_cells_z * spacing - (VEC<VType>::rect.e.z - VEC<VType>::rect.s.z)) / 2;

				double x_pos = start_x + spacing * i + (-0.5 + prng.rand()) * variation;
				double y_pos = start_y + spacing * j + (-0.5 + prng.rand()) * variation;
				double z_pos = start_z + spacing * k + (-0.5 + prng.rand()) * variation;

				sites[i + j * num_cells_x + k * num_cells_x * num_cells_y] = DBL3(x_pos, y_pos, z_pos);
			}
		}
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL3 meshcell_pos) -> int {

		double distance = GetMagnitude(VEC<VType>::rect.size().x, VEC<VType>::rect.size().y);
		int cellidx = 0;

		for (int idx = 0; idx < sites.size(); idx++) {

			double new_distance = get_distance(sites[idx], meshcell_pos);

			if (new_distance < distance) {

				distance = new_distance;
				cellidx = idx;
			}
		}

		return cellidx;
	};

	//mark cells in markers
#pragma omp parallel for
	for (int i = 0; i < VEC<VType>::n.x; i++) {
		for (int j = 0; j < VEC<VType>::n.y; j++) {
			for (int k = 0; k < VEC<VType>::n.z; k++) {

				DBL3 meshcell_pos = DBL3(i + 0.5, j + 0.5, k + 0.5) & VEC<VType>::h;

				int cellidx = find_voronoi_cellidx(meshcell_pos);

				markers[i + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] = cellidx;
			}
		}
	}

	//now pass over cells in mesh and detect Voronoi cell boundaries - mark them empty
	for (int i = 0; i < VEC<VType>::n.x; i++) {
		for (int j = 0; j < VEC<VType>::n.y; j++) {
			for (int k = 0; k < VEC<VType>::n.z; k++) {

				int idx = i + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y;

				bool cell_empty = false;
				int cell_marker = markers[idx];

				//cells which belong to voronoi sites outside of the mesh will be marked as empty (so we get granular edges effect)
				DBL3 site_abs_pos = sites[cell_marker] + VEC<VType>::rect.s;
				if (!VEC<VType>::rect.contains(site_abs_pos)) cell_empty = true;

				if ((ngbrFlags[idx] & NF_NPX) && !cell_empty) {

					if (markers[(i + 1) + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[(i + 1) + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				if ((ngbrFlags[idx] & NF_NNX) && !cell_empty) {

					if (markers[(i - 1) + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[(i - 1) + j * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				if ((ngbrFlags[idx] & NF_NPY) && !cell_empty) {

					if (markers[i + (j + 1) * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[i + (j + 1) * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				if ((ngbrFlags[idx] & NF_NNY) && !cell_empty) {

					if (markers[i + (j - 1) * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[i + (j - 1) * VEC<VType>::n.x + k * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				if ((ngbrFlags[idx] & NF_NPZ) && !cell_empty) {

					if (markers[i + j * VEC<VType>::n.x + (k + 1) * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[i + j * VEC<VType>::n.x + (k + 1) * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				if ((ngbrFlags[idx] & NF_NNZ) && !cell_empty) {

					if (markers[i + j * VEC<VType>::n.x + (k - 1) * VEC<VType>::n.x * VEC<VType>::n.y] != cell_marker && ngbrFlags[i + j * VEC<VType>::n.x + (k - 1) * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY)
						cell_empty = true;
				}

				//detected boundary : mark empty cell
				if (cell_empty) {

					mark_empty(idx);
				}
			}
		}
	}

	set_ngbrFlags();

	return true;
}