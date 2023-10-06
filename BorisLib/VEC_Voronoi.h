#pragma once

#include "VEC.h"
#include "BLib_prng.h"

//simple brute-force Voronoi diagram generation.
//when you have time use a proper algorithm:
//1. Steven Fortune's algorithm
//2. Bowyer-Watson algorithm for Delaunay triangulation, then obtain Voronoi diagram from it
//Need to  look into a better algorithm for 3D also

//----------------------------------------------------- GENERATE 2D

template <typename VType>
void VEC<VType>::aux_generate_Voronoi2D(double spacing, BorisRand& prng, std::function<VType(void)>& value_generator)
{
	//spacing determines number of voronoi cells
	int num_cells = round((rect.e.x - rect.s.x) * (rect.e.y - rect.s.y) / (spacing * spacing));

	//generate voronoi cell sites and coefficients for each cell
	std::vector<std::pair<DBL2, VType>> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return;

	//generate sites first
	for (int idx = 0; idx < sites.size(); idx++) {

		double x_pos = prng.rand() * rect.size().x;
		double y_pos = prng.rand() * rect.size().y;

		sites[idx] = std::pair<DBL2, VType>(DBL2(x_pos, y_pos), VType());
	}

	//now generate coefficients for each cell - better to separate it from the sites generation so you can use the same prng seed to generate the same topographical Voronoi cells
	for (int idx = 0; idx < sites.size(); idx++) {

		sites[idx].second = value_generator();
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL2 meshcell_pos) -> int {

		double distance = GetMagnitude(rect.size().x, rect.size().y);
		int cellidx = 0;

		for (int idx = 0; idx < sites.size(); idx++) {

			double new_distance = get_distance(sites[idx].first, meshcell_pos);

			if (new_distance < distance) {

				distance = new_distance;
				cellidx = idx;
			}
		}

		return cellidx;
	};

#pragma omp parallel for
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {

			DBL2 meshcell_pos = DBL2(i + 0.5, j + 0.5) & DBL2(h.x, h.y);

			int cellidx = find_voronoi_cellidx(meshcell_pos);

			quantity[i + j * n.x] = sites[cellidx].second;
		}
	}
}

template <typename VType>
void VEC<VType>::aux_generate_uVoronoi2D(double spacing, double variation, BorisRand& prng, std::function<VType(void)>& value_generator)
{
	//calculate number of cells along x and y, and add extra outside of the mesh (so we can get a granular edge effect)
	int num_cells_x = floor((rect.e.x - rect.s.x) / spacing) + 3;
	int num_cells_y = floor((rect.e.y - rect.s.y) / spacing) + 3;
	int num_cells = num_cells_x * num_cells_y;

	//generate voronoi cell sites and coefficients for each cell
	std::vector<std::pair<DBL2, VType>> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return;

	//generate sites first on regular grid, but with some random variation controlled by variation parameter
	for (int i = 0; i < num_cells_x; i++) {
		for (int j = 0; j < num_cells_y; j++) {

			double start_x = spacing / 2 - (num_cells_x * spacing - (rect.e.x - rect.s.x)) / 2;
			double start_y = spacing / 2 - (num_cells_y * spacing - (rect.e.y - rect.s.y)) / 2;

			double x_pos = start_x + spacing * i + (-0.5 + prng.rand()) * variation;
			double y_pos = start_y + spacing * j + (-0.5 + prng.rand()) * variation;

			sites[i + j * num_cells_x] = std::pair<DBL2, VType>(DBL2(x_pos, y_pos), VType());
		}
	}

	//now generate coefficients for each cell - better to separate it from the sites generation so you can use the same prng seed to generate the same topographical Voronoi cells
	for (int idx = 0; idx < sites.size(); idx++) {

		sites[idx].second = value_generator();
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL2 meshcell_pos) -> int {

		double distance = GetMagnitude(rect.size().x, rect.size().y);
		int cellidx = 0;

		for (int idx = 0; idx < sites.size(); idx++) {

			double new_distance = get_distance(sites[idx].first, meshcell_pos);

			if (new_distance < distance) {

				distance = new_distance;
				cellidx = idx;
			}
		}

		return cellidx;
	};

#pragma omp parallel for
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {

			DBL2 meshcell_pos = DBL2(i + 0.5, j + 0.5) & DBL2(h.x, h.y);

			int cellidx = find_voronoi_cellidx(meshcell_pos);

			quantity[i + j * n.x] = sites[cellidx].second;
		}
	}
}

//----------------------------------------------------- GENERATE 3D

template <typename VType>
void VEC<VType>::aux_generate_Voronoi3D(double spacing, BorisRand& prng, std::function<VType(void)>& value_generator)
{
	//spacing determines number of voronoi cells
	int num_cells = round((rect.e.x - rect.s.x) * (rect.e.y - rect.s.y) * (rect.e.z - rect.s.z) / (spacing * spacing * spacing));

	//generate voronoi cell sites and coefficients for each cell
	std::vector<std::pair<DBL3, VType>> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return;

	//generate sites first
	for (int idx = 0; idx < sites.size(); idx++) {

		double x_pos = prng.rand() * rect.size().x;
		double y_pos = prng.rand() * rect.size().y;
		double z_pos = prng.rand() * rect.size().z;

		sites[idx] = std::pair<DBL3, VType>(DBL3(x_pos, y_pos, z_pos), VType());
	}

	//now generate coefficients for each cell - better to separate it from the sites generation so you can use the same prng seed to generate the same topographical Voronoi cells
	for (int idx = 0; idx < sites.size(); idx++) {

		sites[idx].second = value_generator();
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL3 meshcell_pos) -> int {

		double distance = GetMagnitude(rect.size().x, rect.size().y, rect.size().z);
		int cellidx = 0;

		for (int idx = 0; idx < sites.size(); idx++) {

			double new_distance = get_distance(sites[idx].first, meshcell_pos);

			if (new_distance < distance) {

				distance = new_distance;
				cellidx = idx;
			}
		}

		return cellidx;
	};

#pragma omp parallel for
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {
			for (int k = 0; k < n.z; k++) {

				DBL3 meshcell_pos = DBL3(i + 0.5, j + 0.5, k + 0.5) & h;

				int cellidx = find_voronoi_cellidx(meshcell_pos);

				quantity[i + j * n.x + k * n.x * n.y] = sites[cellidx].second;
			}
		}
	}
}

template <typename VType>
void VEC<VType>::aux_generate_uVoronoi3D(double spacing, double variation, BorisRand& prng, std::function<VType(void)>& value_generator)
{
	//calculate number of cells along x and y, and add extra outside of the mesh (so we can get a granular edge effect)
	int num_cells_x = floor((rect.e.x - rect.s.x) / spacing) + 3;
	int num_cells_y = floor((rect.e.y - rect.s.y) / spacing) + 3;
	int num_cells_z = floor((rect.e.z - rect.s.z) / spacing) + 3;
	int num_cells = num_cells_x * num_cells_y * num_cells_z;

	//generate voronoi cell sites and coefficients for each cell
	std::vector<std::pair<DBL3, VType>> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return;

	//generate sites first on regular grid, but with some random variation controlled by variation parameter
	for (int i = 0; i < num_cells_x; i++) {
		for (int j = 0; j < num_cells_y; j++) {
			for (int k = 0; k < num_cells_z; k++) {

				double start_x = spacing / 2 - (num_cells_x * spacing - (rect.e.x - rect.s.x)) / 2;
				double start_y = spacing / 2 - (num_cells_y * spacing - (rect.e.y - rect.s.y)) / 2;
				double start_z = spacing / 2 - (num_cells_z * spacing - (rect.e.z - rect.s.z)) / 2;

				double x_pos = start_x + spacing * i + (-0.5 + prng.rand()) * variation;
				double y_pos = start_y + spacing * j + (-0.5 + prng.rand()) * variation;
				double z_pos = start_z + spacing * k + (-0.5 + prng.rand()) * variation;

				sites[i + j * num_cells_x + k * num_cells_x * num_cells_y] = std::pair<DBL3, VType>(DBL3(x_pos, y_pos, z_pos), VType());
			}
		}
	}

	//now generate coefficients for each cell - better to separate it from the sites generation so you can use the same prng seed to generate the same topographical Voronoi cells
	for (int idx = 0; idx < sites.size(); idx++) {

		sites[idx].second = value_generator();
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL3 meshcell_pos) -> int {

		double distance = GetMagnitude(rect.size().x, rect.size().y, rect.size().z);
		int cellidx = 0;

		for (int idx = 0; idx < sites.size(); idx++) {

			double new_distance = get_distance(sites[idx].first, meshcell_pos);

			if (new_distance < distance) {

				distance = new_distance;
				cellidx = idx;
			}
		}

		return cellidx;
	};

#pragma omp parallel for
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {
			for (int k = 0; k < n.z; k++) {

				DBL3 meshcell_pos = DBL3(i + 0.5, j + 0.5, k + 0.5) & h;

				int cellidx = find_voronoi_cellidx(meshcell_pos);

				quantity[i + j * n.x + k * n.x * n.y] = sites[cellidx].second;
			}
		}
	}
}

//----------------------------------------------------- GENERATE 2D BOUNDARY

template <typename VType>
void VEC<VType>::aux_generate_VoronoiBoundary2D(double spacing, VType base_value, BorisRand& prng, std::function<VType(void)>& value_generator)
{
	//spacing determines number of voronoi cells
	int num_cells = round((rect.e.x - rect.s.x) * (rect.e.y - rect.s.y) / (spacing * spacing));

	//generate voronoi cell sites and coefficients for each cell
	std::vector<DBL2> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return;

	//cell markers
	std::vector<int> markers;
	if (!malloc_vector(markers, n.x * n.y)) return;

	//generate sites first
	for (int idx = 0; idx < sites.size(); idx++) {

		double x_pos = prng.rand() * rect.size().x;
		double y_pos = prng.rand() * rect.size().y;

		sites[idx] = DBL2(x_pos, y_pos);
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL2 meshcell_pos) -> int {

		double distance = GetMagnitude(rect.size().x, rect.size().y);
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
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {

			DBL2 meshcell_pos = DBL2(i + 0.5, j + 0.5) & DBL2(h.x, h.y);

			int cellidx = find_voronoi_cellidx(meshcell_pos);

			markers[i + j * n.x] = cellidx;
		}
	}

	set(base_value);

	//now pass over cells in mesh and detect Voronoi cell boundaries
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {

			int idx = i + j * n.x;

			bool boundary_cell = false;
			int cell_marker = markers[idx];

			if (!boundary_cell && markers[idx] != -1 && i + 1 < n.x) { if (markers[(i + 1) + j * n.x] != cell_marker && markers[(i + 1) + j * n.x] != -1) boundary_cell = true; }
			if (!boundary_cell && markers[idx] != -1 && i > 0) { if (markers[(i - 1) + j * n.x] != cell_marker && markers[(i - 1) + j * n.x] != -1) boundary_cell = true; }

			if (!boundary_cell && markers[idx] != -1 && j + 1 < n.y) { if (markers[i + (j + 1) * n.x] != cell_marker && markers[i + (j + 1) * n.x] != -1) boundary_cell = true; }
			if (!boundary_cell && markers[idx] != -1 && j > 0) { if (markers[i + (j - 1) * n.x] != cell_marker && markers[i + (j - 1) * n.x] != -1) boundary_cell = true; }

			//detected boundary
			if (boundary_cell) {

				VType value = value_generator();

				for (int k = 0; k < n.z; k++) {

					markers[i + j * n.x + k * n.x * n.y] = -1;
					quantity[i + j * n.x + k * n.x * n.y] = value;
				}
			}
		}
	}
}

template <typename VType>
void VEC<VType>::aux_generate_uVoronoiBoundary2D(double spacing, double variation, VType base_value, BorisRand& prng, std::function<VType(void)>& value_generator)
{
	//calculate number of cells along x and y, and add extra outside of the mesh (so we can get a granular edge effect)
	int num_cells_x = floor((rect.e.x - rect.s.x) / spacing) + 3;
	int num_cells_y = floor((rect.e.y - rect.s.y) / spacing) + 3;
	int num_cells = num_cells_x * num_cells_y;

	//generate voronoi cell sites and coefficients for each cell
	std::vector<DBL2> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return;

	//cell markers
	std::vector<int> markers;
	if (!malloc_vector(markers, n.x * n.y)) return;

	//generate sites first on regular grid, but with some random variation controlled by variation parameter
	for (int i = 0; i < num_cells_x; i++) {
		for (int j = 0; j < num_cells_y; j++) {

			double start_x = spacing / 2 - (num_cells_x * spacing - (rect.e.x - rect.s.x)) / 2;
			double start_y = spacing / 2 - (num_cells_y * spacing - (rect.e.y - rect.s.y)) / 2;

			double x_pos = start_x + spacing * i + (-0.5 + prng.rand()) * variation;
			double y_pos = start_y + spacing * j + (-0.5 + prng.rand()) * variation;

			sites[i + j * num_cells_x] = DBL2(x_pos, y_pos);
		}
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL2 meshcell_pos) -> int {

		double distance = GetMagnitude(rect.size().x, rect.size().y);
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
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {

			DBL2 meshcell_pos = DBL2(i + 0.5, j + 0.5) & DBL2(h.x, h.y);

			int cellidx = find_voronoi_cellidx(meshcell_pos);

			markers[i + j * n.x] = cellidx;
		}
	}

	set(base_value);

	//now pass over cells in mesh and detect Voronoi cell boundaries
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {

			int idx = i + j * n.x;

			bool boundary_cell = false;
			int cell_marker = markers[idx];

			if (!boundary_cell && markers[idx] != -1 && i + 1 < n.x) { if (markers[(i + 1) + j * n.x] != cell_marker && markers[(i + 1) + j * n.x] != -1) boundary_cell = true; }
			if (!boundary_cell && markers[idx] != -1 && i > 0) { if (markers[(i - 1) + j * n.x] != cell_marker && markers[(i - 1) + j * n.x] != -1) boundary_cell = true; }

			if (!boundary_cell && markers[idx] != -1 && j + 1 < n.y) { if (markers[i + (j + 1) * n.x] != cell_marker && markers[i + (j + 1) * n.x] != -1) boundary_cell = true; }
			if (!boundary_cell && markers[idx] != -1 && j > 0) { if (markers[i + (j - 1) * n.x] != cell_marker && markers[i + (j - 1) * n.x] != -1) boundary_cell = true; }

			//detected boundary
			if (boundary_cell) {

				VType value = value_generator();

				for (int k = 0; k < n.z; k++) {

					markers[i + j * n.x + k * n.x * n.y] = -1;
					quantity[i + j * n.x + k * n.x * n.y] = value;
				}
			}
		}
	}
}

//----------------------------------------------------- GENERATE 3D BOUNDARY

template <typename VType>
void VEC<VType>::aux_generate_VoronoiBoundary3D(double spacing, VType base_value, BorisRand& prng, std::function<VType(void)>& value_generator)
{
	//spacing determines number of voronoi cells
	int num_cells = round((rect.e.x - rect.s.x) * (rect.e.y - rect.s.y) * (rect.e.z - rect.s.z) / (spacing * spacing * spacing));

	//generate voronoi cell sites and coefficients for each cell
	std::vector<DBL3> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return;

	//cell markers
	std::vector<int> markers;
	if (!malloc_vector(markers, n.dim())) return;

	//generate sites first
	for (int idx = 0; idx < sites.size(); idx++) {

		double x_pos = prng.rand() * rect.size().x;
		double y_pos = prng.rand() * rect.size().y;
		double z_pos = prng.rand() * rect.size().z;

		sites[idx] = DBL3(x_pos, y_pos, z_pos);
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL3 meshcell_pos) -> int {

		double distance = GetMagnitude(rect.size().x, rect.size().y);
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
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {
			for (int k = 0; k < n.z; k++) {

				DBL3 meshcell_pos = DBL3(i + 0.5, j + 0.5, k + 0.5) & h;

				int cellidx = find_voronoi_cellidx(meshcell_pos);

				markers[i + j * n.x + k * n.x * n.y] = cellidx;
			}
		}
	}

	set(base_value);

	//now pass over cells in mesh and detect Voronoi cell boundaries
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {
			for (int k = 0; k < n.z; k++) {

				int idx = i + j * n.x + k * n.x * n.y;

				bool boundary_cell = false;
				int cell_marker = markers[idx];

				if (!boundary_cell && markers[idx] != -1 && i + 1 < n.x) { if (markers[(i + 1) + j * n.x + k * n.x * n.y] != cell_marker && markers[(i + 1) + j * n.x + k * n.x * n.y] != -1) boundary_cell = true; }
				if (!boundary_cell && markers[idx] != -1 && i > 0) { if (markers[(i - 1) + j * n.x + k * n.x * n.y] != cell_marker && markers[(i - 1) + j * n.x + k * n.x * n.y] != -1) boundary_cell = true; }

				if (!boundary_cell && markers[idx] != -1 && j + 1 < n.y) { if (markers[i + (j + 1) * n.x + k * n.x * n.y] != cell_marker && markers[i + (j + 1) * n.x + k * n.x * n.y] != -1) boundary_cell = true; }
				if (!boundary_cell && markers[idx] != -1 && j > 0) { if (markers[i + (j - 1) * n.x + k * n.x * n.y] != cell_marker && markers[i + (j - 1) * n.x + k * n.x * n.y] != -1) boundary_cell = true; }

				if (!boundary_cell && markers[idx] != -1 && k + 1 < n.z) { if (markers[i + j * n.x + (k + 1) * n.x * n.y] != cell_marker && markers[i + j * n.x + (k + 1) * n.x * n.y] != -1) boundary_cell = true; }
				if (!boundary_cell && markers[idx] != -1 && k > 0) { if (markers[i + j * n.x + (k - 1) * n.x * n.y] != cell_marker && markers[i + j * n.x + (k - 1) * n.x * n.y] != -1) boundary_cell = true; }

				//detected boundary : mark empty cell
				if (boundary_cell) {

					markers[idx] = -1;

					quantity[idx] = value_generator();
				}
			}
		}
	}
}

template <typename VType>
void VEC<VType>::aux_generate_uVoronoiBoundary3D(double spacing, double variation, VType base_value, BorisRand& prng, std::function<VType(void)>& value_generator)
{
	//calculate number of cells along x and y, and add extra outside of the mesh (so we can get a granular edge effect)
	int num_cells_x = floor((rect.e.x - rect.s.x) / spacing) + 3;
	int num_cells_y = floor((rect.e.y - rect.s.y) / spacing) + 3;
	int num_cells_z = floor((rect.e.z - rect.s.z) / spacing) + 3;
	int num_cells = num_cells_x * num_cells_y * num_cells_z;

	//generate voronoi cell sites and coefficients for each cell
	std::vector<DBL3> sites;
	if (!num_cells || !malloc_vector(sites, num_cells)) return;

	//cell markers
	std::vector<int> markers;
	if (!malloc_vector(markers, n.dim())) return;

	//generate sites first on regular grid, but with some random variation controlled by variation parameter
	for (int i = 0; i < num_cells_x; i++) {
		for (int j = 0; j < num_cells_y; j++) {
			for (int k = 0; k < num_cells_z; k++) {

				double start_x = spacing / 2 - (num_cells_x * spacing - (rect.e.x - rect.s.x)) / 2;
				double start_y = spacing / 2 - (num_cells_y * spacing - (rect.e.y - rect.s.y)) / 2;
				double start_z = spacing / 2 - (num_cells_z * spacing - (rect.e.z - rect.s.z)) / 2;

				double x_pos = start_x + spacing * i + (-0.5 + prng.rand()) * variation;
				double y_pos = start_y + spacing * j + (-0.5 + prng.rand()) * variation;
				double z_pos = start_z + spacing * k + (-0.5 + prng.rand()) * variation;

				sites[i + j * num_cells_x + k * num_cells_x * num_cells_y] = DBL3(x_pos, y_pos, z_pos);
			}
		}
	}

	//naive method : for each mesh cell just search the sites vector to find coefficient to apply
	auto find_voronoi_cellidx = [&](DBL3 meshcell_pos) -> int {

		double distance = GetMagnitude(rect.size().x, rect.size().y);
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
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {
			for (int k = 0; k < n.z; k++) {

				DBL3 meshcell_pos = DBL3(i + 0.5, j + 0.5, k + 0.5) & h;

				int cellidx = find_voronoi_cellidx(meshcell_pos);

				markers[i + j * n.x + k * n.x * n.y] = cellidx;
			}
		}
	}

	set(base_value);

	//now pass over cells in mesh and detect Voronoi cell boundaries
	for (int i = 0; i < n.x; i++) {
		for (int j = 0; j < n.y; j++) {
			for (int k = 0; k < n.z; k++) {

				int idx = i + j * n.x + k * n.x * n.y;

				bool boundary_cell = false;
				int cell_marker = markers[idx];

				if (!boundary_cell && markers[idx] != -1 && i + 1 < n.x) { if (markers[(i + 1) + j * n.x + k * n.x * n.y] != cell_marker && markers[(i + 1) + j * n.x + k * n.x * n.y] != -1) boundary_cell = true; }
				if (!boundary_cell && markers[idx] != -1 && i > 0) { if (markers[(i - 1) + j * n.x + k * n.x * n.y] != cell_marker && markers[(i - 1) + j * n.x + k * n.x * n.y] != -1) boundary_cell = true; }

				if (!boundary_cell && markers[idx] != -1 && j + 1 < n.y) { if (markers[i + (j + 1) * n.x + k * n.x * n.y] != cell_marker && markers[i + (j + 1) * n.x + k * n.x * n.y] != -1) boundary_cell = true; }
				if (!boundary_cell && markers[idx] != -1 && j > 0) { if (markers[i + (j - 1) * n.x + k * n.x * n.y] != cell_marker && markers[i + (j - 1) * n.x + k * n.x * n.y] != -1) boundary_cell = true; }

				if (!boundary_cell && markers[idx] != -1 && k + 1 < n.z) { if (markers[i + j * n.x + (k + 1) * n.x * n.y] != cell_marker && markers[i + j * n.x + (k + 1) * n.x * n.y] != -1) boundary_cell = true; }
				if (!boundary_cell && markers[idx] != -1 && k > 0) { if (markers[i + j * n.x + (k - 1) * n.x * n.y] != cell_marker && markers[i + j * n.x + (k - 1) * n.x * n.y] != -1) boundary_cell = true; }

				//detected boundary : mark empty cell
				if (boundary_cell) {

					markers[idx] = -1;

					quantity[idx] = value_generator();
				}
			}
		}
	}
}

//----------------------------------------------------- GENERATOR METHODS

//voronoi 2D: set VEC dimensions (force 2D in xy plane) and generate random values in the given range with each value fixed in a voronoi cell, and average spacing (prng instantiated with given seed).
template <typename VType>
bool VEC<VType>::generate_Voronoi2D(DBL3 new_h, Rect new_rect, DBL2 range, double spacing, unsigned seed)
{
	//force 2D in xy plane
	new_h.k = new_rect.size().k;

	if (!resize(new_h, new_rect) || spacing <= 0) return false;

	BorisRand prng(seed);

	std::function<VType(void)> value_generator = [&](void) -> VType {

		return prng.rand() * (range.j - range.i) + range.i;
	};

	aux_generate_Voronoi2D(spacing, prng, value_generator);

	return true;
}

//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
template <typename VType>
bool VEC<VType>::generate_uVoronoi2D(DBL3 new_h, Rect new_rect, DBL2 range, double spacing, double variation, unsigned seed)
{
	//force 2D in xy plane
	new_h.k = new_rect.size().k;

	if (!resize(new_h, new_rect) || spacing <= 0 || variation < 0) return false;

	BorisRand prng(seed);

	std::function<VType(void)> value_generator = [&](void) -> VType {

		return prng.rand() * (range.j - range.i) + range.i;
	};

	aux_generate_uVoronoi2D(spacing, variation, prng, value_generator);

	return true;
}

//voronoi 3D: set VEC dimensions and generate random values in the given range with each value fixed in a voronoi cell, and average spacing (prng instantiated with given seed).
template <typename VType>
bool VEC<VType>::generate_Voronoi3D(DBL3 new_h, Rect new_rect, DBL2 range, double spacing, unsigned seed)
{
	if (!resize(new_h, new_rect) || spacing <= 0) return false;

	BorisRand prng(seed);

	std::function<VType(void)> value_generator = [&](void) -> VType {

		return prng.rand() * (range.j - range.i) + range.i;
	};

	aux_generate_Voronoi3D(spacing, prng, value_generator);

	return true;
}

//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
template <typename VType>
bool VEC<VType>::generate_uVoronoi3D(DBL3 new_h, Rect new_rect, DBL2 range, double spacing, double variation, unsigned seed)
{
	if (!resize(new_h, new_rect) || spacing <= 0 || variation < 0) return false;

	BorisRand prng(seed);

	std::function<VType(void)> value_generator = [&](void) -> VType {

		return prng.rand() * (range.j - range.i) + range.i;
	};

	aux_generate_uVoronoi3D(spacing, variation, prng, value_generator);

	return true;
}

//voronoi boundary 2D: set VEC dimensions (force 2D in xy plane) and generate voronoi 2d tessellation with average spacing. Set coefficient values randomly in the given range only at the Voronoi cell boundaries (prng instantiated with given seed).
template <typename VType>
bool VEC<VType>::generate_VoronoiBoundary2D(DBL3 new_h, Rect new_rect, DBL2 range, VType base_value, double spacing, unsigned seed)
{
	//force 2D in xy plane
	new_h.k = new_rect.size().k;

	if (!resize(new_h, new_rect) || spacing <= 0) return false;

	BorisRand prng(seed);

	std::function<VType(void)> value_generator = [&](void) -> VType {

		return prng.rand() * (range.j - range.i) + range.i;
	};

	aux_generate_VoronoiBoundary2D(spacing, base_value, prng, value_generator);

	return true;
}

//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
template <typename VType>
bool VEC<VType>::generate_uVoronoiBoundary2D(DBL3 new_h, Rect new_rect, DBL2 range, VType base_value, double spacing, double variation, unsigned seed)
{
	//force 2D in xy plane
	new_h.k = new_rect.size().k;

	if (!resize(new_h, new_rect) || spacing <= 0 || variation < 0) return false;

	BorisRand prng(seed);

	std::function<VType(void)> value_generator = [&](void) -> VType {

		return prng.rand() * (range.j - range.i) + range.i;
	};

	aux_generate_uVoronoiBoundary2D(spacing, variation, base_value, prng, value_generator);

	return true;
}

//voronoi boundary 3D: set VEC dimensions and generate voronoi 3d tessellation with average spacing. Set coefficient values randomly in the given range only at the Voronoi cell boundaries (prng instantiated with given seed).
template <typename VType>
bool VEC<VType>::generate_VoronoiBoundary3D(DBL3 new_h, Rect new_rect, DBL2 range, VType base_value, double spacing, unsigned seed)
{
	if (!resize(new_h, new_rect) || spacing <= 0) return false;

	BorisRand prng(seed);

	std::function<VType(void)> value_generator = [&](void) -> VType {

		return prng.rand() * (range.j - range.i) + range.i;
	};

	aux_generate_VoronoiBoundary3D(spacing, base_value, prng, value_generator);

	return true;
}

//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
template <typename VType>
bool VEC<VType>::generate_uVoronoiBoundary3D(DBL3 new_h, Rect new_rect, DBL2 range, VType base_value, double spacing, double variation, unsigned seed)
{
	if (!resize(new_h, new_rect) || spacing <= 0 || variation < 0) return false;

	BorisRand prng(seed);

	std::function<VType(void)> value_generator = [&](void) -> VType {

		return prng.rand() * (range.j - range.i) + range.i;
	};

	aux_generate_uVoronoiBoundary3D(spacing, variation, base_value, prng, value_generator);

	return true;
}

//voronoi rotations 2D: set VEC dimensions (force 2D in xy plane) and generate voronoi 2d tessellation with average spacing. This method is applicable only to DBL3 PType, where a rotation operation is applied, fixed in each Voronoi cell. 
//The rotation uses the values for polar (theta) and azimuthal (phi) angles specified in given ranges. prng instantiated with given seed.
template <>
inline bool VEC<DBL3>::generate_VoronoiRotation2D(DBL3 new_h, Rect new_rect, DBL2 theta, DBL2 phi, double spacing, unsigned seed)
{
	//force 2D in xy plane
	new_h.k = new_rect.size().k;

	if (!resize(new_h, new_rect) || spacing <= 0) return false;

	BorisRand prng(seed);

	std::function<DBL3(void)> value_generator = [&](void) -> DBL3 {

		double theta_value = prng.rand() * (theta.j - theta.i) + theta.i;
		double phi_value = prng.rand() * (phi.j - phi.i) + phi.i;

		DBL3 unit_vector = Polar_to_Cartesian(DBL3(1.0, theta_value, phi_value));

		return unit_vector;
	};

	aux_generate_Voronoi2D(spacing, prng, value_generator);

	return true;
}

//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
template <>
inline bool VEC<DBL3>::generate_uVoronoiRotation2D(DBL3 new_h, Rect new_rect, DBL2 theta, DBL2 phi, double spacing, double variation, unsigned seed)
{
	//force 2D in xy plane
	new_h.k = new_rect.size().k;

	if (!resize(new_h, new_rect) || spacing <= 0 || variation < 0) return false;

	BorisRand prng(seed);

	std::function<DBL3(void)> value_generator = [&](void) -> DBL3 {

		double theta_value = prng.rand() * (theta.j - theta.i) + theta.i;
		double phi_value = prng.rand() * (phi.j - phi.i) + phi.i;

		DBL3 unit_vector = Polar_to_Cartesian(DBL3(1.0, theta_value, phi_value));

		return unit_vector;
	};

	aux_generate_uVoronoi2D(spacing, variation, prng, value_generator);

	return true;
}

//voronoi rotations 3D: set VEC dimensions and generate voronoi 3d tessellation with average spacing. This method is applicable only to DBL3 PType, where a rotation operation is applied, fixed in each Voronoi cell. 
//The rotation uses the values for polar (theta) and azimuthal (phi) angles specified in given ranges in degrees. prng instantiated with given seed.
template <>
inline bool VEC<DBL3>::generate_VoronoiRotation3D(DBL3 new_h, Rect new_rect, DBL2 theta, DBL2 phi, double spacing, unsigned seed)
{
	if (!resize(new_h, new_rect) || spacing <= 0) return false;

	BorisRand prng(seed);

	std::function<DBL3(void)> value_generator = [&](void) -> DBL3 {

		double theta_value = prng.rand() * (theta.j - theta.i) + theta.i;
		double phi_value = prng.rand() * (phi.j - phi.i) + phi.i;

		DBL3 unit_vector = Polar_to_Cartesian(DBL3(1.0, theta_value, phi_value));

		return unit_vector;
	};

	aux_generate_Voronoi3D(spacing, prng, value_generator);

	return true;
}

//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
template <>
inline bool VEC<DBL3>::generate_uVoronoiRotation3D(DBL3 new_h, Rect new_rect, DBL2 theta, DBL2 phi, double spacing, double variation, unsigned seed)
{
	if (!resize(new_h, new_rect) || spacing <= 0 || variation < 0) return false;

	BorisRand prng(seed);

	std::function<DBL3(void)> value_generator = [&](void) -> DBL3 {

		double theta_value = prng.rand() * (theta.j - theta.i) + theta.i;
		double phi_value = prng.rand() * (phi.j - phi.i) + phi.i;

		DBL3 unit_vector = Polar_to_Cartesian(DBL3(1.0, theta_value, phi_value));

		return unit_vector;
	};

	aux_generate_uVoronoi3D(spacing, variation, prng, value_generator);

	return true;
}
