#pragma once

#include "VEC.h"
#include "VEC_MeshTransfer.h"

//--------------------------------------------HELPER METHODS

template <typename VType>
bool VEC<VType>::mapmesh_newdims(const SZ3& new_n)
{
	if (new_n == n) return true;

	std::vector<VType> quantity_new;

	auto map_quantity = [&](std::vector<VType>& quantity_old) -> bool
	{
		//abort if quantity_new cannot be resized to new dimensions
		if (!malloc_vector(quantity_new, new_n.dim())) return false;

		//now also transfer mesh values to new dimensions
		DBL3 sourceIdx = (DBL3)n / new_n;

#pragma omp parallel for
		for (int idx = 0; idx < new_n.dim(); idx++) {

			int _x = (int)floor((idx % new_n.x) * sourceIdx.x);
			int _y = (int)floor(((idx / new_n.x) % new_n.y) * sourceIdx.y);
			int _z = (int)floor((idx / (new_n.x * new_n.y)) * sourceIdx.z);

			quantity_new[idx] = quantity_old[_x + _y * n.x + _z * (n.x * n.y)];
		}

		//set allocated memory in quantity
		quantity_old.swap(quantity_new);
		return true;
	};

	bool success = map_quantity(quantity);

	for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
		success &= map_quantity(quantity_extra[sidx - 1]);

	set_quantity_pointers();

	if (success) {

		//set new size
		n = new_n;
		return true;
	}
	else return false;
}

template <typename VType>
SZ3 VEC<VType>::get_n_from_h_and_rect(const DBL3& h_, const Rect& rect_) const
{
	//calculate new n from rect and current h
	SZ3 new_n = round(rect_ / h_);
	if (new_n.x < 1) new_n.x = 1;
	if (new_n.y < 1) new_n.y = 1;
	if (new_n.z < 1) new_n.z = 1;

	return new_n;
}

//from current rectangle and h value set n. h may also need to be adjusted since n must be an integer
template <typename VType>
bool VEC<VType>::set_n_adjust_h(void)
{
	//make sure the cellsize divides the mesh rectangle

	//calculate new n from rect and current h
	SZ3 new_n = get_n_from_h_and_rect(h, rect);

	//adjust h for new n (but save it first in case memory allocation fails)
	DBL3 h_save = h;
	set_h(rect / new_n);

	//now set n - either map to new sizes (if current object is not empty) or allocate memory for new object
	if (!resize(new_n)) {

		//failed : go back to previous h (n unchanged)
		set_h(h_save);
		return false;
	}
	else return true;
}

//set cellsize, adjusting sublattice position vectors
template <typename VType>
void VEC<VType>::set_h(DBL3 new_h)
{
	//adjust sublattice position vectors
	for (int sidx = 0; sidx < get_num_sublattices(); sidx++) {

		if (h != DBL3()) r_xyz[sidx] = r_xyz[sidx] & (new_h / h);
		//if cellsize is null just leave r_xyz as it is (this could be first time a cellsize is set, but r_xyz could already have correct value)
	}

	//new cellsize
	h = new_h;
}

//set pointers in pquantity (must call whenever quantity and quantity_extra are resized)
template <typename VType>
void VEC<VType>::set_quantity_pointers(void)
{
	pquantity.resize(get_num_sublattices());
	pquantity[0] = &quantity;
	for (int sidx = 1; sidx < pquantity.size(); sidx++)
		pquantity[sidx] = &quantity_extra[sidx - 1];
}

//--------------------------------------------CONSTRUCTORS

template <typename VType>
VEC<VType>::VEC(void) :
	transfer(this), transfer2(this)
{
	pquantity.resize(1);
	pquantity[0] = &quantity;
	r_xyz.resize(1);
	r_xyz[0] = DBL3();
}

template <typename VType>
VEC<VType>::VEC(const SZ3& n_) :
	transfer(this), transfer2(this), n(n_)
{
	pquantity.resize(1);
	pquantity[0] = &quantity;
	r_xyz.resize(1);
	r_xyz[0] = DBL3();

	//make sure memory is assigned to set size
	if (!malloc_vector(quantity, n.dim())) n = SZ3();
}

template <typename VType>
VEC<VType>::VEC(const DBL3& h_, const Rect& rect_) :
	transfer(this), transfer2(this), h(h_), rect(rect_)
{
	pquantity.resize(1);
	pquantity[0] = &quantity;
	r_xyz.resize(1);
	r_xyz[0] = DBL3();

	//make sure memory is assigned to set size
	if (!set_n_adjust_h()) {

		h = DBL3();
		rect = Rect();
	}
}

template <typename VType>
VEC<VType>::VEC(const DBL3& h_, const Rect& rect_, VType value) :
	transfer(this), transfer2(this), h(h_), rect(rect_)
{
	pquantity.resize(1);
	pquantity[0] = &quantity;
	r_xyz.resize(1);
	r_xyz[0] = DBL3();

	//make sure memory is assigned to set size and value set
	if (set_n_adjust_h()) {

		quantity.assign(n.dim(), value);
	}
	else {

		h = DBL3();
		rect = Rect();
	}
}

//--------------------------------------------SIZING

template <typename VType>
bool VEC<VType>::resize(const SZ3& new_n)
{
	if (new_n == n) return true;

	//check if memory for new size can be allocated
	if (!mreserve_vector(quantity, new_n.dim())) return false;
	
	for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
		if (!mreserve_vector(quantity_extra[sidx - 1], new_n.dim())) return false;

	//always clear any transfer info when VEC changes size
	transfer.clear();
	transfer2.clear();

	if (n == SZ3(0)) {

		bool success = malloc_vector(quantity, new_n.dim());
		for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
			success &= malloc_vector(quantity_extra[sidx - 1], new_n.dim());

		set_quantity_pointers();

		//current zero size : set new size and rect
		if (success) {

			n = new_n;
			SetMeshRect();
			return true;
		}
	}
	else {

		//remap to new size and set rect
		if (mapmesh_newdims(new_n)) {

			SetMeshRect();
			return true;
		}
	}

	//if here then couldn't allocate memory : fail and previous size maintained
	return false;
}

template <typename VType>
bool VEC<VType>::resize(const DBL3& new_h, const Rect& new_rect)
{
	if (new_h == h && new_rect == rect) return true;

	//always clear any transfer info when VEC changes size
	transfer.clear();
	transfer2.clear();

	//save h and rect in case we cannot resize
	DBL3 save_h = h;
	Rect save_rect = rect;

	//set new h and rect for now
	set_h(new_h);
	rect = new_rect;

	if (!set_n_adjust_h()) {

		//failed : go back to previous dimensions
		set_h(save_h);
		rect = save_rect;

		return false;
	}
	else return true;
}

template <typename VType>
bool VEC<VType>::assign(const SZ3& new_n, VType value)
{
	if (!mreserve_vector(quantity, new_n.dim())) return false;

	for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
		if (!mreserve_vector(quantity_extra[sidx - 1], new_n.dim())) return false;

	//always clear any transfer info when VEC changes size
	if (new_n != n) {

		transfer.clear();
		transfer2.clear();
	}

	n = new_n;
	
	quantity.assign(n.dim(), value);
	for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
		quantity_extra[sidx - 1].assign(n.dim(), value);
	
	set_quantity_pointers();

	SetMeshRect();

	return true;
}

//multiple sub-lattice version
template <typename VType>
bool VEC<VType>::assign(const SZ3& new_n, std::vector<VType> values)
{
	if (!values.size()) return false;
	if (!mreserve_vector(quantity, new_n.dim())) return false;

	for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
		if (!mreserve_vector(quantity_extra[sidx - 1], new_n.dim())) return false;

	//always clear any transfer info when VEC changes size
	if (new_n != n) {

		transfer.clear();
		transfer2.clear();
	}

	n = new_n;

	quantity.assign(n.dim(), values[0]);
	for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
		if (sidx < values.size()) quantity_extra[sidx - 1].assign(n.dim(), values[sidx]);
		else quantity_extra[sidx - 1].assign(n.dim(), values[0]);

	set_quantity_pointers();

	SetMeshRect();

	return true;
}

//works like resize(h_, rect_) but sets given value also
template <typename VType>
bool VEC<VType>::assign(const DBL3& new_h, const Rect& new_rect, VType value)
{
	//calculate new n from rect and current h
	SZ3 new_n = get_n_from_h_and_rect(new_h, new_rect);

	if (!mreserve_vector(quantity, new_n.dim())) return false;

	for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
		if (!mreserve_vector(quantity_extra[sidx - 1], new_n.dim())) return false;

	//always clear any transfer info when VEC changes size
	if (new_h != h || new_rect != rect) {

		transfer.clear();
		transfer2.clear();
	}

	//now set dimensions, allocate memory and set value
	set_h(new_rect / new_n);
	rect = new_rect;
	n = new_n;

	quantity.assign(n.dim(), value);
	for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
		quantity_extra[sidx - 1].assign(n.dim(), value);

	set_quantity_pointers();

	return true;
}

//multiple sub-lattice version
template <typename VType>
bool VEC<VType>::assign(const DBL3& new_h, const Rect& new_rect, std::vector<VType> values)
{
	if (!values.size()) return false;

	//calculate new n from rect and current h
	SZ3 new_n = get_n_from_h_and_rect(new_h, new_rect);

	if (!mreserve_vector(quantity, new_n.dim())) return false;

	for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
		if (!mreserve_vector(quantity_extra[sidx - 1], new_n.dim())) return false;

	//always clear any transfer info when VEC changes size
	if (new_h != h || new_rect != rect) {

		transfer.clear();
		transfer2.clear();
	}

	//now set dimensions, allocate memory and set value
	set_h(new_rect / new_n);
	rect = new_rect;
	n = new_n;

	quantity.assign(n.dim(), values[0]);
	for (int sidx = 1; sidx < get_num_sublattices(); sidx++)
		if (sidx < values.size()) quantity_extra[sidx - 1].assign(n.dim(), values[sidx]);
		else quantity_extra[sidx - 1].assign(n.dim(), values[0]);

	set_quantity_pointers();

	return true;
}

template <typename VType>
void VEC<VType>::clear(void)
{
	transfer.clear();
	transfer2.clear();

	n = SZ3(0);

	quantity.clear();
	quantity.shrink_to_fit();
	
	clear_extra_sublattices();

	set_quantity_pointers();

	SetMeshRect();
}