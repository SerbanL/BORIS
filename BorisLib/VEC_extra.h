#pragma once

#include "VEC.h"

//--------------------------------------------EXTRA SUBLATTICES : VEC_extra.h
//set number of sublattices, where r_xyz_.size() >= 1
//initial values on new sublattices will be copied from base sublattice
template <typename VType>
bool VEC<VType>::set_number_of_sublattices(std::vector<DBL3> r_xyz_)
{
	if (!r_xyz_.size()) return false;

	//add new sublattices if required (i.e. leave any existing ones, including the base sublattice)
	for (int sidx = r_xyz.size(); sidx < r_xyz_.size(); sidx++) {

		if (!add_sublattice(r_xyz_[sidx])) return false;
	}

	//however set all r_xyz values as indicated
	r_xyz.resize(r_xyz_.size());
	for (int sidx = 0; sidx < r_xyz.size(); sidx++) r_xyz[sidx] = r_xyz_[sidx];

	return true;
}


//add an extra sublattice with position vector re_xyz_ (normalized to cellsizes)
template <typename VType>
bool VEC<VType>::add_sublattice(DBL3 r_xyz_)
{
	std::vector<VType> quantity_new;
	if (!malloc_vector(quantity_new, n.dim())) return false;

	quantity_extra.push_back(std::vector<VType>{});
	quantity_extra.back().swap(quantity_new);
	r_xyz.push_back(r_xyz_);

	//copy values from base sublattice
	for (int idx = 0; idx < quantity.size(); idx++)
		quantity_extra.back()[idx] = quantity[idx];

	set_quantity_pointers();

	return true;
}

//set sublattice sidx vector position (sidx must be a correct index, 0 is for the base lattice)
template <typename VType>
void VEC<VType>::set_sublattice_position(int sidx, DBL3 r_xyz_)
{
	if (sidx < get_num_sublattices()) r_xyz[sidx] = r_xyz_;
}

//delete all extra sublattices, leaving only the base lattice
template <typename VType>
void VEC<VType>::clear_extra_sublattices(void)
{
	if (get_num_sublattices() > 1) {

		for (int sidx = 1; sidx < get_num_sublattices(); sidx++) {

			quantity_extra[sidx - 1].clear();
			quantity_extra[sidx - 1].shrink_to_fit();
		}

		quantity_extra.clear();

		r_xyz.erase(r_xyz.begin() + 1, r_xyz.end());

		set_quantity_pointers();
	}
}

//set value in cell idx for all sub-lattices
template <typename VType>
void VEC<VType>::set_sublattices_value(int idx, VType value)
{
	for (int sidx = 0; sidx < get_num_sublattices(); sidx++) (*pquantity[sidx])[idx] = value;
}

//set value in cell idx for all sub-lattices, different values for each sub-lattice
template <typename VType>
void VEC<VType>::set_sublattices_value(int idx, const std::vector<VType>& value)
{
	for (int sidx = 0; sidx < get_num_sublattices(); sidx++) 
		if (sidx < value.size()) (*pquantity[sidx])[idx] = value[sidx];
		else (*pquantity[sidx])[idx] = quantity[idx];
}