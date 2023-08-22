#pragma once

#include "cuVEC_VC.h"

//-------------------------------- GRADIENT OPERATOR

//gradient operator. Use Neumann boundary conditions (homogeneous).
//Can be used at composite media boundaries where sided differentials will be used instead.
template <typename VType>
__device__ cuVAL3<VType> cuVEC_VC<VType>::grad_neu(int idx) const
{
	cuVAL3<VType> diff = cuVAL3<VType>();

	if (!(ngbrFlags[idx] & NF_NOTEMPTY)) return diff;

	//x direction
	if ((ngbrFlags[idx] & NF_BOTHX) == NF_BOTHX) {

		//inner point along this direction
		diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOX)) {

		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPX) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[j*pUVA_haloVEC_right->n.x + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NNX) diff.x = (halo_val - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			else diff.x = (halo_val - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.x);
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[(pUVA_haloVEC_left->n.x - 1) + j * pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - halo_val) / (2 * cuVEC<VType>::h.x);
			else diff.x = (cuVEC<VType>::quantity[idx] - halo_val) / (2 * cuVEC<VType>::h.x);
		}
	}
	//Is it a CMBND boundary? - if not then use homogeneous Neumann condition (differential zero at the boundary)
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDX)) {

		if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x;
		if (ngbrFlags[idx] & NF_NNX) diff.x = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x;
	}
	else if (ngbrFlags[idx] & NF_NGBRX) {

		if (ngbrFlags[idx] & NF_NPX) {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cuVEC<VType>::quantity[idx + 1] - cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.x);
			}
		}
		else {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.x - 1)] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			}
		}
	}

	//y direction
	if ((ngbrFlags[idx] & NF_BOTHY) == NF_BOTHY) {

		diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOY)) {

		int i = idx % cuVEC<VType>::n.x;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPY) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNY) diff.y = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			else diff.y = (halo_val - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.y);
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + (pUVA_haloVEC_left->n.y - 1)*pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - halo_val) / (2 * cuVEC<VType>::h.y);
			else diff.y = (cuVEC<VType>::quantity[idx] - halo_val) / (2 * cuVEC<VType>::h.y);
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDY)) {

		if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y;
		if (ngbrFlags[idx] & NF_NNY) diff.y = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y;
	}
	else if (ngbrFlags[idx] & NF_NGBRY) {

		if (ngbrFlags[idx] & NF_NPY) {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.y);
			}
		}
		else {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
		}
	}

	//z direction
	if ((ngbrFlags[idx] & NF_BOTHZ) == NF_BOTHZ) {

		diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOZ)) {

		int i = idx % cuVEC<VType>::n.x;
		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;

		if (ngbrFlags2[idx] & NF2_HALOPZ) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + j * pUVA_haloVEC_right->n.x] : halo_p[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNZ) diff.z = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			else diff.z = (halo_val - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.z);
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + j * pUVA_haloVEC_left->n.x + (pUVA_haloVEC_left->n.z - 1)*pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - halo_val) / (2 * cuVEC<VType>::h.z);
			else diff.z = (cuVEC<VType>::quantity[idx] - halo_val) / (2 * cuVEC<VType>::h.z);
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDZ)) {

		if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z;
		if (ngbrFlags[idx] & NF_NNZ) diff.z = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z;
	}
	else if (ngbrFlags[idx] & NF_NGBRZ) {

		if (ngbrFlags[idx] & NF_NPZ) {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.z);
			}
		}
		else {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
		}
	}

	return diff;
}

//gradient operator. Use non-homogeneous Neumann boundary conditions.
//Can be used at composite media boundaries where sided differentials will be used instead.
//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions - the class Class_BDiff must define a method bdiff returning a cuVAL3<VType> and taking an int (the cell index)
template <typename VType>
template <typename Class_BDiff>
__device__ cuVAL3<VType> cuVEC_VC<VType>::grad_nneu(int idx, const Class_BDiff& bdiff_class) const
{
	cuVAL3<VType> diff = cuVAL3<VType>();

	if (!(ngbrFlags[idx] & NF_NOTEMPTY)) return diff;

	//x direction
	if ((ngbrFlags[idx] & NF_BOTHX) == NF_BOTHX) {

		//inner point along this direction
		diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOX)) {

		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPX) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[j*pUVA_haloVEC_right->n.x + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NNX) diff.x = (halo_val - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			else diff.x = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x + bdiff_class.bdiff(idx).x) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[(pUVA_haloVEC_left->n.x - 1) + j * pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - halo_val) / (2 * cuVEC<VType>::h.x);
			else diff.x = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.x + bdiff_class.bdiff(idx).x) / 2;
		}
	}
	//Is it a CMBND boundary? - if not then use non-homogeneous Neumann condition
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDX)) {

		if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x;
		if (ngbrFlags[idx] & NF_NNX) diff.x = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x;
	}
	else if (ngbrFlags[idx] & NF_NGBRX) {

		cuVAL3<VType> bdiff_val = bdiff_class.bdiff(idx);

		if (ngbrFlags[idx] & NF_NPX) {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cuVEC<VType>::quantity[idx + 1] - cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = ((cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x + bdiff_val.x) / 2;
			}
		}
		else {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.x - 1)] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x + bdiff_val.x) / 2;
			}
		}
	}

	//y direction
	if ((ngbrFlags[idx] & NF_BOTHY) == NF_BOTHY) {

		diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOY)) {

		int i = idx % cuVEC<VType>::n.x;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPY) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNY) diff.y = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			else diff.y = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y + bdiff_class.bdiff(idx).y) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + (pUVA_haloVEC_left->n.y - 1)*pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - halo_val) / (2 * cuVEC<VType>::h.y);
			else diff.y = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.y + bdiff_class.bdiff(idx).y) / 2;
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDY)) {

		if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y;
		if (ngbrFlags[idx] & NF_NNY) diff.y = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y;
	}
	else if (ngbrFlags[idx] & NF_NGBRY) {

		cuVAL3<VType> bdiff_val = bdiff_class.bdiff(idx);

		if (ngbrFlags[idx] & NF_NPY) {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = ((cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y + bdiff_val.y) / 2;
			}
		}
		else {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y + bdiff_val.y) / 2;
			}
		}
	}

	//z direction
	if ((ngbrFlags[idx] & NF_BOTHZ) == NF_BOTHZ) {

		diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOZ)) {

		int i = idx % cuVEC<VType>::n.x;
		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;

		if (ngbrFlags2[idx] & NF2_HALOPZ) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + j * pUVA_haloVEC_right->n.x] : halo_p[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNZ) diff.z = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			else diff.z = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z + bdiff_class.bdiff(idx).z) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + j * pUVA_haloVEC_left->n.x + (pUVA_haloVEC_left->n.z - 1)*pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - halo_val) / (2 * cuVEC<VType>::h.z);
			else diff.z = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.z + bdiff_class.bdiff(idx).z) / 2;
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDZ)) {

		if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z;
		if (ngbrFlags[idx] & NF_NNZ) diff.z = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z;
	}
	else if (ngbrFlags[idx] & NF_NGBRZ) {

		cuVAL3<VType> bdiff_val = bdiff_class.bdiff(idx);

		if (ngbrFlags[idx] & NF_NPZ) {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = ((cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z + bdiff_val.z) / 2;
			}
		}
		else {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z + bdiff_val.z) / 2;
			}
		}
	}

	return diff;
}

//Same as above but boundary conditions specified using a constant
template <typename VType>
__device__ cuVAL3<VType> cuVEC_VC<VType>::grad_nneu(int idx, const cuVAL3<VType>& bdiff) const
{
	cuVAL3<VType> diff = cuVAL3<VType>();

	if (!(ngbrFlags[idx] & NF_NOTEMPTY)) return diff;

	//x direction
	if ((ngbrFlags[idx] & NF_BOTHX) == NF_BOTHX) {

		//inner point along this direction
		diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOX)) {

		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPX) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[j*pUVA_haloVEC_right->n.x + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NNX) diff.x = (halo_val - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			else diff.x = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x + bdiff.x) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[(pUVA_haloVEC_left->n.x - 1) + j * pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - halo_val) / (2 * cuVEC<VType>::h.x);
			else diff.x = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.x + bdiff.x) / 2;
		}
	}
	//Is it a CMBND boundary? - if not then use non-homogeneous Neumann condition
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDX)) {

		if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x;
		if (ngbrFlags[idx] & NF_NNX) diff.x = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x;
	}
	else if (ngbrFlags[idx] & NF_NGBRX) {

		if (ngbrFlags[idx] & NF_NPX) {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cuVEC<VType>::quantity[idx + 1] - cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = ((cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x + bdiff.x) / 2;
			}
		}
		else {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.x - 1)] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x + bdiff.x) / 2;
			}
		}
	}

	//y direction
	if ((ngbrFlags[idx] & NF_BOTHY) == NF_BOTHY) {

		diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOY)) {

		int i = idx % cuVEC<VType>::n.x;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPY) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNY) diff.y = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			else diff.y = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y + bdiff.y) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + (pUVA_haloVEC_left->n.y - 1)*pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - halo_val) / (2 * cuVEC<VType>::h.y);
			else diff.y = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.y + bdiff.y) / 2;
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDY)) {

		if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y;
		if (ngbrFlags[idx] & NF_NNY) diff.y = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y;
	}
	else if (ngbrFlags[idx] & NF_NGBRY) {

		if (ngbrFlags[idx] & NF_NPY) {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = ((cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y + bdiff.y) / 2;
			}
		}
		else {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y + bdiff.y) / 2;
			}
		}
	}

	//z direction
	if ((ngbrFlags[idx] & NF_BOTHZ) == NF_BOTHZ) {

		diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOZ)) {

		int i = idx % cuVEC<VType>::n.x;
		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;

		if (ngbrFlags2[idx] & NF2_HALOPZ) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + j * pUVA_haloVEC_right->n.x] : halo_p[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNZ) diff.z = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			else diff.z = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z + bdiff.z) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + j * pUVA_haloVEC_left->n.x + (pUVA_haloVEC_left->n.z - 1)*pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - halo_val) / (2 * cuVEC<VType>::h.z);
			else diff.z = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.z + bdiff.z) / 2;
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDZ)) {

		if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z;
		if (ngbrFlags[idx] & NF_NNZ) diff.z = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z;
	}
	else if (ngbrFlags[idx] & NF_NGBRZ) {

		if (ngbrFlags[idx] & NF_NPZ) {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = ((cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z + bdiff.z) / 2;
			}
		}
		else {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z + bdiff.z) / 2;
			}
		}
	}

	return diff;
}

//gradient operator. Use Dirichlet conditions if set, else Neumann boundary conditions (homogeneous).
//Can be used at composite media boundaries where sided differentials will be used instead.
template <typename VType>
__device__ cuVAL3<VType> cuVEC_VC<VType>::grad_diri(int idx) const
{
	cuVAL3<VType> diff = cuVAL3<VType>();

	if (!(ngbrFlags[idx] & NF_NOTEMPTY)) return diff;

	//x direction
	if ((ngbrFlags[idx] & NF_BOTHX) == NF_BOTHX) {

		//inner point along this direction
		diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOX)) {

		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPX) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[j*pUVA_haloVEC_right->n.x + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NNX) diff.x = (halo_val - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			else diff.x = (halo_val - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.x);
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[(pUVA_haloVEC_left->n.x - 1) + j * pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - halo_val) / (2 * cuVEC<VType>::h.x);
			else diff.x = (cuVEC<VType>::quantity[idx] - halo_val) / (2 * cuVEC<VType>::h.x);
		}
	}
	//not an inner point along this direction - Use Dirichlet?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_DIRICHLETX)) {

		if (ngbrFlags2[idx] & NF2_DIRICHLETPX) diff.x = (cuVEC<VType>::quantity[idx + 1] + cuVEC<VType>::quantity[idx] - 2 * get_dirichlet_value(NF2_DIRICHLETPX, idx)) / (2 * cuVEC<VType>::h.x);
		else								   diff.x = (2 * get_dirichlet_value(NF2_DIRICHLETNX, idx) - cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
	}
	//Not Dirichlet, is it a CMBND boundary? - if not this either then use homogeneous Neumann condition
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDX)) {

		if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x;
		if (ngbrFlags[idx] & NF_NNX) diff.x = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x;
	}
	else if (ngbrFlags[idx] & NF_NGBRX) {

		if (ngbrFlags[idx] & NF_NPX) {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cuVEC<VType>::quantity[idx + 1] - cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.x);
			}
		}
		else {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.x - 1)] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			}
		}
	}

	//y direction
	if ((ngbrFlags[idx] & NF_BOTHY) == NF_BOTHY) {

		diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOY)) {

		int i = idx % cuVEC<VType>::n.x;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPY) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNY) diff.y = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			else diff.y = (halo_val - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.y);
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + (pUVA_haloVEC_left->n.y - 1)*pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - halo_val) / (2 * cuVEC<VType>::h.y);
			else diff.y = (cuVEC<VType>::quantity[idx] - halo_val) / (2 * cuVEC<VType>::h.y);
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_DIRICHLETY)) {

		if (ngbrFlags2[idx] & NF2_DIRICHLETPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] + cuVEC<VType>::quantity[idx] - 2 * get_dirichlet_value(NF2_DIRICHLETPY, idx)) / (2 * cuVEC<VType>::h.y);
		else								   diff.y = (2 * get_dirichlet_value(NF2_DIRICHLETNY, idx) - cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDY)) {

		if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y;
		if (ngbrFlags[idx] & NF_NNY) diff.y = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y;
	}
	else if (ngbrFlags[idx] & NF_NGBRY) {

		if (ngbrFlags[idx] & NF_NPY) {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.y);
			}
		}
		else {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
		}
	}

	//z direction
	if ((ngbrFlags[idx] & NF_BOTHZ) == NF_BOTHZ) {

		diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOZ)) {

		int i = idx % cuVEC<VType>::n.x;
		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;

		if (ngbrFlags2[idx] & NF2_HALOPZ) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + j * pUVA_haloVEC_right->n.x] : halo_p[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNZ) diff.z = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			else diff.z = (halo_val - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.z);
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + j * pUVA_haloVEC_left->n.x + (pUVA_haloVEC_left->n.z - 1)*pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - halo_val) / (2 * cuVEC<VType>::h.z);
			else diff.z = (cuVEC<VType>::quantity[idx] - halo_val) / (2 * cuVEC<VType>::h.z);
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_DIRICHLETZ)) {

		if (ngbrFlags2[idx] & NF2_DIRICHLETPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] + cuVEC<VType>::quantity[idx] - 2 * get_dirichlet_value(NF2_DIRICHLETPZ, idx)) / (2 * cuVEC<VType>::h.z);
		else								   diff.z = (2 * get_dirichlet_value(NF2_DIRICHLETNZ, idx) - cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDZ)) {

		if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z;
		if (ngbrFlags[idx] & NF_NNZ) diff.z = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z;
	}
	else if (ngbrFlags[idx] & NF_NGBRZ) {

		if (ngbrFlags[idx] & NF_NPZ) {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / (2 * cuVEC<VType>::h.z);
			}
		}
		else {
			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {


				diff.z = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
		}
	}

	return diff;
}

//gradient operator. Use Dirichlet conditions if set, else non-homogeneous Neumann boundary conditions.
//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions - the class Class_BDiff must define a method bdiff returning a cuVAL3<VType> and taking an int (the cell index)
//Can be used at composite media boundaries where sided differentials will be used instead.
template <typename VType>
template <typename Class_BDiff>
__device__ cuVAL3<VType> cuVEC_VC<VType>::grad_diri_nneu(int idx, const Class_BDiff& bdiff_class) const
{
	cuVAL3<VType> diff = cuVAL3<VType>();

	if (!(ngbrFlags[idx] & NF_NOTEMPTY)) return diff;

	//x direction
	if ((ngbrFlags[idx] & NF_BOTHX) == NF_BOTHX) {

		//inner point along this direction
		diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOX)) {

		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPX) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[j*pUVA_haloVEC_right->n.x + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NNX) diff.x = (halo_val - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			else diff.x = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x + bdiff_class.bdiff(idx).x) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[(pUVA_haloVEC_left->n.x - 1) + j * pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - halo_val) / (2 * cuVEC<VType>::h.x);
			else diff.x = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.x + bdiff_class.bdiff(idx).x) / 2;
		}
	}
	//not an inner point along this direction - Use Dirichlet?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_DIRICHLETX)) {

		if (ngbrFlags2[idx] & NF2_DIRICHLETPX) diff.x = (cuVEC<VType>::quantity[idx + 1] + cuVEC<VType>::quantity[idx] - 2 * get_dirichlet_value(NF2_DIRICHLETPX, idx)) / (2 * cuVEC<VType>::h.x);
		else								   diff.x = (2 * get_dirichlet_value(NF2_DIRICHLETNX, idx) - cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
	}
	//Not Dirichlet, is it a CMBND boundary? - if not this either then use homogeneous Neumann condition
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDX)) {

		if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x;
		if (ngbrFlags[idx] & NF_NNX) diff.x = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x;
	}
	else if (ngbrFlags[idx] & NF_NGBRX) {

		cuVAL3<VType> bdiff_val = bdiff_class.bdiff(idx);

		if (ngbrFlags[idx] & NF_NPX) {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cuVEC<VType>::quantity[idx + 1] - cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = ((cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x + bdiff_val.x) / 2;
			}
		}
		else {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.x - 1)] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x + bdiff_val.x) / 2;
			}
		}
	}

	//y direction
	if ((ngbrFlags[idx] & NF_BOTHY) == NF_BOTHY) {

		diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOY)) {

		int i = idx % cuVEC<VType>::n.x;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPY) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNY) diff.y = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			else diff.y = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y + bdiff_class.bdiff(idx).y) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + (pUVA_haloVEC_left->n.y - 1)*pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - halo_val) / (2 * cuVEC<VType>::h.y);
			else diff.y = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.y + bdiff_class.bdiff(idx).y) / 2;
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_DIRICHLETY)) {

		if (ngbrFlags2[idx] & NF2_DIRICHLETPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] + cuVEC<VType>::quantity[idx] - 2 * get_dirichlet_value(NF2_DIRICHLETPY, idx)) / (2 * cuVEC<VType>::h.y);
		else								   diff.y = (2 * get_dirichlet_value(NF2_DIRICHLETNY, idx) - cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDY)) {

		if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y;
		if (ngbrFlags[idx] & NF_NNY) diff.y = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y;
	}
	else if (ngbrFlags[idx] & NF_NGBRY) {

		cuVAL3<VType> bdiff_val = bdiff_class.bdiff(idx);

		if (ngbrFlags[idx] & NF_NPY) {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = ((cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y + bdiff_val.y) / 2;
			}
		}
		else {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y + bdiff_val.y) / 2;
			}
		}
	}

	//z direction
	if ((ngbrFlags[idx] & NF_BOTHZ) == NF_BOTHZ) {

		diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOZ)) {

		int i = idx % cuVEC<VType>::n.x;
		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;

		if (ngbrFlags2[idx] & NF2_HALOPZ) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + j * pUVA_haloVEC_right->n.x] : halo_p[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNZ) diff.z = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			else diff.z = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z + bdiff_class.bdiff(idx).z) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + j * pUVA_haloVEC_left->n.x + (pUVA_haloVEC_left->n.z - 1)*pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - halo_val) / (2 * cuVEC<VType>::h.z);
			else diff.z = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.z + bdiff_class.bdiff(idx).z) / 2;
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_DIRICHLETZ)) {

		if (ngbrFlags2[idx] & NF2_DIRICHLETPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] + cuVEC<VType>::quantity[idx] - 2 * get_dirichlet_value(NF2_DIRICHLETPZ, idx)) / (2 * cuVEC<VType>::h.z);
		else								   diff.z = (2 * get_dirichlet_value(NF2_DIRICHLETNZ, idx) - cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDZ)) {

		if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z;
		if (ngbrFlags[idx] & NF_NNZ) diff.z = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z;
	}
	else if (ngbrFlags[idx] & NF_NGBRZ) {

		cuVAL3<VType> bdiff_val = bdiff_class.bdiff(idx);

		if (ngbrFlags[idx] & NF_NPZ) {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = ((cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z + bdiff_val.z) / 2;
			}
		}
		else {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z + bdiff_val.z) / 2;
			}
		}
	}

	return diff;
}

//Same as above but boundary conditions specified using a constant
template <typename VType>
__device__ cuVAL3<VType> cuVEC_VC<VType>::grad_diri_nneu(int idx, const cuVAL3<VType>& bdiff) const
{
	cuVAL3<VType> diff = cuVAL3<VType>();

	if (!(ngbrFlags[idx] & NF_NOTEMPTY)) return diff;

	//x direction
	if ((ngbrFlags[idx] & NF_BOTHX) == NF_BOTHX) {

		//inner point along this direction
		diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOX)) {

		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPX) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[j*pUVA_haloVEC_right->n.x + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NNX) diff.x = (halo_val - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			else diff.x = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x + bdiff.x) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[(pUVA_haloVEC_left->n.x - 1) + j * pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - halo_val) / (2 * cuVEC<VType>::h.x);
			else diff.x = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.x + bdiff.x) / 2;
		}
	}
	//not an inner point along this direction - Use Dirichlet?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_DIRICHLETX)) {

		if (ngbrFlags2[idx] & NF2_DIRICHLETPX) diff.x = (cuVEC<VType>::quantity[idx + 1] + cuVEC<VType>::quantity[idx] - 2 * get_dirichlet_value(NF2_DIRICHLETPX, idx)) / (2 * cuVEC<VType>::h.x);
		else								   diff.x = (2 * get_dirichlet_value(NF2_DIRICHLETNX, idx) - cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
	}
	//Not Dirichlet, is it a CMBND boundary? - if not this either then use homogeneous Neumann condition
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDX)) {

		if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x;
		if (ngbrFlags[idx] & NF_NNX) diff.x = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x;
	}
	else if (ngbrFlags[idx] & NF_NGBRX) {

		if (ngbrFlags[idx] & NF_NPX) {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cuVEC<VType>::quantity[idx + 1] - cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = ((cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x + bdiff.x) / 2;
			}
		}
		else {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.x - 1)] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x + bdiff.x) / 2;
			}
		}
	}

	//y direction
	if ((ngbrFlags[idx] & NF_BOTHY) == NF_BOTHY) {

		diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOY)) {

		int i = idx % cuVEC<VType>::n.x;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPY) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNY) diff.y = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			else diff.y = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y + bdiff.y) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + (pUVA_haloVEC_left->n.y - 1)*pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - halo_val) / (2 * cuVEC<VType>::h.y);
			else diff.y = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.y + bdiff.y) / 2;
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_DIRICHLETY)) {

		if (ngbrFlags2[idx] & NF2_DIRICHLETPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] + cuVEC<VType>::quantity[idx] - 2 * get_dirichlet_value(NF2_DIRICHLETPY, idx)) / (2 * cuVEC<VType>::h.y);
		else								   diff.y = (2 * get_dirichlet_value(NF2_DIRICHLETNY, idx) - cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDY)) {

		if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y;
		if (ngbrFlags[idx] & NF_NNY) diff.y = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y;
	}
	else if (ngbrFlags[idx] & NF_NGBRY) {

		if (ngbrFlags[idx] & NF_NPY) {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = ((cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y + bdiff.y) / 2;
			}
		}
		else {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y + bdiff.y) / 2;
			}
		}
	}

	//z direction
	if ((ngbrFlags[idx] & NF_BOTHZ) == NF_BOTHZ) {

		diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOZ)) {

		int i = idx % cuVEC<VType>::n.x;
		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;

		if (ngbrFlags2[idx] & NF2_HALOPZ) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + j * pUVA_haloVEC_right->n.x] : halo_p[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNZ) diff.z = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			else diff.z = ((halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z + bdiff.z) / 2;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + j * pUVA_haloVEC_left->n.x + (pUVA_haloVEC_left->n.z - 1)*pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - halo_val) / (2 * cuVEC<VType>::h.z);
			else diff.z = ((cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.z + bdiff.z) / 2;
		}
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_DIRICHLETZ)) {

		if (ngbrFlags2[idx] & NF2_DIRICHLETPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] + cuVEC<VType>::quantity[idx] - 2 * get_dirichlet_value(NF2_DIRICHLETPZ, idx)) / (2 * cuVEC<VType>::h.z);
		else								   diff.z = (2 * get_dirichlet_value(NF2_DIRICHLETNZ, idx) - cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
	}
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_CMBNDZ)) {

		if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z;
		if (ngbrFlags[idx] & NF_NNZ) diff.z = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z;
	}
	else if (ngbrFlags[idx] & NF_NGBRZ) {

		if (ngbrFlags[idx] & NF_NPZ) {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = ((cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z + bdiff.z) / 2;
			}
		}
		else {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = ((cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z + bdiff.z) / 2;
			}
		}
	}

	return diff;
}

//gradient operator (specializations defined for 1. VType = double, RType = DBL3). Use sided differentials (also at composite media boundaries)
template <typename VType>
__device__ cuVAL3<VType> cuVEC_VC<VType>::grad_sided(int idx) const
{
	cuVAL3<VType> diff = cuVAL3<VType>();

	if (!(ngbrFlags[idx] & NF_NOTEMPTY)) return diff;

	//x direction
	if ((ngbrFlags[idx] & NF_BOTHX) == NF_BOTHX) {

		//inner point along this direction
		diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOX)) {

		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPX) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[j*pUVA_haloVEC_right->n.x + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NNX) diff.x = (halo_val - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			else diff.x = (halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[(pUVA_haloVEC_left->n.x - 1) + j * pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[j + k * cuVEC<VType>::n.y]);

			if (ngbrFlags[idx] & NF_NPX) diff.x = (cuVEC<VType>::quantity[idx + 1] - halo_val) / (2 * cuVEC<VType>::h.x);
			else diff.x = (cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.x;
		}
	}
	//use sided differentials if one of the neighbors is present
	else if (ngbrFlags[idx] & NF_NGBRX) {

		if (ngbrFlags[idx] & NF_NPX) {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cuVEC<VType>::quantity[idx + 1] - cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = (cuVEC<VType>::quantity[idx + 1] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.x;
			}
		}
		else {

			//is it a pbc along x? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCX) {

				diff.x = (cu_get_sign(pbc_x) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.x - 1)] - cuVEC<VType>::quantity[idx - 1]) / (2 * cuVEC<VType>::h.x);
			}
			else {

				diff.x = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - 1]) / cuVEC<VType>::h.x;
			}
		}
	}

	//y direction
	if ((ngbrFlags[idx] & NF_BOTHY) == NF_BOTHY) {

		diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOY)) {

		int i = idx % cuVEC<VType>::n.x;
		int k = idx / (cuVEC<VType>::n.x*cuVEC<VType>::n.y);

		if (ngbrFlags2[idx] & NF2_HALOPY) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + k * pUVA_haloVEC_right->n.x*pUVA_haloVEC_right->n.y] : halo_p[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNY) diff.y = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			else diff.y = (halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + (pUVA_haloVEC_left->n.y - 1)*pUVA_haloVEC_left->n.x + k * pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + k * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPY) diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - halo_val) / (2 * cuVEC<VType>::h.y);
			else diff.y = (cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.y;
		}
	}
	else if (ngbrFlags[idx] & NF_NGBRY) {

		if (ngbrFlags[idx] & NF_NPY) {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.y;
			}
		}
		else {

			//is it a pbc along y? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCY) {

				diff.y = (cu_get_sign(pbc_y) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.y - 1) * cuVEC<VType>::n.x] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / (2 * cuVEC<VType>::h.y);
			}
			else {

				diff.y = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x]) / cuVEC<VType>::h.y;
			}
		}
	}

	//z direction
	if ((ngbrFlags[idx] & NF_BOTHZ) == NF_BOTHZ) {

		diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
	}
	//use halo region?
	else if (using_extended_flags && (ngbrFlags2[idx] & NF2_HALOZ)) {

		int i = idx % cuVEC<VType>::n.x;
		int j = (idx / cuVEC<VType>::n.x) % cuVEC<VType>::n.y;

		if (ngbrFlags2[idx] & NF2_HALOPZ) {

			VType halo_val = (pUVA_haloVEC_right ? (*pUVA_haloVEC_right)[i + j * pUVA_haloVEC_right->n.x] : halo_p[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NNZ) diff.z = (halo_val - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			else diff.z = (halo_val - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z;
		}
		else {

			VType halo_val = (pUVA_haloVEC_left ? (*pUVA_haloVEC_left)[i + j * pUVA_haloVEC_left->n.x + (pUVA_haloVEC_left->n.z - 1)*pUVA_haloVEC_left->n.x*pUVA_haloVEC_left->n.y] : halo_n[i + j * cuVEC<VType>::n.x]);

			if (ngbrFlags[idx] & NF_NPZ) diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - halo_val) / (2 * cuVEC<VType>::h.z);
			else diff.z = (cuVEC<VType>::quantity[idx] - halo_val) / cuVEC<VType>::h.z;
		}
	}
	else if (ngbrFlags[idx] & NF_NGBRZ) {

		if (ngbrFlags[idx] & NF_NPZ) {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx + (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = (cuVEC<VType>::quantity[idx + cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx]) / cuVEC<VType>::h.z;
			}
		}
		else {

			//is it a pbc along z? If yes, then we are guaranteed to have a "neighbor" on the other side, so use it; otherwise apply boundary condition.
			if (ngbrFlags[idx] & NF_PBCZ) {

				diff.z = (cu_get_sign(pbc_z) * cuVEC<VType>::quantity[idx - (cuVEC<VType>::n.z - 1) * cuVEC<VType>::n.x*cuVEC<VType>::n.y] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / (2 * cuVEC<VType>::h.z);
			}
			else {

				diff.z = (cuVEC<VType>::quantity[idx] - cuVEC<VType>::quantity[idx - cuVEC<VType>::n.x*cuVEC<VType>::n.y]) / cuVEC<VType>::h.z;
			}
		}
	}

	return diff;
}