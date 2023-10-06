#pragma once

#include "VEC.h"

//--------------------------------------------MULTIPLE ENTRIES SETTERS

template <typename VType>
void VEC<VType>::set(VType value)
{
#pragma omp parallel for
	for (int idx = 0; idx < quantity.size(); idx++) {

		quantity[idx] = value;
	}
}

//set value in box (i.e. in cells entirely included in box)
template <typename VType>
void VEC<VType>::setbox(const Box& box, VType value)
{
#pragma omp parallel for
	for (int j = (box.s.y >= 0 ? box.s.y : 0); j < (box.e.y <= n.y ? box.e.y : n.y); j++) {
		for (int k = (box.s.z >= 0 ? box.s.z : 0); k < (box.e.z <= n.z ? box.e.z : n.z); k++) {
			for (int i = (box.s.x >= 0 ? box.s.x : 0); i < (box.e.x <= n.x ? box.e.x : n.x); i++) {

				quantity[i + j * n.x + k * n.x*n.y] = value;
			}
		}
	}
}

//set value in rectangle (i.e. in cells intersecting the rectangle), where the rectangle is relative to this VEC's rectangle.
template <typename VType>
void VEC<VType>::setrect(const Rect& rectangle, VType value)
{
	if (!rect.intersects(rectangle + rect.s)) return;

	Box cells_box = box_from_rect_max(rectangle + rect.s);

	setbox(cells_box, value);
}

template <typename VType>
template <typename PType>
void VEC<VType>::renormalize(PType new_norm)
{
#pragma omp parallel for
	for (int idx = 0; idx < n.dim(); idx++) {

		PType curr_norm = GetMagnitude(quantity[idx]);

		if (IsNZ(curr_norm)) quantity[idx] *= new_norm / curr_norm;
	}
}

//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions
template <typename VType>
void VEC<VType>::copy_values(const VEC<VType>& copy_this, Rect dstRect, Rect srcRect, double multiplier)
{
	if (dstRect.IsNull()) dstRect = rect - rect.s;
	if (srcRect.IsNull()) srcRect = copy_this.rect - copy_this.rect.s;

	Box cells_box_dst = box_from_rect_max(dstRect + rect.s);
	SZ3 dst_n = cells_box_dst.size();
	DBL3 lRatio = dstRect.size() / srcRect.size();

#pragma omp parallel for
	for (int j = 0; j < dst_n.j; j++) {
		for (int k = 0; k < dst_n.k; k++) {
			for (int i = 0; i < dst_n.i; i++) {

				int idx_out = (i + cells_box_dst.s.i) + (j + cells_box_dst.s.j) * n.x + (k + cells_box_dst.s.k) * n.x * n.y;

				//destination cell rectangle
				Rect dst_cell_rect_rel = get_cellrect(idx_out) - rect.s - dstRect.s;

				//now map this to source rectangle
				Rect src_cell_rect_rel = Rect(dst_cell_rect_rel.s & lRatio, dst_cell_rect_rel.e & lRatio) + srcRect.s;

				if (idx_out < n.dim()) {

					quantity[idx_out] = copy_this.average(src_cell_rect_rel) * multiplier;
				}
			}
		}
	}
}

//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions
//can specify destination and source rectangles in relative coordinates
//this is intended for VECs where copy_this cellsize is much larger than that in this VEC, and instead of setting all values the same, thermalize_func generator will generate values
//e.g. this is useful for copying values from a micromagnetic mesh into an atomistic mesh, where the atomistic spins are generated according to a distribution setup in thermalize_func
//thermalize_func returns the value to set, and takes parameters VType (value in the larger cell from copy_this which is being copied), and int, int (index of larger cell from copy_this which is being copied, and index of destination cell)
template <typename VType>
void VEC<VType>::copy_values_thermalize(const VEC<VType>& copy_this, std::function<VType(VType, int, int)>& thermalize_func, Rect dstRect, Rect srcRect)
{
	if (dstRect.IsNull()) dstRect = rect - rect.s;
	if (srcRect.IsNull()) srcRect = copy_this.rect - copy_this.rect.s;

	Box cells_box_src = copy_this.box_from_rect_max(srcRect + copy_this.rect.s);
	SZ3 src_n = cells_box_src.size();

	DBL3 lRatio = dstRect.size() / srcRect.size();

	//go over source cells (not destination cells  as in copy_values)
#pragma omp parallel for
	for (int j = 0; j < src_n.j; j++) {
		for (int k = 0; k < src_n.k; k++) {
			for (int i = 0; i < src_n.i; i++) {

				int idx_src = (i + cells_box_src.s.i) + (j + cells_box_src.s.j) * copy_this.n.x + (k + cells_box_src.s.k) * copy_this.n.x * copy_this.n.y;

				//source cell rectangle in absolute coordinates
				Rect src_cell_rect_abs = copy_this.get_cellrect(idx_src);

				//source cell rectangle relative to srcRect in copy_this VEC
				Rect src_cell_rect_rel = src_cell_rect_abs - copy_this.rect.s - srcRect.s;

				//map to a rectangle in this VEC, with relative coordinates
				Rect dst_cell_rect_rel = Rect(src_cell_rect_rel.s & lRatio, src_cell_rect_rel.e & lRatio) + dstRect.s;

				//box of cells for destination
				Box dst_box = box_from_rect_max(dst_cell_rect_rel + rect.s);

				//go through all destination cells contained in the current copy_this cell

				for (int jbox = dst_box.s.j; jbox < dst_box.e.j; jbox++) {
					for (int kbox = dst_box.s.k; kbox < dst_box.e.k; kbox++) {
						for (int ibox = dst_box.s.i; ibox < dst_box.e.i; ibox++) {

							int idx_dst = ibox + jbox * n.x + kbox * n.x * n.y;
							quantity[idx_dst] = thermalize_func(copy_this[idx_src], idx_src, idx_dst);
						}
					}
				}
			}
		}
	}
}
