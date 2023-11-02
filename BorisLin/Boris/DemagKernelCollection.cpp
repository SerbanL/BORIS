#include "stdafx.h"
#include "DemagKernelCollection.h"

#ifdef MODULE_COMPILATION_SDEMAG

//-------------------------- KERNEL TYPES

BError KerType::AllocateKernels(Rect from_rect, Rect this_rect, SZ3 N_)
{
	BError error(__FUNCTION__);

	FreeKernels();

	N = N_;

	zshifted = false;
	xshifted = false;

	//check if internal demag
	if (from_rect == this_rect) {

		if (N.z == 1) {

			if (!Kdiag_real.resize(SZ3(N.x / 2 + 1, N.y / 2 + 1, 1))) return error(BERROR_OUTOFMEMORY_CRIT);
			if (!malloc_vector(K2D_odiag, (N.x / 2 + 1) * (N.y / 2 + 1))) return error(BERROR_OUTOFMEMORY_CRIT);
		}
		else {

			if (!Kdiag_real.resize(SZ3(N.x / 2 + 1, N.y / 2 + 1, N.z / 2 + 1))) return error(BERROR_OUTOFMEMORY_CRIT);
			if (!Kodiag_real.resize(SZ3(N.x / 2 + 1, N.y / 2 + 1, N.z / 2 + 1))) return error(BERROR_OUTOFMEMORY_CRIT);
		}
	}
	//not internal demag
	else {

		//see if we can use z-shifted kernels instead of full kernels
		DBL3 shift = from_rect.s - this_rect.s;

		//1. z shifted

		//z shifted (for 2D) : can use real kernels of reduced dimensions
		//Kxx, Kyy, Kzz : real
		//Kxy : real
		//Kxz, Kyz : imaginary (so must adjust for i multiplication when multiplying kernels)

		//z shifted for 3D : can use kernels of reduced dimensions but must be complex
		//
		//Kxx : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
		//Kyy : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
		//Kzz : y - symmetrical (+), z - Re part symmetrical (+), Im part inv. symmetric (-)
		//
		//Kxy : y - inv. symmetric (-), z - Re part symmetrical  (+), Im part inv. symmetric (-)
		//Kxz : y - symmetrical  (+), z - Re part inv. symmetric (-), Im part symmetrical  (+)
		//Kyz : y - inv. symmetric (-), z - Re part inv. symmetric (-), Im part symmetrical  (+)

		//2. x shifted

		//x shifted (for 2D and 3D) : can use kernels of reduced dimensions but must be complex
		//
		//Kxx, Kyy, Kzz : symmetrical (+) in y and z directions.
		//
		//Kxy : y - inv. symmetric (-), z - symmetrical  (+)
		//Kxz : y - symmetrical  (+), z - inv. symmetric (-)
		//Kyz : y - inv. symmetric (-), z - inv. symmetric (-)

		if ((IsZ(shift.x) && IsZ(shift.y)) || (IsZ(shift.y) && IsZ(shift.z))) {

			if (IsZ(shift.x) && IsZ(shift.y)) zshifted = true;
			if (IsZ(shift.y) && IsZ(shift.z)) xshifted = true;

			if (N.z == 1) {

				if (zshifted) {

					if (!Kdiag_real.resize(SZ3(N.x / 2 + 1, N.y / 2 + 1, N.z))) return error(BERROR_OUTOFMEMORY_CRIT);
					if (!Kodiag_real.resize(SZ3(N.x / 2 + 1, N.y / 2 + 1, N.z))) return error(BERROR_OUTOFMEMORY_CRIT);
				}
				else if (xshifted) {

					if (!Kdiag_cmpl.resize(SZ3(N.x / 2 + 1, N.y / 2 + 1, N.z))) return error(BERROR_OUTOFMEMORY_CRIT);
					if (!Kodiag_cmpl.resize(SZ3(N.x / 2 + 1, N.y / 2 + 1, N.z))) return error(BERROR_OUTOFMEMORY_CRIT);
				}
			}
			else {

				if (!Kdiag_cmpl.resize(SZ3(N.x / 2 + 1, N.y / 2 + 1, N.z / 2 + 1))) return error(BERROR_OUTOFMEMORY_CRIT);
				if (!Kodiag_cmpl.resize(SZ3(N.x / 2 + 1, N.y / 2 + 1, N.z / 2 + 1))) return error(BERROR_OUTOFMEMORY_CRIT);
			}
		}
		
		else {
		
			//full complex kernels

			if (!Kdiag_cmpl.resize(SZ3(N.x / 2 + 1, N.y, N.z))) return error(BERROR_OUTOFMEMORY_CRIT);
			if (!Kodiag_cmpl.resize(SZ3(N.x / 2 + 1, N.y, N.z))) return error(BERROR_OUTOFMEMORY_CRIT);
		}
	}

	return error;
}

void KerType::FreeKernels(void)
{
	K2D_odiag.clear();
	K2D_odiag.shrink_to_fit();

	Kdiag_cmpl.clear();
	Kodiag_cmpl.clear();

	Kdiag_real.clear();
	Kodiag_real.clear();

	kernel_calculated = false;
}

//-------------------------- MEMORY ALLOCATION

BError DemagKernelCollection::AllocateKernelMemory(void)
{
	BError error(__FUNCTION__);

	size_t num_meshes = Rect_collection.size();

	//size the kernels vector, but do not allocate any kernels here -> allocated them if needed just before calculating them
	//this way instead of always allocating new kernels we might be able to reuse other kernels which have already been allocated (reduce redundancy)
	kernels.clear();
	kernels.assign(num_meshes, nullptr);

	inverse_shifted.assign(num_meshes, false);

	return error;
}

//-------------------------- SETTERS

//Set all the rectangles participating in convolution. This determines the number of kernels needed : one for each mesh.
BError DemagKernelCollection::Set_Rect_Collection(std::vector<Rect>& Rect_collection_, Rect this_rect_, double h_max_, int self_contribution_index_)
{
	BError error(__FUNCTION__);

	//set new rect collection, rectangle for this collection and h_max

	Rect_collection = Rect_collection_;
	this_rect = this_rect_;
	h_max = h_max_;
	self_contribution_index = self_contribution_index_;

	error = AllocateKernelMemory();

	return error;
}

#endif


