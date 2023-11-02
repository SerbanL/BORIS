#include "stdafx.h"
#include "DemagKernelCollectionCUDA.h"

#if COMPILECUDA == 1

#if defined(MODULE_COMPILATION_SDEMAG) || defined(MODULE_COMPILATION_DEMAG)

//-------------------------- MEMORY ALLOCATION

BError DemagKernelCollectionCUDA::AllocateKernelMemory(void)
{
	BError error(__FUNCTION__);

	size_t num_meshes = Rect_collection.size();

	//size the kernels vector, but do not allocate any kernels here -> allocated them if needed just before calculating them
	//this way instead of always allocating new kernels we might be able to reuse other kernels which have already been allocated (reduce redundancy)
	kernels.clear();
	kernels.assign(num_meshes, nullptr);

	//build kernels_gpu after filling kernels with calculation method
	kernels_gpu.clear();

	inverse_shifted.assign(num_meshes, false);
	zshifted.assign(num_meshes, false);
	xshifted.assign(num_meshes, false);

	inverse_shifted_gpu.clear();

	return error;
}

//-------------------------- SETTERS

//Set all the rectangles participating in convolution. This determines the number of kernels needed : one for each mesh.
BError DemagKernelCollectionCUDA::Set_Rect_Collection(std::vector<cuRect>& Rect_collection_, cuRect this_rect_, cuBReal h_max_, int self_contribution_index_)
{
	BError error(__FUNCTION__);

	Rect_collection = Rect_collection_;
	this_rect = this_rect_;
	h_max = h_max_;
	self_contribution_index = self_contribution_index_;

	error = AllocateKernelMemory();

	return error;
}

#endif

#endif

