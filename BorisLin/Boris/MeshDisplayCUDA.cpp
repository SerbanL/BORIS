#include "stdafx.h"

#include "MeshDisplayCUDA.h"

#if COMPILECUDA == 1

#include "mGPUConfig.h"

#include "BorisLib.h"

MeshDisplayCUDA::MeshDisplayCUDA(void) :
	display_vec_vc_vec_CUDA(mGPU),
	display_vec_vec_CUDA(mGPU),
	display_vec_vc_sca_CUDA(mGPU),
	display_vec_sca_CUDA(mGPU)
{
#if SINGLEPRECISION == 1
	pdisplay_vec_vc_vec = new VEC_VC<FLT3>();
	pdisplay2_vec_vc_vec = new VEC_VC<FLT3>();

	//vectorial cuVEC
	pdisplay_vec_vec = new VEC<FLT3>();
	pdisplay2_vec_vec = new VEC<FLT3>();

	//scalar cuVEC_VC
	pdisplay_vec_vc_sca = new VEC_VC<float>();

	//scalar cuVEC
	pdisplay_vec_sca = new VEC<float>();
#else
	pdisplay_vec_vc_vec = new VEC_VC<DBL3>();
	pdisplay2_vec_vc_vec = new VEC_VC<DBL3>();

	//vectorial cuVEC
	pdisplay_vec_vec = new VEC<DBL3>();
	pdisplay2_vec_vec = new VEC<DBL3>();

	//scalar cuVEC_VC
	pdisplay_vec_vc_sca = new VEC_VC<double>();

	//scalar cuVEC
	pdisplay_vec_sca = new VEC<double>();
#endif
}

MeshDisplayCUDA::~MeshDisplayCUDA()
{
	delete pdisplay_vec_vc_vec;
	delete pdisplay2_vec_vc_vec;

	delete pdisplay_vec_vec;
	delete pdisplay2_vec_vec;
	
	delete pdisplay_vec_vc_sca;
	delete pdisplay_vec_sca;
}

bool MeshDisplayCUDA::prepare_display(SZ3 n_quantity, Rect meshRect, double detail_level, mcu_VEC_VC(cuReal3)& cu_obj_quantity)
{
	//the cu_obj managed quantity cannot be empty
	if (!cu_obj_quantity.size_cpu().dim()) return false;
	
	DBL3 hdisp(detail_level);
	SZ3 ndisp = round(meshRect / hdisp);

	if (ndisp.x < 1) ndisp.x = 1;
	if (ndisp.y < 1) ndisp.y = 1;
	if (ndisp.z < 1) ndisp.z = 1;
	if (ndisp.x > n_quantity.x)  ndisp.x = n_quantity.x;
	if (ndisp.y > n_quantity.y)  ndisp.y = n_quantity.y;
	if (ndisp.z > n_quantity.z)  ndisp.z = n_quantity.z;

	hdisp = meshRect / ndisp;

	//resize the gpu and cpu vecs if needed - they must always have the same size
	if (!pdisplay_vec_vc_vec->resize(hdisp, meshRect)) return false;
	if (!display_vec_vc_vec_CUDA.resize(hdisp, meshRect)) return false;

	//extract from cuda M to coarser cuVEC
	cu_obj_quantity.extract_cuvec(display_vec_vc_vec_CUDA);
	//copy from coarser cuVEC to its cpu version
	display_vec_vc_vec_CUDA.copy_to_cpuvec(*pdisplay_vec_vc_vec);

	return true;
}

bool MeshDisplayCUDA::prepare_display(SZ3 n_quantity, Rect meshRect, double detail_level, mcu_VEC_VC(cuReal3)& cu_obj_quantity, mcu_VEC_VC(cuReal3)& cu_obj_quantity2)
{
	//the cu_obj managed quantity cannot be empty
	if (!cu_obj_quantity.size_cpu().dim() || !cu_obj_quantity2.size_cpu().dim()) return false;

	DBL3 hdisp(detail_level);
	SZ3 ndisp = round(meshRect / hdisp);

	if (ndisp.x < 1) ndisp.x = 1;
	if (ndisp.y < 1) ndisp.y = 1;
	if (ndisp.z < 1) ndisp.z = 1;
	if (ndisp.x > n_quantity.x)  ndisp.x = n_quantity.x;
	if (ndisp.y > n_quantity.y)  ndisp.y = n_quantity.y;
	if (ndisp.z > n_quantity.z)  ndisp.z = n_quantity.z;

	hdisp = meshRect / ndisp;

	//resize the gpu and cpu vecs if needed - they must always have the same size
	if (!pdisplay_vec_vc_vec->resize(hdisp, meshRect) || !pdisplay2_vec_vc_vec->resize(hdisp, meshRect)) return false;
	
	if (!display_vec_vc_vec_CUDA.resize(hdisp, meshRect)) return false;

	//1.

	//extract from cuda M to coarser cuVEC
	cu_obj_quantity.extract_cuvec(display_vec_vc_vec_CUDA);

	//copy from coarser cuVEC to its cpu version
	display_vec_vc_vec_CUDA.copy_to_cpuvec(*pdisplay_vec_vc_vec);

	//2.

	//extract from cuda M to coarser cuVEC
	cu_obj_quantity2.extract_cuvec(display_vec_vc_vec_CUDA);

	//copy from coarser cuVEC to its cpu version
	display_vec_vc_vec_CUDA.copy_to_cpuvec(*pdisplay2_vec_vc_vec);

	return true;
}

bool MeshDisplayCUDA::prepare_display(SZ3 n_quantity, Rect meshRect, double detail_level, mcu_VEC_VC(cuBReal)& cu_obj_quantity)
{
	//the cu_obj managed quantity cannot be empty
	if (!cu_obj_quantity.size_cpu().dim()) return false;

	DBL3 hdisp(detail_level);
	SZ3 ndisp = round(meshRect / hdisp);

	if (ndisp.x < 1) ndisp.x = 1;
	if (ndisp.y < 1) ndisp.y = 1;
	if (ndisp.z < 1) ndisp.z = 1;
	if (ndisp.x > n_quantity.x)  ndisp.x = n_quantity.x;
	if (ndisp.y > n_quantity.y)  ndisp.y = n_quantity.y;
	if (ndisp.z > n_quantity.z)  ndisp.z = n_quantity.z;

	hdisp = meshRect / ndisp;

	//resize the gpu and cpu vecs if needed - they must always have the same size
	if (!pdisplay_vec_vc_sca->resize(hdisp, meshRect)) return false;
	if (!display_vec_vc_sca_CUDA.resize(hdisp, meshRect)) return false;

	//extract from cuda M to coarser cuVEC
	cu_obj_quantity.extract_cuvec(display_vec_vc_sca_CUDA);
	//copy from coarser cuVEC to its cpu version
	display_vec_vc_sca_CUDA.copy_to_cpuvec(*pdisplay_vec_vc_sca);

	return true;
}

bool MeshDisplayCUDA::prepare_display(SZ3 n_quantity, Rect meshRect, double detail_level, mcu_VEC(cuReal3)& cu_obj_quantity)
{
	//the cu_obj managed quantity cannot be empty
	if (!cu_obj_quantity.size_cpu().dim()) return false;

	DBL3 hdisp(detail_level);
	SZ3 ndisp = round(meshRect / hdisp);

	if (ndisp.x < 1) ndisp.x = 1;
	if (ndisp.y < 1) ndisp.y = 1;
	if (ndisp.z < 1) ndisp.z = 1;
	if (ndisp.x > n_quantity.x)  ndisp.x = n_quantity.x;
	if (ndisp.y > n_quantity.y)  ndisp.y = n_quantity.y;
	if (ndisp.z > n_quantity.z)  ndisp.z = n_quantity.z;

	hdisp = meshRect / ndisp;

	//resize the gpu and cpu vecs if needed - they must always have the same size
	if (!pdisplay_vec_vec->resize(hdisp, meshRect)) return false;
	if (!display_vec_vec_CUDA.resize(hdisp, meshRect)) return false;

	//extract from cuda M to coarser cuVEC
	cu_obj_quantity.extract_cuvec(display_vec_vec_CUDA);
	//copy from coarser cuVEC to its cpu version
	display_vec_vec_CUDA.copy_to_cpuvec(*pdisplay_vec_vec);

	return true;
}

bool MeshDisplayCUDA::prepare_display(SZ3 n_quantity, Rect meshRect, double detail_level, mcu_VEC(cuReal3)& cu_obj_quantity, mcu_VEC(cuReal3)& cu_obj_quantity2)
{
	//the cu_obj managed quantity cannot be empty
	if (!cu_obj_quantity.size_cpu().dim() || !cu_obj_quantity2.size_cpu().dim()) return false;

	DBL3 hdisp(detail_level);
	SZ3 ndisp = round(meshRect / hdisp);

	if (ndisp.x < 1) ndisp.x = 1;
	if (ndisp.y < 1) ndisp.y = 1;
	if (ndisp.z < 1) ndisp.z = 1;
	if (ndisp.x > n_quantity.x)  ndisp.x = n_quantity.x;
	if (ndisp.y > n_quantity.y)  ndisp.y = n_quantity.y;
	if (ndisp.z > n_quantity.z)  ndisp.z = n_quantity.z;

	hdisp = meshRect / ndisp;

	//resize the gpu and cpu vecs if needed - they must always have the same size
	if (!pdisplay_vec_vec->resize(hdisp, meshRect) || !pdisplay2_vec_vec->resize(hdisp, meshRect)) return false;
	
	if (!display_vec_vec_CUDA.resize(hdisp, meshRect)) return false;

	//1.

	//extract from cuda M to coarser cuVEC
	cu_obj_quantity.extract_cuvec(display_vec_vec_CUDA);
	
	//copy from coarser cuVEC to its cpu version
	display_vec_vec_CUDA.copy_to_cpuvec(*pdisplay_vec_vec);

	//2.

	//extract from cuda M to coarser cuVEC
	cu_obj_quantity2.extract_cuvec(display_vec_vec_CUDA);

	//copy from coarser cuVEC to its cpu version
	display_vec_vec_CUDA.copy_to_cpuvec(*pdisplay2_vec_vec);

	return true;
}

bool MeshDisplayCUDA::prepare_display(SZ3 n_quantity, Rect meshRect, double detail_level, mcu_VEC(cuBReal)& cu_obj_quantity)
{
	//the cu_obj managed quantity cannot be empty
	if (!cu_obj_quantity.size_cpu().dim()) return false;

	DBL3 hdisp(detail_level);
	SZ3 ndisp = round(meshRect / hdisp);

	if (ndisp.x < 1) ndisp.x = 1;
	if (ndisp.y < 1) ndisp.y = 1;
	if (ndisp.z < 1) ndisp.z = 1;
	if (ndisp.x > n_quantity.x)  ndisp.x = n_quantity.x;
	if (ndisp.y > n_quantity.y)  ndisp.y = n_quantity.y;
	if (ndisp.z > n_quantity.z)  ndisp.z = n_quantity.z;

	hdisp = meshRect / ndisp;

	//resize the gpu and cpu vecs if needed - they must always have the same size
	if (!pdisplay_vec_sca->resize(hdisp, meshRect)) return false;
	if (!display_vec_sca_CUDA.resize(hdisp, meshRect)) return false;

	//extract from cuda M to coarser cuVEC
	cu_obj_quantity.extract_cuvec(display_vec_sca_CUDA);
	//copy from coarser cuVEC to its cpu version
	display_vec_sca_CUDA.copy_to_cpuvec(*pdisplay_vec_sca);

	return true;
}

#endif