#pragma once

#include "VEC.h"
#include "ProgramState.h"

////////////////////////////////////////////////////////////////////////////////////////////////// VEC_VC<VType>
//
// extends VEC with vector calculus operations

/////////////////////////////////////////////////////////////////////

template <typename VType> class CGSolve;

struct CMBNDInfo;

template <typename VType>
class VEC_VC : 
	public VEC<VType>,
	public ProgramState<VEC_VC<VType>, 
	std::tuple<SZ3, DBL3, Rect, std::vector<VType>, std::vector<int>, std::vector<int>, int,
	std::vector<VType>, std::vector<VType>, std::vector<VType>, std::vector<VType>, std::vector<VType>, std::vector<VType>,
	DBL2, DBL2, DBL2, DBL2, DBL2, DBL2, DBL2, DBL3, int, int, int, bool, bool>,
	std::tuple<>>
{

	friend CGSolve<VType>;

//the following are used as masks for ngbrFlags. 32 bits in total (4 bytes for an int)

//neighbor existence masks (+x, -x, +y, -y, +z, -z). Bits 0, 1, 2, 3, 4, 5
#define NF_NPX	1
#define NF_NNX	2
#define NF_NPY	4
#define NF_NNY	8
#define NF_NPZ	16
#define NF_NNZ	32

//Existance of at least one neighbor along a given axis : use as masks (test using &).
#define NF_NGBRX	(NF_NPX + NF_NNX)	//test bits 0 and 1
#define NF_NGBRY	(NF_NPY + NF_NNY)	//test bits 2 and 3
#define NF_NGBRZ	(NF_NPZ + NF_NNZ)   //test bits 4 and 5

//existence of both neighbors along axes x, y, z
//the use this check mask with value and see if the result is the same as the mask, e.g. if ((ngbrFlags[idx] & NF_BOTHX) == NF_BOTHX) { both neighbors are present }
#define NF_BOTHX	(NF_NPX + NF_NNX)
#define NF_BOTHY	(NF_NPY + NF_NNY)
#define NF_BOTHZ	(NF_NPZ + NF_NNZ)

//periodic boundary condition along x. Set at x sides only if there is a neighbor present at the other side - this is how we know which side to use, +x or -x : bit 6
#define NF_PBCX	64

//periodic boundary condition along y. Set at y sides only if there is a neighbor present at the other side - this is how we know which side to use, +y or -y : bit 7
#define NF_PBCY	128

//periodic boundary condition along z. Set at z sides only if there is a neighbor present at the other side - this is how we know which side to use, +z or -z : bit 8
#define NF_PBCZ	256

//mask for all pbc flags
#define NF_PBC (NF_PBCX + NF_PBCY + NF_PBCZ)

//mask to check for cell with zero value set : bit 9
#define NF_NOTEMPTY	512

//this is not necessarily an empty cell, but mark them to be skipped during computations for some algorithms (e.g. moving mesh algorithm where the ends of the magnetic mesh must not be updated by the ODE solver) : bit 10
#define NF_SKIPCELL	1024

//NOTE for faces : only presence of lower faces is marked, i.e. those which contain the cube origin
//If next cell cannot be checked because
//lower face with x surface normal (yz face) present, i.e. there is a non-empty cell which contains it : bit 11
#define NF_FACEX	2048
//lower face with y surface normal (xz face) present, i.e. there is a non-empty cell which contains it : bit 12
#define NF_FACEY	4096
//lower face with z surface normal (xy face) present, i.e. there is a non-empty cell which contains it : bit 13
#define NF_FACEZ	8192

//NOTE for edges : only presence of lower edges is marked, i.e. those which contain the cube origin
//lower x edge present : bit 14
#define NF_EDGEX	16384
//lower y edge present : bit 15
#define NF_EDGEY	32768
//lower z edge present : bit 16
#define NF_EDGEZ	65536

//off-axis neighbor at +x, +y, 0z (xy differentials) : bit 17
#define NF_XY_PXPY	131072
//off-axis neighbor at +x, -y, 0z (xy differentials) : bit 18
#define NF_XY_PXNY	262144
//off-axis neighbor at -x, +y, 0z (xy differentials) : bit 19
#define NF_XY_NXPY	524288
//off-axis neighbor at -x, -y, 0z (xy differentials) : bit 20
#define NF_XY_NXNY	1048576
//off-axis neighbor at +x, 0y, +z (xz differentials) : bit 21
#define NF_XZ_PXPZ	2097152
//off-axis neighbor at +x, 0y, -z (xz differentials) : bit 22
#define NF_XZ_PXNZ	4194304
//off-axis neighbor at -x, 0y, +z (xz differentials) : bit 23
#define NF_XZ_NXPZ	8388608
//off-axis neighbor at -x, 0y, -z (xz differentials) : bit 24
#define NF_XZ_NXNZ	16777216
//off-axis neighbor at 0x, +y, +z (yz differentials) : bit 25
#define NF_YZ_PYPZ	33554432
//off-axis neighbor at 0x, +y, -z (yz differentials) : bit 26
#define NF_YZ_PYNZ	67108864
//off-axis neighbor at 0x, -y, +z (yz differentials) : bit 27
#define NF_YZ_NYPZ	134217728
//off-axis neighbor at 0x, -y, -z (yz differentials) : bit 28
#define NF_YZ_NYNZ	268435456

//for off-axis neighbors stencil, indicate if mixed second order differential stencil is available, i.e. at least 2 columns must have at least 2 non-empty cells.
//Better to use 3 additional bits to speed up these checks rather than build it every time from the above bits.

//off-axis stencil available in XY plane : bit 29
#define NF_XY_OASTENCIL	536870912
//off-axis stencil available in XZ plane : bit 30
#define NF_XZ_OASTENCIL	1073741824
//off-axis stencil available in YZ plane : bit 31
#define NF_YZ_OASTENCIL	2147483648

//check for full off-axis stencil in a plane as (ngbrFlags[idx] & NF_XY_FULL) == NF_XY_FULL
#define NF_XY_FULL	(NF_XY_PXPY + NF_XY_PXNY + NF_XY_NXPY + NF_XY_NXNY)
#define NF_XZ_FULL	(NF_XZ_PXPZ + NF_XZ_PXNZ + NF_XZ_NXPZ + NF_XZ_NXNZ)
#define NF_YZ_FULL	(NF_YZ_PYPZ + NF_YZ_PYNZ + NF_YZ_NYPZ + NF_YZ_NYNZ)

//Extended flags

//Robin boundary conditions flags
//cell on positive x side of boundary : bit 0 -> use robin_nx values
#define NF2_ROBINPX	1
//cell on negative x side of boundary : bit 1 -> use robin_px values, etc.
#define NF2_ROBINNX	2
//cell on positive y side of boundary : bit 2
#define NF2_ROBINPY	4
//cell on negative y side of boundary : bit 3
#define NF2_ROBINNY	8
//cell on positive z side of boundary : bit 4
#define NF2_ROBINPZ	16
//cell on negative z side of boundary : bit 5
#define NF2_ROBINNZ	32
//flag Robin boundary with a void cell (use robin_v values) : bit 6
#define NF2_ROBINV	64

//mask for all Robin flags
#define NF2_ROBIN	(NF2_ROBINPX + NF2_ROBINNX + NF2_ROBINPY + NF2_ROBINNY + NF2_ROBINPZ + NF2_ROBINNZ + NF2_ROBINV)
//masks for Robin flags along the x, y, or z axes
#define NF2_ROBINX	(NF2_ROBINPX + NF2_ROBINNX)
#define NF2_ROBINY	(NF2_ROBINPY + NF2_ROBINNY)
#define NF2_ROBINZ	(NF2_ROBINPZ + NF2_ROBINNZ)

//these are used in conjunction with dirichlet vectors to indicate dirichlet boundary conditions should be used
//cell on +x side of boundary : bit 7
#define NF2_DIRICHLETPX	128
//cell on -x side of boundary : bit 8
#define NF2_DIRICHLETNX	256
//cell on +y side of boundary : bit 9
#define NF2_DIRICHLETPY	512
//cell on -y side of boundary : bit 10
#define NF2_DIRICHLETNY	1024
//cell on +z side of boundary : bit 11
#define NF2_DIRICHLETPZ	2048
//cell on -z side of boundary : bit 12
#define NF2_DIRICHLETNZ	4096

//masks for x, y, z directions for Dirichlet cells
#define NF2_DIRICHLETX (NF2_DIRICHLETPX + NF2_DIRICHLETNX)
#define NF2_DIRICHLETY (NF2_DIRICHLETPY + NF2_DIRICHLETNY)
#define NF2_DIRICHLETZ (NF2_DIRICHLETPZ + NF2_DIRICHLETNZ)
#define NF2_DIRICHLET (NF2_DIRICHLETX + NF2_DIRICHLETY + NF2_DIRICHLETZ)

//BITS 13 to 18 inclusive are used for halo flags in CUDA version. These are not needed here.

//composite media boundary cells (used to flag cells where boundary conditions must be applied). These flags are not set using set_ngbrFlags, but must be externally set
//cell on positive x side of boundary : bit 19
#define NF2_CMBNDPX	524288
//cell on negative x side of boundary : bit 20
#define NF2_CMBNDNX	1048576
//cell on positive y side of boundary : bit 21
#define NF2_CMBNDPY	2097152
//cell on negative y side of boundary : bit 22
#define NF2_CMBNDNY	4194304
//cell on positive z side of boundary : bit 23
#define NF2_CMBNDPZ	8388608
//cell on negative z side of boundary : bit 24
#define NF2_CMBNDNZ	16777216

//mask for all cmbnd flags
#define NF2_CMBND	(NF2_CMBNDPX + NF2_CMBNDNX + NF2_CMBNDPY + NF2_CMBNDNY + NF2_CMBNDPZ + NF2_CMBNDNZ)
//masks for cmbnd flags along the x, y, or z axes
#define NF2_CMBNDX	(NF2_CMBNDPX + NF2_CMBNDNX)
#define NF2_CMBNDY	(NF2_CMBNDPY + NF2_CMBNDNY)
#define NF2_CMBNDZ	(NF2_CMBNDPZ + NF2_CMBNDNZ)

//Halo bits 13 - 18 not used with VEC_VC, only with cuVEC_VC.
//If adding new bits, then add them from bit 19 onwards.

private:

	//NEIGHBORS

	//mark cells with various flags to indicate properties of neighboring cells
	//ngbrFlags2 defines additional flags. Only allocate memory if these additional flags are enabled - this is more memory efficient + I need to do it this way to keep older save files backward compatible.
	std::vector<int> ngbrFlags, ngbrFlags2;

	//if true then faces and edges flags in ngbrFlags are also calculated. turn on only if needed.
	bool calculate_faces_and_edges = false;

	int nonempty_cells = 0;

	//DIRICHLET

	//store dirichlet boundary conditions at mesh sides - only allocate memory as required
	//these vectors are of sizes equal to 1 cell deep at each respective side. dirichlet_nx are the dirichlet values at the -x side of the mesh, etc.
	std::vector<VType> dirichlet_nx, dirichlet_px, dirichlet_ny, dirichlet_py, dirichlet_nz, dirichlet_pz;

	//ROBIN

	//Robin boundary conditions values : diff_norm(u) = alpha * (u - VEC<VType>::h), where diff_norm means differential along surface normal (e.g. positive sign at +x boundary, negative sign at -x boundary).
	//alpha is a positive constant : robins_nx.i. Note, if this is zero then homogeneous Neumann boundary condition results.
	//h is a value (e.g. ambient temperature for heat equation): robins_nx.j
	//nx, px, etc... for mesh boundaries - use these when flagged with NF2_ROBINNX etc. and not flagged with NF2_ROBINV
	DBL2 robin_px, robin_nx;
	DBL2 robin_py, robin_ny;
	DBL2 robin_pz, robin_nz;
	//robin_v applies for boundary conditions at void cells - more precisely for cells next to a void cell. Use this when flagged with NF2_ROBINNX etc. and also flagged with NF2_ROBINV
	DBL2 robin_v;

	//CMBND

	//when cmbnd flags set (with set_cmbnd_flags), set this to true (clear_cmbnd_flags will set it to false). This is checked by use_extended_flags as cmbnd flags are set in ngbrFlags2.
	bool cmbnd_conditions_set = false;

	//PBC AND AUXILIARY

	//when used with moving mesh algorithms calls to shift... functions may be used. If the shift requested is smaller than the cellsize then we cannot perform the shift. 
	//Add it to shift_debt and on next shift call we might be able to shift the mesh values.
	DBL3 shift_debt;

	//Periodic boundary conditions for evaluating differential operators. If these are set then neighbor flags are calculated accordingly, and applied when evaluating operators.
	int pbc_x = 0;
	int pbc_y = 0;
	int pbc_z = 0;

private:

	//--------------------------------------------IMPORTANT FLAG MANIPULATION METHODS : VEC_VC_flags.h

	//set size of ngbrFlags to new_n also mapping shape from current size to new size (if current zero size set solid shape). Memory must be reserved in ngbrFlags to guarantee success. Also VEC<VType>::n should still have the old value : call this before changing it.
	void resize_ngbrFlags(SZ3 new_n);

	//initialization method for neighbor flags : set flags at size VEC<VType>::n, counting neighbors etc.
	//Set empty cell values using information in linked_vec (keep same shape) - this must have same rectangle
	template <typename LVType>
	void set_ngbrFlags(const VEC_VC<LVType> &linked_vec, bool do_reset = true);

	//initialization method for neighbor flags : set flags at size VEC<VType>::n, counting neighbors etc. Use current shape in ngbrFlags
	void set_ngbrFlags(bool do_reset = true);

	//calculate faces and edges flags - called by set_ngbrFlags if calculate_faces_and_edges is true
	void set_faces_and_edges_flags(void);

	//from NF2_DIRICHLET type flag and cell_idx return boundary value from one of the dirichlet vectors
	VType get_dirichlet_value(int dirichlet_flag, int cell_idx) const;

	//set robin flags from robin values and shape. Doesn't affect any other flags. Call from set_ngbrFlags after counting neighbors, and after setting robin values
	void set_robin_flags(void);

	//set pbc flags depending on set conditions and currently calculated flags - ngbrFlags must already be calculated before using this
	void set_pbc_flags(void);

	//check if we need to use ngbrFlags2 (allocate memory etc.)
	bool use_extended_flags(void);

	//---------------------------------------------MULTIPLE ENTRIES SETTERS - VEC SHAPE MASKS : VEC_VEC_shapemask.h

	//auxiliary function for generating shapes, where the shape is defined in shape_method
	//shape_method takes two parameters: distance from shape centre and dimensions of shape; if point is within shape then return true
	//generate shape at centre_pos with given rotation
	//make given number of repetitions, with displacements between repetitions
	//shape set using the indicated method (or, not, xor) and default_value
	void shape_setter(std::function<bool(DBL3, DBL3)>& shape_method, MeshShape shape, VType default_value);

	//set composite shape: differs slightly from single elementary shape setter: this adds the composite shape into the mesh, so any subtractive shapes are only subtracted when forming the composite shape, not subtracted from the mesh
	void shape_setter(std::vector<std::function<bool(DBL3, DBL3)>> shape_methods, std::vector<MeshShape> shapes, VType default_value);

	//similar to shape_setter, but sets value in composite shape where both the mesh and composite shapes are not empty
	void shape_valuesetter(std::vector<std::function<bool(DBL3, DBL3)>> shape_methods, std::vector<MeshShape> shapes, VType value);

	//get average value from composite shape
	VType shape_valuegetter(std::vector<std::function<bool(DBL3, DBL3)>> shape_methods, std::vector<MeshShape> shapes);

public:

	//--------------------------------------------CONSTRUCTORS : VEC_VC_mng.h

	VEC_VC(void);

	VEC_VC(const SZ3& n_);

	VEC_VC(const DBL3& h_, const Rect& rect_);

	VEC_VC(const DBL3& h_, const Rect& rect_, VType value);

	~VEC_VC() {}

	//implement ProgramState method
	void RepairObjectState() 
	{ 
		//any mesh VEC<VType>::transfer info will have to be remade
		VEC<VType>::transfer.clear(); 
	}

	//--------------------------------------------SPECIAL DATA ACCESS (typically used for copy to/from cuVECs)

	std::vector<int>& ngbrFlags_ref(void) { return ngbrFlags; }
	std::vector<int>& ngbrFlags2_ref(void) { return ngbrFlags2; }

	std::vector<VType>& dirichlet_px_ref(void) { return dirichlet_px; }
	std::vector<VType>& dirichlet_nx_ref(void) { return dirichlet_nx; }
	std::vector<VType>& dirichlet_py_ref(void) { return dirichlet_py; }
	std::vector<VType>& dirichlet_ny_ref(void) { return dirichlet_ny; }
	std::vector<VType>& dirichlet_pz_ref(void) { return dirichlet_pz; }
	std::vector<VType>& dirichlet_nz_ref(void) { return dirichlet_nz; }

	int& nonempty_cells_ref(void) { return nonempty_cells; }

	DBL2& robin_px_ref(void) { return robin_px; }
	DBL2& robin_nx_ref(void) { return robin_nx; }
	DBL2& robin_py_ref(void) { return robin_py; }
	DBL2& robin_ny_ref(void) { return robin_ny; }
	DBL2& robin_pz_ref(void) { return robin_pz; }
	DBL2& robin_nz_ref(void) { return robin_nz; }
	DBL2& robin_v_ref(void) { return robin_v; }

	DBL3& shift_debt_ref(void) { return shift_debt; }

	bool& cmbnd_conditions_set_ref(void) { return cmbnd_conditions_set; }

	bool& calculate_faces_and_edges_ref(void) { return calculate_faces_and_edges; }

	int& pbc_x_ref(void) { return pbc_x; }
	int& pbc_y_ref(void) { return pbc_y; }
	int& pbc_z_ref(void) { return pbc_z; }

	//--------------------------------------------SIZING : VEC_VC_mng.h

	//sizing methods return true or false (failed to resize) - if failed then no changes made

	//resize and set shape using linked vec
	template <typename LVType>
	bool resize(const SZ3& new_n, const VEC_VC<LVType> &linked_vec);
	//resize but keep shape
	bool resize(const SZ3& new_n);

	//resize and set shape using linked vec
	template <typename LVType>
	bool resize(const DBL3& new_h, const Rect& new_rect, const VEC_VC<LVType> &linked_vec);
	//resize but keep shape
	bool resize(const DBL3& new_h, const Rect& new_rect);

	//set value and shape from linked vec
	template <typename LVType>
	bool assign(const SZ3& new_n, VType value, const VEC_VC<LVType> &linked_vec);
	//set value but keep shape - empty cells will retain zero value : i.e. set value everywhere but in empty cells
	bool assign(const SZ3& new_n, VType value);

	//set value and shape from linked vec
	template <typename LVType>
	bool assign(const DBL3& new_h, const Rect& new_rect, VType value, const VEC_VC<LVType> &linked_vec);
	//set value but keep shape - empty cells will retain zero value : i.e. set value everywhere but in empty cells
	bool assign(const DBL3& new_h, const Rect& new_rect, VType value);

	void clear(void);

	void shrink_to_fit(void);

	//--------------------------------------------SUB-VECS : VEC_VC_subvec.h

	//get a copy from this VEC_VC, as a sub-VEC_VC defined by box; same cellsize maintained; any transfer data not copied.
	//dirichlet conditions not copied
	VEC_VC<VType> subvec(Box box);

	//--------------------------------------------PROPERTY CHECHING

	int is_pbc_x(void) const { return pbc_x; }
	int is_pbc_y(void) const { return pbc_y; }
	int is_pbc_z(void) const { return pbc_z; }

	//--------------------------------------------FLAG CHECKING : VEC_VC_flags.h

	int get_nonempty_cells(void) const { return nonempty_cells; }

	bool is_not_empty(int index) const { return (ngbrFlags[index] & NF_NOTEMPTY); }
	bool is_not_empty(const INT3& ijk) const { return (ngbrFlags[ijk.i + ijk.j*VEC<VType>::n.x + ijk.k*VEC<VType>::n.x*VEC<VType>::n.y] & NF_NOTEMPTY); }
	bool is_not_empty(const DBL3& rel_pos) const { return (ngbrFlags[int(rel_pos.x / VEC<VType>::h.x) + int(rel_pos.y / VEC<VType>::h.y) * VEC<VType>::n.x + int(rel_pos.z / VEC<VType>::h.z) * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY); }

	bool is_empty(int index) const { return !(ngbrFlags[index] & NF_NOTEMPTY); }
	bool is_empty(const INT3& ijk) const { return !(ngbrFlags[ijk.i + ijk.j*VEC<VType>::n.x + ijk.k*VEC<VType>::n.x*VEC<VType>::n.y] & NF_NOTEMPTY); }
	bool is_empty(const DBL3& rel_pos) const { return !(ngbrFlags[int(rel_pos.x / VEC<VType>::h.x) + int(rel_pos.y / VEC<VType>::h.y) * VEC<VType>::n.x + int(rel_pos.z / VEC<VType>::h.z) * VEC<VType>::n.x * VEC<VType>::n.y] & NF_NOTEMPTY); }

	//check if all cells intersecting the rectangle (absolute coordinates) are empty
	bool is_empty(const Rect& rectangle) const;
	//check if all cells intersecting the rectangle (absolute coordinates) are not empty
	bool is_not_empty(const Rect& rectangle) const;

	bool is_not_cmbnd(int index) const { return !(ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBND)); }
	bool is_not_cmbnd(const DBL3& rel_pos) const { return !(ngbrFlags2.size() && (ngbrFlags2[int(rel_pos.x / VEC<VType>::h.x) + int(rel_pos.y / VEC<VType>::h.y) * VEC<VType>::n.x + int(rel_pos.z / VEC<VType>::h.z) * VEC<VType>::n.x * VEC<VType>::n.y] & NF2_CMBND)); }
	bool is_not_cmbnd(const INT3& ijk) const { return !(ngbrFlags2.size() && (ngbrFlags2[ijk.i + ijk.j*VEC<VType>::n.x + ijk.k*VEC<VType>::n.x*VEC<VType>::n.y] & NF2_CMBND)); }

	bool is_cmbnd(int index) const { return (ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBND)); }
	bool is_cmbnd(const DBL3& rel_pos) const { return (ngbrFlags2.size() && (ngbrFlags2[int(rel_pos.x / VEC<VType>::h.x) + int(rel_pos.y / VEC<VType>::h.y) * VEC<VType>::n.x + int(rel_pos.z / VEC<VType>::h.z) * VEC<VType>::n.x * VEC<VType>::n.y] & NF2_CMBND)); }
	bool is_cmbnd(const INT3& ijk) const { return (ngbrFlags2.size() && (ngbrFlags2[ijk.i + ijk.j*VEC<VType>::n.x + ijk.k*VEC<VType>::n.x*VEC<VType>::n.y] & NF2_CMBND)); }

	bool is_cmbnd_px(int index) const { return (ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBNDPX)); }
	bool is_cmbnd_nx(int index) const { return (ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBNDNX)); }
	bool is_cmbnd_py(int index) const { return (ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBNDPY)); }
	bool is_cmbnd_ny(int index) const { return (ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBNDNY)); }
	bool is_cmbnd_pz(int index) const { return (ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBNDPZ)); }
	bool is_cmbnd_nz(int index) const { return (ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBNDNZ)); }

	bool is_cmbnd_x(int index) const { return (ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBNDX)); }
	bool is_cmbnd_y(int index) const { return (ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBNDY)); }
	bool is_cmbnd_z(int index) const { return (ngbrFlags2.size() && (ngbrFlags2[index] & NF2_CMBNDZ)); }

	bool is_skipcell(int index) const { return (ngbrFlags[index] & NF_SKIPCELL); }
	bool is_skipcell(const DBL3& rel_pos) const { return (ngbrFlags[int(rel_pos.x / VEC<VType>::h.x) + int(rel_pos.y / VEC<VType>::h.y) * VEC<VType>::n.x + int(rel_pos.z / VEC<VType>::h.z) * VEC<VType>::n.x * VEC<VType>::n.y] & NF_SKIPCELL); }
	bool is_skipcell(const INT3& ijk) const { return (ngbrFlags[ijk.i + ijk.j * VEC<VType>::n.x + ijk.k * VEC<VType>::n.x * VEC<VType>::n.y] & NF_SKIPCELL); }

	bool is_dirichlet(int index) const { return ngbrFlags2.size() && (ngbrFlags2[index] & NF2_DIRICHLET); }

	bool is_dirichlet_px(int index) const { return ngbrFlags2.size() && (ngbrFlags2[index] & NF2_DIRICHLETPX); }
	bool is_dirichlet_nx(int index) const { return ngbrFlags2.size() && (ngbrFlags2[index] & NF2_DIRICHLETNX); }
	bool is_dirichlet_py(int index) const { return ngbrFlags2.size() && (ngbrFlags2[index] & NF2_DIRICHLETPY); }
	bool is_dirichlet_ny(int index) const { return ngbrFlags2.size() && (ngbrFlags2[index] & NF2_DIRICHLETNY); }
	bool is_dirichlet_pz(int index) const { return ngbrFlags2.size() && (ngbrFlags2[index] & NF2_DIRICHLETPZ); }
	bool is_dirichlet_nz(int index) const { return ngbrFlags2.size() && (ngbrFlags2[index] & NF2_DIRICHLETNZ); }

	bool is_dirichlet_x(int index) const { return ngbrFlags2.size() && (ngbrFlags2[index] & NF2_DIRICHLETX); }
	bool is_dirichlet_y(int index) const { return ngbrFlags2.size() && (ngbrFlags2[index] & NF2_DIRICHLETY); }
	bool is_dirichlet_z(int index) const { return ngbrFlags2.size() && (ngbrFlags2[index] & NF2_DIRICHLETZ); }

	//are all neighbors available? (for 2D don't check the z neighbors)
	bool is_interior(int index) const { return (((ngbrFlags[index] & NF_BOTHX) == NF_BOTHX) && ((ngbrFlags[index] & NF_BOTHY) == NF_BOTHY) && (VEC<VType>::n.z == 1 || ((ngbrFlags[index] & NF_BOTHZ) == NF_BOTHZ))); }
	
	//are all neighbors in the xy plane available?
	bool is_plane_interior(int index) const { return (((ngbrFlags[index] & NF_BOTHX) == NF_BOTHX) && ((ngbrFlags[index] & NF_BOTHY) == NF_BOTHY)); }

	//return number of neighbors present (pbcs not taken into consideration)
	int ngbr_count(int index) const { return ((ngbrFlags[index] & NF_NPX) == NF_NPX) + ((ngbrFlags[index] & NF_NNX) == NF_NNX) + ((ngbrFlags[index] & NF_NPY) == NF_NPY) + ((ngbrFlags[index] & NF_NNY) == NF_NNY) + ((ngbrFlags[index] & NF_NPZ) == NF_NPZ) + ((ngbrFlags[index] & NF_NNZ) == NF_NNZ); }

	//populate neighbors (must have 6 elements) with indexes of neighbors for idx cell, setting -1 for cells which are not neighbors (empty or due to boundaries)
	//order is +x, -x, +y, -y, +z, -z
	void get_neighbors(int idx, std::vector<int>& neighbors);

	//--------------------------------------------SET CELL FLAGS - EXTERNAL USE : VEC_VC_flags.h

	//set dirichlet boundary conditions from surface_rect (must be a rectangle intersecting with one of the surfaces of this mesh) and value
	//return false on memory allocation failure only, otherwise return true even if surface_rect was not valid
	bool set_dirichlet_conditions(const Rect& surface_rect, VType value);

	//clear all dirichlet flags and vectors
	void clear_dirichlet_flags(void);

	//set pbc conditions : setting any to false clears flags
	void set_pbc(int pbc_x_, int pbc_y_, int pbc_z_);

	//clear all pbc flags : can also be achieved setting all flags to false in set_pbc but this one is more readable
	void clear_pbc(void);

	//clear all composite media boundary flags
	void clear_cmbnd_flags(void);

	//mark cells included in this rectangle (absolute coordinates) to be skipped during some computations (if status true, else clear the skip cells flags in this rectangle)
	void set_skipcells(const Rect& rectangle, bool status = true);

	//clear all skip cell flags
	void clear_skipcells(void);

	void set_robin_conditions(DBL2 robin_v_, DBL2 robin_px_, DBL2 robin_nx_, DBL2 robin_py_, DBL2 robin_ny_, DBL2 robin_pz_, DBL2 robin_nz_);

	//clear all Robin boundary conditions and values
	void clear_robin_conditions(void);

	//mark cell as not empty / empty : internal use only; routines that use these must finish with recalculating ngbrflags as neighbours will have changed
	void mark_not_empty(int index) { ngbrFlags[index] |= NF_NOTEMPTY; }
	void mark_empty(int index) { ngbrFlags[index] &= ~NF_NOTEMPTY; VEC<VType>::quantity[index] = VType(); }

	//similar to set_ngbrFlags, but do not reset externally set flags, usable at runtime if shape changes
	void set_ngbrFlags_shapeonly(void) { set_ngbrFlags(false); }

	//when enabled then set_faces_and_edges_flags method will be called by set_ngbrFlags every time it is executed
	//if false then faces and edges flags not calculated to avoid extra unnecessary initialization work
	void set_calculate_faces_and_edges(bool status) { calculate_faces_and_edges = status; if (calculate_faces_and_edges) set_faces_and_edges_flags(); }

	//--------------------------------------------CALCULATE COMPOSITE MEDIA BOUNDARY VALUES : VEC_VC_cmbnd.h

	//set cmbnd flags by identifying contacts with other vecs (listed in pVECs); this primary mesh index in that vector is given here as it needs to be stored in CMBNDInfo
	//if you set check_neighbors to false, neighbors on primary and secondary sides are not checked (i.e. cells 2 and -2)
	std::vector<CMBNDInfo> set_cmbnd_flags(int primary_mesh_idx, std::vector<VEC_VC<VType>*> &pVECs, bool check_neighbors = true);

	//calculate and set values at composite media boundary cells in this mesh for a given contacting mesh (the "secondary" V_sec) and given contact description (previously calculated using set_cmbnd_flags)
	//The values are calculated based on the continuity of a potential and flux normal to the interface. The potential is this VEC_VC, call it V, and the flux is the function f(V) = a_func + b_func * V', where the V' differential direction is perpendicular to the interface.
	//The second order differential of V perpendicular to the interface, V'', is also used and specified using the methods diff2.
	//Boundary with labelled cells either side, on the left is the secondary mesh, on the right is the primary mesh (i.e. the mesh for which we are setting boundary values using this method now): -2 -1 | 1 2 
	//a_func_sec is for the secondary side and takes a position for cell -1, a position shift to add to position to reach cell -2 and finally a stencil to use when obtaining values at cells -1 and -2 (use weighted_average)
	//b_func_sec similar
	//diff2_sec takes a position and stencil (since only need cell -1). It also takes a position shift vector perpendicular to the interface and pointing from primary to secondary.
	//a_func_pri takes indexes for cells 1 and 2. It also takes a position shift vector perpendicular to the interface and pointing from primary to secondary.
	//b_func_pri takes indexes for cells 1 and 2
	//diff2_pri takes index for cell 1. It also takes a position shift vector perpendicular to the interface and pointing from primary to secondary.
	//also need instances for the secondary and primary objects whose classes contain the above methods
	template <typename Owner>
	void set_cmbnd_continuous(VEC_VC<VType> &V_sec, CMBNDInfo& contact,
		std::function<VType(const Owner&, DBL3, DBL3, DBL3)> a_func_sec, std::function<VType(const Owner&, int, int, DBL3)> a_func_pri,
		std::function<double(const Owner&, DBL3, DBL3, DBL3)> b_func_sec, std::function<double(const Owner&, int, int)> b_func_pri,
		std::function<VType(const Owner&, DBL3, DBL3, DBL3)> diff2_sec, std::function<VType(const Owner&, int, DBL3)> diff2_pri,
		Owner& instance_sec, Owner& instance_pri);
	
	//calculate cmbnd values based on continuity of flux only. The potential is allowed to drop across the interface as:
	//f_sec(V) = f_pri(V) = A + B * delV, where f_sec and f_pri are the fluxes on the secondary and primary sides of the interface, and delV = V_pri - V_sec, the drop in potential across the interface.
	//Thus in addition to the functions in set_cmbnd_continuous we need two extra functions A, B.
	//Note, this reduces to the fully continuous case for B tending to infinity and A = VType(0)
	template <typename Owner, typename SOwner>
	void set_cmbnd_continuousflux(VEC_VC<VType> &V_sec, CMBNDInfo& contact,
		std::function<VType(const Owner&, DBL3, DBL3, DBL3)> a_func_sec, std::function<VType(const Owner&, int, int, DBL3)> a_func_pri,
		std::function<double(const Owner&, DBL3, DBL3, DBL3)> b_func_sec, std::function<double(const Owner&, int, int)> b_func_pri,
		std::function<VType(const Owner&, DBL3, DBL3, DBL3)> diff2_sec, std::function<VType(const Owner&, int, DBL3)> diff2_pri,
		std::function<VType(const SOwner&, int, int, DBL3, DBL3, DBL3, Owner&, Owner&)> A_func, std::function<double(const SOwner&, int, int, DBL3, DBL3, DBL3, Owner&, Owner&)> B_func,
		Owner& instance_sec, Owner& instance_pri, SOwner& instance_s);

	//most general case of composite media boundary conditions
	//calculate cmbnd values based on set boundary flux values; both the flux and potential is allowed to be discontinuous across the interface.
	//Fluxes at the interface are specified as: f_sec(V) = A_sec + B_sec * delVs, f_pri(V) = A_pri + B_pri * delVs, with directions from secondary to primary
	//B functions may return a double, or a DBL33 (3x3 matrix) if VType is DBL3 (Cartesian vector).
	//delVs = c_pri * V_pri - c_sec * V_sec, c are double values specified by given functions
	template <typename Owner, typename SOwner, typename BType,
		std::enable_if_t<(std::is_same<VType, double>::value && std::is_same<BType, double>::value) ||
		(std::is_same<VType, DBL3>::value && (std::is_same<BType, double>::value || std::is_same<BType, DBL33>::value))>* = nullptr>
	void set_cmbnd_discontinuous(
		VEC_VC<VType> &V_sec, CMBNDInfo& contact,
		std::function<VType(const Owner&, DBL3, DBL3, DBL3)> a_func_sec, std::function<VType(const Owner&, int, int, DBL3)> a_func_pri,
		std::function<double(const Owner&, DBL3, DBL3, DBL3)> b_func_sec, std::function<double(const Owner&, int, int)> b_func_pri,
		std::function<VType(const Owner&, DBL3, DBL3, DBL3)> diff2_sec, std::function<VType(const Owner&, int, DBL3)> diff2_pri,
		std::function<VType(const SOwner&, int, int, DBL3, DBL3, DBL3, Owner&, Owner&)> A_func_sec, std::function<BType(const SOwner&, int, int, DBL3, DBL3, DBL3, Owner&, Owner&)> B_func_sec,
		std::function<VType(const SOwner&, int, int, DBL3, DBL3, DBL3, Owner&, Owner&)> A_func_pri, std::function<BType(const SOwner&, int, int, DBL3, DBL3, DBL3, Owner&, Owner&)> B_func_pri,
		std::function<double(const Owner&, DBL3, DBL3)> c_func_sec, std::function<double(const Owner&, int)> c_func_pri,
		Owner& instance_sec, Owner& instance_pri, SOwner& instance_s);


	//--------------------------------------------MULTIPLE ENTRIES SETTERS - SHAPE CHANGERS : VEC_VC_shape.h

	//set value in box (i.e. in cells entirely included in box) - all cells become non-empty cells irrespective of value set
	void setbox(const Box& box, VType value = VType());

	//set value in rectangle (i.e. in cells intersecting the rectangle), where the rectangle is relative to this VEC's rectangle - all cells become non-empty cells irrespective of value set
	void setrect(const Rect& rectangle, VType value = VType());

	//delete rectangle, where the rectangle is relative to this VEC's rectangle, by setting empty cell values - all cells become empty cells
	void delrect(const Rect& rectangle, bool recalculate_flags = true);

	//mask values in cells using bitmap image : white -> empty cells. black -> keep values. Apply mask up to given z depth depending on grayscale value (all if default 0 value).
	bool apply_bitmap_mask(const std::vector<unsigned char>& bitmap, double zDepth = 0.0);

	//--------------------------------------------MULTIPLE ENTRIES SETTERS - SHAPE GENERATORS : VEC_VC_genshape.h, VEC_VC_Voronoi.h

	//roughen a mesh side (side = "-x", "x", "-y", "y", "-z", "z") to given depth (same units as VEC<VType>::h) with prng instantiated with given seed
	bool generate_roughside(std::string side, double depth, unsigned seed);

	//Roughen mesh top and bottom surfaces using a jagged pattern to given depth and peak spacing (same units as VEC<VType>::h) with prng instantiated with given seed.
	//Rough both top and bottom if sides is empty, else it should be either -z or z.
	bool generate_jagged_surfaces(double depth, double spacing, unsigned seed, std::string sides);

	//Generate 2D Voronoi cells with boundaries between cells set to empty
	bool generate_Voronoi2D_Grains(double spacing, unsigned seed);

	//Generate 3D Voronoi cells with boundaries between cells set to empty
	bool generate_Voronoi3D_Grains(double spacing, unsigned seed);

	//Generate uniform 2D Voronoi cells with boundaries between cells set to empty
	//There's an additional variation parameter (same units as spacing) which controls grain uniformity (e.g. if zero they are all square)
	bool generate_uVoronoi2D_Grains(double spacing, double variation, unsigned seed);

	//Generate uniform 3D Voronoi cells with boundaries between cells set to empty
	//There's an additional variation parameter (same units as spacing) which controls grain uniformity (e.g. if zero they are all square)
	bool generate_uVoronoi3D_Grains(double spacing, double variation, unsigned seed);

	//---------------------------------------------MULTIPLE ENTRIES SETTERS - VEC SHAPE MASKS : VEC_VEC_shapemask.h

	//shape_setter auxiliary method is used to set the actual shape (setshape = true) by default. set it to false if you just want to retrieve the shape definition method is std::function without actually setting it

	//disk
	std::function<bool(DBL3, DBL3)> shape_disk(MeshShape shape, VType default_value, bool setshape = true);

	//rectangle
	std::function<bool(DBL3, DBL3)> shape_rect(MeshShape shape, VType default_value, bool setshape = true);

	//triangle
	std::function<bool(DBL3, DBL3)> shape_triangle(MeshShape shape, VType default_value, bool setshape = true);

	//ellipsoid
	std::function<bool(DBL3, DBL3)> shape_ellipsoid(MeshShape shape, VType default_value, bool setshape = true);

	//pyramid
	std::function<bool(DBL3, DBL3)> shape_pyramid(MeshShape shape, VType default_value, bool setshape = true);

	//tetrahedron
	std::function<bool(DBL3, DBL3)> shape_tetrahedron(MeshShape shape, VType default_value, bool setshape = true);

	//cone
	std::function<bool(DBL3, DBL3)> shape_cone(MeshShape shape, VType default_value, bool setshape = true);

	//torus
	std::function<bool(DBL3, DBL3)> shape_torus(MeshShape shape, VType default_value, bool setshape = true);

	//set a composite shape using combination of the above elementary shapes
	void shape_set(std::vector<MeshShape> shapes, VType default_value);

	//set a composite shape using combination of the above elementary shapes but:
	//only set value in non-empty parts of mesh, which are also non-empty parts of shape
	//uses the shape_valuesetter auxiliary method
	void shape_setvalue(std::vector<MeshShape> shapes, VType value);

	//--------------------------------------------MULTIPLE ENTRIES SETTERS - OTHERS : VEC_VC_oper.h

	//exactly the same as assign value - do not use assign as it is slow (sets flags)
	void setnonempty(VType value = VType());

	//set value in non-empty cells only in given rectangle (relative coordinates)
	void setrectnonempty(const Rect& rectangle, VType value = VType());

	//set value in solid object only containing relpos
	void setobject(VType value, DBL3 relpos);

	//re-normalize all non-zero values to have the new magnitude (multiply by new_norm and divide by current magnitude)
	template <typename PType = decltype(GetMagnitude(std::declval<VType>()))>
	void renormalize(PType new_norm);

	//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions; from flags only copy the shape but not the boundary condition values or anything else - these are reset
	void copy_values(const VEC<VType>& copy_this, Rect dstRect = Rect(), Rect srcRect = Rect(), double multiplier = 1.0, bool recalculate_flags = true);
	void copy_values(const VEC_VC<VType>& copy_this, Rect dstRect = Rect(), Rect srcRect = Rect(), double multiplier = 1.0, bool recalculate_flags = true);

	//copy values from copy_this but keep current dimensions - if necessary map values from copy_this to local dimensions
	//can specify destination and source rectangles in relative coordinates
	//this is intended for VECs where copy_this cellsize is much larger than that in this VEC, and instead of setting all values the same, thermalize_func generator will generate values
	//e.g. this is useful for copying values from a micromagnetic mesh into an atomistic mesh, where the atomistic spins are generated according to a distribution setup in thermalize_func
	//thermalize_func returns the value to set, and takes parameters VType (value in the larger cell from copy_this which is being copied), and int, int (index of larger cell from copy_this which is being copied, and index of destination cell)
	void copy_values_thermalize(const VEC_VC<VType>& copy_this, std::function<VType(VType, int, int)>& thermalize_func, Rect dstRect = Rect(), Rect srcRect = Rect(), bool recalculate_flags = true);

	//shift along an axis - use this for moving mesh algorithms. Fast parallel code.
	void shift_x(double delta, const Rect& shift_rect, bool recalculate_flags = true);

	//shift along an axis - use this for moving mesh algorithms. Fast parallel code.
	void shift_y(double delta, const Rect& shift_rect, bool recalculate_flags = true);

	//--------------------------------------------ARITHMETIC OPERATIONS ON ENTIRE VEC : VEC_VC_arith.h

	//scale all stored values by the given constant
	void scale_values(double constant);

	//add values from add_this but keep current dimensions - if necessary map values from add_this to local dimensions
	void add_values(const VEC<VType>& add_this);
	void add_values(const VEC_VC<VType>& add_this);

	//subtract values from sub_this but keep current dimensions - if necessary map values from sub_this to local dimensions
	void sub_values(const VEC<VType>& sub_this);
	void sub_values(const VEC_VC<VType>& sub_this);

	void operator+=(double constant);
	void operator-=(double constant);
	void operator*=(double constant);
	void operator/=(double constant);

	//--------------------------------------------AVERAGING/SUMMING OPERATIONS : VEC_VC_avg.h

	//overload VEC method : use NF_NOTEMPTY flags instead here
	VType average_nonempty(const Box& box) const;
	//average over non-empty cells over given rectangle (relative to this VEC's VEC<VType>::rect)
	VType average_nonempty(const Rect& rectangle = Rect()) const;

	//parallel processing versions - do not call from parallel code!!!
	VType average_nonempty_omp(const Box& box) const;
	VType average_nonempty_omp(const Rect& rectangle = Rect()) const;

	//summing functions - do not call from parallel code!!!
	VType sum_nonempty_omp(const Box& box) const;
	VType sum_nonempty_omp(const Rect& rectangle = Rect()) const;

	//get average value in composite shape (defined in VEC_VC_shapemask.h)
	VType shape_getaverage(std::vector<MeshShape> shapes);

	//--------------------------------------------NUMERICAL PROPERTIES : VEC_VC_nprops.h

	template <typename PType = decltype(GetMagnitude(std::declval<VType>()))>
	VAL2<PType> get_minmax(const Box& box) const;
	template <typename PType = decltype(GetMagnitude(std::declval<VType>()))>
	VAL2<PType> get_minmax(const Rect& rectangle = Rect()) const;

	template <typename PType = decltype(GetMagnitude(std::declval<VType>()))>
	VAL2<PType> get_minmax_component_x(const Box& box) const;
	template <typename PType = decltype(GetMagnitude(std::declval<VType>()))>
	VAL2<PType> get_minmax_component_x(const Rect& rectangle = Rect()) const;

	template <typename PType = decltype(GetMagnitude(std::declval<VType>()))>
	VAL2<PType> get_minmax_component_y(const Box& box) const;
	template <typename PType = decltype(GetMagnitude(std::declval<VType>()))>
	VAL2<PType> get_minmax_component_y(const Rect& rectangle = Rect()) const;

	template <typename PType = decltype(GetMagnitude(std::declval<VType>()))>
	VAL2<PType> get_minmax_component_z(const Box& box) const;
	template <typename PType = decltype(GetMagnitude(std::declval<VType>()))>
	VAL2<PType> get_minmax_component_z(const Rect& rectangle = Rect()) const;

	//--------------------------------------------SPECIAL NUMERICAL PROPERTIES : VEC_VC_nprops.h

	//Robin value is of the form alpha * (Tb - Ta). alpha and Ta are known from values set robin_nx, robin_px, ...
	//Tb is quantity value at boundary. Here we will return Robin value for x, y, z axes for which any shift component is nonzero (otherwise zero for that component)
	//e.g. if shift.x is non-zero then Tb value is obtained at rel_pos + (shift.x, 0, 0) using extrapolation from values at rel_pos and rel_pos - (shift.x, 0, 0) -> both these values should be inside the mesh, else return zero.
	DBL3 get_robin_value(const DBL3& rel_pos, const DBL3& shift);

	//for given cell index, find if any neighboring cells are empty and get distance (shift) valuer to them along each axis
	//if any shift is zero this means both cells are present either side, or both are missing
	//NOTE : this is intended to be used with get_robin_value method to determine the shift value, and rel_pos will be position corresponding to idx
	DBL3 get_shift_to_emptycell(int idx);

	//--------------------------------------------EXTRACT A LINE PROFILE : VEC_VC_extract.h

	//for all these methods use wrap-around when extracting profiles if points on profile exceed mesh boundaries

	//extract profile in profile_storage temporary vector, returned through reference: extract starting at start in the direction end - step, with given step; use average to extract profile with given stencil, excluding zero points (assumed empty)
	//all coordinates are relative positions
	std::vector<VType>& extract_profile(DBL3 start, DBL3 end, double step, DBL3 stencil);

	//--------------------------------------------OPERATORS and ALGORITHMS

	//----LAPLACE OPERATOR : VEC_VC_del.h

	//calculate Laplace operator at cell with given index. Use Neumann boundary conditions (homogeneous).
	//Returns zero at composite media boundary cells.
	VType delsq_neu(int idx) const;

	//calculate Laplace operator at cell with given index. Use non-homogeneous Neumann boundary conditions with the specified boundary differential.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	//Returns zero at composite media boundary cells.
	VType delsq_nneu(int idx, const VAL3<VType>& bdiff) const;

	//calculate Laplace operator at cell with given index. Use Dirichlet conditions if set, else Neumann boundary conditions (homogeneous).
	//Returns zero at composite media boundary cells.
	VType delsq_diri(int idx) const;

	//calculate Laplace operator at cell with given index. Use Dirichlet conditions if set, else non-homogeneous Neumann boundary conditions.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	//Returns zero at composite media boundary cells.
	VType delsq_diri_nneu(int idx, const VAL3<VType>& bdiff) const;

	//calculate Laplace operator at cell with given index. Use Robin boundary conditions (defaulting to Neumann if not set).
	//Returns zero at composite media boundary cells.
	//The K constant is used in Robin boundary condition calculations, where -K*diff_norm(T) = alpha*(Tboundary - Tambient) is the flux normal to the boundary - K is the thermal conductivity in the heat equation
	VType delsq_robin(int idx, double K) const;

	//----GRADIENT OPERATOR : VEC_VC_grad.h

	//gradient operator. Use Neumann boundary conditions (homogeneous).
	//Can be used at composite media boundaries where sided differentials will be used instead.
	VAL3<VType> grad_neu(int idx) const;

	//gradient operator. Use non-homogeneous Neumann boundary conditions.
	//Can be used at composite media boundaries where sided differentials will be used instead.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	VAL3<VType> grad_nneu(int idx, const VAL3<VType>& bdiff) const;

	//gradient operator. Use Dirichlet conditions if set, else Neumann boundary conditions (homogeneous).
	//Can be used at composite media boundaries where sided differentials will be used instead.
	VAL3<VType> grad_diri(int idx) const;

	//gradient operator. Use Dirichlet conditions if set, else non-homogeneous Neumann boundary conditions.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	//Can be used at composite media boundaries where sided differentials will be used instead.
	VAL3<VType> grad_diri_nneu(int idx, const VAL3<VType>& bdiff) const;

	//gradient operator. Use sided differentials at boundaries (including at composite media boundaries)
	VAL3<VType> grad_sided(int idx) const;

	//----DIVERGENCE OPERATOR : VEC_VC_div.h

	//divergence operator. Use Neumann boundary conditions (homogeneous).
	//Can be used at composite media boundaries where sided differentials will be used instead.
	//div operator can be applied if VType is a VAL3<Type>, returning Type
	double div_neu(int idx) const;
	
	//divergence operator. Use non-homogeneous Neumann boundary conditions.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	//Can be used at composite media boundaries where sided differentials will be used instead.
	//div operator can be applied if VType is a VAL3<Type>, returning Type
	double div_nneu(int idx, const VAL3<VType>& bdiff) const;

	//divergence operator. Use Dirichlet conditions if set, else Neumann boundary conditions (homogeneous).
	//Can be used at composite media boundaries where sided differentials will be used instead.
	//div operator can be applied if VType is a VAL3<Type>, returning Type
	double div_diri(int idx) const;

	//divergence operator. Use Dirichlet conditions if set, else non-homogeneous Neumann boundary conditions.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	//Can be used at composite media boundaries where sided differentials will be used instead.
	//div operator can be applied if VType is a VAL3<Type>, returning Type
	double div_diri_nneu(int idx, const VAL3<VType>& bdiff) const;

	//divergence operator. Use sided differentials (also at composite media boundaries)
	//div operator can be applied if VType is a VAL3<Type>, returning Type
	double div_sided(int idx) const;
	
	//divergence operator of epsilon3(VEC<VType>::quantity[idx]), i.e. multiply this VEC_VC by the unit antisymmetric tensor of rank3 before taking divergence. 
	//Use Neumann boundary conditions (homogeneous).
	//Can be used at composite media boundaries where sided differentials will be used instead.
	VType diveps3_neu(int idx) const;

	//divergence operator of epsilon3(VEC<VType>::quantity[idx]), i.e. multiply this VEC_VC by the unit antisymmetric tensor of rank3 before taking divergence. 
	//Use Dirichlet conditions if set, else Neumann boundary conditions(homogeneous).
	//Can be used at composite media boundaries where sided differentials will be used instead.
	VType diveps3_diri(int idx) const;

	//divergence operator of epsilon3(VEC<VType>::quantity[idx]), i.e. multiply this VEC_VC by the unit antisymmetric tensor of rank3 before taking divergence. 
	//Use sided differentials (also at composite media boundaries)
	VType diveps3_sided(int idx) const;

	//----CURL OPERATOR : VEC_VC_curl.h

	//curl operator. Use Neumann boundary conditions (homogeneous).
	//Can be used at composite media boundaries where sided differentials will be used instead.
	//can only be applied if VType is a VAL3
	VType curl_neu(int idx) const;

	//curl operator. Use non-homogeneous Neumann boundary conditions.
	//Can be used at composite media boundaries where sided differentials will be used instead.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	//can only be applied if VType is a VAL3
	VType curl_nneu(int idx, const VAL3<VType>& bdiff) const;

	//curl operator. Use Dirichlet conditions if set, else Neumann boundary conditions (homogeneous).
	//Can be used at composite media boundaries where sided differentials will be used instead.
	//can only be applied if VType is a VAL3
	VType curl_diri(int idx) const;

	//curl operator. Use Dirichlet conditions if set, else non-homogeneous Neumann boundary conditions.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	//Can be used at composite media boundaries where sided differentials will be used instead.
	//can only be applied if VType is a VAL3
	VType curl_diri_nneu(int idx, const VAL3<VType>& bdiff) const;

	//curl operator. Use sided differentials at boundaries (including at composite media boundaries)
	//can only be applied if VType is a VAL3
	VType curl_sided(int idx) const;

	//---- SECOND ORDER DIFFERENTIALS : VEC_VC_diff2.h

	//homogeneous second order.
	//Use Neumann boundary conditions.
	//Returns zero at composite media boundary cells
	VType dxx_neu(int idx) const;
	VType dyy_neu(int idx) const;
	VType dzz_neu(int idx) const;

	//Use non-homogeneous Neumann boundary conditions.
	//Returns zero at composite media boundary cells
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	VType dxx_nneu(int idx, const VAL3<VType>& bdiff) const;
	VType dyy_nneu(int idx, const VAL3<VType>& bdiff) const;
	VType dzz_nneu(int idx, const VAL3<VType>& bdiff) const;

	//Use Dirichlet boundary conditions, else Neumann boundary conditions (homogeneous).
	//Returns zero at composite media boundary cells.
	VType dxx_diri(int idx) const;
	VType dyy_diri(int idx) const;
	VType dzz_diri(int idx) const;

	//Use Dirichlet boundary conditions, else non-homogeneous Neumann boundary conditions.
	//Returns zero at composite media boundary cells.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	VType dxx_diri_nneu(int idx, const VAL3<VType>& bdiff) const;
	VType dyy_diri_nneu(int idx, const VAL3<VType>& bdiff) const;
	VType dzz_diri_nneu(int idx, const VAL3<VType>& bdiff) const;

	//Use Robin boundary conditions (defaulting to Neumann if not set).
	//Returns zero at composite media boundary cells.
	//The K constant is used in Robin boundary condition calculations, where -K*diff_norm(T) = alpha*(Tboundary - Tambient) is the flux normal to the boundary - K is the thermal conductivity in the heat equation
	VType dxx_robin(int idx, double K) const;
	VType dyy_robin(int idx, double K) const;
	VType dzz_robin(int idx, double K) const;

	//mixed second order

	//Use Neumann boundary conditions(homogeneous).
	//Can be used at composite media boundaries where sided differentials will be used instead.
	//dxy same as dyx
	VType dxy_neu(int idx) const;

	//dxz same as dzx
	VType dxz_neu(int idx) const;

	//dyz same as dzy
	VType dyz_neu(int idx) const;

	//----NEIGHBOR SUM : VEC_VC_ngbrsum.h

	//calculate 6-point neighbor sum at given index
	//missing neighbors not added, including at boundaries, but taking into account pbc
	VType ngbr_sum(int idx) const;

	//same as ngbr_sum but sum normalised values only; for scalar values this is a trivial operation, but for vectors it's not.
	VType ngbr_dirsum(int idx) const;

	//calculate 6-point anisotropic neighbor sum at given index as rij x Vj over j points neighboring the point i at this index.
	//missing neighbors not added, including at boundaries, but taking into account pbc
	//only used if VType is a VAL3
	VType anisotropic_ngbr_sum(int idx) const;

	//same as anisotropic_ngbr_sum but sum normalised values only; for scalar values this is a trivial operation, but for vectors it's not.
	VType anisotropic_ngbr_dirsum(int idx) const;

	//calculate 6-point anisotropic neighbor sum at given index as (rij x z) x Vj over j points neighboring the point i at this index.
	//missing neighbors not added, including at boundaries, but taking into account pbc
	//only used if VType is a VAL3
	VType zanisotropic_ngbr_sum(int idx) const;
	//calculate 6-point anisotropic neighbor sum at given index as (rij x y) x Vj over j points neighboring the point i at this index.
	VType yanisotropic_ngbr_sum(int idx) const;
	//calculate 6-point anisotropic neighbor sum at given index as (rij x x) x Vj over j points neighboring the point i at this index.
	VType xanisotropic_ngbr_sum(int idx) const;

	//same as zanisotropic_ngbr_sum but sum normalised values only; for scalar values this is a trivial operation, but for vectors it's not.
	VType zanisotropic_ngbr_dirsum(int idx) const;
	VType yanisotropic_ngbr_dirsum(int idx) const;
	VType xanisotropic_ngbr_dirsum(int idx) const;

	//----LAPLACE / POISSON EQUATION : VEC_VC_solve.h

	//Take one SOR iteration for Laplace equation on this VEC. Return error (maximum change in VEC<VType>::quantity from one iteration to the next)
	//Dirichlet boundary conditions used, defaulting to Neumann boundary conditions where not set, and composite media boundary cells skipped (use boundary conditions to set cmbnd cells after calling this)
	//Return un-normalized error (maximum change in VEC<VType>::quantity from one iteration to the next) - first - and maximum value  -second - divide them to obtain normalized error
	DBL2 IterateLaplace_SOR(double relaxation_param = 1.9);

	//For Poisson equation we need a function to specify the RHS of the equation delsq V = F : use Poisson_RHS
	//F must be a member const method of Owner taking an index value (the index ranges over this VEC) and returning a double value : F(index) evaluated at the index-th cell.
	//Return un-normalized error (maximum change in VEC<VType>::quantity from one iteration to the next) - first - and maximum value  -second - divide them to obtain normalized error
	//Dirichlet boundary conditions used, defaulting to Neumann boundary conditions where not set, and composite media boundary cells skipped (use boundary conditions to set cmbnd cells after calling this)
	template <typename Owner>
	DBL2 IteratePoisson_SOR(std::function<VType(const Owner&, int)> Poisson_RHS, Owner& instance, double relaxation_param = 1.9);

	//This solves delsq V = F + M * V : For M use Tensor_RHS (For VType double M returns type double, For VType DBL3 M returns DBL33)
	//For Poisson equation we need a function to specify the RHS of the equation delsq V = F : use Poisson_RHS
	//F must be a member const method of Owner taking an index value (the index ranges over this VEC) and returning a double value : F(index) evaluated at the index-th cell.
	//Return un-normalized error (maximum change in VEC<VType>::quantity from one iteration to the next) - first - and maximum value  -second - divide them to obtain normalized error
	//Dirichlet boundary conditions used, defaulting to Neumann boundary conditions where not set, and composite media boundary cells skipped (use boundary conditions to set cmbnd cells after calling this)
	template <typename Owner, typename MType>
	DBL2 IteratePoisson_SOR(std::function<VType(const Owner&, int)> Poisson_RHS, std::function<MType(const Owner&, int)> Tensor_RHS, Owner& instance, double relaxation_param = 1.9);

	//Poisson equation solved using SOR, but using non-homogeneous Neumann boundary condition - this is evaluated using the bdiff call-back method.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	//Return un-normalized error (maximum change in VEC<VType>::quantity from one iteration to the next) - first - and maximum value  -second - divide them to obtain normalized error
	template <typename Owner>
	DBL2 IteratePoisson_SOR(std::function<VType(const Owner&, int)> Poisson_RHS, std::function<VAL3<VType>(const Owner&, int)> bdiff, Owner& instance, double relaxation_param = 1.9);

	//This solves delsq V = F + M * V : For M use Tensor_RHS (For VType double M returns type double, For VType DBL3 M returns DBL33)
	//Poisson equation solved using SOR, but using non-homogeneous Neumann boundary condition - this is evaluated using the bdiff call-back method.
	//NOTE : the boundary differential is specified with 3 components, one for each of +x, +y, +z surface normal directions
	//Return un-normalized error (maximum change in VEC<VType>::quantity from one iteration to the next) - first - and maximum value  -second - divide them to obtain normalized error
	template <typename Owner, typename MType>
	DBL2 IteratePoisson_SOR(std::function<VType(const Owner&, int)> Poisson_RHS, std::function<MType(const Owner&, int)> Tensor_RHS, std::function<VAL3<VType>(const Owner&, int)> bdiff, Owner& instance, double relaxation_param = 1.9);
};