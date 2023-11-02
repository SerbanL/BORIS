#pragma once

#include "BorisLib.h"

#include "ErrorHandler.h"
#include "OVF2_Handlers.h"
#include "SimSharedData.h"

#include "Boris_Enums_Defs.h"
#include "ParametersDefs.h"

#if COMPILECUDA == 1
#include "MaterialParameterPolicyCUDA.h"
#include "mGPUConfig.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//Class holding a single material parameter with any associated temperature dependence and spatial variation.
//This is designed to be used just as the type stored (PType) would be used on its own in mathematical equations.
//
//PType is the parameter type
//SType is the type used for spatial variation.
//e.g. most parameters just use a scalar for the spatial variation, but there are special cases where we need a vector, e.g. DBL3 for anisotropy symmetry axes, specifying rotations for spatial variation.
//
//This is achieved by updating the parameter output value (current_value) if the temperature changes.
//1. For uniform temperature just call update(Temperature) whenever the uniform base temperature is changed
//2. For non-uniform temperature the value must be updated in every computational cell before being used, using current_value = get(Temperature). 
//This should be done by an external variadic template function if there are multiple parameters to update.

template <typename PType, typename SType>
class MatP :
	public SimulationSharedData,
	public ProgramState<MatP<PType, SType>, 
	std::tuple<
	PType, PType,
	std::vector<PType>, std::vector<PType>,
	std::vector<double>, std::vector<double>, std::vector<double>, 
	VEC<SType>, 
	std::string, int, std::string, 
	TEquation<double>, TEquation<double, double, double, double>>, 
	std::tuple<>>
{

private:

#if COMPILECUDA == 1
	//when CUDA enabled this will point to the cu_obj managed MatPCUDA object which mirrors this cpu memory parameter in gpu memory.
	//Pointer assigned and nulled in MeshParamsCUDA ctor and dtor. Use it to update the MatPCUDA parameter accordingly when this parameter changes (if not nullptr).
	//To use it, cast it to MatPCUDA<conversion(PType)>. The conversion is done to a cuBReal variant, e.g. double/float to cuBReal, DBL3/FLT3 to cuReal3 etc.
	void *p_cu_obj_mpcuda = nullptr;
#endif

	//the parameter value at 0K : when constructing it, this is the value to set. This is also the value displayed/changed in the console. Temperaure dependence set separately.
	PType value_at_0K;
	//if using multiple sub-lattices these are the values to use (i.e. use values_at_0K instead of value_at_0K)
	std::vector<PType> values_at_0K;

	//this is the value that is actually read out and is obtained from value_at_0K depending on the set temperature
	PType current_value;
	//if using multiple sub-lattices these are the values to use (i.e. use current_values instead of current_value)
	std::vector<PType> current_values;

	//NOTE : scalings apply to all sub-lattices the same

	//array describing temperature scaling of value_at_0K. The index in this vector is the temperature value in K, i.e. values are set at 1K increments starting at 0K
	std::vector<double> t_scaling;

	//additional components for dual or vector quantities if set
	std::vector<double> t_scaling_y;
	std::vector<double> t_scaling_z;

	//temperature scaling equation, if set : takes only the parameter T (temperature); Tc constant : Curie temperature, Tb constant : base temperature
	TEquation<double> Tscaling_eq;

	//pointers to special functions, calculated elsewhere
	std::shared_ptr<Funcs_Special> pCurieWeiss = nullptr;
	std::shared_ptr<Funcs_Special> pLongRelSus = nullptr;
	std::shared_ptr<Funcs_Special> pCurieWeiss1 = nullptr;
	std::shared_ptr<Funcs_Special> pCurieWeiss2 = nullptr;
	std::shared_ptr<Funcs_Special> pLongRelSus1 = nullptr;
	std::shared_ptr<Funcs_Special> pLongRelSus2 = nullptr;
	std::shared_ptr<Funcs_Special> pAlpha1 = nullptr;
	std::shared_ptr<Funcs_Special> pAlpha2 = nullptr;

#if COMPILECUDA == 1
	//as above but CUDA version : need to keep this here not in MatPCUDA since MatPCUDA is cu_obj managed and TEquationCUDA is meant to be held in cpu memory
	//instead you need to obtain the top-level ManagedFunctionCUDA from TEquationCUDA and store that in MatPCUDA using a pointer; then it can be used in device functions.
	//object made here with set_p_cu_obj_mpcuda, invoked by MeshParamsCUDA ctor, and destroyed with null_p_cu_obj_mpcuda, invoked by MeshParamsCUDA dtor
	mTEquationCUDA<cuBReal>* pTscaling_CUDAeq = nullptr;

	//CUDA versions of special functions, calculated elsewhere
	mcu_SpecialFunc* pCurieWeiss_CUDA = nullptr;
	mcu_SpecialFunc* pLongRelSus_CUDA = nullptr;
	mcu_SpecialFunc* pCurieWeiss1_CUDA = nullptr;
	mcu_SpecialFunc* pCurieWeiss2_CUDA = nullptr;
	mcu_SpecialFunc* pLongRelSus1_CUDA = nullptr;
	mcu_SpecialFunc* pLongRelSus2_CUDA = nullptr;
	mcu_SpecialFunc* pAlpha1_CUDA = nullptr;
	mcu_SpecialFunc* pAlpha2_CUDA = nullptr;
#endif

	//VEC with spatial dependence scaling of value_at_0K. The VEC rect must match the rect of the mesh to which this MatP applies.
	VEC<SType> s_scaling;

	//spatial variation scaling equation : function of x, y, z, t (stage time); takes parameters Lx, Ly, Lz
	TEquation<double, double, double, double> Sscaling_eq;

#if COMPILECUDA == 1
	//see comments about pTscaling_CUDAeq
	//object made here int set_p_cu_obj_mpcuda, invoked by MeshParamsCUDA ctor, and destroyed with null_p_cu_obj_mpcuda, invoked by MeshParamsCUDA dtor
	mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>* pSscaling_CUDAeq = nullptr;
#endif

	//temperature scaling setting info. this is used if the temperature dependence is set using an array, and this would contain the file name from which the temperature dependence was loaded
	std::string t_scaling_info = temperature_dependence_type(MATPTDEP_NONE);

	//value from MATPVAR_ enum
	int s_scaling_type = MATPVAR_NONE;

	//spatial scaling setting info : text with ";" separators. First field is the scaling set type (e.g. none, custom, random, jagged, etc.); The other fields are parameters for spatial generators.
	std::string s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_NONE);

private:

	//---------Output value update : CUDA helpers

#if COMPILECUDA == 1
	//update just the value at 0K and current value in the corresponding MatPCUDA
	void update_cuda_value(void);

	//full update of corresponding MatPCUDA
	void update_cuda_object(void);
#endif

	//---------Set spatial dependence : generator methods

	//custom, set from an image file : coefficients vary from 0 : black to 1 : white
	bool set_s_custom(DBL3 h, Rect rect, double offset, double scale, std::string fileName, const std::function<std::vector<unsigned char>(std::string, INT2)>& bitmap_loader);

	//set value in given composite shape
	bool set_s_shape(DBL3 h, Rect rect, std::vector<MeshShape> shapes, SType value);

	//set s_scaling VEC by loading it form the named OVF2 file -  data types must match, but data is stretched to given mesh dimensions.
	bool set_s_ovf2(DBL3 h, Rect rect, std::string fileName);

	//random points in given range
	bool set_s_random(DBL3 h, Rect rect, DBL2 range, int seed);

	//random points in given range at given square spacing in xy plane : in between fill using bilinear interpolation
	bool set_s_jagged(DBL3 h, Rect rect, DBL2 range, double spacing, int seed);

	//polynomial slopes at slides with exponent n, starting from maximum value at surfaces, proceeding inwards towards minimum up to a depth of length * ratio
	//abl_x, abl_y, abl_z : depth ratio for negative side, depth ratio for positive side
	//values : minimum centre, maximum outer, polynomial exponent
	bool set_s_ablpol(DBL3 h, Rect rect, DBL2 abl_x, DBL2 abl_y, DBL2 abl_z, DBL3 values);

	//tanh slopes at slides with exponent n, starting from maximum value at surfaces, proceeding inwards towards minimum up to a depth of length * ratio
	//tanh profile centered
	//abl_x, abl_y, abl_z : depth ratio for negative side, depth ratio for positive side
	//values : minimum centre, maximum outer, tanh sigma in nm
	bool set_s_abltanh(DBL3 h, Rect rect, DBL2 abl_x, DBL2 abl_y, DBL2 abl_z, DBL3 values);

	//exponential slopes at slides with exponent n, starting from maximum value at surfaces, proceeding inwards towards minimum up to a depth of length * ratio
	//abl_x, abl_y, abl_z : depth ratio for negative side, depth ratio for positive side
	//values : minimum centre, maximum outer, exponential sigma in nm
	bool set_s_ablexp(DBL3 h, Rect rect, DBL2 abl_x, DBL2 abl_y, DBL2 abl_z, DBL3 values);

	//generate circular defects with a tanh radial profile with values in the given range, diameter range and average spacing (prng instantiated with given seed). The defect positioning is random.
	bool set_s_defects(DBL3 h, Rect rect, DBL2 range, DBL2 diameter_range, double spacing, int seed);

	//generate line faults in the given range length, orientation length(degrees azimuthal) and average spacing (prng instantiated with given seed).
	bool set_s_faults(DBL3 h, Rect rect, DBL2 range, DBL2 length_range, DBL2 orientation_range, double spacing, int seed);

	//generate voronoi 2d tessellation with values in the given range, and average spacing (prng instantiated with given seed). The values range is used to generate coefficient multipliers which are constant in each voronoi cell.
	bool set_s_voronoi2d(DBL3 h, Rect rect, DBL2 range, double spacing, int seed);
	//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
	bool set_s_uvoronoi2d(DBL3 h, Rect rect, DBL2 range, double spacing, double variation, int seed);

	//generate voronoi 3d tessellation with values in the given range, and average spacing (prng instantiated with given seed). The values range is used to generate coefficient multipliers which are constant in each voronoi cell.
	bool set_s_voronoi3d(DBL3 h, Rect rect, DBL2 range, double spacing, int seed);
	//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
	bool set_s_uvoronoi3d(DBL3 h, Rect rect, DBL2 range, double spacing, double variation, int seed);

	//generate voronoi 2d tessellation with average spacing. Set coefficient values randomly in the given range only at the Voronoi cell boundaries (prng instantiated with given seed).
	bool set_s_voronoiboundary2d(DBL3 h, Rect rect, DBL2 range, double spacing, int seed);
	//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
	bool set_s_uvoronoiboundary2d(DBL3 h, Rect rect, DBL2 range, double spacing, double variation, int seed);

	//generate voronoi 3d tessellation with average spacing. Set coefficient values randomly in the given range only at the Voronoi cell boundaries (prng instantiated with given seed).
	bool set_s_voronoiboundary3d(DBL3 h, Rect rect, DBL2 range, double spacing, int seed);
	//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
	bool set_s_uvoronoiboundary3d(DBL3 h, Rect rect, DBL2 range, double spacing, double variation, int seed);

	//generate voronoi 2d tessellation with average spacing. This method is applicable only to DBL3 PType, where a rotation operation is applied, fixed in each Voronoi cell. 
	//The rotation uses the values for polar (theta) and azimuthal (phi) angles specified in given ranges. prng instantiated with given seed.
	bool set_s_voronoirotation2d(DBL3 h, Rect rect, DBL2 theta, DBL2 phi, double spacing, int seed);
	//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
	bool set_s_uvoronoirotation2d(DBL3 h, Rect rect, DBL2 theta, DBL2 phi, double spacing, double variation, int seed);

	//generate voronoi 3d tessellation with average spacing. This method is applicable only to DBL3 PType, where a rotation operation is applied, fixed in each Voronoi cell. 
	//The rotation uses the values for polar (theta) and azimuthal (phi) angles specified in given ranges. prng instantiated with given seed.
	bool set_s_voronoirotation3d(DBL3 h, Rect rect, DBL2 theta, DBL2 phi, double spacing, int seed);
	//as above but Voronoi sites on regular grid with set variation (results in uniform grains, with uniformity controlled by variation parameter)
	bool set_s_uvoronoirotation3d(DBL3 h, Rect rect, DBL2 theta, DBL2 phi, double spacing, double variation, int seed);

public:

	//---------Constructors

	MatP(void) :
		SimulationSharedData(),
		Tscaling_eq({"T"}),
		Sscaling_eq({"x", "y", "z", "t"}),
		ProgramState<MatP<PType, SType>,
		std::tuple<
		PType, PType,
		std::vector<PType>, std::vector<PType>,
		std::vector<double>, std::vector<double>, std::vector<double>,
		VEC<SType>,
		std::string, int, std::string,
		TEquation<double>, TEquation<double, double, double, double>>,
		std::tuple<>>
		(this, { 
		VINFO(value_at_0K), VINFO(current_value),
		VINFO(values_at_0K), VINFO(current_values),
		VINFO(t_scaling), VINFO(t_scaling_y), VINFO(t_scaling_z), 
		VINFO(s_scaling), 
		VINFO(t_scaling_info), VINFO(s_scaling_type), VINFO(s_scaling_info), 
		VINFO(Tscaling_eq), VINFO(Sscaling_eq) }, {})
	{
		value_at_0K = PType();
		current_value = value_at_0K;
	}

	MatP(PType value) :
		SimulationSharedData(),
		Tscaling_eq({ "T" }),
		Sscaling_eq({ "x", "y", "z", "t" }),
		value_at_0K(value),
		ProgramState<MatP<PType, SType>, 
		std::tuple<
		PType, PType,
		std::vector<PType>, std::vector<PType>,
		std::vector<double>, std::vector<double>, std::vector<double>, 
		VEC<SType>, 
		std::string, int, std::string, 
		TEquation<double>, TEquation<double, double, double, double>>, 
		std::tuple<>>
		(this, {
		VINFO(value_at_0K), VINFO(current_value),
		VINFO(values_at_0K), VINFO(current_values),
		VINFO(t_scaling), VINFO(t_scaling_y), VINFO(t_scaling_z),
		VINFO(s_scaling),
		VINFO(t_scaling_info), VINFO(s_scaling_type), VINFO(s_scaling_info),
		VINFO(Tscaling_eq), VINFO(Sscaling_eq) }, {})
	{
		current_value = value_at_0K;	//default to 0K value
	}

	//An assignment from a MatP to another must always have matched template parameters. 
	//Conversion when parameters are mismatched should not be used, but we need to define the conversion operator to use the copy_parameters method since the compiler sees such a conversion in the compilation tree as theoretically possible (so throws error)
	template <typename PType_, typename SType_>
	operator MatP<PType_, SType_>&() { return *(new MatP<PType_, SType_>()); }

	//---------Assignment operator - copy a MatP to another of same type

	MatP<PType, SType>& operator=(const MatP<PType, SType>& copy_this);

	//---------Implement ProgramState method

	void RepairObjectState(void) 
	{
		//this is needed so older simulation files show temperature dependence sensibly, otherwise they'll end up showing "none" when there is a temperature dependence set.
		if (is_tdep()) {

			if (t_scaling_info == temperature_dependence_type(MATPTDEP_NONE)) t_scaling_info = temperature_dependence_type(MATPTDEP_ARRAY);
		}
		else t_scaling_info = temperature_dependence_type(MATPTDEP_NONE);
	}

	//---------Sub-lattices

	void set_number_of_sublattices(int num_sublattices)
	{
		if (num_sublattices > 1) {

			values_at_0K.assign(num_sublattices, value_at_0K);
			current_values.assign(num_sublattices, current_value);
		}
		else {

			values_at_0K.clear();
			current_values.clear();
		}

		//update cuda object also if set
#if COMPILECUDA == 1
		if (p_cu_obj_mpcuda) update_cuda_object();
#endif
	}

	int get_num_sublattices(void) { return values_at_0K.size(); }

	//---------Output value update

	//update output value for given temperature (use if base temperature changes), including updating of CUDA value if set
	void update(double Temperature = 0);

	//set current value for given temperature, but do not update CUDA value
	void set_current(double Temperature);

	//---------Output value get

	//get value at given temperature and no spatial scaling, but do not update output (meant for use with non-uniform temperature; for uniform temperature it's faster to just read the value through the conversion operator)
	//Use with temperature dependence : YES, spatial variation : NO
	PType get_scaled_value(const PType& value_to_scale, double Temperature = 0);
	//single sub-lattice
	PType get(double Temperature = 0) { return get_scaled_value(value_at_0K, Temperature); }
	//for multiple sub-lattices
	PType get(int sidx, double Temperature = 0) 
	{ 
		if (sidx < values_at_0K.size()) return get_scaled_value(values_at_0K[sidx], Temperature);
		else return get_scaled_value(value_at_0K, Temperature);
	}

	//get value (at base temperature) with spatial scaling (must be set so check before!).
	//Use with temperature dependence : NO, spatial variation : YES
	PType get_scaled_value(const PType& value_to_scale, const DBL3& position, double stime);
	//single sub-lattice
	PType get(const DBL3& position, double stime) { return get_scaled_value(current_value, position, stime); }
	//for multiple sub-lattices
	PType get(int sidx, const DBL3& position, double stime) 
	{ 
		if (sidx < current_values.size()) return get_scaled_value(current_values[sidx], position, stime);
		else return get_scaled_value(current_value, position, stime);
	}

	//get value with spatial scaling (must be set so check before!) and temperature dependence
	//Use with temperature dependence : YES, spatial variation : YES
	PType get_scaled_value(const PType& value_to_scale, const DBL3& position, double stime, double Temperature);
	//single sub-lattice
	PType get(const DBL3& position, double stime, double Temperature) { return get_scaled_value(value_at_0K, position, stime, Temperature); }
	//for multiple sub-lattices
	PType get(int sidx, const DBL3& position, double stime, double Temperature) 
	{ 
		if (sidx < values_at_0K.size()) return get_scaled_value(values_at_0K[sidx], position, stime, Temperature);
		else return get_scaled_value(value_at_0K, position, stime, Temperature);
	}

	//get 0K value
	PType get0(void) const { return value_at_0K; }
	//for multiple sub-lattices
	PType get0(int sidx) const { return values_at_0K[sidx]; }
	//get total value by summing over sub-lattices
	PType get0_total(void) const
	{
		PType value = PType();
		for (int sidx = 0; sidx < values_at_0K.size(); sidx++) value += values_at_0K[sidx];
		return value;
	}

	//get current output value (at base temperature with no spatial scaling)
	PType get_current(void) const { return current_value; }
	//for multiple sub-lattices
	PType get_current(int sidx) const { return current_values[sidx]; }
	//get total value by summing over sub-lattices
	PType get_current_total(void) const
	{
		PType value = PType();
		for (int sidx = 0; sidx < current_values.size(); sidx++) value += current_values[sidx];
		return value;
	}

	//get spatial scaling value : 1.0 if not set.
	SType get_s_scaling_value(const DBL3& position, double stime);

	//---------Set scaling equation (temperature)

	//set scaling text equation
	void set_t_scaling_equation(const std::string& equationText, const vector_key<double>& userConstants, double T_Curie, double base_temperature);

	//update set equations with user constants, mesh dimensions (where applicable), material Curie temperature and base temperature (where applicable)
	bool update_equations(const vector_key<double>& userConstants, DBL3 meshDimensions, double T_Curie, double base_temperature);

	//set special functions in text equation, also increasing ref count locally if passed in : the Funcs_Special objects are computed externally.
	void set_t_scaling_special_functions(
		std::shared_ptr<Funcs_Special> pCurieWeiss_ = nullptr,
		std::shared_ptr<Funcs_Special> pLongRelSus_ = nullptr,
		std::shared_ptr<Funcs_Special> pCurieWeiss1_ = nullptr, std::shared_ptr<Funcs_Special> pCurieWeiss2_ = nullptr,
		std::shared_ptr<Funcs_Special> pLongRelSus1_ = nullptr, std::shared_ptr<Funcs_Special> pLongRelSus2_ = nullptr,
		std::shared_ptr<Funcs_Special> pAlpha1_ = nullptr, std::shared_ptr<Funcs_Special> pAlpha2_ = nullptr);

#if COMPILECUDA == 1
	//set special functions in text equation, also increasing ref count locally if passed in : the Funcs_Special objects are computed externally.
	void set_t_scaling_special_functions_CUDA(
		mcu_SpecialFunc* pCurieWeiss_CUDA_,
		mcu_SpecialFunc* pLongRelSus_CUDA_,
		mcu_SpecialFunc* pCurieWeiss1_CUDA_, mcu_SpecialFunc* pCurieWeiss2_CUDA_,
		mcu_SpecialFunc* pLongRelSus1_CUDA_, mcu_SpecialFunc* pLongRelSus2_CUDA_,
		mcu_SpecialFunc* pAlpha1_CUDA_, mcu_SpecialFunc* pAlpha2_CUDA_);
#endif

	//---------Set scaling array (temperature)

	//clear any temperature dependence
	void clear_t_scaling(void);

	//calculate t_scaling from given temperature dependence of scaling coefficients
	bool set_t_scaling_array(std::vector<double> temp_arr, std::vector<double> scaling_arr, std::vector<double> scaling_arr_y, std::vector<double> scaling_arr_z);

	//set scaling array directly, where the value indexes must correspond to temperature values (e.g. t_scaling[0] is for temperature 0 K, etc.).
	void set_precalculated_t_scaling_array(std::vector<double>& scaling_arr, std::vector<double>& scaling_arr_y, std::vector<double>& scaling_arr_z);

	//---------Set scaling info (temperature)

	void set_t_scaling_info(std::string info_text);

	//---------Set spatial dependence

	//clear spatial scaling
	void clear_s_scaling(void);

	//update spatial scaling VEC for cellsize and rectangle, depending on currently set spatial scaling
	bool update_s_scaling(DBL3 h, Rect rect);

	//set parameter spatial variation using a given generator and arguments (arguments passed as a std::string to be interpreted and converted using ToNum)
	BError set_s_scaling(DBL3 h, Rect rect, MATPVAR_ generatorID, std::string generatorArgs, const std::function<std::vector<unsigned char>(std::string, INT2)>& bitmap_loader);

	//set value in given composite shape : this calls set_s_shape in turn
	BError set_s_scaling_shape(DBL3 h, Rect rect, std::vector<MeshShape> shapes, SType value);

	//set spatial scaling text equation
	void set_s_scaling_equation(const std::string& equationText, const vector_key<double>& userConstants, DBL3 meshDimensions);

	//if s_scaling is set, then check there are no zero values for non-empty cells in M. Only enabled for SType fundamental types (double), since GetMagnitude(M[idx]) is a fundamental type.
	//if any zeroes found, then set s_scaling value to match the magnitude of the value held in M at that cell, where the 0K value is passed in as Ms.
	template <typename VType, typename SType_ = SType, std::enable_if_t<std::is_fundamental<SType_>::value>* = nullptr>
	void fix_s_scaling_zeros(VEC_VC<VType>& M, double Ms)
	{
		//s_scaling must have same size as M
		if (s_scaling.size() != M.size()) return;

#pragma omp parallel for
		for (int idx = 0; idx < M.linear_size(); idx++) {

			if (M.is_not_empty(idx) && s_scaling.is_empty(idx)) {

				s_scaling[idx] = GetMagnitude(M[idx]) / Ms;
			}
		}

#if COMPILECUDA == 1
		if (p_cu_obj_mpcuda) update_cuda_object();
#endif
	}

	//---------Get temperature dependence

	//get temperature scaling coefficients as an array from 0K up to and including max_temperature
	bool get_temperature_scaling(double max_temperature, std::vector<double>& get_scaling_x, std::vector<double>& get_scaling_y, std::vector<double>& get_scaling_z);

	//---------Get spatial variation

	void calculate_s_scaling(VEC<double>& displayVEC_SCA, double stime);
	void calculate_s_scaling(VEC<DBL3>& displayVEC_VEC, double stime);

	//---------Read value

	//use the material parameter just like a PType type to read from it
	operator PType() const { return current_value; }
	//for multiple sub-lattices
	PType operator()(int sidx) const { return current_values[sidx]; }
	//get all sub-lattices values
	const std::vector<PType>& get_sublattices(void) const { return current_values; }

	//---------Set value

	//set value at 0K, and make current value equal to it
	MatP<PType, SType>& operator=(PType set_value);

	//set value at 0K, and make current value equal to it - multiple sub-lattice version
	MatP<PType, SType>& operator=(std::vector<PType> set_values);

	//---------Get Info (intended for console display)

	//returns a std::string describing the set temperature dependence ("none", "array/filename" or set equation : "equation text") 
	std::string get_info_string(void) const { return t_scaling_info; }

	//returns a std::string describing the set spatial dependence ("none", "array" or set equation : "name parameters...") 
	std::string get_varinfo_string(void) const { return s_scaling_info; }

	//does it have a temperature dependence?
	bool is_tdep(void) const { return (Tscaling_eq.is_set() || t_scaling.size()); }

	//does it have a temperature dependence specified using a text equation? (equation may still not be set due to missing constants)
	bool is_t_equation_set(void) const { return Tscaling_eq.show_equation().length(); }

	//does it have a spatial dependence?
	bool is_sdep(void) const { return (Sscaling_eq.is_set() || s_scaling.linear_size()); }

	//does it have a spatial dependence specified using a text equation?
	bool is_s_equation_set(void) const { return Sscaling_eq.is_set(); }

	//return number of components of value, e.g. double : 1, DBL2 : 2, DBL3 : 3
	int num_components(void) const;

	//---------Comparison operators

	//Not currently needed

	//---------Arithmetic operators

	//the return type might not simply be PType! (e.g. PType is a DBL3, so '*' for DBL3 is a scalar product which results in a double). 
	//Need to tell the compiler exactly how to determine the return type. Similar problems may occur for the other arithmetic operators

	//multiply with a MatP - '^' is used to signify vector product for VAL3
	template <typename PType_, typename SType_>
	auto operator^(const MatP<PType_, SType_>& rhs) const -> decltype(std::declval<PType_>() ^ std::declval<PType_>())
	{
		return current_value ^ rhs.current_value;
	}

	//multiply with a MatP - '&' is used to signify component-by-component product for VAL3
	template <typename PType_, typename SType_>
	auto operator&(const MatP<PType_, SType_>& rhs) const -> decltype(std::declval<PType_>() & std::declval<PType_>())
	{
		return current_value & rhs.current_value;
	}

	//multiply with a MatP
	auto operator*(const MatP& rhs) const -> decltype(std::declval<PType>() * std::declval<PType>())
	{
		return current_value * rhs.current_value;
	}

	//multiply with a value on the RHS
	template <typename PType_, std::enable_if_t<!std::is_same<PType_, MatP<PType, SType>>::value>* = nullptr>
	auto operator*(const PType_& rhs) const -> decltype(std::declval<PType>() * std::declval<PType_>())
	{
		return current_value * rhs;
	}

	//multiply with a value on the LHS
	template <typename _PType, std::enable_if_t<!std::is_same<_PType, MatP<PType, SType>>::value>* = nullptr>
	friend auto operator*(const _PType& lhs, const MatP<PType, SType>& rhs) -> decltype(std::declval<_PType>() * std::declval<PType>())
	{
		return lhs * rhs.current_value;
	}

	//divide by a MatP
	auto operator/(const MatP& rhs) const -> decltype(std::declval<PType>() / std::declval<PType>())
	{
		return current_value / rhs.current_value;
	}

	//division with a value on the RHS
	template <typename PType_, std::enable_if_t<!std::is_same<PType_, MatP<PType, SType>>::value>* = nullptr>
	auto operator/(const PType_& rhs) const -> decltype(std::declval<PType>() / std::declval<PType_>())
	{
		return current_value / rhs;
	}

	//division with a value on the LHS
	template <typename _PType, std::enable_if_t<!std::is_same<_PType, MatP<PType, SType>>::value>* = nullptr>
	friend auto operator/(const _PType& lhs, const MatP<PType, SType>& rhs) -> decltype(std::declval<_PType>() / std::declval<PType>())
	{
		return lhs / rhs.current_value;
	}

	//addition with a MatP
	auto operator+(const MatP& rhs) const -> decltype(std::declval<PType>() + std::declval<PType>())
	{
		return current_value + rhs.current_value;
	}

	//addition with a value on the RHS
	template <typename PType_, std::enable_if_t<!std::is_same<PType_, MatP<PType, SType>>::value>* = nullptr>
	auto operator+(const PType_& rhs) const -> decltype(std::declval<PType>() + std::declval<PType_>())
	{
		return current_value + rhs;
	}

	//addition with a value on the LHS
	template <typename _PType, std::enable_if_t<!std::is_same<_PType, MatP<PType, SType>>::value>* = nullptr>
	friend auto operator+(const _PType& lhs, const MatP<PType, SType>& rhs) -> decltype(std::declval<_PType>() + std::declval<PType>())
	{
		return lhs + rhs.current_value;
	}

	//difference with a MatP
	auto operator-(const MatP& rhs) const -> decltype(std::declval<PType>() - std::declval<PType>())
	{
		return current_value - rhs.current_value;
	}

	//difference with a value on the RHS
	template <typename PType_, std::enable_if_t<!std::is_same<PType_, MatP<PType, SType>>::value>* = nullptr>
	auto operator-(const PType_& rhs) const -> decltype(std::declval<PType>() - std::declval<PType_>())
	{
		return current_value - rhs;
	}

	//difference with a value on the LHS
	template <typename _PType, std::enable_if_t<!std::is_same<_PType, MatP<PType, SType>>::value>* = nullptr>
	friend auto operator-(const _PType& lhs, const MatP<PType, SType>& rhs) -> decltype(std::declval<_PType>() - std::declval<PType>())
	{
		return lhs - rhs.current_value;
	}

	//---------Special access methods

	std::vector<double>& t_scaling_ref(void) { return t_scaling; }
	std::vector<double>& t_scaling_y_ref(void) { return t_scaling_y; }
	std::vector<double>& t_scaling_z_ref(void) { return t_scaling_z; }

	VEC<SType>& s_scaling_ref(void) { return s_scaling; }

#if COMPILECUDA == 1
	template <typename mcu_MatPCUDA_RType>
	void set_p_cu_obj_mpcuda(mcu_MatPCUDA_RType* ptr)
	{
		pTscaling_CUDAeq = new mTEquationCUDA<cuBReal>(mGPU);
		pSscaling_CUDAeq = new mTEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>(mGPU);

		p_cu_obj_mpcuda = ptr;
	}
	
	void null_p_cu_obj_mpcuda(void) 
	{ 
		if (pTscaling_CUDAeq) delete pTscaling_CUDAeq;
		pTscaling_CUDAeq = nullptr;
		if (pSscaling_CUDAeq) delete pSscaling_CUDAeq;
		pSscaling_CUDAeq = nullptr;

		p_cu_obj_mpcuda = nullptr; 
	}
	
	//only invoked when CUDA switched on, so objects constructed (pTscaling_CUDAeq and pSscaling_CUDAeq constructed in set_p_cu_obj_mpcuda, called before MatPCUDA is set from MatP - set_from_cpu)
	TEquationCUDA<cuBReal>& Tscaling_CUDAeq_ref(int device_idx) { return pTscaling_CUDAeq->get_managed_object(device_idx); }
	TEquationCUDA<cuBReal, cuBReal, cuBReal, cuBReal>& Sscaling_CUDAeq_ref(int device_idx) { return pSscaling_CUDAeq->get_managed_object(device_idx); }
#endif
};

//---------Assignment operator - copy a MatP to another of same type

template <typename PType, typename SType>
MatP<PType, SType>& MatP<PType, SType>::operator=(const MatP<PType, SType>& copy_this)
{
	//copy value at 0K:
	value_at_0K = copy_this.value_at_0K;
	values_at_0K = copy_this.values_at_0K;

	//copy current value:
	current_value = copy_this.current_value;
	current_values = copy_this.current_values;

	//copy temperature scaling array:
	clear_vector(t_scaling);
	t_scaling = copy_this.t_scaling;
	clear_vector(t_scaling_y);
	t_scaling_y = copy_this.t_scaling_y;
	clear_vector(t_scaling_z);
	t_scaling_z = copy_this.t_scaling_z;

	//now copy scaling equation
	Tscaling_eq = copy_this.Tscaling_eq;
	Sscaling_eq = copy_this.Sscaling_eq;

	t_scaling_info = copy_this.t_scaling_info;

	//copy any spatial scaling
	
	//s_scaling may be empty but copy_this.s_scaling not; in this case the following command doesn't have an effect, but this copy operator should be followed by an update function which will generate the required s_scaling (update_s_scaling called)
	//Note, this copy is still needed here in case s_scaling and copy_this.s_scaling match in dimensions but have different values - update_s_scaling will skip if it sees matching dimensions.
	s_scaling.copy_values(copy_this.s_scaling);

	//s_scaling info so the update will generate the required s_scaling
	s_scaling_type = copy_this.s_scaling_type;
	s_scaling_info = copy_this.s_scaling_info;

	//pointers to special functions, calculated elsewhere
	pCurieWeiss = copy_this.pCurieWeiss;
	pLongRelSus = copy_this.pLongRelSus;
	pCurieWeiss1 = copy_this.pCurieWeiss1;
	pCurieWeiss2 = copy_this.pCurieWeiss2;
	pLongRelSus1 = copy_this.pLongRelSus1;
	pLongRelSus2 = copy_this.pLongRelSus2;
	pAlpha1 = copy_this.pAlpha1;
	pAlpha2 = copy_this.pAlpha2;

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return *this;
}

//---------Get Info

//return number of components of value, e.g. double : 1, DBL2 : 2, DBL3 : 3
template <>
inline int MatP<double, double>::num_components(void) const { return 1; }

template <>
inline int MatP<DBL2, double>::num_components(void) const { return 2; }

template <>
inline int MatP<DBL3, double>::num_components(void) const { return 3; }
template <>
inline int MatP<DBL3, DBL3>::num_components(void) const { return 3; }

//---------- TEMPERATURE ONLY

//get value at given temperature and no spatial scaling, but do not update output (meant for use with non-uniform temperature; for uniform temperature it's faster to just read the value through the conversion operator)
//Use with temperature dependence : YES, spatial variation : NO
template <>
inline double MatP<double, double>::get_scaled_value(const double& value_to_scale, double Temperature)
{
	if (Tscaling_eq.is_set()) {

		//use pre-set equation
		return value_to_scale * Tscaling_eq.evaluate(Temperature);
	}
	else if (t_scaling.size()) {

		//use custom temperature scaling
		int index = (int)floor_epsilon(Temperature);
		if (index + 1 < (int)t_scaling.size() && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1
			return (value_to_scale * (t_scaling[index] * (double(index + 1) - Temperature) + t_scaling[index + 1] * (Temperature - double(index))));
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else return (value_to_scale * t_scaling.back());
	}
	//no temperature dependence set
	else return value_to_scale;
}

template <>
inline DBL2 MatP<DBL2, double>::get_scaled_value(const DBL2& value_to_scale, double Temperature)
{
	if (Tscaling_eq.is_set()) {

		//use pre-set equation

		//component-by-component product if dual equation
		if (Tscaling_eq.is_set_dual()) return (value_to_scale & Tscaling_eq.evaluate_dual(Temperature));
		
		//just constant multiplication for a scalar equation
		else return value_to_scale * Tscaling_eq.evaluate(Temperature);
	}
	else if (t_scaling.size()) {

		//use custom temperature scaling
		int index = (int)floor_epsilon(Temperature);
		
		if (index + 1 < (int)t_scaling.size() && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1

			//component-by-component product if dual equation
			if (t_scaling_y.size()) return (value_to_scale & (DBL2(t_scaling[index], t_scaling_y[index]) * (double(index + 1) - Temperature) + DBL2(t_scaling[index + 1], t_scaling_y[index + 1]) * (Temperature - double(index))));

			//just constant multiplication for a scalar equation
			else return (value_to_scale * (t_scaling[index] * (double(index + 1) - Temperature) + t_scaling[index + 1] * (Temperature - double(index))));
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (t_scaling_y.size()) return (value_to_scale & DBL2(t_scaling.back(), t_scaling_y.back()));
			else return (value_to_scale * t_scaling.back());
		}
	}
	//no temperature dependence set
	else return value_to_scale;
}

template <>
inline DBL3 MatP<DBL3, double>::get_scaled_value(const DBL3& value_to_scale, double Temperature)
{
	if (Tscaling_eq.is_set()) {

		//component-by-component product if vector equation
		if (Tscaling_eq.is_set_vector()) return (value_to_scale & Tscaling_eq.evaluate_vector(Temperature));
		
		//just constant multiplication for a scalar equation
		else return value_to_scale * Tscaling_eq.evaluate(Temperature);
	}
	else if (t_scaling.size()) {

		//use custom temperature scaling
		int index = (int)floor_epsilon(Temperature);
		if (index + 1 < (int)t_scaling.size() && index >= 0) {

			//component-by-component product if vector equation
			if (t_scaling_z.size()) return (value_to_scale & (DBL3(t_scaling[index], t_scaling_y[index], t_scaling_z[index]) * (double(index + 1) - Temperature) + DBL3(t_scaling[index + 1], t_scaling_y[index + 1], t_scaling_z[index + 1]) * (Temperature - double(index))));

			//just constant multiplication for a scalar equation
			else return (value_to_scale * (t_scaling[index] * (double(index + 1) - Temperature) + t_scaling[index + 1] * (Temperature - double(index))));
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (t_scaling_z.size()) return (value_to_scale & DBL3(t_scaling.back(), t_scaling_y.back(), t_scaling_z.back()));
			else return (value_to_scale * t_scaling.back());
		}
	}
	//no temperature dependence set
	else return value_to_scale;
}

template <>
inline DBL3 MatP<DBL3, DBL3>::get_scaled_value(const DBL3& value_to_scale, double Temperature)
{
	if (Tscaling_eq.is_set()) {

		//use pre-set equation

		//component-by-component product if vector equation
		if (Tscaling_eq.is_set_vector()) return (value_to_scale & Tscaling_eq.evaluate_vector(Temperature));

		//just constant multiplication for a scalar equation
		else return value_to_scale * Tscaling_eq.evaluate(Temperature);
	}
	else if (t_scaling.size()) {

		//use custom temperature scaling
		int index = (int)floor_epsilon(Temperature);
		if (index + 1 < (int)t_scaling.size() && index >= 0) {

			//component-by-component product if vector equation
			if (t_scaling_z.size()) return (value_to_scale & (DBL3(t_scaling[index], t_scaling_y[index], t_scaling_z[index]) * (double(index + 1) - Temperature) + DBL3(t_scaling[index + 1], t_scaling_y[index + 1], t_scaling_z[index + 1]) * (Temperature - double(index))));

			//just constant multiplication for a scalar equation
			else return (value_to_scale * (t_scaling[index] * (double(index + 1) - Temperature) + t_scaling[index + 1] * (Temperature - double(index))));
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (t_scaling_z.size()) return (value_to_scale & DBL3(t_scaling.back(), t_scaling_y.back(), t_scaling_z.back()));
			else return (value_to_scale * t_scaling.back());
		}
	}
	//no temperature dependence set
	else return value_to_scale;
}

//---------- SPATIAL ONLY

//get value (at base temperature) with spatial scaling (must be set so check before!)
//Use with temperature dependence : NO, spatial variation : YES
template <>
inline double MatP<double, double>::get_scaled_value(const double& value_to_scale, const DBL3& position, double stime)
{
	if (s_scaling_type == MATPVAR_EQUATION) return value_to_scale * Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
	else return value_to_scale * s_scaling[position];
}

template <>
inline DBL2 MatP<DBL2, double>::get_scaled_value(const DBL2& value_to_scale, const DBL3& position, double stime)
{
	if (s_scaling_type == MATPVAR_EQUATION) return value_to_scale * Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
	else return value_to_scale * s_scaling[position];
}

template <>
inline DBL3 MatP<DBL3, double>::get_scaled_value(const DBL3& value_to_scale, const DBL3& position, double stime)
{
	if (s_scaling_type == MATPVAR_EQUATION) return value_to_scale * Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
	else return value_to_scale * s_scaling[position];
}

template <>
inline DBL3 MatP<DBL3, DBL3>::get_scaled_value(const DBL3& value_to_scale, const DBL3& position, double stime)
{
	//rotation not scalar multiplication
	if (s_scaling_type == MATPVAR_EQUATION) return rotate_polar(value_to_scale, Sscaling_eq.evaluate_vector(position.x, position.y, position.z, stime));
	else return rotate_polar(value_to_scale, s_scaling[position]);
}

//-----------------

//get spatial scaling value : 1.0 if not set.
template <>
inline double MatP<double, double>::get_s_scaling_value(const DBL3& position, double stime)
{
	if (s_scaling_type == MATPVAR_EQUATION) return Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
	else if (s_scaling.linear_size()) return s_scaling[position];
	else return 1.0;
}

template <>
inline double MatP<DBL2, double>::get_s_scaling_value(const DBL3& position, double stime)
{
	if (s_scaling_type == MATPVAR_EQUATION) return Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
	else if (s_scaling.linear_size()) return s_scaling[position];
	else return 1.0;
}

template <>
inline double MatP<DBL3, double>::get_s_scaling_value(const DBL3& position, double stime)
{
	if (s_scaling_type == MATPVAR_EQUATION) return Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
	else if (s_scaling.linear_size()) return s_scaling[position];
	else return 1.0;
}

template <>
inline DBL3 MatP<DBL3, DBL3>::get_s_scaling_value(const DBL3& position, double stime)
{
	if (s_scaling_type == MATPVAR_EQUATION) return Sscaling_eq.evaluate_vector(position.x, position.y, position.z, stime);
	else if (s_scaling.linear_size()) return s_scaling[position];
	else return DBL3(1.0, 0.0, 0.0);
}

//---------- TEMPERATURE and SPATIAL

//get value with spatial scaling (must be set so check before!) and temperature dependence
//Use with temperature dependence : YES, spatial variation : YES
template <>
inline double MatP<double, double>::get_scaled_value(const double& value_to_scale, const DBL3& position, double stime, double Temperature)
{
	if (Tscaling_eq.is_set()) {

		//use pre-set equation
		if (s_scaling_type == MATPVAR_EQUATION) {

			return (value_to_scale * Tscaling_eq.evaluate(Temperature)) * Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
		}
		else {

			return (value_to_scale * Tscaling_eq.evaluate(Temperature)) * s_scaling[position];
		}
	}
	else if (t_scaling.size()) {

		//use custom temperature scaling
		int index = (int)floor_epsilon(Temperature);
		if (index + 1 < (int)t_scaling.size() && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1
			if (s_scaling_type == MATPVAR_EQUATION) return (value_to_scale * (t_scaling[index] * (double(index + 1) - Temperature) + t_scaling[index + 1] * (Temperature - double(index)))) * Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
			else return (value_to_scale * (t_scaling[index] * (double(index + 1) - Temperature) + t_scaling[index + 1] * (Temperature - double(index)))) * s_scaling[position];
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (s_scaling_type == MATPVAR_EQUATION) return (value_to_scale * t_scaling.back()) * Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
			else return (value_to_scale * t_scaling.back()) * s_scaling[position];
		}
	}
	//no temperature dependence set
	else {
		
		if (s_scaling_type == MATPVAR_EQUATION) return value_to_scale * Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
		else return value_to_scale * s_scaling[position];
	}
}

template <>
inline DBL2 MatP<DBL2, double>::get_scaled_value(const DBL2& value_to_scale, const DBL3& position, double stime, double Temperature)
{
	if (Tscaling_eq.is_set()) {

		//use pre-set equation

		double s = (s_scaling_type == MATPVAR_EQUATION ? Sscaling_eq.evaluate(position.x, position.y, position.z, stime) : s_scaling[position]);

		//component-by-component product if dual equation
		if (Tscaling_eq.is_set_dual()) return (value_to_scale & Tscaling_eq.evaluate_dual(Temperature)) * s;

		//just constant multiplication for a scalar equation
		else return value_to_scale * Tscaling_eq.evaluate(Temperature) * s;
	}
	else if (t_scaling.size()) {

		double s = (s_scaling_type == MATPVAR_EQUATION ? Sscaling_eq.evaluate(position.x, position.y, position.z, stime) : s_scaling[position]);

		//use custom temperature scaling
		int index = (int)floor_epsilon(Temperature);
		if (index + 1 < (int)t_scaling.size() && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1

			//component-by-component product if dual equation
			if (t_scaling_y.size()) return (value_to_scale & (DBL2(t_scaling[index], t_scaling_y[index]) * (double(index + 1) - Temperature) + DBL2(t_scaling[index + 1], t_scaling_y[index + 1]) * (Temperature - double(index)))) * s;

			//just constant multiplication for a scalar equation
			else return (value_to_scale * (t_scaling[index] * (double(index + 1) - Temperature) + t_scaling[index + 1] * (Temperature - double(index)))) * s;
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (t_scaling_y.size()) return (value_to_scale & DBL2(t_scaling.back(), t_scaling_y.back())) * s;
			else return (value_to_scale * t_scaling.back()) * s;
		}
	}
	//no temperature dependence set
	else {

		if (s_scaling_type == MATPVAR_EQUATION) return value_to_scale * Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
		else return value_to_scale * s_scaling[position];
	}
}

template <>
inline DBL3 MatP<DBL3, double>::get_scaled_value(const DBL3& value_to_scale, const DBL3& position, double stime, double Temperature)
{
	if (Tscaling_eq.is_set()) {

		double s = (s_scaling_type == MATPVAR_EQUATION ? Sscaling_eq.evaluate(position.x, position.y, position.z, stime) : s_scaling[position]);

		//component-by-component product if dual equation
		if (Tscaling_eq.is_set_vector()) return (value_to_scale & Tscaling_eq.evaluate_vector(Temperature)) * s;

		//just constant multiplication for a scalar equation
		else return value_to_scale * Tscaling_eq.evaluate(Temperature) * s;
	}
	else if (t_scaling.size()) {

		double s = (s_scaling_type == MATPVAR_EQUATION ? Sscaling_eq.evaluate(position.x, position.y, position.z, stime) : s_scaling[position]);

		//use custom temperature scaling
		int index = (int)floor_epsilon(Temperature);
		if (index + 1 < (int)t_scaling.size() && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1
			
			//component-by-component product if vector equation
			if (t_scaling_z.size()) return (value_to_scale & (DBL3(t_scaling[index], t_scaling_y[index], t_scaling_z[index]) * (double(index + 1) - Temperature) + DBL3(t_scaling[index + 1], t_scaling_y[index + 1], t_scaling_z[index + 1]) * (Temperature - double(index)))) * s;

			//just constant multiplication for a scalar equation
			else return (value_to_scale * (t_scaling[index] * (double(index + 1) - Temperature) + t_scaling[index + 1] * (Temperature - double(index)))) * s;
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (t_scaling_z.size()) return (value_to_scale & DBL3(t_scaling.back(), t_scaling_y.back(), t_scaling_z.back())) * s;
			else return (value_to_scale * t_scaling.back()) * s;
		}
	}
	//no temperature dependence set
	else {

		if (s_scaling_type == MATPVAR_EQUATION) return value_to_scale * Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
		else return value_to_scale * s_scaling[position];
	}
}

template <>
inline DBL3 MatP<DBL3, DBL3>::get_scaled_value(const DBL3& value_to_scale, const DBL3& position, double stime, double Temperature)
{
	if (Tscaling_eq.is_set()) {

		//use pre-set equation

		DBL3 s = (s_scaling_type == MATPVAR_EQUATION ? Sscaling_eq.evaluate_vector(position.x, position.y, position.z, stime) : s_scaling[position]);

		//component-by-component product if dual equation
		if (Tscaling_eq.is_set_vector()) return rotate_polar(value_to_scale & Tscaling_eq.evaluate_vector(Temperature), s);

		//just constant multiplication for a scalar equation
		else return rotate_polar(value_to_scale * Tscaling_eq.evaluate(Temperature), s);
	}
	else if (t_scaling.size()) {

		DBL3 s = (s_scaling_type == MATPVAR_EQUATION ? Sscaling_eq.evaluate_vector(position.x, position.y, position.z, stime) : s_scaling[position]);

		//use custom temperature scaling
		int index = (int)floor_epsilon(Temperature);
		if (index + 1 < (int)t_scaling.size() && index >= 0) {

			//use linear interpolation for temperature in range index to index + 1

			//component-by-component product if vector equation
			if (t_scaling_z.size()) return rotate_polar(value_to_scale & (DBL3(t_scaling[index], t_scaling_y[index], t_scaling_z[index]) * (double(index + 1) - Temperature) + DBL3(t_scaling[index + 1], t_scaling_y[index + 1], t_scaling_z[index + 1]) * (Temperature - double(index))), s);

			//just constant multiplication for a scalar equation
			else return rotate_polar(value_to_scale * (t_scaling[index] * (double(index + 1) - Temperature) + t_scaling[index + 1] * (Temperature - double(index))), s);
		}
		//for temperatures higher than the loaded array just use the last scaling value
		else {

			if (t_scaling_z.size()) return rotate_polar(value_to_scale & DBL3(t_scaling.back(), t_scaling_y.back(), t_scaling_z.back()), s);
			else return rotate_polar(value_to_scale * t_scaling.back(), s);
		}
	}
	//no temperature dependence set
	else {

		if (s_scaling_type == MATPVAR_EQUATION) return rotate_polar(value_to_scale, Sscaling_eq.evaluate_vector(position.x, position.y, position.z, stime));
		else return rotate_polar(value_to_scale, s_scaling[position]);
	}
}

//---------Output value update

template <typename PType, typename SType>
void MatP<PType, SType>::update(double Temperature)
{
	current_value = get(Temperature);
	
	if (current_values.size()) {

		for (int sidx = 0; sidx < current_values.size(); sidx++)
			current_values[sidx] = get(sidx, Temperature);
	}

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_value();
#endif
}

//set current value for given temperature, but do not update CUDA value
template <typename PType, typename SType>
void MatP<PType, SType>::set_current(double Temperature)
{
	current_value = get(Temperature);

	if (current_values.size()) {

		for (int sidx = 0; sidx < current_values.size(); sidx++)
			current_values[sidx] = get(sidx, Temperature);
	}
}

//---------Set value

template <typename PType, typename SType>
MatP<PType, SType>& MatP<PType, SType>::operator=(PType set_value)
{
	value_at_0K = set_value;
	current_value = value_at_0K;	//default to 0K value when using this

	//multiple sub-lattices
	if (values_at_0K.size()) {

		for (int sidx = 0; sidx < values_at_0K.size(); sidx++) {

			values_at_0K[sidx] = set_value;
			current_values[sidx] = set_value;
		}
	}

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_value();
#endif

	return *this;
}

//set value at 0K, and make current value equal to it - multiple sub-lattice version
template <typename PType, typename SType>
MatP<PType, SType>& MatP<PType, SType>::operator=(std::vector<PType> set_values)
{
	value_at_0K = set_values[0];
	current_value = value_at_0K;	//default to 0K value when using this

	//multiple sub-lattices
	if (values_at_0K.size()) {

		for (int sidx = 0; sidx < values_at_0K.size(); sidx++) {

			if (sidx < set_values.size()) {

				values_at_0K[sidx] = set_values[sidx];
				current_values[sidx] = set_values[sidx];
			}
			else {

				values_at_0K[sidx] = value_at_0K;
				current_values[sidx] = current_value;
			}
		}
	}

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_value();
#endif

	return *this;
}

//---------Set scaling equation (temperature)

//set scaling text equation
template <typename PType, typename SType>
void MatP<PType, SType>::set_t_scaling_equation(const std::string& equationText, const vector_key<double>& userConstants, double T_Curie, double base_temperature)
{
	std::vector<std::pair<std::string, double>> constants(userConstants.size());
	for (int idx = 0; idx < constants.size(); idx++) {

		constants[idx] = { userConstants.get_key_from_index(idx), userConstants[idx] };
	}

	Tscaling_eq.set_constants(constants, false);
	Tscaling_eq.make_from_string(equationText, { {"Tc", T_Curie}, {"Tb", base_temperature} });

	//set special function in Text equation (pre-calculated)
	set_t_scaling_special_functions();
	   	  
	//clear scaling array - you either use scaling array method, or equation - or none - not both
	t_scaling.clear();
	t_scaling_y.clear();
	t_scaling_z.clear();

	//set temperature scaling info
	t_scaling_info = Tscaling_eq.show_equation();

	update(base_temperature);

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif
}

//update set equtions with user constants, mesh dimensions (where applicable), material Curie temperature and base temperature (where applicable)
template <typename PType, typename SType>
bool MatP<PType, SType>::update_equations(const vector_key<double>& userConstants, DBL3 meshDimensions, double T_Curie, double base_temperature)
{
	bool success = true;

	std::vector<std::pair<std::string, double>> constants(userConstants.size());
	for (int idx = 0; idx < constants.size(); idx++) {

		constants[idx] = { userConstants.get_key_from_index(idx), userConstants[idx] };
	}

	Tscaling_eq.set_constants(constants, false);
	Tscaling_eq.set_constant("Tc", T_Curie, false);
	Tscaling_eq.set_constant("Tb", base_temperature, false);
	success = Tscaling_eq.remake_equation();

	//set special functions in Text equation (pre-calculated)
	if (success) set_t_scaling_special_functions();

	Sscaling_eq.set_constants(constants, false);
	Sscaling_eq.set_constants({ {"Lx", meshDimensions.x}, {"Ly", meshDimensions.y}, {"Lz", meshDimensions.z} }, false);
	success &= Sscaling_eq.remake_equation();

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//set special functions in text equation, also increasing ref count locally if passed in : the Funcs_Special objects are computed externally.
template <typename PType, typename SType>
void MatP<PType, SType>::set_t_scaling_special_functions(
	std::shared_ptr<Funcs_Special> pCurieWeiss_,
	std::shared_ptr<Funcs_Special> pLongRelSus_,
	std::shared_ptr<Funcs_Special> pCurieWeiss1_, std::shared_ptr<Funcs_Special> pCurieWeiss2_,
	std::shared_ptr<Funcs_Special> pLongRelSus1_, std::shared_ptr<Funcs_Special> pLongRelSus2_,
	std::shared_ptr<Funcs_Special> pAlpha1_, std::shared_ptr<Funcs_Special> pAlpha2_)
{
	//increase reference count locally if passed in
	if (pCurieWeiss_) pCurieWeiss = pCurieWeiss_;
	if (pLongRelSus_) pLongRelSus = pLongRelSus_;
	if (pCurieWeiss1_) pCurieWeiss1 = pCurieWeiss1_;
	if (pCurieWeiss2_) pCurieWeiss2 = pCurieWeiss2_;
	if (pLongRelSus1_) pLongRelSus1 = pLongRelSus1_;
	if (pLongRelSus2_) pLongRelSus2 = pLongRelSus2_;
	if (pAlpha1_) pAlpha1 = pAlpha1_;
	if (pAlpha2_) pAlpha2 = pAlpha2_;

	//set them in text equation if available
	if (Tscaling_eq.is_set()) {

		if (pCurieWeiss) {

			Tscaling_eq.Set_SpecialFunction(EqComp::FUNC_CURIEWEISS, pCurieWeiss);

#if COMPILECUDA == 1
			//Update CUDA text equation also : set calculated special function in GPU memory, then make sure it's also set in the CUDA text equation
			if (pTscaling_CUDAeq && pCurieWeiss_CUDA) pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_CURIEWEISS, pCurieWeiss_CUDA);
#endif
		}

		if (pCurieWeiss1 && pCurieWeiss2) {

			Tscaling_eq.Set_SpecialFunction(EqComp::FUNC_CURIEWEISS1, pCurieWeiss1);
			Tscaling_eq.Set_SpecialFunction(EqComp::FUNC_CURIEWEISS2, pCurieWeiss2);

#if COMPILECUDA == 1
			//Update CUDA text equation also : set calculated special function in GPU memory, then make sure it's also set in the CUDA text equation
			if (pTscaling_CUDAeq && pCurieWeiss1_CUDA && pCurieWeiss2_CUDA) {

				pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_CURIEWEISS1, pCurieWeiss1_CUDA);
				pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_CURIEWEISS2, pCurieWeiss2_CUDA);
			}
#endif
		}

		if (pLongRelSus) {

			Tscaling_eq.Set_SpecialFunction(EqComp::FUNC_LONGRELSUS, pLongRelSus);

#if COMPILECUDA == 1
			//Update CUDA text equation also : set calculated special function in GPU memory, then make sure it's also set in the CUDA text equation
			if (pTscaling_CUDAeq && pLongRelSus_CUDA) pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_LONGRELSUS, pLongRelSus_CUDA);
#endif
		}

		if (pLongRelSus1 && pLongRelSus2) {
			
			Tscaling_eq.Set_SpecialFunction(EqComp::FUNC_LONGRELSUS1, pLongRelSus1);
			Tscaling_eq.Set_SpecialFunction(EqComp::FUNC_LONGRELSUS2, pLongRelSus2);

#if COMPILECUDA == 1
			//Update CUDA text equation also : set calculated special function in GPU memory, then make sure it's also set in the CUDA text equation
			if (pTscaling_CUDAeq && pLongRelSus1_CUDA && pLongRelSus2_CUDA) {

				pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_LONGRELSUS1, pLongRelSus1_CUDA);
				pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_LONGRELSUS2, pLongRelSus2_CUDA);
			}
#endif
		}

		if (pAlpha1 && pAlpha2) {

			Tscaling_eq.Set_SpecialFunction(EqComp::FUNC_ALPHA1, pAlpha1);
			Tscaling_eq.Set_SpecialFunction(EqComp::FUNC_ALPHA2, pAlpha2);

#if COMPILECUDA == 1
			//Update CUDA text equation also : set calculated special function in GPU memory, then make sure it's also set in the CUDA text equation
			if (pTscaling_CUDAeq && pAlpha1_CUDA && pAlpha2_CUDA) {

				pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_ALPHA1, pAlpha1_CUDA);
				pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_ALPHA1, pAlpha2_CUDA);
			}
#endif
		}
	}

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif
}

#if COMPILECUDA == 1
//set special functions in text equation, also increasing ref count locally if passed in : the Funcs_Special objects are computed externally.
template <typename PType, typename SType>
void MatP<PType, SType>::set_t_scaling_special_functions_CUDA(
	mcu_SpecialFunc* pCurieWeiss_CUDA_,
	mcu_SpecialFunc* pLongRelSus_CUDA_,
	mcu_SpecialFunc* pCurieWeiss1_CUDA_, mcu_SpecialFunc* pCurieWeiss2_CUDA_,
	mcu_SpecialFunc* pLongRelSus1_CUDA_, mcu_SpecialFunc* pLongRelSus2_CUDA_,
	mcu_SpecialFunc* pAlpha1_CUDA_, mcu_SpecialFunc* pAlpha2_CUDA_)
{
	pCurieWeiss_CUDA = pCurieWeiss_CUDA_;
	pLongRelSus_CUDA = pLongRelSus_CUDA_;
	pCurieWeiss1_CUDA = pCurieWeiss1_CUDA_;
	pCurieWeiss2_CUDA = pCurieWeiss2_CUDA_;
	pLongRelSus1_CUDA = pLongRelSus1_CUDA_;
	pLongRelSus2_CUDA = pLongRelSus2_CUDA_;
	pAlpha1_CUDA = pAlpha1_CUDA_;
	pAlpha2_CUDA = pAlpha2_CUDA_;

	if (Tscaling_eq.is_set()) {

		pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_CURIEWEISS, pCurieWeiss_CUDA);
		pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_LONGRELSUS, pLongRelSus_CUDA);
		pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_CURIEWEISS1, pCurieWeiss1_CUDA);
		pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_CURIEWEISS2, pCurieWeiss2_CUDA);
		pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_LONGRELSUS1, pLongRelSus1_CUDA);
		pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_LONGRELSUS2, pLongRelSus2_CUDA);
		pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_ALPHA1, pAlpha1_CUDA);
		pTscaling_CUDAeq->Set_SpecialFunction(EqComp::FUNC_ALPHA2, pAlpha2_CUDA);
	}
}
#endif

//---------Set scaling array

//clear any temperature dependence
template <typename PType, typename SType>
void MatP<PType, SType>::clear_t_scaling(void)
{
	//reset equation
	Tscaling_eq.clear();

	//clear scaling array
	t_scaling.clear();
	t_scaling_y.clear();
	t_scaling_z.clear();

	//temperature scaling info
	t_scaling_info = temperature_dependence_type(MATPTDEP_NONE);

	//update output value to value at 0K
	update();

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif
}

//calculate t_scaling from given temperature dependence of scaling coefficients
template <typename PType, typename SType>
bool MatP<PType, SType>::set_t_scaling_array(std::vector<double> temp_arr, std::vector<double> scaling_arr, std::vector<double> scaling_arr_y, std::vector<double> scaling_arr_z)
{
	//vectors must have same size and must have at least 2 points
	if (temp_arr.size() < 2 || temp_arr.size() != scaling_arr.size()) return false;
	if (scaling_arr_y.size() && temp_arr.size() != scaling_arr_y.size()) return false;
	if (scaling_arr_z.size() && temp_arr.size() != scaling_arr_z.size()) return false;
	if (scaling_arr_z.size() && !scaling_arr_y.size()) return false;

	//prep work: make sure temperature values are strictly increasing with corresponding order in scaling arrays kept : bundle everything in a vector of tuples to work on
	std::vector<std::tuple<double, double, double, double>> arr_bundle(temp_arr.size());
#pragma omp parallel for
	for (int idx = 0; idx < arr_bundle.size(); idx++) {

		double x = scaling_arr[idx], y = 0.0, z = 0.0;
		if (scaling_arr_y.size()) y = scaling_arr_y[idx];
		if (scaling_arr_z.size()) z = scaling_arr_z[idx];

		arr_bundle[idx] = { temp_arr[idx], x, y, z };
	}

	//make sure temperature is in increasing order
	//Version with parallel sort requires <execution>
	//std::sort(std::execution::par_unseq, arr_bundle.begin(), arr_bundle.end());
	//Without parallel sort
	std::sort(arr_bundle.begin(), arr_bundle.end());

	//extract from tuple back to vectors
	std::transform(arr_bundle.begin(), arr_bundle.end(), temp_arr.begin(), [](auto const& tup) { return std::get<0>(tup); });
	std::transform(arr_bundle.begin(), arr_bundle.end(), scaling_arr.begin(), [](auto const& tup) { return std::get<1>(tup); });
	if (scaling_arr_y.size()) std::transform(arr_bundle.begin(), arr_bundle.end(), scaling_arr_y.begin(), [](auto const& tup) { return std::get<2>(tup); });
	if (scaling_arr_z.size()) std::transform(arr_bundle.begin(), arr_bundle.end(), scaling_arr_z.begin(), [](auto const& tup) { return std::get<3>(tup); });

	//make sure temperature values are not repeated - values must be strictly increasing
	if (scaling_arr_y.size() && scaling_arr_z.size()) delete_repeats(temp_arr, scaling_arr, scaling_arr_y, scaling_arr_z);
	else if (scaling_arr_y.size()) delete_repeats(temp_arr, scaling_arr, scaling_arr_y);
	else delete_repeats(temp_arr, scaling_arr);

	auto calculate_t_scaling = [](std::vector<double>& temp_arr, std::vector<double>& scaling_arr, std::vector<double>& t_scaling) -> bool {

		//temperature values cannot be negative
		if (temp_arr[0] < 0) return false;

		//set tscaling size : covers temperature from 0K up to maximum possible temperature in 1K increments.
		double max_temp = minimum(temp_arr.back(), MAX_TEMPERATURE);
		t_scaling.resize((size_t)(floor_epsilon(max_temp) + 1));

		int temp_arr_idx = 0;

		//first temperature point : 0K
		double temp_a = 0;
		//next temperature point - will use interpolation to fill in any integer temperature values in between; this may coincide with temp_a
		double temp_b = temp_arr[0];

		//first scaling coefficient (at 0K) : extrapolate from first 2 points in the arrays
		double scaling_a = interpolate(DBL2(temp_arr[0], scaling_arr[0]), DBL2(temp_arr[1], scaling_arr[1]), 0);
		//next scaling coefficient
		double scaling_b = scaling_arr[0];

		//we know what the first value must be:
		t_scaling[temp_a] = scaling_a;

		for (int t_value = 1; t_value < (int)t_scaling.size(); t_value++) {

			//have we finished this [Ta, Tb) interval? Use a while since the temperature points in temp_arr may be finely spaced (less than 1K intervals)
			while ((double)t_value >= temp_b) {

				temp_arr_idx++;

				//have we reached the end? i.e. was temp_b the last temperature value in temp_arr? This can only happen if temp_b is an integer value, in which case t_value = temp_b now.
				if (temp_arr_idx >= (int)temp_arr.size()) {

					t_scaling[t_value] = scaling_arr.back();
					break;
				}

				//next [Ta, Tb) interval
				temp_a = temp_b;
				temp_b = temp_arr[temp_arr_idx];

				scaling_a = scaling_b;
				scaling_b = scaling_arr[temp_arr_idx];
			}

			//t_value is now in the interval [temp_a, temp_b). Set t_scaling value at t_value index using interpolation
			t_scaling[t_value] = interpolate(DBL2(temp_a, scaling_a), DBL2(temp_b, scaling_b), t_value);
		}

		return true;
	};

	if (!calculate_t_scaling(temp_arr, scaling_arr, t_scaling)) return false;

	//further components if required
	if (scaling_arr_y.size()) if (!calculate_t_scaling(temp_arr, scaling_arr_y, t_scaling_y)) return false;
	if (scaling_arr_z.size()) if (!calculate_t_scaling(temp_arr, scaling_arr_z, t_scaling_z)) return false;

	//no scaling equation - using array instead
	Tscaling_eq.clear();

	//temperature scaling info
	t_scaling_info = temperature_dependence_type(MATPTDEP_ARRAY);

	update();

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif
	
	return true;
}

template <typename PType, typename SType>
void MatP<PType, SType>::set_precalculated_t_scaling_array(std::vector<double>& scaling_arr, std::vector<double>& scaling_arr_y, std::vector<double>& scaling_arr_z)
{
	t_scaling = scaling_arr;

	if (scaling_arr_y.size() && scaling_arr_y.size() == scaling_arr.size()) t_scaling_y = scaling_arr_y;
	if (scaling_arr_z.size() && scaling_arr_z.size() == scaling_arr.size()) t_scaling_z = scaling_arr_z;

	//no scaling equation - using array instead
	Tscaling_eq.clear();

	//temperature scaling info
	t_scaling_info = temperature_dependence_type(MATPTDEP_ARRAY);

	update();

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif
}

//---------Set scaling info (temperature)

template <typename PType, typename SType>
void MatP<PType, SType>::set_t_scaling_info(std::string info_text)
{
	t_scaling_info = info_text;

	if (!t_scaling_info.length()) {

		//empty std::string: set default info

		if (!is_tdep()) t_scaling_info = temperature_dependence_type(MATPTDEP_NONE);
		else {

			if (is_t_equation_set()) t_scaling_info = temperature_dependence_type(MATPTDEP_EQUATION);
			else t_scaling_info = temperature_dependence_type(MATPTDEP_ARRAY);
		}
	}
}

//---------Set spatial dependence

//clear spatial scaling
template <typename PType, typename SType>
void MatP<PType, SType>::clear_s_scaling(void)
{
	s_scaling.clear();
	s_scaling_type = MATPVAR_NONE;
	s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_NONE);

	Sscaling_eq.clear();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif
}

//update spatial scaling VEC for cellsize and rectangle, depending on currently set spatial scaling
template <typename PType, typename SType>
bool MatP<PType, SType>::update_s_scaling(DBL3 h, Rect rect)
{
	//nothing to do
	if (s_scaling_type == MATPVAR_NONE) return true;

	//there is a spatial scaling set
	if (s_scaling_type == MATPVAR_MASK || s_scaling_type == MATPVAR_OVF2 || s_scaling_type == MATPVAR_SHAPE) {

		if (s_scaling_type == MATPVAR_MASK) {

			//for mask spatial scaling just stretch to new sizes, but keep it 2D always
			INT3 cells = round(rect / h);
			if (s_scaling.linear_size()) s_scaling.resize(rect / SZ3(cells.x, cells.y, 1), rect);
		}
		else {

			//for ovf2 and shape scaling just stretch to new sizes, but keep it 2D if already 2D only
			INT3 cells = round(rect / h);
			if (s_scaling.linear_size()) {

				if (s_scaling.h.z == 1) s_scaling.resize(rect / SZ3(cells.x, cells.y, 1), rect);
				else s_scaling.resize(rect / SZ3(cells), rect);
			}
		}
	}
	else if (s_scaling_type == MATPVAR_EQUATION) {

		Sscaling_eq.set_constants({ {"Lx", rect.size().x}, {"Ly", rect.size().y}, {"Lz", rect.size().z} });
	}
	else {

		//if cellsize and rect match then nothing to do
		if ((s_scaling.h == h) && (s_scaling.rect == rect)) return true;

		//fields in s_scaling_info first contains the generator name, then its args - all separated using "; " or ", "
		std::vector<std::string> fields = split(s_scaling_info, ", ", "; ");

		//get rid of first field, then recombine into a std::string using " " as separators
		BError error = set_s_scaling(h, rect, (MATPVAR_)s_scaling_type, combine(subvec(fields, 1), " "), std::function<std::vector<unsigned char>(std::string, INT2)>());
		if (error) return false;
	}

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return true;
}

//set parameter spatial variation using a given generator and arguments (arguments passed as a std::string to be interpreted and converted using ToNum)
template <typename PType, typename SType>
BError MatP<PType, SType>::set_s_scaling(DBL3 h, Rect rect, MATPVAR_ generatorID, std::string generatorArgs, const std::function<std::vector<unsigned char>(std::string, INT2)>& bitmap_loader)
{
	BError error(__FUNCTION__);

	Sscaling_eq.clear();

	switch (generatorID) {

	case MATPVAR_NONE:
		clear_s_scaling();
		break;

	case MATPVAR_MASK:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() < 3) return error(BERROR_INCORRECTVALUE);

		double offset = ToNum(fields[0]);
		double scale = ToNum(fields[1]);
		std::string fileName = combine(subvec(fields, 2), " ");

		if (!set_s_custom(h, rect, offset, scale, fileName, bitmap_loader)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_SHAPE:
	{
		if (!set_s_shape(h, rect, {}, SType())) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_OVF2:
	{
		std::string fileName = generatorArgs;

		if (!set_s_ovf2(h, rect, fileName)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_RANDOM:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 3) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		int seed = ToNum(fields[2]);

		if (!set_s_random(h, rect, range, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_JAGGED:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 4) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		double spacing = ToNum(fields[2], "m");
		int seed = ToNum(fields[3]);

		if (!set_s_jagged(h, rect, range, spacing, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_ABLPOL:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 9) return error(BERROR_INCORRECTVALUE);

		DBL2 abl_x = DBL2(ToNum(fields[0]), ToNum(fields[1]));
		DBL2 abl_y = DBL2(ToNum(fields[2]), ToNum(fields[3]));
		DBL2 abl_z = DBL2(ToNum(fields[4]), ToNum(fields[5]));
		DBL3 values = DBL3(ToNum(fields[6]), ToNum(fields[7]), ToNum(fields[8]));

		if (!set_s_ablpol(h, rect, abl_x, abl_y, abl_z, values)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_ABLTANH:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 9) return error(BERROR_INCORRECTVALUE);

		DBL2 abl_x = DBL2(ToNum(fields[0]), ToNum(fields[1]));
		DBL2 abl_y = DBL2(ToNum(fields[2]), ToNum(fields[3]));
		DBL2 abl_z = DBL2(ToNum(fields[4]), ToNum(fields[5]));
		DBL3 values = DBL3(ToNum(fields[6]), ToNum(fields[7]), ToNum(fields[8]));

		if (!set_s_abltanh(h, rect, abl_x, abl_y, abl_z, values)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_ABLEXP:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 9) return error(BERROR_INCORRECTVALUE);

		DBL2 abl_x = DBL2(ToNum(fields[0]), ToNum(fields[1]));
		DBL2 abl_y = DBL2(ToNum(fields[2]), ToNum(fields[3]));
		DBL2 abl_z = DBL2(ToNum(fields[4]), ToNum(fields[5]));
		DBL3 values = DBL3(ToNum(fields[6]), ToNum(fields[7]), ToNum(fields[8]));

		if (!set_s_ablexp(h, rect, abl_x, abl_y, abl_z, values)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_DEFECTS:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 6) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		DBL2 diameter_range = ToNum(fields[2] + " " + fields[3], "m");
		double spacing = ToNum(fields[4], "m");
		int seed = ToNum(fields[5]);

		if (!set_s_defects(h, rect, range, diameter_range, spacing, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_FAULTS:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 8) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		DBL2 length_range = ToNum(fields[2] + " " + fields[3], "m");
		DBL2 orientation_range = ToNum(fields[4] + " " + fields[5]);
		double spacing = ToNum(fields[6], "m");
		int seed = ToNum(fields[7]);

		if (!set_s_faults(h, rect, range, length_range, orientation_range, spacing, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_VORONOI2D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 4) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		double spacing = ToNum(fields[2], "m");
		int seed = ToNum(fields[3]);

		if (!set_s_voronoi2d(h, rect, range, spacing, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_UVORONOI2D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 5) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		double spacing = ToNum(fields[2], "m");
		double variation = ToNum(fields[3], "m");
		int seed = ToNum(fields[4]);

		if (!set_s_uvoronoi2d(h, rect, range, spacing, variation, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_VORONOI3D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 4) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		double spacing = ToNum(fields[2], "m");
		int seed = ToNum(fields[3]);

		if (!set_s_voronoi3d(h, rect, range, spacing, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_UVORONOI3D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 5) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		double spacing = ToNum(fields[2], "m");
		double variation = ToNum(fields[3], "m");
		int seed = ToNum(fields[4]);

		if (!set_s_uvoronoi3d(h, rect, range, spacing, variation, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_VORONOIBND2D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 4) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		double spacing = ToNum(fields[2], "m");
		int seed = ToNum(fields[3]);

		if (!set_s_voronoiboundary2d(h, rect, range, spacing, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_UVORONOIBND2D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 5) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		double spacing = ToNum(fields[2], "m");
		double variation = ToNum(fields[3], "m");
		int seed = ToNum(fields[4]);

		if (!set_s_uvoronoiboundary2d(h, rect, range, spacing, variation, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_VORONOIBND3D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 4) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		double spacing = ToNum(fields[2], "m");
		int seed = ToNum(fields[3]);

		if (!set_s_voronoiboundary3d(h, rect, range, spacing, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_UVORONOIBND3D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 5) return error(BERROR_INCORRECTVALUE);

		DBL2 range = ToNum(fields[0] + " " + fields[1]);
		double spacing = ToNum(fields[2], "m");
		double variation = ToNum(fields[3], "m");
		int seed = ToNum(fields[4]);

		if (!set_s_uvoronoiboundary3d(h, rect, range, spacing, variation, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_VORONOIROT2D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 6) return error(BERROR_INCORRECTVALUE);

		DBL2 range_theta = ToNum(fields[0] + " " + fields[1]);
		DBL2 range_phi = ToNum(fields[2] + " " + fields[3]);
		double spacing = ToNum(fields[4], "m");
		int seed = ToNum(fields[5]);

		if (!set_s_voronoirotation2d(h, rect, range_theta, range_phi, spacing, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_UVORONOIROT2D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 7) return error(BERROR_INCORRECTVALUE);

		DBL2 range_theta = ToNum(fields[0] + " " + fields[1]);
		DBL2 range_phi = ToNum(fields[2] + " " + fields[3]);
		double spacing = ToNum(fields[4], "m");
		double variation = ToNum(fields[5], "m");
		int seed = ToNum(fields[6]);

		if (!set_s_uvoronoirotation2d(h, rect, range_theta, range_phi, spacing, variation, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_VORONOIROT3D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 6) return error(BERROR_INCORRECTVALUE);

		DBL2 range_theta = ToNum(fields[0] + " " + fields[1]);
		DBL2 range_phi = ToNum(fields[2] + " " + fields[3]);
		double spacing = ToNum(fields[4], "m");
		int seed = ToNum(fields[5]);

		if (!set_s_voronoirotation3d(h, rect, range_theta, range_phi, spacing, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;

	case MATPVAR_UVORONOIROT3D:
	{
		std::vector<std::string> fields = split(generatorArgs, " ");

		if (fields.size() != 7) return error(BERROR_INCORRECTVALUE);

		DBL2 range_theta = ToNum(fields[0] + " " + fields[1]);
		DBL2 range_phi = ToNum(fields[2] + " " + fields[3]);
		double spacing = ToNum(fields[4], "m");
		double variation = ToNum(fields[5], "m");
		int seed = ToNum(fields[6]);

		if (!set_s_uvoronoirotation3d(h, rect, range_theta, range_phi, spacing, variation, seed)) error(BERROR_OUTOFMEMORY_NCRIT);
	}
	break;
	}

	return error;
}

//set value in given composite shape : this calls set_s_shape in turn
template <typename PType, typename SType>
BError MatP<PType, SType>::set_s_scaling_shape(DBL3 h, Rect rect, std::vector<MeshShape> shapes, SType value)
{
	BError error(__FUNCTION__);

	Sscaling_eq.clear();

	if (!set_s_shape(h, rect, shapes, value)) error(BERROR_OUTOFMEMORY_NCRIT);

	return error;
}

//set spatial scaling text equation
template <typename PType, typename SType>
void MatP<PType, SType>::set_s_scaling_equation(const std::string& equationText, const vector_key<double>& userConstants, DBL3 meshDimensions)
{
	std::vector<std::pair<std::string, double>> constants(userConstants.size());
	for (int idx = 0; idx < constants.size(); idx++) {

		constants[idx] = { userConstants.get_key_from_index(idx), userConstants[idx] };
	}

	Sscaling_eq.set_constants(constants, false);
	Sscaling_eq.make_from_string(equationText, { {"Lx", meshDimensions.x}, {"Ly", meshDimensions.y}, {"Lz", meshDimensions.z} });

	//clear scaling array - you either use scaling array method, or equation - or none - not both
	s_scaling.clear();

	s_scaling_type = MATPVAR_EQUATION;
	s_scaling_info = Sscaling_eq.show_equation();

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif
}

//---------Set spatial dependence : generator methods

//custom, set from an image file : coefficients vary from 0 : black to 1 : white
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_custom(DBL3 h, Rect rect, double offset, double scale, std::string fileName, const std::function<std::vector<unsigned char>(std::string, INT2)>& bitmap_loader)
{
	if (GetFileTermination(fileName) != ".png")
		fileName += ".png";
	
	if (!GetFilenameDirectory(fileName).length()) fileName = directory + fileName;

	INT3 cells = round(rect / h);
	bool success = s_scaling.generate_custom_2D(SZ3(cells.x, cells.y, 1), rect, offset, scale, bitmap_loader(fileName, INT2(cells.x, cells.y)));

	//don't display path or termination in paramvar desription
	ExtractFilenameDirectory(fileName);
	ExtractFilenameTermination(fileName);

	s_scaling_type = MATPVAR_MASK;
	s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_MASK) + "; " + ToString(offset) + "; " + ToString(scale) + "; " + fileName;

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	//allow setting a wrong filename without causing error messages
	return true;
}

//shape : set value only in shape
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_shape(DBL3 h, Rect rect, std::vector<MeshShape> shapes, SType value)
{
	bool success = true;

	//make sure the data is loaded for the current mesh dimensions
	success = s_scaling.resize(h, rect);

	if (success) {

		s_scaling_type = MATPVAR_SHAPE;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_SHAPE);

		if (shapes.size()) {

			s_scaling.shape_setvalue(shapes, value);
		}
	}

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//set s_scaling VEC by loading it form the named OVF2 file -  data types must match, but data is stretched to given mesh dimensions.
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_ovf2(DBL3 h, Rect rect, std::string fileName)
{
	if (GetFileTermination(fileName) != ".ovf")
		fileName += ".ovf";

	if (!GetFilenameDirectory(fileName).length()) fileName = directory + fileName;

	OVF2 ovf;
	ovf.Read_OVF2(fileName, s_scaling);

	//make sure the data is loaded for the current mesh dimensions
	s_scaling.resize(h, rect);

	ExtractFilenameDirectory(fileName);
	ExtractFilenameTermination(fileName);
	s_scaling_type = MATPVAR_OVF2;
	s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_OVF2) + "; " + fileName;

#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	//allow setting a wrong filename without causing error messages
	return true;
}

//random points in given range
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_random(DBL3 h, Rect rect, DBL2 range, int seed)
{
	bool success = s_scaling.generate_random(h, rect, range, seed);

	if (success) {

		s_scaling_type = MATPVAR_RANDOM;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_RANDOM) + "; " + ToString(range) + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//random points in given range at given square spacing in xy plane : in between fill using bilinear interpolation
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_jagged(DBL3 h, Rect rect, DBL2 range, double spacing, int seed)
{
	bool success = s_scaling.generate_jagged(h, rect, range, spacing, seed);

	if (success) {

		s_scaling_type = MATPVAR_JAGGED;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_JAGGED) + "; " + ToString(range) + "; " + ToString(spacing, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//polynomial slopes at slides with exponent n, starting from maximum value at surfaces, proceeding inwards towards minimum up to a depth of length * ratio
//abl_x, abl_y, abl_z : depth ratio for negative side, depth ratio for positive side
//values : minimum centre, maximum outer, polynomial exponent
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_ablpol(DBL3 h, Rect rect, DBL2 abl_x, DBL2 abl_y, DBL2 abl_z, DBL3 values)
{
	bool success = s_scaling.generate_ablpol(h, rect, abl_x, abl_y, abl_z, values);

	if (success) {

		s_scaling_type = MATPVAR_ABLPOL;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_ABLPOL) + "; " + ToString(abl_x) + "; " + ToString(abl_y) + "; " + ToString(abl_z) + "; " + ToString(values);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//tanh slopes at slides with exponent n, starting from maximum value at surfaces, proceeding inwards towards minimum up to a depth of length * ratio
//tanh profile centered
//abl_x, abl_y, abl_z : depth ratio for negative side, depth ratio for positive side
//values : minimum centre, maximum outer, tanh sigma in nm
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_abltanh(DBL3 h, Rect rect, DBL2 abl_x, DBL2 abl_y, DBL2 abl_z, DBL3 values)
{
	bool success = s_scaling.generate_abltanh(h, rect, abl_x, abl_y, abl_z, values);

	if (success) {

		s_scaling_type = MATPVAR_ABLTANH;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_ABLTANH) + "; " + ToString(abl_x) + "; " + ToString(abl_y) + "; " + ToString(abl_z) + "; " + ToString(values);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//exponential slopes at slides with exponent n, starting from maximum value at surfaces, proceeding inwards towards minimum up to a depth of length * ratio
//abl_x, abl_y, abl_z : depth ratio for negative side, depth ratio for positive side
//values : minimum centre, maximum outer, exponential sigma in nm
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_ablexp(DBL3 h, Rect rect, DBL2 abl_x, DBL2 abl_y, DBL2 abl_z, DBL3 values)
{
	bool success = s_scaling.generate_ablexp(h, rect, abl_x, abl_y, abl_z, values);

	if (success) {

		s_scaling_type = MATPVAR_ABLEXP;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_ABLEXP) + "; " + ToString(abl_x) + "; " + ToString(abl_y) + "; " + ToString(abl_z) + "; " + ToString(values);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//generate circular defects with a tanh radial profile with values in the given range, diameter range and average spacing(prng instantiated with given seed).The defect positioning is random.
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_defects(DBL3 h, Rect rect, DBL2 range, DBL2 diameter_range, double spacing, int seed)
{
	bool success = s_scaling.generate_defects(h, rect, range, range.i, diameter_range, spacing, seed);

	if (success) {

		s_scaling_type = MATPVAR_DEFECTS;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_DEFECTS) + "; " + ToString(range) + "; " + ToString(diameter_range, "m") + "; " + ToString(spacing, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//generate circular defects with a tanh radial profile with values in the given range, diameter range and average spacing(prng instantiated with given seed).The defect positioning is random.
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_faults(DBL3 h, Rect rect, DBL2 range, DBL2 length_range, DBL2 orientation_range, double spacing, int seed)
{
	bool success = s_scaling.generate_faults(h, rect, range, 1.0, length_range, orientation_range, spacing, seed);

	if (success) {

		s_scaling_type = MATPVAR_FAULTS;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_FAULTS) + "; " + ToString(range) + "; " + ToString(length_range, "m") + "; " + ToString(orientation_range) + "; " + ToString(spacing, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//generate voronoi 2d tessellation with values in the given range, and average spacing (prng instantiated with given seed). The values range is used to generate coefficient multipliers which are constant in each voronoi cell.
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_voronoi2d(DBL3 h, Rect rect, DBL2 range, double spacing, int seed)
{
	bool success = s_scaling.generate_Voronoi2D(h, rect, range, spacing, seed);

	if (success) {

		s_scaling_type = MATPVAR_VORONOI2D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_VORONOI2D) + "; " + ToString(range) + "; " + ToString(spacing, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_uvoronoi2d(DBL3 h, Rect rect, DBL2 range, double spacing, double variation, int seed)
{
	bool success = s_scaling.generate_uVoronoi2D(h, rect, range, spacing, variation, seed);

	if (success) {

		s_scaling_type = MATPVAR_UVORONOI2D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_UVORONOI2D) + "; " + ToString(range) + "; " + ToString(spacing, "m") + "; " + ToString(variation, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//generate voronoi 3d tessellation with values in the given range, and average spacing (prng instantiated with given seed). The values range is used to generate coefficient multipliers which are constant in each voronoi cell.
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_voronoi3d(DBL3 h, Rect rect, DBL2 range, double spacing, int seed)
{
	bool success = s_scaling.generate_Voronoi3D(h, rect, range, spacing, seed);

	if (success) {

		s_scaling_type = MATPVAR_VORONOI3D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_VORONOI3D) + "; " + ToString(range) + "; " + ToString(spacing, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_uvoronoi3d(DBL3 h, Rect rect, DBL2 range, double spacing, double variation, int seed)
{
	bool success = s_scaling.generate_uVoronoi3D(h, rect, range, spacing, variation, seed);

	if (success) {

		s_scaling_type = MATPVAR_UVORONOI3D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_UVORONOI3D) + "; " + ToString(range) + "; " + ToString(spacing, "m") + "; " + ToString(variation, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//generate voronoi 2d tessellation with average spacing. Set coefficient values randomly in the given range only at the Voronoi cell boundaries (prng instantiated with given seed).
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_voronoiboundary2d(DBL3 h, Rect rect, DBL2 range, double spacing, int seed)
{
	bool success = s_scaling.generate_VoronoiBoundary2D(h, rect, range, SType(1), spacing, seed);

	if (success) {

		s_scaling_type = MATPVAR_VORONOIBND2D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_VORONOIBND2D) + "; " + ToString(range) + "; " + ToString(spacing, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_uvoronoiboundary2d(DBL3 h, Rect rect, DBL2 range, double spacing, double variation, int seed)
{
	bool success = s_scaling.generate_uVoronoiBoundary2D(h, rect, range, SType(1), spacing, variation, seed);

	if (success) {

		s_scaling_type = MATPVAR_UVORONOIBND2D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_UVORONOIBND2D) + "; " + ToString(range) + "; " + ToString(spacing, "m") + "; " + ToString(variation, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//generate voronoi 3d tessellation with average spacing. Set coefficient values randomly in the given range only at the Voronoi cell boundaries (prng instantiated with given seed).
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_voronoiboundary3d(DBL3 h, Rect rect, DBL2 range, double spacing, int seed)
{
	bool success = s_scaling.generate_VoronoiBoundary3D(h, rect, range, SType(1), spacing, seed);

	if (success) {

		s_scaling_type = MATPVAR_VORONOIBND3D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_VORONOIBND3D) + "; " + ToString(range) + "; " + ToString(spacing, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_uvoronoiboundary3d(DBL3 h, Rect rect, DBL2 range, double spacing, double variation, int seed)
{
	bool success = s_scaling.generate_uVoronoiBoundary3D(h, rect, range, SType(1), spacing, variation, seed);

	if (success) {

		s_scaling_type = MATPVAR_UVORONOIBND3D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_UVORONOIBND3D) + "; " + ToString(range) + "; " + ToString(spacing, "m") + "; " + ToString(variation, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//generate voronoi 2d tessellation with average spacing. This method is applicable only to DBL3 PType, where a rotation operation is applied, fixed in each Voronoi cell. 
//The rotation uses the values for polar (theta) and azimuthal (phi) angles specified in given ranges. prng instantiated with given seed.
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_voronoirotation2d(DBL3 h, Rect rect, DBL2 theta, DBL2 phi, double spacing, int seed)
{
	bool success = s_scaling.generate_VoronoiRotation2D(h, rect, theta, phi, spacing, seed);

	if (success) {

		s_scaling_type = MATPVAR_VORONOIROT2D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_VORONOIROT2D) + "; " + ToString(theta) + "; " + ToString(phi) + "; " + ToString(spacing, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_uvoronoirotation2d(DBL3 h, Rect rect, DBL2 theta, DBL2 phi, double spacing, double variation, int seed)
{
	bool success = s_scaling.generate_uVoronoiRotation2D(h, rect, theta, phi, spacing, variation, seed);

	if (success) {

		s_scaling_type = MATPVAR_UVORONOIROT2D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_UVORONOIROT2D) + "; " + ToString(theta) + "; " + ToString(phi) + "; " + ToString(spacing, "m") + "; " + ToString(variation, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//generate voronoi 3d tessellation with average spacing. This method is applicable only to DBL3 PType, where a rotation operation is applied, fixed in each Voronoi cell. 
//The rotation uses the values for polar (theta) and azimuthal (phi) angles specified in given ranges. prng instantiated with given seed.
template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_voronoirotation3d(DBL3 h, Rect rect, DBL2 theta, DBL2 phi, double spacing, int seed)
{
	bool success = s_scaling.generate_VoronoiRotation3D(h, rect, theta, phi, spacing, seed);

	if (success) {

		s_scaling_type = MATPVAR_VORONOIROT3D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_VORONOIROT3D) + "; " + ToString(theta) + "; " + ToString(phi) + "; " + ToString(spacing, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

template <typename PType, typename SType>
bool MatP<PType, SType>::set_s_uvoronoirotation3d(DBL3 h, Rect rect, DBL2 theta, DBL2 phi, double spacing, double variation, int seed)
{
	bool success = s_scaling.generate_uVoronoiRotation3D(h, rect, theta, phi, spacing, variation, seed);

	if (success) {

		s_scaling_type = MATPVAR_UVORONOIROT3D;
		s_scaling_info = vargenerator_descriptor.get_key_from_ID(MATPVAR_UVORONOIROT3D) + "; " + ToString(theta) + "; " + ToString(phi) + "; " + ToString(spacing, "m") + "; " + ToString(variation, "m") + "; " + ToString(seed);
	}
	else clear_s_scaling();

	//update cuda object also if set
#if COMPILECUDA == 1
	if (p_cu_obj_mpcuda) update_cuda_object();
#endif

	return success;
}

//---------Get temperature dependence

//get temperature scaling coefficients as an array from 0K up to and including max_temperature
template <typename PType, typename SType>
bool MatP<PType, SType>::get_temperature_scaling(double max_temperature, std::vector<double>& get_scaling_x, std::vector<double>& get_scaling_y, std::vector<double>& get_scaling_z)
{
	if (max_temperature < 0 || max_temperature > MAX_TEMPERATURE) return false;

	if (!malloc_vector(get_scaling_x, (int)floor_epsilon(max_temperature) + 1)) return false;

	if (Tscaling_eq.is_set_dual() || t_scaling_y.size()) if (!malloc_vector(get_scaling_y, (int)floor_epsilon(max_temperature) + 1)) return false;
	if (Tscaling_eq.is_set_vector() || t_scaling_z.size()) if (!malloc_vector(get_scaling_z, (int)floor_epsilon(max_temperature) + 1)) return false;

	for (int t_value = 0; t_value <= max_temperature; t_value++) {

		if (Tscaling_eq.is_set()) {

			//use pre-set equation
			if (Tscaling_eq.is_set_vector()) {

				DBL3 value = Tscaling_eq.evaluate_vector((double)t_value);

				get_scaling_x[t_value] = value.x;
				get_scaling_y[t_value] = value.y;
				get_scaling_z[t_value] = value.z;
			}
			else if (Tscaling_eq.is_set_dual()) {

				DBL2 value = Tscaling_eq.evaluate_dual((double)t_value);

				get_scaling_x[t_value] = value.x;
				get_scaling_y[t_value] = value.y;
			}
			else get_scaling_x[t_value] = Tscaling_eq.evaluate((double)t_value);
		}
		else if (t_value < t_scaling.size()) {

			get_scaling_x[t_value] = t_scaling[t_value];

			if (t_scaling_y.size()) get_scaling_y[t_value] = t_scaling_y[t_value];
			if (t_scaling_z.size()) get_scaling_z[t_value] = t_scaling_z[t_value];
		}
		else if (t_scaling.size()) {

			get_scaling_x[t_value] = t_scaling.back();

			if (t_scaling_y.size()) get_scaling_y[t_value] = t_scaling_y.back();
			if (t_scaling_z.size()) get_scaling_z[t_value] = t_scaling_z.back();
		}
		else get_scaling_x[t_value] = 1.0;
	}

	return true;
}

//---------Get spatial variation

//scalar equation
template <>
inline void MatP<double, double>::calculate_s_scaling(VEC<double>& displayVEC_SCA, double stime)
{
	if (Sscaling_eq.is_set()) {

#pragma omp parallel for
		for (int idx = 0; idx < displayVEC_SCA.linear_size(); idx++) {

			DBL3 position = displayVEC_SCA.cellidx_to_position(idx);
			displayVEC_SCA[idx] = Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
		}
	}
}

//scalar equation
template <>
inline void MatP<DBL2, double>::calculate_s_scaling(VEC<double>& displayVEC_SCA, double stime)
{
	if (Sscaling_eq.is_set()) {

#pragma omp parallel for
		for (int idx = 0; idx < displayVEC_SCA.linear_size(); idx++) {

			DBL3 position = displayVEC_SCA.cellidx_to_position(idx);
			displayVEC_SCA[idx] = Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
		}
	}
}

//scalar equation
template <>
inline void MatP<DBL3, double>::calculate_s_scaling(VEC<double>& displayVEC_SCA, double stime)
{
	if (Sscaling_eq.is_set()) {

#pragma omp parallel for
		for (int idx = 0; idx < displayVEC_SCA.linear_size(); idx++) {

			DBL3 position = displayVEC_SCA.cellidx_to_position(idx);
			displayVEC_SCA[idx] = Sscaling_eq.evaluate(position.x, position.y, position.z, stime);
		}
	}
}

//N/A : keep compiler happy
template <>
inline void MatP<DBL3, DBL3>::calculate_s_scaling(VEC<double>& displayVEC_SCA, double stime) {}

//N/A : keep compiler happy
template <>
inline void MatP<double, double>::calculate_s_scaling(VEC<DBL3>& displayVEC_VEC, double stime) {}

//N/A : keep compiler happy
template <>
inline void MatP<DBL2, double>::calculate_s_scaling(VEC<DBL3>& displayVEC_VEC, double stime) {}

//N/A : keep compiler happy
template <>
inline void MatP<DBL3, double>::calculate_s_scaling(VEC<DBL3>& displayVEC_VEC, double stime) {}

//vector equation
template <>
inline void MatP<DBL3, DBL3>::calculate_s_scaling(VEC<DBL3>& displayVEC_VEC, double stime)
{
	if (Sscaling_eq.is_set()) {

#pragma omp parallel for
		for (int idx = 0; idx < displayVEC_VEC.linear_size(); idx++) {

			DBL3 position = displayVEC_VEC.cellidx_to_position(idx);
			displayVEC_VEC[idx] = Sscaling_eq.evaluate_vector(position.x, position.y, position.z, stime);
		}
	}
}