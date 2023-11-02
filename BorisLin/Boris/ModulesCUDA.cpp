#include "stdafx.h"

#include "ModulesCUDA.h"
#if COMPILECUDA == 1

//-------------------------- Energies and Torques

//Calculate the energy density in the given rect only
cuBReal ModulesCUDA::GetEnergyDensity(cuRect avRect)
{
	cuBReal energy1 = 0.0, energy2 = 0.0;

	if (Module_energy.linear_size_cpu()) {

		energy1 = Module_energy.average_nonempty(avRect);
	}

	if (Module_energy2.linear_size_cpu()) {

		energy2 = Module_energy2.average_nonempty(avRect);
		return (energy1 + energy2) / 2;
	}
	else return energy1;
}

//-------------------------- Effective field and energy VECs

//Make sure memory is allocated correctly for display data if used, else free memory
BError ModulesCUDA::Update_Module_Display_VECs(cuReal3 h, cuRect meshRect, bool Module_Heff_used, bool Module_Energy_used, bool twosublattice)
{
	BError error(CLASS_STR(ModulesCUDA));

	//1. Heff - sub-lattice A

	if (Module_Heff.size_cpu().dim()) {

		if (Module_Heff_used && !Module_Heff.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (!Module_Heff_used) Module_Heff.clear();
	}
	else if (Module_Heff_used) {

		if (!Module_Heff.assign(h, meshRect, cuReal3())) return error(BERROR_OUTOFGPUMEMORY_CRIT);
	}

	//2. Heff - sub-lattice B

	if (Module_Heff2.size_cpu().dim()) {

		if (twosublattice && Module_Heff_used && !Module_Heff2.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else Module_Heff2.clear();
	}
	else if (twosublattice && Module_Heff_used) {

		if (!Module_Heff2.assign(h, meshRect, cuReal3())) return error(BERROR_OUTOFGPUMEMORY_CRIT);
	}

	//3. Energy Density - sub-lattice A

	if (Module_energy.size_cpu().dim()) {

		if (Module_Energy_used && !Module_energy.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else if (!Module_Energy_used) Module_energy.clear();
	}
	else if (Module_Energy_used) {

		if (!Module_energy.assign(h, meshRect, 0.0)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
	}

	//4. Energy Density - sub-lattice B

	if (Module_energy2.size_cpu().dim()) {

		if (twosublattice && Module_Energy_used && !Module_energy2.resize(h, meshRect)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
		else Module_energy2.clear();
	}
	else if (twosublattice && Module_Energy_used) {

		if (!Module_energy2.assign(h, meshRect, 0.0)) return error(BERROR_OUTOFGPUMEMORY_CRIT);
	}

	return error;
}

#endif