#pragma once

#include "Simulation.h"

#include "ColorDefs.h"

void Simulation::MakeIOInfo(void)
{
	std::string versionupdate_info =
		std::string("[tc1,1,0,1/tc]<b>Program Version Status</b>") +
		std::string("\n[tc1,1,0,1/tc]click: open download page\n");

	ioInfo.push_back(versionupdate_info, IOI_PROGRAMUPDATESTATUS);

	//Data box entry, showing the label of a given entry in Simulation::dataBoxList : minorId is the minor id of elements in Simulation::dataBoxList (major id there is always 0), auxId is the number of the interactive object in the list (i.e. entry number as it appears in data box in order). textId is the mesh name (if associated with this data type)
	//Note this entry must always represent the entry in Simulation::dataBoxList with the index in auxId.
	//IOI_DATABOXFIELDLABEL

	//A set or available module for a given mesh: minorId in InteractiveObjectProperties is an entry from MOD_ enum identifying the module, auxId contains the unique mesh id number this module refers to
	//IOI_MODULE
	//super-mesh module : minor type is an entry from MOD_ enum
	//IOI_SMODULE

	std::string modulegeneric_info =
		std::string("[tc1,1,0,1/tc]<b>computational module</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: on, red: off</i>") +
		std::string("\n[tc1,1,0,1/tc]left-click: enable") +
		std::string("\n[tc1,1,0,1/tc]right-click: disable\n");

	ioInfo.set(modulegeneric_info + std::string("<i><b>Stoner-Wohlfarth demag"), INT2(IOI_MODULE, MOD_DEMAG_N));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Full demag field"), INT2(IOI_MODULE, MOD_DEMAG));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Dipole-dipole interaction"), INT2(IOI_MODULE, MOD_ATOM_DIPOLEDIPOLE));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Direct exchange interaction"), INT2(IOI_MODULE, MOD_EXCHANGE));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Dzyaloshinskii-Moriya interaction - bulk"), INT2(IOI_MODULE, MOD_DMEXCHANGE));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Dzyaloshinskii-Moriya interaction - interfacial"), INT2(IOI_MODULE, MOD_IDMEXCHANGE));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Dzyaloshinskii-Moriya interaction - vector interfacial"), INT2(IOI_MODULE, MOD_VIDMEXCHANGE));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Surface exchange interaction\n<i><b>Couple to adjacent meshes along z\n<i><b>with surfexchange module enabled"), INT2(IOI_MODULE, MOD_SURFEXCHANGE));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Applied field term"), INT2(IOI_MODULE, MOD_ZEEMAN));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Magneto-Optical term"), INT2(IOI_MODULE, MOD_MOPTICAL));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Magnetocrystalline anisotropy: uniaxial"), INT2(IOI_MODULE, MOD_ANIUNI));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Magnetocrystalline anisotropy: cubic"), INT2(IOI_MODULE, MOD_ANICUBI));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Magnetocrystalline anisotropy: biaxial"), INT2(IOI_MODULE, MOD_ANIBI));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Magnetocrystalline anisotropy: tensorial"), INT2(IOI_MODULE, MOD_ANITENS));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Magneto-elastic term"), INT2(IOI_MODULE, MOD_MELASTIC));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Charge and spin transport"), INT2(IOI_MODULE, MOD_TRANSPORT));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Tunneling magneto-resistance"), INT2(IOI_MODULE, MOD_TMR));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Heat equation solver"), INT2(IOI_MODULE, MOD_HEAT));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Spin-orbit torque field\n<i><b>Results in DL and FL spin-orbit torques"), INT2(IOI_MODULE, MOD_SOTFIELD));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Spin-orbit torque field\n<i><b>Results in DL and FL Slonczewski torques"), INT2(IOI_MODULE, MOD_STFIELD));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Physical roughness\n<i><b>Demag term corrections when approximating shapes\n<i><b>from a fine mesh with a coarse mesh."), INT2(IOI_MODULE, MOD_ROUGHNESS));

	ioInfo.set(modulegeneric_info + std::string("<i><b>Supermesh demag field"), INT2(IOI_SMODULE, MODS_SDEMAG));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Oersted field in electric supermesh"), INT2(IOI_SMODULE, MODS_OERSTED));
	ioInfo.set(modulegeneric_info + std::string("<i><b>Stray field from dipole meshes"), INT2(IOI_SMODULE, MODS_STRAYFIELD));

	//Module used for effective field display for a given mesh: minorId in InteractiveObjectProperties is an entry from MOD_ enum identifying the module, auxId contains the unique mesh id number this module refers to, textId is the MOD_ value used for display
	//IOI_DISPLAYMODULE:

	std::string IOI_DISPLAYMODULE_info =
		std::string("[tc1,1,0,1/tc]<b>Module for Heff and E display</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>orange: selected, red: off</i>") +
		std::string("\n[tc1,1,0,1/tc]left-click: select") +
		std::string("\n[tc1,1,0,1/tc]right-click: deselect\n");

	ioInfo.push_back(IOI_DISPLAYMODULE_info, IOI_DISPLAYMODULE);

	//Available/set ode : minorId is an entry from ODE_ (the equation)
	//IOI_ODE

	std::string ode_info =
		std::string("[tc1,1,0,1/tc]<b>dM/dt equation</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: on, red: off</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change state\n");

	ioInfo.push_back(ode_info, IOI_ODE);

	//Available/set ode in atomistic meshes: minorId is an entry from ODE_ (the equation)
	//IOI_ATOMODE

	std::string atomode_info =
		std::string("[tc1,1,0,1/tc]<b>dM/dt atomistic equation</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: on, red: off</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change state\n");

	ioInfo.push_back(atomode_info, IOI_ATOMODE);

	//Set ODE time step: textId is the value
	//IOI_ODEDT

	std::string odedt_info =
		std::string("[tc1,1,0,1/tc]<b>ODE Time Step</b>") +
		std::string("\n[tc1,1,0,1/tc]double-click: edit\n");

	ioInfo.push_back(odedt_info, IOI_ODEDT);

	//Set heat equation time step: textId is the value
	//IOI_HEATDT

	std::string odeheatdt_info =
		std::string("[tc1,1,0,1/tc]<b>Heat Equation Time Step</b>") +
		std::string("\n[tc1,1,0,1/tc]double-click: edit\n");

	ioInfo.push_back(odeheatdt_info, IOI_HEATDT);

	//Set elastodynamics equation time step: textId is the value
	//IOI_ELDT

	std::string odeeldt_info =
		std::string("[tc1,1,0,1/tc]<b>Elastodynamics Equation Time Step</b>") +
		std::string("\n[tc1,1,0,1/tc]double-click: edit\n");

	ioInfo.push_back(odeeldt_info, IOI_ELDT);

	//Link elastodynamics time-step to ODE dT flag : auxId is the value
	//IOI_LINKELDT

	std::string linkeldt_info =
		std::string("[tc1,1,0,1/tc]<b>Link eldT to ODE dT</b>") +
		std::string("\n[tc1,1,0,1/tc]click: change\n");

	ioInfo.push_back(linkeldt_info, IOI_LINKELDT);

	//Set stochastic time-step: textId is the value
	//IOI_STOCHDT

	std::string stochdt_info =
		std::string("[tc1,1,0,1/tc]<b>Stochastic Time Step</b>") +
		std::string("\n[tc1,1,0,1/tc]double-click: edit\n");

	ioInfo.push_back(stochdt_info, IOI_STOCHDT);

	//Link stochastic time-step to ODE dT flag : auxId is the value
	//IOI_LINKSTOCHDT

	std::string linkstochdt_info =
		std::string("[tc1,1,0,1/tc]<b>Link dTstoch to ODE dT</b>") +
		std::string("\n[tc1,1,0,1/tc]click: change\n");

	ioInfo.push_back(linkstochdt_info, IOI_LINKSTOCHDT);

	//Set evaluation speedup time-step: textId is the value
	//IOI_SPEEDUPDT

	std::string speedupdt_info =
		std::string("[tc1,1,0,1/tc]<b>Evaluation Speedup Time Step</b>") +
		std::string("\n[tc1,1,0,1/tc]double-click: edit\n");

	ioInfo.push_back(speedupdt_info, IOI_SPEEDUPDT);

	//Link evaluation speedup time-step to ODE dT flag : auxId is the value
	//IOI_LINKSPEEDUPDT

	std::string linkspeedupdt_info =
		std::string("[tc1,1,0,1/tc]<b>Link dTspeedup to ODE dT</b>") +
		std::string("\n[tc1,1,0,1/tc]click: change\n");

	ioInfo.push_back(linkspeedupdt_info, IOI_LINKSPEEDUPDT);

	//Available/set evaluation method for ode : minorId is an entry from ODE_ as : micromagnetic equation value + 100 * atomistic equation value, auxId is the EVAL_ entry (the evaluation method), textId is the name of the evaluation method
	//IOI_ODE_EVAL

	std::string ode_eval_info =
		std::string("[tc1,1,0,1/tc]<b>Evaluation method for dM/dt equation</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: on, red: off, gray: unavailable</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change state\n");

	ioInfo.push_back(ode_eval_info, IOI_ODE_EVAL);

	//Shows a mesh name : minorId is the unique mesh id number, textId is the mesh name (below are similar objects but used in different lists, so these lists need updating differently).
	//auxId is also used : value 1 means update list, value 0 means do not update list (but delete line if mesh is deleted).
	//IOI_MESH_FORPARAMS
	//IOI_MESH_FORPARAMSTEMP
	//IOI_MESH_FORPARAMSVAR
	//IOI_MESH_FORMODULES
	//IOI_MESH_FORMESHLIST
	//IOI_MESH_FORDISPLAYOPTIONS
	//IOI_MESH_FORTEMPERATURE
	//IOI_MESH_FORHEATBOUNDARIES
	//IOI_MESH_FORCURIEANDMOMENT
	//IOI_MESH_FORTMODEL
	//IOI_MESH_FORPBC
	//IOI_MESH_FOREXCHCOUPLING
	//IOI_MESH_FORSTOCHASTICITY
	//IOI_MESH_FORELASTICITY
	//IOI_MESH_FORSPEEDUP

	std::string mesh_info =
		std::string("[tc1,1,0,1/tc]<b>mesh name</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: selected, red: not selected</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit") +
		std::string("\n[tc1,1,0,1/tc]left-click: select") +
		std::string("\n[tc1,1,0,1/tc]right-click: delete");

	ioInfo.push_back(mesh_info, IOI_MESH_FORPARAMS);
	ioInfo.push_back(mesh_info, IOI_MESH_FORPARAMSTEMP);
	ioInfo.push_back(mesh_info, IOI_MESH_FORPARAMSVAR);
	ioInfo.push_back(mesh_info, IOI_MESH_FORMODULES);
	ioInfo.push_back(mesh_info, IOI_MESH_FORDISPLAYMODULES);
	ioInfo.push_back(mesh_info, IOI_MESH_FORMESHLIST);
	ioInfo.push_back(mesh_info, IOI_MESH_FORDISPLAYOPTIONS);
	ioInfo.push_back(mesh_info, IOI_MESH_FORTEMPERATURE);
	ioInfo.push_back(mesh_info, IOI_MESH_FORHEATBOUNDARIES);
	ioInfo.push_back(mesh_info, IOI_MESH_FORCURIEANDMOMENT);
	ioInfo.push_back(mesh_info, IOI_MESH_FORTMODEL);
	ioInfo.push_back(mesh_info, IOI_MESH_FORPBC);
	ioInfo.push_back(mesh_info, IOI_MESH_FOREXCHCOUPLING);
	ioInfo.push_back(mesh_info, IOI_MESH_FORSTOCHASTICITY);
	ioInfo.push_back(mesh_info, IOI_MESH_FORELASTICITY);
	ioInfo.push_back(mesh_info, IOI_MESH_FORSPEEDUP);
	ioInfo.push_back(mesh_info, IOI_MESH_FORSKYPOSDMUL);
	ioInfo.push_back(mesh_info, IOI_MESH_FORMC);
	ioInfo.push_back(mesh_info, IOI_MESH_FORDIPOLESHIFT);
	ioInfo.push_back(mesh_info, IOI_MESH_FORTMR);

	//Shows ferromagnetic super-mesh rectangle (unit m) : textId is the mesh rectangle for the ferromagnetic super-mesh
	//IOI_FMSMESHRECTANGLE

	std::string fsmeshrect_info =
		std::string("[tc1,1,0,1/tc]<b>Ferromagnetic supermesh rectangle</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Smallest rectangle containing</i>\n[tc1,1,0,1/tc]<i>all ferromagnetic meshes</i>");

	ioInfo.push_back(fsmeshrect_info, IOI_FMSMESHRECTANGLE);

	//Shows electric super-mesh rectangle (unit m) : textId is the mesh rectangle for the electric super-mesh
	//IOI_ESMESHRECTANGLE

	std::string esmeshrect_info =
		std::string("[tc1,1,0,1/tc]<b>Electric supermesh rectangle</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Smallest rectangle containing</i>\n[tc1,1,0,1/tc]<i>all meshes with</i>\n[tc1,1,0,1/tc]<i>enabled Transport module</i>");

	ioInfo.push_back(esmeshrect_info, IOI_ESMESHRECTANGLE);

	//Shows ferromagnetic super-mesh cellsize (units m) : textId is the mesh cellsize for the ferromagnetic super-mesh
	//IOI_FMSMESHCELLSIZE

	std::string fsmeshcell_info =
		std::string("[tc1,1,0,1/tc]<b>Ferromagnetic supermesh cellsize</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Discretization cellsize</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(fsmeshcell_info, IOI_FMSMESHCELLSIZE);

	//Shows electric super-mesh cellsize (units m) : textId is the mesh cellsize for the electric super-mesh
	//IOI_ESMESHCELLSIZE

	std::string esmeshcell_info =
		std::string("[tc1,1,0,1/tc]<b>Electric supermesh cellsize</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Discretization cellsize</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(esmeshcell_info, IOI_ESMESHCELLSIZE);

	//Shows mesh rectangle (units m) : minorId is the unique mesh id number, textId is the mesh rectangle
	//IOI_MESHRECTANGLE

	std::string meshrect_info =
		std::string("[tc1,1,0,1/tc]<b>Mesh rectangle</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(meshrect_info, IOI_MESHRECTANGLE);

	//Shows magnetic mesh cellsize (units m) : minorId is the unique mesh id number, auxId is enabled/disabled status, textId is the mesh cellsize
	//IOI_MESHCELLSIZE

	std::string meshcell_info =
		std::string("[tc1,1,0,1/tc]<b>Ferromagnetic cellsize</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Discretization cellsize</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(meshcell_info, IOI_MESHCELLSIZE);

	//Shows electric mesh cellsize (units m) : minorId is the unique mesh id number, auxId is enabled/disabled status, textId is the mesh cellsize
	//IOI_MESHECELLSIZE

	std::string emeshcell_info =
		std::string("[tc1,1,0,1/tc]<b>Transport solver cellsize</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Discretization cellsize</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(emeshcell_info, IOI_MESHECELLSIZE);

	//Shows thermal mesh cellsize (units m) : minorId is the unique mesh id number, auxId is enabled/disabled status, textId is the mesh cellsize
	//IOI_MESHTCELLSIZE

	std::string tmeshcell_info =
		std::string("[tc1,1,0,1/tc]<b>Heat solver cellsize</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Discretization cellsize</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(tmeshcell_info, IOI_MESHTCELLSIZE);

	//Shows mesh cellsize (units m) : minorId is the unique mesh id number, auxId is enabled/disabled status, textId is the mesh cellsize
	//IOI_MESHMCELLSIZE:

	std::string mechmeshcell_info =
		std::string("[tc1,1,0,1/tc]<b>Mechanical solver cellsize</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Discretization cellsize</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(mechmeshcell_info, IOI_MESHMCELLSIZE);

	//Shows stochastic cellsize (units m) : minorId is the unique mesh id number, auxId is enabled/disabled status, textId is the mesh cellsize
	//IOI_MESHSCELLSIZE:

	std::string stochmeshcell_info =
		std::string("[tc1,1,0,1/tc]<b>Stochasticity cellsize</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Stochastic fields discretization</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(stochmeshcell_info, IOI_MESHSCELLSIZE);

	//Shows link stochastic flag : minorId is the unique mesh id number, auxId is the value off (0), on (1), N/A (-1)
	//IOI_LINKSTOCHASTIC

	std::string linkstoch_info =
		std::string("[tc1,1,0,1/tc]<b>Stochasticity cellsize flag</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Link to magnetic cellsize</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change");

	ioInfo.push_back(linkstoch_info, IOI_LINKSTOCHASTIC);

	//Shows macrocell size (units m) for atomistic meshes: minorId is the unique mesh id number, auxId is enabled/disabled status, textId is the mesh cellsize
	//IOI_MESHDMCELLSIZE

	std::string dmmeshcell_info =
		std::string("[tc1,1,0,1/tc]<b>Atomistic macrocell size</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Used for demagnetising/dipole-dipole field</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(dmmeshcell_info, IOI_MESHDMCELLSIZE);

	//Shows evaluation speedup type: auxId is the type value.
	//IOI_SPEEDUPMODE

	std::string speedup_info =
		std::string("[tc1,1,0,1/tc]<b>Evaluation speedup settings</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>Used to reduce number of demag field evaluations</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(speedup_info, IOI_SPEEDUPMODE);

	//Simulation output data, specifically used for showing values in console : minorId is the DATA_ id, textId is the data handle
	//IOI_SHOWDATA
	//Simulation output data, specifically used to construct output data list : minorId is the DATA_ id, textId is the data handle
	//IOI_DATA

	std::string showdata_info_generic =
		std::string("[tc1,1,0,1/tc]<b>Output data</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: show value") +
		std::string("\n[tc1,1,0,1/tc]right-click: pin to data box") +
		std::string("\n[tc1,1,0,1/tc]drag: pin to data box\n");

	ioInfo.set(showdata_info_generic + std::string("<i><b>Simulation stage and step</i>"), INT2(IOI_SHOWDATA, DATA_STAGESTEP));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Simulation total time</i>"), INT2(IOI_SHOWDATA, DATA_TIME));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Simulation stage time</i>"), INT2(IOI_SHOWDATA, DATA_STAGETIME));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Simulation iterations</i>"), INT2(IOI_SHOWDATA, DATA_ITERATIONS));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Simulation stage iterations</i>"), INT2(IOI_SHOWDATA, DATA_SITERATIONS));
	ioInfo.set(showdata_info_generic + std::string("<i><b>magnetization solver time step</i>"), INT2(IOI_SHOWDATA, DATA_DT));
	ioInfo.set(showdata_info_generic + std::string("<i><b>magnetization relaxation |mxh|</i>"), INT2(IOI_SHOWDATA, DATA_MXH));
	ioInfo.set(showdata_info_generic + std::string("<i><b>magnetization relaxation |dm/dt|</i>"), INT2(IOI_SHOWDATA, DATA_DMDT));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average magnetization</i>"), INT2(IOI_SHOWDATA, DATA_AVM));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average magnetization sub-lattice B</i>"), INT2(IOI_SHOWDATA, DATA_AVM2));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Thermodynamic average magnetization</i>"), INT2(IOI_SHOWDATA, DATA_THAVM));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Magnetization magnitude min-max</i>"), INT2(IOI_SHOWDATA, DATA_M_MINMAX));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Magnetization component x min-max</i>"), INT2(IOI_SHOWDATA, DATA_MX_MINMAX));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Magnetization component y min-max</i>"), INT2(IOI_SHOWDATA, DATA_MY_MINMAX));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Magnetization component z min-max</i>"), INT2(IOI_SHOWDATA, DATA_MZ_MINMAX));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Monte-Carlo cone angle (deg.) and target acceptance.</i>"), INT2(IOI_SHOWDATA, DATA_MONTECARLOPARAMS));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Applied magnetic field</i>"), INT2(IOI_SHOWDATA, DATA_HA));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average charge current density</i>"), INT2(IOI_SHOWDATA, DATA_JC));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average spin x-current density</i>"), INT2(IOI_SHOWDATA, DATA_JSX));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average spin y-current density</i>"), INT2(IOI_SHOWDATA, DATA_JSY));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average spin z-current density</i>"), INT2(IOI_SHOWDATA, DATA_JSZ));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average m x dm/dt</i>"), INT2(IOI_SHOWDATA, DATA_RESPUMP));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average dm/dt</i>"), INT2(IOI_SHOWDATA, DATA_IMSPUMP));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average m x dm/dt sub-lattice 2</i>"), INT2(IOI_SHOWDATA, DATA_RESPUMP2));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average dm/dt sub-lattice 2</i>"), INT2(IOI_SHOWDATA, DATA_IMSPUMP2));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average mA x dmB/dt for 2 sub-lattice</i>"), INT2(IOI_SHOWDATA, DATA_RESPUMP12));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average mB x dmA/dt for 2 sub-lattice</i>"), INT2(IOI_SHOWDATA, DATA_RESPUMP21));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average charge potential</i>"), INT2(IOI_SHOWDATA, DATA_V));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average spin accumulation</i>"), INT2(IOI_SHOWDATA, DATA_S));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average electrical conductivity</i>"), INT2(IOI_SHOWDATA, DATA_ELC));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Potential drop between electrodes</i>"), INT2(IOI_SHOWDATA, DATA_POTENTIAL));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Charge current into\n<i><b>ground electrode and\n<i>net current (error term)</i>"), INT2(IOI_SHOWDATA, DATA_CURRENT));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Total resistance (V/I)</i>"), INT2(IOI_SHOWDATA, DATA_RESISTANCE));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average mechanical displacement</i>"), INT2(IOI_SHOWDATA, DATA_AVU));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average diagonal strain (xx, yy, zz)</i>"), INT2(IOI_SHOWDATA, DATA_AVSTRAINDIAG));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average off-diagonal strain (yz, xz, xy)</i>"), INT2(IOI_SHOWDATA, DATA_AVSTRAINODIAG));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Energy density: demag</i>"), INT2(IOI_SHOWDATA, DATA_E_DEMAG));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Energy density: exchange</i>"), INT2(IOI_SHOWDATA, DATA_E_EXCH));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Torque: exchange</i>"), INT2(IOI_SHOWDATA, DATA_T_EXCH));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Energy density: surface exchange</i>"), INT2(IOI_SHOWDATA, DATA_E_SURFEXCH));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Torque: surface exchange</i>"), INT2(IOI_SHOWDATA, DATA_T_SURFEXCH));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Energy density: applied H field</i>"), INT2(IOI_SHOWDATA, DATA_E_ZEE));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Torque: applied H field</i>"), INT2(IOI_SHOWDATA, DATA_T_ZEE));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Energy density: dipole strayfield</i>"), INT2(IOI_SHOWDATA, DATA_E_STRAY));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Torque: dipole strayfield</i>"), INT2(IOI_SHOWDATA, DATA_T_STRAY));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Energy density: magneto-optical field</i>"), INT2(IOI_SHOWDATA, DATA_E_MOPTICAL));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Energy density: applied mechanical stress</i>"), INT2(IOI_SHOWDATA, DATA_E_MELASTIC));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Energy density: anisotropy</i>"), INT2(IOI_SHOWDATA, DATA_E_ANIS));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Torque: anisotropy</i>"), INT2(IOI_SHOWDATA, DATA_T_ANIS));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Torque: anisotropy</i>"), INT2(IOI_SHOWDATA, DATA_T_ANIS));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Energy density: roughness</i>"), INT2(IOI_SHOWDATA, DATA_E_ROUGH));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Energy density: Total</i>"), INT2(IOI_SHOWDATA, DATA_E_TOTAL));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Domain wall shift\n<i><b>for moving mesh</i>"), INT2(IOI_SHOWDATA, DATA_DWSHIFT));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Domain wall position and width\n<i><b>Fit along x through rectangle centre</i>"), INT2(IOI_SHOWDATA, DATA_DWPOS_X));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Domain wall position and width\n<i><b>Fit along y through rectangle centre</i>"), INT2(IOI_SHOWDATA, DATA_DWPOS_Y));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Domain wall position and width\n<i><b>Fit along z through rectangle centre</i>"), INT2(IOI_SHOWDATA, DATA_DWPOS_Z));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Skyrmion shift in the xy plane\n<i><b>Only use with output save data</i>"), INT2(IOI_SHOWDATA, DATA_SKYSHIFT));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Skyrmion shift in the xy plane\n<i><b>Additional saving of x and y axis diameters\n<i><b>Only use with output save data</i>"), INT2(IOI_SHOWDATA, DATA_SKYPOS));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Topological Charge"), INT2(IOI_SHOWDATA, DATA_Q_TOPO));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Transport solver:\n<i><b>V iterations to convergence</i>"), INT2(IOI_SHOWDATA, DATA_TRANSPORT_ITERSTOCONV));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Transport solver:\n<i><b>S iterations to convergence</i>"), INT2(IOI_SHOWDATA, DATA_TRANSPORT_SITERSTOCONV));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Transport solver:\n<i><b>achieved convergence error</i>"), INT2(IOI_SHOWDATA, DATA_TRANSPORT_CONVERROR));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Tunneling magnetoresistance (Ohms)</i>"), INT2(IOI_SHOWDATA, DATA_TMR));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average temperature</i>"), INT2(IOI_SHOWDATA, DATA_TEMP));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Average lattice temperature</i>"), INT2(IOI_SHOWDATA, DATA_TEMP_L));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Heat solver time step</i>"), INT2(IOI_SHOWDATA, DATA_HEATDT));
	ioInfo.set(showdata_info_generic + std::string("<i><b>Command buffer data extraction</i>"), INT2(IOI_SHOWDATA, DATA_COMMBUFFER));

	std::string data_info_generic =
		std::string("[tc1,1,0,1/tc]<b>Output data</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: add to output list") +
		std::string("\n[tc1,1,0,1/tc]right-click: pin to data box") +
		std::string("\n[tc1,1,0,1/tc]drag: pin to data box\n");

	ioInfo.set(data_info_generic + std::string("<i><b>Simulation stage and step</i>"), INT2(IOI_DATA, DATA_STAGESTEP));
	ioInfo.set(data_info_generic + std::string("<i><b>Simulation total time</i>"), INT2(IOI_DATA, DATA_TIME));
	ioInfo.set(data_info_generic + std::string("<i><b>Simulation stage time</i>"), INT2(IOI_DATA, DATA_STAGETIME));
	ioInfo.set(data_info_generic + std::string("<i><b>Simulation iterations</i>"), INT2(IOI_DATA, DATA_ITERATIONS));
	ioInfo.set(data_info_generic + std::string("<i><b>Simulation stage iterations</i>"), INT2(IOI_DATA, DATA_SITERATIONS));
	ioInfo.set(data_info_generic + std::string("<i><b>magnetization solver time step</i>"), INT2(IOI_DATA, DATA_DT));
	ioInfo.set(data_info_generic + std::string("<i><b>magnetization relaxation |mxh|</i>"), INT2(IOI_DATA, DATA_MXH));
	ioInfo.set(data_info_generic + std::string("<i><b>magnetization relaxation |dm/dt|</i>"), INT2(IOI_DATA, DATA_DMDT));
	ioInfo.set(data_info_generic + std::string("<i><b>Average magnetization</i>"), INT2(IOI_DATA, DATA_AVM));
	ioInfo.set(data_info_generic + std::string("<i><b>Average magnetization sub-lattice B</i>"), INT2(IOI_DATA, DATA_AVM2));
	ioInfo.set(data_info_generic + std::string("<i><b>Thermodynamic average magnetization</i>"), INT2(IOI_DATA, DATA_THAVM));
	ioInfo.set(data_info_generic + std::string("<i><b>Magnetization magnitude min-max</i>"), INT2(IOI_DATA, DATA_M_MINMAX));
	ioInfo.set(data_info_generic + std::string("<i><b>Magnetization component x min-max</i>"), INT2(IOI_DATA, DATA_MX_MINMAX));
	ioInfo.set(data_info_generic + std::string("<i><b>Magnetization component y min-max</i>"), INT2(IOI_DATA, DATA_MY_MINMAX));
	ioInfo.set(data_info_generic + std::string("<i><b>Magnetization component z min-max</i>"), INT2(IOI_DATA, DATA_MZ_MINMAX));
	ioInfo.set(data_info_generic + std::string("<i><b>Monte-Carlo cone angle (deg.) and target acceptance.</i>"), INT2(IOI_DATA, DATA_MONTECARLOPARAMS));
	ioInfo.set(data_info_generic + std::string("<i><b>Applied magnetic field</i>"), INT2(IOI_DATA, DATA_HA));
	ioInfo.set(data_info_generic + std::string("<i><b>Average charge current density</i>"), INT2(IOI_DATA, DATA_JC));
	ioInfo.set(data_info_generic + std::string("<i><b>Average spin x-current density</i>"), INT2(IOI_DATA, DATA_JSX));
	ioInfo.set(data_info_generic + std::string("<i><b>Average spin y-current density</i>"), INT2(IOI_DATA, DATA_JSY));
	ioInfo.set(data_info_generic + std::string("<i><b>Average spin z-current density</i>"), INT2(IOI_DATA, DATA_JSZ));
	ioInfo.set(data_info_generic + std::string("<i><b>Average m x dm/dt</i>"), INT2(IOI_DATA, DATA_RESPUMP));
	ioInfo.set(data_info_generic + std::string("<i><b>Average dm/dt</i>"), INT2(IOI_DATA, DATA_IMSPUMP));
	ioInfo.set(data_info_generic + std::string("<i><b>Average m x dm/dt sub-lattice 2</i>"), INT2(IOI_DATA, DATA_RESPUMP2));
	ioInfo.set(data_info_generic + std::string("<i><b>Average dm/dt sub-lattice 2</i>"), INT2(IOI_DATA, DATA_IMSPUMP2));
	ioInfo.set(data_info_generic + std::string("<i><b>Average mA x dmB/dt for 2 sub-lattice 2</i>"), INT2(IOI_DATA, DATA_RESPUMP12));
	ioInfo.set(data_info_generic + std::string("<i><b>Average mB x dmA/dt for 2 sub-lattice 2</i>"), INT2(IOI_DATA, DATA_RESPUMP21));
	ioInfo.set(data_info_generic + std::string("<i><b>Average charge potential</i>"), INT2(IOI_DATA, DATA_V));
	ioInfo.set(data_info_generic + std::string("<i><b>Average spin accumulation</i>"), INT2(IOI_DATA, DATA_S));
	ioInfo.set(data_info_generic + std::string("<i><b>Average electrical conductivity</i>"), INT2(IOI_DATA, DATA_ELC));
	ioInfo.set(data_info_generic + std::string("<i><b>Potential drop between electrodes</i>"), INT2(IOI_DATA, DATA_POTENTIAL));
	ioInfo.set(data_info_generic + std::string("<i><b>Charge current into\n<i><b>ground electrode and\n<i><b>net current (error term)</i>"), INT2(IOI_DATA, DATA_CURRENT));
	ioInfo.set(data_info_generic + std::string("<i><b>Total resistance (V/I)</i>"), INT2(IOI_DATA, DATA_RESISTANCE));
	ioInfo.set(data_info_generic + std::string("<i><b>Average mechanical displacement</i>"), INT2(IOI_DATA, DATA_AVU));
	ioInfo.set(data_info_generic + std::string("<i><b>Average diagonal strain (xx, yy, zz)</i>"), INT2(IOI_DATA, DATA_AVSTRAINDIAG));
	ioInfo.set(data_info_generic + std::string("<i><b>Average off-diagonal strain (yz, xz, xy)</i>"), INT2(IOI_DATA, DATA_AVSTRAINODIAG));
	ioInfo.set(data_info_generic + std::string("<i><b>Energy density: demag</i>"), INT2(IOI_DATA, DATA_E_DEMAG));
	ioInfo.set(data_info_generic + std::string("<i><b>Energy density: exchange</i>"), INT2(IOI_DATA, DATA_E_EXCH));
	ioInfo.set(data_info_generic + std::string("<i><b>Torque: exchange</i>"), INT2(IOI_DATA, DATA_T_EXCH));
	ioInfo.set(data_info_generic + std::string("<i><b>Energy density: surface exchange</i>"), INT2(IOI_DATA, DATA_E_SURFEXCH));
	ioInfo.set(data_info_generic + std::string("<i><b>Torque: surface exchange</i>"), INT2(IOI_DATA, DATA_T_SURFEXCH));
	ioInfo.set(data_info_generic + std::string("<i><b>Energy density: applied H field</i>"), INT2(IOI_DATA, DATA_E_ZEE));
	ioInfo.set(data_info_generic + std::string("<i><b>Torque: applied H field</i>"), INT2(IOI_DATA, DATA_T_ZEE));
	ioInfo.set(data_info_generic + std::string("<i><b>Energy density: dipole strayfield</i>"), INT2(IOI_DATA, DATA_E_STRAY));
	ioInfo.set(data_info_generic + std::string("<i><b>Torque: dipole strayfield</i>"), INT2(IOI_DATA, DATA_T_STRAY));
	ioInfo.set(data_info_generic + std::string("<i><b>Energy density: magneto-optical field</i>"), INT2(IOI_DATA, DATA_E_MOPTICAL));
	ioInfo.set(data_info_generic + std::string("<i><b>Energy density: applied mechanical stress</i>"), INT2(IOI_DATA, DATA_E_MELASTIC));
	ioInfo.set(data_info_generic + std::string("<i><b>Energy density: anisotropy</i>"), INT2(IOI_DATA, DATA_E_ANIS));
	ioInfo.set(data_info_generic + std::string("<i><b>Torque: anisotropy</i>"), INT2(IOI_DATA, DATA_T_ANIS));
	ioInfo.set(data_info_generic + std::string("<i><b>Energy density: roughness</i>"), INT2(IOI_DATA, DATA_E_ROUGH));
	ioInfo.set(data_info_generic + std::string("<i><b>Energy density: Total</i>"), INT2(IOI_DATA, DATA_E_TOTAL));
	ioInfo.set(data_info_generic + std::string("<i><b>Domain wall shift\n<i><b>for moving mesh</i>"), INT2(IOI_DATA, DATA_DWSHIFT));
	ioInfo.set(data_info_generic + std::string("<i><b>Domain wall position and width\n<i><b>Fit along x through rectangle centre</i>"), INT2(IOI_DATA, DATA_DWPOS_X));
	ioInfo.set(data_info_generic + std::string("<i><b>Domain wall position and width\n<i><b>Fit along y through rectangle centre</i>"), INT2(IOI_DATA, DATA_DWPOS_Y));
	ioInfo.set(data_info_generic + std::string("<i><b>Domain wall position and width\n<i><b>Fit along z through rectangle centre</i>"), INT2(IOI_DATA, DATA_DWPOS_Z));
	ioInfo.set(data_info_generic + std::string("<i><b>Skyrmion shift in the xy plane\n<i><b>Rectangle must circumscribe skyrmion</i>"), INT2(IOI_DATA, DATA_SKYSHIFT));
	ioInfo.set(data_info_generic + std::string("<i><b>Skyrmion shift in the xy plane\n<i><b>Also save x and y axis diameters\n<i><b>Rectangle must circumscribe skyrmion</i>"), INT2(IOI_DATA, DATA_SKYPOS));
	ioInfo.set(data_info_generic + std::string("<i><b>Topological Charge</i>"), INT2(IOI_DATA, DATA_Q_TOPO));
	ioInfo.set(data_info_generic + std::string("<i><b>Transport solver:\n<i><b>V iterations to convergence</i>"), INT2(IOI_DATA, DATA_TRANSPORT_ITERSTOCONV));
	ioInfo.set(data_info_generic + std::string("<i><b>Transport solver:\n<i><b>S iterations to convergence</i>"), INT2(IOI_DATA, DATA_TRANSPORT_SITERSTOCONV));
	ioInfo.set(data_info_generic + std::string("<i><b>Transport solver:\n<i><b>achieved convergence error</i>"), INT2(IOI_DATA, DATA_TRANSPORT_CONVERROR));
	ioInfo.set(data_info_generic + std::string("<i><b>Tunneling magnetoresistance (Ohms)</i>"), INT2(IOI_DATA, DATA_TMR));
	ioInfo.set(data_info_generic + std::string("<i><b>Average temperature</i>"), INT2(IOI_DATA, DATA_TEMP));
	ioInfo.set(data_info_generic + std::string("<i><b>Average lattice temperature</i>"), INT2(IOI_DATA, DATA_TEMP_L));
	ioInfo.set(data_info_generic + std::string("<i><b>Heat solver time step</i>"), INT2(IOI_DATA, DATA_HEATDT));
	ioInfo.set(data_info_generic + std::string("<i><b>Command buffer data extraction</i>"), INT2(IOI_DATA, DATA_COMMBUFFER));

	//Show currently set directory : textId is the directory
	//IOI_DIRECTORY

	std::string dir_info =
		std::string("[tc1,1,0,1/tc]<b>Working directory</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(dir_info, IOI_DIRECTORY);

	//Show currently set save data file : textId is the file name
	//IOI_SAVEDATAFILE

	std::string savedatafile_info =
		std::string("[tc1,1,0,1/tc]<b>Output data file (.txt)</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(savedatafile_info, IOI_SAVEDATAFILE);

	//Show currently set image filename base : textId is the file name
	//IOI_SAVEIMAGEFILEBASE

	std::string saveimagefile_info =
		std::string("[tc1,1,0,1/tc]<b>Image save file base</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(saveimagefile_info, IOI_SAVEIMAGEFILEBASE);

	//Show flag status for data/image saving during a simulation : minorId is the flag value (boolean)
	//IOI_SAVEDATAFLAG

	std::string savedataflag_info =
		std::string("[tc1,1,0,1/tc]<b>Data saving switch</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: on, red: off</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status");

	ioInfo.push_back(savedataflag_info, IOI_SAVEDATAFLAG);

	//IOI_SAVEIMAGEFLAG

	std::string saveimageflag_info =
		std::string("[tc1,1,0,1/tc]<b>Image saving switch</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: on, red: off</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status");

	ioInfo.push_back(saveimageflag_info, IOI_SAVEIMAGEFLAG);

	//Show set output data : minorId is the minor id of elements in Simulation::saveDataList (major id there is always 0), auxId is the number of the interactive object in the list as it appears in the console, textId is the configured output data. 
	//Note this entry must always represent the entry in Simulation::saveDataList with the index in auxId.
	//IOI_OUTDATA

	std::string showoutdata_info =
		std::string("[tc1,1,0,1/tc]<b>Set output data entry</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit entry") +
		std::string("\n[tc1,1,0,1/tc]right-click: delete entry") +
		std::string("\n[tc1,1,0,1/tc]drag: move to change order\n");

	ioInfo.push_back(showoutdata_info, IOI_OUTDATA);

	//Shows a possible stage type, used for adding generic stages to the simulation schedule : minorId is the stage type (SS_ enum value, which is the majorId from stageDescriptors), textId is the stage setting handle
	//IOI_STAGE

	std::string stage_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Simulation schedule stage type</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: add to schedule\n");

	ioInfo.set(stage_generic_info + std::string("<i><b>Nothing to set, just run solvers"), INT2(IOI_STAGE, SS_RELAX));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set H field in Cartesian coordinates"), INT2(IOI_STAGE, SS_HFIELDXYZ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set H field sequence in Cartesian coordinates\n<i><b>Start to stop H values in a number of steps"), INT2(IOI_STAGE, SS_HFIELDXYZSEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set H field sequence in Polar coordinates\n<i><b>Start to stop H values in a number of steps"), INT2(IOI_STAGE, SS_HPOLARSEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set FMR H field sequence in Cartesian coordinates\n<i><b>Bias field, rf field, rf field steps, rf field cycles"), INT2(IOI_STAGE, SS_HFMR));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set field using custom equation"), INT2(IOI_STAGE, SS_HFIELDEQUATION));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set field sequence using custom equation"), INT2(IOI_STAGE, SS_HFIELDEQUATIONSEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set from file in current directory\n<i><b>File must have tab-spaced columns as:\n<i><b>time and value all in S.I. units.\n<i><b>Time resolution set by stage time condition."), INT2(IOI_STAGE, SS_HFIELDFILE));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set fixed voltage drop between electrodes"), INT2(IOI_STAGE, SS_V));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set fixed voltage sequence\n<i><b>Start to stop V values in a number of steps"), INT2(IOI_STAGE, SS_VSEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set a sinusoidal voltage sequence\n<i><b>Amplitude, steps per cycle, cycles"), INT2(IOI_STAGE, SS_VSIN));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set a cosine voltage sequence\n<i><b>Amplitude, steps per cycle, cycles"), INT2(IOI_STAGE, SS_VCOS));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set potential using custom equation"), INT2(IOI_STAGE, SS_VEQUATION));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set potential sequence using custom equation"), INT2(IOI_STAGE, SS_VEQUATIONSEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set from file in current directory\n<i><b>File must have tab-spaced columns as:\n<i><b>time and value all in S.I. units.\n<i><b>Time resolution set by stage time condition."), INT2(IOI_STAGE, SS_VFILE));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set fixed current into ground electrode"), INT2(IOI_STAGE, SS_I));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set fixed current sequence\n<i><b>Start to stop I values in a number of steps"), INT2(IOI_STAGE, SS_ISEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set a sinusoidal current sequence\n<i><b>Amplitude, steps per cycle, cycles"), INT2(IOI_STAGE, SS_ISIN));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set a cosine current sequence\n<i><b>Amplitude, steps per cycle, cycles"), INT2(IOI_STAGE, SS_ICOS));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set current using custom equation"), INT2(IOI_STAGE, SS_IEQUATION));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set current sequence using custom equation"), INT2(IOI_STAGE, SS_IEQUATIONSEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set from file in current directory\n<i><b>File must have tab-spaced columns as:\n<i><b>time and value all in S.I. units.\n<i><b>Time resolution set by stage time condition."), INT2(IOI_STAGE, SS_IFILE));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set base temperature value"), INT2(IOI_STAGE, SS_T));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set base temperature sequence\n<i><b>Start to stop T values in a number of steps"), INT2(IOI_STAGE, SS_TSEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set base temperature using custom equation"), INT2(IOI_STAGE, SS_TEQUATION));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set base temperature sequence using custom equation"), INT2(IOI_STAGE, SS_TEQUATIONSEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set from file in current directory\n<i><b>File must have tab-spaced columns as:\n<i><b>time and value all in S.I. units.\n<i><b>Time resolution set by stage time condition."), INT2(IOI_STAGE, SS_TFILE));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set heat source value"), INT2(IOI_STAGE, SS_Q));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set heat source sequence\n<i><b>Start to stop Q values in a number of steps"), INT2(IOI_STAGE, SS_QSEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set heat source using custom equation"), INT2(IOI_STAGE, SS_QEQUATION));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set heat source sequence using custom equation"), INT2(IOI_STAGE, SS_QEQUATIONSEQ));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set from file in current directory\n<i><b>File must have tab-spaced columns as:\n<i><b>time and value all in S.I. units.\n<i><b>Time resolution set by stage time condition."), INT2(IOI_STAGE, SS_QFILE));
	ioInfo.set(stage_generic_info + std::string("<i><b>Set uniform stress in polar coordinates"), INT2(IOI_STAGE, SS_TSIGPOLAR));
	ioInfo.set(stage_generic_info + std::string("<i><b>Use Monte-Carlo algorithm in atomistic meshes."), INT2(IOI_STAGE, SS_MONTECARLO));

	//Shows a stage added to the simulation schedule : minorId is the minor id of elements in Simulation::simStages (major id there is always 0), auxId is the number of the interactive object in the list, textId is the configured stage text
	//Note this entry must always represent the entry in Simulation::simStages with the index in auxId.
	//IOI_SETSTAGE

	std::string setstage_info =
		std::string("[tc1,1,0,1/tc]<b>Set simulation schedule stage</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: delete\n") +
		std::string("\n[tc1,1,0,1/tc]drag: move to change order\n");

	ioInfo.push_back(setstage_info, IOI_SETSTAGE);

	//Shows the value to set for the simulation schedule stage : minorId is the minor id of elements in Simulation::simStages (major id there is always 0), auxId is the number of the interactive object in the list, textId is the value as a std::string
	//IOI_SETSTAGEVALUE

	std::string stagevalue_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Simulation schedule stage value</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.set(stagevalue_generic_info + std::string(""), INT2(IOI_SETSTAGEVALUE, SS_RELAX));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Hx, Hy, Hz"), INT2(IOI_SETSTAGEVALUE, SS_HFIELDXYZ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Hstart; Hstop; Steps: Hstep = (Hstop - Hstart) / Steps"), INT2(IOI_SETSTAGEVALUE, SS_HFIELDXYZSEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Hstart; Hstop; Steps: Hstep = (Hstop - Hstart) / Steps\n<i><b>H values in polar coordinates as:\n<i><b>strength value, polar angle, azimuthal angle"), INT2(IOI_SETSTAGEVALUE, SS_HPOLARSEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Hbias; Hrf; rf steps; rf cycles\n<i><b>rf steps is the rf cycle discretization\n<i><b>rf cycles is the number of periods"), INT2(IOI_SETSTAGEVALUE, SS_HFMR));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Text vector equation"), INT2(IOI_SETSTAGEVALUE, SS_HFIELDEQUATION));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Steps: Text vector equation"), INT2(IOI_SETSTAGEVALUE, SS_HFIELDEQUATIONSEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Set values from file."), INT2(IOI_SETSTAGEVALUE, SS_HFIELDFILE));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Fixed voltage drop between electrodes"), INT2(IOI_SETSTAGEVALUE, SS_V));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Vstart; Vstop; Steps: Vstep = (Vstop - Vstart) / Steps"), INT2(IOI_SETSTAGEVALUE, SS_VSEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Amplitude, steps per cycle, cycles"), INT2(IOI_SETSTAGEVALUE, SS_VSIN));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Amplitude, steps per cycle, cycles"), INT2(IOI_SETSTAGEVALUE, SS_VCOS));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Text equation"), INT2(IOI_SETSTAGEVALUE, SS_VEQUATION));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Steps: Text equation"), INT2(IOI_SETSTAGEVALUE, SS_VEQUATIONSEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Set values from file."), INT2(IOI_SETSTAGEVALUE, SS_VFILE));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Fixed current into ground electrode"), INT2(IOI_SETSTAGEVALUE, SS_I));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Istart; Istop; Steps: Istep = (Istop - Istart) / Steps"), INT2(IOI_SETSTAGEVALUE, SS_ISEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Amplitude, steps per cycle, cycles"), INT2(IOI_SETSTAGEVALUE, SS_ISIN));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Amplitude, steps per cycle, cycles"), INT2(IOI_SETSTAGEVALUE, SS_ICOS));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Text equation"), INT2(IOI_SETSTAGEVALUE, SS_IEQUATION));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Steps: Text equation"), INT2(IOI_SETSTAGEVALUE, SS_IEQUATIONSEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Set values from file."), INT2(IOI_SETSTAGEVALUE, SS_IFILE));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Base temperature value"), INT2(IOI_SETSTAGEVALUE, SS_T));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Tstart; Tstop; Steps: Tstep = (Tstop - Tstart) / Steps"), INT2(IOI_SETSTAGEVALUE, SS_TSEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Text equation"), INT2(IOI_SETSTAGEVALUE, SS_TEQUATION));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Steps: Text equation"), INT2(IOI_SETSTAGEVALUE, SS_TEQUATIONSEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Set values from file."), INT2(IOI_SETSTAGEVALUE, SS_TFILE));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Heat source value"), INT2(IOI_SETSTAGEVALUE, SS_Q));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Qstart; Qstop; Steps: Qstep = (Qstop - Qstart) / Steps"), INT2(IOI_SETSTAGEVALUE, SS_QSEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Text equation"), INT2(IOI_SETSTAGEVALUE, SS_QEQUATION));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Steps: Text equation"), INT2(IOI_SETSTAGEVALUE, SS_QEQUATIONSEQ));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>Set values from file."), INT2(IOI_SETSTAGEVALUE, SS_QFILE));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>magnitude, theta, polar"), INT2(IOI_SETSTAGEVALUE, SS_TSIGPOLAR));
	ioInfo.set(stagevalue_generic_info + std::string("<i><b>acceptance_rate"), INT2(IOI_SETSTAGEVALUE, SS_MONTECARLO));

	//Shows the stop condition for the simulation schedule stage : minorId is the minor id of elements in Simulation::simStages (major id there is always 0), auxId is the number of the interactive object in the list, textId is the stop type and value as a std::string
	//IOI_STAGESTOPCONDITION

	std::string stagestop_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Stage stop condition</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.set(stagestop_generic_info + std::string("<i><b>No stopping condition set"), INT2(IOI_STAGESTOPCONDITION, STOP_NOSTOP));
	ioInfo.set(stagestop_generic_info + std::string("<i><b>Stop when stage iterations value reached"), INT2(IOI_STAGESTOPCONDITION, STOP_ITERATIONS));
	ioInfo.set(stagestop_generic_info + std::string("<i><b>Stop when |mxh| falls below value"), INT2(IOI_STAGESTOPCONDITION, STOP_MXH));
	ioInfo.set(stagestop_generic_info + std::string("<i><b>Stop when |dm/dt| falls below value"), INT2(IOI_STAGESTOPCONDITION, STOP_DMDT));
	ioInfo.set(stagestop_generic_info + std::string("<i><b>Stop when stage time value reached"), INT2(IOI_STAGESTOPCONDITION, STOP_TIME));
	ioInfo.set(stagestop_generic_info + std::string("<i><b>Stop when |mxh| falls below value or when stage iterations value reached"), INT2(IOI_STAGESTOPCONDITION, STOP_MXH_ITER));
	ioInfo.set(stagestop_generic_info + std::string("<i><b>Stop when |dm/dt| falls below value or when stage iterations value reached"), INT2(IOI_STAGESTOPCONDITION, STOP_DMDT_ITER));

	//Shows the saving condition for the simulation schedule stage : minorId is the minor id of elements in Simulation::simStages (major id there is always 0), auxId is the DSAVE_ value for this data save type, textId is the save type and value as a std::string
	//IOI_DSAVETYPE

	std::string stagesave_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Data save condition</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: on, red: off</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.set(stagesave_generic_info + std::string("<i><b>No saving set"), INT2(IOI_DSAVETYPE, DSAVE_NONE));
	ioInfo.set(stagesave_generic_info + std::string("<i><b>Save at the end of the stage"), INT2(IOI_DSAVETYPE, DSAVE_STAGE));
	ioInfo.set(stagesave_generic_info + std::string("<i><b>Save at the end of every step\n<i><b>in this stage"), INT2(IOI_DSAVETYPE, DSAVE_STEP));
	ioInfo.set(stagesave_generic_info + std::string("<i><b>Save after every given iterations"), INT2(IOI_DSAVETYPE, DSAVE_ITER));
	ioInfo.set(stagesave_generic_info + std::string("<i><b>Save after every given time interval"), INT2(IOI_DSAVETYPE, DSAVE_TIME));

	//Shows a stop condition, used to apply the same condition to all simulation stages : minorId is the STOP_ value, textId is the stop type handle
	//IOI_STAGESTOPCONDITIONALL

	std::string stagestopall_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Stage stop condition\n[tc1,1,0,1/tc]<b>for all stages</b>") +
		std::string("\n[tc1,1,0,1/tc]click: set\n");

	ioInfo.set(stagestopall_generic_info + std::string("<i><b>No stopping condition set"), INT2(IOI_STAGESTOPCONDITIONALL, STOP_NOSTOP));
	ioInfo.set(stagestopall_generic_info + std::string("<i><b>Stop when stage iterations value reached"), INT2(IOI_STAGESTOPCONDITIONALL, STOP_ITERATIONS));
	ioInfo.set(stagestopall_generic_info + std::string("<i><b>Stop when |mxh| falls below value"), INT2(IOI_STAGESTOPCONDITIONALL, STOP_MXH));
	ioInfo.set(stagestopall_generic_info + std::string("<i><b>Stop when |dm/dt| falls below value"), INT2(IOI_STAGESTOPCONDITIONALL, STOP_DMDT));
	ioInfo.set(stagestopall_generic_info + std::string("<i><b>Stop when stage time value reached"), INT2(IOI_STAGESTOPCONDITIONALL, STOP_TIME));
	ioInfo.set(stagestopall_generic_info + std::string("<i><b>Stop when |mxh| falls below value or when stage iterations value reached"), INT2(IOI_STAGESTOPCONDITIONALL, STOP_MXH_ITER));
	ioInfo.set(stagestopall_generic_info + std::string("<i><b>Stop when |dm/dt| falls below value or when stage iterations value reached"), INT2(IOI_STAGESTOPCONDITIONALL, STOP_DMDT_ITER));

	//Shows a data save condition, used to apply the same condition to all simulation stages : minorId is the DSAVE_ value, textId is the save type handle
	//IOI_DSAVETYPEALL

	std::string stagesaveall_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Data save condition\n[tc1,1,0,1/tc]<b>for all stages</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: on, red: off</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.set(stagesaveall_generic_info + std::string("<i><b>No saving set"), INT2(IOI_DSAVETYPEALL, DSAVE_NONE));
	ioInfo.set(stagesaveall_generic_info + std::string("<i><b>Save at the end of the stage"), INT2(IOI_DSAVETYPEALL, DSAVE_STAGE));
	ioInfo.set(stagesaveall_generic_info + std::string("<i><b>Save at the end of every step\n<i><b>in this stage"), INT2(IOI_DSAVETYPEALL, DSAVE_STEP));
	ioInfo.set(stagesaveall_generic_info + std::string("<i><b>Save every given iterations"), INT2(IOI_DSAVETYPEALL, DSAVE_ITER));
	ioInfo.set(stagesaveall_generic_info + std::string("<i><b>Stop every given time interval"), INT2(IOI_DSAVETYPEALL, DSAVE_TIME));

	//Shows parameter and value for a given mesh : minorId is the major id of elements in SimParams::simParams (i.e. an entry from PARAM_ enum), auxId is the unique mesh id number, textId is the parameter handle and value
	//IOI_MESHPARAM

	std::string param_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Material parameter</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.set(param_generic_info + std::string("<i><b>Electron gyromagnetic ratio relative value"), INT2(IOI_MESHPARAM, PARAM_GREL));
	ioInfo.set(param_generic_info + std::string("<i><b>Electron gyromagnetic ratio relative value"), INT2(IOI_MESHPARAM, PARAM_GREL_AFM));
	ioInfo.set(param_generic_info + std::string("<i><b>Gilbert damping"), INT2(IOI_MESHPARAM, PARAM_GDAMPING));
	ioInfo.set(param_generic_info + std::string("<i><b>Gilbert damping"), INT2(IOI_MESHPARAM, PARAM_GDAMPING_AFM));
	ioInfo.set(param_generic_info + std::string("<i><b>Saturation magnetization"), INT2(IOI_MESHPARAM, PARAM_MS));
	ioInfo.set(param_generic_info + std::string("<i><b>Saturation magnetization"), INT2(IOI_MESHPARAM, PARAM_MS_AFM));
	ioInfo.set(param_generic_info + std::string("<i><b>Demag factors Nx, Ny"), INT2(IOI_MESHPARAM, PARAM_DEMAGXY));
	ioInfo.set(param_generic_info + std::string("<i><b>Exchange stiffness"), INT2(IOI_MESHPARAM, PARAM_A));
	ioInfo.set(param_generic_info + std::string("<i><b>Exchange stiffness"), INT2(IOI_MESHPARAM, PARAM_A_AFM));
	ioInfo.set(param_generic_info + std::string("<i><b>Homogeneous AFM coupling"), INT2(IOI_MESHPARAM, PARAM_A_AFH));
	ioInfo.set(param_generic_info + std::string("<i><b>Nonhomogeneous AFM coupling"), INT2(IOI_MESHPARAM, PARAM_A_AFNH));
	ioInfo.set(param_generic_info + std::string("<i><b>Exchange parameter to critical temperature ratio\n<i><b>Used with 2-sublattice model."), INT2(IOI_MESHPARAM, PARAM_AFTAU));
	ioInfo.set(param_generic_info + std::string("<i><b>Exchange parameter to critical temperature ratio\n<i><b>Used with 2-sublattice model, cross-coupling"), INT2(IOI_MESHPARAM, PARAM_AFTAUCROSS));
	ioInfo.set(param_generic_info + std::string("<i><b>Dzyaloshinskii-Moriya exchange"), INT2(IOI_MESHPARAM, PARAM_D));
	ioInfo.set(param_generic_info + std::string("<i><b>Dzyaloshinskii-Moriya exchange"), INT2(IOI_MESHPARAM, PARAM_D_AFM));
	ioInfo.set(param_generic_info + std::string("<i><b>Homogeneous DMI constant"), INT2(IOI_MESHPARAM, PARAM_DMI_DH));
	ioInfo.set(param_generic_info + std::string("<i><b>Homogeneous DMI direction unit vector"), INT2(IOI_MESHPARAM, PARAM_DMI_DH_DIR));
	ioInfo.set(param_generic_info + std::string("<i><b>Interfacial DMI symmetry axis\n<i><b>Cartesian unit vector"), INT2(IOI_MESHPARAM, PARAM_DMI_DIR));
	ioInfo.set(param_generic_info + std::string("<i><b>Bilinear surface exchange\n<i><b>Top mesh sets value"), INT2(IOI_MESHPARAM, PARAM_J1));
	ioInfo.set(param_generic_info + std::string("<i><b>Biquadratic surface exchange\n<i><b>Top mesh sets value"), INT2(IOI_MESHPARAM, PARAM_J2));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetocrystalline anisotropy"), INT2(IOI_MESHPARAM, PARAM_K1));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetocrystalline anisotropy"), INT2(IOI_MESHPARAM, PARAM_K2));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetocrystalline anisotropy"), INT2(IOI_MESHPARAM, PARAM_K3));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetocrystalline anisotropy 2-sublattice"), INT2(IOI_MESHPARAM, PARAM_K1_AFM));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetocrystalline anisotropy 2-sublattice"), INT2(IOI_MESHPARAM, PARAM_K2_AFM));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetocrystalline anisotropy 2-sublattice"), INT2(IOI_MESHPARAM, PARAM_K3_AFM));
	ioInfo.set(param_generic_info + std::string("<i><b>Anisotropy symmetry axis 1\n<i><b>Cartesian unit vector"), INT2(IOI_MESHPARAM, PARAM_EA1));
	ioInfo.set(param_generic_info + std::string("<i><b>Anisotropy symmetry axis 2\n<i><b>Cartesian unit vector"), INT2(IOI_MESHPARAM, PARAM_EA2));
	ioInfo.set(param_generic_info + std::string("<i><b>Anisotropy symmetry axis 3\n<i><b>Cartesian unit vector"), INT2(IOI_MESHPARAM, PARAM_EA3));
	ioInfo.set(param_generic_info + std::string("<i><b>Relative longitudinal\n<i><b>susceptibility for LLB"), INT2(IOI_MESHPARAM, PARAM_SUSREL));
	ioInfo.set(param_generic_info + std::string("<i><b>Relative longitudinal\n<i><b>susceptibility for LLB 2-sublattice"), INT2(IOI_MESHPARAM, PARAM_SUSREL_AFM));
	ioInfo.set(param_generic_info + std::string("<i><b>Relative transverse\n<i><b>susceptibility for LLB"), INT2(IOI_MESHPARAM, PARAM_SUSPREL));
	ioInfo.set(param_generic_info + std::string("<i><b>Applied field coefficient"), INT2(IOI_MESHPARAM, PARAM_HA));
	ioInfo.set(param_generic_info + std::string("<i><b>Magneto-optical field coefficient"), INT2(IOI_MESHPARAM, PARAM_HMO));
	ioInfo.set(param_generic_info + std::string("<i><b>Stochasticity efficiency parameter"), INT2(IOI_MESHPARAM, PARAM_S_EFF));
	ioInfo.set(param_generic_info + std::string("<i><b>Set temperature coefficient"), INT2(IOI_MESHPARAM, PARAM_T));
	ioInfo.set(param_generic_info + std::string("<i><b>Heat source"), INT2(IOI_MESHPARAM, PARAM_Q));
	ioInfo.set(param_generic_info + std::string("<i><b>Base electrical conductivity"), INT2(IOI_MESHPARAM, PARAM_ELC));
	ioInfo.set(param_generic_info + std::string("<i><b>TMR RA product for parallel state"), INT2(IOI_MESHPARAM, PARAM_RATMR_P));
	ioInfo.set(param_generic_info + std::string("<i><b>TMR RA product for antiparallel state"), INT2(IOI_MESHPARAM, PARAM_RATMR_AP));
	ioInfo.set(param_generic_info + std::string("<i><b>Anisotropic magnetoresistance"), INT2(IOI_MESHPARAM, PARAM_AMR));
	ioInfo.set(param_generic_info + std::string("<i><b>Tunnelling anisotropic magnetoresistance"), INT2(IOI_MESHPARAM, PARAM_TAMR));
	ioInfo.set(param_generic_info + std::string("<i><b>Current spin polarization"), INT2(IOI_MESHPARAM, PARAM_P));
	ioInfo.set(param_generic_info + std::string("<i><b>Zhang-Li non-adiabaticity"), INT2(IOI_MESHPARAM, PARAM_BETA));
	ioInfo.set(param_generic_info + std::string("<i><b>Electron diffusion"), INT2(IOI_MESHPARAM, PARAM_DE));
	ioInfo.set(param_generic_info + std::string("<i><b>Diffusion spin polarization"), INT2(IOI_MESHPARAM, PARAM_BETAD));
	ioInfo.set(param_generic_info + std::string("<i><b>Spin-Hall angle\n<i><b>In FM meshes used for SOTField module"), INT2(IOI_MESHPARAM, PARAM_SHA));
	ioInfo.set(param_generic_info + std::string("<i><b>Field-like spin torque coefficient\n<i><b>Used for SOTField and STField modules in FM meshes"), INT2(IOI_MESHPARAM, PARAM_FLSOT));
	ioInfo.set(param_generic_info + std::string("<i><b>Slonczewski macrospin torques parameters."), INT2(IOI_MESHPARAM, PARAM_STQ));
	ioInfo.set(param_generic_info + std::string("<i><b>Slonczewski macrospin torques parameters."), INT2(IOI_MESHPARAM, PARAM_STA));
	ioInfo.set(param_generic_info + std::string("<i><b>Slonczewski macrospin torques parameters."), INT2(IOI_MESHPARAM, PARAM_STQ2));
	ioInfo.set(param_generic_info + std::string("<i><b>Slonczewski macrospin torques parameters."), INT2(IOI_MESHPARAM, PARAM_STA2));
	ioInfo.set(param_generic_info + std::string("<i><b>Slonczewski macrospin torques polarization vector; or SOT symmetry axis."), INT2(IOI_MESHPARAM, PARAM_STP));
	ioInfo.set(param_generic_info + std::string("<i><b>Inverse spin-Hall angle"), INT2(IOI_MESHPARAM, PARAM_ISHA));
	ioInfo.set(param_generic_info + std::string("<i><b>Spin-flip length"), INT2(IOI_MESHPARAM, PARAM_LSF));
	ioInfo.set(param_generic_info + std::string("<i><b>Exchange rotation length"), INT2(IOI_MESHPARAM, PARAM_LEX));
	ioInfo.set(param_generic_info + std::string("<i><b>Spin dephasing length"), INT2(IOI_MESHPARAM, PARAM_LPH));
	ioInfo.set(param_generic_info + std::string("<i><b>Interface spin conductances (majority, minority)\n<i><b>Top mesh sets value even if N"), INT2(IOI_MESHPARAM, PARAM_GI));
	ioInfo.set(param_generic_info + std::string("<i><b>Interface spin mixing conductance (real, imaginary)\n<i><b>Set to zero for continuous N-F interface\n<i><b>Top mesh sets value even if N"), INT2(IOI_MESHPARAM, PARAM_GMIX));
	ioInfo.set(param_generic_info + std::string("<i><b>Spin torque efficiency in the bulk"), INT2(IOI_MESHPARAM, PARAM_TSEFF));
	ioInfo.set(param_generic_info + std::string("<i><b>Spin torque efficiency at interfaces"), INT2(IOI_MESHPARAM, PARAM_TSIEFF));
	ioInfo.set(param_generic_info + std::string("<i><b>Spin pumping efficiency"), INT2(IOI_MESHPARAM, PARAM_PUMPEFF));
	ioInfo.set(param_generic_info + std::string("<i><b>Charge pumping efficiency"), INT2(IOI_MESHPARAM, PARAM_CPUMP_EFF));
	ioInfo.set(param_generic_info + std::string("<i><b>Topological Hall efficiency"), INT2(IOI_MESHPARAM, PARAM_THE_EFF));
	ioInfo.set(param_generic_info + std::string("<i><b>Carrier density"), INT2(IOI_MESHPARAM, PARAM_NDENSITY));
	ioInfo.set(param_generic_info + std::string("<i><b>Seebeck coefficient"), INT2(IOI_MESHPARAM, PARAM_SEEBECK));
	ioInfo.set(param_generic_info + std::string("<i><b>Joule heating effect efficiency"), INT2(IOI_MESHPARAM, PARAM_JOULE_EFF));
	ioInfo.set(param_generic_info + std::string("<i><b>Thermal conductivity"), INT2(IOI_MESHPARAM, PARAM_THERMCOND));
	ioInfo.set(param_generic_info + std::string("<i><b>Mass density"), INT2(IOI_MESHPARAM, PARAM_DENSITY));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetoelastic coefficients (B1, B2)"), INT2(IOI_MESHPARAM, PARAM_MECOEFF));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetostriction coefficients (B1, B2)"), INT2(IOI_MESHPARAM, PARAM_MMECOEFF));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetoelastic coefficients (see manual)"), INT2(IOI_MESHPARAM, PARAM_MECOEFF2));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetostriction coefficients (see manual)"), INT2(IOI_MESHPARAM, PARAM_MMECOEFF2));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetoelastic coefficients (see manual)"), INT2(IOI_MESHPARAM, PARAM_MECOEFF3));
	ioInfo.set(param_generic_info + std::string("<i><b>Magnetostriction coefficients (see manual)"), INT2(IOI_MESHPARAM, PARAM_MMECOEFF3));
	ioInfo.set(param_generic_info + std::string("<i><b>Young's modulus"), INT2(IOI_MESHPARAM, PARAM_YOUNGSMOD));
	ioInfo.set(param_generic_info + std::string("<i><b>Poisson's ratio"), INT2(IOI_MESHPARAM, PARAM_POISSONRATIO));
	ioInfo.set(param_generic_info + std::string("<i><b>Stiffness constants - c11, c12, c44"), INT2(IOI_MESHPARAM, PARAM_STIFFC_CUBIC));
	ioInfo.set(param_generic_info + std::string("<i><b>Stiffness constants - c22, c23, c55"), INT2(IOI_MESHPARAM, PARAM_STIFFC_2));
	ioInfo.set(param_generic_info + std::string("<i><b>Stiffness constants - c33, c13, c66"), INT2(IOI_MESHPARAM, PARAM_STIFFC_3));
	ioInfo.set(param_generic_info + std::string("<i><b>Stiffness constants - c14, c15, c16"), INT2(IOI_MESHPARAM, PARAM_STIFFC_S));
	ioInfo.set(param_generic_info + std::string("<i><b>Mechanical damping"), INT2(IOI_MESHPARAM, PARAM_MDAMPING));
	ioInfo.set(param_generic_info + std::string("<i><b>Coefficient of thermal expansion"), INT2(IOI_MESHPARAM, PARAM_THERMEL));
	ioInfo.set(param_generic_info + std::string("<i><b>Specific heat capacity"), INT2(IOI_MESHPARAM, PARAM_SHC));
	ioInfo.set(param_generic_info + std::string("<i><b>Electronic specific heat capacity"), INT2(IOI_MESHPARAM, PARAM_SHC_E));
	ioInfo.set(param_generic_info + std::string("<i><b>Electron coupling constant\n<i><b>2TM : electron-lattice"), INT2(IOI_MESHPARAM, PARAM_G_E));

	ioInfo.set(param_generic_info + std::string("<i><b>Intrinsic damping - SC"), INT2(IOI_MESHPARAM, PARAM_ATOM_DAMPING));
	ioInfo.set(param_generic_info + std::string("<i><b>Atomistic magnetic moment - SC"), INT2(IOI_MESHPARAM, PARAM_ATOM_MUS));
	ioInfo.set(param_generic_info + std::string("<i><b>Heisenberg exchange energy - SC"), INT2(IOI_MESHPARAM, PARAM_ATOM_J));
	ioInfo.set(param_generic_info + std::string("<i><b>Atomistic DMI exchange energy - SC"), INT2(IOI_MESHPARAM, PARAM_ATOM_D));
	ioInfo.set(param_generic_info + std::string("<i><b>Surface exchange energy"), INT2(IOI_MESHPARAM, PARAM_ATOM_JS));
	ioInfo.set(param_generic_info + std::string("<i><b>Secondary surface exchange energy"), INT2(IOI_MESHPARAM, PARAM_ATOM_JS2));
	ioInfo.set(param_generic_info + std::string("<i><b>Atomistic anisotropy - SC"), INT2(IOI_MESHPARAM, PARAM_ATOM_K1));
	ioInfo.set(param_generic_info + std::string("<i><b>Atomistic anisotropy - SC"), INT2(IOI_MESHPARAM, PARAM_ATOM_K2));
	ioInfo.set(param_generic_info + std::string("<i><b>Atomistic anisotropy - SC"), INT2(IOI_MESHPARAM, PARAM_ATOM_K3));
	ioInfo.set(param_generic_info + std::string("<i><b>Anisotropy symmetry axis 1"), INT2(IOI_MESHPARAM, PARAM_ATOM_EA1));
	ioInfo.set(param_generic_info + std::string("<i><b>Anisotropy symmetry axis 2"), INT2(IOI_MESHPARAM, PARAM_ATOM_EA2));
	ioInfo.set(param_generic_info + std::string("<i><b>Anisotropy symmetry axis 3"), INT2(IOI_MESHPARAM, PARAM_ATOM_EA3));

	//Shows parameter temperature dependence for a given mesh : minorId is the major id of elements in SimParams::simParams (i.e. an entry from PARAM_ enum), auxId is the unique mesh id number, textId is the parameter temperature dependence setting
	//IOI_MESHPARAMTEMP

	std::string paramtemp_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Material parameter\n[tc1,1,0,1/tc]<b>temperature dependence</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: none</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: clear\n");

	ioInfo.push_back(paramtemp_generic_info, IOI_MESHPARAMTEMP);

	//Shows parameter spatial dependence for a given mesh : minorId is the major id of elements in SimParams::simParams (i.e. an entry from PARAM_ enum), auxId is the unique mesh id number, textId is the parameter spatial dependence setting
	//IOI_MESHPARAMVAR

	std::string paramvar_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Material parameter spatial variation</b>") +
		std::string("\n[tc1,1,0,1/tc]<i>green: display selected, red: not selected</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]left-click: display select\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: clear\n");

	ioInfo.push_back(paramvar_generic_info, IOI_MESHPARAMVAR);

	//Shows a possible temperature dependence : minorId is the type (entry from MATPTDEP_ enum)
	//IOI_MESHPARAMTEMPTYPE

	std::string paramtemptype_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Temperature dependence type\n") +
		std::string("\n[tc1,1,0,1/tc]drag: move to parameter to set\n");

	ioInfo.set(paramtemptype_generic_info + std::string("<i><b>No temperature dependence"), INT2(IOI_MESHPARAMTEMPTYPE, MATPTDEP_NONE));
	ioInfo.set(paramtemptype_generic_info + std::string("<i><b>Array"), INT2(IOI_MESHPARAMTEMPTYPE, MATPTDEP_ARRAY));
	ioInfo.set(paramtemptype_generic_info + std::string("<i><b>Custom equation"), INT2(IOI_MESHPARAMTEMPTYPE, MATPTDEP_EQUATION));

	//Shows a possible generator name for mesh parameter spatial dependence : minorId is the MATPVAR_ enum value, textId is the generator name
	//IOI_MESHPARAMVARGENERATOR 

	std::string paramvargen_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Spatial dependence generator") +
		std::string("\n[tc1,1,0,1/tc]drag: move to parameter to set\n");

	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Set from png mask file with grayscale.\n<i><b>offset, scale, filename\n<i><b>black = 0, white = 1"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_MASK));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Set scaling in given shape only."), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_SHAPE));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Set from ovf2 file (mapped to current dimensions)."), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_OVF2));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Random with range (min, max) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_RANDOM));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Jagged with range (min, max)\n<i><b>spacing (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_JAGGED));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Polynomial slopes at sides along each axis\n<b>to length ratio from max at surface to min at centre\n<i>ratio_-x, ratio_+x;\n<i>ratio_-y, ratio_+y;\n<i>ratio_-z, ratio_+z;\n<i>min, max, polynomial exponent;<b>"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_ABLPOL));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Tanh slopes at sides along each axis\n<b>to length ratio from max at surface to min at centre\n<i>ratio_-x, ratio_+x;\n<i>ratio_-y, ratio_+y;\n<i>ratio_-z, ratio_+z;\n<i>min, max, tanh sigma in nm;<b>"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_ABLTANH));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Exp slopes at sides along each axis\n<b>to length ratio from max at surface to min at centre\n<i>ratio_-x, ratio_+x;\n<i>ratio_-y, ratio_+y;\n<i>ratio_-z, ratio_+z;\n<i>min, max, exp sigma in nm;<b>"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_ABLEXP));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Circular defects with value range (min, max)\n<i><b>diameter range (min, max)\n<i><b>average spacing (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_DEFECTS));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Line faults with value range (min, max)\n<i><b>length range (m) (min, max)\n<i><b>orientation range in degrees (min, max)\n<i><b>average spacing (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_FAULTS));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Voronoi 2D cells with value range (min, max)\n<i><b>average spacing (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_VORONOI2D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Uniform Voronoi 2D cells with value range (min, max)\n<i><b>spacing (m) variation (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_UVORONOI2D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Voronoi 3D cells with value range (min, max)\n<i><b>average spacing (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_VORONOI3D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Uniform Voronoi 3D cells with value range (min, max)\n<i><b>spacing (m) variation (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_UVORONOI3D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Voronoi 2D cells with value range (min, max)\n<i><b>average spacing (m) and seed\n<i><b>Apply to Voronoi cell bounaries."), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_VORONOIBND2D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Uniform Voronoi 2D cells with value range (min, max)\n<i><b>spacing (m) variation (m) and seed\n<i><b>Apply to Voronoi cell bounaries."), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_UVORONOIBND2D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Voronoi 3D cells with value range (min, max)\n<i><b>average spacing (m) and seed\n<i><b>Apply to Voronoi cell bounaries."), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_VORONOIBND3D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Uniform Voronoi 3D cells with value range (min, max)\n<i><b>spacing (m) variation (m) and seed\n<i><b>Apply to Voronoi cell bounaries."), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_UVORONOIBND3D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Voronoi 2D cells for rotations.\n<i><b>Rotate vectorial parameters\n<i><b>through polar degrees (min, max)\n<i><b>and azimuthal degrees (min, max).\n<i><b>Average spacing (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_VORONOIROT2D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Uniform Voronoi 2D cells for rotations.\n<i><b>Rotate vectorial parameters\n<i><b>through polar degrees (min, max)\n<i><b>and azimuthal degrees (min, max).\n<i><b>spacing (m) variation (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_UVORONOIROT2D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Voronoi 3D cells for rotations.\n<i><b>Rotate vectorial parameters\n<i><b>through polar degrees (min, max)\n<i><b>and azimuthal degrees (min, max).\n<i><b>Average spacing (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_VORONOIROT3D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>Uniform Voronoi 3D cells for rotations.\n<i><b>Rotate vectorial parameters\n<i><b>through polar degrees (min, max)\n<i><b>and azimuthal degrees (min, max).\n<i><b>spacing (m) variation (m) and seed"), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_UVORONOIROT3D));
	ioInfo.set(paramvargen_generic_info + std::string("<i><b>User text equation.\n<i><b>Scalar or vector equation:\n<i><b>define vector equation for rotations only.\n<i><b>Rotation specified using unit vector:\n<i><b>use direction cosines."), INT2(IOI_MESHPARAMVARGENERATOR, MATPVAR_EQUATION));

	//Shows mesh display option for a given mesh : minorId is the MESHDISPLAY_ value, auxId is the unique mesh id number, textId is the MESHDISPLAY_ handle
	//IOI_MESHDISPLAY

	std::string meshdisplay_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Mesh quantity to display") +
		std::string("\n[tc1,1,0,1/tc]<i>green (foreground): on, red: off</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>orange (background): on, red: off</i>") +
		std::string("\n[tc1,1,0,1/tc]left-click: change foreground state\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: change background state\n");

	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Nothing displayed"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_NONE));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>magnetization"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_MAGNETIZATION));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Atomic Moments"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_MOMENT));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>magnetization sub-lattice 2"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_MAGNETIZATION2));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>AF magnetization"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_MAGNETIZATION12));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Effective H field (total or module)"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_EFFECTIVEFIELD));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Effective H field sub-lattice 2 (total or module)"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_EFFECTIVEFIELD2));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>AF effective H field (total or module)"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_EFFECTIVEFIELD12));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Module energy density spatial variation"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_ENERGY));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Module sub-lattice 2 energy density spatial variation"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_ENERGY2));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Charge current density"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_CURRDENSITY));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Charge potential"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_VOLTAGE));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Electrical conductivity"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_ELCOND));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Spin accumulation"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_SACCUM));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Spin x-current density"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_JSX));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Spin y-current density"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_JSY));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Spin z-current density"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_JSZ));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Bulk spin torque"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_TS));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Interfacial spin torque"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_TSI));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Temperature"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_TEMPERATURE));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Mechanical Displacement"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_UDISP));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Strain : xx, yy, zz"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_STRAINDIAG));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Strain : yz, xz, xy"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_STRAINODIAG));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Mesh parameter spatial variation"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_PARAMVAR));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Roughness"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_ROUGHNESS));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Custom, Vectorial"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_CUSTOM_VEC));
	ioInfo.set(meshdisplay_generic_info + std::string("<i><b>Custom, Scalar"), INT2(IOI_MESHDISPLAY, MESHDISPLAY_CUSTOM_SCA));

	//Shows dual mesh display transparency values : textId is the DBL2 value as a std::string
	//IOI_MESHDISPLAYTRANSPARENCY,

	std::string meshdisplay_transparency_info =
		std::string("[tc1,1,0,1/tc]<b>Transparency values for dual mesh display") +
		std::string("\n[tc1,1,0,1/tc]<i>Foreground, Background</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>0: transparent, 1: opaque</i>") +
		std::string("\n[tc1,1,0,1/tc]double-click: edit\n");

	ioInfo.push_back(meshdisplay_transparency_info, IOI_MESHDISPLAYTRANSPARENCY);

	//Shows mesh display threshold values : textId is the DBL2 value as a std::string
	//IOI_MESHDISPLAYTHRESHOLDS,

	std::string meshdisplay_thresholds_info =
		std::string("[tc1,1,0,1/tc]<b>Threshold values for display") +
		std::string("\n[tc1,1,0,1/tc]<i>Minimum, Maximum</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>Set both to 0 to disable.</i>") +
		std::string("\n[tc1,1,0,1/tc]double-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: clear\n");

	ioInfo.push_back(meshdisplay_thresholds_info, IOI_MESHDISPLAYTHRESHOLDS);

	//Shows mesh display threshold trigger type : auxId is the trigger option
	//IOI_MESHDISPLAYTHRESHOLDTRIG

	std::string meshdisplay_thresholdtrig_info =
		std::string("[tc1,1,0,1/tc]<b>Threshold trigger component") +
		std::string("\n[tc1,1,0,1/tc]<i>X, Y, Z, magnitude</i>") +
		std::string("\n[tc1,1,0,1/tc]left-click: change state forward") +
		std::string("\n[tc1,1,0,1/tc]right-click: change state backward\n");

	ioInfo.push_back(meshdisplay_thresholdtrig_info, IOI_MESHDISPLAYTHRESHOLDTRIG);

	//Shows super-mesh display option : minorId is the MESHDISPLAY_ value, textId is the MESHDISPLAY_ handle
	//IOI_SMESHDISPLAY

	std::string smeshdisplay_generic_info =
		std::string("[tc1,1,0,1/tc]<b>Supermesh quantity to display") +
		std::string("\n[tc1,1,0,1/tc]<i>green: on, red: off</i>\n") +
		std::string("\n[tc1,1,0,1/tc]<i>When set, individual mesh\n") +
		std::string("\n[tc1,1,0,1/tc]<i>display not enabled</i>\n") +
		std::string("\n[tc1,1,0,1/tc]click: change state\n");

	ioInfo.set(smeshdisplay_generic_info + std::string("<i><b>Nothing displayed"), INT2(IOI_SMESHDISPLAY, MESHDISPLAY_NONE));
	ioInfo.set(smeshdisplay_generic_info + std::string("<i><b>Demagnetising field (supermesh convolution only)"), INT2(IOI_SMESHDISPLAY, MESHDISPLAY_SM_DEMAG));
	ioInfo.set(smeshdisplay_generic_info + std::string("<i><b>Oersted field"), INT2(IOI_SMESHDISPLAY, MESHDISPLAY_SM_OERSTED));
	ioInfo.set(smeshdisplay_generic_info + std::string("<i><b>Total dipole stray field"), INT2(IOI_SMESHDISPLAY, MESHDISPLAY_SM_STRAYH));

	//Shows mesh vectorial quantity display option : minorId is the unique mesh id number, auxId is the display option
	//IOI_MESHVECREP
	//IOI_SMESHVECREP

	std::string meshdisplay_option_info =
		std::string("[tc1,1,0,1/tc]<b>Vectorial quantity display option") +
		std::string("\n[tc1,1,0,1/tc]<i>full, X, Y, Z, direction, magnitude</i>") +
		std::string("\n[tc1,1,0,1/tc]left-click: change state forward") +
		std::string("\n[tc1,1,0,1/tc]right-click: change state backward\n");

	ioInfo.push_back(meshdisplay_option_info, IOI_MESHVECREP);
	ioInfo.push_back(meshdisplay_option_info, IOI_SMESHVECREP);

	//Shows movingmesh trigger settings : minorId is the unique mesh id number (if set), auxId is the trigger state (used or not used), textId is the mesh name (if set)
	//IOI_MOVINGMESH

	std::string movingmesh_info =
		std::string("[tc1,1,0,1/tc]<b>Moving mesh algorithm") +
		std::string("\n[tc1,1,0,1/tc]<i>green: on, red: off</i>\n") +
		std::string("\n[tc1,1,0,1/tc]<i>When set, trigger is\n") +
		std::string("\n[tc1,1,0,1/tc]<i>set on given mesh</i>\n") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: clear\n");

	ioInfo.push_back(movingmesh_info, IOI_MOVINGMESH);

	//Shows movingmesh trigger settings : minorId is the unique mesh id number (if set), auxId is the trigger state (used or not used), textId is the mesh name (if set)
	//IOI_MOVINGMESHASYM

	std::string movingmeshasym_info =
		std::string("[tc1,1,0,1/tc]<b>Moving mesh symmetry") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(movingmeshasym_info, IOI_MOVINGMESHASYM);

	//Shows movingmesh trigger settings : minorId is the unique mesh id number (if set), auxId is the trigger state (used or not used), textId is the mesh name (if set)
	//IOI_MOVINGMESHTHRESH

	std::string movingmeshthresh_info =
		std::string("[tc1,1,0,1/tc]<b>Moving mesh threshold") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(movingmeshthresh_info, IOI_MOVINGMESHTHRESH);

	//Shows electrode box. minorId is the minor Id in STransport::electrode_boxes, auxId is the number of the interactive object in the list (electrode index), textId is the electrode rect as a std::string
	//IOI_ELECTRODERECT

	std::string electroderect_info =
		std::string("[tc1,1,0,1/tc]<b>Electrode rectangle") +
		std::string("\n[tc1,1,0,1/tc]<i>Sets fixed potential at") +
		std::string("\n[tc1,1,0,1/tc]<i>intersections with mesh sides") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: delete\n");

	ioInfo.push_back(electroderect_info, IOI_ELECTRODERECT);

	//Shows electrode potential. minorId is the electrode index, textId is potential value as a std::string
	//IOI_ELECTRODEPOTENTIAL

	std::string electrodepotential_info =
		std::string("[tc1,1,0,1/tc]<b>Electrode potential") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: delete\n");

	ioInfo.push_back(electrodepotential_info, IOI_ELECTRODEPOTENTIAL);

	//Shows electrode ground setting. minorId is the electrode index, auxId is the setting (0 : not ground, 1 : ground)
	//IOI_ELECTRODEGROUND

	std::string electrodeground_info =
		std::string("[tc1,1,0,1/tc]<b>Electrode ground setting") +
		std::string("\n[tc1,1,0,1/tc]<i>green: ground, red: not ground</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>Only one electrode can be a ground") +
		std::string("\n[tc1,1,0,1/tc]click: set ground\n");

	ioInfo.push_back(electrodeground_info, IOI_ELECTRODEGROUND);

	//Shows constant current source setting. auxId is the setting.
	//IOI_CONSTANTCURRENTSOURCE

	std::string powersupply_info =
		std::string("[tc1,1,0,1/tc]<b>Power supply mode") +
		std::string("\n[tc1,1,0,1/tc]<i>Constant current or constant voltage</i>") +
		std::string("\n[tc1,1,0,1/tc]click: switch mode\n");

	ioInfo.push_back(powersupply_info, IOI_CONSTANTCURRENTSOURCE);

	//Shows transport solver convergence error. textId is the convergence error value.
	//IOI_TSOLVERCONVERROR

	std::string tsolvererror_info =
		std::string("[tc1,1,0,1/tc]<b>Transport solver error - V") +
		std::string("\n[tc1,1,0,1/tc]<i>Convergence error setting</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(tsolvererror_info, IOI_TSOLVERCONVERROR);

	//Shows transport solver timeout iterations. auxId is the timeout value.
	//IOI_TSOLVERTIMEOUT

	std::string tsolvertout_info =
		std::string("[tc1,1,0,1/tc]<b>Transport solver timeout - V") +
		std::string("\n[tc1,1,0,1/tc]<i>Maximum number of iterations</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>allowed to reach convergence</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(tsolvertout_info, IOI_TSOLVERTIMEOUT);

	//Shows spin transport solver convergence error. textId is the convergence error value.
	//IOI_SSOLVERCONVERROR

	std::string ssolvererror_info =
		std::string("[tc1,1,0,1/tc]<b>Transport solver error - S") +
		std::string("\n[tc1,1,0,1/tc]<i>Convergence error setting</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(ssolvererror_info, IOI_SSOLVERCONVERROR);

	//Shows spin transport solver timeout iterations. auxId is the timeout value.
	//IOI_SSOLVERTIMEOUT

	std::string ssolvertout_info =
		std::string("[tc1,1,0,1/tc]<b>Transport solver timeout - S") +
		std::string("\n[tc1,1,0,1/tc]<i>Maximum number of iterations</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>allowed to reach convergence</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(ssolvertout_info, IOI_SSOLVERTIMEOUT);


	//Shows SOR damping values when used in fixed damping mode. textId is the DBL2 damping value as a std::string. (DBL2 since we need different damping values for V and S solvers)
	//IOI_SORDAMPING

	std::string sordampingvalues_info =
		std::string("[tc1,1,0,1/tc]<b>SOR fixed damping") +
		std::string("\n[tc1,1,0,1/tc]<i>SOR damping for (V, S) solvers</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(sordampingvalues_info, IOI_SORDAMPING);

	//Shows tmr type setting. minorId is the unique mesh id number, auxId is the value.
	//IOI_TMRTYPE

	std::string tmr_info =
		std::string("[tc1,1,0,1/tc]<b>TMR setting") +
		std::string("\n[tc1,1,0,1/tc]<i>Used to calculate TMR angle dependence</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change\n");

	ioInfo.push_back(tmr_info, IOI_TMRTYPE);

	//Static transport solver state. auxId is the value (0/1)
	//IOI_STATICTRANSPORT

	std::string statictransport_info =
		std::string("[tc1,1,0,1/tc]<b>Static transport solver") +
		std::string("\n[tc1,1,0,1/tc]<i>If set, transport solver iterated only</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>at end of a step or stage.</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>You should also set a high timeout.</i>") +
		std::string("\n[tc1,1,0,1/tc]click: switch mode\n");

	ioInfo.push_back(statictransport_info, IOI_STATICTRANSPORT);

	//Disabled transport solver state. auxId is the value (0/1)
	//IOI_DISABLEDTRANSPORT

	std::string disabledtransport_info =
		std::string("[tc1,1,0,1/tc]<b>Transport solver status") +
		std::string("\n[tc1,1,0,1/tc]<i>If disabled, transport solver</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>does not iterate.</i>") +
		std::string("\n[tc1,1,0,1/tc]click: switch mode\n");

	ioInfo.push_back(disabledtransport_info, IOI_DISABLEDTRANSPORT);

	//Shows mesh temperature. minorId is the unique mesh id number, textId is the temperature value
	//IOI_BASETEMPERATURE

	std::string basetemperature_info =
		std::string("[tc1,1,0,1/tc]<b>Mesh base temperature") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(basetemperature_info, IOI_BASETEMPERATURE);

	//Shows ambient temperature for heat equation Robin boundary conditions. minorId is the unique mesh id number, auxId is enabled/disabled status (Heat module must be active), textId is the temperature value
	//IOI_AMBIENT_TEMPERATURE

	std::string ambienttemperature_info =
		std::string("[tc1,1,0,1/tc]<b>Mesh ambient temperature") +
		std::string("\n[tc1,1,0,1/tc]<i>Air temperature outside of mesh</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(ambienttemperature_info, IOI_AMBIENT_TEMPERATURE);

	//Shows alpha value (W/m^2K) for heat equation Robin boundary conditions. minorId is the unique mesh id number, auxId is enabled/disabled status (Heat module must be active), textId is the value
	//IOI_ROBIN_ALPHA

	std::string robinalpha_info =
		std::string("[tc1,1,0,1/tc]<b>Robin heat flux coefficient") +
		std::string("\n[tc1,1,0,1/tc]<i>Boundary heat flux</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>for Newton's law of cooling:</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>flux = coeff * (T_boundary - T_ambient)</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(robinalpha_info, IOI_ROBIN_ALPHA);

	//Shows temperature insulating side setting for heat equation. minorId is the unique mesh id number, auxId is the status (Heat module must be active) : -1 disabled (gray), 0 not insulating (green), 1 insulating (red), textId represents the side : "x", "-x", "y", "-y", "z", "-z"
	//IOI_INSULATINGSIDE

	std::string insulatingside_info =
		std::string("[tc1,1,0,1/tc]<b>Insulating mesh side setting") +
		std::string("\n[tc1,1,0,1/tc]<i>green: not insulating, red: insulating</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>If insulating no heat flux allowed</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>else Newton's law of cooling applies</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(insulatingside_info, IOI_INSULATINGSIDE);

	//Shows mesh Curie temperature. minorId is the unique mesh id number, auxId is available/not available status (must be ferromagnetic mesh), textId is the temperature value
	//IOI_CURIETEMP

	std::string curietemperature_info =
		std::string("[tc1,1,0,1/tc]<b>Mesh Curie temperature") +
		std::string("\n[tc1,1,0,1/tc]<i>When set, parameter temperature</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>dependencies automatically calculated for:</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>damping, Ms, A, P, K1, K2, susrel</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(curietemperature_info, IOI_CURIETEMP);

	//Shows indicative material Curie temperature. minorId is the unique mesh id number, auxId is available/not available status (must be ferromagnetic mesh), textId is the temperature value
	//IOI_CURIETEMPMATERIAL

	curietemperature_info =
		std::string("[tc1,1,0,1/tc]<b>Material Curie temperature") +
		std::string("\n[tc1,1,0,1/tc]<i>This is the actual value for the material</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>Not used in calculations, only indicative</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(curietemperature_info, IOI_CURIETEMPMATERIAL);

	//Shows atomic moment multiple of Bohr magneton. minorId is the unique mesh id number, auxId is available/not available status (must be ferromagnetic mesh), textId is the value
	//IOI_ATOMICMOMENT

	std::string atomicmoment_info =
		std::string("[tc1,1,0,1/tc]<b>Magnetic moment in Bohr magnetons") +
		std::string("\n[tc1,1,0,1/tc]<i>Used to calculate parameter</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>temperature dependencies when T_Curie > 0</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>In particular field-dependence is introduced</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(atomicmoment_info, IOI_ATOMICMOMENT);

	//Shows atomic moment multiple of Bohr magneton for AF meshes. minorId is the unique mesh id number, auxId is available/not available status (must be antiferromagnetic mesh), textId is the value
	//IOI_ATOMICMOMENT_AFM

	std::string atomicmoment_afm_info =
		std::string("[tc1,1,0,1/tc]<b>Magnetic moment in Bohr magnetons") +
		std::string("\n[tc1,1,0,1/tc]<i>Used to calculate parameter</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>temperature dependencies when T_Neel > 0</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>In particular field-dependence is introduced</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(atomicmoment_afm_info, IOI_ATOMICMOMENT_AFM);

	//Shows Tc tau couplings. minorId is the unique mesh id number, auxId is available/not available status (must be antiferromagnetic mesh), textId is the value
	//IOI_TAU

	std::string tau_info =
		std::string("[tc1,1,0,1/tc]<b>Tc tau couplings") +
		std::string("\n[tc1,1,0,1/tc]<i>Used to calculate parameter</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>temperature dependencies when T_Neel > 0</i>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(tau_info, IOI_TAU);

	//Shows temperature model type for mesh. minorId is the unique mesh id number, auxId is the model identifier (entry from TMTYPE_ enum)
	//IOI_TMODEL

	std::string tmodel_info =
		std::string("[tc1,1,0,1/tc]<b>Temperature model type") +
		std::string("\n[tc1,1,0,1/tc]click: change\n");

	ioInfo.push_back(tmodel_info, IOI_TMODEL);

	//Shows cuda enabled/disabled or n/a state. auxId is enabled (1)/disabled(0)/not available(-1) status.
	//IOI_CUDASTATE

	std::string cudastate_info =
		std::string("[tc1,1,0,1/tc]<b>CUDA computations state") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>When set, all computations done on the GPU</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>otherwise done on the CPU</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(cudastate_info, IOI_CUDASTATE);

	//Shows CUDA device information and state. minorId is the device number (from 1 up), auxId is enabled (1)/disabled(0)/not available(-1) status. 
	//IOI_CUDADEVICE

	std::string cudadevice_info =
		std::string("[tc1,1,0,1/tc]<b>CUDA device selector") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set, gray: N/A</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(cudadevice_info, IOI_CUDADEVICE);

	//Shows gpu free memory. auxId is the value
	//IOI_GPUMEMFREE

	std::string gpufreemem_info =
		std::string("[tc1,1,0,1/tc]<b>GPU free memory");

	ioInfo.push_back(gpufreemem_info, IOI_GPUMEMFREE);

	//Shows gpu total memory. auxId is the value
	//IOI_GPUMEMTOTAL

	std::string gputotalmem_info =
		std::string("[tc1,1,0,1/tc]<b>GPU total memory");

	ioInfo.push_back(gputotalmem_info, IOI_GPUMEMTOTAL);

	//Shows cpu free memory. auxId is the value
	//IOI_CPUMEMFREE

	std::string cpufreemem_info =
		std::string("[tc1,1,0,1/tc]<b>CPU free memory");

	ioInfo.push_back(cpufreemem_info, IOI_CPUMEMFREE);

	//Shows cpu total memory. auxId is the value
	//IOI_CPUMEMTOTAL

	std::string cputotalmem_info =
		std::string("[tc1,1,0,1/tc]<b>CPU total memory");

	ioInfo.push_back(cputotalmem_info, IOI_CPUMEMTOTAL);

	//Shows scale_rects enabled/disabled state. auxId is enabled (1)/disabled(0) status.
	//IOI_SCALERECTSSTATUS

	std::string scalerects_info =
		std::string("[tc1,1,0,1/tc]<b>Scale mesh rectangles status") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>When set, changing a mesh size will</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>change all other meshes in proportion</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(scalerects_info, IOI_SCALERECTSSTATUS);

	//Shows coupled_to_dipoles enabled/disabled state. auxId is enabled (1)/disabled(0) status.
	//IOI_COUPLEDTODIPOLESSTATUS

	std::string dipolecouple_info =
		std::string("[tc1,1,0,1/tc]<b>Dipole exchange coupling status") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>When set, for dipole-magnetic mesh contacts</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>moments at interface cells will be frozen in dipole direction</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(dipolecouple_info, IOI_COUPLEDTODIPOLESSTATUS);

	//Shows dipole velocity value. minorId is the unique mesh id number. textId is the value
	//IOI_DIPOLEVELOCITY

	std::string dipolevelocity_info =
		std::string("[tc1,1,0,1/tc]<b>Dipole velocity") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(dipolevelocity_info, IOI_DIPOLEVELOCITY);

	//Shows dipole shift clipping value. minorId is the unique mesh id number. textId is the value
	//IOI_DIPOLESHIFTCLIP

	std::string dipoleclipping_info =
		std::string("[tc1,1,0,1/tc]<b>Dipole position clipping") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(dipoleclipping_info, IOI_DIPOLESHIFTCLIP);

	//Shows strain set equation. minorId is the unique mesh id number. textId is the equation. auxId is enabled(1)/disabled(0) status.
	//IOI_STRAINEQUATION:
	//IOI_SHEARSTRAINEQUATION:

	std::string strainequation_info =
		std::string("[tc1,1,0,1/tc]<b>User set strain equations") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: disable\n");

	ioInfo.push_back(strainequation_info, IOI_STRAINEQUATION);
	ioInfo.push_back(strainequation_info, IOI_SHEARSTRAINEQUATION);

	//Shows fixed surface rectangle. minorId is the minor Id in SMElastic::fixed_u_surfaces, auxId is the number of the interactive object in the list (electrode index), textId is the surface rect as a std::string
	//IOI_SURFACEFIX

	std::string surfacefix_info =
		std::string("[tc1,1,0,1/tc]<b>Fixed surface rectangle") +
		std::string("\n[tc1,1,0,1/tc]<i>Sets fixed surface for") +
		std::string("\n[tc1,1,0,1/tc]<i>elastodynamics solver") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: delete\n");

	ioInfo.push_back(surfacefix_info, IOI_SURFACEFIX);

	//Shows stress surface rectangle. minorId is the minor Id in SMElastic::stress_surfaces_rect, auxId is the number of the interactive object in the list (electrode index), textId is the surface rect as a std::string
	//IOI_SURFACESTRESS

	std::string surfacestress_info =
		std::string("[tc1,1,0,1/tc]<b>Stress surface rectangle") +
		std::string("\n[tc1,1,0,1/tc]<i>Sets stress surface for") +
		std::string("\n[tc1,1,0,1/tc]<i>elastodynamics solver") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: delete\n");

	ioInfo.push_back(surfacestress_info, IOI_SURFACESTRESS);

	//Shows stress surface equation. minorId is the index in SMElastic::stress_surfaces_equations, auxId is the number of the interactive object in the list (electrode index), textId is the equation
	//IOI_SURFACESTRESSEQ

	std::string surfacestresseq_info =
		std::string("[tc1,1,0,1/tc]<b>Stress surface equation") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n") +
		std::string("\n[tc1,1,0,1/tc]right-click: delete\n");

	ioInfo.push_back(surfacestresseq_info, IOI_SURFACESTRESSEQ);

	//Shows log_errors enabled/disabled state. auxId is enabled (1)/disabled(0) status.
	//IOI_ERRORLOGSTATUS:

	std::string logerrors_info =
		std::string("[tc1,1,0,1/tc]<b>Log errors status") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(logerrors_info, IOI_ERRORLOGSTATUS);

	//Shows start_check_updates enabled/disabled state. auxId is enabled (1)/disabled(0) status.
	//IOI_UPDATESTATUSCHECKSTARTUP

	std::string start_check_updates_info =
		std::string("[tc1,1,0,1/tc]<b>Check for updates on startup") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(start_check_updates_info, IOI_UPDATESTATUSCHECKSTARTUP);

	//Shows start_scriptserver enabled/disabled state. auxId is enabled (1)/disabled(0) status.
	//IOI_SCRIPTSERVERSTARTUP

	std::string start_scriptserver_info =
		std::string("[tc1,1,0,1/tc]<b>Start script server on startup") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(start_scriptserver_info, IOI_SCRIPTSERVERSTARTUP);

	//Shows number of threads. auxId is the value.
	//IOI_THREADS

	std::string threads_info =
		std::string("[tc1,1,0,1/tc]<b>Number of threads for CPU computations") +
		std::string("\n[tc1,1,0,1/tc]right-click: maximum") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(threads_info, IOI_THREADS);

	//Shows server port. auxId is the value.
	//IOI_SERVERPORT

	std::string serverport_info =
		std::string("[tc1,1,0,1/tc]<b>Script server port") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(serverport_info, IOI_SERVERPORT);

	//Shows server password. textId is the password.
	//IOI_SERVERPWD

	std::string serverpwd_info =
		std::string("[tc1,1,0,1/tc]<b>Script server password") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(serverpwd_info, IOI_SERVERPWD);

	//Shows server sleep time in ms. auxId is the value.
	//IOI_SERVERSLEEPMS

	std::string serversleepms_info =
		std::string("[tc1,1,0,1/tc]<b>Script server sleep time in ms") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(serversleepms_info, IOI_SERVERSLEEPMS);

	//IOI_MESHEXCHCOUPLING

	std::string ec_info =
		std::string("[tc1,1,0,1/tc]<b>Neighboring mesh exchange coupling status") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>When set, neighboring ferromagnetic meshes</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>in contact with this one will be exchange coupled.</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(ec_info, IOI_MESHEXCHCOUPLING);

	//Shows mesh roughness refinement value. minorId is the unique mesh id number, textId is the value
	//IOI_REFINEROUGHNESS

	std::string roughness_refine_info =
		std::string("[tc1,1,0,1/tc]<b>Mesh roughness") +
		std::string("\n[tc1,1,0,1/tc]<b>cells refinement multiplier\n") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit\n");

	ioInfo.push_back(roughness_refine_info, IOI_REFINEROUGHNESS);

	//Multi-layered convolution configuration
	//IOI_MULTICONV, IOI_2DMULTICONV, IOI_NCOMMONSTATUS, IOI_NCOMMON

	std::string IOI_MULTICONV_info =
		std::string("[tc1,1,0,1/tc]<b>Multi-layered convolution status") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set, gray: N/A (</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>When set, use multi-layered convolution</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>instead of super-mesh convolution.</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(IOI_MULTICONV_info, IOI_MULTICONV);

	IOI_MULTICONV_info =
		std::string("[tc1,1,0,1/tc]<b>2D Multi-layered convolution status") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set, gray: N/A (</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>When set, force multi-layered convolution</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>to 2D in each layer.</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(IOI_MULTICONV_info, IOI_2DMULTICONV);

	IOI_MULTICONV_info =
		std::string("[tc1,1,0,1/tc]<b>Use default discretisation status") +
		std::string("\n[tc1,1,0,1/tc]<i>green: set, red: not set, gray: N/A (</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>When set, use default common discretisation</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>for multi-layered convolution.</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(IOI_MULTICONV_info, IOI_NCOMMONSTATUS);

	IOI_MULTICONV_info =
		std::string("[tc1,1,0,1/tc]<b>Common discretisation") +
		std::string("\n[tc1,1,0,1/tc]<i>Common discretisation</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>for multi-layered convolution.</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(IOI_MULTICONV_info, IOI_NCOMMON);

	//Shows status of gpu kernels demag initialization. auxId is the status (0 : Off, 1 : On)
	//IOI_GPUKERNELS

	std::string IOI_GPUKERNELS_info =
		std::string("[tc1,1,0,1/tc]<b>GPU kernels initialization</b>") +
		std::string("\n[tc1,1,0,1/tc]click: change");

	ioInfo.push_back(IOI_GPUKERNELS_info, IOI_GPUKERNELS);

	//Shows materials database in use. textId is the name of the database, including the path.
	//IOI_LOCALMDB

	std::string mdbfile_info =
		std::string("[tc1,1,0,1/tc]<b>Materials database file (.mdb)</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(mdbfile_info, IOI_LOCALMDB);

	//Adaptive time step control
	//IOI_ODERELERRFAIL, IOI_ODEDTINCR, IOI_ODEDTMIN, IOI_ODEDTMAX

	std::string astep_ctrl_info =
		std::string("[tc1,1,0,1/tc]<b>Adaptive time step control</b>") +
		std::string("\n[tc1,1,0,1/tc]Fail above this error.") + 
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(astep_ctrl_info, IOI_ODERELERRFAIL);

	astep_ctrl_info =
		std::string("[tc1,1,0,1/tc]<b>Adaptive time step control</b>") +
		std::string("\n[tc1,1,0,1/tc]dT increase factor.") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(astep_ctrl_info, IOI_ODEDTINCR);

	astep_ctrl_info =
		std::string("[tc1,1,0,1/tc]<b>Adaptive time step control</b>") +
		std::string("\n[tc1,1,0,1/tc]Minimum dT.") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(astep_ctrl_info, IOI_ODEDTMIN);

	astep_ctrl_info =
		std::string("[tc1,1,0,1/tc]<b>Adaptive time step control</b>") +
		std::string("\n[tc1,1,0,1/tc]Maximum dT.") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(astep_ctrl_info, IOI_ODEDTMAX);

	//Shows PBC setting for individual demag modules. minorId is the unique mesh id number, auxId is the pbc images number (0 disables pbc) (must be ferromagnetic mesh);
	//IOI_PBC_X,
	//IOI_PBC_Y,
	//IOI_PBC_Z

	std::string pbc_info =
		std::string("[tc1,1,0,1/tc]<b>Periodic Boundary Conditions") +
		std::string("\n[tc1,1,0,1/tc]<i>Applicable for magnetization</i>") +
		std::string("\n[tc1,1,0,1/tc]left-click: set") +
		std::string("\n[tc1,1,0,1/tc]right-click: clear") + 
		std::string("\n[tc1,1,0,1/tc]double-click: edit\n");

	ioInfo.push_back(pbc_info, IOI_PBC_X);
	ioInfo.push_back(pbc_info, IOI_PBC_Y);
	ioInfo.push_back(pbc_info, IOI_PBC_Z);

	//Shows PBC setting for supermesh/multilayered demag. auxId is the pbc images number (0 disables pbc)
	//IOI_SPBC_X
	//IOI_SPBC_Y
	//IOI_SPBC_Z

	ioInfo.push_back(pbc_info, IOI_SPBC_X);
	ioInfo.push_back(pbc_info, IOI_SPBC_Y);
	ioInfo.push_back(pbc_info, IOI_SPBC_Z);

	//Shows individual shape control flag. auxId is the value (0/1)
	//IOI_INDIVIDUALSHAPE

	std::string IOI_INDIVIDUALSHAPE_info =
		std::string("[tc1,1,0,1/tc]<b>Individual shape control status flag") +
		std::string("\n[tc1,1,0,1/tc]<i>When On, shapes are applied only</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>to primary displayed quantities.</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>When Off, all primary quantities</i>") +
		std::string("\n[tc1,1,0,1/tc]<i>are modified.</i>") +
		std::string("\n[tc1,1,0,1/tc]click: change status\n");

	ioInfo.push_back(IOI_INDIVIDUALSHAPE_info, IOI_INDIVIDUALSHAPE);

	//Shows image cropping settings : textId has the DBL4 value as text
	//IOI_IMAGECROPPING

	std::string IOI_IMAGECROPPING_info =
		std::string("[tc1,1,0,1/tc]<b>Image save cropping, normalized.") +
		std::string("\n[tc1,1,0,1/tc]double-click: edit\n");

	ioInfo.push_back(IOI_IMAGECROPPING_info, IOI_IMAGECROPPING);

	//Show user constant for text equations : minorId is the index in Simulation::userConstants, auxId is the number of the interactive object in the list as it appears in the console, textId is the constant name and value std::string 
	//Note this entry must always represent the entry in Simulation::userConstants with the index in auxId.
	//IOI_USERCONSTANT

	std::string showuserconstant_info =
		std::string("[tc1,1,0,1/tc]<b>User constant</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit entry") +
		std::string("\n[tc1,1,0,1/tc]right-click: delete entry\n");

	ioInfo.push_back(showuserconstant_info, IOI_USERCONSTANT);

	//Show skypos diameter multiplier : minorId is the unique mesh id number, textId is the multiplier as a std::string
	//IOI_SKYPOSDMUL

	std::string IOI_SKYPOSDMUL_info =
		std::string("[tc1,1,0,1/tc]<b>Skypos diameter multiplier.</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit entry");

	ioInfo.push_back(IOI_SKYPOSDMUL_info, IOI_SKYPOSDMUL);

	//Shows dwpos fitting component. auxId is the value (-1, 0, 1, 2)
	//IOI_DWPOSCOMPONENT

	std::string IOI_DWPOSCOMPONENT_info =
		std::string("[tc1,1,0,1/tc]<b>dwpos fitting component.</b>") +
		std::string("\n[tc1,1,0,1/tc]click: change");

	ioInfo.push_back(IOI_DWPOSCOMPONENT_info, IOI_DWPOSCOMPONENT);

	//Shows Monte-Carlo computation type (serial/parallel) : minorId is the unique mesh id number, auxId is the status (0 : parallel, 1 : serial, -1 : N/A)
	//IOI_MCCOMPUTATION

	std::string IOI_MCCOMPUTATION_info =
		std::string("[tc1,1,0,1/tc]<b>Monte Carlo Computation Type</b>") +
		std::string("\n[tc1,1,0,1/tc]click: toggle");

	ioInfo.push_back(IOI_MCCOMPUTATION_info, IOI_MCCOMPUTATION);

	//Shows Monte-Carlo algorithm type : minorId is the unique mesh id number, auxId is the type (0 : classical, 1 : constrained, -1 : N/A), textId is the constrained DBL3 direction.
	//IOI_MCTYPE

	std::string IOI_MCTYPE_info =
		std::string("[tc1,1,0,1/tc]<b>Monte Carlo Algorithm Type</b>") +
		std::string("\n[tc1,1,0,1/tc]right-click: change") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(IOI_MCTYPE_info, IOI_MCTYPE);

	//Shows Monte-Carlo disabled/enabled status : minorId is the unique mesh id number, auxId is the status (1 : disabled, 0 : enabled).
	//IOI_MCDISABLED

	std::string IOI_MCDISABLED_info =
		std::string("[tc1,1,0,1/tc]<b>Monte Carlo Algorithm Status</b>") +
		std::string("\n[tc1,1,0,1/tc]click: change");

	ioInfo.push_back(IOI_MCDISABLED_info, IOI_MCDISABLED);

	//Shows Monte-Carlo computefields state flag : auxId is the state (0: disabled, 1: enabled)
	//IOI_MCCOMPUTEFIELDS:

	std::string IOI_MCCOMPUTEFIELDS_info =
		std::string("[tc1,1,0,1/tc]<b>Monte Carlo computefields state</b>") +
		std::string("\n[tc1,1,0,1/tc]click: toggle");

	ioInfo.push_back(IOI_MCCOMPUTEFIELDS_info, IOI_MCCOMPUTEFIELDS);

	//Shows shape rotation setting: textId is the value as text (DBL3)
	//IOI_SHAPEROT

	std::string IOI_SHAPEROT_info =
		std::string("[tc1,1,0,1/tc]<b>Shape generation modifier: rotation</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(IOI_SHAPEROT_info, IOI_SHAPEROT);

	//Shows shape repetition setting: textId is the value as text (INT3)
	//IOI_SHAPEREP

	std::string IOI_SHAPEREP_info =
		std::string("[tc1,1,0,1/tc]<b>Shape generation modifier: repetitions</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(IOI_SHAPEREP_info, IOI_SHAPEREP);

	//Shows shape displacement setting: textId is the value as text (DBL3)
	//IOI_SHAPEDSP

	std::string IOI_SHAPEDSP_info =
		std::string("[tc1,1,0,1/tc]<b>Shape generation modifier: displacement</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(IOI_SHAPEDSP_info, IOI_SHAPEDSP);

	//Shows shape method setting: textId is the value as text (method)
	//IOI_SHAPEMET

	std::string IOI_SHAPEMET_info =
		std::string("[tc1,1,0,1/tc]<b>Shape generation modifier: method</b>") +
		std::string("\n[tc1,1,0,1/tc]click: change");

	ioInfo.push_back(IOI_SHAPEMET_info, IOI_SHAPEMET);

	//Shows display render detail level: textId is the value
	//IOI_DISPRENDER_DETAIL

	std::string IOI_DISPRENDER_DETAIL_info =
		std::string("[tc1,1,0,1/tc]<b>Display detail level cellsize</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit") +
		std::string("\n[tc1,1,0,1/tc]mouse wheel: change");

	ioInfo.push_back(IOI_DISPRENDER_DETAIL_info, IOI_DISPRENDER_DETAIL);

	//Shows display render threshold 1: auxId is the value
	//IOI_DISPRENDER_THRESH1,

	std::string IOI_DISPRENDER_THRESH1_info =
		std::string("[tc1,1,0,1/tc]<b>Display threshold 1: simpler elements</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(IOI_DISPRENDER_THRESH1_info, IOI_DISPRENDER_THRESH1);

	//Shows display render threshold 2: auxId is the value
	//IOI_DISPRENDER_THRESH2,

	std::string IOI_DISPRENDER_THRESH2_info =
		std::string("[tc1,1,0,1/tc]<b>Display threshold 2: ommit obscured</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(IOI_DISPRENDER_THRESH2_info, IOI_DISPRENDER_THRESH2);

	//Shows display render threshold 3: auxId is the value
	//IOI_DISPRENDER_THRESH3

	std::string IOI_DISPRENDER_THRESH3_info =
		std::string("[tc1,1,0,1/tc]<b>Display threshold 3: checkerboard</b>") +
		std::string("\n[tc1,1,0,1/tc]dbl-click: edit");

	ioInfo.push_back(IOI_DISPRENDER_THRESH3_info, IOI_DISPRENDER_THRESH3);
}

//---------------------------------------------------- MAKE INTERACTIVE OBJECT : Auxiliary method

template <typename ... PType>
std::string Simulation::MakeIO(IOI_ identifier, PType ... params)
{
	std::vector<std::string> params_str = make_vector<std::string>(ToString(params)...);

#if GRAPHICS == 1
	//std::string objectText, IOI_ majorId, int minorId = -1, int auxId = -1, std::string textId = "", D2D1_COLOR_F bgrndCol = MESSAGECOLOR
	auto MakeInteractiveObject = [&](std::string objectText, IOI_ majorId, int minorId = -1, int auxId = -1, std::string textId = "", D2D1_COLOR_F bgrndCol = MESSAGECOLOR) -> std::string 
	{
		if (!textId.length()) textId = objectText;

		std::string newObject;

		std::string bgrndColString = ToString(bgrndCol.r) + "," + ToString(bgrndCol.g) + "," + ToString(bgrndCol.b) + "," + ToString(bgrndCol.a);

		newObject = "[io" + ToString((int)majorId) + "," + ToString(minorId) + "," + ToString((int)auxId) + "," + textId + "/io]<b>[or][tc1,1,1,1/tc][bc" + bgrndColString + "/bc] " + objectText + " </io>";

		return newObject;
	};
#else
	//std::string objectText, IOI_ majorId, int minorId = -1, int auxId = -1, std::string textId = "", D2D1_COLOR_F bgrndCol = MESSAGECOLOR
	auto MakeInteractiveObject = [&](std::string objectText, IOI_ majorId, int minorId = -1, int auxId = -1, std::string textId = "", int bgrndCol = MESSAGECOLOR) -> std::string 
	{
		return objectText;
	};
#endif

	switch (identifier) {

	case IOI_PROGRAMUPDATESTATUS:
		return MakeInteractiveObject("Checking for updates...", IOI_PROGRAMUPDATESTATUS, 0, -1, "", UNAVAILABLECOLOR);
		break;

	case IOI_FMSMESHRECTANGLE:
		return MakeInteractiveObject(ToString(SMesh.GetFMSMeshRect(), "m"), IOI_FMSMESHRECTANGLE, 0, 0, ToString(SMesh.GetFMSMeshRect(), "m"));
		break;

	case IOI_FMSMESHCELLSIZE:
		return MakeInteractiveObject(ToString(SMesh.GetFMSMeshCellsize(), "m"), IOI_FMSMESHCELLSIZE, 0, 0, ToString(SMesh.GetFMSMeshCellsize(), "m"));
		break;

	case IOI_ESMESHRECTANGLE:
		return MakeInteractiveObject(ToString(SMesh.GetESMeshRect(), "m"), IOI_ESMESHRECTANGLE, 0, 0, ToString(SMesh.GetESMeshRect(), "m"));
		break;

	case IOI_ESMESHCELLSIZE:
		return MakeInteractiveObject(ToString(SMesh.GetESMeshCellsize(), "m"), IOI_ESMESHCELLSIZE, 0, 0, ToString(SMesh.GetESMeshCellsize(), "m"));
		break;

	case IOI_MESH_FORMESHLIST:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);
			std::string meshName = SMesh().get_key_from_index(meshIndex);

			return MakeInteractiveObject(meshName, IOI_MESH_FORMESHLIST, SMesh[meshIndex]->get_id(), 1, meshName);
		}
		break;

	case IOI_MESHRECTANGLE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject(ToString(SMesh[meshIndex]->GetMeshRect(), "m"), IOI_MESHRECTANGLE, SMesh[meshIndex]->get_id(), 0, ToString(SMesh[meshIndex]->GetMeshRect(), "m"));
		}
		break;

	case IOI_MESHCELLSIZE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (SMesh[meshIndex]->MComputation_Enabled())
				return MakeInteractiveObject(ToString(SMesh[meshIndex]->GetMeshCellsize(), "m"), IOI_MESHCELLSIZE, SMesh[meshIndex]->get_id(), 1, ToString(SMesh[meshIndex]->GetMeshCellsize(), "m"), ONCOLOR);
			else
				return MakeInteractiveObject("N/A", IOI_MESHCELLSIZE, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
		}
		break;

	case IOI_MESHECELLSIZE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (SMesh[meshIndex]->EComputation_Enabled())
				return MakeInteractiveObject(ToString(SMesh[meshIndex]->GetMeshECellsize(), "m"), IOI_MESHECELLSIZE, SMesh[meshIndex]->get_id(), 1, ToString(SMesh[meshIndex]->GetMeshECellsize(), "m"), ONCOLOR);
			else
				return MakeInteractiveObject("N/A", IOI_MESHECELLSIZE, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
		}
		break;
		
	case IOI_MESHTCELLSIZE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (SMesh[meshIndex]->TComputation_Enabled())
				return MakeInteractiveObject(ToString(SMesh[meshIndex]->GetMeshTCellsize(), "m"), IOI_MESHTCELLSIZE, SMesh[meshIndex]->get_id(), 1, ToString(SMesh[meshIndex]->GetMeshTCellsize(), "m"), ONCOLOR);
			else
				return MakeInteractiveObject("N/A", IOI_MESHTCELLSIZE, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
		}
		break;
		
	case IOI_MESHMCELLSIZE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (SMesh[meshIndex]->MechComputation_Enabled())
				return MakeInteractiveObject(ToString(SMesh[meshIndex]->GetMeshMCellsize(), "m"), IOI_MESHMCELLSIZE, SMesh[meshIndex]->get_id(), 1, ToString(SMesh[meshIndex]->GetMeshMCellsize(), "m"), ONCOLOR);
			else
				return MakeInteractiveObject("N/A", IOI_MESHMCELLSIZE, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
		}
		break;

	case IOI_MESHSCELLSIZE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (SMesh[meshIndex]->MComputation_Enabled() && !SMesh[meshIndex]->is_atomistic()) {

				return MakeInteractiveObject(ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetMeshSCellsize(), "m"), IOI_MESHSCELLSIZE, SMesh[meshIndex]->get_id(), 1, ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetMeshSCellsize(), "m"), ONCOLOR);
			}
			else {

				return MakeInteractiveObject("N/A", IOI_MESHSCELLSIZE, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
			}
		}
		break;

	//Shows link stochastic flag : minorId is the unique mesh id number, auxId is the value off (0), on (1), N/A (-1)
	case IOI_LINKSTOCHASTIC:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (SMesh[meshIndex]->MComputation_Enabled() && !SMesh[meshIndex]->is_atomistic()) {

				return MakeInteractiveObject("On", IOI_LINKSTOCHASTIC, SMesh[meshIndex]->get_id(), 1);
			}
			else {

				return MakeInteractiveObject("N/A", IOI_LINKSTOCHASTIC, SMesh[meshIndex]->get_id(), -1, "N/A", OFFCOLOR);
			}
		}
		break;

	case IOI_MESHDMCELLSIZE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (SMesh[meshIndex]->MComputation_Enabled() && SMesh[meshIndex]->is_atomistic()) {

				return MakeInteractiveObject(ToString(dynamic_cast<Atom_Mesh*>(SMesh[meshIndex])->Get_Demag_Cellsize(), "m"), IOI_MESHDMCELLSIZE, SMesh[meshIndex]->get_id(), 1, ToString(dynamic_cast<Atom_Mesh*>(SMesh[meshIndex])->Get_Demag_Cellsize(), "m"), ONCOLOR);
			}
			else {

				return MakeInteractiveObject("N/A", IOI_MESHDMCELLSIZE, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
			}
		}
		break;

	//Shows evaluation speedup type: auxId is the type value.
	case IOI_SPEEDUPMODE:
		return MakeInteractiveObject(" None ", IOI_SPEEDUPMODE, 0, 0);
		break;

	case IOI_SMESHDISPLAY:
		if (params_str.size() == 1) {

			int displayOption = ToNum(params_str[0]);

			return MakeInteractiveObject(displayHandles(displayOption), IOI_SMESHDISPLAY, displayOption, -1, displayHandles(displayOption));
		}
		break;

	case IOI_MESHDISPLAY:
		if (params_str.size() == 2) {

			int meshIndex = ToNum(params_str[0]);
			int displayOption = ToNum(params_str[1]);

			return MakeInteractiveObject(displayHandles(displayOption), IOI_MESHDISPLAY, displayOption, SMesh[meshIndex]->get_id(), displayHandles(displayOption));
		}
		break;

	case IOI_MESHDISPLAYTRANSPARENCY:
		if (params_str.size() == 1) {

			return MakeInteractiveObject(params_str[0], IOI_MESHDISPLAYTRANSPARENCY, -1, -1, params_str[0]);
		}
		break;

	case IOI_MESHDISPLAYTHRESHOLDS:
		if (params_str.size() == 1) {

			return MakeInteractiveObject(params_str[0], IOI_MESHDISPLAYTHRESHOLDS, -1, -1, params_str[0]);
		}
		break;

	case IOI_MESHDISPLAYTHRESHOLDTRIG:
		return MakeInteractiveObject("Z", IOI_MESHDISPLAYTHRESHOLDTRIG, -1, (int)VEC3REP_Z);
		break;

	case IOI_MESHVECREP:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject("full", IOI_MESHVECREP, SMesh[meshIndex]->get_id(), 0);
		}
		break;

	case IOI_SMESHVECREP:
		return MakeInteractiveObject("full", IOI_SMESHVECREP, 0, 0);
		break;

	case IOI_MESH_FORDISPLAYOPTIONS:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			std::string meshName = SMesh().get_key_from_index(meshIndex);

			return MakeInteractiveObject(meshName, IOI_MESH_FORDISPLAYOPTIONS, SMesh[meshIndex]->get_id(), 1, meshName);
		}
		break;

	case IOI_SMODULE:
		if (params_str.size() == 1) {

			int moduleID = ToNum(params_str[0]);

			return MakeInteractiveObject(moduleHandles(moduleID), IOI_SMODULE, moduleID);
		}
		break;

	case IOI_MODULE:
		if (params_str.size() == 2) {

			int meshIndex = ToNum(params_str[0]);
			int moduleID = ToNum(params_str[1]);

			std::string meshName = SMesh().get_key_from_index(meshIndex);

			return MakeInteractiveObject(moduleHandles(moduleID), IOI_MODULE, moduleID, SMesh[meshIndex]->get_id());
		}
		break;

	case IOI_DISPLAYMODULE:
		if (params_str.size() == 2) {

			int meshIndex = ToNum(params_str[0]);
			int moduleID = ToNum(params_str[1]);

			std::string meshName = SMesh().get_key_from_index(meshIndex);

			return MakeInteractiveObject(moduleHandles(moduleID), IOI_DISPLAYMODULE, moduleID, SMesh[meshIndex]->get_id(), "0");
		}
		break;

	case IOI_MESH_FORMODULES:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			std::string meshName = SMesh().get_key_from_index(meshIndex);

			return MakeInteractiveObject(meshName, IOI_MESH_FORMODULES, SMesh[meshIndex]->get_id(), 1, meshName);
		}
		break;

	case IOI_MESH_FORDISPLAYMODULES:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			std::string meshName = SMesh().get_key_from_index(meshIndex);

			return MakeInteractiveObject(meshName, IOI_MESH_FORDISPLAYMODULES, SMesh[meshIndex]->get_id(), 1, meshName);
		}
		break;

	case IOI_ODE:
		if (params_str.size() == 1) {

			int odeId = ToNum(params_str[0]);

			std::string odeHandle = odeHandles(odeId);

			return MakeInteractiveObject(odeHandle, IOI_ODE, odeId);
		}
		break;

	case IOI_ATOMODE:
		if (params_str.size() == 1) {

			int odeId = ToNum(params_str[0]);

			std::string atom_odeHandle = atom_odeHandles(odeId);

			return MakeInteractiveObject(atom_odeHandle, IOI_ATOMODE, odeId);
		}
		break;

	case IOI_ODEDT:
		return MakeInteractiveObject("0", IOI_ODEDT, 0, 0, "0");
		break;

	case IOI_HEATDT:
		return MakeInteractiveObject("0", IOI_HEATDT, 0, 0, "0");
		break;

	case IOI_ELDT:
		return MakeInteractiveObject("0", IOI_ELDT, 0, 0, "0");
		break;

	case IOI_LINKELDT:
		return MakeInteractiveObject("On", IOI_LINKELDT, 0, 1);
		break;

	case IOI_STOCHDT:
		return MakeInteractiveObject("0", IOI_STOCHDT, 0, 0, "0");
		break;

	case IOI_LINKSTOCHDT:
		return MakeInteractiveObject("On", IOI_LINKSTOCHDT, 0, 1);
		break;

	case IOI_SPEEDUPDT:
		return MakeInteractiveObject("0", IOI_SPEEDUPDT, 0, 0, "0");
		break;

	case IOI_LINKSPEEDUPDT:
		return MakeInteractiveObject("On", IOI_LINKSPEEDUPDT, 0, 1);
		break;

	case IOI_ODE_EVAL:
		if (params_str.size() == 1) {

			int evalId = ToNum(params_str[0]);

			std::string evalHandle = odeEvalHandles(evalId);

			//mark the set ode with ODE_ERROR, so the state handler will be forced to update the console object with correct state and color
			return MakeInteractiveObject(evalHandle, IOI_ODE_EVAL, ODE_ERROR, evalId, evalHandle);
		}
		break;

	case IOI_SHOWDATA:
		if (params_str.size() == 1) {

			int dataID = ToNum(params_str[0]);

			std::string dataName = dataDescriptor.get_key_from_ID(dataID);

			return MakeInteractiveObject(dataName, IOI_SHOWDATA, dataID, 0, dataName, UNAVAILABLECOLOR);
		}
		break;

	case IOI_DATA:
		if (params_str.size() == 1) {

			int dataID = ToNum(params_str[0]);

			std::string dataName = dataDescriptor.get_key_from_ID(dataID);

			return MakeInteractiveObject(dataName, IOI_DATA, dataID, 0, dataName, UNAVAILABLECOLOR);
		}
		break;

	case IOI_DIRECTORY:
		if (params_str.size() == 1) {

			std::string directory = params_str[0];

			return MakeInteractiveObject(directory, IOI_DIRECTORY, 0, 0, directory);
		}
		break;

	case IOI_SAVEDATAFILE:
		if (params_str.size() == 1) {

			std::string savedataFile = params_str[0];

			return MakeInteractiveObject(savedataFile, IOI_SAVEDATAFILE, 0, 0, savedataFile);
		}
		break;

	case IOI_SAVEDATAFLAG:
		return MakeInteractiveObject("On", IOI_SAVEDATAFLAG);
		break;

	case IOI_SAVEIMAGEFILEBASE:
		if (params_str.size() == 1) {

			std::string imageSaveFileBase = params_str[0];

			return MakeInteractiveObject(imageSaveFileBase, IOI_SAVEIMAGEFILEBASE, 0, 0, imageSaveFileBase);
		}
		break;

	case IOI_SAVEIMAGEFLAG:
		return MakeInteractiveObject("Off", IOI_SAVEIMAGEFLAG);
		break;

	case IOI_OUTDATA:
		if (params_str.size() == 1) {

			int index_in_list = ToNum(params_str[0]);

			std::string outputdata_text = Build_SetOutputData_Text(index_in_list);

			return MakeInteractiveObject(outputdata_text, IOI_OUTDATA, saveDataList.get_id_from_index(index_in_list).minor, index_in_list, outputdata_text);
		}
		break;

	case IOI_STAGE:
		if (params_str.size() == 1) {

			int stageID = ToNum(params_str[0]);

			std::string stagetype_text = stageDescriptors.get_key_from_ID(stageID);

			return MakeInteractiveObject(stagetype_text, IOI_STAGE, stageID, 0, stagetype_text, UNAVAILABLECOLOR);
		}
		break;

	case IOI_SETSTAGE:
		if (params_str.size() == 1) {

			int index_in_list = ToNum(params_str[0]);

			std::string outputdata_text = Build_SetStages_Text(index_in_list);

			return MakeInteractiveObject(outputdata_text, IOI_SETSTAGE, simStages.get_id_from_index(index_in_list).minor, index_in_list, outputdata_text);
		}
		break;

	case IOI_SETSTAGEVALUE:
		if (params_str.size() == 1) {

			int index_in_list = ToNum(params_str[0]);

			std::string setstage_text = simStages[index_in_list].get_value_string();
			
			return MakeInteractiveObject(setstage_text, IOI_SETSTAGEVALUE, simStages.get_id_from_index(index_in_list).minor, index_in_list, setstage_text);
		}
		break;

	case IOI_STAGESTOPCONDITION:
		if (params_str.size() == 1) {

			int index_in_list = ToNum(params_str[0]);

			std::string stopcondition_text = Build_SetStages_StopConditionText(index_in_list);

			return MakeInteractiveObject(stopcondition_text, IOI_STAGESTOPCONDITION, simStages.get_id_from_index(index_in_list).minor, index_in_list, stopcondition_text);
		}
		break;

	case IOI_DSAVETYPE:
		if (params_str.size() == 2) {

			int index_in_list = ToNum(params_str[0]);
			int dataSaveID = ToNum(params_str[1]);

			int dsaveIdx = dataSaveDescriptors.get_index_from_ID(dataSaveID);
			//std::string savecondition_text = Build_SetStages_SaveConditionText(index_in_list, dsaveIdx);
			std::string savecondition_text = dataSaveDescriptors.get_key_from_ID(dataSaveID);

			return MakeInteractiveObject(savecondition_text, IOI_DSAVETYPE, simStages.get_id_from_index(index_in_list).minor, dataSaveID, savecondition_text);
		}
		break;

	case IOI_STAGESTOPCONDITIONALL:
		if (params_str.size() == 1) {

			int stopID = ToNum(params_str[0]);

			std::string stopcondition_text = stageStopDescriptors.get_key_from_ID(stopID);

			return MakeInteractiveObject(stopcondition_text, IOI_STAGESTOPCONDITIONALL, stopID, 0, stopcondition_text, UNAVAILABLECOLOR);
		}
		break;

	case IOI_DSAVETYPEALL:
		if (params_str.size() == 1) {

			int dsaveID = ToNum(params_str[0]);

			std::string savetype_text = dataSaveDescriptors.get_key_from_ID(dsaveID);

			return MakeInteractiveObject(savetype_text, IOI_DSAVETYPEALL, dsaveID, 0, savetype_text, UNAVAILABLECOLOR);
		}
		break;

	case IOI_MESHPARAM:
		if (params_str.size() == 2) {

			int meshIndex = ToNum(params_str[0]);
			int paramId = ToNum(params_str[1]);

			std::string meshParam_text = Build_MeshParams_Text(meshIndex, (PARAM_)paramId);

			return MakeInteractiveObject(meshParam_text, IOI_MESHPARAM, paramId, SMesh[meshIndex]->get_id(), SMesh().get_key_from_index(meshIndex) + std::string("\t") + meshParam_text);
		}
		break;

	case IOI_MESH_FORPARAMS:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			int meshId = SMesh[meshIndex]->get_id();

			return MakeInteractiveObject(SMesh.key_from_meshId(meshId), IOI_MESH_FORPARAMS, meshId, 0, SMesh.key_from_meshId(meshId));
		}
		break;

	case IOI_MESHPARAMTEMP:
		if (params_str.size() == 2) {

			int meshIndex = ToNum(params_str[0]);
			int paramId = ToNum(params_str[1]);

			std::string tempDescriptor_text = SMesh[meshIndex]->get_paraminfo_string((PARAM_)paramId);

			return MakeInteractiveObject(tempDescriptor_text, IOI_MESHPARAMTEMP, paramId, SMesh[meshIndex]->get_id(), SMesh().get_key_from_index(meshIndex) + std::string("\t") + tempDescriptor_text, (SMesh[meshIndex]->is_paramtemp_set((PARAM_)paramId) ? ONCOLOR : OFFCOLOR));
		}
		break;

	case IOI_MESHPARAMVAR:
		if (params_str.size() == 2) {

			int meshIndex = ToNum(params_str[0]);
			int paramId = ToNum(params_str[1]);

			std::string varDescriptor_text = SMesh[meshIndex]->get_paramvarinfo_string((PARAM_)paramId);

			return MakeInteractiveObject(varDescriptor_text, IOI_MESHPARAMVAR, paramId, SMesh[meshIndex]->get_id(), SMesh().get_key_from_index(meshIndex) + std::string("\t") + varDescriptor_text, (SMesh[meshIndex]->is_paramvar_set((PARAM_)paramId) ? ONCOLOR : OFFCOLOR));
		}
		break;

	case IOI_MESHPARAMTEMPTYPE:
		if (params_str.size() == 1) {

			int dependenceID = ToNum(params_str[0]);
			
			std::string type = temperature_dependence_type(dependenceID);

			return MakeInteractiveObject(type, IOI_MESHPARAMTEMPTYPE, dependenceID, 0, type);
		}
		break;

	case IOI_MESHPARAMVARGENERATOR:
		if (params_str.size() == 1) {

			int generatorID = ToNum(params_str[0]);

			std::string generatorName = vargenerator_descriptor.get_key_from_ID(generatorID);

			return MakeInteractiveObject(generatorName, IOI_MESHPARAMVARGENERATOR, generatorID, 0, generatorName);
		}
		break;

	case IOI_MESH_FORPARAMSTEMP:
	case IOI_MESH_FORPARAMSVAR:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			std::string meshName = SMesh.key_from_meshIdx(meshIndex);

			return MakeInteractiveObject(meshName, identifier, SMesh[meshIndex]->get_id(), 0, meshName);
		}
		break;

	case IOI_MESH_FORTEMPERATURE:
	case IOI_MESH_FORCURIEANDMOMENT:
	case IOI_MESH_FORTMODEL:
	case IOI_MESH_FORPBC:
	case IOI_MESH_FOREXCHCOUPLING:
	case IOI_MESH_FORSTOCHASTICITY:
	case IOI_MESH_FORELASTICITY:
	case IOI_MESH_FORSPEEDUP:
	case IOI_MESH_FORSKYPOSDMUL:
	case IOI_MESH_FORMC:
	case IOI_MESH_FORDIPOLESHIFT:
	case IOI_MESH_FORTMR:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);
			std::string meshName = SMesh().get_key_from_index(meshIndex);

			return MakeInteractiveObject(meshName, identifier, SMesh[meshIndex]->get_id(), 1, meshName);
		}
		break;

	case IOI_MOVINGMESH:
		if (SMesh.IsMovingMeshSet()) {

			int meshId = SMesh.GetId_of_MoveMeshTrigger();
			std::string meshName = SMesh.key_from_meshId(meshId);

			return MakeInteractiveObject(meshName, IOI_MOVINGMESH, meshId, true, meshName, ONCOLOR);
		}
		else {

			return MakeInteractiveObject("None", IOI_MOVINGMESH, -1, false, "", OFFCOLOR);
		}
		break;

	case IOI_MOVINGMESHASYM:
		return MakeInteractiveObject("Antisymmetric", IOI_MOVINGMESHASYM, 0, true);
		break;

	case IOI_MOVINGMESHTHRESH:
		return MakeInteractiveObject("0", IOI_MOVINGMESHTHRESH, 0, 0, "0");
		break;

	case IOI_CONSTANTCURRENTSOURCE:
		return MakeInteractiveObject(" ", IOI_CONSTANTCURRENTSOURCE, -1, !SMesh.CallModuleMethod(&STransport::UsingConstantCurrentSource));
		break;

	case IOI_ELECTRODERECT:
		if (params_str.size() == 1) {

			int el_index = ToNum(params_str[0]);

			std::pair<Rect, double> elInfo = SMesh.CallModuleMethod(&STransport::GetElectrodeInfo, el_index);
			int electrode_id = SMesh.CallModuleMethod(&STransport::GetElectrodeid, el_index);

			return MakeInteractiveObject(ToString(elInfo.first, "m"), IOI_ELECTRODERECT, electrode_id, el_index, ToString(elInfo.first, "m"));
		}
		break;

	case IOI_ELECTRODEPOTENTIAL:
		if (params_str.size() == 1) {

			int el_index = ToNum(params_str[0]);

			std::pair<Rect, double> elInfo = SMesh.CallModuleMethod(&STransport::GetElectrodeInfo, el_index);

			return MakeInteractiveObject(ToString(elInfo.second, "V"), IOI_ELECTRODEPOTENTIAL, el_index, 0, ToString(elInfo.second, "V"));
		}
		break;

	case IOI_ELECTRODEGROUND:
		if (params_str.size() == 1) {

			int el_index = ToNum(params_str[0]);

			if (SMesh.CallModuleMethod(&STransport::IsGroundElectrode, el_index)) {

				return MakeInteractiveObject("Grnd", IOI_ELECTRODEGROUND, el_index, true, "", ONCOLOR);
			}
			else {

				return MakeInteractiveObject("Grnd", IOI_ELECTRODEGROUND, el_index, false, "", OFFCOLOR);
			}
		}
		break;

	case IOI_TSOLVERCONVERROR:
	{
		double convergence_error = SMesh.CallModuleMethod(&STransport::GetConvergenceError);

		return MakeInteractiveObject(ToString(convergence_error), IOI_TSOLVERCONVERROR);
	}
	break;

	case IOI_TSOLVERTIMEOUT:
	{
		int iters_timeout = SMesh.CallModuleMethod(&STransport::GetConvergenceTimeout);

		return MakeInteractiveObject(ToString(iters_timeout), IOI_TSOLVERTIMEOUT);
	}
	break;

	case IOI_SSOLVERCONVERROR:
	{
		double convergence_error = SMesh.CallModuleMethod(&STransport::GetSConvergenceError);

		return MakeInteractiveObject(ToString(convergence_error), IOI_SSOLVERCONVERROR);
	}
	break;

	case IOI_SSOLVERTIMEOUT:
	{
		int iters_timeout = SMesh.CallModuleMethod(&STransport::GetSConvergenceTimeout);

		return MakeInteractiveObject(ToString(iters_timeout), IOI_SSOLVERTIMEOUT);
	}
	break;

	case IOI_SORDAMPING:
	{
		DBL2 fixed_SOR_damping = SMesh.CallModuleMethod(&STransport::GetSORDamping);

		return MakeInteractiveObject(ToString(fixed_SOR_damping), IOI_SORDAMPING, -1, -1, ToString(fixed_SOR_damping));
	}
	break;

	case IOI_TMRTYPE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);
			return MakeInteractiveObject(" ", IOI_TMRTYPE, SMesh[meshIndex]->get_id(), -1);
		}
		break;

	case IOI_STATICTRANSPORT:
	{
		if (static_transport_solver) return MakeInteractiveObject("On", IOI_STATICTRANSPORT, 0, 1, "", ONCOLOR);
		else return MakeInteractiveObject("Off", IOI_STATICTRANSPORT, 0, 0, "", OFFCOLOR);
	}
	break;

	case IOI_DISABLEDTRANSPORT:
	{
		if (disabled_transport_solver) return MakeInteractiveObject("Disabled", IOI_DISABLEDTRANSPORT, 0, 1, "", OFFCOLOR);
		else return MakeInteractiveObject("Enabled", IOI_DISABLEDTRANSPORT, 0, 0, "", OFFCOLOR);
	}
	break;

	case IOI_BASETEMPERATURE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject(ToString(SMesh[meshIndex]->GetAverageTemperature(), "K"), IOI_BASETEMPERATURE, SMesh[meshIndex]->get_id(), -1, ToString(SMesh[meshIndex]->GetAverageTemperature(), "K"));
		}
		break;

	case IOI_AMBIENT_TEMPERATURE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			double T_ambient = SMesh[meshIndex]->CallModuleMethod(&HeatBase::GetAmbientTemperature);

			return MakeInteractiveObject(ToString(T_ambient, "K"), IOI_AMBIENT_TEMPERATURE, SMesh[meshIndex]->get_id(), 1, ToString(T_ambient, "K"));
		}
		break;

	case IOI_ROBIN_ALPHA:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			double alpha_boundary = SMesh[meshIndex]->CallModuleMethod(&HeatBase::GetAlphaBoundary);

			return MakeInteractiveObject(ToString(alpha_boundary, "W/m2K"), IOI_ROBIN_ALPHA, SMesh[meshIndex]->get_id(), 1, ToString(alpha_boundary, "W/m2K"));
		}
		break;

	case IOI_INSULATINGSIDE:
		if (params_str.size() == 2) {

			int meshIndex = ToNum(params_str[0]);
			std::string side_literal = params_str[1];

			return MakeInteractiveObject(side_literal + ": No", IOI_INSULATINGSIDE, SMesh[meshIndex]->get_id(), 0, side_literal);
		}
		break;

	case IOI_MESH_FORHEATBOUNDARIES:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);
			std::string meshName = SMesh().get_key_from_index(meshIndex);

			return MakeInteractiveObject(meshName, IOI_MESH_FORHEATBOUNDARIES, SMesh[meshName]->get_id(), 1, meshName);
		}
		break;

	case IOI_CURIETEMP:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (!SMesh[meshIndex]->is_atomistic()) {

				return MakeInteractiveObject(ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetCurieTemperature(), "K"), IOI_CURIETEMP, SMesh[meshIndex]->get_id(), 1, ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetCurieTemperature(), "K"));
			}
			else {

				return MakeInteractiveObject("N/A", IOI_CURIETEMP, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
			}
		}
		break;

	case IOI_CURIETEMPMATERIAL:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (!SMesh[meshIndex]->is_atomistic()) {

				return MakeInteractiveObject(ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetCurieTemperatureMaterial(), "K"), IOI_CURIETEMPMATERIAL, SMesh[meshIndex]->get_id(), 1, ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetCurieTemperatureMaterial(), "K"));
			}
			else {

				return MakeInteractiveObject("N/A", IOI_CURIETEMPMATERIAL, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
			}
		}
		break;

	case IOI_ATOMICMOMENT:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (!SMesh[meshIndex]->is_atomistic()) {

				return MakeInteractiveObject(ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetAtomicMoment(), "uB"), IOI_ATOMICMOMENT, SMesh[meshIndex]->get_id(), 1, ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetAtomicMoment(), "uB"));
			}
			else {

				return MakeInteractiveObject("N/A", IOI_ATOMICMOMENT, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
			}
		}
		break;

	case IOI_ATOMICMOMENT_AFM:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (!SMesh[meshIndex]->is_atomistic()) {

				return MakeInteractiveObject(ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetAtomicMoment_AFM(), "uB"), IOI_ATOMICMOMENT_AFM, SMesh[meshIndex]->get_id(), 1, ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetAtomicMoment_AFM(), "uB"));
			}
			else {

				return MakeInteractiveObject("N/A", IOI_ATOMICMOMENT_AFM, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
			}
		}
		break;

	case IOI_TAU:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (!SMesh[meshIndex]->is_atomistic()) {

				return MakeInteractiveObject(ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetTcCoupling()), IOI_TAU, SMesh[meshIndex]->get_id(), 1, ToString(dynamic_cast<Mesh*>(SMesh[meshIndex])->GetTcCoupling()));
			}
			else {

				return MakeInteractiveObject("N/A", IOI_TAU, SMesh[meshIndex]->get_id(), 0, "N/A", OFFCOLOR);
			}
		}
		break;

	case IOI_TMODEL:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject(" 1TM ", IOI_TMODEL, SMesh[meshIndex]->get_id(), 1);
		}
		break;

	//Shows cuda enabled/disabled or n/a state. auxId is enabled (1)/disabled(0)/not available(-1) status.
	case IOI_CUDASTATE:
		if (params_str.size() == 1) {

			int status = ToNum(params_str[0]);

			if(!cudaAvailable) return MakeInteractiveObject("N/A", IOI_CUDASTATE, -1, -1, "", UNAVAILABLECOLOR);
			else {

				if(status == 0) return MakeInteractiveObject("Off", IOI_CUDASTATE, -1, 0, "", OFFCOLOR);
				else return MakeInteractiveObject("On", IOI_CUDASTATE, -1, 1, "", ONCOLOR);
			}
		}
		break;

	//Shows CUDA device information and state. minorId is the device number (from 1 up), auxId is enabled (1)/disabled(0)/not available(-1) status. 
	case IOI_CUDADEVICE:
		if (params_str.size() == 1) {

			int device = ToNum(params_str[0]);

			if (!cudaAvailable) return MakeInteractiveObject("N/A", IOI_CUDADEVICE, device, -1, "", UNAVAILABLECOLOR);
#if COMPILECUDA == 1
			else {

				if (cudaDeviceVersions[device].first != __CUDA_ARCH__) {

					return MakeInteractiveObject(cudaDeviceVersions[device].second, IOI_CUDADEVICE, device, -1, "", UNAVAILABLECOLOR);
				}
				else {

					return MakeInteractiveObject(cudaDeviceVersions[device].second, IOI_CUDADEVICE, device, 0, "", OFFCOLOR);
				}
			}
#else
			return MakeInteractiveObject("N/A", IOI_CUDADEVICE, device, -1, "", UNAVAILABLECOLOR);
#endif
		}
		break;

	//Shows gpu free memory. auxId is the value
	case IOI_GPUMEMFREE:
		if (params_str.size() == 1) {

			size_t mem_size = ToNum(params_str[0]);

			return MakeInteractiveObject(params_str[0], IOI_GPUMEMFREE, mem_size);
		}
		break;

	//Shows gpu total memory. auxId is the value
	case IOI_GPUMEMTOTAL:
		if (params_str.size() == 1) {

			size_t mem_size = ToNum(params_str[0]);

			return MakeInteractiveObject(params_str[0], IOI_GPUMEMTOTAL, mem_size);
		}
		break;

	//Shows cpu free memory. auxId is the value
	case IOI_CPUMEMFREE:
		if (params_str.size() == 1) {

			size_t mem_size = ToNum(params_str[0]);

			return MakeInteractiveObject(params_str[0], IOI_CPUMEMFREE, mem_size);
		}
		break;

	//Shows cpu total memory. auxId is the value
	case IOI_CPUMEMTOTAL:
		if (params_str.size() == 1) {

			size_t mem_size = ToNum(params_str[0]);

			return MakeInteractiveObject(params_str[0], IOI_CPUMEMTOTAL, mem_size);
		}
		break;
		
	//Shows scale_rects enabled/disabled state. auxId is enabled (1)/disabled(0) status.
	case IOI_SCALERECTSSTATUS:
		if (params_str.size() == 1) {

			int status = ToNum(params_str[0]);

			if (status == 0) return MakeInteractiveObject("Off", IOI_SCALERECTSSTATUS, -1, 0, "", OFFCOLOR);
			else return MakeInteractiveObject("On", IOI_SCALERECTSSTATUS, -1, 1, "", ONCOLOR);
		}
		break;

	//Shows coupled_to_dipoles enabled/disabled state. auxId is enabled (1)/disabled(0) status.
	case IOI_COUPLEDTODIPOLESSTATUS:
		if (params_str.size() == 1) {

			int status = ToNum(params_str[0]);

			if (status == 0) return MakeInteractiveObject("Off", IOI_COUPLEDTODIPOLESSTATUS, -1, 0, "", OFFCOLOR);
			else return MakeInteractiveObject("On", IOI_COUPLEDTODIPOLESSTATUS, -1, 1, "", ONCOLOR);
		}
		break;

	//Shows dipole velocity value. minorId is the unique mesh id number. textId is the value
	case IOI_DIPOLEVELOCITY:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject(" ", IOI_DIPOLEVELOCITY, SMesh[meshIndex]->get_id(), 0, "", ONCOLOR);
		}
		break;

	//Shows diagonal strain set equation. minorId is the unique mesh id number. textId is the equation. auxId is enabled(1)/disabled(0) status.
	case IOI_STRAINEQUATION:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject(" None ", IOI_STRAINEQUATION, SMesh[meshIndex]->get_id(), 0, "", OFFCOLOR);
		}
		break;

	//Shows shear strain set equation. minorId is the unique mesh id number. textId is the equation. auxId is enabled(1)/disabled(0) status.
	case IOI_SHEARSTRAINEQUATION:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject(" None ", IOI_SHEARSTRAINEQUATION, SMesh[meshIndex]->get_id(), 0, "", OFFCOLOR);
		}
		break;

	case IOI_SURFACEFIX:
		if (params_str.size() == 1) {

			int surface_index = ToNum(params_str[0]);

			Rect rect = SMesh.CallModuleMethod(&SMElastic::Get_Fixed_Surface, surface_index);
			int surface_id = SMesh.CallModuleMethod(&SMElastic::Get_Fixed_Surface_id, surface_index);

			return MakeInteractiveObject(ToString(rect, "m"), IOI_SURFACEFIX, surface_id, surface_index, ToString(rect, "m"));
		}
		break;

	case IOI_SURFACESTRESS:
		if (params_str.size() == 1) {

			int surface_index = ToNum(params_str[0]);

			Rect rect = SMesh.CallModuleMethod(&SMElastic::Get_Stress_Surface, surface_index);
			int surface_id = SMesh.CallModuleMethod(&SMElastic::Get_Stress_Surface_id, surface_index);

			return MakeInteractiveObject(ToString(rect, "m"), IOI_SURFACESTRESS, surface_id, surface_index, ToString(rect, "m"));
		}
		break;

	case IOI_SURFACESTRESSEQ:
		if (params_str.size() == 1) {

			int surface_index = ToNum(params_str[0]);

			std::string equation = SMesh.CallModuleMethod(&SMElastic::Get_Stress_Surface_Equation, surface_index);

			return MakeInteractiveObject(equation, IOI_SURFACESTRESSEQ, surface_index, 0, equation);
		}
		break;

	//Shows dipole shift clipping value. minorId is the unique mesh id number. textId is the value
	case IOI_DIPOLESHIFTCLIP:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject(" ", IOI_DIPOLESHIFTCLIP, SMesh[meshIndex]->get_id(), 0, "", ONCOLOR);
		}
		break;

	//Shows log_errors enabled/disabled state. auxId is enabled (1)/disabled(0) status.
	case IOI_ERRORLOGSTATUS:
		if (params_str.size() == 1) {

			int status = ToNum(params_str[0]);

			if (status == 0) return MakeInteractiveObject("Off", IOI_ERRORLOGSTATUS, -1, 0, "", OFFCOLOR);
			else return MakeInteractiveObject("On", IOI_ERRORLOGSTATUS, -1, 1, "", ONCOLOR);
		}
		break;

	//Shows start_check_updates enabled/disabled state. auxId is enabled (1)/disabled(0) status.
	case IOI_UPDATESTATUSCHECKSTARTUP:
		if (params_str.size() == 1) {

			int status = ToNum(params_str[0]);

			if (status == 0) return MakeInteractiveObject("Off", IOI_UPDATESTATUSCHECKSTARTUP, -1, 0, "", OFFCOLOR);
			else return MakeInteractiveObject("On", IOI_UPDATESTATUSCHECKSTARTUP, -1, 1, "", ONCOLOR);
		}
		break;

	//Shows start_scriptserver enabled/disabled state. auxId is enabled (1)/disabled(0) status.
	case IOI_SCRIPTSERVERSTARTUP:
		if (params_str.size() == 1) {

			int status = ToNum(params_str[0]);

			if (status == 0) return MakeInteractiveObject("Off", IOI_SCRIPTSERVERSTARTUP, -1, 0, "", OFFCOLOR);
			else return MakeInteractiveObject("On", IOI_SCRIPTSERVERSTARTUP, -1, 1, "", ONCOLOR);
		}
		break;

	//Shows number of threads. auxId is the value.
	case IOI_THREADS:
		if (params_str.size() == 1) {

			int threads = ToNum(params_str[0]);

			return MakeInteractiveObject(ToString(threads), IOI_THREADS, -1, threads);
		}
		break;

	//Shows server port. auxId is the value.
	case IOI_SERVERPORT:
		if (params_str.size() == 1) {

			std::string port = params_str[0];

			return MakeInteractiveObject(port, IOI_SERVERPORT, -1, ToNum(port));
		}
		break;

	//Shows server password. textId is the password.
	case IOI_SERVERPWD:
		if (params_str.size() == 1) {

			std::string password = params_str[0];

			//for display only
			if (!password.length()) return MakeInteractiveObject(" ", IOI_SERVERPWD, -1, 0, "");
			else return MakeInteractiveObject(password, IOI_SERVERPWD, -1, 0, password);
		}
		break;

	//Shows server sleep time in ms. auxId is the value.
	case IOI_SERVERSLEEPMS:
		if (params_str.size() == 1) {

			int sleepms = ToNum(params_str[0]);

			return MakeInteractiveObject(ToString(sleepms), IOI_SERVERSLEEPMS, -1, sleepms);
		}
		break;

	case IOI_MESHEXCHCOUPLING:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			if (SMesh[meshIndex]->MComputation_Enabled()) {

				return MakeInteractiveObject("Off", IOI_MESHEXCHCOUPLING, SMesh[meshIndex]->get_id(), 0, "", OFFCOLOR);
			}
			else {

				return MakeInteractiveObject("N/A", IOI_MESHEXCHCOUPLING, SMesh[meshIndex]->get_id(), -1, "", UNAVAILABLECOLOR);
			}
		}
		break;

	//Shows mesh roughness refinement value. minorId is the unique mesh id number, auxId is enabled (1)/disabled(0) status. textId is the value
	case IOI_REFINEROUGHNESS:
		if (params_str.size() == 1) {

			std::string meshName = params_str[0];

			if (SMesh[meshName]->IsModuleSet(MOD_ROUGHNESS))
				return MakeInteractiveObject(ToString(SMesh[meshName]->CallModuleMethod(&Roughness::get_refine)), IOI_REFINEROUGHNESS, SMesh[meshName]->get_id(), 1, ToString(SMesh[meshName]->CallModuleMethod(&Roughness::get_refine)), ONCOLOR);
			else 
				return MakeInteractiveObject("N/A", IOI_REFINEROUGHNESS, SMesh[meshName]->get_id(), 0, "N/A", UNAVAILABLECOLOR);
		}
		break;

	//Shows status of multi-layered convolution. auxId is the status (-1 : N/A, 0 : Off, 1 : On)
	case IOI_MULTICONV:
		if (params_str.size() == 1) {

			return MakeInteractiveObject("N/A", IOI_MULTICONV, 0, -1, "", UNAVAILABLECOLOR);
		}
		break;

	//Shows status of gpu kernels demag initialization. auxId is the status (0 : Off, 1 : On)
	case IOI_GPUKERNELS:
		if (params_str.size() == 1) {

			int status = ToNum(params_str[0]);

			if (status) return MakeInteractiveObject("On", IOI_GPUKERNELS, 0, status, "", ONCOLOR);
			else return MakeInteractiveObject("Off", IOI_GPUKERNELS, 0, status, "", OFFCOLOR);
		}
		break;

	//Shows status of force 2D multi-layered convolution. auxId is the status (-1 : N/A, 0 : Off, 1 : On)
	case IOI_2DMULTICONV:
		if (params_str.size() == 1) {

			return MakeInteractiveObject("N/A", IOI_2DMULTICONV, 0, -1, "", UNAVAILABLECOLOR);
		}
		break;

	//Shows status of use default n for multi-layered convolution. auxId is the status (-1 : N/A, 0 : Off, 1 : On)
	case IOI_NCOMMONSTATUS:
		if (params_str.size() == 1) {

			return MakeInteractiveObject("N/A", IOI_NCOMMONSTATUS, 0, -1, "", UNAVAILABLECOLOR);
		}
		break;

	//Shows n_common for multi-layered convolution. auxId is the status (-1 : N/A, otherwise available). textId is the value as a SZ3.
	case IOI_NCOMMON:
		if (params_str.size() == 1) {

			return MakeInteractiveObject("N/A", IOI_NCOMMON, 0, -1, "0, 0, 0", UNAVAILABLECOLOR);
		}
		break;

	case IOI_LOCALMDB:
		if (params_str.size() == 1) {

			std::string mdbFile = params_str[0];

			return MakeInteractiveObject(mdbFile, IOI_LOCALMDB, 0, 0, mdbFile);
		}
		break;

	case IOI_ODERELERRFAIL:
		if (params_str.size() == 1) {

			std::string value = params_str[0];

			return MakeInteractiveObject(value, IOI_ODERELERRFAIL, 0, 0, value);
		}
		break;

	case IOI_ODEDTINCR:
		if (params_str.size() == 1) {

			std::string value = params_str[0];

			return MakeInteractiveObject(value, IOI_ODEDTINCR, 0, 0, value);
		}
		break;

	case IOI_ODEDTMIN:
		if (params_str.size() == 1) {

			std::string value = params_str[0];

			return MakeInteractiveObject(value, IOI_ODEDTMIN, 0, 0, value);
		}
		break;

	case IOI_ODEDTMAX:
		if (params_str.size() == 1) {

			std::string value = params_str[0];

			return MakeInteractiveObject(value, IOI_ODEDTMAX, 0, 0, value);
		}
		break;

	case IOI_PBC_X:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject("0", IOI_PBC_X, SMesh[meshIndex]->get_id(), 0, "", OFFCOLOR);
		}
		break;

	case IOI_PBC_Y:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject("0", IOI_PBC_Y, SMesh[meshIndex]->get_id(), 0, "", OFFCOLOR);
		}
		break;

	case IOI_PBC_Z:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject("0", IOI_PBC_Z, SMesh[meshIndex]->get_id(), 0, "", OFFCOLOR);
		}
		break;

	case IOI_SPBC_X:
		return MakeInteractiveObject("N/A", IOI_SPBC_X, 0, 0, "", UNAVAILABLECOLOR);
		break;

	case IOI_SPBC_Y:
		return MakeInteractiveObject("N/A", IOI_SPBC_Y, 0, 0, "", UNAVAILABLECOLOR);
		break;

	case IOI_SPBC_Z:
		return MakeInteractiveObject("N/A", IOI_SPBC_Z, 0, 0, "", UNAVAILABLECOLOR);
		break;

	case IOI_INDIVIDUALSHAPE:
		if (params_str.size() == 1) {

			int status = ToNum(params_str[0]);

			if (status) {

				return MakeInteractiveObject("On", IOI_INDIVIDUALSHAPE, 0, 1, "", ONCOLOR);
			}
			else {

				return MakeInteractiveObject("Off", IOI_INDIVIDUALSHAPE, 0, 0, "", OFFCOLOR);
			}
		}
		break;

	case IOI_IMAGECROPPING:
		return MakeInteractiveObject("0, 0, 1, 1", IOI_IMAGECROPPING, 0, 0, "0, 0, 1, 1");
		break;
	
	case IOI_USERCONSTANT:
		if (params_str.size() == 3) {

			std::string constant_name = params_str[0];
			double value = ToNum(params_str[1]);
			int index_in_list = ToNum(params_str[2]);

			std::string userConstant_text = Build_EquationConstants_Text(index_in_list);

			return MakeInteractiveObject(userConstant_text, IOI_USERCONSTANT, index_in_list, index_in_list, userConstant_text);
		}
		break;

	case IOI_SKYPOSDMUL:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);
			double multiplier = SMesh[meshIndex]->Get_skypos_dmul();

			return MakeInteractiveObject(ToString(multiplier), IOI_SKYPOSDMUL, SMesh[meshIndex]->get_id(), -1, ToString(multiplier), ONCOLOR);
		}
		break;

	case IOI_DWPOSCOMPONENT:
		if (params_str.size() == 1) {

			int component = ToNum(params_str[0]);

			switch (component) {

			default:
			case -1:
				return MakeInteractiveObject("Auto", IOI_DWPOSCOMPONENT, 0, component);
				break;

			case 0:
				return MakeInteractiveObject("x", IOI_DWPOSCOMPONENT, 0, component);
				break;

			case 1:
				return MakeInteractiveObject("y", IOI_DWPOSCOMPONENT, 0, component);
				break;

			case 2:
				return MakeInteractiveObject("z", IOI_DWPOSCOMPONENT, 0, component);
				break;
			}
		}
		break;

	case IOI_MCCOMPUTATION:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject(" Parallel ", IOI_MCCOMPUTATION, SMesh[meshIndex]->get_id(), 0);
		}
		break;
	
	case IOI_MCTYPE:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject(" Classical ", IOI_MCTYPE, SMesh[meshIndex]->get_id(), 0, "");
		}
		break;

	case IOI_MCDISABLED:
		if (params_str.size() == 1) {

			int meshIndex = ToNum(params_str[0]);

			return MakeInteractiveObject(" Disabled ", IOI_MCDISABLED, SMesh[meshIndex]->get_id(), 1, "", OFFCOLOR);
		}
		break;

	case IOI_MCCOMPUTEFIELDS:
		if (params_str.size() == 1) {

			int status = ToNum(params_str[0]);

			if (status) {

				return MakeInteractiveObject("Enabled", IOI_MCCOMPUTEFIELDS, 0, 1, "", ONCOLOR);
			}
			else {

				return MakeInteractiveObject("Disabled", IOI_MCCOMPUTEFIELDS, 0, 0, "", OFFCOLOR);
			}
		}
		break;

	case IOI_SHAPEROT:
		if (params_str.size() == 1) {

			return MakeInteractiveObject(params_str[0], IOI_SHAPEROT, 0, 0, params_str[0]);
		}
		break;

	case IOI_SHAPEREP:
		if (params_str.size() == 1) {

			return MakeInteractiveObject(params_str[0], IOI_SHAPEREP, 0, 0, params_str[0]);
		}
		break;

	case IOI_SHAPEDSP:
		if (params_str.size() == 1) {

			DBL3 value = ToNum(params_str[0], "m");

			return MakeInteractiveObject(ToString(value, "m"), IOI_SHAPEDSP, 0, 0, ToString(value, "m"));
		}
		break;

	case IOI_SHAPEMET:
		if (params_str.size() == 1) {

			return MakeInteractiveObject(params_str[0], IOI_SHAPEMET, 0, 0, params_str[0]);
		}
		break;

	case IOI_DISPRENDER_DETAIL:
		if (params_str.size() == 1) {

			double value = ToNum(params_str[0], "m");

			return MakeInteractiveObject(ToString(value, "m"), IOI_DISPRENDER_DETAIL, 0, 0, ToString(value, "m"));
		}
		break;

	case IOI_DISPRENDER_THRESH1:
		if (params_str.size() == 1) {

			return MakeInteractiveObject(params_str[0], IOI_DISPRENDER_THRESH1, 0, ToNum(params_str[0]));
		}
		break;

	case IOI_DISPRENDER_THRESH2:
		if (params_str.size() == 1) {

			return MakeInteractiveObject(params_str[0], IOI_DISPRENDER_THRESH2, 0, ToNum(params_str[0]));
		}
		break;

	case IOI_DISPRENDER_THRESH3:
		if (params_str.size() == 1) {

			return MakeInteractiveObject(params_str[0], IOI_DISPRENDER_THRESH3, 0, ToNum(params_str[0]));
		}
		break;
	}

	return "";
}

