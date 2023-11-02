#include "stdafx.h"
#include "Simulation.h"

void Simulation::AddGenericStage(SS_ stageType, std::string meshName) 
{
	switch(stageType) {

	case SS_RELAX:
	{
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_MONTECARLO:
	{
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_ITERATIONS));
		stageConfig.set_value(0.5);
		stageConfig.set_stopvalue(10000);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_HFIELDXYZ:
	{
		//zero field with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(DBL3(0, 0, 0));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_HFIELDXYZSEQ:
	{
		//zero field with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(SEQ3(DBL3(-1e5, 0, 0), DBL3(1e5, 0, 0), 100));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_HPOLARSEQ:
	{
		//zero field with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(SEQP(DBL3(-1e5, 90, 0), DBL3(1e5, 90, 0), 100));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_HFMR:
	{
		//Bias field along y with Hrf along x. 1 GHz.
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_TIME), meshName);
		stageConfig.set_value(COSSEQ3(DBL3(0, 1e6, 0), DBL3(1e3, 0, 0), 20, 100));
		stageConfig.set_stopvalue(50e-12);

		simStages.push_back(stageConfig);
	}
	break;
	
	case SS_HFIELDEQUATION:
	{
		//zero field with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(std::string("0, 0, 0"));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_HFIELDEQUATIONSEQ:
	{
		//zero field with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(StringSequence(std::string("1: 0, 0, 0")));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_HFIELDFILE:
	{
		//zero field with STOP_TIME
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_TIME), meshName);
		stageConfig.set_value(FILESEQ3(directory, "file.txt", 1e-9));
		stageConfig.set_stopvalue(1e-9);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_V:
	{
		//zero potential with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH));
		stageConfig.set_value(0.0);
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_VSEQ:
	{
		//V 0.0 to 1.0 V in 10 steps with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH));
		stageConfig.set_value(SEQ(0.0, 1.0, 10));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_VEQUATION:
	{
		//zero potential with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH));
		stageConfig.set_value(std::string("0"));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_VEQUATIONSEQ:
	{
		//zero potential with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH));
		stageConfig.set_value(StringSequence(std::string("1: 0")));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_VFILE:
	{
		//zero field with STOP_TIME
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_TIME), meshName);
		stageConfig.set_value(FILESEQ(directory, "file.txt", 1e-9));
		stageConfig.set_stopvalue(1e-9);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_I:
	{
		//zero current with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH));
		stageConfig.set_value(0.0);
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_ISEQ:
	{
		//I 0.0 to 1.0 mA in 10 steps with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH));
		stageConfig.set_value(SEQ(0.0, 1.0e-3, 10));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_IEQUATION:
	{
		//zero current with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH));
		stageConfig.set_value(std::string("0"));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_IEQUATIONSEQ:
	{
		//zero current with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH));
		stageConfig.set_value(StringSequence(std::string("1: 0")));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_IFILE:
	{
		//zero field with STOP_TIME
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_TIME), meshName);
		stageConfig.set_value(FILESEQ(directory, "file.txt", 1e-9));
		stageConfig.set_stopvalue(1e-9);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_T:
	{
		//zero temperature with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(0.0);
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_TSEQ:
	{
		//T 0.0 to 300K in 10 steps with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(SEQ(0.0, 300.0, 10));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_TEQUATION:
	{
		//zero temperature with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(std::string("0"));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;
		
	case SS_TEQUATIONSEQ:
	{
		//zero temperature with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(StringSequence(std::string("1: 0")));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_TFILE:
	{
		//zero field with STOP_TIME
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_TIME), meshName);
		stageConfig.set_value(FILESEQ(directory, "file.txt", 1e-9));
		stageConfig.set_stopvalue(1e-9);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_Q:
	{
		//1e19 W/m3 heat source with STOP_TIME of 10ns
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_TIME), meshName);
		stageConfig.set_value(1e19);
		stageConfig.set_stopvalue(10e-9);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_QSEQ:
	{
		//Q 0.0 to 1e19 W/m3 in 10 steps with STOP_TIME of 1ns
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_TIME), meshName);
		stageConfig.set_value(SEQ(0.0, 1e19, 10));
		stageConfig.set_stopvalue(1e-9);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_QEQUATION:
	{
		//zero Q with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(std::string("0"));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_QEQUATIONSEQ:
	{
		//zero Q with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(StringSequence(std::string("1: 0")));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_QFILE:
	{
		//zero field with STOP_TIME
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_TIME), meshName);
		stageConfig.set_value(FILESEQ(directory, "file.txt", 1e-9));
		stageConfig.set_stopvalue(1e-9);

		simStages.push_back(stageConfig);
	}
	break;

	case SS_TSIGPOLAR:
	{
		//zero stress with STOP_MXH
		StageConfig stageConfig = StageConfig(stageDescriptors(stageType), stageStopDescriptors(STOP_MXH), meshName);
		stageConfig.set_value(DBL3(0, 0, 0));
		stageConfig.set_stopvalue(1e-4);

		simStages.push_back(stageConfig);
	}
	break;
	}
}

void Simulation::DeleteStage(int stageIndex) 
{
	simStages.erase(stageIndex);
}

void Simulation::SetGenericStopCondition(int index, STOP_ stopType) 
{
	if(!GoodIdx(simStages.last(), index)) return;

	switch(stopType) {

	case STOP_NOSTOP:
		simStages[index].set_stoptype( stageStopDescriptors(stopType) );
		simStages[index].clear_stopvalue();
		break;

	case STOP_ITERATIONS:
		simStages[index].set_stoptype( stageStopDescriptors(stopType) );
		simStages[index].set_stopvalue(1000);
		break;

	case STOP_MXH:
		simStages[index].set_stoptype( stageStopDescriptors(stopType) );
		simStages[index].set_stopvalue(1e-4);
		break;

	case STOP_DMDT:
		simStages[index].set_stoptype(stageStopDescriptors(stopType));
		simStages[index].set_stopvalue(1e-5);
		break;

	case STOP_TIME:
		simStages[index].set_stoptype( stageStopDescriptors(stopType) );
		simStages[index].set_stopvalue(10e-9);
		break;

	case STOP_MXH_ITER:
		simStages[index].set_stoptype(stageStopDescriptors(stopType));
		simStages[index].set_stopvalue(DBL2(1e-4, 10000));
		break;

	case STOP_DMDT_ITER:
		simStages[index].set_stoptype(stageStopDescriptors(stopType));
		simStages[index].set_stopvalue(DBL2(1e-5, 10000));
		break;
	}
}

void Simulation::SetGenericDataSaveCondition(int index, DSAVE_ dsaveType)
{
	switch(dsaveType) {

	case DSAVE_NONE:
		simStages[index].set_dsavetype( dataSaveDescriptors(dsaveType) );
		simStages[index].clear_dsavevalue();
		break;

	case DSAVE_STAGE:
		simStages[index].set_dsavetype( dataSaveDescriptors(dsaveType) );
		simStages[index].clear_dsavevalue();
		break;

	case DSAVE_STEP:
		simStages[index].set_dsavetype( dataSaveDescriptors(dsaveType) );
		simStages[index].clear_dsavevalue();
		break;

	case DSAVE_ITER:
		simStages[index].set_dsavetype( dataSaveDescriptors(dsaveType) );
		simStages[index].set_dsavevalue(100);
		break;

	case DSAVE_TIME:
		simStages[index].set_dsavetype( dataSaveDescriptors(dsaveType) );
		simStages[index].set_dsavevalue(1e-9);
		break;
	}
}

void Simulation::EditStageType(int index, SS_ stageType, std::string meshName) 
{
	//if same stage type as before just change the mesh name
	if(GoodIdx(simStages.last(), index) && simStages[index].stage_type() == stageType) {
		
		simStages[index].set_meshname(meshName);
	}
	else {

		//new stage type at this index so set a generic stage to start off with
		AddGenericStage(stageType, meshName);
		simStages.move(simStages.last(), index);
		simStages.erase(index + 1);
	}
}

void Simulation::EditStageValue(int stageIndex, std::string value_string) 
{
	bool adjust_special = false;

	//in some special cases, changing the stage value can result in a stage type change
	if (simStages[stageIndex].stage_type() == SS_HFIELDEQUATION || simStages[stageIndex].stage_type() == SS_HFIELDEQUATIONSEQ) {

		size_t pos = value_string.find(":");
		//change from equation to equation sequence
		if (pos != std::string::npos && simStages[stageIndex].stage_type() == SS_HFIELDEQUATION) {

			adjust_special = true;

			AddGenericStage(SS_HFIELDEQUATIONSEQ, simStages[stageIndex].meshname());
		}
		//change from equation sequence to equation
		else if (pos == std::string::npos && simStages[stageIndex].stage_type() == SS_HFIELDEQUATIONSEQ) {

			adjust_special = true;

			AddGenericStage(SS_HFIELDEQUATION, simStages[stageIndex].meshname());
		}
	}

	else if (simStages[stageIndex].stage_type() == SS_VEQUATION || simStages[stageIndex].stage_type() == SS_VEQUATIONSEQ) {

		size_t pos = value_string.find(":");
		//change from equation to equation sequence
		if (pos != std::string::npos && simStages[stageIndex].stage_type() == SS_VEQUATION) {

			adjust_special = true;

			AddGenericStage(SS_VEQUATIONSEQ, simStages[stageIndex].meshname());
		}
		//change from equation sequence to equation
		else if (pos == std::string::npos && simStages[stageIndex].stage_type() == SS_VEQUATIONSEQ) {

			adjust_special = true;

			AddGenericStage(SS_VEQUATION, simStages[stageIndex].meshname());
		}
	}

	else if (simStages[stageIndex].stage_type() == SS_IEQUATION || simStages[stageIndex].stage_type() == SS_IEQUATIONSEQ) {

		size_t pos = value_string.find(":");
		//change from equation to equation sequence
		if (pos != std::string::npos && simStages[stageIndex].stage_type() == SS_IEQUATION) {

			adjust_special = true;

			AddGenericStage(SS_IEQUATIONSEQ, simStages[stageIndex].meshname());
		}
		//change from equation sequence to equation
		else if (pos == std::string::npos && simStages[stageIndex].stage_type() == SS_IEQUATIONSEQ) {

			adjust_special = true;

			AddGenericStage(SS_IEQUATION, simStages[stageIndex].meshname());
		}
	}

	else if (simStages[stageIndex].stage_type() == SS_TEQUATION || simStages[stageIndex].stage_type() == SS_TEQUATIONSEQ) {

		size_t pos = value_string.find(":");
		//change from equation to equation sequence
		if (pos != std::string::npos && simStages[stageIndex].stage_type() == SS_TEQUATION) {

			adjust_special = true;

			AddGenericStage(SS_TEQUATIONSEQ, simStages[stageIndex].meshname());
		}
		//change from equation sequence to equation
		else if (pos == std::string::npos && simStages[stageIndex].stage_type() == SS_TEQUATIONSEQ) {

			adjust_special = true;

			AddGenericStage(SS_TEQUATION, simStages[stageIndex].meshname());
		}
	}

	else if (simStages[stageIndex].stage_type() == SS_QEQUATION || simStages[stageIndex].stage_type() == SS_QEQUATIONSEQ) {

		size_t pos = value_string.find(":");
		//change from equation to equation sequence
		if (pos != std::string::npos && simStages[stageIndex].stage_type() == SS_QEQUATION) {

			adjust_special = true;

			AddGenericStage(SS_QEQUATIONSEQ, simStages[stageIndex].meshname());
		}
		//change from equation sequence to equation
		else if (pos == std::string::npos && simStages[stageIndex].stage_type() == SS_QEQUATIONSEQ) {

			adjust_special = true;

			AddGenericStage(SS_QEQUATION, simStages[stageIndex].meshname());
		}
	}

	if (!adjust_special) {

		//edit current stage
		simStages[stageIndex].set_stagevalue_fromstring(value_string);
	}
	else {

		//current stage type has been changed : we have a new generic stage of the required type at the end

		//set required value and copy current generate stage data
		simStages[simStages.last()].set_stagevalue_fromstring(value_string);
		simStages[simStages.last()].copy_stage_general_data(simStages[stageIndex]);

		//replace current stage
		simStages.move(simStages.last(), stageIndex);
		simStages.erase(stageIndex + 1);
	}
}

void Simulation::EditStageStopCondition(int index, STOP_ stopType, std::string stopValueString) 
{
	//if same stop condition as before just change the stop value
	if(GoodIdx(simStages.last(), index) && simStages[index].stop_condition() == stopType) {

		if(stopValueString.length()) simStages[index].set_stopvalue_fromstring(stopValueString);
	}
	else {

		SetGenericStopCondition(index, stopType);
		if(stopValueString.length()) simStages[index].set_stopvalue_fromstring(stopValueString);
	}
}

void Simulation::EditDataSaveCondition(int index, DSAVE_ dsaveType, std::string dsaveValueString)
{
	//if same saving condition as before just change the value
	if(GoodIdx(simStages.last(), index) && simStages[index].dsave_type() == dsaveType) {

		if(dsaveValueString.length()) simStages[index].set_dsavevalue_fromstring(dsaveValueString);
	}
	else {

		SetGenericDataSaveCondition(index, dsaveType);
		if(dsaveValueString.length()) simStages[index].set_dsavevalue_fromstring(dsaveValueString);
	}
}

void Simulation::UpdateStageMeshNames(std::string oldMeshName, std::string newMeshName) 
{
	for(int idx = 0; idx < simStages.size(); idx++) {

		if (simStages[idx].meshname() == oldMeshName) {

			simStages[idx].set_meshname(newMeshName);
		}
	}
}

INT2 Simulation::Check_and_GetStageStep()
{
	//first make sure stage value is correct - this could only happen if stages have been deleted. If incorrect just reset back to 0.
	if(stage_step.major >= simStages.size()) stage_step = INT2();

	//mak sure step value is correct - if incorrect reset back to zero.
	if(stage_step.minor > simStages[stage_step.major].number_of_steps()) stage_step.minor = 0;

	return stage_step;
}

void Simulation::CheckSimulationSchedule(void) 
{
	//if stage index exceeds number of stages then just set it to the end : stages must have been deleted whilst simulation running.
	if(stage_step.major >= simStages.size()) stage_step.major = simStages.last();

	switch( simStages[ stage_step.major ].stop_condition() ) {

	case STOP_NOSTOP:
		break;

	case STOP_ITERATIONS:

		if( SMesh.GetStageIteration() >= (int)simStages[ stage_step.major ].get_stopvalue() ) AdvanceSimulationSchedule();
		break;

	case STOP_MXH:

		if( SMesh.Get_mxh() <= (double)simStages[ stage_step.major ].get_stopvalue() ) AdvanceSimulationSchedule();
		break;

	case STOP_DMDT:

		if (SMesh.Get_dmdt() <= (double)simStages[stage_step.major].get_stopvalue()) AdvanceSimulationSchedule();
		break;

	case STOP_TIME:
	{
		double stime = SMesh.GetStageTime();
		double tstop = (double)simStages[stage_step.major].get_stopvalue();
		double dT = SMesh.GetTimeStep();
		if (tstop - stime < dT * 0.01) AdvanceSimulationSchedule();
	}
		break;

	case STOP_MXH_ITER:
	{
		DBL2 stop = simStages[stage_step.major].get_stopvalue();
		if (SMesh.Get_mxh() <= stop.i || SMesh.GetStageIteration() >= stop.j) AdvanceSimulationSchedule();
	}
	break;

	case STOP_DMDT_ITER: 
	{
		DBL2 stop = simStages[stage_step.major].get_stopvalue();
		if (SMesh.Get_dmdt() <= stop.i || SMesh.GetStageIteration() >= stop.j) AdvanceSimulationSchedule();
	}
	break;
	}
}

void Simulation::CheckSaveDataConditions()
{
	switch (simStages[stage_step.major].dsave_type()) {

	case DSAVE_NONE:
	case DSAVE_STAGE:
	case DSAVE_STEP:
		//step and stage save data is done in AdvanceSimulationSchedule when step or stage ending is detected
		break;

	case DSAVE_ITER:
		if (!(SMesh.GetIteration() % (int)simStages[stage_step.major].get_dsavevalue())) SaveData();
		break;

	case DSAVE_TIME:
	{
		double stime = SMesh.GetStageTime();
		double tsave = (double)simStages[stage_step.major].get_dsavevalue();
		//important to use last time step not current time step, otherwise with adaptive timestep multiple saves will result if dT keeps increasing
		double dT = SMesh.GetLastTimeStep();

		double delta = stime - floor(stime / tsave + (dT / tsave) * 0.5) * tsave;
		if (delta < dT * (1 - (dT / tsave) * 0.5)) {

			//the == 0.0 check is fine since this is intended to trigger save when simulation starts
			if (stime < last_time_save || last_time_save == 0.0) {

				last_time_save = stime;
				SaveData();
			}
			//this last check is needed to avoid multiple saves around same tsave increment, which can happen when adaptive time-step is used
			else if (stime - last_time_save > tsave * 0.5) SaveData();
		}
	}
	break;
	}
}

void Simulation::AdvanceSimulationSchedule(void)
{
	//assume stage_step.major is correct

	//do we need to iterate the transport solver? 
	//if static_transport_solver is true then the transport solver was stopped from iterating before reaching the end of a stage or step
	if (static_transport_solver) {

		//turn off flag for now to enable iterating the transport solver
		static_transport_solver = false;

#if COMPILECUDA == 1
		if (cudaEnabled) {

			SMesh.UpdateTransportSolverCUDA();
		}
		else {

			SMesh.UpdateTransportSolver();
		}
#else
		SMesh.UpdateTransportSolver();
#endif

		//turn flag back on
		static_transport_solver = true;
		}

	//first try to increment the step number
	if (stage_step.minor < simStages[stage_step.major].number_of_steps()) {

		//save data at end of current step?
		if (simStages[stage_step.major].dsave_type() == DSAVE_STEP) SaveData();
		//if not step condition still check if we need to save data (e.g. to save last point)
		else CheckSaveDataConditions();

		//next step and set value for it
		stage_step.minor++;
		SetSimulationStageValue();
	}
	else {

		//save data at end of current stage?
		if (simStages[stage_step.major].dsave_type() == DSAVE_STAGE || simStages[stage_step.major].dsave_type() == DSAVE_STEP) SaveData();
		//if not step condition still check if we need to save data (e.g. to save last point)
		else CheckSaveDataConditions();

		//next stage
		stage_step.minor = 0;
		if (!single_stage_run) {

			stage_step.major++;

			//if not at the end then set stage value for given stage_step
			if (stage_step.major < simStages.size()) {

				SetSimulationStageValue();
			}
			else {

				//schedule reached end: stop simulation. Note, since this routine is called from Simulate routine, which runs on the THREAD_LOOP thread, cannot stop THREAD_LOOP from within it: stop it from another thread.
				//use a dedicated thread id to stop loop, since we need to be sure it's not blocking (alternatively could configure it with set_nonblocking_thread). Better to use dedicated thread just to make sure we don't have to wait for some other thread to finish.
				single_call_launch(&Simulation::StopSimulation, THREAD_LOOP_STOP);

				//set stage step to start of last stage but without resetting anything : the idea is user can edit the stopping value for the last stage, e.g. add more time to it, then run simulation some more
				//if you want a complete reset then reset command must be issued in console
				stage_step = INT2(stage_step.major - 1, 0);
			}
		}
		else single_call_launch(&Simulation::StopSimulation, THREAD_LOOP_STOP);
	}
}

void Simulation::SetSimulationStageValue(int stage_index) 
{
	if (stage_index < 0) SMesh.NewStageODE();

	int stage = (stage_index >= 0 ? stage_index : stage_step.major);
	int step = (stage_index >= 0 ? 0 : stage_step.minor);

	//assume stage_step is correct (if called from AdvanceSimulationSchedule it will be. could also be called directly at the start of a simulation with stage_step reset, so it's also correct).

	switch(simStages[stage].stage_type()) {

	case SS_RELAX:
	case SS_MONTECARLO:
	break;

	case SS_HFIELDXYZ:
	case SS_HFIELDXYZSEQ:
	case SS_HPOLARSEQ:
	case SS_HFMR:
	case SS_HFIELDFILE:
	{
		std::string meshName = simStages[stage].meshname();

		DBL3 appliedField = simStages[stage].get_value<DBL3>(step);

		if (SMesh.contains(meshName)) SMesh[meshName]->CallModuleMethod(&ZeemanBase::SetField, appliedField);
		else if (meshName == SMesh.superMeshHandle) {

			for (int idx = 0; idx < SMesh.size(); idx++) {
				SMesh[idx]->CallModuleMethod(&ZeemanBase::SetField, appliedField);
			}
		}
	}
	break;

	case SS_TSIGPOLAR:
	{
		std::string meshName = simStages[stage].meshname();

		DBL3 appliedStress = simStages[stage].get_value<DBL3>(step);

		if (SMesh.contains(meshName)) SMesh[meshName]->CallModuleMethod(&MElastic::SetUniformStress, appliedStress);
		else if (meshName == SMesh.superMeshHandle) {

			for (int idx = 0; idx < SMesh.size(); idx++) {
				SMesh[idx]->CallModuleMethod(&MElastic::SetUniformStress, appliedStress);
			}
		}
	}
	break;

	case SS_HFIELDEQUATION:
	{
		std::string meshName = simStages[stage].meshname();

		std::string equation_text = simStages[stage].get_value<std::string>(step);

		if (SMesh.contains(meshName)) SMesh[meshName]->CallModuleMethod(&ZeemanBase::SetFieldEquation, equation_text, 0);
		else if (meshName == SMesh.superMeshHandle) {

			for (int idx = 0; idx < SMesh.size(); idx++) {
				SMesh[idx]->CallModuleMethod(&ZeemanBase::SetFieldEquation, equation_text, 0);
			}
		}
	}
	break;

	case SS_HFIELDEQUATIONSEQ:
	{
		std::string meshName = simStages[stage].meshname();

		std::string equation_text = simStages[stage].get_value<std::string>(step);
		//for a equation sequence we have "n: actual equation", where n is the number of steps
		std::string equation_equation_text = equation_text.substr(equation_text.find_first_of(':') + 1);

		if (SMesh.contains(meshName)) SMesh[meshName]->CallModuleMethod(&ZeemanBase::SetFieldEquation, equation_equation_text, step);
		else if (meshName == SMesh.superMeshHandle) {

			for (int idx = 0; idx < SMesh.size(); idx++) {
				SMesh[idx]->CallModuleMethod(&ZeemanBase::SetFieldEquation, equation_equation_text, step);
			}
		}
	}
	break;

	case SS_VEQUATION:
	{
		std::string equation_text = simStages[stage].get_value<std::string>(step);

		SMesh.CallModuleMethod(&STransport::SetPotentialEquation, equation_text, 0);
	}
	break;

	case SS_VEQUATIONSEQ:
	{
		std::string equation_text = simStages[stage].get_value<std::string>(step);

		//for a equation sequence we have "n: actual equation", where n is the number of steps
		std::string equation_equation_text = equation_text.substr(equation_text.find_first_of(':') + 1);

		SMesh.CallModuleMethod(&STransport::SetPotentialEquation, equation_equation_text, step);
	}
	break;

	case SS_V:
	case SS_VSEQ:
	case SS_VFILE:
	{
		double potential = simStages[stage].get_value<double>(step);

		SMesh.CallModuleMethod(&STransport::SetPotential, potential, true);
	}
	break;

	case SS_I:
	case SS_ISEQ:
	case SS_IFILE:
	{
		double current = simStages[stage].get_value<double>(step);

		SMesh.CallModuleMethod(&STransport::SetCurrent, current, true);
	}
	break;

	case SS_IEQUATION:
	{
		std::string equation_text = simStages[stage].get_value<std::string>(step);

		SMesh.CallModuleMethod(&STransport::SetCurrentEquation, equation_text, 0);
	}
	break;

	case SS_IEQUATIONSEQ:
	{
		std::string equation_text = simStages[stage].get_value<std::string>(step);

		//for a equation sequence we have "n: actual equation", where n is the number of steps
		std::string equation_equation_text = equation_text.substr(equation_text.find_first_of(':') + 1);

		SMesh.CallModuleMethod(&STransport::SetCurrentEquation, equation_equation_text, step);
	}
	break;

	case SS_T:
	case SS_TSEQ:
	case SS_TFILE:
	{
		std::string meshName = simStages[stage].meshname();

		double temperature = simStages[stage].get_value<double>(step);

		if (SMesh.contains(meshName)) SMesh[meshName]->SetBaseTemperature(temperature);
		else if (meshName == SMesh.superMeshHandle) {

			//all meshes
			for (int idx = 0; idx < SMesh.size(); idx++) {

				SMesh[idx]->SetBaseTemperature(temperature);
			}
		}
	}
	break;

	case SS_TEQUATION:
	{
		std::string meshName = simStages[stage].meshname();

		std::string equation_text = simStages[stage].get_value<std::string>(step);

		if (SMesh.contains(meshName)) SMesh[meshName]->SetBaseTemperatureEquation(equation_text, 0);
		else if (meshName == SMesh.superMeshHandle) {

			//all meshes
			for (int idx = 0; idx < SMesh.size(); idx++) {

				SMesh[idx]->SetBaseTemperatureEquation(equation_text, 0);
			}
		}
	}
	break;

	case SS_TEQUATIONSEQ:
	{
		std::string meshName = simStages[stage].meshname();

		std::string equation_text = simStages[stage].get_value<std::string>(step);

		//for a equation sequence we have "n: actual equation", where n is the number of steps
		std::string equation_equation_text = equation_text.substr(equation_text.find_first_of(':') + 1);

		if (SMesh.contains(meshName)) SMesh[meshName]->SetBaseTemperatureEquation(equation_text, step);
		else if (meshName == SMesh.superMeshHandle) {

			//all meshes
			for (int idx = 0; idx < SMesh.size(); idx++) {

				SMesh[idx]->SetBaseTemperatureEquation(equation_text, step);
			}
		}
	}
	break;

	case SS_Q:
	case SS_QSEQ:
	case SS_QFILE:
	{
		std::string meshName = simStages[stage].meshname();

		double Qvalue = simStages[stage].get_value<double>(step);

		if (SMesh.contains(meshName)) SMesh[meshName]->Q = Qvalue;
		else if (meshName == SMesh.superMeshHandle) {

			//all meshes
			for (int idx = 0; idx < SMesh.size(); idx++) {

				SMesh[idx]->Q = Qvalue;
			}
		}
	}
	break;

	case SS_QEQUATION:
	{
		std::string meshName = simStages[stage].meshname();

		std::string equation_text = simStages[stage].get_value<std::string>(step);

		if (SMesh.contains(meshName)) SMesh[meshName]->CallModuleMethod(&HeatBase::SetQEquation, equation_text, 0);
		else if (meshName == SMesh.superMeshHandle) {

			for (int idx = 0; idx < SMesh.size(); idx++) {
				SMesh[idx]->CallModuleMethod(&HeatBase::SetQEquation, equation_text, 0);
			}
		}
	}
	break;

	case SS_QEQUATIONSEQ:
	{
		std::string meshName = simStages[stage].meshname();

		std::string equation_text = simStages[stage].get_value<std::string>(step);

		//for a equation sequence we have "n: actual equation", where n is the number of steps
		std::string equation_equation_text = equation_text.substr(equation_text.find_first_of(':') + 1);

		if (SMesh.contains(meshName)) SMesh[meshName]->CallModuleMethod(&HeatBase::SetQEquation, equation_text, step);
		else if (meshName == SMesh.superMeshHandle) {

			for (int idx = 0; idx < SMesh.size(); idx++) {
				SMesh[idx]->CallModuleMethod(&HeatBase::SetQEquation, equation_text, step);
			}
		}
	}
	break;
	}
}