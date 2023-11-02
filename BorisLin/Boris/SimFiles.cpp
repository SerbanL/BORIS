#include "stdafx.h"
#include "Simulation.h"

BError Simulation::SaveSimulation(std::string fileName)
{
	BError error(__FUNCTION__);

	StopSimulation();

	if (GetFileTermination(fileName) != ".bsm")
		fileName += ".bsm";

	if (!GetFilenameDirectory(fileName).length()) fileName = directory + fileName;

	std::ofstream bdout;
	bdout.open(fileName.c_str(), std::ios::out | std::ios::binary);

	if (bdout.is_open()) {

		//save version number header
		bdout << simfile_header << Program_Version << std::endl;

		bdout.precision(CONVERSIONPRECISION);

		//before saving, switch CUDA off, so everything is stored and up to date in cpu memory
		if (cudaEnabled) {

			error = SMesh.SwitchCUDAState(false);

			if (!error) {

				//Set cudaEnabled to true so it's saved with the right value
				cudaEnabled = true;
				SaveObjectState(bdout);

				//switch CUDA back on if it was on before (cudaEnabled = false s function takes effect)
				cudaEnabled = false;
				error = SMesh.SwitchCUDAState(true);
			}
		}
		else SaveObjectState(bdout);

		bdout.close();

		if (!error) currentSimulationFile = fileName;

		return error;
	}
	else return error(BERROR_COULDNOTSAVEFILE);
}

BError Simulation::LoadSimulation(std::string fileName)
{
	BError error(__FUNCTION__);

	bool success = true;

	StopSimulation();

	//clear screen before loading new simulation : if any interactive objects on screen are updated whilst simulation is loading the program can crash.
	ClearScreen();
	BD.DisplayConsoleMessage("Loading simulation ... please wait.");

	if(GetFileTermination(fileName) != ".bsm")
		fileName += ".bsm";

	if (!GetFilenameDirectory(fileName).length()) fileName = directory + fileName;
	else directory = GetFilenameDirectory(fileName);

	std::string save_directory = directory;

	std::ifstream bdin;
	bdin.open(fileName.c_str(), std::ios::in | std::ios::binary);

	if (bdin.is_open()) {

		//get first line to check for header and version number
		char line[FILEROWCHARS];
		bdin.getline(line, FILEROWCHARS);

		size_t pos = std::string(line).find(simfile_header);
		if (pos != std::string::npos) {

			//header found, get version
			int SimFile_Version_Number = ToNum(std::string(line).substr(simfile_header.length()));

			//check if simulation file version matches program version : if it doesn't then first load default state then attempt to load simulation file
			if (SimFile_Version_Number < Program_Version) {

				//older simulation file version

				//close original stream so we can load default state
				bdin.close();

				std::string default_file = GetUserDocumentsPath() + boris_data_directory + boris_simulations_directory + std::string("default.bsm");
				bdin.open(default_file.c_str(), std::ios::in | std::ios::binary);

				if (cudaAvailable && cudaEnabled) error = SMesh.SwitchCUDAState(false);

				if (!error) success = LoadObjectState(bdin);

				bdin.close();

				if (!success) return error(BERROR_COULDNOTLOADFILE_CRIT);

				//at this point default state is loaded, so re-open the original simulation file
				bdin.open(fileName.c_str(), std::ios::in | std::ios::binary);

				//chuck away the first line with the header
				bdin.getline(line, FILEROWCHARS);
				if (!bdin.is_open()) return error(BERROR_COULDNOTLOADFILE);

				//now proceed as normal. Loading from default state if versions do not match means any newer data structures not saved in the old simulation file version are set to their default state so the program should behave as expected.
			}
			else if (SimFile_Version_Number > Program_Version) {

				//cannot load a newer simulation file with an older program version
				bdin.close();
				return error(BERROR_COULDNOTLOADFILE_VERSIONMISMATCH);
			}
		}
		else {

			//header not found : not a valid simulation file
			bdin.close();
			return error(BERROR_COULDNOTLOADFILE);
		}

		//before loading make sure CUDA is switched off so it doesn't cause problems with loading
		if (cudaAvailable && cudaEnabled) error = SMesh.SwitchCUDAState(false);

		if (!error) success = LoadObjectState(bdin);

		bdin.close();

		//it's possible to load a file with cudaEnabled = true when cuda is not enabled on the machine.
		//If cuda not available must make sure the cudaEnabled flag is off too.
		if (!cudaAvailable) cudaEnabled = false;

		if (!success) return error(BERROR_COULDNOTLOADFILE_CRIT);

		//cudaEnabled will have been modified depending on what was stored in the simulation file, but CUDA is still switched off at this point; switch CUDA on if indicated
#if COMPILECUDA == 1
		if (cudaEnabled && !error) {
		
			//a previous simulation file might have been configured for a cuda device != on a multi-GPU machine; might not be available on current machine
			if (cudaDeviceSelect >= cudaDeviceVersions.size() || cudaDeviceVersions[cudaDeviceSelect].first != __CUDA_ARCH__) {

				//back to default, since cuda available this device should be present
				select_cuda_devices({ 0 });
			}

			if (cudaDeviceVersions[cudaDeviceSelect].first == __CUDA_ARCH__) {

				//cudaEnabled will be set to true by SwitchCUDAState, but need to set it to false first to take effect
				cudaEnabled = false;
				error = SMesh.SwitchCUDAState(true, { cudaDeviceSelect });
			}
		}
#endif

		//check for any errors in loading program state
		if (!error) error = SMesh.Error_On_Create();
		if (!error) error = SMesh.UpdateConfiguration(UPDATECONFIG_FORCEUPDATE);

		if (error) return error;

		currentSimulationFile = fileName;
		//set directory as the simulation file directory
		directory = save_directory;

		ClearScreen();
		UpdateScreen();
	}
	else return error(BERROR_COULDNOTLOADFILE);

	return error;
}