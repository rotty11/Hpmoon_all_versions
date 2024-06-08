/**
 * @file config.cpp
 * @author Juan José Escobar Pérez
 * @date 14/02/2016
 * @brief File with the necessary implementation for extracting the program configuration parameters from command-line or from the XML configuration
 *
 */

/********************************* Includes *******************************/

#include "config.h"

using namespace std;

/********************************* Methods ********************************/

Config::Config(const char **argv, const int argc) {


	/************ Init the parser ***********/

	ezParser parser;
	parser.overview = "High computing: Genetic algorithm in OpenCL";
	parser.syntax = "./bin/hpmoonCL [OPTIONS]";
	parser.example = "./bin/hpmoonCL -h\n";
	parser.example += "./bin/hpmoonCL -conf \"config.xml\" -devt GPU -wg 4096 -wl 512 -db \"datos.txt\"\n";
	parser.example += "./bin/hpmoonCL -conf \"config.xml\" -ts 4 -maxf 85 -plotimg \"imgPareto\" -db \"datos.txt\"\n\n";
	parser.footer = "Cluster HPMoon (C) 2016\n";

	// Options
	parser.add("", 0, 0, 0, "Display usage instructions.", "-h"); // Display help
	parser.add("", 1, 1, 0, "Name of the file containing the XML configuration file.", "-conf"); // XML
	parser.add("", 0, 1, 0, "Name of the file containing the database.", "-db"); // DataBase
	parser.add("", 0, 1, 0, "Number of generations.", "-g"); // Number of generations
	parser.add("", 0, 1, 0, "Maximum number of features initially set to \"1\".", "-maxf"); // Max. features
	parser.add("", 0, 1, 0, "Number of individuals competing in the tournament.", "-ts"); // Tournament size
	parser.add("", 0, 1, 0, "Name of the file containing the fitness of the individuals in the first Pareto front.", "-plotdata"); // Gnuplot data
	parser.add("", 0, 1, 0, "Name of the file containing the gnuplot code for data display.", "-plotsrc"); // Gnuplot code
	parser.add("", 0, 1, 0, "Name of the file containing the image with the data (graphic).", "-plotimg"); // Gnuplot image
	parser.add("", 0, 1, 0, "Type of device that will run the program.", "-devt"); // Device type
	parser.add("", 0, 1, 0, "Name of the platform vendor that will run the program.", "-vendor"); // Platform vendor
	parser.add("", 0, 1, 0, "Name of the device that will run the program.", "-devn"); // Device name
	parser.add("", 0, 1, 0, "Total number of work-items (threads) that will run the program.", "-wg"); // Global work-items
	parser.add("", 0, 1, 0, "Number of work-items (threads) per compute unit that will run the program.", "-wl"); // Local work-items
	parser.add("", 0, 1, 0, "Maximum number of individuals to be processed in a single execution of the kernel.", "-maxind"); // Max. individuals
	parser.add("", 0, 1, 0, "Name of the file containing the kernels with the OpenCL code.", "-ke"); // Kernels

	parser.parse(argc, argv);
	string option;


	/************ Check important parameters ***********/

	// Help option
	if (parser.isSet("-h")) {
		parser.getUsage(option);
		fprintf(stdout, "%s", option.c_str());
		exit(0);
	}

	// Missing options
	vector<string> badOptions;
	if (!parser.gotRequired(badOptions)) {
		for (int i = 0; i < badOptions.size(); ++i) {
			fprintf(stderr, "ERROR: Missing required option %s\n", badOptions[i].c_str());
		}
		exit(-1);
	}

	// Missing arguments
	if (!parser.gotExpected(badOptions)) {
		for (int i = 0; i < badOptions.size(); ++i) {
			fprintf(stderr, "Got unexpected number of arguments for option %s\n\n", badOptions[i].c_str());
		}
		parser.getUsage(option);
		fprintf(stdout, "%s", option.c_str());
		exit(-1);
	}


	/************ Get the parameters ***********/

	////////////////////// -conf value
	parser.get("-conf") -> getString(option);
	XMLDocument configDoc;
	configDoc.LoadFile(option.c_str());


	////////////////////// -db value
	if (parser.isSet("-db")) {
		parser.get("-db") -> getString(option);
	}
	else {
		option = configDoc.FirstChildElement("Config") -> FirstChildElement("DataBaseFileName") -> GetText();
	}
	this -> dataBaseFileName = new char[option.length() + 1];
	strcpy(this -> dataBaseFileName, option.c_str());


	////////////////////// -g value
	if (parser.isSet("-g")) {
		parser.get("-g") -> getInt(this -> nGenerations);
	}
	else {
		configDoc.FirstChildElement("Config") ->  FirstChildElement("NGenerations") -> QueryIntText(&(this -> nGenerations));
	}


	////////////////////// -maxf value
	if (parser.isSet("-maxf")) {
		parser.get("-maxf") -> getInt(this -> maxFeatures);
	}
	else {
		configDoc.FirstChildElement("Config") ->  FirstChildElement("MaxFeatures") -> QueryIntText(&(this -> maxFeatures));
	}


	////////////////////// -ts value
	if (parser.isSet("-ts")) {
		parser.get("-ts") -> getInt(this -> tourSize);
	}
	else {
		configDoc.FirstChildElement("Config") ->  FirstChildElement("TournamentSize") -> QueryIntText(&(this -> tourSize));
	}


	////////////////////// -plotdata value
	if (parser.isSet("-plotdata")) {
		parser.get("-plotdata") -> getString(option);
	}
	else {
		option = configDoc.FirstChildElement("Config") ->  FirstChildElement("DataFileName") -> GetText();
	}
	this -> dataFileName = new char[option.length() + 1];
	strcpy(this -> dataFileName, option.c_str());


	////////////////////// -plotsrc value
	if (parser.isSet("-plotsrc")) {
		parser.get("-plotsrc") -> getString(option);
	}
	else {
		option = configDoc.FirstChildElement("Config") ->  FirstChildElement("PlotFileName") -> GetText();
	}
	this -> plotFileName = new char[option.length() + 1];
	strcpy(this -> plotFileName, option.c_str());


	////////////////////// -plotimg value
	if (parser.isSet("-plotimg")) {
		parser.get("-plotimg") -> getString(option);
	}
	else {
		option = configDoc.FirstChildElement("Config") ->  FirstChildElement("ImageFileName") -> GetText();
	}
	this -> imageFileName = new char[option.length() + 1];
	strcpy(this -> imageFileName, option.c_str());


	////////////////////// -devt value
	const char *aux;
	if (parser.isSet("-devt")) {
		parser.get("-devt") -> getString(option);
		aux = option.c_str();
	}
	else {
		aux = configDoc.FirstChildElement("Config") -> FirstChildElement("OpenCL") -> FirstChildElement("DeviceType") -> GetText();
	}
	if (strcmp(aux, "CPU") == 0) {
		this -> deviceType = CL_DEVICE_TYPE_CPU;
	}
	else if (strcmp(aux, "GPU") == 0) {
		this -> deviceType = CL_DEVICE_TYPE_GPU;
	}
	else {
		fprintf(stderr, "Error: The device type specified in configuration must be CPU or GPU\n");
		exit(-1);
	}


	////////////////////// -vendor value
	if (parser.isSet("-vendor")) {
		parser.get("-vendor") -> getString(option);
	}
	else {
		if (this -> deviceType == CL_DEVICE_TYPE_CPU) {
			option = configDoc.FirstChildElement("Config") ->  FirstChildElement("OpenCL") -> FirstChildElement("CpuPlatformVendor") -> GetText();
		}
		else {
			option = configDoc.FirstChildElement("Config") ->  FirstChildElement("OpenCL") -> FirstChildElement("GpuPlatformVendor") -> GetText();
		}
	}
	this -> platformVendor = new char[option.length() + 1];
	strcpy(this -> platformVendor, option.c_str());


	////////////////////// -devn value
	if (parser.isSet("-devn")) {
		parser.get("-devn") -> getString(option);
	}
	else {
		if (this -> deviceType == CL_DEVICE_TYPE_CPU) {
			option = configDoc.FirstChildElement("Config") ->  FirstChildElement("OpenCL") -> FirstChildElement("CpuDeviceName") -> GetText();
		}
		else {
			option = configDoc.FirstChildElement("Config") ->  FirstChildElement("OpenCL") -> FirstChildElement("GpuDeviceName") -> GetText();
		}
	}
	this -> deviceName = new char[option.length() + 1];
	strcpy(this -> deviceName, option.c_str());


	////////////////////// -wg value
	if (parser.isSet("-wg")) {
		parser.get("-wg") -> getString(option);
		this -> wiGlobal = (size_t) atoi(option.c_str());
	}
	else {
		this -> wiGlobal = (size_t) atoi(configDoc.FirstChildElement("Config") -> FirstChildElement("OpenCL") -> FirstChildElement("WiGlobal") -> GetText());
	}


	////////////////////// -wl value
	if (parser.isSet("-wl")) {
		parser.get("-wl") -> getString(option);
		this -> wiLocal = (size_t) atoi(option.c_str());
	}
	else {
		this -> wiLocal = (size_t) atoi(configDoc.FirstChildElement("Config") -> FirstChildElement("OpenCL") -> FirstChildElement("WiLocal") -> GetText());
	}


	////////////////////// If the device is the CPU, the local work-size must be "1". On the contrary the program will fail
	if (this -> deviceType == CL_DEVICE_TYPE_CPU && this -> wiLocal != 1) {
		this -> wiLocal = 1;
		fprintf(stderr, "Warning: If the device is the CPU, the local work-size must be 1. Local work-size has been set to 1\n");
	}


	////////////////////// -maxind value
	if (parser.isSet("-maxind")) {
		parser.get("-maxind") -> getInt(this -> maxIndividualsOnGpuKernel);
	}
	else {
		configDoc.FirstChildElement("Config") -> FirstChildElement("OpenCL") -> FirstChildElement("MaxIndividualsOnGpuKernel") -> QueryIntText(&(this -> maxIndividualsOnGpuKernel));
	}


	////////////////////// -ke value
	if (parser.isSet("-ke")) {
		parser.get("-ke") -> getString(option);
	}
	else {
		option = configDoc.FirstChildElement("Config") ->  FirstChildElement("OpenCL") -> FirstChildElement("KernelsFileName") -> GetText();
	}
	this -> kernelsFileName = new char[option.length() + 1];
	strcpy(this -> kernelsFileName, option.c_str());
}