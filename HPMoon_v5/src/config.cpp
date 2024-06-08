/**
 * @file config.cpp
 * @author Juan José Escobar Pérez
 * @date 14/02/2016
 * @brief File with the necessary implementation for extracting the program configuration parameters from command-line or from the XML configuration
 *
 */

/********************************* Includes *******************************/

#include "clUtils.h"
#include "tinyxml2.h"
#include "ezParser.h"

using namespace tinyxml2;
using namespace ez;

/********************************* Methods ********************************/

/**
 * @brief The destructor
 */
Config::~Config() {

	if (this != NULL && this -> nDevices > 0) {

		// Resources used are released
		delete[] this -> devices;
		delete[] this -> computeUnits;
		delete[] this -> wiLocal;
	}
}


/**
 * @brief The constructor with parameters
 * @param argv The command-line parameters
 * @param argc Number of arguments
 * @return An object containing all the configuration parameters
 */
Config::Config(const char **argv, const int argc) {


	/************ Init the parser ***********/

	ezParser parser;
	parser.overview = "High computing: Genetic algorithm in OpenCL and OpenMP";
	parser.syntax = "./bin/hpmoon [OPTIONS]";
	parser.example = "./bin/hpmoon -h\n";
	parser.example += "./bin/hpmoon -l\n";
	parser.example += "./bin/hpmoon -conf \"config.xml\" -nd 3 -cu 8,13,24 -wl 512,1024,1 -db \"db/datos.txt\"\n";
	parser.example += "./bin/hpmoon -conf \"config.xml\" -nd 1 -cu 4 -wl 1 -db \"db/datos.txt\"\n";
	parser.example += "./bin/hpmoon -conf \"config.xml\" -nexec 1 -ts 4 -maxf 85 -plotimg \"imgPareto\" -db \"db/datos.txt\"\n\n";
	parser.footer = "Cluster HPMoon (C) 2017\n";

	// Options
	parser.add("", 0, 0, 0, "Display usage instructions.", "-h"); // Display help
	parser.add("", 0, 0, 0, "List all available OpenCL devices.", "-l"); // List OpenCL devices
	parser.add("", 1, 1, 0, "Name of the file containing the XML configuration file.", "-conf"); // XML
	parser.add("", 0, 1, 0, "Number of subpopulations (only for islands-based model).", "-ns"); // Number of subpopulations
	parser.add("", 0, 1, 0, "Number of individuals in each subpopulation.", "-ss"); // Subpopulation size
	parser.add("", 0, 1, 0, "Maximum number of instances to be taken from the database.", "-ni"); // Maximum number of instances
	parser.add("", 0, 1, 0, "Number of executions of the program (only for benchmarks).", "-nexec"); // Number of executions
	parser.add("", 0, 1, 0, "Name of the file containing the database.", "-db"); // DataBase
	parser.add("", 0, 1, 0, "Number of individuals migrations between subpopulations.", "-nm"); // Number of migrations
	parser.add("", 0, 1, 0, "Number of generations before each migration.", "-g"); // Number of generations
	parser.add("", 0, 1, 0, "Maximum number of features initially set to \"1\".", "-maxf"); // Max. features
	parser.add("", 0, 1, 0, "Number of individuals competing in the tournament.", "-ts"); // Tournament size
	parser.add("", 0, 1, 0, "Name of the file containing the fitness of the individuals in the first Pareto front.", "-plotdata"); // Gnuplot data
	parser.add("", 0, 1, 0, "Name of the file containing the gnuplot code for data display.", "-plotsrc"); // Gnuplot code
	parser.add("", 0, 1, 0, "Name of the file containing the image with the data (graphic).", "-plotimg"); // Gnuplot image
	parser.add("", 0, 1, 0, "Number of OpenCL devices that will run the program. Set to 0 to run in sequential mode.", "-nd"); // Number of devices
	parser.add("", 0, 1, 0, "The devices name that will run the program.", "-devn"); // Devices name
	parser.add("", 0, 1, 0, "Number of compute units that will run the program.", "-cu"); // Compute units
	parser.add("", 0, 1, 0, "Number of work-items (threads) per compute unit of each device that will run the program.", "-wl"); // Local work-items
	parser.add("", 0, 1, 0, "Maximum number of individuals to be processed in a single execution of the kernel.", "-maxind"); // Max. individuals
	parser.add("", 0, 1, 0, "Name of the file containing the kernels with the OpenCL code.", "-ke"); // Kernels

	parser.parse(argc, argv);
	string option;


	/************ Check important parameters ***********/

	// Display help
	if (parser.isSet("-h")) {
		parser.getUsage(option);
		fprintf(stdout, "%s", option.c_str());
		exit(0);
	}

	// List OpenCL devices
	if (parser.isSet("-l")) {
		listDevices();
		exit(0);
	}

	// Missing options
	vector<string> badOptions;
	if (!parser.gotRequired(badOptions)) {
		for (int i = 0; i < badOptions.size(); ++i) {
			fprintf(stderr, "Error: Missing required option %s\n", badOptions[i].c_str());
		}
		exit(-1);
	}

	// Missing arguments
	if (!parser.gotExpected(badOptions)) {
		for (int i = 0; i < badOptions.size(); ++i) {
			fprintf(stderr, "Error: Got unexpected number of arguments for option %s\n\n", badOptions[i].c_str());
		}
		parser.getUsage(option);
		fprintf(stdout, "%s", option.c_str());
		exit(-1);
	}


	/************ Get the XML/command-line parameters ***********/

	////////////////////// -conf value
	parser.get("-conf") -> getString(option);
	XMLDocument configDoc;
	configDoc.LoadFile(option.c_str());


	////////////////////// -ns value
	if (parser.isSet("-ns")) {
		parser.get("-ns") -> getInt(this -> nSubpopulations);
	}
	else {
		configDoc.FirstChildElement("Config") -> FirstChildElement("NSubpopulations") -> QueryIntText(&(this -> nSubpopulations));
	}
	if (this -> nSubpopulations < 1) {
		fprintf(stderr, "Error: The number of subpopulations must be 1 or higher\n");
		exit(-1);
	}


	////////////////////// -ss value
	if (parser.isSet("-ss")) {
		parser.get("-ss") -> getInt(this -> subpopulationSize);
	}
	else {
		configDoc.FirstChildElement("Config") -> FirstChildElement("SubpopulationSize") -> QueryIntText(&(this -> subpopulationSize));
	}
	if (this -> subpopulationSize < 4) {
		fprintf(stderr, "Error: The number of individuals must be 4 or higher\n");
		exit(-1);
	}


	////////////////////// -ni value
	if (parser.isSet("-ni")) {
		parser.get("-ni") -> getInt(this -> nInstances);
	}
	else {
		configDoc.FirstChildElement("Config") -> FirstChildElement("NInstances") -> QueryIntText(&(this -> nInstances));
	}


	////////////////////// -nexec value
	if (parser.isSet("-nexec")) {
		parser.get("-nexec") -> getInt(this -> nExecutions);
	}
	else {
		configDoc.FirstChildElement("Config") -> FirstChildElement("NExecutions") -> QueryIntText(&(this -> nExecutions));
	}
	if (this -> nExecutions < 1) {
		fprintf(stderr, "Error: The number of executions of the program must be 1 or higher\n");
		exit(-1);
	}


	////////////////////// -db value
	if (parser.isSet("-db")) {
		parser.get("-db") -> getString(this -> dataBaseFileName);
	}
	else {
		this -> dataBaseFileName = configDoc.FirstChildElement("Config") -> FirstChildElement("DataBaseFileName") -> GetText();
	}


	////////////////////// -nm value
	if (parser.isSet("-nm")) {
		parser.get("-nm") -> getInt(this -> nMigrations);
	}
	else {
		configDoc.FirstChildElement("Config") -> FirstChildElement("NMigrations") -> QueryIntText(&(this -> nMigrations));
	}
	if (this -> nMigrations < 1) {
		fprintf(stderr, "Error: The number of migrations must be 1 or higher\n");
		exit(-1);
	}
	if (this -> nMigrations > 1 && this -> nSubpopulations == 1) {
		fprintf(stderr, "Error: If there is only one subpopulation, the number of migrations must be 1\n");
		exit(-1);
	}


	////////////////////// -g value
	if (parser.isSet("-g")) {
		parser.get("-g") -> getInt(this -> nGenerations);
	}
	else {
		configDoc.FirstChildElement("Config") -> FirstChildElement("NGenerations") -> QueryIntText(&(this -> nGenerations));
	}
	if (this -> nGenerations < 0) {
		fprintf(stderr, "Error: The number of generations must be 0 or higher\n");
		exit(-1);
	}
	if (this -> nGenerations == 0 && this -> nMigrations > 1) {
		fprintf(stderr, "Error: If the number of generations is 0, the number of migrations must be 1\n");
		exit(-1);
	}


	////////////////////// -maxf value
	if (parser.isSet("-maxf")) {
		parser.get("-maxf") -> getInt(this -> maxFeatures);
	}
	else {
		configDoc.FirstChildElement("Config") -> FirstChildElement("MaxFeatures") -> QueryIntText(&(this -> maxFeatures));
	}
	if (this -> maxFeatures < 1) {
		fprintf(stderr, "Error: The maximum initial number of features must be 1 or higher\n");
		exit(-1);
	}


	////////////////////// -ts value
	if (parser.isSet("-ts")) {
		parser.get("-ts") -> getInt(this -> tourSize);
	}
	else {
		configDoc.FirstChildElement("Config") -> FirstChildElement("TournamentSize") -> QueryIntText(&(this -> tourSize));
	}
	if (this -> tourSize < 2) {
		fprintf(stderr, "Error: The number of individuals in the tournament must be 2 or higher\n");
		exit(-1);
	}


	////////////////////// -plotdata value
	if (parser.isSet("-plotdata")) {
		parser.get("-plotdata") -> getString(this -> dataFileName);
	}
	else {
		this -> dataFileName = configDoc.FirstChildElement("Config") -> FirstChildElement("DataFileName") -> GetText();
	}


	////////////////////// -plotsrc value
	if (parser.isSet("-plotsrc")) {
		parser.get("-plotsrc") -> getString(this -> plotFileName);
	}
	else {
		this -> plotFileName = configDoc.FirstChildElement("Config") -> FirstChildElement("PlotFileName") -> GetText();
	}


	////////////////////// -plotimg value
	if (parser.isSet("-plotimg")) {
		parser.get("-plotimg") -> getString(this -> imageFileName);
	}
	else {
		this -> imageFileName = configDoc.FirstChildElement("Config") -> FirstChildElement("ImageFileName") -> GetText();
	}


	////////////////////// -nd value
	if (parser.isSet("-nd")) {
		parser.get("-nd") -> getInt(this -> nDevices);
	}
	else {
		configDoc.FirstChildElement("Config") -> FirstChildElement("OpenCL") -> FirstChildElement("NDevices") -> QueryIntText(&(this -> nDevices));
	}

	// Check if OpenCL mode is active or not
	if (this -> nDevices != 0) {
		this -> benchmarkMode = (this -> nDevices < 0) ? true : false;
		this -> nDevices = abs(this -> nDevices);


		////////////////////// -devn value
		if (parser.isSet("-devn")) {
			parser.get("-devn") -> getString(option);
		}
		else {
			option = configDoc.FirstChildElement("Config") -> FirstChildElement("OpenCL") -> FirstChildElement("Devices") -> GetText();
		}
		this -> nDevices = min(this -> nDevices, split(option, this -> devices));


		////////////////////// -cu value
		if (parser.isSet("-cu")) {
			parser.get("-cu") -> getString(option);
		}
		else {
			option = configDoc.FirstChildElement("Config") -> FirstChildElement("OpenCL") -> FirstChildElement("ComputeUnits") -> GetText();
		}
		if (split(option, this -> computeUnits) < this -> nDevices) {
			fprintf(stderr, "Error: Specified lower number of compute units than number of devices\n");
			exit(-1);
		}


		////////////////////// -wl value
		if (parser.isSet("-wl")) {
			parser.get("-wl") -> getString(option);
		}
		else {
			option = configDoc.FirstChildElement("Config") -> FirstChildElement("OpenCL") -> FirstChildElement("WiLocal") -> GetText();
		}
		if (split(option, this -> wiLocal) < this -> nDevices) {
			fprintf(stderr, "Error: Specified lower number of local work-items than number of devices\n");
			exit(-1);
		}


		////////////////////// -maxind value
		if (parser.isSet("-maxind")) {
			parser.get("-maxind") -> getInt(this -> maxIndividualsOnGpuKernel);
		}
		else {
			configDoc.FirstChildElement("Config") -> FirstChildElement("OpenCL") -> FirstChildElement("MaxIndividualsOnGpuKernel") -> QueryIntText(&(this -> maxIndividualsOnGpuKernel));
		}
		if (this -> maxIndividualsOnGpuKernel < 1) {
			fprintf(stderr, "Error: The maximum of individuals on the GPU kernel must be 1 or higher\n");
			exit(-1);
		}


		////////////////////// -ke value
		if (parser.isSet("-ke")) {
			parser.get("-ke") -> getString(this -> kernelsFileName);
		}
		else {
			this -> kernelsFileName = configDoc.FirstChildElement("Config") -> FirstChildElement("OpenCL") -> FirstChildElement("KernelsFileName") -> GetText();
		}
	}
	else {
		this -> benchmarkMode = false;
	}


	/************ Set and get the internal parameters ***********/

	////////////////////// Number of centroids
	this -> nCentroids = 3;


	////////////////////// Maximum iterations for K-means algorithm
	this -> maxIterKmeans = 20;


	////////////////////// The size of the pool (the half of the subpopulation size)
	this -> poolSize = this -> subpopulationSize >> 1;


	////////////////////// Number of individuals in each subpopulation (including parents and children)
	this -> familySize = this -> subpopulationSize << 1;


	////////////////////// Number of individuals in the world
	this -> worldSize = this -> nSubpopulations * this -> subpopulationSize;


	////////////////////// Total number of individuals in the world (including parents and children)
	this -> totalIndividuals = this -> worldSize << 1;


	////////////////////// Number of features of the database
	if (N_FEATURES < 4) {
		fprintf(stderr, "Error: The number of features must be 4 or higher\n");
		exit(-1);
	}
	this -> nFeatures = N_FEATURES;


	////////////////////// Number of objectives
	this -> nObjectives = 2;
}


/**
 * @brief Split a string into tokens separated by commas (,)
 * @param str The string to be split
 * @param tokens A pointer containing the tokens
 * @return The number of obtained tokens
 */
int split(const string str, string *&tokens) {

	stringstream ss(str);
	string token;
	int nTokens = 0;
	while(getline(ss, token, ',')) {
		++nTokens;
	}

	tokens = new string[nTokens];
	if (nTokens > 0) {
		stringstream aux(str);
		for (int i = 0; i < nTokens; ++i) {
			getline(aux, tokens[i], ',');
		}
	}

	return nTokens;
}