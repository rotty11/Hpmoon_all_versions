/**
 * @file config.cpp
 * @author Juan José Escobar Pérez
 * @date 14/02/2016
 * @brief File with the necessary implementation for extracting the program configuration parameters from command-line or from the XML configuration
 */

/********************************* Includes *******************************/

#include "clUtils.h"
#include "cmdParser.h"
#include "tinyxml2.h"
#include <mpi.h>
#include <sstream> // stringstream...

using namespace tinyxml2;

/********************************* Methods ********************************/

/**
 * @brief The destructor
 */
Config::~Config() {

	// Resources used are released
	if (this != NULL && (this -> ompThreads == 0 || this -> nDevices > 1)) {
		delete[] this -> devices;
		delete[] this -> computeUnits;
		delete[] this -> wiLocal;
	}
}


/**
 * @brief The constructor with parameters
 * @param argc Number of arguments
 * @param argv The command-line parameters
 * @return An object containing all configuration parameters
 */
Config::Config(const int argc, const char **argv) {

	int rank = MPI::COMM_WORLD.Get_rank();
	int size = MPI::COMM_WORLD.Get_size();


	/************ Init the parser ***********/

	CmdParser parser("High computing: Genetic algorithm in OpenCL and OpenMP, and distributed using OpenMPI.", "mpirun [MPI OPTIONS] ./bin/hpmoon [ARGS]", "Cluster HPMoon (C) 2018 v8.0");
	parser.addExample("./bin/hpmoon -h");
	parser.addExample("./bin/hpmoon -l");
	parser.addExample("mpirun --bind-to none --map-by node --host localhost ./bin/hpmoon -conf \"config.xml\" -ns 2 -trdb \"db/TRdata.txt\"");
	parser.addExample("mpirun --bind-to none --map-by node --host node0,localhost ./bin/hpmoon -conf \"config.xml\" -ss 480 -ngm 3 -trdb \"db/TRdata.txt\" -trnorm");
	parser.addExample("mpirun --bind-to none --map-by node --host node0,node1 ./bin/hpmoon -conf \"config.xml\" -ts 4 -maxf 85 -plotimg \"imgPareto\"");

	// Options
	parser.addArg("-h", false, "Display usage instructions."); // Display help
	parser.addArg("-l", false, "List all available OpenCL devices."); // List OpenCL devices
	parser.addArg("-conf", true, "Name of the file containing the XML configuration file."); // XML
	parser.addArg("-ns", true, "Number of subpopulations (only for islands-based model)."); // Number of subpopulations
	parser.addArg("-ss", true, "Number of individuals in each subpopulation."); // Subpopulation size
	parser.addArg("-ngm", true, "Number of migrations between subpopulations of different nodes."); // Number of global migrations
	parser.addArg("-g", true, "Number of generations before each migration."); // Number of generations
	parser.addArg("-maxf", true, "Maximum number of features initially set to \"1\"."); // Max. features
	parser.addArg("-plotdata", true, "Name of the file containing the fitness of the individuals in the first Pareto front."); // Gnuplot data
	parser.addArg("-plotsrc", true, "Name of the file containing the gnuplot code for data display."); // Gnuplot code
	parser.addArg("-plotimg", true, "Name of the file containing the image with the data (graphic)."); // Gnuplot image
	parser.addArg("-trni", true, "Maximum number of instances to be taken from the training database."); // Maximum number of training instances
	parser.addArg("-trdb", true, "Name of the file containing the training database."); // Training database
	parser.addArg("-trnorm", false, "If the training database must be normalized or not."); // Normalization of the training database
	parser.addArg("-ts", true, "Number of individuals competing in the tournament."); // Tournament size
	parser.addArg("-ke", true, "Name of the file containing the kernels with the OpenCL code."); // Kernels

	// Parse and check the missing arguments
	check(!parser.parse(argv, argc), "%s\n", CFG_ERROR_PARSE_ARGUMENTS);


	/************ Check important parameters ***********/

	// Display help
	if (parser.isSet("-h")) {
		if (rank == 0) {
			parser.printHelp();
		}
		MPI::Finalize();
		exit(0);
	}

	// List OpenCL devices
	if (parser.isSet("-l")) {
		listDevices(rank);
		MPI::Finalize();
		exit(0);
	}


	/************ Get the XML/command-line parameters ***********/

	////////////////////// -conf value
	XMLDocument configDoc;
	check(configDoc.LoadFile(parser.getValue<char*>("-conf")) != XML_SUCCESS, "%s\n", CFG_ERROR_XML_READ);
	XMLElement *root = configDoc.FirstChildElement();


	////////////////////// -ns value
	if (parser.isSet("-ns")) {
		this -> nSubpopulations = parser.getValue<int>("-ns");
	}
	else {
		root -> FirstChildElement("NSubpopulations") -> QueryIntText(&(this -> nSubpopulations));
	}
	check(this -> nSubpopulations < 1, "%s\n", CFG_ERROR_SUBPOPS_MIN);


	////////////////////// -ss value
	if (parser.isSet("-ss")) {
		this -> subpopulationSize = parser.getValue<int>("-ss");
	}
	else {
		root -> FirstChildElement("SubpopulationSize") -> QueryIntText(&(this -> subpopulationSize));
	}
	check(this -> subpopulationSize < 4, "%s\n", CFG_ERROR_SUBPOPS_SIZE);


	////////////////////// -ngm value
	if (parser.isSet("-ngm")) {
		this -> nGlobalMigrations = parser.getValue<int>("-ngm");
	}
	else {
		root -> FirstChildElement("NGlobalMigrations") -> QueryIntText(&(this -> nGlobalMigrations));
	}
	check(this -> nGlobalMigrations < 1, "%s\n", CFG_ERROR_MIGRATIONS_MIN);
	check(this -> nGlobalMigrations > 1 && this -> nSubpopulations == 1, "%s\n", CFG_ERROR_MIGRATIONS_ONE);


	////////////////////// -g value
	if (parser.isSet("-g")) {
		this -> nGenerations = parser.getValue<int>("-g");
	}
	else {
		root -> FirstChildElement("NGenerations") -> QueryIntText(&(this -> nGenerations));
	}
	check(this -> nGenerations < 0, "%s\n", CFG_ERROR_GENERATIONS_MIN);
	check(this -> nGenerations == 0 && (this -> nGlobalMigrations > 1), "%s\n", CFG_ERROR_GENERATIONS_ONE);


	////////////////////// -maxf value
	if (parser.isSet("-maxf")) {
		this -> maxFeatures = parser.getValue<int>("-maxf");
	}
	else {
		root -> FirstChildElement("MaxFeatures") -> QueryIntText(&(this -> maxFeatures));
	}
	check(this -> maxFeatures < 1, "%s\n", CFG_ERROR_MAXFEAT_MIN);


	////////////////////// -plotdata value
	this -> dataFileName = (parser.isSet("-plotdata")) ? parser.getValue<char*>("-plotdata") : root -> FirstChildElement("DataFileName") -> GetText();


	////////////////////// -plotsrc value
	this -> plotFileName = (parser.isSet("-plotsrc")) ? parser.getValue<char*>("-plotsrc") : root -> FirstChildElement("PlotFileName") -> GetText();


	////////////////////// -plotimg value
	this -> imageFileName = (parser.isSet("-plotimg")) ? parser.getValue<char*>("-plotimg") : root -> FirstChildElement("ImageFileName") -> GetText();


	////////////////////// -trni value
	XMLElement *parent = root -> FirstChildElement("TrDatabase");
	if (parser.isSet("-trni")) {
		this -> trNInstances = parser.getValue<int>("-trni");
	}
	else {
		parent -> FirstChildElement("NInstances") -> QueryIntText(&(this -> trNInstances));
	}

	////////////////////// -trdb value
	this -> trDataBaseFileName = (parser.isSet("-trdb")) ? parser.getValue<char*>("-trdb") : parent -> FirstChildElement("FileName") -> GetText();


	////////////////////// -trnorm value
	if (parser.isSet("-trnorm")) {
		this -> trNormalize = true;
	}
	else {
		parent -> FirstChildElement("Normalize") -> QueryBoolText(&(this -> trNormalize));
	}


	////////////////////// -ts value
	if (parser.isSet("-ts")) {
		this -> tourSize = parser.getValue<int>("-ts");
	}
	else {
		root -> FirstChildElement("TournamentSize") -> QueryIntText(&(this -> tourSize));
	}
	check(this -> tourSize < 2 || this -> tourSize > this -> subpopulationSize, "%s\n", CFG_ERROR_TOURNAMENT_SIZE);

	if (rank > 0 || (rank == 0 && size == 1)) {

		////////////////////// Devices number
		parent = root -> FirstChildElement("Devices") -> FirstChildElement();
		for (int i = 1; i < rank; ++i) {
			parent = parent -> NextSiblingElement("NDevices");
			check(parent == NULL, "%s\n", CFG_ERROR_OPENCL_INFO);
		}
		parent -> QueryIntText(&(this -> nDevices));
		check(this -> nDevices < 0, "%s\n", CFG_ERROR_NDEVICES_MIN);

		// Check if OpenCL mode is active or not
		if (this -> nDevices > 0) {
			std::string option;

			////////////////////// Devices Name
			option = parent -> NextSiblingElement("Names") -> GetText();
			this -> nDevices = std::min(this -> nDevices, split(option, this -> devices));


			////////////////////// Compute Units
			option = parent -> NextSiblingElement("ComputeUnits") -> GetText();
			check(split(option, this -> computeUnits) < this -> nDevices, "%s\n", CFG_ERROR_CU_LOWER);


			////////////////////// Local work-items
			option = parent -> NextSiblingElement("WiLocal") -> GetText();
			check(split(option, this -> wiLocal) < this -> nDevices, "%s\n", CFG_ERROR_WI_LOWER);


			////////////////////// -ke value
			this -> kernelsFileName = (parser.isSet("-ke")) ? parser.getValue<char*>("-ke") : parent -> NextSiblingElement("KernelsFileName") -> GetText();
		}


		////////////////////// CPU threads value
		parent -> NextSiblingElement("CpuThreads") -> QueryIntText(&(this -> ompThreads));
		check(this -> ompThreads < 0 || (this -> ompThreads == 0 && this -> nDevices == 0), "%s\n", CFG_ERROR_THREADS_MIN);
	}


	/************ Set and get the internal parameters ***********/

	////////////////////// Number of centroids for K-means algorithm
	this -> K = 3;


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


	////////////////////// Number of features of the training database
	this -> nFeatures = N_FEATURES;
	check(N_FEATURES < 4, "%s\n", CFG_ERROR_FEATURES_MIN);


	////////////////////// Number of objectives
	this -> nObjectives = 2;


	////////////////////// MPI rank process
	this -> mpiRank = rank;


	////////////////////// Total number of processes in the global communicator
	this -> mpiSize = size;
	check(this -> mpiSize < 1, "%s\n", CFG_ERROR_SIZE_MIN);
}


/**
 * @brief Split a string into tokens separated by commas (,)
 * @param str The string to be split
 * @param tokens A pointer containing the tokens
 * @return The number of obtained tokens
 */
int split(const std::string str, std::string *&tokens) {

	std::stringstream ss(str);
	std::string token;
	int nTokens = 0;
	while(getline(ss, token, ',')) {
		++nTokens;
	}

	tokens = new std::string[nTokens];
	if (nTokens > 0) {
		std::stringstream aux(str);
		for (int i = 0; i < nTokens; ++i) {
			getline(aux, tokens[i], ',');
		}
	}

	return nTokens;
}


/**
 * @brief Check the condition. If it is true, a message is showed and the program will abort
 * @param cond The condition to be evaluated
 * @param format The format of the arguments
 * @param ... The corresponding messages to be showed in error case
 */
void check(const bool cond, const char *const format, ...) {

	if (cond) {
		va_list args;
		va_start(args, format);
		fprintf(stderr, "Process %d: ", MPI::COMM_WORLD.Get_rank());
		vfprintf(stderr, format, args);
		va_end(args);
		MPI::COMM_WORLD.Abort(-1);
	}
}
