/**
 * @file config.h
 * @author Juan José Escobar Pérez
 * @date 14/02/2016
 * @brief Header file for extracting the program configuration parameters from command-line or from the XML configuration. Also, internal parameters are considered
 */

#ifndef CONFIG_H
#define CONFIG_H

/********************************* Includes *******************************/

#include <fstream> // fstream

/******************************** Constants *******************************/

//const char *const CFG_ERROR_MISSING_ARGUMENT = "Error: Got unexpected number of arguments for option";
const char *const CFG_ERROR_PARSE_ARGUMENTS = "Error: Missing required value of the argument or nothing to parse\n";
const char *const CFG_ERROR_XML_READ = "Error: Could not read the XML file\n";
const char *const CFG_ERROR_SUBPOPS_MIN = "Error: The number of subpopulations must be 1 or higher\n";
const char *const CFG_ERROR_SUBPOPS_SIZE = "Error: The number of individuals per subpopulation must be 4 or higher\n";
const char *const CFG_ERROR_MIGRATIONS_MIN = "Error: The number of migrations must be 1 or higher\n";
const char *const CFG_ERROR_MIGRATIONS_ONE = "Error: If there is only one subpopulation, the number of migrations must be 1\n";
const char *const CFG_ERROR_GENERATIONS_MIN = "Error: The number of generations must be 0 or higher\n";
const char *const CFG_ERROR_GENERATIONS_ONE = "Error: If the number of generations is 0, the number of migrations must be 1\n";
const char *const CFG_ERROR_MAXFEAT_MIN = "Error: The maximum initial number of features must be 1 or higher\n";
const char *const CFG_ERROR_TOURNAMENT_SIZE = "Error: The number of individuals in the tournament must be between 2 and the subpopulation size\n";
const char *const CFG_ERROR_OPENCL_INFO = "Error: Specified in the XML file lower number of OpenCL device information than number of MPI slave processes\n";
const char *const CFG_ERROR_NDEVICES_MIN = "Error: The number of devices must be 0 or higher\n";
const char *const CFG_ERROR_CU_LOWER = "Error: Specified lower number of compute units than number of devices\n";
const char *const CFG_ERROR_WI_LOWER = "Error: Specified lower number of local work-items than number of devices\n";
const char *const CFG_ERROR_FEATURES_MIN = "Error: The number of features must be 4 or higher\n";
const char *const CFG_ERROR_SIZE_MIN = "Error: The minimum number of MPI processes must be 2 or higher\n";

/******************************** Structures ******************************/

/**
 * @brief Structure containing the configuration parameters
 */
typedef struct Config {


	/********************************* XML/command-line parameters ********************************/

	/**
	 * @brief The parameter indicating the number of subpopulations (only for islands-based model)
	 */
	int nSubpopulations;


	/**
	 * @brief The parameter indicating the number of individuals in each subpopulation
	 */
	int subpopulationSize;


	/**
	 * @brief The parameter indicating the number of instances of the database
	 */
	int nInstances;


	/**
	 * @brief The parameter indicating the name of the file containing the database
	 */
	std::string dataBaseFileName;


	/**
	 * @brief The parameter indicating the number of generations to generate before each migration
	 */
	int nGenerations;


	/**
	 * @brief The parameter indicating the maximum number of features initially set to "1"
	 */
	int maxFeatures;


	/**
	 * @brief The parameter indicating the name of the file containing the fitness of the individuals in the first Pareto front
	 */
	std::string dataFileName;


	/**
	 * @brief The parameter indicating the name of the file containing the gnuplot code for data display
	 */
	std::string plotFileName;


	/**
	 * @brief The parameter indicating the name of the file containing the image with the data (graphic)
	 */
	std::string imageFileName;


	/**
	 * @brief The parameter indicating the number of migrations between subpopulations of different nodes
	 */
	int nGlobalMigrations;


	/**
	 * @brief The parameter indicating the number of migrations between subpopulations of the same node
	 */
	int nLocalMigrations;


	/**
	 * @brief The parameter indicating the number of individuals competing in the tournament
	 */
	int tourSize;


	/**
	 * @brief The parameter indicating the number of OpenCL devices that will run the program
	 */
	int nDevices;


	/**
	 * @brief The parameter indicating the devices name that will run the program
	 */
	std::string *devices;


	/**
	 * @brief The parameter indicating the number of compute units of each device that will run the program
	 */
	std::string *computeUnits;


	/**
	 * @brief The parameter indicating the number of work-items (threads) per compute unit of each device that will run the program
	 */
	std::string *wiLocal;


	/**
	 * @brief The parameter indicating the name of the file containing the kernels with the OpenCL code
	 */
	std::string kernelsFileName;


	/********************************* Internal parameters ********************************/


	/**
	 * @brief The parameter indicating the number centroids (clusters)
	 */
	int nCentroids;


	/**
	 * @brief The parameter indicating the number of maximum iterations for the convergence of K-means
	 */
	int maxIterKmeans;


	/**
	 * @brief The parameter indicating the size of the pool (the half of the subpopulation size)
	 */
	int poolSize;


	/**
	 * @brief The parameter indicating the number of individuals in each subpopulation (including parents and children)
	 */
	int familySize;


	/**
	 * @brief The parameter indicating the number of individuals in the world
	 */
	int worldSize;


	/**
	 * @brief The parameter indicating the total number of individuals in the world (including parents and children)
	 */
	int totalIndividuals;


	/**
	 * @brief The parameter indicating the number of features of the database
	 */
	int nFeatures;


	/**
	 * @brief The parameter indicating the number of objectives
	 */
	unsigned char nObjectives;


	/**
	 * @brief The parameter indicating the MPI rank process
	 */
	int mpiRank;


	/**
	 * @brief The parameter indicating the total number of processes in the global communicator
	 */
	int mpiSize;


	/********************************* Methods ********************************/

	/**
	 * @brief The constructor with parameters
	 * @param argc Number of arguments
	 * @param argv The command-line parameters
	 * @return An object containing all configuration parameters
	 */
	Config(const int argc, const char **argv);


	/**
	 * @brief The destructor
	 */
	~Config();

} Config;

/********************************* Methods ********************************/

/**
 * @brief Split a string into tokens separated by commas (,)
 * @param str The string to be split
 * @param tokens A pointer containing the tokens
 * @return The number of obtained tokens
 */
int split(const std::string str, std::string *&tokens);


/**
 * @brief Check the condition. If it is true, a message is showed and the program will abort
 * @param cond The condition to be evaluated
 * @param message The corresponding message to be showed in error case
 * @param stop Indicates if the program must be aborted or not in error case
 */
void check(const bool cond, const char *const message = NULL, const bool stop = true);

#endif