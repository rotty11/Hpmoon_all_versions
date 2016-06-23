/**
 * @file config.h
 * @author Juan José Escobar Pérez
 * @date 14/02/2016
 * @brief Header file for extracting the program configuration parameters from command-line or from the XML configuration
 *
 */

#ifndef CONFIG_H
#define CONFIG_H

/********************************* Includes *******************************/

#include "tinyxml2.h"
#include "ezParser.h"
#include <CL/cl.h> // OpenCL

using namespace tinyxml2;
using namespace ez;

/******************************** Structures ******************************/

/**
 * @brief Structure containing the configuration parameters
 */
typedef struct Config {


	/**
	 * @brief The parameter indicating the name of the file containing the database
	 */
	char *dataBaseFileName;


	/**
	 * @brief The parameter indicating the number of generations to generate (iterations of the program)
	 */
	int nGenerations;


	/**
	 * @brief The parameter indicating the maximum number of features initially set to "1"
	 */
	int maxFeatures;


	/**
	 * @brief The parameter indicating the number of individuals competing in the tournament
	 */
	int tourSize;


	/**
	 * @brief The parameter indicating the name of the file containing the fitness of the individuals in the first Pareto front
	 */
	char *dataFileName;


	/**
	 * @brief The parameter indicating the name of the file containing the gnuplot code for data display
	 */
	char *plotFileName;


	/**
	 * @brief The parameter indicating the name of the file containing the image with the data (graphic)
	 */
	char *imageFileName;


	/**
	 * @brief The parameter indicating the type of device that will run the program
	 */
	cl_device_type deviceType;


	/**
	 * @brief The parameter indicating the name of the device vendor that will run the program
	 */
	char *platformVendor;


	/**
	 * @brief The parameter indicating the name of the device model that will run the program
	 */
	char *deviceName;


	/**
	 * @brief The parameter indicating the total number of work-items (threads) that will run the program
	 */
	size_t wiGlobal;


	/**
	 * @brief The parameter indicating the number of work-items (threads) per compute unit that will run the program
	 */
	size_t wiLocal;


	/**
	 * @brief The parameter indicating the maximum number of individuals to be processed in a single execution of the kernel
	 */
	int maxIndividualsOnGpuKernel;


	/**
	 * @brief The parameter indicating the name of the file containing the kernels with the OpenCL code
	 */
	char *kernelsFileName;


	/********************************* Methods ********************************/

	/**
	* @brief The constructor
	* @param argv The command-line parameters
	* @param argc Number of arguments
	*/
	Config(const char **argv, const int argc);
} Config;

#endif