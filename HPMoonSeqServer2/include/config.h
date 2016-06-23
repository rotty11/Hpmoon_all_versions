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


	/********************************* Methods ********************************/

	/**
	* @brief The constructor
	* @param argv The command-line parameters
	* @param argc Number of arguments
	*/
	Config(const char **argv, const int argc);
} Config;

#endif