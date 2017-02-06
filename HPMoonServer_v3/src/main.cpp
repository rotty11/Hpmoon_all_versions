/**
 * @file main.cpp
 * @author Juan José Escobar Pérez
 * @date 08/07/2015
 * @brief Multiobjective genetic algorithm
 *
 * Multiobjective genetic algorithm running on a general purpose processor
 *
 */

/********************************* Includes ********************************/

#include "bd.h"
#include "ag.h"
#include "evaluation.h"


/**
 * @brief Main program
 * @param argc The number of arguments of the program
 * @param argv Arguments of the program
 * @return Returns nothing if successful or a negative number if incorrect
 */
int main(const int argc, const char **argv) {


	/********** Get the configuration data from the XML file or from the command-line ***********/

	const Config conf(argv, argc);

	// Generic variables
	float dataBase[conf.nInstances * conf.nFeatures];
	float dataBaseTransposed[conf.nInstances * conf.nFeatures];
	int selInstances[conf.nCentroids];
	double referencePoint[conf.nObjectives];
	Individual populationOrig[conf.totalIndividuals];


	/********** Get the data base and its normalization ***********/

	readDataBase(dataBase, &conf);
	normDataBase(dataBase, &conf);
	transposeDataBase(dataBase, dataBaseTransposed, &conf); // Matrix transposed


	/********** Get the initial "conf.nCentroids" centroids ***********/

	getCentroids(selInstances, &conf);


	/********** Set the reference point ***********/

	// The reference point will be (X_1 = 1.0, X_2 = 1.0, .... X_conf.nObjectives = 1.0)
	for (int i = 0; i < conf.nObjectives; ++i) {
		referencePoint[i] = 1.0;
	}


	/********** Initialize the population and the individuals ***********/

	srand((u_int) time(NULL));

	// Population will have the parents and children (left half and right half respectively)
	// This way is better for the performance
	initPopulation(populationOrig, &conf);


	/********** OpenCL init ***********/

	// OpenCL mode
	if (conf.nDevices > 0) {

		CLDevice *devices = createDevices(dataBase, selInstances, dataBaseTransposed, &conf);

		// Benchmark mode (Sequential, heterogeneous and all devices per separate)
		if (conf.benchmarkMode) {

			// Each device per separate
			for (int i = 0; i < conf.nDevices; ++i) {
				agAlgorithm(populationOrig, devices[i].deviceType, &devices[i], NULL, NULL, referencePoint, &conf);
			}

			// Heterogeneous mode if more than 1 device is available
			if (conf.nDevices > 1) {
				agAlgorithm(populationOrig, CL_DEVICE_TYPE_ALL, devices, NULL, NULL, referencePoint, &conf);
			}

			// Sequential mode
			agAlgorithm(populationOrig, CL_DEVICE_TYPE_DEFAULT, NULL, dataBase, selInstances, referencePoint, &conf);
		}

		// Only 1 device (CPU or GPU)
		else if (conf.nDevices == 1) {
			agAlgorithm(populationOrig, devices[0].deviceType, &devices[0], NULL, NULL, referencePoint, &conf);
		}

		// Heterogeneous mode if more than 1 device is available
		else {
			agAlgorithm(populationOrig, CL_DEVICE_TYPE_ALL, devices, NULL, NULL, referencePoint, &conf);
		}


		/********** Resources used are released ***********/

		// OpenCL resources
		delete[] devices;
	}

	// Sequential mode
	else {
		agAlgorithm(populationOrig, CL_DEVICE_TYPE_DEFAULT, NULL, dataBase, selInstances, referencePoint, &conf);
	}

	// Generation of the Gnuplot file for display the Pareto front
	generateGnuplot(referencePoint, &conf);
}