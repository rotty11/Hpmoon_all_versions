/**
 * @file main.cpp
 * @author Juan José Escobar Pérez
 * @date 08/07/2015
 * @brief Multi-objective genetic algorithm
 *
 * Multi-objective genetic algorithm running on a general purpose processor
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


	/********** Get the data base and its normalization ***********/

	const float *const dataBase = getDataBase(&conf);
	const float *const dataBaseTransposed = transposeDataBase(dataBase, &conf); // Matrix transposed


	/********** Get the initial "conf.nCentroids" centroids ***********/

	srand((uint) time(NULL));
	const int *const selInstances = getCentroids(&conf);


	/********** Set the reference point ***********/

	// The reference point will be (X_1 = 1.0, X_2 = 1.0, .... X_conf.nObjectives = 1.0)
	double *const referencePoint = new double[conf.nObjectives];
	for (int i = 0; i < conf.nObjectives; ++i) {
		referencePoint[i] = 1.0;
	}


	/********** Initialize the subpopulations and the individuals ***********/

	// Subpopulations will have the parents and children (left half and right half respectively)
	// This way is better for the performance
	const Individual *const subpopsOrig = initSubpopulations(&conf);


	/********** Genetic algorithm ***********/

	CLDevice *devices = createDevices(dataBase, selInstances, dataBaseTransposed, &conf);

	// Benchmark mode (Sequential, heterogeneous and all devices per separate)
	if (conf.benchmarkMode) {

		// Each device per separate
		for (int i = 0; i < conf.nDevices; ++i) {
			agIslands(subpopsOrig, 1, &devices[i], NULL, NULL, referencePoint, &conf);
		}

		// Heterogeneous mode if more than 1 device is available
		if (conf.nDevices > 1) {
			agIslands(subpopsOrig, conf.nDevices, devices, NULL, NULL, referencePoint, &conf);
		}

		// Sequential mode
		agIslands(subpopsOrig, 0, NULL, dataBase, selInstances, referencePoint, &conf);
	}

	// Sequential, only 1 device (CPU or GPU) or heterogeneous mode if more than 1 device is available
	else {
		agIslands(subpopsOrig, conf.nDevices, devices, dataBase, selInstances, referencePoint, &conf);
	}

	// Generation of the Gnuplot file for display the Pareto front
	generateGnuplot(referencePoint, &conf);


	/********** Resources used are released ***********/

	delete[] devices;
	delete[] dataBase;
	delete[] dataBaseTransposed;
	delete[] selInstances;
	delete[] referencePoint;
	delete[] subpopsOrig;
}
