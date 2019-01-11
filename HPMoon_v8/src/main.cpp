/**
 * @file main.cpp
 * @author Juan José Escobar Pérez
 * @date 08/07/2015
 * @brief Multi-objective genetic algorithm
 *
 * Multi-objective genetic algorithm running on a general purpose processor
 */

/********************************* Includes ********************************/

#include "bd.h"
#include "ag.h"
#include "evaluation.h"


/**
 * @brief Main program
 * @param argc The number of arguments of the program
 * @param argv Arguments of the program
 */
int main(const int argc, const char **argv) {


	/********** Initialize the MPI environment ***********/

	MPI::Init_thread(MPI_THREAD_MULTIPLE);


	/********** Get the configuration data from the XML file or from the command-line ***********/

	Config conf(argc, argv);
	Individual *subpops;
	int *selInstances;
	srand((uint) time(NULL) + conf.mpiRank); // "+ rank" is necessary in MPI

	// Master
	if (conf.mpiRank == 0 && conf.mpiSize > 1) {

		// Initialize the subpopulations and the individuals
		// Subpopulations will have the parents and children (left half and right half respectively)
		subpops = createSubpopulations(&conf);

		// Get the initial "conf.K" centroids and share them with the workers
		selInstances = getCentroids(&conf);
		MPI::COMM_WORLD.Bcast(selInstances, conf.K, MPI::INT, 0);


		/********** Genetic algorithm ***********/

		agIslands(subpops, NULL, NULL, NULL, &conf);
	}

	// Workers
	else {

		// Get the databases and its normalization if it is required
		const float *const trDataBase = getDataBase(&conf);
		const float *const transposedTrDataBase = transposeDataBase(trDataBase, &conf); // Transposed database

		// I am the master and I work alone
		if (conf.mpiSize == 1) {
			subpops = createSubpopulations(&conf);
			selInstances = getCentroids(&conf);
		}

		// Get the initial "conf.K" centroids from the master
		else {
			selInstances = new int[conf.K];
			MPI::COMM_WORLD.Bcast(selInstances, conf.K, MPI::INT, 0);
		}


		/********** Genetic algorithm ***********/

		// Sequential, only 1 device (CPU or GPU) or heterogeneous mode if more than 1 device is available
		CLDevice *devices = createDevices(trDataBase, selInstances, transposedTrDataBase, &conf);
		agIslands(subpops, devices, trDataBase, selInstances, &conf);

		// Exclusive variables used by the workers are released
		delete[] devices;
		delete[] trDataBase;
		delete[] transposedTrDataBase;
	}

	// Variables used by both master and workers are released
	delete[] subpops;
	delete[] selInstances;

	// Finish the MPI environment
	MPI::Finalize();
}
