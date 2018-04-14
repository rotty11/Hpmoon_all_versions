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
 * @return Returns nothing if successful or a negative number if incorrect
 */
int main(const int argc, const char **argv) {


	/********** Initialize the MPI environment ***********/

	MPI_Init(NULL, NULL);


	/********** Get the configuration data from the XML file or from the command-line ***********/

	const Config conf(argc, argv);
	srand((uint) time(NULL) + conf.mpiRank); // "+ rank" is necessary in MPI

	// Master
	if (conf.mpiRank == 0) {

		// Initialize the subpopulations and the individuals
		// Subpopulations will have the parents and children (left half and right half respectively)
		// This way is better for the performance
		Individual *subpops = initSubpopulations(&conf);

		// Get the initial "conf.nCentroids" centroids
		int *selInstances = getCentroids(&conf);
		MPI_Bcast(selInstances, conf.nCentroids, MPI_INT, 0, MPI_COMM_WORLD);


		/********** Genetic algorithm ***********/

		agIslands(subpops, NULL, NULL, NULL, false, &conf);

		// Generation of the Gnuplot file for display the Pareto front
		generateGnuplot(&conf);

		// Exclusive variables used by the master are released
		delete[] subpops;
		delete[] selInstances;
	}

	// Slaves
	else {

		// Get the databases and its normalization
		const float *const dataBase = getDataBase(&conf);
		const float *const dataBaseTransposed = transposeDataBase(dataBase, &conf); // Matrix transposed

		// Get the initial "conf.nCentroids" centroids from the master
		int selInstances[conf.nCentroids];
		MPI_Bcast(selInstances, conf.nCentroids, MPI_INT, 0, MPI_COMM_WORLD);


		/********** Genetic algorithm ***********/

		// Sequential, only 1 device (CPU or GPU) or heterogeneous mode if more than 1 device is available
		CLDevice *devices = createDevices(dataBase, selInstances, dataBaseTransposed, &conf);
		agIslands(NULL, devices, dataBase, selInstances, false, &conf);

		// Exclusive variables used by the slaves are released
		delete[] devices;
		delete[] dataBase;
		delete[] dataBaseTransposed;
	}

	// Finish the MPI environment
	MPI_Finalize();
}