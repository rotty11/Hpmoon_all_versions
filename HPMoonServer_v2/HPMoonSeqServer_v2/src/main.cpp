/**
 * @file main.cpp
 * @author Juan José Escobar Pérez
 * @date 15/06/2015
 * @brief Multiobjective genetic algorithm
 *
 * Multiobjective genetic algorithm running on a general purpose processor
 *
 */

/********************************* Includes ********************************/

#include "config.h"
#include "bd.h"
#include "initialization.h"
#include "evaluation.h"
#include "sort.h"
#include "tournament.h"
#include "crossover.h"


/**
 * @brief Main program
 * @param argc The number of arguments of the program
 * @param argv Arguments of the program
 * @return Returns nothing if successful or a negative number if incorrect
 */
int main(const int argc, const char **argv) {


	/********** Get the configuration data from the XML file or from the command-line ***********/

	const Config conf(argv, argc);

	// Time measure
	clock_t timeStart;


	/********** Check program restrictions ***********/

	if (POPULATION_SIZE < 4) {
		fprintf(stderr, "Error: The number of individuals must be 4 or higher\n");
		exit(-1);
	}

	if (N_FEATURES < 4 || N_INSTANCES < 4) {
		fprintf(stderr, "Error: The number of features and number of instances must be 4 or higher\n");
		exit(-1);
	}

	if (N_OBJECTIVES != 2) {
		fprintf(stderr, "Error: The number of objectives must be 2. If you want to increase this number, the module \"evaluation\" must be modified\n");
		exit(-1);
	}

	if (conf.maxFeatures < 1) {
		fprintf(stderr, "Error: The maximum initial number of features must be 1 or higher\n");
		exit(-1);
	}

	if (conf.tourSize < 2) {
		fprintf(stderr, "Error: The number of individuals in the tournament must be 2 or higher\n");
		exit(-1);
	}


	/********** Get the data base ***********/

	float dataBase[N_INSTANCES * N_FEATURES];
	readDataBase(dataBase, conf.dataBaseFileName, N_INSTANCES, N_FEATURES);

	// Data base normalization
	normDataBase(dataBase, N_INSTANCES, N_FEATURES);


	/******* Start the time measure *********/

	timeStart = clock();


	/********** Initialize the population and the individuals ***********/

	srand((unsigned int) time(NULL));
	const int totalIndividuals = POPULATION_SIZE << 1;

	// Population will have the parents and children (left half and right half respectively)
	// This way is better for the performance
	individual *population = initPopulation(totalIndividuals, N_OBJECTIVES, N_FEATURES, conf.maxFeatures);


	/********** Multiobjective individual evaluation ***********/

	// Get the initial "KMEANS" centroids ***********/
	int selInstances[KMEANS];
	getCentroids(selInstances, N_INSTANCES);

	evaluation(population, 0, POPULATION_SIZE, dataBase, N_INSTANCES, N_FEATURES, N_OBJECTIVES, selInstances);


	/********** Sort the population with the "Non-Domination-Sort" method ***********/

	int nIndFront0 = nonDominationSort(population, POPULATION_SIZE, N_OBJECTIVES, N_INSTANCES, N_FEATURES);


	/********** Get the population quality (calculating the hypervolume) ***********/

	// The reference point will be (X_1 = 1.0, X_2 = 1.0, .... X_N_OBJECTIVES = 1.0)
	double referencePoint[N_OBJECTIVES];
	for (int i = 0; i < N_OBJECTIVES; ++i) {
		referencePoint[i] = 1.0;
	}

	float popHypervolume = getHypervolume(population, nIndFront0, N_OBJECTIVES, referencePoint);


	/********** Start the evolution process ***********/

	const int poolSize = POPULATION_SIZE >> 1;
	int pool[poolSize];
	for (int g = 0; g < conf.nGenerations; ++g) {

		// Fill the mating pool
		fillPool(pool, poolSize, conf.tourSize, POPULATION_SIZE);

		// Perform crossover
		int nChildren = crossover(population, POPULATION_SIZE, pool, poolSize, N_OBJECTIVES, N_FEATURES);

		// Multiobjective individual evaluation
		int lastChild = POPULATION_SIZE + nChildren;
		evaluation(population, POPULATION_SIZE, lastChild, dataBase, N_INSTANCES, N_FEATURES, N_OBJECTIVES, selInstances);
		
		// The crowding distance of the parents is initialized again for the next nonDominationSort
		for (int i = 0;  i < POPULATION_SIZE; ++i) {
			population[i].crowding = 0.0f;
		}

		// Replace population
		// Parents and children are sorted by rank and crowding distance.
		// The first "populationSize" individuals will advance the next generation
		nIndFront0 = nonDominationSort(population, POPULATION_SIZE + nChildren, N_OBJECTIVES, N_INSTANCES, N_FEATURES);

		// Get the population quality (calculating the hypervolume)
		popHypervolume = getHypervolume(population, nIndFront0, N_OBJECTIVES, referencePoint);
	}

	// Finish the time measure
	double ms = ((double) (clock() - timeStart) / CLOCKS_PER_SEC) * 1000.0;
	fprintf(stdout, "%.16g\t%.6g\n", ms, popHypervolume);

	// Generation of the data file and Gnuplot file for display the Pareto front
	generateGnuplot(conf.dataFileName, conf.plotFileName, conf.imageFileName, population, nIndFront0, N_OBJECTIVES, referencePoint);


	/********** Resources used are released ***********/

	// The individuals (parents and children)
	delete[] population;
}