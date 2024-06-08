/**
 * @file ag.cpp
 * @author Juan José Escobar Pérez
 * @date 14/07/2016
 * @brief File with the necessary implementation for the main functions of the genetic algorithm
 *
 */

/********************************* Includes *******************************/

#include "ag.h"
#include "evaluation.h"
#include <string.h> // memset...
#include <omp.h> // OpenMP

/********************************* Methods ********************************/


/**
 * @brief Allocates memory for parents and children. Also, they are initialized
 * @param population The first population
 * @param conf The structure with all configuration parameters
 */
void initPopulation(Individual *population, const Config *conf) {


	/********** Initialization of the population and the individuals ***********/

	// Allocates memory for parents and children
	for (int i = 0; i < conf -> totalIndividuals; ++i) {
		memset(population[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
			population[i].fitness[obj] = 0.0f;
		}
		population[i].nSelFeatures = 0;
		population[i].rank = -1;
		population[i].crowding = 0.0f;
	}

	// Only the parents are initialized
	for (int i = 0; i < conf -> populationSize; ++i) {

		// Set the "1" value at most "conf -> maxFeatures" decision variables
		for (int mf = 0; mf < conf -> maxFeatures; ++mf) {
			int randomFeature = rand() % conf -> nFeatures;
			if (!(population[i].chromosome[randomFeature] & 1)) {
				population[i].nSelFeatures += (population[i].chromosome[randomFeature] = 1);
				
			}
		}
	}
}


/**
 * @brief Competition between randomly selected individuals. The best individuals are stored in the pool
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 */
void fillPool(int *pool, const Config *conf) {

	// Fill pool
	for (int i = 0; i < conf -> poolSize; ++i) {
		int bestCandidate = rand() % conf -> populationSize;
		for (int j = 0; j < conf -> tourSize - 1; ++j) {
			bool repeated;

			// Avoid repeated candidates
			do {
				int randomCandidate = rand() % conf -> populationSize;
				if (randomCandidate != bestCandidate) {
					repeated = false;

					// At this point, the individuals already are sorted by rank and crowding distance
					// Therefore, lower index is better
					if (randomCandidate < bestCandidate) {
						bestCandidate = randomCandidate;
					}
				}
				else {
					repeated = true;
				}
			} while (repeated);
		}

		pool[i] = bestCandidate;
	}
}


/**
 * @brief Perform binary crossover between two individuals (2-point crossover)
 * @param population Current population
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 * @return The number of generated children
 */
int crossover2p(Individual *population, const int *pool, const Config *conf) {

	int childrenSize = 0;
	for (int i = 0; i < conf -> poolSize; ++i) {

		// 90% probability perform crossover. Two childen are generated
		if ((rand() / (float) RAND_MAX) < 0.9f) {
			int parent1 = rand() % conf -> poolSize;
			int parent2 = rand() % conf -> poolSize;

			// Avoid repeated parents
			while (parent1 == parent2) {
				parent2 = rand() % conf -> poolSize;
			}

			// Initialize the two children
			population[conf -> populationSize + childrenSize].nSelFeatures = 0;
			population[conf -> populationSize + childrenSize + 1].nSelFeatures = 0;
			population[conf -> populationSize + childrenSize].rank = -1;
			population[conf -> populationSize + childrenSize + 1].rank = -1;
			population[conf -> populationSize + childrenSize].crowding = 0.0f;
			population[conf -> populationSize + childrenSize + 1].crowding = 0.0f;

			for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
				population[conf -> populationSize + childrenSize].fitness[obj] = 0.0f;
				population[conf -> populationSize + childrenSize + 1].fitness[obj] = 0.0f;
			}

			// Perform crossover for each decision variable in the chromosome
			// Crossover between two points
			int point1 = rand() % conf -> nFeatures;
			int point2 = rand() % conf -> nFeatures;
			if (point1 > point2) {
				int temp = point1;
				point1 = point2;
				point2 = temp;
			}

			// First part
			for (int f = 0; f < point1; ++f) {

				// Generate the f-th element of the first child
				population[conf -> populationSize + childrenSize].nSelFeatures += (population[conf -> populationSize + childrenSize].chromosome[f] = population[pool[parent1]].chromosome[f]);

				// Generate the f-th element of the second child
				population[conf -> populationSize + childrenSize + 1].nSelFeatures += (population[conf -> populationSize + childrenSize + 1].chromosome[f] = population[pool[parent2]].chromosome[f]);
			}

			// Second part
			for (int f = point1; f < point2; ++f) {

				// Generate the f-th element of the first child
				population[conf -> populationSize + childrenSize].nSelFeatures += (population[conf -> populationSize + childrenSize].chromosome[f] = population[pool[parent2]].chromosome[f]);

				// Generate the f-th element of the second child
				population[conf -> populationSize + childrenSize + 1].nSelFeatures += (population[conf -> populationSize + childrenSize + 1].chromosome[f] = population[pool[parent1]].chromosome[f]);
			}

			// Third part
			for (int f = point2; f < conf -> nFeatures; ++f) {

				// Generate the f-th element of the first child
				population[conf -> populationSize + childrenSize].nSelFeatures += (population[conf -> populationSize + childrenSize].chromosome[f] = population[pool[parent1]].chromosome[f]);

				// Generate the f-th element of the second child
				population[conf -> populationSize + childrenSize + 1].nSelFeatures += (population[conf -> populationSize + childrenSize + 1].chromosome[f] = population[pool[parent2]].chromosome[f]);
			}

			// At least one decision variable must be set to "1"
			if (population[conf -> populationSize + childrenSize].nSelFeatures == 0) {
				int randomFeature = rand() % conf -> nFeatures;
				population[conf -> populationSize + childrenSize].chromosome[randomFeature] = 1;
				population[conf -> populationSize + childrenSize].nSelFeatures = 1;
			}

			if (population[conf -> populationSize + childrenSize + 1].nSelFeatures == 0) {
				int randomFeature = rand() % conf -> nFeatures;
				population[conf -> populationSize + childrenSize + 1].chromosome[randomFeature] = 1;
				population[conf -> populationSize + childrenSize + 1].nSelFeatures = 1;
			}

			childrenSize += 2;
		}

		// 10% probability perform mutation. One child is generated
		// Mutation is based on random mutation
		else {
			int parent = rand() % conf -> poolSize;

			// Initialize the child
			population[conf -> populationSize + childrenSize].nSelFeatures = 0;
			population[conf -> populationSize + childrenSize].rank = -1;
			population[conf -> populationSize + childrenSize].crowding = 0.0f;

			for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
				population[conf -> populationSize + childrenSize].fitness[obj] = 0.0f;
			}

			// Perform mutation on each element of the selected parent
			for (int f = 0; f < conf -> nFeatures; ++f) {

				// 10% probability perform mutation for each decision variable in the chromosome
				if ((rand() / (float) RAND_MAX) < 0.1f) {
					if (population[pool[parent]].chromosome[f] & 1) {
						population[conf -> populationSize + childrenSize].chromosome[f] = 0;
					}
					else {
						population[conf -> populationSize + childrenSize].chromosome[f] = 1;
						population[conf -> populationSize + childrenSize].nSelFeatures++;
					}
				}
				else {
					population[conf -> populationSize + childrenSize]. nSelFeatures += (population[conf -> populationSize + childrenSize].chromosome[f] = population[pool[parent]].chromosome[f]);
				}
			}

			// At least one decision variable must be set to "1"
			if (population[conf -> populationSize + childrenSize].nSelFeatures == 0) {
				population[conf -> populationSize + childrenSize].chromosome[rand() % conf -> nFeatures] = 1;
				population[conf -> populationSize + childrenSize].nSelFeatures = 1;
			}

			++childrenSize;
		}
	}

	// The not generated children are reinitialized
	for (int i = conf -> populationSize + childrenSize; i < conf -> totalIndividuals; ++i) {
		memset(population[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		memset(population[i].fitness, 0, conf -> nObjectives * sizeof(float));
		population[i].nSelFeatures = 0;
		population[i].rank = -1;
		population[i].crowding = 0.0f;
	}

	return childrenSize;
}


/**
 * @brief Perform binary crossover between two individuals (uniform crossover)
 * @param population Current population
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 * @return The number of generated children
 */
int crossoverUniform(Individual *population, const int *pool, const Config *conf) {

	int childrenSize = 0;
	for (int i = 0; i < conf -> poolSize; ++i) {

		// 90% probability perform crossover. Two childen are generated
		if ((rand() / (float) RAND_MAX) < 0.75f) {
			int parent1 = rand() % conf -> poolSize;
			int parent2 = rand() % conf -> poolSize;

			// Avoid repeated parents
			while (parent1 == parent2) {
				parent2 = rand() % conf -> poolSize;
			}

			// Initialize the two children
			population[conf -> populationSize + childrenSize].nSelFeatures = population[conf -> populationSize + childrenSize + 1].nSelFeatures = 0;
			population[conf -> populationSize + childrenSize].rank = population[conf -> populationSize + childrenSize + 1].rank = -1;
			population[conf -> populationSize + childrenSize].crowding = population[conf -> populationSize + childrenSize + 1].crowding = 0.0f;

			for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
				population[conf -> populationSize + childrenSize].fitness[obj] = population[conf -> populationSize + childrenSize + 1].fitness[obj] = 0.0f;
			}

			// Perform crossover for each decision variable in the chromosome
			// Uniform crossover
			for (int f = 0; f < conf -> nFeatures; ++f) {

				// 50% probability perform copy the decision variable of the other parent
				if ((population[pool[parent1]].chromosome[f] != population[pool[parent2]].chromosome[f]) && ((rand() / (float) RAND_MAX) < 0.5f)) {
					population[conf -> populationSize + childrenSize].nSelFeatures += (population[conf -> populationSize + childrenSize].chromosome[f] = population[pool[parent2]].chromosome[f]);
					population[conf -> populationSize + childrenSize + 1].nSelFeatures += (population[conf -> populationSize + childrenSize + 1].chromosome[f] = population[pool[parent1]].chromosome[f]);
				}
				else {
					population[conf -> populationSize + childrenSize].nSelFeatures += (population[conf -> populationSize + childrenSize].chromosome[f] = population[pool[parent1]].chromosome[f]);
					population[conf -> populationSize + childrenSize + 1].nSelFeatures += (population[conf -> populationSize + childrenSize + 1].chromosome[f] = population[pool[parent2]].chromosome[f]);
				}
			}

			// At least one decision variable must be set to "1"
			if (population[conf -> populationSize + childrenSize].nSelFeatures == 0) {
				population[conf -> populationSize + childrenSize].chromosome[rand() % conf -> nFeatures] = population[conf -> populationSize + childrenSize].nSelFeatures = 1;
			}

			if (population[conf -> populationSize + childrenSize + 1].nSelFeatures == 0) {
				population[conf -> populationSize + childrenSize + 1].chromosome[rand() % conf -> nFeatures] = population[conf -> populationSize + childrenSize + 1].nSelFeatures = 1;
			}

			childrenSize += 2;
		}

		// 10% probability perform mutation. One child is generated
		// Mutation is based on random mutation
		else {
			int parent = rand() % conf -> poolSize;

			// Initialize the child
			population[conf -> populationSize + childrenSize].nSelFeatures = 0;
			population[conf -> populationSize + childrenSize].rank = -1;
			population[conf -> populationSize + childrenSize].crowding = 0.0f;

			for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
				population[conf -> populationSize + childrenSize].fitness[obj] = 0.0f;
			}

			// Perform mutation on each element of the selected parent
			for (int f = 0; f < conf -> nFeatures; ++f) {

				// 10% probability perform mutation (gen level)
				float probability = (rand() / (float) RAND_MAX);
				if (probability < 0.1f) {
					if ((rand() / (float) RAND_MAX) > 0.01f) {
						population[conf -> populationSize + childrenSize].chromosome[f] = 0;
					}
					else {
						population[conf -> populationSize + childrenSize].nSelFeatures += (population[conf -> populationSize + childrenSize].chromosome[f] = 1);
					}
				}
				else {
					population[conf -> populationSize + childrenSize].nSelFeatures += (population[conf -> populationSize + childrenSize].chromosome[f] = population[pool[parent]].chromosome[f]);
				}
			}

			// At least one decision variable must be set to "1"
			if (population[conf -> populationSize + childrenSize].nSelFeatures == 0) {
				population[conf -> populationSize + childrenSize].chromosome[rand() % conf -> nFeatures] = population[conf -> populationSize + childrenSize].nSelFeatures = 1;
			}

			++childrenSize;
		}
	}

	// The not generated children are reinitialized
	for (int i = conf -> populationSize + childrenSize; i < conf -> totalIndividuals; ++i) {
		memset(population[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		memset(population[i].fitness, 0, conf -> nObjectives * sizeof(float));
		population[i].nSelFeatures = 0;
		population[i].rank = -1;
		population[i].crowding = 0.0f;
	}

	return childrenSize;
}


/**
 * @brief Genetic algorithm running in different modes: Sequential, CPU or GPU only and Heterogeneous (full cooperation between all available OpenCL devices)
 * @param populationOrig The original population
 * @param executionMode The execution mode (Sequential, CPU, GPU or Heterogeneous)
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param dataBase The database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param referencePoint The reference point used for the hypervolume calculation
 * @param conf The structure with all configuration parameters
 */
void agAlgorithm(Individual *populationOrig, cl_device_type executionMode, CLDevice *devicesObject, const float *const dataBase, const int *const selInstances, const double *const referencePoint, const Config *conf) {

	// Generic variables
	Individual population[conf -> totalIndividuals];
	int pool[conf -> poolSize];
	float popHypervolume = 0.0f;
	int nIndFront0;

	// Time measure
	double timeStart;
	double totalTime = 0.0;


	/********** Run mutiple times for calculate the mean of the execution times ***********/

	for (int nExec = 0; nExec < conf -> nExecutions; ++nExec) {

		// Reinit the population
		for (int i = 0; i < conf -> totalIndividuals; ++i) {
			population[i] = populationOrig[i];
		}


		/******* Measure *******/

		timeStart = omp_get_wtime();


		/********** Multiobjective individual evaluation ***********/

		if (executionMode == CL_DEVICE_TYPE_DEFAULT) {
			evaluationSEQ(population, 0, conf -> populationSize, dataBase, selInstances, conf);
		}
		else if (executionMode == CL_DEVICE_TYPE_ALL) {
			evaluationCL(population, 0, conf -> populationSize, devicesObject, conf -> nDevices, conf);
		}
		else {
			evaluationCL(population, 0, conf -> populationSize, devicesObject, 1, conf);
		}


		/********** Sort the population with the "Non-Domination-Sort" method ***********/

		nIndFront0 = nonDominationSort(population, conf -> populationSize, conf);


		/********** Start the evolution process ***********/

		for (int g = 0; g < conf -> nGenerations; ++g) {

			// Fill the mating pool
			fillPool(pool, conf);

			// Perform crossover
			int nChildren = crossoverUniform(population, pool, conf);

			// Multiobjective individual evaluation
			if (executionMode == CL_DEVICE_TYPE_DEFAULT) {
				evaluationSEQ(population, conf -> populationSize, conf -> populationSize + nChildren, dataBase, selInstances, conf);
			}
			else if (executionMode == CL_DEVICE_TYPE_ALL) {
				evaluationCL(population, conf -> populationSize, conf -> populationSize + nChildren, devicesObject, conf -> nDevices, conf);
			}
			else {
				evaluationCL(population, conf -> populationSize, conf -> populationSize + nChildren, devicesObject, 1, conf);
			}

			// The crowding distance of the parents is initialized again for the next nonDominationSort
			for (int i = 0;  i < conf -> populationSize; ++i) {
				population[i].crowding = 0.0f;
			}

			// Replace population
			// Parents and children are sorted by rank and crowding distance.
			// The first "populationSize" individuals will advance the next generation
			nIndFront0 = nonDominationSort(population, conf -> populationSize + nChildren, conf);
		}


		/********** Get the population quality (calculating the hypervolume) ***********/

		popHypervolume += getHypervolume(population, nIndFront0, referencePoint, conf);
		totalTime += omp_get_wtime() - timeStart;
	}

	// Finish the time measure
	fprintf(stdout, "%.10g\t%.6g\n", (totalTime * 1000.0) / conf -> nExecutions, popHypervolume / conf -> nExecutions);

	// Generation of the data file for Gnuplot
	if (executionMode == CL_DEVICE_TYPE_ALL) {
		generateDataPlot(population, nIndFront0, "Heterogeneous", conf);
	}
	else if (executionMode == CL_DEVICE_TYPE_DEFAULT) {
		generateDataPlot(population, nIndFront0, "Sequential", conf);
	}
	else {
		generateDataPlot(population, nIndFront0, devicesObject[0].deviceName, conf);
	}
}