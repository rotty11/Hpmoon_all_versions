/**
 * @file ag.cpp
 * @author Juan José Escobar Pérez
 * @date 09/12/2016
 * @brief File with the necessary implementation for the main functions of the islands-based genetic algorithm model
 *
 */

/********************************* Includes *******************************/

#include "ag.h"
#include "evaluation.h"
#include <string.h> // memset...
#include <omp.h> // OpenMP

/********************************* Methods ********************************/


/**
 * @brief Allocates memory for all subpopulations (parents and children). Also, they are initialized
 * @param conf The structure with all configuration parameters
 * @return The first subpopulations
 */
Individual* initSubpopulations(const Config *const conf) {


	/********** Initialization of the subpopulations and the individuals ***********/

	// Allocates memory for parents and children
	Individual *subpops = new Individual[conf -> totalIndividuals];
	for (int i = 0; i < conf -> totalIndividuals; ++i) {
		memset(subpops[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
			subpops[i].fitness[obj] = 0.0f;
		}
		subpops[i].nSelFeatures = 0;
		subpops[i].rank = -1;
		subpops[i].crowding = 0.0f;
	}

	// Only the parents of each subpopulation are initialized
	for (int it = 0; it < conf -> totalIndividuals; it += conf -> familySize) {
		for (int i = it; i < it + conf -> subpopulationSize; ++i) {

			// Set the "1" value at most "conf -> maxFeatures" decision variables
			for (int mf = 0; mf < conf -> maxFeatures; ++mf) {
				int randomFeature = rand() % conf -> nFeatures;
				if (!(subpops[i].chromosome[randomFeature] & 1)) {
					subpops[i].nSelFeatures += (subpops[i].chromosome[randomFeature] = 1);
					
				}
			}
		}
	}

	return subpops;
}


/**
 * @brief Competition between randomly selected individuals. The best individuals are stored in the pool
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 */
void fillPool(int *const pool, const Config *const conf) {

	// Fill pool
	for (int i = 0; i < conf -> poolSize; ++i) {
		int bestCandidate = (rand() % conf -> subpopulationSize);
		for (int j = 0; j < conf -> tourSize - 1; ++j) {
			bool repeated;

			// Avoid repeated candidates
			do {
				int randomCandidate = (rand() % conf -> subpopulationSize);
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
 * @param subpop Current subpopulation
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 * @return The number of generated children
 */
int crossover2p(Individual *const subpop, const int *const pool, const Config *const conf) {

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
			subpop[conf -> subpopulationSize + childrenSize].nSelFeatures = 0;
			subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures = 0;
			subpop[conf -> subpopulationSize + childrenSize].rank = subpop[conf -> subpopulationSize + childrenSize + 1].rank = -1;
			subpop[conf -> subpopulationSize + childrenSize].crowding = subpop[conf -> subpopulationSize + childrenSize + 1].crowding = 0.0f;

			for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
				subpop[conf -> subpopulationSize + childrenSize].fitness[obj] = 0.0f;
				subpop[conf -> subpopulationSize + childrenSize + 1].fitness[obj] = 0.0f;
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
				subpop[conf -> subpopulationSize + childrenSize].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = subpop[pool[parent1]].chromosome[f]);
				
				// Generate the f-th element of the second child
				subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize + 1].chromosome[f] = subpop[pool[parent2]].chromosome[f]);
			}

			// Second part
			for (int f = point1; f < point2; ++f) {

				// Generate the f-th element of the first child
				subpop[conf -> subpopulationSize + childrenSize].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = subpop[pool[parent2]].chromosome[f]);
				
				// Generate the f-th element of the second child
				subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize + 1].chromosome[f] = subpop[pool[parent1]].chromosome[f]);
			}

			// Third part
			for (int f = point2; f < conf -> nFeatures; ++f) {

				// Generate the f-th element of the first child
				subpop[conf -> subpopulationSize + childrenSize].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = subpop[pool[parent1]].chromosome[f]);

				// Generate the f-th element of the second child
				subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize + 1].chromosome[f] = subpop[pool[parent2]].chromosome[f]);
			}

			// At least one decision variable must be set to "1"
			if (subpop[conf -> subpopulationSize + childrenSize].nSelFeatures == 0) {
				int randomFeature = rand() % conf -> nFeatures;
				subpop[conf -> subpopulationSize + childrenSize].chromosome[randomFeature] = 1;
				subpop[conf -> subpopulationSize + childrenSize].nSelFeatures = 1;
			}

			if (subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures == 0) {
				int randomFeature = rand() % conf -> nFeatures;
				subpop[conf -> subpopulationSize + childrenSize + 1].chromosome[randomFeature] = 1;
				subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures = 1;
			}

			childrenSize += 2;
		}

		// 10% probability perform mutation. One child is generated
		// Mutation is based on random mutation
		else {
			int parent = rand() % conf -> poolSize;

			// Initialize the child
			subpop[conf -> subpopulationSize + childrenSize].nSelFeatures = 0;

			for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
				subpop[conf -> subpopulationSize + childrenSize].fitness[obj] = 0.0f;
			}

			// Perform mutation on each element of the selected parent
			for (int f = 0; f < conf -> nFeatures; ++f) {

				// 10% probability perform mutation for each decision variable in the chromosome
				if ((rand() / (float) RAND_MAX) < 0.1f) {
					if (subpop[pool[parent]].chromosome[f] & 1) {
						subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = 0;
					}
					else {
						subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = 1;
						subpop[conf -> subpopulationSize + childrenSize].nSelFeatures++;
					}
				}
				else {
					subpop[conf -> subpopulationSize + childrenSize]. nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = subpop[pool[parent]].chromosome[f]);
				}
			}

			// At least one decision variable must be set to "1"
			if (subpop[conf -> subpopulationSize + childrenSize].nSelFeatures == 0) {
				subpop[conf -> subpopulationSize + childrenSize].chromosome[rand() % conf -> nFeatures] = 1;
				subpop[conf -> subpopulationSize + childrenSize].nSelFeatures = 1;
			}

			++childrenSize;
		}
	}

	// The children not generated are reinitialized
	for (int i = conf -> subpopulationSize + childrenSize; i < conf -> familySize; ++i) {
		memset(subpop[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		memset(subpop[i].fitness, 0, conf -> nObjectives * sizeof(float));
		subpop[i].nSelFeatures = 0;
		subpop[i].rank = -1;
		subpop[i].crowding = 0.0f;
	}

	return childrenSize;
}


/**
 * @brief Perform binary crossover between two individuals (uniform crossover)
 * @param subpop Current subpopulation
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 * @return The number of generated children
 */
int crossoverUniform(Individual *const subpop, const int *const pool, const Config *const conf) {

	int childrenSize = 0;
	for (int i = 0; i < conf -> poolSize; ++i) {

		// 75% probability perform crossover. Two childen are generated
		if ((rand() / (float) RAND_MAX) < 0.75f) {
			int parent1 = rand() % conf -> poolSize;
			int parent2 = rand() % conf -> poolSize;

			// Avoid repeated parents
			while (parent1 == parent2) {
				parent2 = rand() % conf -> poolSize;
			}

			// Initialize the two children
			subpop[conf -> subpopulationSize + childrenSize].nSelFeatures = subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures = 0;
			subpop[conf -> subpopulationSize + childrenSize].rank = subpop[conf -> subpopulationSize + childrenSize + 1].rank = -1;
			subpop[conf -> subpopulationSize + childrenSize].crowding = subpop[conf -> subpopulationSize + childrenSize + 1].crowding = 0.0f;

			for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
				subpop[conf -> subpopulationSize + childrenSize].fitness[obj] = subpop[conf -> subpopulationSize + childrenSize + 1].fitness[obj] = 0.0f;
			}

			// Perform crossover for each decision variable in the chromosome
			// Uniform crossover
			for (int f = 0; f < conf -> nFeatures; ++f) {

				// 50% probability perform copy the decision variable of the other parent
				if ((subpop[pool[parent1]].chromosome[f] != subpop[pool[parent2]].chromosome[f]) && ((rand() / (float) RAND_MAX) < 0.5f)) {
					subpop[conf -> subpopulationSize + childrenSize].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = subpop[pool[parent2]].chromosome[f]);
					subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize + 1].chromosome[f] = subpop[pool[parent1]].chromosome[f]);
				}
				else {
					subpop[conf -> subpopulationSize + childrenSize].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = subpop[pool[parent1]].chromosome[f]);
					subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize + 1].chromosome[f] = subpop[pool[parent2]].chromosome[f]);
				}
			}

			// At least one decision variable must be set to "1"
			if (subpop[conf -> subpopulationSize + childrenSize].nSelFeatures == 0) {
				subpop[conf -> subpopulationSize + childrenSize].chromosome[rand() % conf -> nFeatures] = subpop[conf -> subpopulationSize + childrenSize].nSelFeatures = 1;
			}

			if (subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures == 0) {
				subpop[conf -> subpopulationSize + childrenSize + 1].chromosome[rand() % conf -> nFeatures] = subpop[conf -> subpopulationSize + childrenSize + 1].nSelFeatures = 1;
			}

			childrenSize += 2;
		}

		// 25% probability perform mutation. One child is generated
		// Mutation is based on random mutation
		else {
			int parent = rand() % conf -> poolSize;

			// Initialize the child
			subpop[conf -> subpopulationSize + childrenSize].nSelFeatures = 0;
			subpop[conf -> subpopulationSize + childrenSize].rank = -1;
			subpop[conf -> subpopulationSize + childrenSize].crowding = 0.0f;

			for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
				subpop[conf -> subpopulationSize + childrenSize].fitness[obj] = 0.0f;
			}

			// Perform mutation on each element of the selected parent
			for (int f = 0; f < conf -> nFeatures; ++f) {

				// 10% probability perform mutation (gen level)
				float probability = (rand() / (float) RAND_MAX);
				if (probability < 0.1f) {
					if ((rand() / (float) RAND_MAX) > 0.01f) {
						subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = 0;
					}
					else {
						subpop[conf -> subpopulationSize + childrenSize].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = 1);
					}
				}
				else {
					subpop[conf -> subpopulationSize + childrenSize].nSelFeatures += (subpop[conf -> subpopulationSize + childrenSize].chromosome[f] = subpop[pool[parent]].chromosome[f]);
				}
			}

			// At least one decision variable must be set to "1"
			if (subpop[conf -> subpopulationSize + childrenSize].nSelFeatures == 0) {
				subpop[conf -> subpopulationSize + childrenSize].chromosome[rand() % conf -> nFeatures] = subpop[conf -> subpopulationSize + childrenSize].nSelFeatures = 1;
			}

			++childrenSize;
		}
	}

	// The children not generated are reinitialized
	for (int i = conf -> subpopulationSize + childrenSize; i < conf -> familySize; ++i) {
		memset(subpop[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		memset(subpop[i].fitness, 0, conf -> nObjectives * sizeof(float));
		subpop[i].nSelFeatures = 0;
		subpop[i].rank = -1;
		subpop[i].crowding = 0.0f;
	}

	return childrenSize;
}


/**
 * @brief Perform the individuals migrations between subpopulations
 * @param subpops The subpopulations
 * @param nIndsFronts0 The number of individuals in the front 0 of each subpopulation
 * @param conf The structure with all configuration parameters
 */
void migration(Individual *const subpops, const int *const nIndsFronts0, const Config *const conf) {

	// Initiallize a vector with the index of the subpopulations
	vector<int> indIndexOrig(conf -> nSubpopulations);
	for (int subpop = 0; subpop < conf -> nSubpopulations; ++subpop) {
		indIndexOrig[subpop] = subpop;
	}

	// From subpopulations randomly choosen some individuals of the front 0 are copied to each subpopulation (the worst individuals are deleted)
	for (int subpop = 0; subpop < conf -> nSubpopulations; ++subpop) {

		// This vector contains the available subpopulations indexes which are randomly choosen for copy the individuals of the front 0
		vector<int> randomIndex(indIndexOrig);

		// The current subpopulation will not copy its own individuals
		randomIndex.erase(randomIndex.begin() + subpop);

		int maxCopy = conf -> subpopulationSize - nIndsFronts0[subpop];
		Individual *ptrDest = subpops + (subpop * conf -> familySize) + conf -> subpopulationSize;
		for (int subpop2 = 1; subpop2 < conf -> nSubpopulations && maxCopy > 0; ++subpop2) {
			int randomSubpop = rand() % randomIndex.size();
			int toCopy = min(maxCopy, nIndsFronts0[randomIndex[randomSubpop]] >> 1);
			Individual *ptrOrig = subpops + (randomIndex[randomSubpop] * conf -> familySize);
			ptrDest -= toCopy;
			memcpy(ptrDest, ptrOrig, toCopy * sizeof(Individual));
			randomIndex.erase(randomIndex.begin() + randomSubpop);
			maxCopy -= toCopy;
		}
	}
}


/**
 * @brief Genetic algorithm running in different modes: Sequential, CPU or GPU only and Heterogeneous (full cooperation between all available OpenCL devices)
 * @param popOrig The original population
 * @param nDevices The number of devices that will execute the genetic algorithm
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param dataBase The database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param referencePoint The reference point used for the hypervolume calculation
 * @param conf The structure with all configuration parameters
 */
void agAlgorithm(const Individual *const popOrig, const int nDevices, CLDevice *const devicesObject, const float *const dataBase, const int *const selInstances, const double *const referencePoint, const Config *const conf) {

	// Generic variables
	Individual *pop = new Individual[conf -> totalIndividuals];
	int nIndFront0;

	// Time measure
	double timeStart;
	vector<double> times;


	/********** Run mutiple times for calculate the mean of the execution times ***********/

	for (int nExec = 0; nExec < conf -> nExecutions; ++nExec) {

		vector<float> hvs;

		// Reinit the population
		memcpy(pop, popOrig, conf -> totalIndividuals * sizeof(Individual));


		/******* Measure *******/

		timeStart = omp_get_wtime();


		/********** Multi-objective individual evaluation ***********/

		if (nDevices == 0) {
			evaluationSEQ(pop, conf -> subpopulationSize, dataBase, selInstances, conf);
		}
		else {
			evaluationCL(pop, conf -> subpopulationSize, devicesObject, nDevices, conf);
		}


		/********** Sort the population with the "Non-Domination-Sort" method ***********/

		nIndFront0 = nonDominationSort(pop, conf -> subpopulationSize, conf);
		hvs.push_back(getHypervolume(pop, nIndFront0, referencePoint, conf));


		/********** Start the evolution process ***********/

		int *pool = new int[conf -> poolSize];
		for (int g = 0; g < conf -> nGenerations; ++g) {

			// Fill the mating pool
			fillPool(pool, conf);

			// Perform crossover
			int nChildren = crossoverUniform(pop, pool, conf);

			// Multi-objective individual evaluation
			if (nDevices == 0) {
				evaluationSEQ(pop + conf -> subpopulationSize, nChildren, dataBase, selInstances, conf);
			}
			else {
				evaluationCL(pop + conf -> subpopulationSize, nChildren, devicesObject, nDevices, conf);
			}

			// The crowding distance of the parents is initialized again for the next nonDominationSort
			for (int i = 0;  i < conf -> subpopulationSize; ++i) {
				pop[i].crowding = 0.0f;
			}

			// Replace population
			// Parents and children are sorted by rank and crowding distance.
			// The first "subpopulationSize" individuals will advance the next generation
			nIndFront0 = nonDominationSort(pop, conf -> subpopulationSize + nChildren, conf);


			/********** Get the population quality (calculating the hypervolume) ***********/

			hvs.push_back(getHypervolume(pop, nIndFront0, referencePoint, conf));
		}

		fprintf(stdout, "%.6g\n", getHypervolume(pop, nIndFront0, referencePoint, conf));

		// Local resources used are released
		delete[] pool;

		// Save the execution time
		times.push_back(omp_get_wtime() - timeStart);
	}

	// Finish the time measure
	for (int r = 0; r < times.size(); ++r) {
		fprintf(stdout, "%.10g\n", times[r] * 1000.0);
	}

	// Generation of the data file for Gnuplot
	if (nDevices == 0) {
		generateDataPlot(pop, nIndFront0, "Sequential", conf);
	}
	else if (nDevices > 1) {
		generateDataPlot(pop, nIndFront0, "Heterogeneous", conf);
	}
	else {
		generateDataPlot(pop, nIndFront0, devicesObject[0].deviceName, conf);
	}


	/********** Resources used are released ***********/

	delete[] pop;
}


/**
 * @brief Island-based genetic algorithm model running in different modes: Sequential, CPU or GPU only and Heterogeneous (full cooperation between all available OpenCL devices)
 * @param subpopsOrig The original subpopulations
 * @param nDevices The number of devices that will execute the genetic algorithm
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param dataBase The database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param referencePoint The reference point used for the hypervolume calculation
 * @param conf The structure with all configuration parameters
 */
void agIslands(const Individual *const subpopsOrig, const int nDevices, CLDevice *const devicesObject, const float *const dataBase, const int *const selInstances, const double *const referencePoint, const Config *const conf) {

	// Generic variables
	Individual *const subpops = new Individual[conf -> totalIndividuals];
	Individual *const finalPop = new Individual[conf -> worldSize];
	int *const nIndsFronts0 = new int[conf -> nSubpopulations];
	int nThreads = max(1, nDevices);
	int finalFront0;

	// Time measure
	double timeStart;
	vector<double> times;


	/********** Run multiple times for calculate the mean of the execution times ***********/

	for (int nExec = 0; nExec < conf -> nExecutions; ++nExec) {

		// Reinit the subpopulations
		memcpy(subpops, subpopsOrig, conf -> totalIndividuals * sizeof(Individual));


		/******* Measure *******/

		timeStart = omp_get_wtime();


		/********** In each migration the individuals are exchanged  ***********/

		for (int mig = 0; mig < conf -> nMigrations; ++mig) {


			/********** Iterate over all subpopulations ***********/

			#pragma omp parallel for num_threads(nThreads) schedule(dynamic, 1)
			for (int ind = 0; ind < conf -> nSubpopulations; ++ind) {

				// Aux
				int popIndex = ind * conf -> familySize;

				if (mig == 0) {


					/********** Multi-objective individual evaluation ***********/

					if (nDevices == 0) {
						evaluationSEQ(subpops + popIndex, conf -> subpopulationSize, dataBase, selInstances, conf);
					}
					else {
						evaluationCL(subpops + popIndex, conf -> subpopulationSize, &devicesObject[omp_get_thread_num()], 1, conf);
					}


					/********** Sort the subpopulation with the "Non-Domination-Sort" method ***********/

					nIndsFronts0[ind] = nonDominationSort(subpops + popIndex, conf -> subpopulationSize, conf);
				}


				/********** Start the evolution process ***********/

				int *pool = new int[conf -> poolSize];
				for (int g = 0; g < conf -> nGenerations; ++g) {

					// Fill the mating pool
					fillPool(pool, conf);

					// Perform crossover
					int nChildren = crossoverUniform(subpops + popIndex, pool, conf);

					// Multi-objective individual evaluation
					if (nDevices == 0) {
						evaluationSEQ(subpops + popIndex + conf -> subpopulationSize, nChildren, dataBase, selInstances, conf);
					}
					else {
						evaluationCL(subpops + popIndex + conf -> subpopulationSize, nChildren, &devicesObject[omp_get_thread_num()], 1, conf);
					}

					// The crowding distance of the parents is initialized again for the next nonDominationSort
					for (int i = popIndex;  i < popIndex + conf -> subpopulationSize; ++i) {
						subpops[i].crowding = 0.0f;
					}

					// Replace subpopulation
					// Parents and children are sorted by rank and crowding distance.
					// The first "subpopulationSize" individuals will advance the next generation
					nIndsFronts0[ind] = nonDominationSort(subpops + popIndex, conf -> subpopulationSize + nChildren, conf);
				}

				// Local resources used are released
				delete[] pool;
			}

			// Migration process
			if (mig != conf -> nMigrations - 1) {
				migration(subpops, nIndsFronts0, conf);

				#pragma omp parallel for num_threads(conf -> nSubpopulations)
				for (int ind = 0; ind < conf -> nSubpopulations; ++ind) {
					int popIndex = ind * conf -> familySize;

					// The crowding distance of the subpopulation is initialized again for the next nonDominationSort
					for (int i = popIndex;  i < popIndex + conf -> subpopulationSize; ++i) {
						subpops[i].crowding = 0.0f;
					}
					nonDominationSort(subpops + popIndex, conf -> subpopulationSize, conf);
				}
			}
		}


		/********** Recombination process ***********/

		for (int ind = 0; ind < conf -> nSubpopulations; ++ind) {
			memcpy(finalPop + (ind * conf -> subpopulationSize), subpops + (ind * conf -> familySize), conf -> subpopulationSize * sizeof(Individual));
		}

		// The crowding distance of the subpopulation is initialized again for the next nonDominationSort
		for (int i = 0;  i < conf -> worldSize; ++i) {
			finalPop[i].crowding = 0.0f;
		}
		finalFront0 = min(conf -> subpopulationSize, nonDominationSort(finalPop, conf -> worldSize, conf));

		// Save the execution time
		times.push_back(omp_get_wtime() - timeStart);

		fprintf(stdout, "%.6g\n", getHypervolume(finalPop, finalFront0, referencePoint, conf));
	}

	// Finish the time measure
	for (int r = 0; r < times.size(); ++r) {
		fprintf(stdout, "%.10g\n", times[r] * 1000.0);
	}

	// Generation of the data file for Gnuplot
	if (nDevices == 0) {
		generateDataPlot(finalPop, finalFront0, "Sequential", conf);
	}
	else if (nDevices > 1) {
		generateDataPlot(finalPop, finalFront0, "Heterogeneous", conf);
	}
	else {
		generateDataPlot(finalPop, finalFront0, devicesObject[0].deviceName, conf);
	}


	/********** Resources used are released ***********/

	delete[] subpops;
	delete[] finalPop;
	delete[] nIndsFronts0;
}
