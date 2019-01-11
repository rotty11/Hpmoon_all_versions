/**
 * @file ag.cpp
 * @author Juan José Escobar Pérez
 * @date 09/12/2016
 * @brief File with the necessary implementation for the main functions of the islands-based genetic algorithm model
 */

/********************************* Includes *******************************/

#include "ag.h"
#include "evaluation.h"
#include <algorithm> // std::max_element
#include <numeric> // std::iota
#include <omp.h> // OpenMP
#include <set> // std::set
#include <string.h> // memcpy, memset

/********************************* Defines ********************************/

#define INITIALIZE 0
#define IGNORE_VALUE 1
#define FINISH 2

/********************************* Methods ********************************/

/**
 * @brief Allocate memory for all subpopulations (parents and children). Also, they are initialized
 * @param conf The structure with all configuration parameters
 * @return The first subpopulations
 */
Individual* initSubpopulations(const Config *const conf) {


	/********** Initialization of the subpopulations and the individuals ***********/

	// Allocate memory for parents and children
	Individual *subpops = new Individual[conf -> totalIndividuals];
	for (int i = 0; i < conf -> totalIndividuals; ++i) {
		memset(subpops[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
			subpops[i].fitness[obj] = 0.0f;
		}
		subpops[i].crowding = 0.0f;
		subpops[i].rank = -1;
		subpops[i].nSelFeatures = 0;
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
 * @brief Tournament between randomly selected individuals. The best individuals are stored in the pool
 * @param conf The structure with all configuration parameters
 * @return The pool with the selected individuals
 */
int* getPool(const Config *const conf) {

	// Create and fill the pool
	int *pool = new int[conf -> poolSize];
	std::set<int> candidates;
	for (int i = 0; i < conf -> poolSize; ++i) {

		// std::set guarantees unique elements in the insert function
		for (int j = 0; j < conf -> tourSize; ++j) {
			candidates.insert(rand() % conf -> subpopulationSize);
		}

		// At this point, the individuals already are sorted by rank and crowding distance
		// Therefore, lower index is better
		pool[i] = *(candidates.begin());
		candidates.clear();
	}

	return pool;
}


/**
 * @brief Perform binary crossover between two individuals (uniform crossover)
 * @param subpop Current subpopulation
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 * @return The number of generated children
 */
int crossoverUniform(Individual *const subpop, const int *const pool, const Config *const conf) {

	// Reset the children
	for (int i = conf -> subpopulationSize; i < conf -> familySize; ++i) {
		memset(subpop[i].chromosome, 0, conf -> nFeatures * sizeof(unsigned char));
		for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
			subpop[i].fitness[obj] = 0.0f;
		}
		subpop[i].nSelFeatures = 0;
		subpop[i].rank = -1;
		subpop[i].crowding = 0.0f;
	}

	Individual *child = &(subpop[conf -> subpopulationSize]);
	for (int i = 0; i < conf -> poolSize; ++i) {

		// 75% probability perform crossover. Two childen are generated
		Individual *parent1 = &(subpop[pool[rand() % conf -> poolSize]]);
		if ((rand() / (float) RAND_MAX) < 0.75f) {

			// Avoid repeated parents
			Individual *parent2 = &(subpop[pool[rand() % conf -> poolSize]]);
			Individual *child2 = child + 1;
			while (parent1 == parent2) {
				parent2 = &(subpop[pool[rand() % conf -> poolSize]]);
			}

			// Perform uniform crossover for each decision variable in the chromosome
			for (int f = 0; f < conf -> nFeatures; ++f) {

				// 50% probability perform copy the decision variable of the other parent
				if ((parent1 -> chromosome[f] != parent2 -> chromosome[f]) && ((rand() / (float) RAND_MAX) < 0.5f)) {
					child -> nSelFeatures += (child -> chromosome[f] = parent2 -> chromosome[f]);
					child2 -> nSelFeatures += (child2 -> chromosome[f] = parent1 -> chromosome[f]);
				}
				else {
					child -> nSelFeatures += (child -> chromosome[f] = parent1 -> chromosome[f]);
					child2 -> nSelFeatures += (child2 -> chromosome[f] = parent2 -> chromosome[f]);
				}
			}

			// At least one decision variable must be set to "1"
			if (child -> nSelFeatures == 0) {
				child -> chromosome[rand() % conf -> nFeatures] = child -> nSelFeatures = 1;
			}

			if (child2 -> nSelFeatures == 0) {
				child2 -> chromosome[rand() % conf -> nFeatures] = child2 -> nSelFeatures = 1;
			}
			child += 2;
		}

		// 25% probability perform mutation. One child is generated
		// Mutation is based on random mutation
		else {

			// Perform mutation on each element of the selected parent
			for (int f = 0; f < conf -> nFeatures; ++f) {

				// 10% probability perform mutation (gen level)
				float probability = (rand() / (float) RAND_MAX);
				if (probability < 0.1f) {
					if ((rand() / (float) RAND_MAX) > 0.01f) {
						child -> chromosome[f] = 0;
					}
					else {
						child -> nSelFeatures += (child -> chromosome[f] = 1);
					}
				}
				else {
					child -> nSelFeatures += (child -> chromosome[f] = parent1 -> chromosome[f]);
				}
			}

			// At least one decision variable must be set to "1"
			if (child -> nSelFeatures == 0) {
				child -> chromosome[rand() % conf -> nFeatures] = child -> nSelFeatures = 1;
			}
			++child;
		}
	}

	return child - (subpop + conf -> subpopulationSize);
}


/**
 * @brief Perform the migrations between subpopulations
 * @param subpops The subpopulations
 * @param nSubpopulations The number of subpopulations involved in the migration
 * @param nIndsFronts0 The number of individuals in the front 0 of each subpopulation
 * @param conf The structure with all configuration parameters
 */
void migration(Individual *const subpops, const int nSubpopulations, const int *const nIndsFronts0, const Config *const conf) {

	// From subpopulations randomly choosen some individuals of the front 0 are copied to each subpopulation (the worst individuals are deleted)
	for (int subpop = 0; subpop < nSubpopulations; ++subpop) {

		// This vector contains the available subpopulations indexes which are randomly choosen for copy the individuals of the front 0
		std::vector<int> randomIndex(nSubpopulations);
		std::iota(randomIndex.begin(), randomIndex.end(), 0);

		// The current subpopulation will not copy its own individuals
		randomIndex.erase(randomIndex.begin() + subpop);
		std::random_shuffle(randomIndex.begin(), randomIndex.end());

		int maxCopy = conf -> subpopulationSize - nIndsFronts0[subpop];
		Individual *ptrDest = subpops + (subpop * conf -> familySize) + conf -> subpopulationSize;
		for (int subpop2 = 0; subpop2 < nSubpopulations - 1 && maxCopy > 0; ++subpop2) {
			int toCopy = std::min(maxCopy, nIndsFronts0[randomIndex[subpop2]] >> 1);
			Individual *ptrOrig = subpops + (randomIndex[subpop2] * conf -> familySize);
			ptrDest -= toCopy;
			memcpy(ptrDest, ptrOrig, toCopy * sizeof(Individual));
			maxCopy -= toCopy;
		}
	}
}


/**
 * @brief Island-based genetic algorithm model running in different modes: Sequential, CPU or GPU only and Heterogeneous (full cooperation between all available OpenCL devices)
 * @param subpops The initial subpopulations
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param trDataBase The training database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param conf The structure with all configuration parameters
 */
void agIslands(Individual *const subpops, CLDevice *const devicesObject, const float *const trDataBase, const int *const selInstances, const Config *const conf) {


	/********** MPI variables ***********/

	MPI_Datatype Individual_MPI_type;
	MPI_Datatype array_of_types[3] = {MPI_UNSIGNED_CHAR, MPI_FLOAT, MPI_INT};
	int array_of_blocklengths[3] = {conf -> nFeatures, conf -> nObjectives + 1, 2};
	MPI_Aint array_of_displacement[3];
	MPI_Status status;


	/******* Measure and start the master-worker algorithm *******/

	MPI_Barrier(MPI_COMM_WORLD);


	/******* Each process dinamically will request subpopulations *******/

	// Master
	if (conf -> mpiRank == 0) {
		double timeStart = omp_get_wtime();
		int *nIndsFronts0 = new int[conf -> nSubpopulations];
		int finalFront0;

		// The master receives the number of subpopulations that each worker can process
		int workerCapacities[conf -> mpiSize - 1];
		MPI_Request requests[conf -> mpiSize - 1];
		for (int p = 1; p < conf -> mpiSize; ++p) {
			MPI_Irecv(&workerCapacities[p - 1], 1, MPI_INT, p, MPI_ANY_TAG, MPI_COMM_WORLD, &requests[p - 1]);
		}

		// The "Individual" datatype must be converted to a MPI datatype and commit it
		array_of_displacement[0] = (size_t) &(subpops[0].chromosome[0]) - (size_t) &(subpops[0]);
		array_of_displacement[1] = (size_t) &(subpops[0].fitness[0]) - (size_t) &(subpops[0]);
		array_of_displacement[2] = (size_t) &(subpops[0].rank) - (size_t) &(subpops[0]);

		MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacement, array_of_types, &Individual_MPI_type);
		MPI_Type_commit(&Individual_MPI_type);

		MPI_Waitall(conf -> mpiSize - 1, requests, MPI_STATUSES_IGNORE);
		int maxChunk = std::min(*std::max_element(workerCapacities, workerCapacities + conf -> mpiSize - 1), conf -> nSubpopulations);


		/********** In each migration the individuals are exchanged between subpopulations of different nodes  ***********/

		for (int gMig = 0; gMig < conf -> nGlobalMigrations; ++gMig) {

			// Send some work to the workers
			int nextWork = 0;
			int sent = 0;
			int mpiTag = (gMig == 0) ? INITIALIZE : IGNORE_VALUE;
			for (int p = 1; p < conf -> mpiSize && nextWork < conf -> nSubpopulations; ++p) {
				int finallyWork = std::min(workerCapacities[p - 1], conf -> nSubpopulations - nextWork);
				int popIndex = nextWork * conf -> familySize;
				MPI_Isend(subpops + popIndex, finallyWork * conf -> familySize, Individual_MPI_type, p, mpiTag, MPI_COMM_WORLD, &requests[p - 1]);
				nextWork += finallyWork;
				++sent;
			}
			MPI_Waitall(sent, requests, MPI_STATUSES_IGNORE);

			// Dynamically distribute the subpopulations
			int receivedWork = 0;
			int receivedPtr = 0;
			while (nextWork < conf -> nSubpopulations) {
				MPI_Recv(subpops + (receivedPtr * conf -> familySize), maxChunk * conf -> familySize, Individual_MPI_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				MPI_Recv(nIndsFronts0 + receivedPtr, maxChunk, MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				MPI_Get_count(&status, MPI_INT, &receivedWork);
				receivedPtr += receivedWork;
				int finallyWork = std::min(workerCapacities[status.MPI_SOURCE - 1], conf -> nSubpopulations - nextWork);
				int popIndex = nextWork * conf -> familySize;
				MPI_Send(subpops + popIndex, finallyWork * conf -> familySize, Individual_MPI_type, status.MPI_SOURCE, mpiTag, MPI_COMM_WORLD);
				nextWork += finallyWork;
			}

			// Receive the remaining work
			while (receivedPtr < conf -> nSubpopulations) {
				MPI_Recv(subpops + (receivedPtr * conf -> familySize), maxChunk * conf -> familySize, Individual_MPI_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				MPI_Recv(nIndsFronts0 + receivedPtr, maxChunk, MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				MPI_Get_count(&status, MPI_INT, &receivedWork);
				receivedPtr += receivedWork;
			}

			// Migration process between subpopulations of different nodes
			if (gMig != conf -> nGlobalMigrations - 1 && conf -> nSubpopulations > 1) {
				migration(subpops, conf -> nSubpopulations, nIndsFronts0, conf);

				#pragma omp parallel for
				for (int sp = 0; sp < conf -> nSubpopulations; ++sp) {
					int popIndex = sp * conf -> familySize;

					// The crowding distance of the subpopulation is initialized again for the next nonDominationSort
					for (int i = popIndex;  i < popIndex + conf -> subpopulationSize; ++i) {
						subpops[i].crowding = 0.0f;
					}
					nonDominationSort(subpops + popIndex, conf -> subpopulationSize, conf);
				}
			}
		}

		// Notify to all workers that the work has finished
		for (int p = 1; p < conf -> mpiSize; ++p) {
			  MPI_Isend(0, 0, MPI_INT, p, FINISH, MPI_COMM_WORLD, &requests[p - 1]);
		}


		/********** Recombination process ***********/

		if (conf -> nSubpopulations > 1) {
			for (int sp = 0; sp < conf -> nSubpopulations; ++sp) {
				memcpy(subpops + (sp * conf -> subpopulationSize), subpops + (sp * conf -> familySize), conf -> subpopulationSize * sizeof(Individual));
			}

			// The crowding distance of the subpopulation is initialized again for the next nonDominationSort
			#pragma omp parallel for
			for (int i = 0;  i < conf -> worldSize; ++i) {
				subpops[i].crowding = 0.0f;
			}
			finalFront0 = std::min(conf -> subpopulationSize, nonDominationSort(subpops, conf -> worldSize, conf));
		}
		else {
			finalFront0 = nIndsFronts0[0];
		}

		// All process must reach this point in order to provide a real time measure
		MPI_Waitall(conf -> mpiSize - 1, requests, MPI_STATUSES_IGNORE);
		MPI_Barrier(MPI_COMM_WORLD);
		fprintf(stdout, "%.10g\n", (omp_get_wtime() - timeStart) * 1000.0);

		// Get the hypervolume
		fprintf(stdout, "%.6g\n", getHypervolume(subpops, finalFront0, conf));

		// Generation of the data file for Gnuplot
		generateDataPlot(subpops, finalFront0, conf);

		// Exclusive variables used by the master are released
		delete[] nIndsFronts0;
		MPI_Type_free(&Individual_MPI_type);
	}

	// Workers
	else {
		// This is only for sequential benchmark
		const bool isSequential = (conf -> nDevices == 0 && conf -> ompThreads < 2);
		const int nDevices = (isSequential) ? conf -> nSubpopulations : std::max(1, conf -> nDevices + (conf -> ompThreads > 0));
		int nChildren[nDevices];
		int nIndsFronts0[nDevices];
		MPI_Request requests[2];

		// The worker tells to the master how many subpopulations can be processed
		MPI_Isend(&nDevices, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &(requests[0]));
		MPI_Request_free(&(requests[0]));

		// Each worker will compute as many subpopulations as OpenCL devices at most
		Individual *subpops = new Individual[nDevices * conf -> familySize];

		// Create MPI datatype for the individuals and commit it
		array_of_displacement[0] = (size_t) &(subpops[0].chromosome[0]) - (size_t) &(subpops[0]);
		array_of_displacement[1] = (size_t) &(subpops[0].fitness[0]) - (size_t) &(subpops[0]);
		array_of_displacement[2] = (size_t) &(subpops[0].rank) - (size_t) &(subpops[0]);

		MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacement, array_of_types, &Individual_MPI_type);
		MPI_Type_commit(&Individual_MPI_type);

		// The worker receives as many subpopulations as number of OpenCL devices at most
		MPI_Recv(subpops, nDevices * conf -> familySize, Individual_MPI_type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		while (status.MPI_TAG != FINISH) {
			int receivedWork;
			MPI_Get_count(&status, Individual_MPI_type, &receivedWork);
			int nSubpopulations = receivedWork / conf -> familySize;
			int nThreads = (isSequential) ? 1 : std::min(nDevices, nSubpopulations);

			if (status.MPI_TAG == INITIALIZE) {


				/********** Multi-objective individuals evaluation over all subpopulations ***********/

				omp_set_nested(1);
				#pragma omp parallel for num_threads(nThreads) schedule(dynamic, 1)
				for (int sp = 0; sp < nSubpopulations; ++sp) {
					int popIndex = sp * conf -> familySize;
					if (isSequential) {
						evaluationCPU(subpops + popIndex, conf -> subpopulationSize, trDataBase, selInstances, 1, conf);
					}
					else if (nSubpopulations == 1) {
						evaluationHET(subpops + popIndex, conf -> subpopulationSize, devicesObject, nDevices, trDataBase, selInstances, conf);
					}
					else {
						evaluationHET(subpops + popIndex, conf -> subpopulationSize, &devicesObject[omp_get_thread_num()], 1, trDataBase, selInstances, conf);
					}

					// Fitness normalization
					normalizeFitness(subpops + popIndex, conf -> subpopulationSize, conf);
				}


				/********** Sort each subpopulation with the "Non-Domination-Sort" method ***********/

				#pragma omp parallel for
				for (int sp = 0; sp < nSubpopulations; ++sp) {
					int popIndex = sp * conf -> familySize;
					nIndsFronts0[sp] = nonDominationSort(subpops + popIndex, conf -> subpopulationSize, conf);
				}
			}


			/********** In each migration the individuals are exchanged between subpopulations of the same node  ***********/

			int nLocalMigrations = (nSubpopulations > 1) ? conf -> nLocalMigrations : 1;
			for (int lMig = 0; lMig < nLocalMigrations; ++lMig) {


				/********** Start the evolution process ***********/

				for (int g = 0; g < conf -> nGenerations; ++g) {


					/********** Fill the mating pool and perform crossover ***********/

					#pragma omp parallel for
					for (int sp = 0; sp < nSubpopulations; ++sp) {
						const int *const pool = getPool(conf);
						int popIndex = sp * conf -> familySize;	
						nChildren[sp] = crossoverUniform(subpops + popIndex, pool, conf);

						// Local resources used are released
						delete[] pool;
					}


					/********** Multi-objective individuals evaluation over all subpopulations ***********/

					#pragma omp parallel for num_threads(nThreads) schedule(dynamic, 1)
					for (int sp = 0; sp < nSubpopulations; ++sp) {
						int popIndex = sp * conf -> familySize;
						if (isSequential) {
							evaluationCPU(subpops + popIndex + conf -> subpopulationSize, nChildren[sp], trDataBase, selInstances, 1, conf);
						}
						else if (nSubpopulations == 1) {
							evaluationHET(subpops + popIndex + conf -> subpopulationSize, nChildren[sp], devicesObject, nDevices, trDataBase, selInstances, conf);
						}
						else {
							evaluationHET(subpops + popIndex + conf -> subpopulationSize, nChildren[sp], &devicesObject[omp_get_thread_num()], 1, trDataBase, selInstances, conf);
						}

						// Fitness normalization
						normalizeFitness(subpops + popIndex + conf -> subpopulationSize, nChildren[sp], conf);
					}


					/********** The crowding distance of the parents is initialized again for the next nonDominationSort ***********/

					#pragma omp parallel for
					for (int sp = 0; sp < nSubpopulations; ++sp) {
						int popIndex = sp * conf -> familySize;
						for (int i = popIndex;  i < popIndex + conf -> subpopulationSize; ++i) {
							subpops[i].crowding = 0.0f;
						}

						// Replace subpopulation
						// Parents and children are sorted by rank and crowding distance.
						// The first "subpopulationSize" individuals will advance the next generation
						nIndsFronts0[sp] = nonDominationSort(subpops + popIndex, conf -> subpopulationSize + nChildren[sp], conf);
					}
				}

				// Migration process between subpopulations of the same node
				if (lMig != nLocalMigrations - 1 && nSubpopulations > 1) {
					migration(subpops, nSubpopulations, nIndsFronts0, conf);

					#pragma omp parallel for
					for (int sp = 0; sp < nSubpopulations; ++sp) {
						int popIndex = sp * conf -> familySize;

						// The crowding distance of the subpopulation is initialized again for the next nonDominationSort
						for (int i = popIndex;  i < popIndex + conf -> subpopulationSize; ++i) {
							subpops[i].crowding = 0.0f;
						}
						nonDominationSort(subpops + popIndex, conf -> subpopulationSize, conf);
					}
				}
			}

			// The worker send to the master the subpopulations already evaluated and will request new work
			MPI_Isend(subpops, receivedWork, Individual_MPI_type, 0, 0, MPI_COMM_WORLD, &(requests[0]));
			MPI_Isend(nIndsFronts0, nSubpopulations, MPI_INT, 0, 0, MPI_COMM_WORLD, &(requests[1]));
			MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
			MPI_Recv(subpops, nDevices * conf -> familySize, Individual_MPI_type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}

		// All process must reach this point in order to provide a real time measure
		MPI_Barrier(MPI_COMM_WORLD);

		// Exclusive variables used by the workers are released
		delete[] subpops;
		MPI_Type_free(&Individual_MPI_type);
	}
}
