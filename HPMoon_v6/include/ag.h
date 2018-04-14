/**
 * @file ag.h
 * @author Juan José Escobar Pérez
 * @date 09/12/2016
 * @brief Header file containing the main functions for the islands-based genetic algorithm model
 */

#ifndef AGISLANDS_H
#define AGISLANDS_H

/********************************* Includes *******************************/

#include "clUtils.h"
#include <mpi.h>

/********************************* Methods ********************************/


/**
 * @brief Allocates memory for all subpopulations (parents and children). Also, they are initialized
 * @param conf The structure with all configuration parameters
 * @return The first subpopulations
 */
Individual* initSubpopulations(const Config *const conf);


/**
 * @brief Tournament between randomly selected individuals. The best individuals are stored in the pool
 * @param conf The structure with all configuration parameters
 * @return The pool with the selected individuals
 */
int* getPool(const Config *const conf);


/**
 * @brief Perform binary crossover between two individuals (uniform crossover)
 * @param subpop Current subpopulation
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 * @return The number of generated children
 */
int crossoverUniform(Individual *const subpop, const int *const pool, const Config *const conf);


/**
 * @brief Perform the migrations between subpopulations
 * @param subpops The subpopulations
 * @param nSubpopulations The number of subpopulations involved in the migration
 * @param nIndsFronts0 The number of individuals in the front 0 of each subpopulation
 * @param conf The structure with all configuration parameters
 */
void migration(Individual *const subpops, const int nSubpopulations, const int *const nIndsFronts0, const Config *const conf);


/**
 * @brief Island-based genetic algorithm model running in different modes: Sequential, CPU or GPU only and Heterogeneous (full cooperation between all available OpenCL devices)
 * @param subpops The initial subpopulations
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param dataBase The database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param cpuMP If the individuals evaluation is performed using OpenMP in CPU
 * @param conf The structure with all configuration parameters
 */
void agIslands(Individual *const subpops, CLDevice *const devicesObject, const float *const dataBase, const int *const selInstances, const bool cpuMP, const Config *const conf);

#endif