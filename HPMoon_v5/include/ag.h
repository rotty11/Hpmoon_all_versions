/**
 * @file ag.h
 * @author Juan José Escobar Pérez
 * @date 09/12/2016
 * @brief Header file containing the main functions for the islands-based genetic algorithm model
 *
 */

#ifndef AGISLANDS_H
#define AGISLANDS_H

/********************************* Includes *******************************/

#include "clUtils.h"

/********************************* Methods ********************************/


/**
 * @brief Allocates memory for all subpopulations (parents and children). Also, they are initialized
 * @param conf The structure with all configuration parameters
 * @return The first subpopulations
 */
Individual* initSubpopulations(const Config *const conf);


/**
 * @brief Competition between randomly selected individuals. The best individuals are stored in the pool
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 */
void fillPool(int *const pool, const Config *const conf);


/**
 * @brief Perform binary crossover between two individuals (uniform crossover)
 * @param subpop Current subpopulation
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 * @return The number of generated children
 */
int crossoverUniform(Individual *const subpop, const int *const pool, const Config *const conf);


/**
 * @brief Perform the individuals migrations between subpopulations
 * @param subpops The subpopulations
 * @param nIndsFronts0 The number of individuals in the front 0 of each subpopulation
 * @param conf The structure with all configuration parameters
 */
void migration(Individual *const subpops, const int *const nIndsFronts0, const Config *const conf);


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
void agIslands(const Individual *const subpopsOrig, const int nDevices, CLDevice *const devicesObject, const float *const dataBase, const int *const selInstances, const double *const referencePoint, const Config *const conf);

#endif