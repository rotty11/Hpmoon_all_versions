/**
 * @file ag.h
 * @author Juan José Escobar Pérez
 * @date 14/07/2015
 * @brief Header file containing the main functions for the genetic algorithm
 *
 */

#ifndef AG_H
#define AG_H

/********************************* Includes *******************************/

#include "clUtils.h"

/********************************* Methods ********************************/


/**
 * @brief Allocates memory for parents and children. Also, they are initialized
 * @param population The first population
 * @param conf The structure with all configuration parameters
 */
void initPopulation(Individual *population, const Config *conf);


/**
 * @brief Competition between randomly selected individuals. The best individuals are stored in the pool
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 */
void fillPool(int *pool, const Config *conf);


/**
 * @brief Perform binary crossover between two individuals (2-point crossover)
 * @param population Current population
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 * @return The number of generated children
 */
int crossover2p(Individual *population, const int *pool, const Config *conf);


/**
 * @brief Perform binary crossover between two individuals (uniform crossover)
 * @param population Current population
 * @param pool Position of the selected individuals for the crossover
 * @param conf The structure with all configuration parameters
 * @return The number of generated children
 */
int crossoverUniform(Individual *population, const int *pool, const Config *conf);


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
void agAlgorithm(Individual *populationOrig, cl_device_type executionMode, CLDevice *devicesObject, const float *const dataBase, const int *const selInstances, const double *const referencePoint, const Config *conf);

#endif