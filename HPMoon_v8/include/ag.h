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
Individual* createSubpopulations(const Config *const conf);


/**
 * @brief Island-based genetic algorithm model
 * @param subpops The initial subpopulations
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param trDataBase The training database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param conf The structure with all configuration parameters
 */
void agIslands(Individual *subpops, CLDevice *const devicesObject, const float *const trDataBase, const int *const selInstances, const Config *const conf);

#endif
