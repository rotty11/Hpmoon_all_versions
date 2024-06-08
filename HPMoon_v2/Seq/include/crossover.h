/**
 * @file crossover.h
 * @author Juan José Escobar Pérez
 * @date 29/06/2015
 * @brief Header file for the crossover between individuals
 *
 */

#ifndef CROSSOVER_H
#define CROSSOVER_H

/********************************* Includes *******************************/

#include "individual.h" // Individual

/********************************* Methods ********************************/

/**
 * @brief Perform binary crossover between two individuals
 * @param pop Current population
 * @param populationSize The size of the population
 * @param pool Position of the selected individuals for the crossover
 * @param poolSize Number of selected individuals for the crossover
 * @param nObjectives The number of objectives
 * @param nFeatures The number of features (columns) of the database
 * @return The number of generated children
 */
int crossover(individual *pop, const int populationSize, const int *pool, const int poolSize, const unsigned char nObjectives, const int nFeatures);

#endif