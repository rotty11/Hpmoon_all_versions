/**
 * @file evaluation.h
 * @author Juan José Escobar Pérez
 * @date 20/06/2015
 * @brief Header file for the evaluation of the individuals
 *
 */

#ifndef EVALUATION_H
#define EVALUATION_H

/********************************* Includes *******************************/

#include "clUtils.h"

/********************************* Methods ********************************/

/**
 * @brief Evaluation of each individual in Sequential mode
 * @param pop Current population
 * @param begin The first individual to evaluate
 * @param end The "end-1" position is the last individual to evaluate
 * @param dataBase The database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param conf The structure with all configuration parameters
 */
void evaluationSEQ(Individual *pop, const int begin, const int end, const float *const dataBase, const int *const selInstances, const Config *conf);


/**
 * @brief Evaluation of each individual on OpenCL devices
 * @param pop Current population
 * @param begin The first individual to evaluate
 * @param end The "end-1" position is the last individual to evaluate
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param nDevices The number of devices that will execute the evaluation
 * @param conf The structure with all configuration parameters
 */
void evaluationCL(Individual *pop, const int begin, const int end, CLDevice *devicesObject, const int nDevices, const Config *conf);


/**
 * @brief Normalize the fitness for each individual
 * @param pop Current population
 * @param begin The first individual to normalize the fitness
 * @param end The "end-1" position is the last individual to normalize the fitness
 * @param conf The structure with all configuration parameters
 */
void normalizeFitness(Individual *pop, const int begin, const int end, const Config *conf);


/**
 * @brief Gets the hypervolume measure of the population
 * @param pop Current population
 * @param nIndFront0 The number of individuals in the front 0
 * @param referencePoint The necessary reference point for calculation
 * @param conf The structure with all configuration parameters
 * @return The value of the hypervolume
 */
float getHypervolume(const Individual *const pop, const int nIndFront0, const double *referencePoint, const Config *conf);


/**
 * @brief Gets the initial centroids (instances choosen randomly)
 * @param selInstances Where the instances choosen as initial centroids will be stored
 * @param conf The structure with all configuration parameters
 */
void getCentroids(int *selInstances, const Config *conf);


/**
 * @brief Generates the gnuplot data
 * @param pop Current population
 * @param nIndFront0 The number of individuals in the front 0
 * @param executionerMode The execution mode
 * @param conf The structure with all configuration parameters
 */
void generateDataPlot(const Individual *const pop, const int nIndFront0, const string executionerMode, const Config *conf);


/**
 * @brief Generates gnuplot code for data display
 * @param referencePoint The reference point used for the hypervolume calculation
 * @param conf The structure with all configuration parameters
 */
void generateGnuplot(const double *const referencePoint, const Config *conf);

#endif