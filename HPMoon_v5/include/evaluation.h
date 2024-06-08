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
 * @param subpop The first individual to evaluate of the current subpopulation
 * @param nIndividuals The number of individuals which will be evaluated
 * @param dataBase The database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param conf The structure with all configuration parameters
 */
void evaluationSEQ(Individual *const subpop, const int nIndividuals, const float *const dataBase, const int *const selInstances, const Config *const conf);


/**
 * @brief Evaluation of each individual on OpenCL devices
 * @param subpop The first individual to evaluate of the current subpopulation
 * @param nIndividuals The number of individuals which will be evaluated
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param nDevices The number of devices that will execute the evaluation
 * @param conf The structure with all configuration parameters
 */
void evaluationCL(Individual *const subpop, const int nIndividuals, CLDevice *const devicesObject, const int nDevices, const Config *const conf);


/**
 * @brief Normalize the fitness for each individual
 * @param subpop The first individual to normalize of the current subpopulation
 * @param nIndividuals The number of individuals which will be normalized
 * @param conf The structure with all configuration parameters
 */
void normalizeFitness(Individual *const subpop, const int nIndividuals, const Config *const conf);


/**
 * @brief Gets the hypervolume measure of the subpopulation
 * @param subpop Current subpopulation
 * @param nIndFront0 The number of individuals in the front 0
 * @param referencePoint The necessary reference point for calculation
 * @param conf The structure with all configuration parameters
 * @return The value of the hypervolume
 */
float getHypervolume(const Individual *const subpop, const int nIndFront0, const double *const referencePoint, const Config *const conf);


/**
 * @brief Gets the initial centroids (instances choosen randomly)
 * @param conf The structure with all configuration parameters
 * @return The instances choosen as initial centroids will be stored
 */
int* getCentroids(const Config *const conf);


/**
 * @brief Generates the gnuplot data
 * @param subpop Current subpopulation
 * @param nIndFront0 The number of individuals in the front 0
 * @param executionerMode The execution mode
 * @param conf The structure with all configuration parameters
 */
void generateDataPlot(const Individual *const subpop, const int nIndFront0, const string executionerMode, const Config *const conf);


/**
 * @brief Generates gnuplot code for data display
 * @param referencePoint The reference point used for the hypervolume calculation
 * @param conf The structure with all configuration parameters
 */
void generateGnuplot(const double *const referencePoint, const Config *const conf);

#endif