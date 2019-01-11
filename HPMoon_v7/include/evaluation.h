/**
 * @file evaluation.h
 * @author Juan José Escobar Pérez
 * @date 20/06/2015
 * @brief Header file for the evaluation of the individuals
 */

#ifndef EVALUATION_H
#define EVALUATION_H

/********************************* Includes *******************************/

#include "clUtils.h"

/******************************** Constants *******************************/

const char *const EV_ERROR_ENQUEUE_INDIVIDUALS = "Error: Could not enqueue the OpenCL object containing the individuals";
const char *const EV_ERROR_KERNEL_ARGUMENT4 = "Error: Could not set the fourth kernel argument";
const char *const EV_ERROR_KERNEL_ARGUMENT5 = "Error: Could not set the fifth kernel argument";
const char *const EV_ERROR_ENQUEUE_KERNEL = "Error: Could not run the kernel";
const char *const EV_ERROR_ENQUEUE_READING = "Error: Could not read the data from the device";
const char *const EV_ERROR_DATA_OPEN = "Error: An error ocurred opening or writting the data file";
const char *const EV_ERROR_PLOT_OPEN = "Error: An error ocurred opening or writting the plot file";
const char *const EV_ERROR_OBJECTIVES_NUMBER = "Error: Gnuplot is only available for two objectives by now. Not generated gnuplot file";

/********************************* Methods ********************************/


/**
 * @brief Evaluation of each individual in CPU in Sequential mode or using OpenMP
 * @param subpop The first individual to evaluate of the current subpopulation
 * @param nIndividuals The number of individuals which will be evaluated
 * @param trDataBase The training database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param nThreads The number of threads to perform the individuals evaluation
 * @param conf The structure with all configuration parameters
 */
void evaluationCPU(Individual *const subpop, const int nIndividuals, const float *const trDataBase, const int *const selInstances, const int nThreads, const Config *const conf);


/**
 * @brief Evaluation of each individual on OpenCL devices
 * @param subpop The first individual to evaluate of the current subpopulation
 * @param nIndividuals The number of individuals which will be evaluated
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param nDevices The number of devices that will execute the evaluation
 * @param trDataBase The training database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param conf The structure with all configuration parameters
 */
void evaluationHET(Individual *const subpop, const int nIndividuals, CLDevice *const devicesObject, const int nDevices, const float *const trDataBase, const int *const selInstances, const Config *const conf);


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
 * @param conf The structure with all configuration parameters
 * @return The value of the hypervolume
 */
float getHypervolume(const Individual *const subpop, const int nIndFront0, const Config *const conf);


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
 * @param conf The structure with all configuration parameters
 */
void generateDataPlot(const Individual *const subpop, const int nIndFront0, const Config *const conf);


/**
 * @brief Generates gnuplot code for data display
 * @param conf The structure with all configuration parameters
 */
void generateGnuplot(const Config *const conf);

#endif
