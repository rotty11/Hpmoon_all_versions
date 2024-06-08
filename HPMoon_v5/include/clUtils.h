/**
 * @file clUtils.h
 * @author Juan José Escobar Pérez
 * @date 01/07/2015
 * @brief Header file containing auxiliary functions for OpenCL
 *
 */

#ifndef CLUTILS_H
#define CLUTILS_H

/********************************* Includes *******************************/

#include "individual.h" // Individual
#include <CL/cl.h> // OpenCL
#include <stdlib.h> // exit
#include <vector> // vector

/********************************* Structures ********************************/

/**
 * @brief Structure containing the OpenCL variables of a device
 */
typedef struct CLDevice {


	/**
	 * @brief Identifier for the device
	 */
	cl_device_id device;


	/**
	 * @brief The device type
	 */
	cl_device_type deviceType;


	/**
	 * @brief The context associated to the device
	 */
	cl_context context;


	/**
	 * @brief The command queue which contains the tasks (reads/writes on the device...)
	 */
	cl_command_queue commandQueue;


	/**
	 * @brief The OpenCL kernel with the implementation of K-means
	 */
	cl_kernel kernel;


	/**
	 * @brief OpenCL object which contains the database
	 */
	cl_mem objDataBase;


	/**
	 * @brief OpenCL object which contains the indexes of the instances choosen as initial centroids
	 */
	cl_mem objSelInstances;


	/**
	 * @brief OpenCL object which contains the current subpopulations
	 */
	cl_mem objSubpopulations;


	/**
	 * @brief OpenCL object which contains the database transposed
	 */
	cl_mem objDataBaseTransposed;


	/**
	 * @brief The number of compute units specified for this device
	 */
	int computeUnits;


	/**
	 * @brief The number of global work-items specified for this device
	 */
	size_t wiGlobal;


	/**
	 * @brief The number of local work-items specified for this device
	 */
	size_t wiLocal;


	/**
	 * @brief The device name
	 */
	string deviceName;


	/********************************* Methods ********************************/

	/**
	 * @brief The destructor
	 */
	~CLDevice();

} CLDevice;


/********************************* Methods ********************************/

/**
 * @brief Creates an array of objects containing the OpenCL variables of each device
 * @param dataBase The database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param dataBaseTransposed Database already transposed
 * @param conf The structure with all configuration parameters
 * @return A pointer containing the objects
 */
CLDevice *createDevices(const float *const dataBase, const int *const selInstances, const float *const dataBaseTransposed, const Config *const conf);


/**
 * @brief Gets the IDs of all available OpenCL devices
 * @return A vector containing the IDs of all devices
 */
vector<cl_device_id> getAllDevices();


/**
 * @brief Prints a list containing the ID of all available OpenCL devices
 */
void listDevices();

#endif