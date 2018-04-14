/**
 * @file clUtils.h
 * @author Juan José Escobar Pérez
 * @date 01/07/2015
 * @brief Header file containing auxiliary functions for OpenCL
 */

#ifndef CLUTILS_H
#define CLUTILS_H

/********************************* Includes *******************************/

#include "individual.h" // Individual
#include <CL/cl.h> // OpenCL
#include <stdlib.h> // exit
#include <vector> // std::vector...

/******************************** Constants *******************************/

const char *const CL_ERROR_PLATFORMS_NUMBER = "Error: Could not get the number of platforms\n";
const char *const CL_ERROR_PLATFORMS_FOUND = "Error: Platforms not found\n";
const char *const CL_ERROR_PLATFORM_ID = "Error: Could not get the platform id\n";
const char *const CL_ERROR_DEVICES_NUMBER = "Error: Could not get the number of devices\n";
const char *const CL_ERROR_DEVICE_ID = "Error: Could not get the device id\n";
const char *const CL_ERROR_DEVICE_NAME = "Error: Could not get the device name\n";
const char *const CL_ERROR_DEVICE_TYPE = "Error: Could not get the device type\n";
const char *const CL_ERROR_DEVICE_MEM = "Error: Could not get the maximum local memory of the device\n";
const char *const CL_ERROR_DEVICE_CONTEXT = "Error: Could not get the context\n";
const char *const CL_ERROR_DEVICE_QUEUE = "Error: Could not get the command queue\n";
const char *const CL_ERROR_FILE_OPEN = "Error: An error ocurred opening the kernel file\n";
const char *const CL_ERROR_PROGRAM_BUILD = "Error: Could not create the program\n";
const char *const CL_ERROR_PROGRAM_ERRORS = "Error: Could not get the compilation errors\n";
const char *const CL_ERROR_KERNEL_BUILD = "Error: Could not create the kernel\n";
const char *const CL_WARNING_CPU_WI = "Warning: If the device is the CPU, the local work-size must be 1. Local work-size has been set to 1\n";
const char *const CL_ERROR_OBJECT_DB = "Error: Could not create the OpenCL object containing the data base\n";
const char *const CL_ERROR_OBJECT_CENTROIDS = "Error: Could not create the OpenCL object containing the indexes of the initial centroids\n";
const char *const CL_ERROR_OBJECT_SUBPOPS = "Error: Could not create the OpenCL object containing the subpopulations\n";
const char *const CL_ERROR_KERNEL_ARGUMENT1 = "Error: Could not set the first kernel argument\n";
const char *const CL_ERROR_KERNEL_ARGUMENT2 = "Error: Could not set the second kernel argument\n";
const char *const CL_ERROR_KERNEL_ARGUMENT3 = "Error: Could not set the third kernel argument\n";
const char *const CL_ERROR_ENQUEUE_DB = "Error: Could not enqueue the OpenCL object containing the data base\n";
const char *const CL_ERROR_ENQUEUE_CENTROIDS = "Error: Could not enqueue the OpenCL object containing the init centroids\n";
const char *const CL_ERROR_OBJECT_DBT = "Error: Could not create the OpenCL object containing the data base transposed\n";
const char *const CL_ERROR_ENQUEUE_DBT = "Error: Could not enqueue the OpenCL object containing the data base transposed\n";
const char *const CL_ERROR_KERNEL_ARGUMENT6 = "Error: Could not set the sixth kernel argument\n";
const char *const CL_ERROR_DEVICE_FOUND = "Error: Not exists the specified device\n";

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
	std::string deviceName;


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
std::vector<cl_device_id> getAllDevices();


/**
 * @brief Prints a list containing the ID of all available OpenCL devices
 * @param mpiRank The MPI process number which is calling the function
 */
void listDevices(const int mpiRank);

#endif