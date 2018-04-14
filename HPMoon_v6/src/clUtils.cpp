/**
 * @file clUtils.cpp
 * @author Juan José Escobar Pérez
 * @date 04/07/2016
 * @brief File with the necessary implementation for the auxiliary OpenCL functions
 */

/********************************* Includes *******************************/

#include "clUtils.h"
#include <string>

/********************************* Methods ********************************/

/**
 * @brief The destructor
 */
CLDevice::~CLDevice() {

	if (this -> device != NULL) {

		// Resources used are released
		clReleaseContext(this -> context);
		clReleaseCommandQueue(this -> commandQueue);
		clReleaseKernel(this -> kernel);
		clReleaseMemObject(this -> objDataBase);
		clReleaseMemObject(this -> objSelInstances);
		clReleaseMemObject(this -> objSubpopulations);

		// GPU case
		if (this -> deviceType == CL_DEVICE_TYPE_GPU) {
			clReleaseMemObject(this -> objDataBaseTransposed);
		}
	}
}


/**
 * @brief Creates an array of objects containing the OpenCL variables of each device
 * @param dataBase The database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param dataBaseTransposed Database already transposed
 * @param conf The structure with all configuration parameters
 * @return A pointer containing the objects
 */
CLDevice *createDevices(const float *const dataBase, const int *const selInstances, const float *const dataBaseTransposed, const Config *const conf) {


	/********** Find the devices specified in configuration and get the deviceIDs ***********/

	// OpenCL variables
	cl_uint numPlatformsDevices;
	cl_device_type deviceType;
	cl_program program;
	cl_kernel kernel;
	cl_int status;

	// Others variables
	std::vector<cl_device_id> allDevices = getAllDevices();
	CLDevice *devices = new CLDevice[conf -> nDevices];

	for (int dev = 0; dev < conf -> nDevices; ++dev) {

		bool found = false;
		for (int allDev = 0; allDev < allDevices.size() && !found; ++allDev) {

			// Get the specified CPU or GPU device
			char dbuff[120];
			check(clGetDeviceInfo(allDevices[allDev], CL_DEVICE_NAME, sizeof(dbuff), dbuff, NULL) != CL_SUCCESS, CL_ERROR_DEVICE_NAME);

			// If the device exists...
			if (conf -> devices[dev] == dbuff) {
				devices[dev].device = allDevices[allDev];
				devices[dev].deviceName = dbuff;
				check(clGetDeviceInfo(devices[dev].device, CL_DEVICE_TYPE, sizeof(cl_device_type), &(devices[dev].deviceType), NULL) != CL_SUCCESS, CL_ERROR_DEVICE_TYPE);


				/********** GPU memory usage ***********/

				if (devices[dev].deviceType == CL_DEVICE_TYPE_GPU) {
					long int usedMemory = conf -> nFeatures * sizeof(cl_uchar); // Chromosome of the individual
					usedMemory += conf -> nInstances * sizeof(cl_uchar); // Mapping buffer
					usedMemory += conf -> nCentroids * conf -> nFeatures * sizeof(cl_float); // Centroids buffer
					usedMemory += conf -> nInstances * sizeof(cl_float); // DistCentroids buffer
					usedMemory += conf -> nCentroids * sizeof(cl_int); // Samples_in_k buffer

					// Get the maximum local memory size
					long int maxMemory;
					check(clGetDeviceInfo(devices[dev].device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(long int), &maxMemory, NULL) != CL_SUCCESS, CL_ERROR_DEVICE_MEM);

					// Avoid exceeding the maximum local memory available
					if (usedMemory > maxMemory - 1024) { // 1024 bytes of margin
						fprintf(stderr, "Error: Local memory exceeded:\n\tMax memory: %ld bytes\n\tAllow memory: %ld bytes\n\tUsed memory: %ld bytes\n", maxMemory, maxMemory - 1024, usedMemory);
						exit(-1);
					}
				}


				/********** Create context ***********/

				devices[dev].context = clCreateContext(NULL, 1, &(devices[dev].device), 0, 0, &status);
				check(status != CL_SUCCESS, CL_ERROR_DEVICE_CONTEXT);


				/********** Create Command queue ***********/

				devices[dev].commandQueue = clCreateCommandQueue(devices[dev].context, devices[dev].device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &status);
				check(status != CL_SUCCESS, CL_ERROR_DEVICE_QUEUE);


				/********** Create kernel ***********/

				// Open the file containing the kernels
				std::fstream kernels(conf -> kernelsFileName.c_str(), std::fstream::in);
				check(!kernels.is_open(), CL_ERROR_FILE_OPEN);

				// Obtain the size
				kernels.seekg(0, kernels.end);
				size_t fSize = kernels.tellg();
				kernels.seekg(0, kernels.beg);

				char *kernelSource = new char[fSize];
				kernels.read(kernelSource, fSize);
				kernels.close();

				// Create program
				program = clCreateProgramWithSource(devices[dev].context, 1, (const char **) &kernelSource, &fSize, &status);
				check(status != CL_SUCCESS, CL_ERROR_PROGRAM_BUILD);

				// Build program for the device in the context
				char buildOptions[196];
				sprintf(buildOptions, "-I include -D N_INSTANCES=%d -D N_FEATURES=%d -D N_OBJECTIVES=%d -D KMEANS=%d -D MAX_ITER_KMEANS=%d", conf -> nInstances, conf -> nFeatures, conf -> nObjectives, conf -> nCentroids, conf -> maxIterKmeans);
				if (clBuildProgram(program, 1, &(devices[dev].device), buildOptions, 0, 0) != CL_SUCCESS) {
					char buffer[4096];
					fprintf(stderr, "Error: Could not build the program\n");
					check(clGetProgramBuildInfo(program, devices[dev].device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL) != CL_SUCCESS, CL_ERROR_PROGRAM_ERRORS);
					fprintf(stderr, "%s\n", buffer);
					exit(-1);
				}

				// Create kernel
				const char *kernelName = (devices[dev].deviceType == CL_DEVICE_TYPE_GPU) ? "kmeansGPU" : "kmeansCPU";
				devices[dev].kernel = clCreateKernel(program, kernelName, &status);
				check(status != CL_SUCCESS, CL_ERROR_KERNEL_BUILD);


				/******* Work-items *******/

				devices[dev].computeUnits = atoi(conf -> computeUnits[dev].c_str());
				devices[dev].wiLocal = atoi(conf -> wiLocal[dev].c_str());
				devices[dev].wiGlobal = devices[dev].computeUnits * devices[dev].wiLocal;

				// If the device is the CPU, the local work-size must be "1". On the contrary the program will fail
				if (devices[dev].deviceType == CL_DEVICE_TYPE_CPU && devices[dev].wiLocal != 1) {
					devices[dev].wiGlobal = devices[dev].computeUnits;
					devices[dev].wiLocal = 1;
					fprintf(stderr, CL_WARNING_CPU_WI);
				}


				/******* Create and write the dataBase buffers and centroids buffers. Create the subpopulations buffer. Set kernel arguments *******/

				// Create buffers
				devices[dev].objDataBase = clCreateBuffer(devices[dev].context, CL_MEM_READ_ONLY, conf -> nInstances * conf -> nFeatures * sizeof(cl_float), 0, &status);
				check(status != CL_SUCCESS, CL_ERROR_OBJECT_DB);

				devices[dev].objSelInstances = clCreateBuffer(devices[dev].context, CL_MEM_READ_ONLY, conf -> nCentroids * sizeof(cl_int), 0, &status);
				check(status != CL_SUCCESS, CL_ERROR_OBJECT_CENTROIDS);

				devices[dev].objSubpopulations = clCreateBuffer(devices[dev].context, CL_MEM_READ_WRITE, conf -> familySize * sizeof(Individual), 0, &status);
				check(status != CL_SUCCESS, CL_ERROR_OBJECT_SUBPOPS);

				// Sets kernel arguments
				check(clSetKernelArg(devices[dev].kernel, 0, sizeof(cl_mem), (void *)&(devices[dev].objSubpopulations)) != CL_SUCCESS, CL_ERROR_KERNEL_ARGUMENT1);
				check(clSetKernelArg(devices[dev].kernel, 1, sizeof(cl_mem), (void *)&(devices[dev].objSelInstances)) != CL_SUCCESS, CL_ERROR_KERNEL_ARGUMENT2);
				check(clSetKernelArg(devices[dev].kernel, 2, sizeof(cl_mem), (void *)&(devices[dev].objDataBase)) != CL_SUCCESS, CL_ERROR_KERNEL_ARGUMENT3);

				// Write buffers
				check(clEnqueueWriteBuffer(devices[dev].commandQueue, devices[dev].objDataBase, CL_FALSE, 0, conf -> nInstances * conf -> nFeatures * sizeof(cl_float), dataBase, 0, NULL, NULL) != CL_SUCCESS, CL_ERROR_ENQUEUE_DB);
				check(clEnqueueWriteBuffer(devices[dev].commandQueue, devices[dev].objSelInstances, CL_FALSE, 0, conf -> nCentroids * sizeof(cl_int), selInstances, 0, NULL, NULL) != CL_SUCCESS, CL_ERROR_ENQUEUE_CENTROIDS);

				if (devices[dev].deviceType == CL_DEVICE_TYPE_GPU) {
					devices[dev].objDataBaseTransposed = clCreateBuffer(devices[dev].context, CL_MEM_READ_ONLY, conf -> nInstances * conf -> nFeatures * sizeof(cl_float), 0, &status);
					check(status != CL_SUCCESS, CL_ERROR_OBJECT_DBT);
					check(clEnqueueWriteBuffer(devices[dev].commandQueue, devices[dev].objDataBaseTransposed, CL_FALSE, 0, conf -> nInstances * conf -> nFeatures * sizeof(cl_float), dataBaseTransposed, 0, NULL, NULL) != CL_SUCCESS, CL_ERROR_ENQUEUE_DBT);

					// Set the new database as kernel argument
					check(clSetKernelArg(devices[dev].kernel, 5, sizeof(cl_mem), (void *)&(devices[dev].objDataBaseTransposed)) != CL_SUCCESS, CL_ERROR_KERNEL_ARGUMENT6);
				}

				// Resources used are released
				delete[] kernelSource;
				clReleaseProgram(program);

				found = true;
				allDevices.erase(allDevices.begin() + allDev);
			}
		}

		check(!found, CL_ERROR_DEVICE_FOUND);
	}

	return devices;
}


/**
 * @brief Gets the IDs of all available OpenCL devices
 * @return A vector containing the IDs of all devices
 */
std::vector<cl_device_id> getAllDevices() {

	// OpenCL variables
	cl_platform_id *platforms;
	cl_uint numPlatforms;
	cl_uint numPlatformsDevices;
	std::vector<cl_device_id> allDevices;
	cl_device_id *devices;
	cl_int status;

	// Get the number of platforms
	check(clGetPlatformIDs(0, NULL, &numPlatforms) != CL_SUCCESS, CL_ERROR_PLATFORMS_NUMBER);
	check(numPlatforms == 0, CL_ERROR_PLATFORMS_FOUND);

	// Get the platforms
	platforms = new cl_platform_id[numPlatforms];
	check(clGetPlatformIDs(numPlatforms, platforms, NULL) != CL_SUCCESS, CL_ERROR_PLATFORM_ID);

	// Search devices in each platform
	for (int i = 0; i < numPlatforms; ++i) {

		// Get the number of devices of this platform
		check((status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, 0, (cl_uint*) &numPlatformsDevices)) != CL_SUCCESS, CL_ERROR_DEVICES_NUMBER);
		devices = new cl_device_id[numPlatformsDevices];

		// Get all devices of this platform
		check(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numPlatformsDevices, devices, NULL) != CL_SUCCESS, CL_ERROR_DEVICE_ID);
		for (int dev = 0; dev < numPlatformsDevices; ++dev) {
			  allDevices.push_back(devices[dev]);
		}

		delete[] devices;
	}

	delete[] platforms;
	return allDevices;
}


/**
 * @brief Prints a list containing the ID of all available OpenCL devices
 * @param mpiRank The MPI process number which is calling the function
 */
void listDevices(const int mpiRank) {

	// OpenCL variables
	std::vector<cl_device_id> allDevices = getAllDevices();
	std::string devices = "Process " + std::to_string(mpiRank) + ": ";
	devices += (allDevices.empty()) ? "No devices in this node" : "Devices in this node:";

	for(int i = 0; i < allDevices.size(); ++i) {
		char nameBuff[128];
		check(clGetDeviceInfo(allDevices[i], CL_DEVICE_NAME, sizeof(nameBuff), nameBuff, NULL) != CL_SUCCESS, CL_ERROR_DEVICE_NAME);
		devices += "\nDevice Name " + std::to_string(i) + ":\t " + nameBuff;
	}
	fprintf(stdout, "%s\n", devices.c_str());
}