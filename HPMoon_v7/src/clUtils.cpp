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

	if (this -> device != NULL && this -> deviceType != CL_DEVICE_TYPE_CPU) {

		// Resources used are released
		clReleaseContext(this -> context);
		clReleaseCommandQueue(this -> commandQueue);
		clReleaseKernel(this -> kernel);
		clReleaseMemObject(this -> objTrDataBase);
		clReleaseMemObject(this -> objTransposedTrDataBase);
		clReleaseMemObject(this -> objSelInstances);
		clReleaseMemObject(this -> objSubpopulations);
	}
}


/**
 * @brief Creates an array of objects containing the OpenCL variables of each device
 * @param trDataBase The training database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param transposedTrDataBase The training database already transposed
 * @param conf The structure with all configuration parameters
 * @return A pointer containing the objects
 */
CLDevice *createDevices(const float *const trDataBase, const int *const selInstances, const float *const transposedTrDataBase, const Config *const conf) {


	/********** Find the OpenCL devices specified in configuration ***********/

	// OpenCL variables
	cl_uint numPlatformsDevices;
	cl_device_type deviceType;
	cl_program program;
	cl_kernel kernel;
	cl_int status;

	// Others variables
	auto allDevices = getAllDevices();
	CLDevice *devices = new CLDevice[conf -> nDevices + (conf -> ompThreads > 0)];

	for (int dev = 0; dev < conf -> nDevices; ++dev) {

		bool found = false;
		for (int allDev = 0; allDev < allDevices.size() && !found; ++allDev) {

			// Get the specified OpenCL device
			char dbuff[120];
			check(clGetDeviceInfo(allDevices[allDev], CL_DEVICE_NAME, sizeof(dbuff), dbuff, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_DEVICE_NAME);

			// If the device exists...
			if (conf -> devices[dev] == dbuff) {
				devices[dev].device = allDevices[allDev];
				devices[dev].deviceName = dbuff;
				check(clGetDeviceInfo(devices[dev].device, CL_DEVICE_TYPE, sizeof(cl_device_type), &(devices[dev].deviceType), NULL) != CL_SUCCESS, "%s\n", CL_ERROR_DEVICE_TYPE);


				/********** Device local memory usage ***********/

				long int usedMemory = conf -> nFeatures * sizeof(cl_uchar); // Chromosome of the individual
				usedMemory += conf -> trNInstances * sizeof(cl_uchar); // Mapping buffer
				usedMemory += conf -> K * conf -> nFeatures * sizeof(cl_float); // Centroids buffer
				usedMemory += conf -> trNInstances * sizeof(cl_float); // DistCentroids buffer
				usedMemory += conf -> K * sizeof(cl_int); // Samples_in_k buffer

				// Get the maximum local memory size
				long int maxMemory;
				check(clGetDeviceInfo(devices[dev].device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(long int), &maxMemory, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_DEVICE_MAXMEM);

				// Avoid exceeding the maximum local memory available. 1024 bytes of margin
				check(usedMemory > maxMemory - 1024, "%s:\n\tMax memory: %ld bytes\n\tAllow memory: %ld bytes\n\tUsed memory: %ld bytes\n", CL_ERROR_DEVICE_LOCALMEM, maxMemory, maxMemory - 1024, usedMemory);


				/********** Create context ***********/

				devices[dev].context = clCreateContext(NULL, 1, &(devices[dev].device), 0, 0, &status);
				check(status != CL_SUCCESS, "%s\n", CL_ERROR_DEVICE_CONTEXT);


				/********** Create Command queue ***********/

				devices[dev].commandQueue = clCreateCommandQueue(devices[dev].context, devices[dev].device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &status);
				check(status != CL_SUCCESS, "%s\n", CL_ERROR_DEVICE_QUEUE);


				/********** Create kernel ***********/

				// Open the file containing the kernels
				std::fstream kernels(conf -> kernelsFileName.c_str(), std::fstream::in);
				check(!kernels.is_open(), "%s\n", CL_ERROR_FILE_OPEN);

				// Obtain the size
				kernels.seekg(0, kernels.end);
				size_t fSize = kernels.tellg();
				kernels.seekg(0, kernels.beg);

				char *kernelSource = new char[fSize];
				kernels.read(kernelSource, fSize);
				kernels.close();

				// Create program
				program = clCreateProgramWithSource(devices[dev].context, 1, (const char **) &kernelSource, &fSize, &status);
				check(status != CL_SUCCESS, "%s\n", CL_ERROR_PROGRAM_BUILD);

				// Build program for the device in the context
				char buildOptions[196];
				sprintf(buildOptions, "-I include -D N_INSTANCES=%d -D N_FEATURES=%d -D N_OBJECTIVES=%d -D K=%d -D MAX_ITER_KMEANS=%d", conf -> trNInstances, conf -> nFeatures, conf -> nObjectives, conf -> K, conf -> maxIterKmeans);
				if (clBuildProgram(program, 1, &(devices[dev].device), buildOptions, 0, 0) != CL_SUCCESS) {
					char buffer[4096];
					fprintf(stderr, "Error: Could not build the program\n");
					check(clGetProgramBuildInfo(program, devices[dev].device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_PROGRAM_ERRORS);
					check(true, "%s\n", buffer);
				}

				// Create kernel
				const char *kernelName = (devices[dev].deviceType == CL_DEVICE_TYPE_GPU) ? "kmeansGPU" : "";
				devices[dev].kernel = clCreateKernel(program, kernelName, &status);
				check(status != CL_SUCCESS, "%s\n", CL_ERROR_KERNEL_BUILD);


				/******* Work-items *******/

				devices[dev].computeUnits = atoi(conf -> computeUnits[dev].c_str());
				devices[dev].wiLocal = atoi(conf -> wiLocal[dev].c_str());
				devices[dev].wiGlobal = devices[dev].computeUnits * devices[dev].wiLocal;


				/******* Create and write the databases and centroids buffers. Create the subpopulations buffer. Set kernel arguments *******/

				// Create buffers
				devices[dev].objSubpopulations = clCreateBuffer(devices[dev].context, CL_MEM_READ_WRITE, conf -> familySize * sizeof(Individual), 0, &status);
				check(status != CL_SUCCESS, "%s\n", CL_ERROR_OBJECT_SUBPOPS);

				devices[dev].objTrDataBase = clCreateBuffer(devices[dev].context, CL_MEM_READ_ONLY, conf -> trNInstances * conf -> nFeatures * sizeof(cl_float), 0, &status);
				check(status != CL_SUCCESS, "%s\n", CL_ERROR_OBJECT_TRDB);

				devices[dev].objTransposedTrDataBase = clCreateBuffer(devices[dev].context, CL_MEM_READ_ONLY, conf -> trNInstances * conf -> nFeatures * sizeof(cl_float), 0, &status);
				check(status != CL_SUCCESS, "%s\n", CL_ERROR_OBJECT_TTRDB);

				devices[dev].objSelInstances = clCreateBuffer(devices[dev].context, CL_MEM_READ_ONLY, conf -> K * sizeof(cl_int), 0, &status);
				check(status != CL_SUCCESS, "%s\n", CL_ERROR_OBJECT_CENTROIDS);

				// Sets kernel arguments
				check(clSetKernelArg(devices[dev].kernel, 0, sizeof(cl_mem), (void *)&(devices[dev].objSubpopulations)) != CL_SUCCESS, "%s\n", CL_ERROR_KERNEL_ARGUMENT1);

				check(clSetKernelArg(devices[dev].kernel, 1, sizeof(cl_mem), (void *)&(devices[dev].objSelInstances)) != CL_SUCCESS, "%s\n", CL_ERROR_KERNEL_ARGUMENT2);

				check(clSetKernelArg(devices[dev].kernel, 2, sizeof(cl_mem), (void *)&(devices[dev].objTrDataBase)) != CL_SUCCESS, "%s\n", CL_ERROR_KERNEL_ARGUMENT3);

				check(clSetKernelArg(devices[dev].kernel, 5, sizeof(cl_mem), (void *)&(devices[dev].objTransposedTrDataBase)) != CL_SUCCESS, "%s\n", CL_ERROR_KERNEL_ARGUMENT6);

				// Write buffers
				check(clEnqueueWriteBuffer(devices[dev].commandQueue, devices[dev].objTrDataBase, CL_FALSE, 0, conf -> trNInstances * conf -> nFeatures * sizeof(cl_float), trDataBase, 0, NULL, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_ENQUEUE_TRDB);
				check(clEnqueueWriteBuffer(devices[dev].commandQueue, devices[dev].objSelInstances, CL_FALSE, 0, conf -> K * sizeof(cl_int), selInstances, 0, NULL, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_ENQUEUE_CENTROIDS);
				check(clEnqueueWriteBuffer(devices[dev].commandQueue, devices[dev].objTransposedTrDataBase, CL_FALSE, 0, conf -> trNInstances * conf -> nFeatures * sizeof(cl_float), transposedTrDataBase, 0, NULL, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_ENQUEUE_TTRDB);

				// Resources used are released
				delete[] kernelSource;
				clReleaseProgram(program);

				found = true;
				allDevices.erase(allDevices.begin() + allDev);
			}
		}

		check(!found, "%s\n", CL_ERROR_DEVICE_FOUND);
	}


	/********** Add the CPU if has been enabled in configuration ***********/

	if (conf -> ompThreads > 0) {
		devices[conf -> nDevices].deviceType = CL_DEVICE_TYPE_CPU;
		devices[conf -> nDevices].computeUnits = conf -> ompThreads;
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
	check(clGetPlatformIDs(0, NULL, &numPlatforms) != CL_SUCCESS, "%s\n", CL_ERROR_PLATFORMS_NUMBER);
	check(numPlatforms == 0, "%s\n", CL_ERROR_PLATFORMS_FOUND);

	// Get the platforms
	platforms = new cl_platform_id[numPlatforms];
	check(clGetPlatformIDs(numPlatforms, platforms, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_PLATFORM_ID);

	// Search devices in each platform
	for (int i = 0; i < numPlatforms; ++i) {

		// Get the number of devices of this platform
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, 0, (cl_uint*) &numPlatformsDevices);
		check(status != CL_SUCCESS && status != CL_DEVICE_NOT_FOUND, "%s\n", CL_ERROR_DEVICES_NUMBER);

		// Get all devices of this platform
		if (numPlatformsDevices > 0) {
			devices = new cl_device_id[numPlatformsDevices];
			check(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numPlatformsDevices, devices, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_DEVICE_ID);
			allDevices.insert(allDevices.end(), devices, devices + numPlatformsDevices);
			delete[] devices;
		}
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
	auto allDevices = getAllDevices();
	std::string devices = "Process " + std::to_string(mpiRank) + ": ";
	devices += (allDevices.empty()) ? "No OpenCL devices in this node" : "OpenCL Devices in this node:";

	for(int i = 0; i < allDevices.size(); ++i) {
		char nameBuff[128];
		size_t maxWorkitems[3];
		unsigned int maxCU;
		check(clGetDeviceInfo(allDevices[i], CL_DEVICE_NAME, sizeof(nameBuff), nameBuff, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_DEVICE_NAME);
		check(clGetDeviceInfo(allDevices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxCU, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_DEVICE_MAXCU);
		check(clGetDeviceInfo(allDevices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, maxWorkitems, NULL) != CL_SUCCESS, "%s\n", CL_ERROR_DEVICE_MAXWORKITEMS);
		devices += "\n\tDevice " + std::to_string(i) + " ->  Name: " + nameBuff + ";  Compute units: " + std::to_string(maxCU) + ";  Max Work-items: " + std::to_string(maxWorkitems[0]);
	}
	fprintf(stdout, "%s\n", devices.c_str());
}
