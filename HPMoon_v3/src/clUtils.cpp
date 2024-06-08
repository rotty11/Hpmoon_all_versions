/**
 * @file clUtils.cpp
 * @author Juan José Escobar Pérez
 * @date 04/07/2016
 * @brief File with the necessary implementation for the auxiliary OpenCL functions
 *
 */

/********************************* Includes *******************************/

#include "clUtils.h"
#include <string.h> // strcmp...

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
		clReleaseMemObject(this -> objPopulation);

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
CLDevice *createDevices(const float *const dataBase, const int *const selInstances, const float *const dataBaseTransposed, const Config *conf) {


	/********** Find the devices specified in configuration and get the deviceIDs ***********/

	// OpenCL variables
	cl_uint numPlatformsDevices;
	cl_device_type deviceType;
	cl_program program;
	cl_kernel kernel;
	cl_int status;

	// Others variables
	vector<cl_device_id> allDevices = getAllDevices();
	CLDevice *devices = new CLDevice[conf -> nDevices];

	for (int dev = 0; dev < conf -> nDevices; ++dev) {

		bool found = false;
		for (int allDev = 0; allDev < allDevices.size() && !found; ++allDev) {

			// Get the specified CPU or GPU device
			char dbuff[120];
			if (clGetDeviceInfo(allDevices[allDev], CL_DEVICE_NAME, sizeof(dbuff), dbuff, NULL) != CL_SUCCESS) {
				fprintf(stderr, "Error: Could not get the information of the device\n");
				exit(-1);
			}

			// If the device exists...
			if (strcmp(dbuff, conf -> devices[dev].c_str()) == 0) {
				devices[dev].device = allDevices[allDev];
				devices[dev].deviceName = dbuff;
				if (clGetDeviceInfo(devices[dev].device, CL_DEVICE_TYPE, sizeof(cl_device_type), &(devices[dev].deviceType), NULL) != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not get the device type\n");
					exit(-1);
				}


				/********** GPU memory usage ***********/

				if (devices[dev].deviceType == CL_DEVICE_TYPE_GPU) {
					long int usedMemory = conf -> nFeatures * sizeof(cl_uchar); // Chromosome of the individual
					usedMemory += conf -> nInstances * sizeof(cl_uchar); // Mapping buffer
					usedMemory += conf -> nCentroids * conf -> nFeatures * sizeof(cl_float); // Centroids buffer
					usedMemory += conf -> nInstances * sizeof(cl_float); // DistCentroids buffer
					usedMemory += conf -> nCentroids * sizeof(cl_int); // Samples_in_k buffer

					// Get the maximum local memory size
					long int maxMemory;
					if (clGetDeviceInfo(devices[dev].device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(long int), &maxMemory, NULL) != CL_SUCCESS) {
						fprintf(stderr, "Error: Could not get the maximum local memory of the device\n");
						exit(-1);
					}

					// Avoid exceeding the maximum local memory available
					if (usedMemory > maxMemory - 1024) { // 1024 bytes of margin
						fprintf(stderr, "Error: Local memory exceeded:\n\tMax memory: %ld bytes\n\tAllow memory: %ld bytes\n\tUsed memory: %ld bytes\n", maxMemory, maxMemory - 1024, usedMemory);
						exit(-1);
					}
				}


				/********** Create context ***********/

				devices[dev].context = clCreateContext(NULL, 1, &(devices[dev].device), 0, 0, &status);
				if (status != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not get the context\n");
					exit(-1);
				}


				/********** Create Command queue ***********/

				devices[dev].commandQueue = clCreateCommandQueue(devices[dev].context, devices[dev].device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &status);
				if (status != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not get the command queue\n");
					exit(-1);
				}


				/********** Create kernel ***********/

				// Open the file containing the kernels
				FILE *kernels = fopen(conf -> kernelsFileName.c_str(), "r");
				if (!kernels) {
					fprintf(stderr, "Error: An error ocurred opening the kernel file\n");
					exit(-1);
				}

				// Obtain the size
				fseek(kernels , 0 , SEEK_END);
				size_t fSize = ftell(kernels);
				char *kernelSource = new char[fSize];
				rewind(kernels);
				fread(kernelSource, 1, fSize, kernels);
				fclose(kernels);

				// Create program
				program = clCreateProgramWithSource(devices[dev].context, 1, (const char **) &kernelSource, &fSize, &status);
				if (status != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not create the program\n");
					exit(-1);
				}

				// Build program for the device in the context
				char buildOptions[156];
				sprintf(buildOptions, "-I include -D N_INSTANCES=%d -D N_FEATURES=%d -D N_OBJECTIVES=%d -D KMEANS=%d -D MAX_ITER_KMEANS=%d", conf -> nInstances, conf -> nFeatures, conf -> nObjectives, conf -> nCentroids, conf -> maxIterKmeans);
				if (clBuildProgram(program, 1, &(devices[dev].device), buildOptions, 0, 0) != CL_SUCCESS) {
					char buffer[4096];
					fprintf(stderr, "Error: Could not build the program\n");
					if (clGetProgramBuildInfo(program, devices[dev].device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL) != CL_SUCCESS) {
						fprintf(stderr, "Error: Could not get the compilation errors\n");
						exit(-1);
					}
					fprintf(stderr, "%s\n", buffer);
					exit(-1);
				}

				// Create kernel
				const char *kernelName = (devices[dev].deviceType == CL_DEVICE_TYPE_GPU) ? "kmeansGPU" : "kmeansCPU";
				devices[dev].kernel = clCreateKernel(program, kernelName, &status);
				if (status != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not create the kernel\n");
					exit(-1);
				}


				/******* Create and write the dataBase buffers and centroids buffer. Create the population buffer. Set kernel arguments *******/

				// Create buffers
				devices[dev].objDataBase = clCreateBuffer(devices[dev].context, CL_MEM_READ_ONLY, conf -> nInstances * conf -> nFeatures * sizeof(cl_float), 0, &status);
				if (status != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not create the OpenCL object containing the data base\n");
					exit(-1);
				}

				devices[dev].objPopulation = clCreateBuffer(devices[dev].context, CL_MEM_READ_WRITE, conf -> totalIndividuals * sizeof(Individual), 0, &status);
				if (status != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not create the OpenCL object containing the population\n");
					exit(-1);
				}

				devices[dev].objSelInstances = clCreateBuffer(devices[dev].context, CL_MEM_READ_ONLY, conf -> nCentroids * sizeof(cl_int), 0, &status);
				if (status != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not create the OpenCL object containing the indexes of the initial centroids\n");
					exit(-1);
				}

				// Sets kernel arguments
				if (clSetKernelArg(devices[dev].kernel, 0, sizeof(cl_mem), (void *)&(devices[dev].objPopulation)) != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not set the first kernel argument \n");
					exit(-1);
				}

				if (clSetKernelArg(devices[dev].kernel, 1, sizeof(cl_mem), (void *)&(devices[dev].objSelInstances)) != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not set the second kernel argument \n");
					exit(-1);
				}

				if (clSetKernelArg(devices[dev].kernel, 2, sizeof(cl_mem), (void *)&(devices[dev].objDataBase)) != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not set the third kernel argument \n");
					exit(-1);
				}

				// Write buffers
				if (clEnqueueWriteBuffer(devices[dev].commandQueue, devices[dev].objDataBase, CL_FALSE, 0, conf -> nInstances * conf -> nFeatures * sizeof(cl_float), dataBase, 0, NULL, NULL) != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not enqueue the OpenCL object containing the data base\n");
					exit(-1);
				}

				if (clEnqueueWriteBuffer(devices[dev].commandQueue, devices[dev].objSelInstances, CL_FALSE, 0, conf -> nCentroids * sizeof(cl_int), selInstances, 0, NULL, NULL) != CL_SUCCESS) {
					fprintf(stderr, "Error: Could not enqueue the OpenCL object containing the init centroids\n");
					exit(-1);
				}

				if (devices[dev].deviceType == CL_DEVICE_TYPE_GPU) {
					devices[dev].objDataBaseTransposed = clCreateBuffer(devices[dev].context, CL_MEM_READ_ONLY, conf -> nInstances * conf -> nFeatures * sizeof(cl_float), 0, &status);
					if (status != CL_SUCCESS) {
						fprintf(stderr, "Error: Could not create the OpenCL object containing the data base transposed\n");
						exit(-1);
					}

					if (clEnqueueWriteBuffer(devices[dev].commandQueue, devices[dev].objDataBaseTransposed, CL_FALSE, 0, conf -> nInstances * conf -> nFeatures * sizeof(cl_float), dataBaseTransposed, 0, NULL, NULL) != CL_SUCCESS) {
						fprintf(stderr, "Error: Could not enqueue the OpenCL object containing the data base transposed\n");
						exit(-1);
					}

					// Set the new database as kernel argument
					if (clSetKernelArg(devices[dev].kernel, 5, sizeof(cl_mem), (void *)&(devices[dev].objDataBaseTransposed)) != CL_SUCCESS) {
						fprintf(stderr, "Error: Could not set the sixth kernel argument \n");
						exit(-1);
					}
				}


				/******* Work-items *******/

				devices[dev].computeUnits = atoi(conf -> computeUnits[dev].c_str());
				devices[dev].wiLocal = atoi(conf -> wiLocal[dev].c_str());
				devices[dev].wiGlobal = devices[dev].computeUnits * devices[dev].wiLocal;

				// If the device is the CPU, the local work-size must be "1". On the contrary the program will fail
				if (devices[dev].deviceType == CL_DEVICE_TYPE_CPU && devices[dev].wiLocal != 1) {
					devices[dev].wiGlobal = devices[dev].computeUnits;
					devices[dev].wiLocal = 1;
					fprintf(stderr, "Warning: If the device is the CPU, the local work-size must be 1. Local work-size has been set to 1\n");
				}

				// Resources used are released
				delete[] kernelSource;
				clReleaseProgram(program);

				found = true;
				allDevices.erase(allDevices.begin() + allDev);
			}
		}

		if (!found) {
			fprintf(stderr, "Error: Not exists the specified device\n");
			exit(-1);
		}
	}

	return devices;	
}


/**
 * @brief Gets the IDs of all available OpenCL devices
 * @return A vector containing the IDs of all devices
 */
vector<cl_device_id> getAllDevices() {

	// OpenCL variables
	cl_platform_id *platforms;
	cl_uint numPlatforms;
	cl_uint numPlatformsDevices;
	vector<cl_device_id> allDevices;
	cl_device_id *devices;
	cl_int status;

	// Get the number of platforms
	if (clGetPlatformIDs(0, NULL, &numPlatforms) != CL_SUCCESS) {
		fprintf(stderr, "Error: Could not get the number of platforms\n");
		exit(-1);
	}

	if (numPlatforms == 0) {
		fprintf(stderr, "Error: Platforms not found\n");
		exit(-1);
	}

	// Get the platforms
	platforms = new cl_platform_id[numPlatforms];
	if (clGetPlatformIDs(numPlatforms, platforms, NULL) != CL_SUCCESS) {
		fprintf(stderr, "Error: Could not get the platform id\n");
		exit(-1);
	}

	// Search devices in each platform
	for (int i = 0; i < numPlatforms; ++i) {

		// Get the number of devices of this platform
		if ((status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, 0, (cl_uint*) &numPlatformsDevices)) != CL_SUCCESS) {
			fprintf(stderr, "Error: Could not get the number of devices\n");
			exit(-1);
		}
		else {
			devices = new cl_device_id[numPlatformsDevices];

			// Get all devices of this platform
			if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numPlatformsDevices, devices, NULL) != CL_SUCCESS) {
				fprintf(stderr, "Error: Could not get the device id\n");
				exit(-1);
			}
			for (int dev = 0; dev < numPlatformsDevices; ++dev) {
				  allDevices.push_back(devices[dev]);
			}

			delete[] devices;
		}
	}

	delete[] platforms;
	return allDevices;
}


/**
 * @brief Prints a list containing the ID of all available OpenCL devices
 */
void listDevices() {

	// OpenCL variables
	vector<cl_device_id> allDevices = getAllDevices();

	for(int i = 0; i < allDevices.size(); ++i) {
		char nameBuff[128];
		if (clGetDeviceInfo(allDevices[i], CL_DEVICE_NAME, sizeof(nameBuff), nameBuff, NULL) != CL_SUCCESS) {
			fprintf(stderr, "Error: Could not get the device name\n");
			exit(-1);
		}
		fprintf(stdout, "\nDevice Name %d:\t %s", i, nameBuff);
	}
	fprintf(stdout, "\n\n");
}