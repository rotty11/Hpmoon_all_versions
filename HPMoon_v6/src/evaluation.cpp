/**
 * @file evaluation.cpp
 * @author Juan José Escobar Pérez
 * @date 20/06/2015
 * @brief File with the necessary implementation for the evaluation of the individuals
 */

/********************************** Includes **********************************/

#include "evaluation.h"
#include "zitzler.h"
#include <omp.h> // OpenMP
#include <math.h> // exp, sqrt, INFINITY

/********************************* Methods ********************************/


/**
 * @brief Evaluation of each individual in CPU in Sequential mode or using OpenMP
 * @param subpop The first individual to evaluate of the current subpopulation
 * @param nIndividuals The number of individuals which will be evaluated
 * @param dataBase The database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param nThreads The number of threads to perform the individuals evaluation
 * @param conf The structure with all configuration parameters
 */
void evaluationCPU(Individual *const subpop, const int nIndividuals, const float *const dataBase, const int *const selInstances, const int nThreads, const Config *const conf) {


	/************ K-means algorithm ***********/

	#pragma omp parallel num_threads(nThreads)
	{
		const int totalCoord = conf -> nCentroids * conf -> nFeatures;

		unsigned char mapping[conf -> nInstances];
		float centroids[totalCoord];
		float distCentroids[conf -> nInstances];
		int samples_in_k[conf -> nCentroids];

		// Evaluate all individuals
		#pragma omp for
		for (int ind = 0; ind < nIndividuals; ++ind) {

			// The centroids will have the selected features of the individual
			for (int k = 0; k < conf -> nCentroids; ++k) {
				int posDataBase = selInstances[k] * conf -> nFeatures;
				int posCentr = k * conf -> nFeatures;

				for (int f = 0; f < conf -> nFeatures; ++f) {
					centroids[posCentr + f] = dataBase[posDataBase + f];
				}
			}

			// Initialize the mapping table
			for (int i = 0; i < conf -> nInstances; ++i) {
				mapping[i] = 0;
			}


			/******************** Convergence process *********************/

			// To avoid poor performance, "conf -> maxIterKmeans" iterations are executed
			for (int maxIter = 0; maxIter < conf -> maxIterKmeans; ++maxIter) {

				for (int k = 0; k < conf -> nCentroids; ++k) {
					samples_in_k[k] = 0;
				}

				// Calculate all distances (Euclidean distance) between each instance and the centroids
				for (int i = 0; i < conf -> nInstances; ++i) {
					float minDist = INFINITY;
					int selectCentroid;
					int pos = conf -> nFeatures * i;
					for (int k = 0, posCentr = 0; k < conf -> nCentroids; ++k, posCentr += conf -> nFeatures) {
						float dist = 0.0f;
						for (int f = 0; f < conf -> nFeatures; ++f) {
							if (subpop[ind].chromosome[f]) {
								float dif = dataBase[pos + f] - centroids[posCentr + f];
								dist += dif * dif;
							}
						}

						if (dist < minDist) {
							minDist = dist;
							selectCentroid = k;
						}
					}

					distCentroids[i] = minDist;
					samples_in_k[selectCentroid]++;
					if (mapping[i] != selectCentroid) {
						mapping[i] = selectCentroid;
					}
				}

				// Update the position of the centroids
				for (int k = 0; k < conf -> nCentroids; ++k) {
					int posCentr = k * conf -> nFeatures;
					for (int f = 0; f < conf -> nFeatures; ++f) {
						float sum = 0.0f;
						if (subpop[ind].chromosome[f]) {
							for (int i = 0; i < conf -> nInstances; ++i) {
								if (mapping[i] == k) {
									sum += dataBase[(conf -> nFeatures * i) + f];
								}
							}
							centroids[posCentr + f] = sum / samples_in_k[k];
						}
					}
				}
			}


			/************ Minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS) *************/

			float sumWithin = 0.0f;
			float sumInter = 0.0f;

			// Within-cluster
			for (int i = 0; i < conf -> nInstances; ++i) {
				sumWithin += sqrt(distCentroids[i]);
			}

			// Inter-cluster
			for (int posCentr = 0; posCentr < totalCoord; posCentr += conf -> nFeatures) {
				for (int i = posCentr + conf -> nFeatures; i < totalCoord; i += conf -> nFeatures) {
					float sum = 0.0f;
					for (int f = 0; f < conf -> nFeatures; ++f) {
						if (subpop[ind].chromosome[f]) {
							sum += (centroids[posCentr + f] - centroids[i + f]) * (centroids[posCentr + f] - centroids[i + f]);
						}
					}
					sumInter += sqrt(sum);
				}
			}

			// First objective function (Within-cluster sum of squares (WCSS))
			subpop[ind].fitness[0] = sumWithin;

			// Second objective function (Inter-cluster sum of squares (ICSS))
			subpop[ind].fitness[1] = sumInter;
		}
	}
}


/**
 * @brief Evaluation of each individual on OpenCL devices
 * @param subpop The first individual to evaluate of the current subpopulation
 * @param nIndividuals The number of individuals which will be evaluated
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param nDevices The number of devices that will execute the evaluation
 * @param dataBase The database which will contain the instances and the features
 * @param selInstances The instances choosen as initial centroids
 * @param conf The structure with all configuration parameters
 */
void evaluationCL_MP(Individual *const subpop, const int nIndividuals, CLDevice *const devicesObject, const int nDevices, const float *const dataBase, const int *const selInstances, const Config *const conf) {


	/************ K-means algorithm in OpenCL ***********/

	int index = 0;

	#pragma omp parallel num_threads(nDevices)
	{
		int begin, end, maxProcessing;
		bool finished = false;
		int threadID = omp_get_thread_num();
		cl_int status;
		cl_event kernelEvent, copyEvent;

		// Start the copy onto the devices
		if (devicesObject[threadID].deviceType != CL_DEVICE_TYPE_CPU) {
			check(clEnqueueWriteBuffer(devicesObject[threadID].commandQueue, devicesObject[threadID].objSubpopulations, CL_FALSE, 0, nIndividuals * sizeof(Individual), subpop, 0, NULL, &copyEvent) != CL_SUCCESS, EV_ERROR_ENQUEUE_INDIVIDUALS);
		}

		// Only 1 device (CPU or GPU)
		if (nDevices == 1) {
			int maxIndividualsOnGpuKernel = 10000; // This param was configurable in previous versions
			maxProcessing = (devicesObject[threadID].deviceType == CL_DEVICE_TYPE_GPU) ? std::min(nIndividuals, maxIndividualsOnGpuKernel) : nIndividuals;
		}

		// Heterogeneous mode
		else {
			maxProcessing = devicesObject[threadID].computeUnits;
		}

		do {
			#pragma omp atomic capture
			{
				begin = index;
				index += maxProcessing;
			}

			if (begin < nIndividuals) {
				end = (begin + maxProcessing >= nIndividuals) ? nIndividuals : begin + maxProcessing;

				if (devicesObject[threadID].deviceType != CL_DEVICE_TYPE_CPU) {

					// Sets new kernel arguments
					check(clSetKernelArg(devicesObject[threadID].kernel, 3, sizeof(int), &begin) != CL_SUCCESS, EV_ERROR_KERNEL_ARGUMENT4);
					check(clSetKernelArg(devicesObject[threadID].kernel, 4, sizeof(int), &end) != CL_SUCCESS, EV_ERROR_KERNEL_ARGUMENT5);

					// Enqueue and execute the kernel
					check((status = clEnqueueNDRangeKernel(devicesObject[threadID].commandQueue, devicesObject[threadID].kernel, 1, NULL, &(devicesObject[threadID].wiGlobal), &(devicesObject[threadID].wiLocal), 1, &copyEvent, &kernelEvent)) != CL_SUCCESS, EV_ERROR_ENQUEUE_KERNEL);

					// Read the data from the devices
					check((status = clEnqueueReadBuffer(devicesObject[threadID].commandQueue, devicesObject[threadID].objSubpopulations, CL_TRUE, begin * sizeof(Individual), (end - begin) * sizeof(Individual), subpop + begin, 1, &kernelEvent, NULL)) != CL_SUCCESS, EV_ERROR_ENQUEUE_READING);
				}
				else {
					evaluationCPU(subpop + begin, end - begin, dataBase, selInstances, devicesObject[threadID].wiGlobal, conf);
				}
			}
			else {
				finished = true;
			}
		} while (!finished);
	}
}


/**
 * @brief Evaluation of each individual on OpenCL devices
 * @param subpop The first individual to evaluate of the current subpopulation
 * @param nIndividuals The number of individuals which will be evaluated
 * @param devicesObject Structure containing the OpenCL variables of a device
 * @param nDevices The number of devices that will execute the evaluation
 * @param conf The structure with all configuration parameters
 */
void evaluationCL(Individual *const subpop, const int nIndividuals, CLDevice *const devicesObject, const int nDevices, const Config *const conf) {


	/************ K-means algorithm in OpenCL ***********/

	int index = 0;

	#pragma omp parallel num_threads(nDevices)
	{
		int begin, end, maxProcessing;
		bool finished = false;
		int threadID = omp_get_thread_num();
		cl_int status;
		cl_event kernelEvent, copyEvent;

		// Start the copy onto the devices
		check(clEnqueueWriteBuffer(devicesObject[threadID].commandQueue, devicesObject[threadID].objSubpopulations, CL_FALSE, 0, nIndividuals * sizeof(Individual), subpop, 0, NULL, &copyEvent) != CL_SUCCESS, EV_ERROR_ENQUEUE_INDIVIDUALS);

		// Only 1 device (CPU or GPU)
		if (nDevices == 1) {
			int maxIndividualsOnGpuKernel = 10000; // This param was configurable in previous versions
			maxProcessing = (devicesObject[threadID].deviceType == CL_DEVICE_TYPE_GPU) ? std::min(nIndividuals, maxIndividualsOnGpuKernel) : nIndividuals;
		}

		// Heterogeneous mode
		else {
			maxProcessing = devicesObject[threadID].computeUnits;
		}

		do {
			#pragma omp atomic capture
			{
				begin = index;
				index += maxProcessing;
			}

			if (begin < nIndividuals) {
				end = (begin + maxProcessing >= nIndividuals) ? nIndividuals : begin + maxProcessing;

				// Sets new kernel arguments
				check(clSetKernelArg(devicesObject[threadID].kernel, 3, sizeof(int), &begin) != CL_SUCCESS, EV_ERROR_KERNEL_ARGUMENT4);
				check(clSetKernelArg(devicesObject[threadID].kernel, 4, sizeof(int), &end) != CL_SUCCESS, EV_ERROR_KERNEL_ARGUMENT5);

				// Enqueue and execute the kernel
				check((status = clEnqueueNDRangeKernel(devicesObject[threadID].commandQueue, devicesObject[threadID].kernel, 1, NULL, &(devicesObject[threadID].wiGlobal), &(devicesObject[threadID].wiLocal), 1, &copyEvent, &kernelEvent)) != CL_SUCCESS, EV_ERROR_ENQUEUE_KERNEL);

				// Read the data from the devices
				check((status = clEnqueueReadBuffer(devicesObject[threadID].commandQueue, devicesObject[threadID].objSubpopulations, CL_TRUE, begin * sizeof(Individual), (end - begin) * sizeof(Individual), subpop + begin, 1, &kernelEvent, NULL)) != CL_SUCCESS, EV_ERROR_ENQUEUE_READING);
			}
			else {
				finished = true;
			}
		} while (!finished);
	}
}


/**
 * @brief Normalize the fitness for each individual
 * @param subpop The first individual to normalize of the current subpopulation
 * @param nIndividuals The number of individuals which will be normalized
 * @param conf The structure with all configuration parameters
 */
void normalizeFitness(Individual *const subpop, const int nIndividuals, const Config *const conf) {

	for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {

		// Fitness vector average
		float average = 0;
		for (int i = 0; i < nIndividuals; ++i) {
			average += subpop[i].fitness[obj];
		}

		average /= nIndividuals;

		// Fitness vector variance
		float variance = 0;
		for (int i = 0; i < nIndividuals; ++i) {
			variance += (subpop[i].fitness[obj] - average) * (subpop[i].fitness[obj] - average);
		}
		variance /= (nIndividuals - 1);

		// Fitness vector standard deviation
		float std_deviation = sqrt(variance);

		// The second objective is a maximization problem. x_new must be negative
		if (obj == 1) {

			// Normalize a set of continuous values using SoftMax (based on the logistic function)
			for (int i = 0; i < nIndividuals; ++i) {
				float x_scaled = (subpop[i].fitness[obj] - average) / std_deviation;
				float x_new = 1.0f / (1.0f + exp(-x_scaled));
				subpop[i].fitness[obj] = -x_new;
			}
		}
		else {

			// Normalize a set of continuous values using SoftMax (based on the logistic function)
			for (int i = 0; i < nIndividuals; ++i) {
				float x_scaled = (subpop[i].fitness[obj] - average) / std_deviation;
				subpop[i].fitness[obj] = 1.0f / (1.0f + exp(-x_scaled));
			}
		}
	}
}


/**
 * @brief Gets the hypervolume measure of the subpopulation
 * @param subpop Current subpopulation
 * @param nIndFront0 The number of individuals in the front 0
 * @param conf The structure with all configuration parameters
 * @return The value of the hypervolume
 */
float getHypervolume(const Individual *const subpop, const int nIndFront0, const Config *const conf) {

	// Generation the points for the calculation of the hypervolume
	double **points = new double*[nIndFront0];
	for (int i = 0; i < nIndFront0; ++i) {
		points[i] = new double[conf -> nObjectives];
		for (unsigned char obj = 0; obj < conf -> nObjectives; ++obj) {
			points[i][obj] = (obj == 0) ? 1 - subpop[i].fitness[obj] : -subpop[i].fitness[obj];
		}
	}

	// The reference point is the origin point
	float hypervolume = fabs(GetHypervolume(points, nIndFront0, conf -> nObjectives));
	for (int i = 0; i < nIndFront0; ++i) {
		delete[] points[i];
	}
	delete[] points;

	return hypervolume;
}


/**
 * @brief Gets the initial centroids (instances choosen randomly)
 * @param conf The structure with all configuration parameters
 * @return The instances choosen as initial centroids will be stored
 */
int* getCentroids(const Config *const conf) {

	// The init centroids will be instances choosen randomly (Forgy's Method)
	int *selInstances = new int[conf -> nCentroids];
	for (int k = 0; k < conf -> nCentroids; ++k) {
		bool exists = false;
		int randomInstance;

		// Avoid repeat centroids
		do {
			randomInstance = rand() % conf -> nInstances;
			exists = false;

			// Look if the generated index already exists
			for (int kk = 0; kk < k && !exists; ++kk) {
				exists = (randomInstance == selInstances[kk]);
			}
		} while (exists);

		selInstances[k] = randomInstance;
	}

	return selInstances;
}


/**
 * @brief Generates the gnuplot data
 * @param subpop Current subpopulation
 * @param nIndFront0 The number of individuals in the front 0
 * @param conf The structure with all configuration parameters
 */
void generateDataPlot(const Individual *const subpop, const int nIndFront0 , const Config *const conf) {

	// Open the data file
	FILE *f_data = fopen(conf -> dataFileName.c_str(), "w");
	check(!f_data, EV_ERROR_DATA_OPEN);

	// Write the data
	fprintf(f_data, "#Objective0");
	for (unsigned char obj = 1; obj < conf -> nObjectives; ++obj) {
		fprintf(f_data, "\tObjective%d", obj);
	}
	for (int i = 0; i < nIndFront0; ++i) {
		fprintf(f_data, "\n%f", subpop[i].fitness[0]);
		for (unsigned char obj = 1; obj < conf -> nObjectives; ++obj) {
			fprintf(f_data, "\t%f", subpop[i].fitness[obj]);
		}
	}

	fclose(f_data);
}


/**
 * @brief Generates gnuplot code for data display
 * @param conf The structure with all configuration parameters
 */
void generateGnuplot(const Config *const conf) {

	// Gnuplot is only available for two objectives
	check(conf -> nObjectives != 2, EV_WARNING_OBJECTIVES_NUMBER, false);

	// Open the gnuplot script file
	FILE *f_plot = fopen(conf -> plotFileName.c_str(), "w");
	check(!f_plot, EV_ERROR_PLOT_OPEN);

	// Write the code
	fprintf(f_plot, "#!/usr/bin/gnuplot\n");
	fprintf(f_plot, "set terminal png size 1024,600\n");
	fprintf(f_plot, "set output '%s.png'\n", conf -> imageFileName.c_str());
	fprintf(f_plot, "set multiplot\n");
	fprintf(f_plot, "set xlabel \"Objective 0\"\n");
	fprintf(f_plot, "set grid\n");
	fprintf(f_plot, "set title \"Pareto front 0\"\n");
	fprintf(f_plot, "set ylabel \"Objective 1\"\n");
	fprintf(f_plot, "set size 0.9,0.9\n");
	fprintf(f_plot, "set origin 0.00,0.05\n");
	fprintf(f_plot, "set key center top\n");
	fprintf(f_plot, "\tplot [0:1][-1:0] \"<echo '1.0 0.0'\" title \"Reference point\" with points pt 3 ps 2,\\\n");
	fprintf(f_plot, "\t'< sort %s' using 1:%d title \"MPI Mode\" with lp;\n", conf -> dataFileName.c_str(), conf -> nObjectives);
	fprintf(f_plot, "set nomultiplot\n");
	fprintf(f_plot, "reset\n");
	fclose(f_plot);
}