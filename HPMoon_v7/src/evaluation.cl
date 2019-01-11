/**
 * @file evaluation.cl
 * @author Juan José Escobar Pérez
 * @date 12/07/2015
 * @brief File with the necessary implementation for the K-means algorithm in OpenCL
 */

/*********************************** Defines *********************************/

/**
 * @brief Structure containing the Individual's parameters
 */
typedef struct Individual {


	/**
	 * @brief Vector denoting the selected features
	 *
	 * Values: Zeros or ones
	 */
	unsigned char chromosome[N_FEATURES];


	/**
	 * @brief Individual fitness for the multi-objective functions
	 *
	 * Values: Each position contains an objective function
	 */
	float fitness[N_OBJECTIVES];


	/**
	 * @brief Crowding distance of the individual
	 *
	 * The values are positives or infinites
	 */
	float crowding;


	/**
	 * @brief Range of the individual (Pareto front)
	 */
	int rank;


	/**
	 * @brief Number of selected features
	 */
	int nSelFeatures;

} Individual;

/********************************* OpenCL Kernels ********************************/


/**
 * @brief Computes the K-means algorithm in a OpenCL GPU device
 * @param subpop OpenCL object which contains the current subpopulation. The object is stored in global memory
 * @param selInstances OpenCL object which contains the instances choosen as initial centroids. The object is stored in constant memory
 * @param trDataBase OpenCL object which contains the training database. The object is stored in global memory
 * @param begin The first individual to be evaluated
 * @param end The "end-1" position is the last individual to be evaluated
 * @param transposedDataBase OpenCL object which contains the transposed training database. The object is stored in global memory
 */
__kernel void kmeansGPU(__global struct Individual *subpop, __constant int *restrict selInstances, __global float *restrict trDataBase, const int begin, const int end, __global float *restrict transposedDataBase) {

	uint localId = get_local_id(0);
	uint localSize = get_local_size(0);
	uint groupId = get_group_id(0);
	uint numGroups = get_num_groups(0);

	const int totalCoord = K * N_FEATURES;

	// The individual is cached into local memory to improve performance
	__local uchar chromosome[N_FEATURES];
	__local uchar mapping[N_INSTANCES];
	__local float centroids_l[K * N_FEATURES];
	__local float distCentroids[N_INSTANCES];
	__local int samples_in_k[K];

	event_t eventInd;
	event_t eventCentr;


	// Each work-group compute an individual (master-slave as a deck algorithm)
	for (int ind = begin + groupId; ind < end; ind += numGroups) {

		// The centroids will have the selected features of the individual
		for (int k = 0; k < K; ++k) {
			async_work_group_copy(centroids_l + (N_FEATURES * k), trDataBase + (selInstances[k] * N_FEATURES), N_FEATURES, eventCentr);
		}

		// The individual is cached to local memory for improve performance
		eventInd = async_work_group_copy(chromosome, subpop[ind].chromosome, N_FEATURES, 0);

		// Initialize the mapping table
		for (int i = localId; i < N_INSTANCES; i += localSize) {
			mapping[i] = 0;
		}

		// Syncpoint
		wait_group_events(1, &eventInd);


		/******************** Convergence process *********************/

		// To avoid poor performance, "MAX_ITER_KMEANS" iterations are executed
		for (int maxIter = 0; maxIter < MAX_ITER_KMEANS/* && !converged*/; ++maxIter) {

			barrier(CLK_LOCAL_MEM_FENCE);
			for (int k = localId; k < K; k += localSize) {
				samples_in_k[k] = 0;
			}

			// Syncpoint
			barrier(CLK_LOCAL_MEM_FENCE);

			// Calculate all distances (Euclidean distance) between each instance and the centroids
			for (int i = localId; i < N_INSTANCES; i += localSize) {
				float minDist = INFINITY;
				int selectCentroid;
				for (int k = 0, posCentr = 0; k < K; ++k, posCentr += N_FEATURES) {
					float dist = 0.0f;
					for (int f = 0; f < N_FEATURES; ++f) {
						if (chromosome[f]) {
							float dif = transposedDataBase[(N_INSTANCES * f) + i] - centroids_l[posCentr + f];
							dist = mad(dif, dif, dist);
						}
					}

					if (dist < minDist) {
						minDist = dist;
						selectCentroid = k;
					}
				}

				distCentroids[i] = minDist;
				atomic_inc(&samples_in_k[selectCentroid]);

				if (mapping[i] != selectCentroid) {
					mapping[i] = selectCentroid;
				}
			}

			// Syncpoint
			barrier(CLK_LOCAL_MEM_FENCE);

			// Update the position of the centroids
			for (int kf = localId; kf < totalCoord; kf += localSize) {
				int k = kf / N_FEATURES;
				int f = kf - (k * N_FEATURES); // kf % N_FEATURES
				if (chromosome[f]) {
					float sum = 0.0f;
					for (int i = 0; i < N_INSTANCES; ++i) {
						if (mapping[i] == k) {
							sum += trDataBase[(N_FEATURES * i) + f];
						}
					}
					centroids_l[kf] = sum / samples_in_k[k];
				}
			}

			// Syncpoint
			barrier(CLK_LOCAL_MEM_FENCE);
		}


		/************ Minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS) *************/

		if (localId == 0) {
			float sumWithin = 0.0f;
			float sumInter = 0.0f;

			// Within-cluster
			for (int i = 0; i < N_INSTANCES; ++i) {
				sumWithin += sqrt(distCentroids[i]);
			}

			// Inter-cluster
			for (int posCentr = 0; posCentr < totalCoord; posCentr += N_FEATURES) {
				for (int i = posCentr + N_FEATURES; i < totalCoord; i += N_FEATURES) {
					float sum = 0.0f;
					for (int f = 0; f < N_FEATURES; ++f) {
						if (chromosome[f]) {
							sum += (centroids_l[posCentr + f] - centroids_l[i + f]) * (centroids_l[posCentr + f] - centroids_l[i + f]);
						}
					}
					sumInter += sqrt(sum);
				}
			}

			// First objective function (Within-cluster sum of squares (WCSS))
			subpop[ind].fitness[0] = sumWithin;//printf("IND %d\n", ind);printf("%f\n", sumWithin);

			// Second objective function (Inter-cluster sum of squares (ICSS))
			subpop[ind].fitness[1] = sumInter;//printf("%f\n", sumInter);
		}

		// Syncpoint
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
