/**
 * @file bd.cpp
 * @author Juan José Escobar Pérez
 * @date 15/06/2015
 * @brief File with the necessary implementation to read and store the database
 *
 */

/********************************* Includes *******************************/

#include "bd.h"
#include <stdlib.h> // exit...
#include <math.h> // exp, sqrt...
#include <sstream> // stringstream

/********************************* Methods ********************************/

/**
 * @brief Reads and normalizes the database
 * @param conf The structure with all configuration parameters
 * @return The database which will contain the instances
 */
float* getDataBase(const Config *const conf) {


	/********** Open the database ***********/

	fstream fData(conf -> dataBaseFileName.c_str(), fstream::in);
	if (!fData.is_open()) {
		fprintf(stderr, "Error: Could not open the database file\n");
		exit(-1);
	}


	/********** Getting the database dimensions ***********/

	int nRows = 1, nCols = 0;
	float dato;
	string line;
	if(!getline(fData, line)) {
		fprintf(stderr, "Error: The database file is empty\n");
		fData.close();
		exit(-1);
	}

	stringstream ss(line);
	while(ss >> dato) {
		++nCols;
	}

	while(getline(fData, line)) {
		++nRows;
		stringstream aux(line);
		int tmp = 0;
		while(aux >> dato) {
			++tmp;
		}
		if (tmp != nCols) {
			fprintf(stderr, "Error: Different number of columns in the row %d\n", nRows);
			exit(-1);
		}
	}

	/********** Check the parameters specified in configuration ***********/

	if (nRows < 4 || nCols < 4) {
		fprintf(stderr, "Error: The database dimensions must be 4x4 or higher\n");
		exit(-1);
	}
	if (conf -> nInstances < 4 || conf -> nInstances > nRows) {
		fprintf(stderr, "Error: The number of instances must be between 4 and %d\n", nRows);
		exit(-1);
	}


	/********** Reading and database storage ***********/

	fData.clear();
	fData.seekg(0);
	int dbSize = conf -> nInstances * nCols;
	float *dataBase = new float[dbSize];
	for (int i = 0; i < dbSize; ++i) {
		fData >> dataBase[i];
	}

	// Normalize, close the database and return it
	normDataBase(dataBase, conf);
	fData.close();
	return dataBase;
}


/**
 * @brief The database is normalized between 0 and 1
 * @param dataBase Database to be normalized
 * @param conf The structure with all configuration parameters
 */
void normDataBase(float *const dataBase, const Config *const conf) {


	/********** Database normalization ***********/

	for(int j = 0; j < conf -> nFeatures; ++j) {

		// Average of the features vector
		float average = 0;
		for(int i = 0; i < conf -> nInstances; ++i) {
			int pos = (conf -> nFeatures * i) + j;
			average += dataBase[pos];
		}

		average /= conf -> nInstances;

		// Variance of the features vector
		float variance = 0;
		for(int i = 0; i < conf -> nInstances; ++i) {
			int pos = (conf -> nFeatures * i) + j;
			variance += (dataBase[pos] - average) * (dataBase[pos] - average);
		}
		variance /= (conf -> nInstances - 1);

		// Standard deviation of the features vector
		float std_deviation = sqrt(variance);

		// Normalize a set of continuous values using SoftMax (based on the logistic function)
		for(int i = 0; i < conf -> nInstances; ++i) {
			int pos = (conf -> nFeatures * i) + j;
			float x_scaled = (dataBase[pos] - average) / std_deviation;
			dataBase[pos] = 1.0f / (1.0f + exp(-x_scaled));
		}
	}
}


/**
 * @brief The database is transposed
 * @param dataBase Database to be transposed
 * @param conf The structure with all configuration parameters
 * @return The database already transposed
 */
float* transposeDataBase(const float *const dataBase, const Config *const conf) {


	/********** Transpose database ***********/

	float *dataBaseTransposed = new float[conf -> nInstances * conf -> nFeatures];
	for(int f = 0, size = 0; f < conf -> nFeatures; ++f) {
		for (int i = 0; i < conf -> nInstances; ++i, ++size) {
			dataBaseTransposed[size] = dataBase[(conf -> nFeatures * i) + f];
		}
	}

	return dataBaseTransposed;
}