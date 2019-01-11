/**
 * @file bd.cpp
 * @author Juan José Escobar Pérez
 * @date 15/06/2015
 * @brief File with the necessary implementation to read and store a database
 */

/********************************* Includes *******************************/

#include "bd.h"
#include <cmath> // exp, sqrt...
#include <sstream> // stringstream

/********************************* Methods ********************************/

/**
 * @brief The database is normalized between 0.0 and 1.0
 * @param dataBase Database to be normalized
 * @param conf The structure with all configuration parameters
 */
void normDataBase(float *const dataBase, const Config *const conf) {


	/********** Database normalization ***********/

	for(int j = 0; j < conf -> nFeatures; ++j) {

		// Average of the features vector
		float average = 0;
		for(int i = 0; i < conf -> trNInstances; ++i) {
			int pos = (conf -> nFeatures * i) + j;
			average += dataBase[pos];
		}

		average /= conf -> trNInstances;

		// Variance of the features vector
		float variance = 0;
		for(int i = 0; i < conf -> trNInstances; ++i) {
			int pos = (conf -> nFeatures * i) + j;
			variance += (dataBase[pos] - average) * (dataBase[pos] - average);
		}
		variance /= (conf -> trNInstances - 1);

		// Standard deviation of the features vector
		float std_deviation = sqrt(variance);

		// Normalize a set of continuous values using SoftMax (based on the logistic function)
		for(int i = 0; i < conf -> trNInstances; ++i) {
			int pos = (conf -> nFeatures * i) + j;
			float x_scaled = (dataBase[pos] - average) / std_deviation;
			dataBase[pos] = 1.0f / (1.0f + exp(-x_scaled));
		}
	}
}


/**
 * @brief Reads and normalizes a database if it is required
 * @param conf The structure with all configuration parameters
 * @return The database which will contain the instances
 */
float* getDataBase(const Config *const conf) {


	/********** Open the database ***********/

	std::fstream fData(conf -> trDataBaseFileName.c_str(), std::fstream::in);
	check(!fData.is_open(), "%s\n", BD_ERROR_FILE_OPEN);


	/********** Getting the database dimensions ***********/

	int nRows = 1, nCols = 0;
	float dato;
	std::string line;
	if(!getline(fData, line)) {
		fData.close();
		check(true, "%s\n", BD_ERROR_FILE_EMPTY);
	}

	std::stringstream ss(line);
	while(ss >> dato) {
		++nCols;
	}

	while(getline(fData, line)) {
		++nRows;
		std::stringstream aux(line);
		int tmp = 0;
		while(aux >> dato) {
			++tmp;
		}
		check(tmp != nCols, "%s %d\n", BD_ERROR_ROW_UNEQUAL, nRows);
	}


	/********** Check the parameters specified in configuration ***********/

	check(nRows < 4 || nCols < 4, "%s\n", BD_ERROR_DIMENSIONS_MIN);
	check(conf -> trNInstances < 4 || conf -> trNInstances > nRows, "%s %d\n", BD_ERROR_INSTANCES_RANGE, nRows);
	check(conf -> nFeatures != nCols, "%s\n", BD_ERROR_COLUMNS_UNEQUAL);


	/********** Reading and database storage ***********/

	fData.clear();
	fData.seekg(0);
	int dbSize = conf -> trNInstances * nCols;
	float *dataBase = new float[dbSize];
	for (int i = 0; i < dbSize; ++i) {
		fData >> dataBase[i];
	}
	fData.close();

	// Normalize the database if it is required and return it
	if (conf -> trNormalize) {
		normDataBase(dataBase, conf);
	}
	return dataBase;
}


/**
 * @brief The database is transposed
 * @param dataBase Database to be transposed
 * @param conf The structure with all configuration parameters
 * @return The database already transposed
 */
float* transposeDataBase(const float *const dataBase, const Config *const conf) {


	/********** Transpose database ***********/

	float *dataBaseTransposed = new float[conf -> trNInstances * conf -> nFeatures];
	for(int f = 0, size = 0; f < conf -> nFeatures; ++f) {
		for (int i = 0; i < conf -> trNInstances; ++i, ++size) {
			dataBaseTransposed[size] = dataBase[(conf -> nFeatures * i) + f];
		}
	}

	return dataBaseTransposed;
}
