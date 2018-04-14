/**
 * @file bd.h
 * @author Juan José Escobar Pérez
 * @date 15/06/2015
 * @brief Header file for data base reading
 */

#ifndef BD_H
#define BD_H

/********************************* Includes *******************************/

#include "config.h" // "Config" datatype

/******************************** Constants *******************************/

const char *const BD_ERROR_FILE_OPEN = "Error: Could not open the database file\n";
const char *const BD_ERROR_FILE_EMPTY = "Error: The database file is empty";
const char *const BD_ERROR_ROW_UNEQUAL = "Error: Different number of columns in the row";
const char *const BD_ERROR_DIMENSIONS_MIN = "Error: The database dimensions must be 4x4 or higher\n";
const char *const BD_ERROR_INSTANCES_RANGE = "Error: The number of instances must be between 4 and";

/********************************* Methods ********************************/

/**
 * @brief Reads and normalizes the database
 * @param conf The structure with all configuration parameters
 * @return The database which will contain the instances
 */
float* getDataBase(const Config *const conf);


/**
 * @brief The database is normalized between 0 and 1
 * @param dataBase Database to be normalized
 * @param conf The structure with all configuration parameters
 */
void normDataBase(float *const dataBase, const Config *const conf);


/**
 * @brief The database is transposed
 * @param dataBase Database to be transposed
 * @param conf The structure with all configuration parameters
 * @return The database already transposed
 */
float* transposeDataBase(const float *const dataBase, const Config *const conf);

#endif