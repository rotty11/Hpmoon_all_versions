/**
 * @file bd.h
 * @author Juan José Escobar Pérez
 * @date 15/06/2015
 * @brief Header file for data base reading
 *
 */

#ifndef BD_H
#define BD_H

/********************************* Includes *******************************/

#include "config.h" // "Config" datatype

/********************************* Methods ********************************/

/**
 * @brief Reading the database
 * @param dataBase The database which will contain the instances
 * @param conf The structure with all configuration parameters
 */
void readDataBase(float *dataBase, const Config *conf);


/**
 * @brief The database is normalized between 0 and 1
 * @param dataBase Database to be normalized
 * @param conf The structure with all configuration parameters
 */
void normDataBase(float *dataBase, const Config *conf);


/**
 * @brief The database is transposed
 * @param dataBase Database to be transposed
 * @param dataBaseTransposed Database already transposed
 * @param conf The structure with all configuration parameters
 */
void transposeDataBase(float *dataBase, float *dataBaseTransposed, const Config *conf);

#endif