/**
 * @file zitzler.h
 * @author Juan José Escobar Pérez
 * @date 26/02/2018
 * @brief Header file for the hypervolume metric calculation
 */

#ifndef ZITZLER_H
#define ZITZLER_H

/********************************* Methods ********************************/

/**
 * @brief Gets the hypervolume measure for the specified set of points
 * @param points The set of points
 * @param nPoints The number of points
 * @param nObjectives The number of objectives
 * @return The hypervolume value
 */
double GetHypervolume(double **points, const int nPoints, const int nObjectives);

#endif