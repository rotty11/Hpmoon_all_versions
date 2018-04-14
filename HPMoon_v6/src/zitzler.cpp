/**
 * @file zitzler.cpp
 * @author Juan José Escobar Pérez
 * @date 26/02/2018
 * @brief File with the necessary implementation for the hypervolume metric calculation.
 * This code is an adaptation of the original code, written in C, and owned by Eckart Zitzler.
 * This program calculates for a given set of objective vectors the volume
 * of the dominated space, enclosed by the nondominated points and the
 * origin. Here, a maximization problem is assumed, for minimization
 * or mixed optimization problem the objective vectors have to be transformed
 * accordingly. The hypervolume metric has been proposed in:\n\n
 * [1] E. Zitzler and L. Thiele. Multiobjective Optimization Using Evolutionary Algorithms:
 *     A Comparative Case Study. Parallel Problem
 *     Solving from Nature - PPSN-V, September 1998, pages 292-301.\n\n
 * A more detailed description and extensions can be found in:\n\n
 * [2] E. Zitzler. Evolutionary Algorithms for Multiobjective Optimization:
 *     Methods and Applications. Swiss Federal Institute of Technology (ETH)
 *     Zurich. Shaker Verlag, Germany, ISBN 3-8265-6831-1, December 1999.\n\n
 * Eckart Zitzler
 * Computer Engineering and Networks Laboratory (TIK)\n
 * Swiss Federal Institute of Technology (ETH) Zurich, Switzerland\n
 * (c)2001
 */

#include "zitzler.h"
#include <stdio.h>
#include <stdlib.h>

#define ERROR(x)  fprintf(stderr, x), fprintf(stderr, "\n"), exit(1)

int  Dominates(double  point1[], double  point2[], int  noObjectives) {

  int  i, betterInAnyObjective;

  betterInAnyObjective = 0;
  for (i = 0; i < noObjectives && point1[i] >= point2[i]; i++)
      if (point1[i] > point2[i])
	betterInAnyObjective = 1;
  return (i >= noObjectives && betterInAnyObjective);
} /* Dominates */

void  Swap(double  *front[], int  i, int  j) {

	double  *temp;

  temp = front[i];
  front[i] = front[j];
  front[j] = temp;
} /* Swap */

int  FilterNondominatedSet(double  *front[], int  noPoints, int  noObjectives)
     /* all nondominated points regarding the first 'noObjectives' dimensions
	are collected; the points referenced by 'front[0..noPoints-1]' are
	considered; 'front' is resorted, such that 'front[0..n-1]' contains
	the nondominated points; n is returned */
{
  int  i, j;
  int  n;

  n = noPoints;
  i = 0;
  while (i < n) {
    j = i + 1;
    while (j < n) {
      if (Dominates(front[i], front[j], noObjectives)) {
	/* remove point 'j' */
	n--;
	Swap(front, j, n);
      }
      else if (Dominates(front[j], front[i], noObjectives)) {
	/* remove point 'i'; ensure that the point copied to index 'i'
	   is considered in the next outer loop (thus, decrement i) */
	n--;
	Swap(front, i, n);
	i--;
	break;
      }
      else
	j++;
    }
    i++;
  }
  return n;
} /* FilterNondominatedSet */


double  SurfaceUnchangedTo(double  *front[], int  noPoints, int  objective)
     /* calculate next value regarding dimension 'objective'; consider
	points referenced in 'front[0..noPoints-1]' */
{
  int     i;
  double  minValue, value;

  if (noPoints < 1)  ERROR("run-time error");
  minValue = front[0][objective];
  for (i = 1; i < noPoints; i++) {
    value = front[i][objective];
    if (value < minValue)  minValue = value;
  }
  return minValue;
} /* SurfaceUnchangedTo */

int  ReduceNondominatedSet(double  *front[], int  noPoints, int  objective,
			   double  threshold)
     /* remove all points which have a value <= 'threshold' regarding the
	dimension 'objective'; the points referenced by
	'front[0..noPoints-1]' are considered; 'front' is resorted, such that
	'front[0..n-1]' contains the remaining points; 'n' is returned */
{
  int  n;
  int  i;

  n = noPoints;
  for (i = 0; i < n; i++)
    if (front[i][objective] <= threshold) {
      n--;
      Swap(front, i, n);
    }
  return n;
} /* ReduceNondominatedSet */

double  CalculateHypervolume(double  *front[], int  noPoints,
			     int  noObjectives)
{
  int     n;
  double  volume, distance;

  volume = 0;
  distance = 0;
  n = noPoints;
  while (n > 0) {
    int     noNondominatedPoints;
    double  tempVolume, tempDistance;

    noNondominatedPoints = FilterNondominatedSet(front, n, noObjectives - 1);
    tempVolume = 0;
    if (noObjectives < 3) {
      if (noNondominatedPoints < 1)  ERROR("run-time error");
      tempVolume = front[0][0];
    }
    else
      tempVolume = CalculateHypervolume(front, noNondominatedPoints, noObjectives - 1);
    tempDistance = SurfaceUnchangedTo(front, n, noObjectives - 1);
    volume += tempVolume * (tempDistance - distance);
    distance = tempDistance;
    n = ReduceNondominatedSet(front, n, noObjectives - 1, distance);
  }
  return volume;
} /* CalculateHypervolume */


/**
 * @brief Gets the hypervolume measure for the specified set of points
 * @param points The set of points
 * @param nPoints The number of points
 * @param nObjectives The number of objectives
 * @return The hypervolume value
 */
double GetHypervolume(double **points, const int nPoints, const int nObjectives) {

	int redSizeFront1 = FilterNondominatedSet(points, nPoints, nObjectives);
	return CalculateHypervolume(points, redSizeFront1, nObjectives);
}