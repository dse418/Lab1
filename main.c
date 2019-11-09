#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// David Shane Elliott
// Monte Carlo estimation of pi
// Based on: https://en.wikipedia.org/wiki/Monte_Carlo_method &
// https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/

// Returns a random value between -1 and 1
double getRand(unsigned int *seed) {
    return (double) rand_r(seed) * 2 / (double) (RAND_MAX) - 1;
}

// Calculates an estimation of pi on a single thread
long double Calculate_Pi_Sequential(long long number_of_tosses) {

    // Declare variables
    int pointsInCircle = 0;
    int pointsInSquare = 0;
    unsigned int seed = (unsigned int) time(NULL);
    double distanceToOrigin, estimatedPi, xCoordinate, yCoordinate;

    // Iterate through each toss
    for(int i = 0; i < number_of_tosses; i++) {
        // Get random x & y coordinates
        xCoordinate = getRand(&seed);
        yCoordinate = getRand(&seed);

        // Calculate distance to origin to determine if circle or both circle & square point
        distanceToOrigin = pow(xCoordinate, 2.0) + pow(yCoordinate, 2.0);

        // Increment circle points if within circle, else increment square points
        if(distanceToOrigin <= 1.0) {
            pointsInCircle++;
        }
        else{
            pointsInSquare++;
        }
    }

    // Calculate estimated pi
    estimatedPi = 4 * (double) pointsInCircle / ((double) pointsInCircle + (double) pointsInSquare);

    return estimatedPi;
}

// Calculates an estimation of pi on parallel threads
long double Calculate_Pi_Parallel(long long number_of_tosses) {

    // Declare variables
    int globalPointsInCircle = 0;
    int globalPointsInSquare = 0;
    unsigned int seed = (unsigned int) time(NULL) + (unsigned int) omp_get_thread_num();
    double distanceToOrigin, estimatedPi, xCoordinate, yCoordinate;

    // OpenMP parallel for pragma with specified reduction and private variables
#pragma omp parallel for num_threads(omp_get_max_threads()) \
    reduction(+: globalPointsInCircle, globalPointsInSquare) private(distanceToOrigin, xCoordinate, yCoordinate)
    for(int i = 0; i < number_of_tosses; i++) {
        // Get random x & y coordinates
        xCoordinate = getRand(&seed);
        yCoordinate = getRand(&seed);

        // Calculate distance to origin to determine if circle or both circle & square point
        distanceToOrigin = pow(xCoordinate, 2.0) + pow(yCoordinate, 2.0);

        // Increment global circle points if within circle, else increment global square points
        if(distanceToOrigin <= 1.0) {
            globalPointsInCircle++;
        }
        else{
            globalPointsInSquare++;
        }
    }

    // Calculate estimated pi
    estimatedPi = 4 * (double) globalPointsInCircle / ((double) globalPointsInCircle + (double) globalPointsInSquare);

    return estimatedPi;
}

int main() {
    struct timeval start, end;

    long long num_tosses = 10000000;

    printf("Timing sequential...\n");
    gettimeofday(&start, NULL);
    long double sequential_pi = Calculate_Pi_Sequential(num_tosses);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    printf("Timing parallel...\n");
    gettimeofday(&start, NULL);
    long double parallel_pi = Calculate_Pi_Parallel(num_tosses);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    // This will print the result to 10 decimal places
    printf("π = %.10Lf (sequential)\n", sequential_pi);
    printf("π = %.10Lf (parallel)", parallel_pi);

    return 0;
}