//
// Created by dxt on 18-12-14.
//

#ifndef SOLARENERGYRAYTRACING_RECEIVERINTERSECTUTIL_H
#define SOLARENERGYRAYTRACING_RECEIVERINTERSECTUTIL_H

#include <SolarScene.h>
#include "global_function.cuh"

void changeSunLightsAndPerturbationToParallel(Sunray *sunray);

std::vector<float> deviceArray2vector(float *d_array, int size);

#endif //SOLARENERGYRAYTRACING_RECEIVERINTERSECTUTIL_H
