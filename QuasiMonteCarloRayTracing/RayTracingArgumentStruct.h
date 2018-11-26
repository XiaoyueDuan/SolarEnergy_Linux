//
// Created by dxt on 18-11-16.
//

#ifndef SOLARENERGYRAYTRACING_RAYTRACINGARGUMENTSTRUCT_H
#define SOLARENERGYRAYTRACING_RAYTRACINGARGUMENTSTRUCT_H

#include "cuda_runtime.h"
#include "SolarScene.h"

struct SunrayArgument {
    float3 *d_samplelights;
    float3 *d_perturbations;
    int pool_size;
    int numberOfLightsPerGroup;
    float3 sunray_direction;

    SunrayArgument() : d_samplelights(nullptr), d_perturbations(nullptr) {}

    SunrayArgument(float3 *sample, float3 *perturbation, int pool_size_, int lightsPerGroup, float3 dir) :
            d_samplelights(sample), d_perturbations(perturbation), pool_size(pool_size_),
            numberOfLightsPerGroup(lightsPerGroup), sunray_direction(dir) {}

    ~SunrayArgument() {
        d_perturbations = nullptr;
        d_samplelights = nullptr;
    }
};

struct HeliostatArgument {
    float3 *d_microHelio_origins;
    float3 *d_microHelio_normals;
    int *d_microHelio_groups;
    int numberOfMicroHeliostats;
    int subHeliostat_id;
    int numberOfSubHeliostats;

    HeliostatArgument() : d_microHelio_origins(nullptr), d_microHelio_normals(nullptr), d_microHelio_groups(nullptr),
                          numberOfMicroHeliostats(0), subHeliostat_id(0), numberOfSubHeliostats(0) {}

    HeliostatArgument
            (float3 *origins, float3 *normals, int *groups, int microHelioSize, int subhelio_id, int subHelioSize) :
            d_microHelio_origins(origins), d_microHelio_normals(normals),
            d_microHelio_groups(groups), numberOfMicroHeliostats(microHelioSize), subHeliostat_id(subhelio_id),
            numberOfSubHeliostats(subHelioSize) {}

    ~HeliostatArgument() {
        d_microHelio_origins = nullptr;
        d_microHelio_normals = nullptr;
        d_microHelio_groups = nullptr;
    }

    void CClear() {
        cudaFree(d_microHelio_origins);
        cudaFree(d_microHelio_normals);
        cudaFree(d_microHelio_groups);

        d_microHelio_origins = nullptr;
        d_microHelio_normals = nullptr;
        d_microHelio_groups = nullptr;
    }
};

#endif //SOLARENERGYRAYTRACING_RAYTRACINGARGUMENTSTRUCT_H
