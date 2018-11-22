//
// Created by dxt on 18-11-16.
//

#ifndef SOLARENERGYRAYTRACING_QUASIMONTECARLORAYTRACER_H
#define SOLARENERGYRAYTRACING_QUASIMONTECARLORAYTRACER_H

#include "SolarScene.h"
#include "RayTracingArgumentStruct.h"

class QuasiMonteCarloRayTracer {
public:
    void rayTracing(SolarScene *solarScene, int heliostat_id);

    /**
     *  Public following function just for test. DO NOT USE THEM STANDALONE.
     */
    void checkValidHeliostatIndex(SolarScene *solarScene, int heliostat_id);

    int receiverGridCombination(int receiver_type, int grid_type);

    HeliostatArgument generateHeliostatArgument(SolarScene *solarScene, int heliostat_id);

    SunrayArgument generateSunrayArgument(Sunray *sunray);

    int setFlatRectangleHeliostatVertexes(float3 *&d_heliostat_vertexes, std::vector<Heliostat *> &heliostats,
            int start_id, int end_id);
};

#endif //SOLARENERGYRAYTRACING_QUASIMONTECARLORAYTRACER_H
