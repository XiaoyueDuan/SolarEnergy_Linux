//
// Created by dxt on 18-12-15.
//

#ifndef SOLARENERGYRAYTRACING_RAYTRACINGPIPELINE_H
#define SOLARENERGYRAYTRACING_RAYTRACINGPIPELINE_H

#include <string>

#include "DataStructure/Receiver/Receiver.cuh"

class RayTracingPipeline {
public:
    static void rayTracing(int argc, char *argv[]);

private:
    static void saveReceiverResult(Receiver *receiver, std::string pathAndName);
};

#endif //SOLARENERGYRAYTRACING_RAYTRACINGPIPELINE_H
