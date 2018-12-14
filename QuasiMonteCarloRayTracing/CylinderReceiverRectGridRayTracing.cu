#include "CylinderReceiverRectGridRayTracing.cuh"
#include "Grid/rectGridDDA.cuh"
#include "Receiver/cylinderReceiverIntersection.cuh"

__global__ void map_raytracing(SunrayArgument sunrayArgument, CylinderReceiver cylinderReceiver,
                               RectGrid rectGrid, HeliostatArgument heliostatArgument,
                               float3 *d_heliostat_vertexes, float factor) {
    long long myId = global_func::getThreadId();
    if (myId >= heliostatArgument.numberOfMicroHeliostats * sunrayArgument.numberOfLightsPerGroup)
        return;

    //	Step 1: whether the incident light is shadowed by other heliostats
    int address = (heliostatArgument.d_microHelio_groups[myId / sunrayArgument.numberOfLightsPerGroup] +
                   myId % sunrayArgument.numberOfLightsPerGroup) % sunrayArgument.pool_size;
    float3 dir = global_func::local2world(sunrayArgument.d_samplelights[address], -sunrayArgument.sunray_direction);
    float3 origin = heliostatArgument.d_microHelio_origins[myId / sunrayArgument.numberOfLightsPerGroup];
    if (rectGridDDA::collision(origin, dir, rectGrid, d_heliostat_vertexes, heliostatArgument)) {
        return;
    }

    //	Step 2: whether the reflect light is shadowed by other heliostats
    float3 normal = heliostatArgument.d_microHelio_normals[myId / sunrayArgument.numberOfLightsPerGroup];
    address = (heliostatArgument.d_microHelio_groups[(myId / sunrayArgument.numberOfLightsPerGroup + 1) %
                                                     sunrayArgument.pool_size] +
               myId % sunrayArgument.numberOfLightsPerGroup) % sunrayArgument.pool_size;
    normal = global_func::local2world(sunrayArgument.d_perturbations[address], normal);
    normal = normalize(normal);

    dir = normalize(reflect(-dir, normal));
    if (rectGridDDA::collision(origin, dir, rectGrid, d_heliostat_vertexes, heliostatArgument)) {
        return;
    }

    // Step 3: intersect with receiver
    cylinderReceiverIntersect::receiver_drawing(cylinderReceiver, origin, dir, normal, factor);
}

void CylinderReceiverRectGridRayTracing(SunrayArgument &sunrayArgument, CylinderReceiver *cylinderReceiver,
                                         RectGrid *rectGrid, HeliostatArgument &heliostatArgument,
                                         float3 *d_subHeliostat_vertexes, float factor) {
    int nThreads = 512;
    dim3 nBlocks;
    global_func::setThreadsBlocks(nBlocks, nThreads,
                                  heliostatArgument.numberOfMicroHeliostats * sunrayArgument.numberOfLightsPerGroup,
                                  true);
    map_raytracing << < nBlocks, nThreads >> >
                                 (sunrayArgument, *cylinderReceiver, *rectGrid, heliostatArgument, d_subHeliostat_vertexes, factor);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}