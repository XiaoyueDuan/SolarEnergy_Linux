#ifndef SOLARENERGYRAYTRACING_RECTANGLERECEIVERRECTGRIDRAYTRACING_CUH
#define SOLARENERGYRAYTRACING_RECTANGLERECEIVERRECTGRIDRAYTRACING_CUH

#include "cuda_runtime.h"
#include "global_function.cuh"
#include "RayTracingArgumentStruct.h"
#include "RectangleReceiver.cuh"
#include "RectGrid.cuh"

void RectangleReceiverRectGridRayTracing(SunrayArgument &sunrayArgument, RectangleReceiver *rectangleReceiver,
                                         RectGrid *rectGrid, HeliostatArgument &heliostatArgument,
                                         float3 *d_subHeliostat_vertexes, float factor);

#endif //SOLARENERGYRAYTRACING_RECTANGLERECEIVERRECTGRIDRAYTRACING_CUH