#ifndef SOLARENERGYRAYTRACING_CYLINDERRECEIVERRECTGRIDRAYTRACING_CUH
#define SOLARENERGYRAYTRACING_CYLINDERRECEIVERRECTGRIDRAYTRACING_CUH

#include "cuda_runtime.h"
#include "global_function.cuh"
#include "RayTracingArgumentStruct.h"
#include "CylinderReceiver.cuh"
#include "RectGrid.cuh"

void CylinderReceiverRectGridRayTracing(SunrayArgument &sunrayArgument, CylinderReceiver *cylinderReceiver,
                                         RectGrid *rectGrid, HeliostatArgument &heliostatArgument,
                                         float3 *d_subHeliostat_vertexes, float factor);

#endif //SOLARENERGYRAYTRACING_CYLINDERRECEIVERRECTGRIDRAYTRACING_CUH