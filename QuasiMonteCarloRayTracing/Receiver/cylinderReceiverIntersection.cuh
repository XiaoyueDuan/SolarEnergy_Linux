//
// Created by dxt on 18-12-14.
//

#ifndef SOLARENERGYRAYTRACING_CYLINDERRECEIVERINTERSECTION_CUH
#define SOLARENERGYRAYTRACING_CYLINDERRECEIVERINTERSECTION_CUH

#include "CylinderReceiver.cuh"

namespace cylinderReceiverIntersect {
    __device__ void receiver_drawing(CylinderReceiver &cylinderReceiver,
            const float3 &orig, const float3 &dir, const float3 &normal, float factor);
}

#endif //SOLARENERGYRAYTRACING_CYLINDERRECEIVERINTERSECTION_CUH