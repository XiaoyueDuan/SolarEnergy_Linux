#ifndef SOLARENERGYRAYTRACING_RECTANGLERECEIVERINTERSECTION_CUH
#define SOLARENERGYRAYTRACING_RECTANGLERECEIVERINTERSECTION_CUH

#include "vector_arithmetic.cuh"
#include "RectangleReceiver.cuh"
#include "receiverIntersectionUtil.cuh"

namespace rectangleReceiverIntersect {
    __device__ void
    receiver_drawing(RectangleReceiver &rectangleReceiver, const float3 &orig, const float3 &dir, const float3 &normal,
                     float factor);
}

#endif //SOLARENERGYRAYTRACING_RECTANGLERECEIVERINTERSECTION_CUH