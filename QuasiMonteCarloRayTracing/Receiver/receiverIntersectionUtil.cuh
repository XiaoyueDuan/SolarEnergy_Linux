//
// Created by dxt on 18-12-14.
//

#ifndef SOLARENERGYRAYTRACING_RECEIVERINTERSECTIONUTIL_CUH
#define SOLARENERGYRAYTRACING_RECEIVERINTERSECTIONUTIL_CUH

#include "vector_arithmetic.cuh"

inline __host__ __device__ float eta_aAlpha(const float &d) {
    if (d <= 1000.0f)
        return 0.99331f - 0.0001176f * d + 1.97f * (1e-8f) * d * d;
    return expf(-0.0001106f * d);
}

inline __host__ __device__ float calEnergy(float distance, float3 dir, float3 normal, float factor) {
    //       cosine(dir, normal)            * eta         * factor(DNI*Ssub*reflective_rate/numberOfLightsPerGroup)
    return fabsf(dot(dir, normal)) * eta_aAlpha(distance) * factor;
}

#endif //SOLARENERGYRAYTRACING_RECEIVERINTERSECTIONUTIL_CUH