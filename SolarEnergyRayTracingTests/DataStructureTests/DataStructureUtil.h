//
// Created by dxt on 18-12-14.
//

#ifndef SOLARENERGYRAYTRACING_DATASTRUCTUREUTIL_H
#define SOLARENERGYRAYTRACING_DATASTRUCTUREUTIL_H

#include "SolarScene.h"

inline bool Float3Eq(float3 n1, float3 n2, float gap) {
    return (n1.x > n2.x - gap && n1.x < n2.x + gap) &&
           (n1.y > n2.y - gap && n1.y < n2.y + gap) &&
           (n1.z > n2.z - gap && n1.z < n2.z + gap);
}

#endif //SOLARENERGYRAYTRACING_DATASTRUCTUREUTIL_H