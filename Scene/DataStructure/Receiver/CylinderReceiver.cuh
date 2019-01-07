//
// Created by dxt on 18-12-13.
//

#ifndef SOLARENERGYRAYTRACING_CYLINDERRECEIVER_CUH
#define SOLARENERGYRAYTRACING_CYLINDERRECEIVER_CUH

#include <math.h>

#include "Receiver.cuh"
#include "vector_arithmetic.cuh"
#include "global_constant.h"

#include <stdio.h>

/**
 * Note:
 *  - size_：
 *      size_.x is the radius of cylinder
 *      size_.y is the height of cylinder
 *      size_.z with no meaning
 * */
class CylinderReceiver : public Receiver {
public:
    __device__ __host__ CylinderReceiver() {}

    __device__ __host__ bool GIntersect(const float3 &orig, const float3 &dir, float &t, float &u, float &v) {
        // If the origin in the cylinder, it won't intersect with it
        if (innerToCylinder(orig)) {
            return false;
        }

        float2 Ro = make_float2(pos_.x - orig.x, pos_.z - orig.z);
        float tp = dot(Ro, normalize(make_float2(dir.x, dir.z)));
        float delta = dot(Ro, Ro) - tp * tp;

        // Return false if:
        //  1) The direction is different
        //  2) No intersection
        float R2 = size_.x * size_.x;
        if (tp < -Epsilon || delta > R2) {
            return false;
        }
        float t_plus = delta <= 0.0f ? size_.x : sqrtf(R2 - delta);
        t = (tp - t_plus) / length(make_float2(dir.x, dir.z));

        float3 intersect_pos = t * dir + orig;
        u = (intersect_pos.y - pos_.y) / size_.y + 0.5f;
        if (u < 0.0f || u > 1.0f) {
            return false;
        }

        float2 intersect_origin_dir = make_float2(intersect_pos.x - pos_.x, intersect_pos.z - pos_.z);
        intersect_origin_dir = normalize(intersect_origin_dir); // (cosine, sine)

        if (intersect_origin_dir.x < -1 || intersect_origin_dir.x > 1) {
            printf("\nerror occurs on intersect position: %f, %f, %f\n",
                   intersect_pos.x, intersect_pos.y, intersect_pos.z);
        }
        v = acosf(intersect_origin_dir.x) / (2 * M_PI);
//        v = (cosine > 1.0f) ? 0 : ((cosine < -1.0f) ? 0.5f : acosf(cosine) / (2 * M_PI));
        if (intersect_origin_dir.y < 0) {
            v = 1 - v;
        }
        return true;
    }

    virtual void CInit(int geometry_info);
    virtual void Cset_resolution(int geometry_info);
    virtual float3 getFocusCenter(const float3 &heliostat_position);

private:
    __device__ __host__ bool innerToCylinder(const float3 &orig) {
        float2 Ro = make_float2(pos_.x - orig.x, pos_.z - orig.z);
        return dot(Ro, Ro) <= size_.x * size_.x;
    }
};

#endif //SOLARENERGYRAYTRACING_CYLINDERRECEIVER_CUH