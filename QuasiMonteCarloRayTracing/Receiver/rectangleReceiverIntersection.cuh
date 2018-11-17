#ifndef SOLARENERGYRAYTRACING_RECTANGLERECEIVERINTERSECTION_CUH
#define SOLARENERGYRAYTRACING_RECTANGLERECEIVERINTERSECTION_CUH

namespace rectangleReceiverIntersect {
    /**
     * Step 3: intersect with receiver
     * */
    inline __host__ __device__ float eta_aAlpha(const float &d) {
        if (d <= 1000.0f)
            return 0.99331f - 0.0001176f * d + 1.97f * (1e-8f) * d * d;
        return expf(-0.0001106f * d);
    }

    inline __host__ __device__ float calEnergy(float distance, float3 dir, float3 normal, float factor) {
        //       cosine(dir, normal)            * eta         * factor(DNI*Ssub*reflective_rate/numberOfLightsPerGroup)
        return fabsf(dot(dir, normal)) * eta_aAlpha(distance) * factor;
    }

    __device__ void
    receiver_drawing(RectangleReceiver &rectangleReceiver, const float3 &orig, const float3 &dir, const float3 &normal,
                     float factor);
}

#endif //SOLARENERGYRAYTRACING_RECTANGLERECEIVERINTERSECTION_CUH