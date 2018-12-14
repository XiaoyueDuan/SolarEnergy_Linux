#include "cylinderReceiverIntersection.cuh"
#include "receiverIntersectionUtil.cuh"

__device__ void cylinderReceiverIntersect::receiver_drawing(CylinderReceiver &cylinderReceiver,
        const float3 &orig, const float3 &dir, const float3 &normal, float factor) {
    //	Step1: Intersect with receiver
    float t, u, v;
    if (!cylinderReceiver.GIntersect(orig, dir, t, u, v))
        return;

    //	Step2: Calculate the energy of the light
    float energy = calEnergy(t, dir, normal, factor);

    //	Step3: Add the energy to the intersect position
    // Intersect location
    int2 row_col = make_int2(u * cylinderReceiver.getResolution().y, v * cylinderReceiver.getResolution().x);
    int address = row_col.x * cylinderReceiver.getResolution().x + row_col.y;  //col_row.y + col_row.x*resolution.y;
    float *image = cylinderReceiver.getDeviceImage();
    atomicAdd(&(image[address]), energy);
}