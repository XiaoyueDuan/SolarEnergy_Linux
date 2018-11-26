#include "rectangleReceiverIntersection.cuh"

__device__ void rectangleReceiverIntersect::receiver_drawing(RectangleReceiver &rectangleReceiver, const float3 &orig,
                                                             const float3 &dir, const float3 &normal,
                                                             float factor) {
    //	Step1: Intersect with receiver
    float t, u, v;
    if (!rectangleReceiver.GIntersect(orig, dir, t, u, v))
        return;

    //	Step2: Calculate the energy of the light
    float energy = calEnergy(t, dir, normal, factor);

    //	Step3: Add the energy to the intersect position
    // Intersect location
    int2 row_col = make_int2(u * rectangleReceiver.getResolution().y, v * rectangleReceiver.getResolution().x);
    int address = row_col.x * rectangleReceiver.getResolution().x + row_col.y;  //col_row.y + col_row.x*resolution.y;
    float *image = rectangleReceiver.getDeviceImage();
    atomicAdd(&(image[address]), energy);
}