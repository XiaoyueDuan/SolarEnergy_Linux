//
// Created by dxt on 18-12-14.
//

#include "receiverIntersectUtil.h"

void changeSunLightsAndPerturbationToParallel(Sunray *sunray) {
    if(sunray== nullptr) {
        return;
    }

    // Generate parallel lights and transfer them to device
    int size = sunray->getNumOfSunshapeLightsPerGroup() * sunray->getNumOfSunshapeGroups();
    float3 *h_samplelights_ans_perturbation = new float3[size];
    for (int i = 0; i < size; ++i) {
        h_samplelights_ans_perturbation[i] = make_float3(0.0f, 1.0f, 0.0f);
    }
    float3 *d_samplelights = nullptr;
    float3 *d_perturbation = nullptr;
    global_func::cpu2gpu(d_samplelights, h_samplelights_ans_perturbation, size);
    global_func::cpu2gpu(d_perturbation, h_samplelights_ans_perturbation, size);

    // Set the perturbation and sample lights of sunray
    sunray->CClear();
    sunray->setDevicePerturbation(d_samplelights);
    sunray->setDeviceSampleLights(d_perturbation);

    // Clear
    delete[] h_samplelights_ans_perturbation;
}

std::vector<float> deviceArray2vector(float *d_array, int size) {
    vector<float> ans(size, 0.0f);
    float *h_array = nullptr;

    global_func::gpu2cpu(h_array, d_array, size);
    for (int i = 0; i < size; ++i) {
        ans[i] = h_array[i];
    }

    delete[] h_array;
    h_array = nullptr;
    return ans;
}