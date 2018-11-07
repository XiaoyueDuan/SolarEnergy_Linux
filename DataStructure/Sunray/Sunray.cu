//
// Created by dxt on 18-10-29.
//

#include "Sunray.cuh"
#include "check_cuda.h"

/*
 * Getters and setters of attributes for Sunray object
 */
const float3 &Sunray::getSunDirection() const {
    return sun_dir_;
}

void Sunray::setSunDirection(float3 sun_dir_) {
    Sunray::sun_dir_ = sun_dir_;
}

float Sunray::getDNI() const {
    return dni_;
}

void Sunray::setDNI(float dni_) {
    Sunray::dni_ = dni_;
}

float Sunray::getCSR() const {
    return csr_;
}

void Sunray::setCSR(float csr_) {
    Sunray::csr_ = csr_;
}

int Sunray::getNumOfSunshapeGroups() const {
    return num_sunshape_groups_;
}

void Sunray::setNumOfSunshapeGroups(int num_sunshape_groups_) {
    Sunray::num_sunshape_groups_ = num_sunshape_groups_;
}

int Sunray::getNumOfSunshapeLightsPerGroup() const {
    return num_sunshape_lights_per_group_;
}

void Sunray::setNumOfSunshapeLightsPerGroup(int num_sunshape_lights_per_group_) {
    Sunray::num_sunshape_lights_per_group_ = num_sunshape_lights_per_group_;
}

float3 *Sunray::getDeviceSampleLights() const {
    return d_samplelights_;
}

void Sunray::setDeviceSampleLights(float3 *d_samplelights_) {
    Sunray::d_samplelights_ = d_samplelights_;
}

float3 *Sunray::getDevicePerturbation() const {
    return d_perturbation_;
}

void Sunray::setDevicePerturbation(float3 *d_perturbation_) {
    Sunray::d_perturbation_ = d_perturbation_;
}

/*
 * Getters and setters of attributes for Sunray object
 */
Sunray::~Sunray() {
    if(d_samplelights_)
        d_samplelights_ = nullptr;
    if(d_perturbation_)
        d_perturbation_ = nullptr;
}

void Sunray::CClear() {
    if (d_samplelights_)
    {
        checkCudaErrors(cudaFree(d_samplelights_));
        d_samplelights_ = nullptr;
    }

    if (d_perturbation_)
    {
        checkCudaErrors(cudaFree(d_perturbation_));
        d_perturbation_ = nullptr;
    }
}
