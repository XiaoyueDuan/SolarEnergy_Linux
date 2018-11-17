//
// Created by dxt on 18-10-29.
//

#ifndef SOLARENERGYRAYTRACING_SUNRAY_CUH
#define SOLARENERGYRAYTRACING_SUNRAY_CUH

#include <cuda_runtime.h>

class Sunray {
public:
    __device__ __host__ Sunray() : d_samplelights_(nullptr), d_perturbation_(nullptr) {}

    __device__ __host__ Sunray(float3 sun_dir, int num_sunshape_groups, int lights_per_group,
                               float dni, float csr) : Sunray() {
        sun_dir_ = sun_dir;
        dni_ = dni;
        csr_ = csr;
        num_sunshape_groups_ = num_sunshape_groups;
        num_sunshape_lights_per_group_ = lights_per_group;
    }

    __device__ __host__ ~Sunray();

    void CClear();

    /**
     * Getters and setters of attributes for Sunray object
     */
    __device__ __host__ float3 getSunDirection() const;
    void setSunDirection(float3 sun_dir_);

    __device__ __host__ float getDNI() const;
    void setDNI(float dni_);

    __device__ __host__ float getCSR() const;
    void setCSR(float csr_);

    __device__ __host__ int getNumOfSunshapeGroups() const;
    void setNumOfSunshapeGroups(int num_sunshape_groups_);

    __device__ __host__ int getNumOfSunshapeLightsPerGroup() const;
    void setNumOfSunshapeLightsPerGroup(int num_sunshape_lights_per_group_);

    __device__ __host__ float3 *getDeviceSampleLights() const;
    void setDeviceSampleLights(float3 *d_samplelights_);

    __device__ __host__ float3 *getDevicePerturbation() const;
    void setDevicePerturbation(float3 *d_perturbation_);

    float getReflectiveRate() const;
    void setReflectiveRate(float reflective_rate_);

private:
    float3 sun_dir_;                        // e.g. 0.306454	-0.790155	0.530793
    float dni_;                             // e.g. 1000.0
    float csr_;                             // e.g. 0.1
    int num_sunshape_groups_;               // e.g. 8
    int num_sunshape_lights_per_group_;     // e.g. 1024
    float reflective_rate_;                 // e.g. 0.88
                                            // (Though it should be a field of heliostat, since all heliostats with the
                                            // same reflective rate in the our system, we put the filed in sunray
                                            // (because the SolarScene only contains one sunrray))
    float3 *d_samplelights_;                // e.g. point to sample lights memory on GPU
                                            //		memory size = num_sunshape_groups_ * num_sunshape_lights_per_group_
    float3 *d_perturbation_;                // e.g. point to the memory on GPU
                                            //		which obeys Gaussian distribution
                                            //		memory size = num_sunshape_groups_ * num_sunshape_lights_per_group_
};

#endif //SOLARENERGYRAYTRACING_SUNRAY_CUH


