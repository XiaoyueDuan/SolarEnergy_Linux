#include <math.h>

#include "global_function.cuh"
#include "SceneProcessor.h"
#include "RandomNumberGenerator/RandomGenerator.cuh"

namespace samplelights {
    float sunshape_intensity(float theta, const float &k, const float &gamma) {
        // theta must be in range [0,4.65)
        return cosf(0.326 * theta) / cosf(0.308 * theta);
    }

    float integration_larger_465_intensity(float theta, float k, float gamma) {
        // theta must be in range [4.65, 9.3]
        return expf(k) / (gamma + 1) * powf(theta, gamma + 1);
    }

    // Description:
    //	- h_k:(return value) slope (length_interval / subarea)
    //	- num_group: num of intervals
    //	- k, gamma: used in intensity calculation
    //	- upper_lm: length_interval = upper_lm / num_group
    float2 *parameters_generate(float *h_k, int num_group, float k, float gamma, float upper_lm) {
        float length_interval = upper_lm / float(num_group);
        float x = 0.0f;
        float2 *h_cdf = new float2[num_group + 1];
        h_cdf[0].x = 0.0f;
        h_cdf[0].y = 0.0f;
        float hist_pre = sunshape_intensity(x, k, gamma);
        float hist_current;
        float subarea;
        for (int i = 1; i <= num_group; ++i) {
            x += length_interval;
            hist_current = sunshape_intensity(x, k, gamma);
            subarea = (hist_pre + hist_current) * length_interval / 2.0f;

            h_cdf[i].x = x;
            h_cdf[i].y = subarea + h_cdf[i - 1].y;
            h_k[i - 1] = length_interval / subarea;

            hist_pre = hist_current;
        }
        return h_cdf;
    }

    __host__ __device__ int max_less_index(float2 *d_cdf, float value, size_t n)// d_cdf[n+1]
    {
        int left = 0, right = n;
        int mid;
        while (left <= right) {
            mid = (left + right) >> 1;
            if (value > d_cdf[mid].y) {
                left = mid + 1;
            } else if (value < d_cdf[mid].y) {
                right = mid - 1;
            } else //value==d_cdf[mid].y
                return mid;
        }
        return right;
    }

    __global__ void linear_interpolate(float *d_0_1, float2 *d_cdf, float *d_k,
                                       float integration_less_465,
                                       float gamma, float A, float B, float C,
                                       size_t n,// d_cdf[n+1]
                                       size_t size) {
        const int myId = global_func::getThreadId();
        if (myId >= size)
            return;

        float u = d_0_1[myId] * A;

        if (u < integration_less_465) {
            int id = max_less_index(d_cdf, u, n);
            d_0_1[myId] = (u - d_cdf[id].y) * d_k[id] + d_cdf[id].x;
        } else {
            d_0_1[myId] = powf((u - integration_less_465) * B + C, 1 / (gamma + 1));
        }
        d_0_1[myId] /= 1000.0f;
        return;
    }

    __global__ void
    map_angle2xyz(float3 *d_turbulance, const float *d_nonUniform, const float *d_uniform, const size_t size) {
        unsigned int myId = global_func::getThreadId();
        if (myId >= size)
            return;

        float theta = d_nonUniform[myId], phi = d_uniform[myId] * 2 * MATH_PI;
        d_turbulance[myId] = global_func::angle2xyz(make_float2(theta, phi));
    }
}

/**
 * Permutation: Generate sample lights with
 *  - theta ~ G(0, disturb_std)
 *  - phi ~ Uniform(0, 2pi)
 * */

bool SceneProcessor::set_perturbation(Sunray &sunray) {
    if(sunray.getDevicePerturbation()) {
        // Make sure the sunray.perturbation are empty.
        // If not, you clean the device perturbation before calling this method
        return false;
    }

    int size = sunray.getNumOfSunshapeGroups() * sunray.getNumOfSunshapeLightsPerGroup();

    //	Step 1:	Allocate memory for sunray.d_perturbation_ on GPU
    float3 *d_permutation = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_permutation, sizeof(float3) * size));

    //	Step 2:	Allocate memory for theta and phi
    float *d_guassian_theta = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_guassian_theta, sizeof(float) * size));
    float *d_uniform_phi = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_uniform_phi, sizeof(float) * size));

    //	Step 3:	Generate theta and phi
    RandomGenerator::gpu_Gaussian(d_guassian_theta, 0.0f, sceneConfiguration->getDisturb_std(), size);
    RandomGenerator::gpu_Uniform(d_uniform_phi, size);

    //	Step 4:	(theta, phi) -> ( x, y, z)
    int nThreads;
    dim3 nBlocks;
    global_func::setThreadsBlocks(nBlocks, nThreads, size);
    samplelights::map_angle2xyz << < nBlocks, nThreads >> > (d_permutation, d_guassian_theta, d_uniform_phi, size);
    sunray.setDevicePerturbation(d_permutation);

    //	Step 5: Cleanup
    checkCudaErrors(cudaFree(d_guassian_theta));
    checkCudaErrors(cudaFree(d_uniform_phi));

    return true;
}


/**
 * sampleLights: Generate sample lights with
 *  - theta ~ Buie distribution
 *  - phi ~ Uniform(0, 2pi)
 * */
bool SceneProcessor::set_samplelights(Sunray &sunray) {
    if(sunray.getDeviceSampleLights()) {
        // Make sure the sunray.samplelights are empty.
        // If not, you clean the device sample lights before calling this method
        return false;
    }

    // input parameters
    int num_group = sceneConfiguration->getInverse_transform_sampling_groups();
    float csr = sunray.getCSR();
    float upper_lm = 4.65f;

    //
    float k = 0.9f * logf(13.5f * csr) * powf(csr, -0.3f);
    float gamma = 2.2f * logf(0.52f * csr) * powf(csr, 0.43f) - 0.1f;
    float integration_value_between_465_930 = samplelights::integration_larger_465_intensity(9.3f, k, gamma) -
                                              samplelights::integration_larger_465_intensity(upper_lm, k, gamma);

    float *h_k = new float[num_group];
    float2 *h_cdf = samplelights::parameters_generate(h_k, num_group, k, gamma, upper_lm);
    float value_less_465 = h_cdf[num_group].y;

    float2 *d_cdf = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_cdf, sizeof(float2) * (num_group + 1)));
    checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, sizeof(float2) * (num_group + 1), cudaMemcpyHostToDevice));
    float *d_k = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_k, sizeof(float) * num_group));
    checkCudaErrors(cudaMemcpy(d_k, h_k, sizeof(float) * num_group, cudaMemcpyHostToDevice));

    // Generate uniform random theta and phi in range [0,1]
    float *d_theta = nullptr;
    int num_random = sunray.getNumOfSunshapeGroups() * sunray.getNumOfSunshapeLightsPerGroup();
    checkCudaErrors(cudaMalloc((void **) &d_theta, sizeof(float) * num_random));
    RandomGenerator::gpu_Uniform(d_theta, num_random);

    float *d_phi = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_phi, sizeof(float) * num_random));
    RandomGenerator::gpu_Uniform(d_phi, num_random);

    int nThreads = 1024;
    int nBlock = (num_random + nThreads - 1) / nThreads;
    float A = value_less_465 + integration_value_between_465_930;
    float B = (gamma + 1) / expf(k);
    float C = powf(upper_lm, gamma + 1);

    // Change to correct theta
    samplelights::linear_interpolate << < nBlock, nThreads >> > (d_theta, d_cdf, d_k, value_less_465, gamma,
            A, B, C, num_group, num_random);

    float3 *d_samplelights = nullptr;
    cudaMalloc((void **) &d_samplelights, sizeof(float3) * num_random);
    samplelights::map_angle2xyz << < nBlock, nThreads >> > (d_samplelights, d_theta, d_phi, num_random);
    sunray.setDeviceSampleLights(d_samplelights);

    // clear
    delete[] h_k;
    delete[] h_cdf;
    h_k = nullptr;
    h_cdf = nullptr;

    checkCudaErrors(cudaFree(d_theta));
    checkCudaErrors(cudaFree(d_phi));
    checkCudaErrors(cudaFree(d_cdf));
    checkCudaErrors(cudaFree(d_k));
    d_theta = nullptr;
    d_phi = nullptr;
    d_cdf = nullptr;
    d_k = nullptr;

    return true;
}

