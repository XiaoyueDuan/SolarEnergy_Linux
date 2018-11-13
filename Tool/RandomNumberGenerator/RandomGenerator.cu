#include "RandomGenerator.cuh"
#include "global_function.cuh"
#include "check_cuda.h"

curandGenerator_t RandomGenerator::gen;

void RandomGenerator::initCudaRandGenerator() {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
}

void RandomGenerator::destroyCudaRandGenerator() {
    gen = nullptr;
    curandDestroyGenerator(gen);
}

__global__ void map_float2int(int *d_iData, float const *d_fData,
                              int low_threshold, int high_threshold, size_t size) {
    unsigned int myId = global_func::getThreadId();
    if (myId >= size)
        return;

    d_iData[myId] = int(d_fData[myId] * (high_threshold - low_threshold) + low_threshold);
}

bool RandomGenerator::gpu_Uniform(int *d_min_max_array, int low_threshold, int high_threshold, int array_length) {
    if (d_min_max_array == NULL)
        return false;

    float *d_uniform = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_uniform, array_length * sizeof(float)));
    curandGenerateUniform(gen, d_uniform, array_length);

    int nThreads;
    dim3 nBlocks;
    if (!global_func::setThreadsBlocks(nBlocks, nThreads, array_length))
        return false;

    map_float2int << < nBlocks, nThreads >> >
                                (d_min_max_array, d_uniform, low_threshold, high_threshold, array_length);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_uniform));
    return true;
}