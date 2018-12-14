//
// Created by dxt on 18-11-14.
//

#include <iostream>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "DataStructureUtil.h"
#include "RectangleHelio.cuh"
#include "global_function.cuh"
#include "RandomNumberGenerator/RandomGenerator.cuh"

class RectangleHelioFixture : public ::testing::Test {
protected:
    void SetUp() {
        rectangleHelio.setSize(make_float3(6.0f, 0.1f, 5.0f));
        rectangleHelio.setGap(make_float2(2.0f, 1.0f));
        rectangleHelio.setRowAndColumn(make_int2(3, 2));
        rectangleHelio.setPixelLength(1.0f);
        rectangleHelio.setNormal(make_float3(0.0f, 1.0f, 0.0f));
        rectangleHelio.setPosition(make_float3(0.0f, 0.0f, 0.0f));
    }

public:
    RectangleHelio rectangleHelio;

    std::vector<float3> convert2vector(float3 *array, int size) {
        std::vector<float3> ans;
        for (int i = 0; i < size; ++i) {
            ans.push_back(array[i]);
        }
        return ans;
    }
};

MATCHER_P(FloatNearPointwise, gap, "Check whether two float3 objects are almost the same") {
    float3 n1 = std::get<0>(arg);
    float3 n2 = std::get<1>(arg);

    *result_listener << "\nExpect value: (" << n1.x << ", " << n1.y << ", " << n1.z << ")\n";
    *result_listener << "Got value: (" << n2.x << ", " << n2.y << ", " << n2.z << ")";
    return Float3Eq(n1, n2, (float) gap);
}

TEST_F(RectangleHelioFixture, HeliostatSetSubHelioOriginsAndNormals) {
    float3 *d_microhelio_vertexs = nullptr;
    float3 *d_microhelio_normals = nullptr;
    int size = rectangleHelio.CGetDiscreteMicroHelioOriginsAndNormals(d_microhelio_vertexs, d_microhelio_normals);
    float3 *h_microhelio_vertexs = nullptr;
    float3 *h_microhelio_normals = nullptr;

    global_func::gpu2cpu(h_microhelio_vertexs, d_microhelio_vertexs, size);
    global_func::gpu2cpu(h_microhelio_normals, d_microhelio_normals, size);

    std::vector<float3> calculated_normals = convert2vector(h_microhelio_normals, size);
    std::vector<float3> calculated_vertexes = convert2vector(h_microhelio_vertexs, size);

    std::vector<float3> expect_normals = std::vector<float3>(12, make_float3(0, 1, 0));
    std::vector<float3> expect_vertexes = std::vector<float3>({make_float3(-2.5f, 0.05f, -2.0f),
                                                               make_float3(-1.5f, 0.05f, -2.0f),
                                                               make_float3(1.5f, 0.05f, -2.0f),
                                                               make_float3(2.5f, 0.05f, -2.0f),
                                                               make_float3(-2.5f, 0.05f, 0.0f),
                                                               make_float3(-1.5f, 0.05f, 0.0f),
                                                               make_float3(1.5f, 0.05f, -0.0f),
                                                               make_float3(2.5f, 0.05f, -0.0f),
                                                               make_float3(-2.5f, 0.05f, 2.0f),
                                                               make_float3(-1.5f, 0.05f, 2.0f),
                                                               make_float3(1.5f, 0.05f, 2.0f),
                                                               make_float3(2.5f, 0.05f, 2.0f)});

    EXPECT_THAT(calculated_normals, ::testing::Pointwise(FloatNearPointwise(1e-3), expect_normals));
    EXPECT_THAT(calculated_vertexes, ::testing::Pointwise(FloatNearPointwise(1e-3), expect_vertexes));

    checkCudaErrors(cudaFree(d_microhelio_normals));
    checkCudaErrors(cudaFree(d_microhelio_vertexs));
    d_microhelio_normals = nullptr;
    d_microhelio_vertexs = nullptr;

    delete[] h_microhelio_normals;
    delete[] h_microhelio_vertexs;
    h_microhelio_normals = nullptr;
    h_microhelio_vertexs = nullptr;
}

TEST_F(RectangleHelioFixture, HeliostatSetSubHelioOriginAndNormalsWithRotate) {
    float3 *d_microhelio_vertexs = nullptr;
    float3 *d_microhelio_normals = nullptr;
    rectangleHelio.setNormal(make_float3(0.0f, 0.0f, 1.0f));
    int size = rectangleHelio.CGetDiscreteMicroHelioOriginsAndNormals(d_microhelio_vertexs, d_microhelio_normals);
    float3 *h_microhelio_vertexs = nullptr;
    float3 *h_microhelio_normals = nullptr;

    global_func::gpu2cpu(h_microhelio_vertexs, d_microhelio_vertexs, size);
    global_func::gpu2cpu(h_microhelio_normals, d_microhelio_normals, size);

    std::vector<float3> calculated_normals = convert2vector(h_microhelio_normals, size);
    std::vector<float3> calculated_vertexes = convert2vector(h_microhelio_vertexs, size);

    std::vector<float3> expect_normals = std::vector<float3>(12, make_float3(0.0f, 0.0f, 1.0f));
    std::vector<float3> expect_vertexes = std::vector<float3>({make_float3(-2.5f, 2.0f, 0.05f),
                                                               make_float3(-1.5f, 2.0f, 0.05f),
                                                               make_float3(1.5f, 2.0f, 0.05f),
                                                               make_float3(2.5f, 2.0f, 0.05f),
                                                               make_float3(-2.5f, 0.0f, 0.05f),
                                                               make_float3(-1.5f, 0.0f, 0.05f),
                                                               make_float3(1.5f, 0.0f, 0.05f),
                                                               make_float3(2.5f, 0.0f, 0.05f),
                                                               make_float3(-2.5f, -2.0f, 0.05f),
                                                               make_float3(-1.5f, -2.0f, 0.05f),
                                                               make_float3(1.5f, -2.0f, 0.05f),
                                                               make_float3(2.5f, -2.0f, 0.05f)});

    EXPECT_THAT(expect_normals, ::testing::Pointwise(FloatNearPointwise(1e-3), calculated_normals));
    EXPECT_THAT(expect_vertexes, ::testing::Pointwise(FloatNearPointwise(1e-3), calculated_vertexes));

    checkCudaErrors(cudaFree(d_microhelio_normals));
    checkCudaErrors(cudaFree(d_microhelio_vertexs));
    d_microhelio_normals = nullptr;
    d_microhelio_vertexs = nullptr;

    delete[] h_microhelio_normals;
    delete[] h_microhelio_vertexs;
    h_microhelio_normals = nullptr;
    h_microhelio_vertexs = nullptr;
}

TEST_F(RectangleHelioFixture, HeliostatSetSubHelioOriginAndNormalsWithRotate2) {
    float3 *d_microhelio_vertexs = nullptr;
    float3 *d_microhelio_normals = nullptr;
    rectangleHelio.setNormal(make_float3(1.0f, 0.0f, 0.0f));
    int size = rectangleHelio.CGetDiscreteMicroHelioOriginsAndNormals(d_microhelio_vertexs, d_microhelio_normals);
    float3 *h_microhelio_vertexs = nullptr;
    float3 *h_microhelio_normals = nullptr;

    global_func::gpu2cpu(h_microhelio_vertexs, d_microhelio_vertexs, size);
    global_func::gpu2cpu(h_microhelio_normals, d_microhelio_normals, size);

    std::vector<float3> calculated_normals = convert2vector(h_microhelio_normals, size);
    std::vector<float3> calculated_vertexes = convert2vector(h_microhelio_vertexs, size);

    std::vector<float3> expect_normals = std::vector<float3>(12, make_float3(1.0f, 0.0f, 0.0f));
    std::vector<float3> expect_vertexes = std::vector<float3>({make_float3(0.05f, 2.0f, 2.50f),
                                                               make_float3(0.05f, 2.0f, 1.50f),
                                                               make_float3(0.05f, 2.0f, -1.50f),
                                                               make_float3(0.05f, 2.0f, -2.50f),
                                                               make_float3(0.05f, 0.0f, 2.50f),
                                                               make_float3(0.05f, 0.0f, 1.50f),
                                                               make_float3(0.05f, 0.0f, -1.50f),
                                                               make_float3(0.05f, 0.0f, -2.50f),
                                                               make_float3(0.05f, -2.0f, 2.50f),
                                                               make_float3(0.05f, -2.0f, 1.50f),
                                                               make_float3(0.05f, -2.0f, -1.50f),
                                                               make_float3(0.05f, -2.0f, -2.50f)});

    EXPECT_THAT(expect_normals, ::testing::Pointwise(FloatNearPointwise(1e-3), calculated_normals));
    EXPECT_THAT(expect_vertexes, ::testing::Pointwise(FloatNearPointwise(1e-3), calculated_vertexes));

    checkCudaErrors(cudaFree(d_microhelio_normals));
    checkCudaErrors(cudaFree(d_microhelio_vertexs));
    d_microhelio_normals = nullptr;
    d_microhelio_vertexs = nullptr;

    delete[] h_microhelio_normals;
    delete[] h_microhelio_vertexs;
    h_microhelio_normals = nullptr;
    h_microhelio_vertexs = nullptr;
}

TEST_F(RectangleHelioFixture, HeliostatSetSubHelioGroups) {
    int size = 32;
    int nGroup = 8;
    RandomGenerator::initCudaRandGenerator();
    int *d_groups = rectangleHelio.generateDeviceMicrohelioGroup(nGroup, size);
    int *h_groups = nullptr;
    global_func::gpu2cpu(h_groups, d_groups, size);


    std::cout << "\nInteger Uniform Random in the range of [0, " << nGroup << ") :" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << h_groups[i] << ' ';
    }
    std::cout << std::endl;

    // Clean up
    RandomGenerator::destroyCudaRandGenerator();
    delete[] h_groups;
    h_groups = nullptr;
    checkCudaErrors(cudaFree(d_groups));
    d_groups = nullptr;
}