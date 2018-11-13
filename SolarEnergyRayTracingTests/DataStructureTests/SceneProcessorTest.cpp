//
// Created by dxt on 18-11-12.
//

#include "global_function.cuh"
#include "gtest/gtest.h"
#include "SceneProcessor.h"
#include "RandomNumberGenerator/RandomGenerator.cuh"

class SceneProcessorFixture : public ::testing::Test {
protected:
    void SetUp() {
        SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
        std::string configuration_path = "test_file/test_configuration.json";
        sceneConfiguration->loadConfiguration(configuration_path);

        sceneProcessor = new SceneProcessor(sceneConfiguration);

        // test needs random numbers
        RandomGenerator::initCudaRandGenerator();
    }

    void TearDown() {
        delete (sceneProcessor);
        sceneProcessor = nullptr;

        RandomGenerator::destroyCudaRandGenerator();
    }

public:
    SceneProcessorFixture() : sceneProcessor(nullptr), sceneConfiguration(nullptr) {}

    SceneProcessor *sceneProcessor;
    SceneConfiguration *sceneConfiguration;

    void print(std::string name, float3 *d_array, int n) {
        float3 *h_array = nullptr;
        global_func::gpu2cpu(h_array, d_array, n);

        std::cout << "\nPrint " << name << std::endl;
        for (int i = 0; i < n; ++i) {
            std::cout << "(" << h_array[i].x << ", " << h_array[i].y << ", " << h_array[i].z << ")" << std::endl;
        }

        delete[] h_array;
        h_array = nullptr;
    }
};

TEST_F(SceneProcessorFixture, set_sunray_content) {
    Sunray sunray;

    /**
     * Usage: Uncomment following in 'set_perturbation(sunray) && set_samplelights(sunray)' in 'SceneProcessor.cu'
     * to see the distribution of random numbers.
     *
     *  #ifdef SCENE_PROCESSOR_TEST_CPP
     *      ...
     *  #endif
     *
     * */
    sceneProcessor->set_sunray_content(sunray);

    int print_size = 10;
    print("Perturbation", sunray.getDevicePerturbation(), min(print_size, sunray.getNumOfSunshapeLightsPerGroup()));
    print("SampleLights", sunray.getDeviceSampleLights(), min(print_size, sunray.getNumOfSunshapeLightsPerGroup()));
}

