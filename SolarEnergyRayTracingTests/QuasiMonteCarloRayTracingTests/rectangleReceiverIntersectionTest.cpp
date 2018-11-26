//
// Created by dxt on 18-11-26.
//

#include <SceneLoader.h>
#include <SceneProcessor.h>
#include <RectangleReceiver.cuh>
#include <RectGrid.cuh>

#include "Receiver/rectangleReceiverIntersection.cuh"
#include "QuasiMonteCarloRayTracer.h"
#include "RectangleReceiverRectGridRayTracing.cuh"
#include "global_function.cuh"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

MATCHER_P(FloatNearPointwise, gap, "Check whether two float values are almost the same\n") {
    float n1 = std::get<0>(arg);
    float n2 = std::get<1>(arg);

    *result_listener << "\nExpect value: " << n1;
    *result_listener << ". But got value: " << n2;
    return n1 < n2 + gap && n1 > n2 - gap;
}

class rectangleReceiverIntersectionFixture : public ::testing::Test {
protected:
    void loadAndProcessScene() {
        solarScene = SolarScene::GetInstance();

        SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
        std::string configuration_path = "test_file/test_configuration.json";
        sceneConfiguration->loadConfiguration(configuration_path);

        std::string scene_path = "test_file/test_scene.scn";
        SceneLoader sceneLoader;
        sceneLoader.SceneFileRead(solarScene, scene_path);

        SceneProcessor sceneProcessor(sceneConfiguration);
        sceneProcessor.processScene(solarScene);
    }

    void SetUp() {
        loadAndProcessScene();
    }

    void TearDown() {
        solarScene->clear();
    }

public:
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

    void compareForHeliostat1Result(std::vector<float> image, float distance, int numberOfSunraysPerGroup, int row,
                                    int col) {
        vector<float> expect_result(row * col, 0.0f);
        float expect_value = rectangleReceiverIntersect::eta_aAlpha(distance) * float(numberOfSunraysPerGroup);
        for (int r = 7; r < 13; ++r) {          //six
            for (int c = 12; c < 20; ++c) {     //eight
                expect_result[r * col + c] = expect_value;
            }
        }

        /**
         * Noted: The precision of float result is as large as 0.1 here.
         * */
        EXPECT_THAT(expect_result, ::testing::Pointwise(FloatNearPointwise(1e-1), image));
    }

    void compareForHeliostat2Result(std::vector<float> image, float distance, int numberOfSunraysPerGroup, int row,
                                    int col) {
        vector<float> expect_result(row * col, 0.0f);
        float expect_value = rectangleReceiverIntersect::eta_aAlpha(distance) * float(numberOfSunraysPerGroup);

        expect_result[6 * col + 11] = expect_value;
        expect_result[6 * col + 12] = expect_value;
        expect_result[6 * col + 13] = expect_value;
        expect_result[6 * col + 14] = expect_value;
        expect_result[6 * col + 17] = expect_value;
        expect_result[6 * col + 18] = expect_value;
        expect_result[6 * col + 19] = expect_value;
        expect_result[6 * col + 20] = expect_value;

        expect_result[7 * col + 11] = expect_value;
        expect_result[7 * col + 20] = expect_value;

        expect_result[9 * col + 11] = expect_value;
        expect_result[9 * col + 20] = expect_value;

        expect_result[10 * col + 11] = expect_value;
        expect_result[10 * col + 20] = expect_value;

        expect_result[12 * col + 11] = expect_value;
        expect_result[12 * col + 20] = expect_value;

        expect_result[13 * col + 11] = expect_value;
        expect_result[13 * col + 12] = expect_value;
        expect_result[13 * col + 13] = expect_value;
        expect_result[13 * col + 14] = expect_value;
        expect_result[13 * col + 17] = expect_value;
        expect_result[13 * col + 18] = expect_value;
        expect_result[13 * col + 19] = expect_value;
        expect_result[13 * col + 20] = expect_value;

        /**
         * Noted: The precision of float result is as large as 0.1 here.
         * */
        EXPECT_THAT(expect_result, ::testing::Pointwise(FloatNearPointwise(1e-1), image));
    }

    void changeSunLightsAndPerturbationToParallel() {
        Sunray *sunray = solarScene->getSunray();

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

    rectangleReceiverIntersectionFixture() : solarScene(nullptr) {}

    QuasiMonteCarloRayTracer QMCRTracer;
    SolarScene *solarScene;
};


TEST_F(rectangleReceiverIntersectionFixture, rectangleReceiverIntersectionParallel) {
    // Change lights to parallel direction
    changeSunLightsAndPerturbationToParallel();

    // Construct arguments
    SunrayArgument sunrayArgument = QMCRTracer.generateSunrayArgument(solarScene->getSunray());
    RectangleReceiver *rectangleReceiver = dynamic_cast<RectangleReceiver *>(solarScene->getReceivers()[0]);
    RectGrid *rectGrid = dynamic_cast<RectGrid *>(solarScene->getGrid0s()[0]);
    float factor = 1.0f;
    float3 *d_subHeliostat_vertexes = nullptr;
    int start_heliostat_id = rectGrid->getStartHeliostatPosition();
    int end_heliostat_id = start_heliostat_id + rectGrid->getNumberOfHeliostats();
    QMCRTracer.setFlatRectangleHeliostatVertexes(d_subHeliostat_vertexes, solarScene->getHeliostats(),
                                                 start_heliostat_id, end_heliostat_id);

    // Heliostat 1
    HeliostatArgument heliostatArgument0 = QMCRTracer.generateHeliostatArgument(solarScene, 0);
    RectangleReceiverRectGridRayTracing(sunrayArgument, rectangleReceiver, rectGrid, heliostatArgument0,
                                        d_subHeliostat_vertexes, factor);

    int2 resolution = rectangleReceiver->getResolution();
    vector<float> image = deviceArray2vector(rectangleReceiver->getDeviceImage(), resolution.y * resolution.x);
    compareForHeliostat1Result(image, 7.85f, sunrayArgument.numberOfLightsPerGroup, resolution.y, resolution.x);

    // Heliostat 2
    HeliostatArgument heliostatArgument1 = QMCRTracer.generateHeliostatArgument(solarScene, 1);
    rectangleReceiver->Cclean_image_content();
    RectangleReceiverRectGridRayTracing(sunrayArgument, rectangleReceiver, rectGrid, heliostatArgument1,
                                        d_subHeliostat_vertexes, factor);
    image = deviceArray2vector(rectangleReceiver->getDeviceImage(), resolution.y * resolution.x);
    compareForHeliostat2Result(image, 11.95f, sunrayArgument.numberOfLightsPerGroup, resolution.y, resolution.x);
}


TEST_F(rectangleReceiverIntersectionFixture, rectangleReceiverIntersection) {
    // Construct arguments
    SunrayArgument sunrayArgument = QMCRTracer.generateSunrayArgument(solarScene->getSunray());
    RectangleReceiver *rectangleReceiver = dynamic_cast<RectangleReceiver *>(solarScene->getReceivers()[0]);
    RectGrid *rectGrid = dynamic_cast<RectGrid *>(solarScene->getGrid0s()[0]);
    float factor = 1.0f;
    float3 *d_subHeliostat_vertexes = nullptr;
    int start_heliostat_id = rectGrid->getStartHeliostatPosition();
    int end_heliostat_id = start_heliostat_id + rectGrid->getNumberOfHeliostats();
    QMCRTracer.setFlatRectangleHeliostatVertexes(d_subHeliostat_vertexes, solarScene->getHeliostats(),
                                                 start_heliostat_id, end_heliostat_id);

    // Heliostat 1
    HeliostatArgument heliostatArgument0 = QMCRTracer.generateHeliostatArgument(solarScene, 0);
    RectangleReceiverRectGridRayTracing(sunrayArgument, rectangleReceiver, rectGrid, heliostatArgument0,
                                        d_subHeliostat_vertexes, factor);

    int2 resolution = rectangleReceiver->getResolution();
    vector<float> image = deviceArray2vector(rectangleReceiver->getDeviceImage(), resolution.y * resolution.x);
    std::cout <<"Heliostat 1:"<< std::endl;
    for (int r = 0; r < resolution.y; ++r) {
        std::cout << std::endl;
        for (int c = 0; c < resolution.x; ++c) {
            std::cout << image[r * resolution.x + c] << "\t\t";
        }
    }

    // Heliostat 2
    HeliostatArgument heliostatArgument1 = QMCRTracer.generateHeliostatArgument(solarScene, 1);
    rectangleReceiver->Cclean_image_content();
    RectangleReceiverRectGridRayTracing(sunrayArgument, rectangleReceiver, rectGrid, heliostatArgument1,
                                        d_subHeliostat_vertexes, factor);
    image = deviceArray2vector(rectangleReceiver->getDeviceImage(), resolution.y * resolution.x);

    std::cout <<"Heliostat 2:"<< std::endl;
    for (int r = 0; r < resolution.y; ++r) {
        std::cout << std::endl;
        for (int c = 0; c < resolution.x; ++c) {
            std::cout << image[r * resolution.x + c] << "\t\t";
        }
    }
}




