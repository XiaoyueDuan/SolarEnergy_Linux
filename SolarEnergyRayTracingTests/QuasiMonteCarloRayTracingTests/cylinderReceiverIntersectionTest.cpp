//
// Created by dxt on 18-12-14.
//

#include <SceneLoader.h>
#include <SceneProcessor.h>
#include <CylinderReceiver.cuh>
#include <RectGrid.cuh>

#include "Receiver/cylinderReceiverIntersection.cuh"
#include "QuasiMonteCarloRayTracer.h"
#include "CylinderReceiverRectGridRayTracing.cuh"
#include "gtest/gtest.h"
#include "receiverIntersectUtil.h"

class cylinderReceiverIntersectionFixture : public ::testing::Test {
protected:
    void loadAndProcessScene() {
        solarScene = SolarScene::GetInstance();

        SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
        std::string configuration_path = "test_file/test_configuration2.json";
        sceneConfiguration->loadConfiguration(configuration_path);

        std::string scene_path = "test_file/test_scene_cylinder_receiver.scn";
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
    void print(vector<float> &array, int2 resolution) {
        for (int r = resolution.y - 1; r >= 0; --r) {
            for (int c = 0; c < resolution.x; ++c) {
                std::cout << array[r * resolution.x + c] << ' ';
            }
            std::cout << std::endl;
        }
    }

    cylinderReceiverIntersectionFixture() : solarScene(nullptr) {}

    QuasiMonteCarloRayTracer QMCRTracer;
    SolarScene *solarScene;
};

TEST_F(cylinderReceiverIntersectionFixture, cylinderReceiverIntersectionParallel) {
    // Change lights to parallel direction
    changeSunLightsAndPerturbationToParallel(solarScene->getSunray());

    // Construct arguments
    SunrayArgument sunrayArgument = QMCRTracer.generateSunrayArgument(solarScene->getSunray());
    CylinderReceiver *cylinderReceiver = dynamic_cast<CylinderReceiver *>(solarScene->getReceivers()[0]);
    RectGrid *rectGrid = dynamic_cast<RectGrid *>(solarScene->getGrid0s()[0]);
    float factor = 1.0f;
    float3 *d_subHeliostat_vertexes = nullptr;
    int start_heliostat_id = rectGrid->getStartHeliostatPosition();
    int end_heliostat_id = start_heliostat_id + rectGrid->getNumberOfHeliostats();
    QMCRTracer.setFlatRectangleHeliostatVertexes(d_subHeliostat_vertexes, solarScene->getHeliostats(),
                                                 start_heliostat_id, end_heliostat_id);

    // Heliostat 1 without shadow(index = 0)
    HeliostatArgument heliostatArgument0 = QMCRTracer.generateHeliostatArgument(solarScene, 0);
    CylinderReceiverRectGridRayTracing(sunrayArgument, cylinderReceiver, rectGrid, heliostatArgument0,
                                       d_subHeliostat_vertexes, factor);

    int2 resolution = cylinderReceiver->getResolution();
    vector<float> image = deviceArray2vector(cylinderReceiver->getDeviceImage(), resolution.y * resolution.x);
    std::cout<<"\nHeliostat 1 without shadowing - (r: "<<resolution.y<<", c:"<<resolution.x<<")"<<std::endl;
    print(image, resolution);

    // Heliostat 1 with shadow(index = 1)
    rectGrid = dynamic_cast<RectGrid *>(solarScene->getGrid0s()[1]);
    start_heliostat_id = rectGrid->getStartHeliostatPosition();
    end_heliostat_id = start_heliostat_id + rectGrid->getNumberOfHeliostats();
    QMCRTracer.setFlatRectangleHeliostatVertexes(d_subHeliostat_vertexes, solarScene->getHeliostats(),
                                                 start_heliostat_id, end_heliostat_id);

    HeliostatArgument heliostatArgument1 = QMCRTracer.generateHeliostatArgument(solarScene, 1);
    cylinderReceiver->Cclean_image_content();
    CylinderReceiverRectGridRayTracing(sunrayArgument, cylinderReceiver, rectGrid, heliostatArgument1,
                                       d_subHeliostat_vertexes, factor);

    image = deviceArray2vector(cylinderReceiver->getDeviceImage(), resolution.y * resolution.x);
    std::cout<<"\nHeliostat 1 with shadowing - (r: "<<resolution.y<<", c:"<<resolution.x<<")"<<std::endl;
    print(image, resolution);
}

TEST_F(cylinderReceiverIntersectionFixture, cylinderReceiverIntersectionForRealGrid3) {
    // Construct arguments
    SunrayArgument sunrayArgument = QMCRTracer.generateSunrayArgument(solarScene->getSunray());
    CylinderReceiver *cylinderReceiver = dynamic_cast<CylinderReceiver *>(solarScene->getReceivers()[0]);
    RectGrid *rectGrid = dynamic_cast<RectGrid *>(solarScene->getGrid0s()[2]);
    float factor = 1.0f;
    float3 *d_subHeliostat_vertexes = nullptr;
    int start_heliostat_id = rectGrid->getStartHeliostatPosition();
    int end_heliostat_id = start_heliostat_id + rectGrid->getNumberOfHeliostats();
    QMCRTracer.setFlatRectangleHeliostatVertexes(d_subHeliostat_vertexes, solarScene->getHeliostats(),
                                                 start_heliostat_id, end_heliostat_id);

    // Heliostat 4 without shadow(index = 3)
    HeliostatArgument heliostatArgument0 = QMCRTracer.generateHeliostatArgument(solarScene, 3);
    CylinderReceiverRectGridRayTracing(sunrayArgument, cylinderReceiver, rectGrid, heliostatArgument0,
                                       d_subHeliostat_vertexes, factor);

    int2 resolution = cylinderReceiver->getResolution();
    vector<float> image = deviceArray2vector(cylinderReceiver->getDeviceImage(), resolution.y * resolution.x);
    std::cout<<"\nHeliostat 1 without shadowing - (r: "<<resolution.y<<", c:"<<resolution.x<<")"<<std::endl;
    print(image, resolution);
}