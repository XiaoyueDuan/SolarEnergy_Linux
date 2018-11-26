//
// Created by dxt on 18-11-20.
//

#include <SceneLoader.h>
#include <SceneProcessor.h>
#include <QuasiMonteCarloRayTracer.h>
#include "NewTypeHeliostatForTest.h"
#include "Grid/rectGridDDA.cuh"
#include "gtest/gtest.h"

class RectGridDDAFixture : public ::testing::Test {
private:
    void loadAndProcessScene() {
        solarScene = SolarScene::GetInstance();

        std::string configuration_path = "test_file/test_configuration.json";

        SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
        sceneConfiguration->loadConfiguration(configuration_path);

        std::string scene_path = "test_file/test_scene.scn";
        SceneLoader sceneLoader;
        sceneLoader.SceneFileRead(solarScene, scene_path);

        Heliostat *h2 = new NewTypeHeliostatForTest();
        h2->setPosition(make_float3(0.0f, 0.0f, 12.0f));
        h2->setSize(make_float3(5.0f, 0.1f, 4.0f));
        h2->setRowAndColumn(make_int2(3, 2));
        h2->setGap(make_float2(1.0f, 0.5f));

        Heliostat *h_tmp = solarScene->getHeliostats()[1];
        solarScene->getHeliostats()[1] = h2;
        delete (h_tmp);

        /**
         * TODO: For unknown reasons, when run the tests from class entrance, the second case will fail. The failure is
         * caused by nullptr sunray in solarScene. But why sunray is nullptr, I just do not understand.
         *
         * Thus, run the tests in this file one by one.
         * */
        SceneProcessor sceneProcessor(sceneConfiguration);
        sceneProcessor.processScene(solarScene);

        // clean up
        h2 = nullptr;
        h_tmp = nullptr;
    }

    void constructRectGrid() {
        rectGrid = dynamic_cast<RectGrid *>(solarScene->getGrid0s()[0]);
        int *d_grid_helio_match = rectGrid->getDeviceGridHeliostatMatch();
        int *h_grid_heliostat_match = nullptr;
        global_func::gpu2cpu(h_grid_heliostat_match, d_grid_helio_match, rectGrid->getNumberOfGridHeliostatMatch());

        int *d_grid_helio_index = rectGrid->getDeviceGridHelioIndex();
        int *h_grid_helio_index = nullptr;
        global_func::gpu2cpu(h_grid_helio_index, d_grid_helio_index,
                             rectGrid->getGridNumber().x * rectGrid->getGridNumber().y * rectGrid->getGridNumber().z + 1);
        rectGrid->CClear();
        rectGrid->setDeviceGridHelioIndex(h_grid_helio_index);
        rectGrid->setDeviceGridHeliostatMatch(h_grid_heliostat_match);
    }

    void setOriginsAndDirections() {
        direction = make_float3(0.0f, 0.0f, -1.0f); // for all rays

        // Heliostat 1
        subHeliostatId1 = 0;
        origin1 = make_float3(-1.5f, 1.0f, 7.9f);
        grid_id1 = 3;
        Heliostat *h1 = solarScene->getHeliostats()[0];
        numberOfSubHeliostat1 = h1->getSubHelioSize();

        // Heliostat 2
        subHeliostatId2 = 1;
        origin21 = make_float3(-1.0f, 0.0f, 11.95f);    //  1. shadowed
        origin22 = make_float3(-2.2f, 0.0f, 11.95f);    //  2. unshadowed
        grid_id2 = 4;
        Heliostat *h2 = solarScene->getHeliostats()[1];
        numberOfSubHeliostat2 = h2->getSubHelioSize();

        // Heliostat 3
        subHeliostatId3 = 7;
        origin31 = make_float3(-2.4f, -1.4f, 19.95f);     //  1. shadowed
        origin32 = make_float3(-3.4f, -3.9f, 19.95f);;    //  2. unshadowed
        grid_id3 = 4;
        Heliostat *h3 = solarScene->getHeliostats()[2];
        numberOfSubHeliostat3 = h3->getSubHelioSize();

        // clean up
        h1 = nullptr;
        h2 = nullptr;
        h3 = nullptr;
    }

    void setSubHeliostatVertexes() {
        float3 *d_heliostat_vertexes = nullptr;
        int heliostatVertexesSize = QMCRTracer.setFlatRectangleHeliostatVertexes(d_heliostat_vertexes,
                solarScene->getHeliostats(), rectGrid->getStartHeliostatPosition(),
                rectGrid->getStartHeliostatPosition() + rectGrid->getNumberOfHeliostats());
        global_func::gpu2cpu(h_heliostat_vertexs, d_heliostat_vertexes, heliostatVertexesSize);

        // clean up
        checkCudaErrors(cudaFree(d_heliostat_vertexes));
        d_heliostat_vertexes = nullptr;
    }

protected:
    void SetUp() {
        loadAndProcessScene();
        constructRectGrid();
        setOriginsAndDirections();
        setSubHeliostatVertexes();
    }

    void TearDown() {
        delete[] rectGrid->getDeviceGridHeliostatMatch();
        delete[] rectGrid->getDeviceGridHelioIndex();
        rectGrid->setDeviceGridHeliostatMatch(nullptr);
        rectGrid->setDeviceGridHelioIndex(nullptr);
        solarScene->clear();

        delete[] h_heliostat_vertexs;
        h_heliostat_vertexs = nullptr;
    }

public:
    RectGridDDAFixture() : solarScene(nullptr), h_heliostat_vertexs(nullptr), rectGrid(nullptr) {}

    SolarScene *solarScene;
    float3 *h_heliostat_vertexs;    // The value is in host in the test
    RectGrid *rectGrid;

    float3 direction; // for all rays

    // Heliostat 1
    int subHeliostatId1;
    float3 origin1;
    int grid_id1;
    int numberOfSubHeliostat1;

    // Heliostat 2
    int subHeliostatId2;
    float3 origin21;    //  1. shadowed
    float3 origin22;    //  2. unshadowed
    int grid_id2;
    int numberOfSubHeliostat2;

    // Heliostat 3
    int subHeliostatId3;
    float3 origin31;    //  1. shadowed
    float3 origin32;    //  2. unshadowed
    int grid_id3;
    int numberOfSubHeliostat3;

    QuasiMonteCarloRayTracer QMCRTracer;
};

TEST_F(RectGridDDAFixture, intersect) {
    // Ray from Heliostat 1
    EXPECT_FALSE(
            rectGridDDA::intersect(origin1, direction, h_heliostat_vertexs, rectGrid->getDeviceGridHeliostatMatch(),
                                   rectGrid->getDeviceGridHelioIndex()[grid_id1],
                                   rectGrid->getDeviceGridHelioIndex()[grid_id1 + 1],
                                   subHeliostatId1, numberOfSubHeliostat1));

    // Ray from Heliostat 2
    //  1. shadowed
    EXPECT_TRUE(
            rectGridDDA::intersect(origin21, direction, h_heliostat_vertexs, rectGrid->getDeviceGridHeliostatMatch(),
                                   rectGrid->getDeviceGridHelioIndex()[grid_id2],
                                   rectGrid->getDeviceGridHelioIndex()[grid_id2 + 1],
                                   subHeliostatId2, numberOfSubHeliostat2));
    //  2. unshadowed
    EXPECT_FALSE(
            rectGridDDA::intersect(origin22, direction, h_heliostat_vertexs, rectGrid->getDeviceGridHeliostatMatch(),
                                   rectGrid->getDeviceGridHelioIndex()[grid_id2],
                                   rectGrid->getDeviceGridHelioIndex()[grid_id2 + 1],
                                   subHeliostatId2, numberOfSubHeliostat2));

    // Ray from Heliostat 3(7th-subHeliostat)
    //  1. shadowed
    EXPECT_TRUE(
            rectGridDDA::intersect(origin31, direction, h_heliostat_vertexs, rectGrid->getDeviceGridHeliostatMatch(),
                                   rectGrid->getDeviceGridHelioIndex()[grid_id3],
                                   rectGrid->getDeviceGridHelioIndex()[grid_id3 + 1],
                                   subHeliostatId3, numberOfSubHeliostat3));
    //  2. unshadowed
    EXPECT_FALSE(
            rectGridDDA::intersect(origin32, direction, h_heliostat_vertexs, rectGrid->getDeviceGridHeliostatMatch(),
                                   rectGrid->getDeviceGridHelioIndex()[grid_id3],
                                   rectGrid->getDeviceGridHelioIndex()[grid_id3 + 1],
                                   subHeliostatId3, numberOfSubHeliostat3));
}

TEST_F(RectGridDDAFixture, collisionShadowAndUnshadow) {
    // Ray from Heliostat 1
    HeliostatArgument heliostatArgument1 = QMCRTracer.generateHeliostatArgument(solarScene, 0);
    EXPECT_FALSE(rectGridDDA::collision(origin1, direction, *rectGrid, h_heliostat_vertexs, heliostatArgument1));

    // Ray from Heliostat 2
    //  1. shadowed
    HeliostatArgument heliostatArgument2 = QMCRTracer.generateHeliostatArgument(solarScene, 1);
    EXPECT_TRUE(rectGridDDA::collision(origin21, direction, *rectGrid, h_heliostat_vertexs, heliostatArgument2));
    //  2. unshadowed
    EXPECT_FALSE(rectGridDDA::collision(origin22, direction, *rectGrid, h_heliostat_vertexs, heliostatArgument2));

    // Ray from Heliostat 3
    //  1. shadowed
    HeliostatArgument heliostatArgument3 = QMCRTracer.generateHeliostatArgument(solarScene, 2);
    EXPECT_TRUE(rectGridDDA::collision(origin31, direction, *rectGrid, h_heliostat_vertexs, heliostatArgument3));
    //  2. unshadowed
    EXPECT_FALSE(rectGridDDA::collision(origin32, direction, *rectGrid, h_heliostat_vertexs, heliostatArgument3));
}

TEST_F(RectGridDDAFixture, collisionRayDirection) {
    float3 dir;
    float3 origin = make_float3(-10.0f, 0.0f, 4.0f);
    float3 grid_interval = rectGrid->getInterval();
    HeliostatArgument heliostatArgument3 = QMCRTracer.generateHeliostatArgument(solarScene, 2);
    // 1. ... < ... < ...
    // direction.x < direction.y < direction.z
    dir = normalize(make_float3(1.0f, 2.0f, 3.0f) * grid_interval);
    rectGridDDA::collision(origin + dir * 0.1, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);
    // direction.x < direction.z < direction.y
    dir = normalize(make_float3(1.0f, 3.0f, 2.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);
    // direction.y < direction.x < direction.z
    dir = normalize(make_float3(2.0f, 1.0f, 3.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);
    // direction.y < direction.z < direction.x
    dir = normalize(make_float3(3.0f, 1.0f, 2.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);
    // direction.z < direction.x < direction.y
    dir = normalize(make_float3(2.0f, 3.0f, 1.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);
    // direction.z < direction.y < direction.x
    dir = normalize(make_float3(3.0f, 2.0f, 1.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);

    // 2. ... < ... = ...
    // direction.x < direction.y = direction.z
    dir = normalize(make_float3(1.0f, 2.0f, 2.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);
    // direction.y < direction.x = direction.z
    dir = normalize(make_float3(2.0f, 1.0f, 2.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);
    // direction.z < direction.x = direction.y
    dir = normalize(make_float3(2.0f, 2.0f, 1.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);

    // 3. ... = ... < ...
    // direction.x = direction.y < direction.z
    dir = normalize(make_float3(1.0f, 1.0f, 2.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);
    // direction.x = direction.z < direction.y
    dir = normalize(make_float3(1.0f, 2.0f, 1.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);
    // direction.y = direction.z < direction.x
    dir = normalize(make_float3(2.0f, 1.0f, 1.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);

    // 4. ... = ... = ...
    // direction.x = direction.y = direction.z
    dir = normalize(make_float3(1.0f, 1.0f, 1.0f) * grid_interval);
    rectGridDDA::collision(origin, dir, *rectGrid, h_heliostat_vertexs, heliostatArgument3);
}