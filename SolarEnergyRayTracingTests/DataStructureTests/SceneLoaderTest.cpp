//
// Created by dxt on 18-11-7.
//

//#include "SceneLoader.cuh"
#include "SceneLoader.h"
#include "gtest/gtest.h"

class SceneLoaderFixture : public ::testing::Test {
protected:
    void SetUp() {
        solarScene = SolarScene::GetInstance();
        sceneLoader = new SceneLoader();
    }

    void TearDown() {
        delete(sceneLoader);
    }

public:
    SceneLoaderFixture():solarScene(nullptr){}

    SolarScene *solarScene;
    SceneLoader *sceneLoader;
};

TEST_F(SceneLoaderFixture, goodExample) {
    std::string goodExamplePath = "test_file/test_scene_good.scn";
    EXPECT_TRUE(sceneLoader->SceneFileRead(solarScene, goodExamplePath));
}

TEST_F(SceneLoaderFixture, badExampleUnknownFields) {
    std::string badExamplePath = "test_file/test_scene_bad_unknownFields.scn";
    EXPECT_FALSE(sceneLoader->SceneFileRead(solarScene, badExamplePath));
}

TEST_F(SceneLoaderFixture, badExampleNonRegularExpressionFormat) {
    std::string badExamplePath = "test_file/test_scene_bad_nonRE.scn";
    EXPECT_FALSE(sceneLoader->SceneFileRead(solarScene, badExamplePath));
}