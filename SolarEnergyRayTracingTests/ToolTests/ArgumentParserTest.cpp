//
// Created by dxt on 18-11-5.
//

#include "ArgumentParser.h"
#include "gtest/gtest.h"

class ArgumentParserFixture : public ::testing::Test {
protected:
    void SetUp() {
        argumentParser = new ArgumentParser();
    }

    void TearDown() {
        delete (argumentParser);
        argumentParser = nullptr;
    }

public:
    ArgumentParserFixture() :
            argumentParser(nullptr) {}

    ArgumentParser *argumentParser;
    char executable_file_name[21] = "executable_file_name";
    char good_configuration_path_option[21] = "--configuration_path";
    char good_scene_path_option[13] = "--scene_path";
    char unknown_option[17] = "--unknown_option";
};

TEST_F(ArgumentParserFixture, parserGoodExample) {
    /**
     *  Noted: the working directory are set as:
     *      /home/dxt/CLionProjects/SolarEnergyRayTracing/SolarEnergyRayTracingTests/ToolTests
     *      (setting method: "the dialog in upper right" -> "Edit Configurations...")
     * */
    char correctConfigurationPath[] = "test_file/test_configuration.json";
    char correctScenePath[] = "test_file/test_scene.scn";

    char *argv[] = {
            executable_file_name,
            good_configuration_path_option,
            correctConfigurationPath,
            good_scene_path_option,
            correctScenePath
    };
    int argc = sizeof(argv) / sizeof(argv[0]);

    EXPECT_TRUE(argumentParser->parser(argc, argv));
}

TEST_F(ArgumentParserFixture, parserGoodExampleWithEmptyOptions) {
    char *argv[] = {
            executable_file_name
    };
    int argc = sizeof(argv) / sizeof(argv[0]);

    EXPECT_TRUE(argumentParser->parser(argc, argv));
}

TEST_F(ArgumentParserFixture, parserGoodExampleWithPartialOptions) {
    char correctConfigurationPath[] = "test_file/test_configuration.json";

    char *argv[] = {
            executable_file_name,
            good_configuration_path_option,
            correctConfigurationPath,
    };
    int argc = sizeof(argv) / sizeof(argv[0]);

    EXPECT_TRUE(argumentParser->parser(argc, argv));
}

TEST_F(ArgumentParserFixture, parserBadExample_nonExistFilePath) {
    char nonExistConfigurationPath[] = "test_file/nonExist_configuration.json";
    char nonExistScenePath[] = "test_file/nonExist_scene.scn";

    char *argv[] = {
            executable_file_name,
            good_configuration_path_option,
            nonExistConfigurationPath,
            good_scene_path_option,
            nonExistScenePath
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    /**
     * TODO: argumentParser->parser(argc, argv) can only be called once.
     *       for example, in this test:
     *       // codes
     *       {
     *          bool ans1 = argumentParser->parser(argc, argv); // ans = false
     *          bool ans2 = argumentParser->parser(argc, argv); // ans = true
     *       }
     * */
    EXPECT_ANY_THROW(argumentParser->parser(argc, argv));
}

TEST_F(ArgumentParserFixture, parserBadExample_nonMatchSuffix) {
    char nonMatchSuffixConfigurationPath[] = "example_configuration";
    char nonMatchSuffixScenePath[] = "example_scene";

    char *argv[] = {
            executable_file_name,
            good_configuration_path_option,
            nonMatchSuffixConfigurationPath,
            good_scene_path_option,
            nonMatchSuffixScenePath
    };
    int argc = sizeof(argv) / sizeof(argv[0]);

    EXPECT_ANY_THROW(argumentParser->parser(argc, argv));
}

TEST_F(ArgumentParserFixture, parserBadExample_unknownOption) {
    char *argv[] = {
            executable_file_name,
            unknown_option
    };
    int argc = sizeof(argv) / sizeof(argv[0]);

    EXPECT_FALSE(argumentParser->parser(argc, argv));
}