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
    char good_output_path_option[14] = "--output_path";
    char good_heliostat_index_option[25] = "--heliostat_indexes_path";
    char unknown_option[17] = "--unknown_option";

    char good_brief_configuration_path_option[3] = "-c";
    char good_brief_scene_path_option[3] = "-s";
    char good_brief_output_path_option[3] = "-o";
    char good_brief_heliostat_index_option[3] = "-h";
    char unknown_brief_option[3] = "-u";
};

TEST_F(ArgumentParserFixture, parserGoodExample) {
    /**
     *  Noted: the working directory are set as:
     *      /home/dxt/CLionProjects/SolarEnergyRayTracing/SolarEnergyRayTracingTests/ToolTests
     *      (setting method: "the dialog in upper right" -> "Edit Configurations...")
     * */
    char correctConfigurationPath[] = "test_file/test_configuration.json";
    char correctScenePath[] = "test_file/test_scene.scn";
    char correctOutputPath[] = "test_output/";
    char correctHeliostatIndexPath[] = "test_file/task_heliostats_index.txt";

    std::cout<<"\n1. With all long arguments";
    char *argv1[] = {
            executable_file_name,
            good_configuration_path_option, correctConfigurationPath,
            good_scene_path_option, correctScenePath,
            good_output_path_option, correctOutputPath,
            good_heliostat_index_option, correctHeliostatIndexPath
    };
    int argc = sizeof(argv1) / sizeof(argv1[0]);
    EXPECT_TRUE(argumentParser->parser(argc, argv1));

    std::cout<<"\n\n2. With all short arguments";
    char *argv2[] = {
            executable_file_name,
            good_brief_configuration_path_option, correctConfigurationPath,
            good_brief_scene_path_option, correctScenePath,
            good_brief_output_path_option, correctOutputPath,
            good_brief_heliostat_index_option, correctHeliostatIndexPath
    };
    argc = sizeof(argv2)/sizeof(argv2[0]);
    EXPECT_TRUE(argumentParser->parser(argc, argv2));

    std::cout<<"\n\n3. With the mixture of short and long arguments";
    char *argv3[] = {
            executable_file_name,
            good_brief_configuration_path_option, correctConfigurationPath,
            good_scene_path_option, correctScenePath,
            good_output_path_option, correctOutputPath,
            good_brief_heliostat_index_option, correctHeliostatIndexPath
    };
    argc = sizeof(argv3)/sizeof(argv3[0]);
    EXPECT_TRUE(argumentParser->parser(argc, argv3));
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
    char correctScenePath[] = "test_file/test_scene.scn";

    std::cout<<"\n1. With partial of long arguments";
    char *argv1[] = {
            executable_file_name,
            good_configuration_path_option, correctConfigurationPath,
    };
    int argc = sizeof(argv1) / sizeof(argv1[0]);
    EXPECT_TRUE(argumentParser->parser(argc, argv1));

    std::cout<<"\n\n2. With partial of short arguments";
    char *argv2[] = {
            executable_file_name,
            good_brief_configuration_path_option, correctConfigurationPath,
    };
    argc = sizeof(argv2) / sizeof(argv2[0]);
    EXPECT_TRUE(argumentParser->parser(argc, argv2));

    std::cout<<"\n\n3. With partial of mixture of short and long arguments";
    char *argv3[] = {
            executable_file_name,
            good_brief_configuration_path_option, correctConfigurationPath,
            good_scene_path_option, correctScenePath
    };
    argc = sizeof(argv3) / sizeof(argv3[0]);
    EXPECT_TRUE(argumentParser->parser(argc, argv3));
}

TEST_F(ArgumentParserFixture, parserBadExample_nonExistFilePath) {
    char nonExistConfigurationPath[] = "test_file/nonExist_configuration.json";
    char nonExistScenePath[] = "test_file/nonExist_scene.scn";
    char nonHeliostatIndexPath[] = "test_file/nonExist_heliostat_index.txt";
    char nonOutputDirectoryPath[] = "non_exist_dir/";

    std::cout<<"\n1. Non-existed configuration";
    char *argv1[] = {
            executable_file_name,
            good_configuration_path_option,
            nonExistConfigurationPath
    };
    int argc = sizeof(argv1) / sizeof(argv1[0]);
    EXPECT_ANY_THROW(argumentParser->parser(argc, argv1));

    std::cout<<"\n\n2. Non-existed scene";
    char *argv2[] = {
            executable_file_name,
            good_scene_path_option,
            nonExistScenePath
    };
    argc = sizeof(argv2) / sizeof(argv2[0]);
    EXPECT_ANY_THROW(argumentParser->parser(argc, argv2));

    std::cout<<"\n\n3. Non-existed heliostat index input file";
    char *argv3[] = {
            executable_file_name,
            good_heliostat_index_option,
            nonHeliostatIndexPath
    };
    argc = sizeof(argv3) / sizeof(argv3[0]);
    EXPECT_ANY_THROW(argumentParser->parser(argc, argv3));

    std::cout<<"\n\n4. Non-existed output directory";
    char *argv4[] = {
            executable_file_name,
            good_output_path_option,
            nonOutputDirectoryPath
    };
    argc = sizeof(argv4) / sizeof(argv4[0]);
    EXPECT_ANY_THROW(argumentParser->parser(argc, argv4));
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