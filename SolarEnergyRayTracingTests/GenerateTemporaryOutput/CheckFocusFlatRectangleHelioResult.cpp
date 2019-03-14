//
// Created by dxt on 19-3-14.
//

#include "RayTracingPipeline.h"
#include "gtest/gtest.h"

TEST(CheckFocusFlatRectangleHelioResult, rayTracingResult) {
    //  1. Set argument
    char configuration_path[] = "CheckFocusFlatRectangleHelioResult/input_files/configure.json";
    char scene_path[] = "CheckFocusFlatRectangleHelioResult/input_files/test_scene_good_for_focusFlatRectangleHelio.scn";
    char heliostat_index_path[] = "CheckFocusFlatRectangleHelioResult/input_files/heliostat_index.txt";
    char output_path[] ="CheckFocusFlatRectangleHelioResult/output_files/";

    char execuation_file_name[] = "execuation_file_name";
    char configuration_path_option[] = "-c";
    char scene_path_option[] = "-s";
    char output_path_option[] = "-o";
    char heliostat_index_option[] = "-h";

    char *argv[] = {
            execuation_file_name,
            configuration_path_option, configuration_path,
            scene_path_option, scene_path,
            output_path_option, output_path,
            heliostat_index_option, heliostat_index_path
    };

    int argc = sizeof(argv) / sizeof(argv[0]);
    RayTracingPipeline::rayTracing(argc, argv);
}
