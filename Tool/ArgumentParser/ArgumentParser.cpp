//
// Created by dxt on 18-11-3.
//

#include <fstream>
#include <getopt.h>
#include <cstring>

#include "ArgumentParser.h"

void ArgumentParser::initialize() {
    configuration_path =
            "/home/dxt/CLionProjects/SolarEnergyRayTracing/InputFiles/example/example_configuration.json";
    scene_path =
            "/home/dxt/CLionProjects/SolarEnergyRayTracing/InputFiles/example/example_scene.scn";

}

bool ArgumentParser::check_valid(std::string file_path, std::string suffix) {
    // 1. Suffix exist at the end of file_path
    if (file_path.substr(file_path.size() - suffix.size(), suffix.size()) != suffix) {
        printf("The path of file '%s' does not consist on '%s' suffix",
               file_path.c_str(), suffix.c_str());
        return false;
    }

    // 2. The path of the file exists to a real file
    std::ifstream f(file_path.c_str());
    return f.good();
}

bool ArgumentParser::parser(int argc, char **argv) {
    initialize();

    static struct option long_options[] = {
            {"configuration_path", required_argument, 0, 'c'},
            {"scene_path",         required_argument, 0, 's'},
            {0, 0,                                    0, 0}
    };

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "c:s:", long_options, &option_index)) != -1) {
        if (c == '?') {
            // unknown arguments
            return false;
        }

        if (c == 'c' ||
            !std::strcmp(long_options[option_index].name, "configuration_path")) {
            configuration_path = optarg;
            printf("\noption -c/configuration_path with value '%s'\n", optarg);
        } else if (c == 's' ||
                   !std::strcmp(long_options[option_index].name, "scene_path")) {
            scene_path = optarg;
            printf("\noption -s/scene_path with value '%s'\n", optarg);
        }
    }

    bool ans = true;
    if(!check_valid(configuration_path, ".json")) {
        printf("cannot find file '%s'\n", configuration_path.c_str());
        ans = false;
    }
    if(!check_valid(scene_path, ".scn")) {
        printf("cannot find file '%s'\n", scene_path.c_str());
        ans = false;
    }
    return ans;
}

const std::string &ArgumentParser::getConfigurationPath() const {
    return configuration_path;
}

bool ArgumentParser::setConfigurationPath(const std::string &configuration_path) {
    if (check_valid(configuration_path, ".json")) {
        ArgumentParser::configuration_path = configuration_path;
        return true;
    }
    return false;
}

const std::string &ArgumentParser::getScenePath() const {
    return scene_path;
}

bool ArgumentParser::setScenePath(const std::string &scene_path) {
    if (check_valid(scene_path, ".scn")) {
        ArgumentParser::scene_path = scene_path;
        return true;
    }
    return false;
}
