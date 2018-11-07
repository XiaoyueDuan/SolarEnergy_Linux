//
// Created by dxt on 18-11-3.
//

#ifndef SOLARENERGYRAYTRACING_ARGUMENTPARSER_H
#define SOLARENERGYRAYTRACING_ARGUMENTPARSER_H

#include <string>

class ArgumentParser {
private:
    std::string configuration_path;
    std::string scene_path;

    void initialize();

    bool check_valid(std::string path, std::string suffix);

public:
    bool parser(int argc, char **argv);

    const std::string &getConfigurationPath() const;

    bool setConfigurationPath(const std::string &configuration_path);

    const std::string &getScenePath() const;

    bool setScenePath(const std::string &scene_path);

};

#endif //SOLARENERGYRAYTRACING_ARGUMENTPARSER_H
