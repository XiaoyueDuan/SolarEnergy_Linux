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
    std::string heliostat_index_load_path;
    std::string output_path;    // such as "$HOME/dir/"

    void initialize();

    void check_valid_file(std::string path, std::string suffix);
    void check_valid_directory(std::string path);

public:
    bool parser(int argc, char **argv);

    const std::string &getConfigurationPath() const;
    bool setConfigurationPath(const std::string &configuration_path);

    const std::string &getScenePath() const;
    bool setScenePath(const std::string &scene_path);

    const std::string &getHeliostatIndexLoadPath() const;
    void setHeliostatIndexLoadPath(const std::string &heliostat_index_load_path);

    const std::string &getOutputPath() const;
    void setOutputPath(const std::string &output_path);

};

#endif //SOLARENERGYRAYTRACING_ARGUMENTPARSER_H
