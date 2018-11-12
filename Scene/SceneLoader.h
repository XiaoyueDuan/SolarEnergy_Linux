//
// Created by dxt on 18-11-6.
//

#ifndef SOLARENERGYRAYTRACING_SCENELOADER_H
#define SOLARENERGYRAYTRACING_SCENELOADER_H

#include <sstream>

#include "SolarScene.h"
#include "FileLoader/RegularExpressionTree.h"

class SceneLoader {
private:
    TreeNode *current_status;
    SceneRegularExpressionTree sceneRETree_;
    void add_ground(SolarScene *solarScene, std::istream &stringstream);
    void add_receiver(SolarScene *solarScene, std::istream &stringstream);
    int add_grid(SolarScene *solarScene,
            std::istream &stringstream, int receiver_index, int heliostat_start_index);
    void add_heliostat(SolarScene *solarScene, std::istream &stringstream,
            int type, float2 gap, int2 matrix);
    void checkScene(SolarScene *solarScene);
public:
    SceneLoader():current_status(nullptr) {}

    bool SceneFileRead(SolarScene *solarScene, std::string filepath);
};

#endif //SOLARENERGYRAYTRACING_SCENELOADER_H
