//
// Created by dxt on 18-11-9.
//

#ifndef SOLARENERGYRAYTRACING_SCENEPROCESSER_H
#define SOLARENERGYRAYTRACING_SCENEPROCESSER_H

#include "SceneConfiguration.h"
#include "SolarScene.h"

class SceneProcessor {
public:
    SceneProcessor() : sceneConfiguration(nullptr) {}

    SceneProcessor(SceneConfiguration *sceneConfigure) : sceneConfiguration(sceneConfigure) {}

    bool processScene(SolarScene *solarScene);

    bool setGridReceiverHeliostatContent(std::vector<Grid *> &grids, std::vector<Receiver *> &receivers,
                                         std::vector<Heliostat *> &heliostats);

    bool set_sunray_content(Sunray &sunray);
    SceneConfiguration *getSceneConfiguration() const;
    void setSceneConfiguration(SceneConfiguration *sceneConfiguration);

private:
    bool set_perturbation(Sunray &sunray);
    bool set_samplelights(Sunray &sunray);

    SceneConfiguration *sceneConfiguration;
};


#endif //SOLARENERGYRAYTRACING_SCENEPROCESSER_H
