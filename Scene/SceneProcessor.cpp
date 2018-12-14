//
// Created by dxt on 18-11-9.
//

#include <stdexcept>
#include "SceneProcessor.h"

bool SceneProcessor::processScene(SolarScene *solarScene) {
    return set_sunray_content(*solarScene->getSunray()) &&
           setGridReceiverHeliostatContent(solarScene->getGrid0s(), solarScene->getReceivers(),
                                           solarScene->getHeliostats());
}

bool SceneProcessor::setGridReceiverHeliostatContent(std::vector<Grid *> &grids, std::vector<Receiver *> &receivers,
                                                     std::vector<Heliostat *> &heliostats) {
    if (!sceneConfiguration) {
        throw std::runtime_error("No scene configuration. Please load it first before process scene.");
    }

    int pixels_per_meter_for_receiver = int(1.0f / sceneConfiguration->getReceiver_pixel_length());
    float heliostat_pixel_length = sceneConfiguration->getHelio_pixel_length();
    float3 sun_direction = sceneConfiguration->getSun_dir();

    for (Grid *grid : grids) {
        grid->Cinit();
        grid->CGridHelioMatch(heliostats);

        Receiver *receiver = receivers[grid->getBelongingReceiverIndex()];
        receiver->CInit(pixels_per_meter_for_receiver);

        for (int i = 0; i < grid->getNumberOfHeliostats(); ++i) {
            int id = i + grid->getStartHeliostatPosition();
            float3 focus_center = receiver->getFocusCenter(heliostats[id]->getPosition());

            heliostats[id]->setPixelLength(heliostat_pixel_length);
            heliostats[id]->CSetNormalAndRotate(focus_center, sun_direction);
        }
    }
    return true;
}

bool SceneProcessor::set_sunray_content(Sunray &sunray) {
    if (!sceneConfiguration) {
        throw std::runtime_error("No scene configuration. Please load it first before process scene.");
    }

    sunray.setSunDirection(sceneConfiguration->getSun_dir());
    sunray.setCSR(sceneConfiguration->getCsr());
    sunray.setDNI(sceneConfiguration->getDni());
    sunray.setNumOfSunshapeGroups(sceneConfiguration->getNum_sunshape_groups());
    sunray.setNumOfSunshapeLightsPerGroup(sceneConfiguration->getNum_sunshape_lights_per_group());
    sunray.setReflectiveRate(sceneConfiguration->getReflected_rate());

    return set_perturbation(sunray) && set_samplelights(sunray);
}

SceneConfiguration *SceneProcessor::getSceneConfiguration() const {
    return sceneConfiguration;
}

void SceneProcessor::setSceneConfiguration(SceneConfiguration *sceneConfiguration) {
    SceneProcessor::sceneConfiguration = sceneConfiguration;
}












