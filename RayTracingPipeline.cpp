//
// Created by dxt on 18-11-26.
//
#include <chrono>

#include "RayTracingPipeline.h"

#include "QuasiMonteCarloRayTracing/QuasiMonteCarloRayTracer.h"
#include "Scene/SceneProcessor.h"
#include "Scene/SceneLoader.h"
#include "Tool/ArgumentParser/ArgumentParser.h"
#include "Tool/ImageSaver/ImageSaver.h"
#include "Scene/DataStructure/Receiver/Receiver.cuh"
#include "Scene/DataStructure/Heliostat/Heliostat.cuh"
#include "Scene/DataStructure/Grid/Grid.cuh"
#include "global_function.cuh"

void RayTracingPipeline(int argc, char *argv[]) {
    ArgumentParser *argumentParser = new ArgumentParser();
    argumentParser->parser(argc, argv);

    SolarScene *solarScene = SolarScene::GetInstance();
    SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
    sceneConfiguration->loadConfiguration(argumentParser->getConfigurationPath());

    SceneLoader sceneLoader;
    sceneLoader.SceneFileRead(solarScene, argumentParser->getScenePath());

    SceneProcessor sceneProcessor(sceneConfiguration);
    sceneProcessor.processScene(solarScene);

    int heliostat_id = 0;
    QuasiMonteCarloRayTracer QMCRTracer;

    auto start = std::chrono::high_resolution_clock::now();
    QMCRTracer.rayTracing(solarScene, heliostat_id);
    auto end = std::chrono::high_resolution_clock::now();
    long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "\nRay tracing time elapse is " << elapsed << " microseconds.";

    Heliostat *heliostat = solarScene->getHeliostats()[heliostat_id];
    Grid *grid = solarScene->getGrid0s()[heliostat->getBelongingGridId()];
    int receiver_id = grid->getBelongingReceiverIndex();
    Receiver *receiver = solarScene->getReceivers()[receiver_id];
    int2 resolution = receiver->getResolution();

    start = std::chrono::high_resolution_clock::now();
    float *h_array = nullptr;
    float *d_array = receiver->getDeviceImage();
    global_func::gpu2cpu(h_array, d_array, resolution.x * resolution.y);
    ImageSaver::saveText("OutputFiles/" + std::to_string(heliostat_id) + ".txt", resolution.y, resolution.x, h_array);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "\nSaving time elapse is " << elapsed << " microseconds.";
}


