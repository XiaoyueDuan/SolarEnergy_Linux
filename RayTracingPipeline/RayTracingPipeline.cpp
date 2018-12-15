//
// Created by dxt on 18-12-15.
//
#include <chrono>

#include "TaskLoader.h"
#include "QuasiMonteCarloRayTracer.h"
#include "SceneLoader.h"
#include "SceneProcessor.h"
#include "ImageSaver/ImageSaver.h"
#include "ArgumentParser/ArgumentParser.h"
#include "global_function.cuh"

#include "RayTracingPipeline.h"

void RayTracingPipeline::rayTracing(int argc, char *argv[]) {
    // 1. Pass argument
    //  1) solar scene file path;
    //  2) configuration file path;
    //  3) the file path saving the indexes of heliostats which will be ray traced
    //  4) the output path
    std::cout << "1. Start loading arguments...";
    ArgumentParser *argumentParser = new ArgumentParser();
    argumentParser->parser(argc, argv);

    // 2. Initialize solar scene
    std::cout << "\n2. Initialize solar scene..." << std::endl;
    //  2.1 configuration
    std::cout << "  2.1 Load configuration from '" << argumentParser->getConfigurationPath() << "'." << std::endl;
    SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
    sceneConfiguration->loadConfiguration(argumentParser->getConfigurationPath());

    //  2.2 load scene
    std::cout << "  2.2 Load scene file from '" << argumentParser->getScenePath() << "'." << std::endl;
    SceneLoader sceneLoader;
    SolarScene *solarScene = SolarScene::GetInstance();
    sceneLoader.SceneFileRead(solarScene, argumentParser->getScenePath());

    //  2.3 process scene
    std::cout << "  2.3 Process scene." << std::endl;
    SceneProcessor sceneProcessor(sceneConfiguration);
    sceneProcessor.processScene(solarScene);

    //  2.4 load heliostats indexes
    std::cout << "  2.4 Load heliostats indexes from '" << argumentParser->getHeliostatIndexLoadPath() << "'." <<
              std::endl;
    TaskLoader taskLoader;
    taskLoader.loadRayTracingHeliostatIndex(argumentParser->getHeliostatIndexLoadPath(), *solarScene);

    // 3. Ray tracing (could be parallel)
    std::cout << "3. Start ray tracing..." << std::endl;
    QuasiMonteCarloRayTracer QMCRTracer;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;
    long long elapsed;

    for (int heliostat_index : taskLoader.getHeliostatIndexesArray()) {
        try {
            // Count the time
            start = std::chrono::high_resolution_clock::now();

            QMCRTracer.rayTracing(solarScene, heliostat_index);

            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            std::cout << "  No." << heliostat_index << " heliostat takes " << elapsed << " microseconds."
                      << std::endl;
        } catch (exception e) {
            std::cerr << "  Failure in No." << heliostat_index << " heliostat ray tracing." << std::endl;
        }
    }

    // 4. Save results
    std::cout << "\n4. Save results in '" << argumentParser->getOutputPath() << "' directory." << std::endl;
    for (int receiver_index : taskLoader.getReceiverIndexesArray()) {
        std::cout << "  Saving No." << receiver_index << " receiver." << std::endl;
        saveReceiverResult(solarScene->getReceivers()[receiver_index],
                           argumentParser->getOutputPath() + std::to_string(receiver_index) + ".txt");
    }

    // Finally, clear the scene
    solarScene->clear();
}

void RayTracingPipeline::saveReceiverResult(Receiver *receiver, std::string pathAndName) {
    int2 resolution = receiver->getResolution();
    float *h_array = nullptr;
    float *d_array = receiver->getDeviceImage();
    global_func::gpu2cpu(h_array, d_array, resolution.x * resolution.y);
    std::cout << "  resolution: (" << resolution.y << ", " << resolution.x << "). ";
    ImageSaver::saveText(pathAndName, resolution.y, resolution.x, h_array);

    // clear
    delete (h_array);
    h_array = nullptr;
    d_array = nullptr;
}