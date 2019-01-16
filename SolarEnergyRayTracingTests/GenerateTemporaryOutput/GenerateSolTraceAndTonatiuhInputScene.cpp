//
// Created by dxt on 19-1-7.
//

#include <SceneLoader.h>
#include <SceneProcessor.h>
#include <random>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <ImageSaver.h>
#include <QuasiMonteCarloRayTracer.h>

#include "gtest/gtest.h"
#include "vector_arithmetic.cuh"
#include "global_function.cuh"

namespace GenerateSolTraceAndTonatiuhInput {
    bool saveResultFloat3Vector(std::string path, vector<float3> &array) {
        std::ofstream fout(path.c_str());

        for (float3 it : array) {
            fout << it.x << "\t\t" << it.y << "\t\t" << it.z << '\n';
        }
        return true;
    }

    void saveReceiverResult(Receiver *receiver, std::string pathAndName) {
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

    void mySetGridReceiverHeliostatContent(SolarScene *solarScene, SceneConfiguration *sceneConfiguration,
                                           std::string tonatiuh_output_path, std::string solTrace_output_path,
                                           float upper_lmt, float bottom_lmt) {
        int pixels_per_meter_for_receiver = int(1.0f / sceneConfiguration->getReceiver_pixel_length());
        float heliostat_pixel_length = sceneConfiguration->getHelio_pixel_length();
        float3 sun_direction = sceneConfiguration->getSun_dir();

        vector<Heliostat *> heliostats = solarScene->getHeliostats();
        vector<Receiver *> receivers = solarScene->getReceivers();
        vector<float3> tonatiuh_format_files;
        vector<float3> solTrace_format_files;
        float interval = upper_lmt - bottom_lmt;

        /**
         * 1. Change the random seed
         * 2. Uncommend the eta_aAlpha in receiverIntersectionUtil.cuh to return 1.0 no matter what kind of arguments are
         * */
        srand(1000);
        for (Grid *grid : solarScene->getGrid0s()) {
            grid->Cinit();
            grid->CGridHelioMatch(heliostats);

            Receiver *receiver = receivers[grid->getBelongingReceiverIndex()];
            receiver->CInit(pixels_per_meter_for_receiver);

            for (int i = 0; i < grid->getNumberOfHeliostats(); ++i) {
                int id = i + grid->getStartHeliostatPosition();
                float3 focus_center = receiver->getFocusCenter(heliostats[id]->getPosition());
                float u = ((float) rand() / (RAND_MAX)) * interval + bottom_lmt;
                focus_center.y += u;
                // Save Tonatiuh results
                tonatiuh_format_files.push_back(focus_center);

                heliostats[id]->setPixelLength(heliostat_pixel_length);
                heliostats[id]->CSetNormalAndRotate(focus_center, sun_direction);

                // Save SolTrace results
                solTrace_format_files.push_back(heliostats[i]->getPosition() + interval * heliostats[i]->getNormal());
            }
        }
        saveResultFloat3Vector(tonatiuh_output_path, tonatiuh_format_files);
        saveResultFloat3Vector(solTrace_output_path, solTrace_format_files);
    }
}

TEST(GenerateSolTraceAndTonatiuhInputSceneFormat, modificationForRayTracingPipeline) {
    // 1. Set argument
    std::string configuration_path = "input_file/6282_configure.json";
    std::string scene_path = "input_file/6282_QMCRT.scn";
    std::string tonatiuh_output_path("output_file/tonatiuh_output_6282.txt");
    std::string solTrace_output_path("output_file/solTrace_output_6282.txt");
    std::string QMCRT_output_image_path("output_file/QMCRT_image_6282(no-atmosphere-decay).txt");

    float upper_lmt = 5;
    float bottom_lmt = -5;
    float interval = upper_lmt - bottom_lmt;

    // 2. Initialize solar scene
    std::cout << "\n2. Initialize solar scene..." << std::endl;
    //  2.1 configuration
    std::cout << "  2.1 Load configuration from '" << configuration_path << "'." << std::endl;
    SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
    sceneConfiguration->loadConfiguration(configuration_path);

    //  2.2 load scene
    std::cout << "  2.2 Load scene file from '" << scene_path << "'." << std::endl;
    SceneLoader sceneLoader;
    SolarScene *solarScene = SolarScene::GetInstance();
    sceneLoader.SceneFileRead(solarScene, scene_path);

    vector<Heliostat *> heliostats = solarScene->getHeliostats();

    //  2.3 process scene
    std::cout << "  2.3 Process scene." << std::endl;
    SceneProcessor sceneProcessor(sceneConfiguration);
    sceneProcessor.set_sunray_content(*solarScene->getSunray());
    GenerateSolTraceAndTonatiuhInput::mySetGridReceiverHeliostatContent(solarScene, sceneConfiguration,
                                                                        tonatiuh_output_path, solTrace_output_path,
                                                                        upper_lmt, bottom_lmt);

    // 3. Ray tracing (could be parallel)
    std::cout << "3. Start ray tracing..." << std::endl;
    QuasiMonteCarloRayTracer QMCRTracer;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;
    long long elapsed;

    for (int heliostat_index = 0; heliostat_index < heliostats.size(); ++heliostat_index) {
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
    std::cout << "\n4. Save results in '" << QMCRT_output_image_path << "' directory." << std::endl;
    GenerateSolTraceAndTonatiuhInput::saveReceiverResult(solarScene->getReceivers()[0], QMCRT_output_image_path);

    // Finally, clear the scene
    solarScene->clear();
}