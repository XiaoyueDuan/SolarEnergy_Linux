//
// Created by dxt on 18-11-16.
//

#include <stdexcept>

#include "check_cuda.h"
#include "cuda_runtime.h"
#include "global_function.cuh"
#include "QuasiMonteCarloRayTracer.h"
#include "RectangleReceiverRectGridRayTracing.cuh"
#include "CylinderReceiverRectGridRayTracing.cuh"
#include "RectGrid.cuh"

void QuasiMonteCarloRayTracer::rayTracing(SolarScene *solarScene, int heliostat_id) {
    checkValidHeliostatIndex(solarScene, heliostat_id);

    // Data structure
    Sunray *sunray = solarScene->getSunray();
    Heliostat *heliostat = solarScene->getHeliostats()[heliostat_id];
    int grid_id = heliostat->getBelongingGridId();
    Grid *grid = solarScene->getGrid0s()[grid_id];
    int receiver_id = grid->getBelongingReceiverIndex();
    Receiver *receiver = solarScene->getReceivers()[receiver_id];

    // Construct the sub-heliostat vertexes array
    float3 *d_subHeliostat_vertexes = nullptr;
    setFlatRectangleHeliostatVertexes(d_subHeliostat_vertexes, solarScene->getHeliostats(),
                                      grid->getStartHeliostatPosition(),
                                      grid->getStartHeliostatPosition() + grid->getNumberOfHeliostats());

    // Construct arguments
    SunrayArgument sunrayArgument = generateSunrayArgument(sunray);
    HeliostatArgument heliostatArgument = generateHeliostatArgument(solarScene, heliostat_id);

    int receiverGridCombinationIndex = receiverGridCombination(receiver->getType(), grid->getGridType());
    switch (receiverGridCombinationIndex) {
        case 0: {
            /** RectangleReceiver v.s. RectGrid*/
            auto rectangleReceiver = dynamic_cast<RectangleReceiver *>(receiver);
            auto rectGrid = dynamic_cast<RectGrid *>(grid);
            float ratio = heliostat->getPixelLength() / receiver->getPixelLength();
            float factor = sunray->getDNI() * ratio * ratio * sunray->getReflectiveRate()
                    / float(sunray->getNumOfSunshapeLightsPerGroup()) ;
            RectangleReceiverRectGridRayTracing(sunrayArgument, rectangleReceiver, rectGrid, heliostatArgument,
                                                d_subHeliostat_vertexes, factor);
            break;
        }
        case 10: {
            /** CylinderReceiver v.s. RectGrid*/
            auto cylinderReceiver = dynamic_cast<CylinderReceiver *>(receiver);
            auto rectGrid = dynamic_cast<RectGrid *>(grid);
            float ratio = heliostat->getPixelLength() / receiver->getPixelLength();
            float factor = sunray->getDNI() * ratio * ratio * sunray->getReflectiveRate()
                           / float(sunray->getNumOfSunshapeLightsPerGroup()) ;
            CylinderReceiverRectGridRayTracing(sunrayArgument, cylinderReceiver, rectGrid, heliostatArgument,
                                                d_subHeliostat_vertexes, factor);
            break;
        }
            /**
             * TODO: Add other branch for different type of receiver or grid.
             */
        default:
            break;
    }

    cudaFree(d_subHeliostat_vertexes);
    d_subHeliostat_vertexes = nullptr;
    heliostatArgument.CClear();
}

void QuasiMonteCarloRayTracer::checkValidHeliostatIndex(SolarScene *solarScene, int heliostat_id) {
    std::string error_message;
    std::string suffix = " is invalid.";
    // 1. Valid heliostat id
    error_message = "The ray tracing heliostat index " + std::to_string(heliostat_id);
    size_t total_heliostat = solarScene->getHeliostats().size();
    if (heliostat_id < 0 || heliostat_id >= total_heliostat) {
        std::string total_heliostat_message =
                "The heliostat index should in the range of [0, " + std::to_string(total_heliostat) + "].";
        throw std::runtime_error(error_message + suffix + total_heliostat_message);
    }

    // 2. Valid gird id
    Heliostat *heliostat = solarScene->getHeliostats()[heliostat_id];
    int grid_id = heliostat->getBelongingGridId();
    error_message += "with belonging grid index " + std::to_string(grid_id);
    size_t total_grid = solarScene->getGrid0s().size();
    if (grid_id < 0 || grid_id >= total_grid) {
        std::string total_grid_message =
                "The grid index should in the range of [0, " + std::to_string(total_grid) + "].";
        throw std::runtime_error(error_message + suffix + total_grid_message);
    }

    // 3. Valid receiver id
    Grid *grid = solarScene->getGrid0s()[grid_id];
    int receiver_id = grid->getBelongingReceiverIndex();
    error_message += "of belonging receiver index " + std::to_string(receiver_id);
    size_t total_receivers = solarScene->getReceivers().size();
    if (receiver_id < 0 || receiver_id >= total_receivers) {
        std::string total_receiver_message =
                "The receiver index should in the range of [0, " + std::to_string(total_receivers) + "].";
        throw std::runtime_error(error_message + suffix + total_receiver_message);
    }
}

int QuasiMonteCarloRayTracer::receiverGridCombination(int receiver_type, int grid_type) {
    return receiver_type * 10 + grid_type;
}

HeliostatArgument QuasiMonteCarloRayTracer::generateHeliostatArgument(SolarScene *solarScene, int heliostat_id) {
    Heliostat *heliostat = solarScene->getHeliostats()[heliostat_id];
    float3 *d_microhelio_origins = nullptr;
    float3 *d_microhelio_normals = nullptr;
    int numberOfMicrohelio =
            heliostat->CGetDiscreteMicroHelioOriginsAndNormals(d_microhelio_origins, d_microhelio_normals);
    int pool_size = solarScene->getSunray()->getNumOfSunshapeGroups() *
                    solarScene->getSunray()->getNumOfSunshapeLightsPerGroup();
    int *d_microhelio_belonging_groups = heliostat->generateDeviceMicrohelioGroup(pool_size, numberOfMicrohelio);

    int subHeliostat_id = 0;
    Grid *grid = solarScene->getGrid0s()[heliostat->getBelongingGridId()];
    for (int i = 0; i < grid->getNumberOfHeliostats(); ++i) {
        int real_id = i + grid->getStartHeliostatPosition();
        if (real_id == heliostat_id) {
            break;
        }
        Heliostat *before_heliostat = solarScene->getHeliostats()[real_id];
        subHeliostat_id += before_heliostat->getSubHelioSize();
    }
    return HeliostatArgument(d_microhelio_origins, d_microhelio_normals, d_microhelio_belonging_groups,
                             numberOfMicrohelio, subHeliostat_id, heliostat->getSubHelioSize());
}

SunrayArgument QuasiMonteCarloRayTracer::generateSunrayArgument(Sunray *sunray) {
    return SunrayArgument(sunray->getDeviceSampleLights(), sunray->getDevicePerturbation(),
                          sunray->getNumOfSunshapeGroups() * sunray->getNumOfSunshapeLightsPerGroup(),
                          sunray->getNumOfSunshapeLightsPerGroup(), sunray->getSunDirection());
}

int QuasiMonteCarloRayTracer::setFlatRectangleHeliostatVertexes(float3 *&d_heliostat_vertexes,
                                                                std::vector<Heliostat *> &heliostats, int start_id,
                                                                int end_id) {
    if (start_id < 0 || start_id > end_id || end_id > heliostats.size()) {
        throw std::runtime_error(
                __FILE__". The index " + std::to_string(start_id) + " and " + std::to_string(end_id) + " is invalid.");
    }

    if(d_heliostat_vertexes!= nullptr) {
        checkCudaErrors(cudaFree(d_heliostat_vertexes));
        d_heliostat_vertexes = nullptr;
    }

    std::vector<float3> subHeliostatVertexes;
    for (int i = start_id; i < end_id; ++i) {
        heliostats[i]->CGetSubHeliostatVertexes(subHeliostatVertexes);
    }
    float3 *h_heliostat_vertexes = new float3[subHeliostatVertexes.size()];
    for (int i = 0; i < subHeliostatVertexes.size(); ++i) {
        h_heliostat_vertexes[i] = subHeliostatVertexes[i];
    }

    global_func::cpu2gpu(d_heliostat_vertexes, h_heliostat_vertexes, 3 * (subHeliostatVertexes.size()));
    delete[] h_heliostat_vertexes;
    h_heliostat_vertexes = nullptr;
    return subHeliostatVertexes.size();
}

