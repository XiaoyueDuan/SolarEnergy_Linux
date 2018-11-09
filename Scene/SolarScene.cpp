//
// Created by dxt on 18-11-1.
//

#include "SolarScene.h"
#include "RandomNumberGenerator/RandomGenerator.cuh"

SolarScene* SolarScene::m_instance;
SolarScene* SolarScene::GetInstance()
{
    if (m_instance == nullptr) {
        m_instance = new SolarScene();
    }
    return m_instance;
}

SolarScene::SolarScene() {
    //init the random
    RandomGenerator::initSeed();
    RandomGenerator::initCudaRandGenerator();
}

SolarScene::~SolarScene() {
//    // 1. free memory on GPU
//    free_scene::gpu_free(receivers);
//    free_scene::gpu_free(grid0s);
//    free_scene::gpu_free(sunray_);
//
//    // 2. free memory on CPU
//    free_scene::cpu_free(receivers);
//    free_scene::cpu_free(grid0s);
//    free_scene::cpu_free(heliostats);
//    free_scene::cpu_free(sunray_);
}

float SolarScene::getGroundLength() const {
    return ground_length_;
}

void SolarScene::setGroundLength(float ground_length_) {
    SolarScene::ground_length_ = ground_length_;
}

float SolarScene::getGroundWidth() const {
    return ground_width_;
}

void SolarScene::setGroundWidth(float ground_width_) {
    SolarScene::ground_width_ = ground_width_;
}

int SolarScene::getNumberOfGrid() const {
    return grid_num_;
}

void SolarScene::setNumberOfGrid(int grid_num_) {
    SolarScene::grid_num_ = grid_num_;
}

void SolarScene::addReceiver(Receiver *receiver) {
    receivers.push_back(receiver);
}

void SolarScene::addGrid(Grid *grid) {
    grid0s.push_back(grid);
}

void SolarScene::addHeliostat(Heliostat *heliostat) {
    heliostats.push_back(heliostat);
}
