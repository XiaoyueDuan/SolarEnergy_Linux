//
// Created by dxt on 18-11-1.
//

#include "destroy.h"
#include "RandomNumberGenerator/RandomGenerator.cuh"
#include "SolarScene.h"

SolarScene* SolarScene::m_instance;
SolarScene* SolarScene::GetInstance()
{
    if (m_instance == nullptr) {
        m_instance = new SolarScene();
    }
    return m_instance;
}

SolarScene::SolarScene():loaded_from_file(false), sunray(nullptr) {
    //init the random
    RandomGenerator::initSeed();
    RandomGenerator::initCudaRandGenerator();

    // Allocate sunray
    sunray = new Sunray();
}

SolarScene::~SolarScene() {
    clear();
    m_instance = nullptr;
}

bool SolarScene::clear() {
    // 1. Free memory on GPU
    free_scene::gpu_free(receivers);
    free_scene::gpu_free(grid0s);
    free_scene::gpu_free(sunray);

    // 2. Free memory on CPU
    free_scene::cpu_free(receivers);
    free_scene::cpu_free(grid0s);
    free_scene::cpu_free(heliostats);
    free_scene::cpu_free(sunray);

    // 3. Clear vector
    receivers.clear();
    grid0s.clear();
    heliostats.clear();
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

bool SolarScene::isLoaded_from_file() const {
    return loaded_from_file;
}

void SolarScene::setLoaded_from_file(bool loaded_from_file) {
    SolarScene::loaded_from_file = loaded_from_file;
}

Sunray *SolarScene::getSunray() {
    return sunray;
}

vector<Grid *> &SolarScene::getGrid0s() {
    return grid0s;
}

vector<Heliostat *> &SolarScene::getHeliostats() {
    return heliostats;
}

vector<Receiver *> &SolarScene::getReceivers() {
    return receivers;
}
