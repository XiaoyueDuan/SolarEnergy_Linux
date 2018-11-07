//
// Created by dxt on 18-11-1.
//

#ifndef SOLARENERGYRAYTRACING_SOLARSCENE_H
#define SOLARENERGYRAYTRACING_SOLARSCENE_H

#include <vector>
#include <string>

#include "Grid.cuh"
#include "Heliostat.cuh"
#include "Receiver.cuh"
#include "Sunray.cuh"

using namespace std;

class SolarScene {
private:
    SolarScene();

    float ground_length_;
    float ground_width_;
    int grid_num_;
    static SolarScene *m_instance;		//Singleton

public:
    static SolarScene* GetInstance();   //static member
    static void InitInstance();
    ~SolarScene();

    bool InitSolarScene();
    bool InitSolarScene(string filepath);
    bool LoadSceneFromFile(string filepath);

    // Call the method only if all grids, heliostats and receivers needs initializing.
    bool InitContent();

    Sunray *sunray;
    vector<Grid *> grid0s;
    vector<Heliostat *> heliostats;
    vector<Receiver *> receivers;
};

#endif //SOLARENERGYRAYTRACING_SOLARSCENE_H
