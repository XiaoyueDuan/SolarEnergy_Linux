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

    static SolarScene *m_instance;		//Singleton

    float ground_length_;
    float ground_width_;
    int grid_num_;

    Sunray *sunray;
    vector<Grid *> grid0s;
    vector<Heliostat *> heliostats;
    vector<Receiver *> receivers;

public:
    static SolarScene* GetInstance();   //static member
    ~SolarScene();

    float getGroundLength() const;
    void setGroundLength(float ground_length_);

    float getGroundWidth() const;
    void setGroundWidth(float ground_width_);

    int getNumberOfGrid() const;
    void setNumberOfGrid(int grid_num_);

    void addReceiver(Receiver *receiver);
    void addGrid(Grid *grid);
    void addHeliostat(Heliostat *heliostat);
};

#endif //SOLARENERGYRAYTRACING_SOLARSCENE_H
