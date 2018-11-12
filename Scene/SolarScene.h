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
    bool loaded_from_file;

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
    bool clear();

    float getGroundLength() const;
    void setGroundLength(float ground_length_);

    float getGroundWidth() const;
    void setGroundWidth(float ground_width_);

    int getNumberOfGrid() const;
    void setNumberOfGrid(int grid_num_);

    bool isLoaded_from_file() const;
    void setLoaded_from_file(bool loaded_from_file);

    void addReceiver(Receiver *receiver);
    void addGrid(Grid *grid);
    void addHeliostat(Heliostat *heliostat);

    Sunray *getSunray();
    vector<Grid *> &getGrid0s();
    vector<Heliostat *> &getHeliostats();
    vector<Receiver *> &getReceivers();
};

#endif //SOLARENERGYRAYTRACING_SOLARSCENE_H
