//
// Created by dxt on 18-10-29.
//

#ifndef SOLARENERGYRAYTRACING_GRID_CUH
#define SOLARENERGYRAYTRACING_GRID_CUH

#include "Heliostat.cuh"
#include <vector>

using namespace std;

class Grid
{
public:
    __device__ __host__ Grid(){}
    // set *d_grid_helio_match_, *d_grid_helio_index_ and num_grid_helio_match_
    virtual int CGridHelioMatch(const vector<Heliostat *> &h_helios) = 0;
    virtual void CClear() = 0;
    virtual void Cinit() = 0;

    int getGridType() const;
    void setGridType(int type_);

    __host__ __device__ float3 getPosition() const {
        return pos_;
    }
    void setPosition(float3 pos_);

    float3 getSize() const;
    void setSize(float3 size_);

    __host__ __device__ float3 getInterval() const {
        return interval_;
    }
    void setInterval(float3 interval_);

    int getHeliostatType() const;
    void setHeliostatType(int helio_type_);

    int getStartHeliostatPosition() const;
    void setStartHeliostatPosition(int start_helio_pos_);

    int getNumberOfHeliostats() const;
    void setNumberOfHeliostats(int num_helios_);

    int getBelongingReceiverIndex() const;
    void setBelongingReceiverIndex(int belonging_receiver_index_);

protected:
    int type_;
    float3 pos_;
    float3 size_;
    float3 interval_;
    int helio_type_;
    int start_helio_pos_;			// the first helio index of the helio lists in this grid
    int num_helios_;				// How many heliostats in the grid
    int belonging_receiver_index_;
};

#endif //SOLARENERGYRAYTRACING_GRID_CUH
