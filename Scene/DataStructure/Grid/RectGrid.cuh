#ifndef SOLARENERGYRAYTRACING_RECTGRID_CUH
#define SOLARENERGYRAYTRACING_RECTGRID_CUH

#include "Grid.cuh"

using namespace std;

class RectGrid :public Grid
{
public:
    virtual void CGridHelioMatch(const vector<Heliostat *> &h_helios);				// set *d_grid_helio_match_, *d_grid_helio_index_ and num_grid_helio_match_
    virtual void CClear();
    virtual void Cinit();

    __device__ __host__ RectGrid():d_grid_helio_index_(nullptr), d_grid_helio_match_(nullptr){}

    __device__ __host__ ~RectGrid()
    {
        if(d_grid_helio_match_)
            d_grid_helio_match_ = nullptr;
        if (d_grid_helio_index_)
            d_grid_helio_index_ = nullptr;
    }

    const int3 &getGridNumber() const;
    void setGridNumber(int3 grid_num_);

    int *getDeviceGridHeliostatMatch() const;
    void setDeviceGridHeliostatMatch(int *d_grid_helio_match_);

    int *getDeviceGridHelioIndex() const;
    void setDeviceGridHelioIndex(int *d_grid_helio_index_);

    size_t getNumberOfGridHeliostatMatch() const;
    void setNumberOfGridHeliostatMatch(size_t num_grid_helio_match_);

private:
    int3 grid_num_;						// x * y * z 's sub-grid
    int *d_grid_helio_match_;			// size = num_grid_helio_match_
    int *d_grid_helio_index_;			// size = size.x * size.y * size.z +1
    size_t num_grid_helio_match_;
};

#endif //SOLARENERGYRAYTRACING_RECTGRID_CUH