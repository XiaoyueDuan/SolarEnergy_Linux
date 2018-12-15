#include "RectGrid.cuh"
#include "global_function.cuh"
#include "check_cuda.h"

void RectGrid::Cinit() {
    grid_num_.x = ceil(size_.x / interval_.x);
    grid_num_.y = ceil(size_.y / interval_.y);
    grid_num_.z = ceil(size_.z / interval_.z);
}

void RectGrid::CClear() {
    if (d_grid_helio_match_) {
        checkCudaErrors(cudaFree(d_grid_helio_match_));
        d_grid_helio_match_ = nullptr;
    }

    if (d_grid_helio_index_) {
        checkCudaErrors(cudaFree(d_grid_helio_index_));
        d_grid_helio_index_ = nullptr;
    }
}

int boxIntersect(int mirrorBeiginId,
                 int subMirrorSize,
                 float3 min_pos,
                 float3 max_pos,
                 const RectGrid &grid,
                 vector<vector<int> > &grid_mirrow_match_vector) {
    int size = 0;
    float3 pos = grid.getPosition();
    float3 interval = grid.getInterval();
    int3 grid_num = grid.getGridNumber();

    int3 min_grid_pos = make_int3((min_pos - pos).x / interval.x,
                                  (min_pos - pos).y / interval.y,
                                  (min_pos - pos).z / interval.z);
    int3 max_grid_pos = make_int3((max_pos - pos).x / interval.x,
                                  (max_pos - pos).y / interval.y,
                                  (max_pos - pos).z / interval.z);

    if (min_grid_pos.x < 0 || min_grid_pos.y < 0 || min_grid_pos.z < 0 ||
        max_grid_pos.x >= grid_num.x || max_grid_pos.y >= grid_num.y || max_grid_pos.z >= grid_num.z) {
        std::cerr << "The heliostats may out of the grid boundary. Please check your file." << std::endl;
    }

    for (int x = max(0, min_grid_pos.x); x <= min(grid_num.x - 1, max_grid_pos.x); ++x) {
        for (int y = max(0, min_grid_pos.y); y <= min(grid_num.y - 1, max_grid_pos.y); ++y) {
            for (int z = max(0, min_grid_pos.z); z <= min(grid_num.z - 1, max_grid_pos.z); ++z) {
                int pos_id = x * grid_num.y * grid_num.z + y * grid_num.z + z;
                for (int i = 0; i < subMirrorSize; ++i) {
                    grid_mirrow_match_vector[pos_id].push_back(mirrorBeiginId + i);
                }
                size += subMirrorSize;
            }
        }
    }

    return size;
}

int RectGrid::CGridHelioMatch(
        const vector<Heliostat *> &h_helios) // set *d_grid_helio_match_, *d_grid_helio_index_ and num_grid_helio_match_
{
    if (d_grid_helio_match_ || d_grid_helio_index_) {
        throw std::runtime_error(
                "The grid and heliostats corresponding relationship should be empty before calling this method");
    }

    int start_subhelio_pos = 0;
    float3 minPos, maxPos;
    float radius = 0.0f;
    num_grid_helio_match_ = 0;

    vector<vector<int> > grid_mirrow_match_vector(grid_num_.x * grid_num_.y * grid_num_.z, vector<int>());
    for (int i = start_helio_pos_; i < start_helio_pos_ + num_helios_; ++i) {
        float3 pos = h_helios[i]->getPosition();
        radius = length(h_helios[i]->getSize()) / 2;

        minPos = pos - radius;
        maxPos = pos + radius;

        num_grid_helio_match_ += boxIntersect(start_subhelio_pos, h_helios[i]->getSubHelioSize(),
                                              minPos, maxPos, *this, grid_mirrow_match_vector);
        start_subhelio_pos += h_helios[i]->getSubHelioSize();
    }

    int *h_grid_helio_index = new int[grid_num_.x * grid_num_.y * grid_num_.z + 1];
    h_grid_helio_index[0] = 0;
    int *h_grid_helio_match = new int[num_grid_helio_match_];

    int index = 0;
    for (int i = 0; i < grid_num_.x * grid_num_.y * grid_num_.z; ++i) {
        h_grid_helio_index[i + 1] = h_grid_helio_index[i] + grid_mirrow_match_vector[i].size();
        for (int j = 0; j < grid_mirrow_match_vector[i].size(); ++j, ++index)
            h_grid_helio_match[index] = grid_mirrow_match_vector[i][j];
    }

    global_func::cpu2gpu(d_grid_helio_match_, h_grid_helio_match, num_grid_helio_match_);
    global_func::cpu2gpu(d_grid_helio_index_, h_grid_helio_index, grid_num_.x * grid_num_.y * grid_num_.z + 1);

    delete[] h_grid_helio_index;
    delete[] h_grid_helio_match;
    h_grid_helio_index = nullptr;
    h_grid_helio_match = nullptr;

    return start_subhelio_pos;
}

/**
 * Getter and setter of attributes for RectGrid
 */
void RectGrid::setGridNumber(int3 grid_num) {
    grid_num_ = grid_num;
}

void RectGrid::setDeviceGridHeliostatMatch(int *d_grid_helio_match) {
    d_grid_helio_match_ = d_grid_helio_match;
}

void RectGrid::setDeviceGridHelioIndex(int *d_grid_helio_index) {
    d_grid_helio_index_ = d_grid_helio_index;
}

size_t RectGrid::getNumberOfGridHeliostatMatch() const {
    return num_grid_helio_match_;
}

void RectGrid::setNumberOfGridHeliostatMatch(size_t num_grid_helio_match) {
    num_grid_helio_match_ = num_grid_helio_match;
}