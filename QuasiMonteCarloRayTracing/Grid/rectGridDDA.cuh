#ifndef SOLARENERGYRAYTRACING_RECTGRIDDDA_CUH
#define SOLARENERGYRAYTRACING_RECTGRIDDDA_CUH

#include "cuda_runtime.h"
#include "RectGrid.cuh"

namespace rectGridDDA {
    template<typename T>
    inline __host__ __device__ T absDivide(const T &denominator, const T &numerator) {
        if (numerator <= Epsilon && numerator >= -Epsilon)
            return T(INT_MAX);
        return abs(denominator / numerator);
    }

    inline __host__ __device__ float calTMax(float dir, float interval, int current_index, float current_pos) {
        return abs(float(current_index + (dir >= 0)) * interval - current_pos);
    }

    /**
     * Intersect with the heliostat within this rectangle grid
     * */
    __host__ __device__ bool intersect(const float3 &orig, const float3 &dir, const float3 *d_heliostat_vertexs,
                                       const int const *d_grid_heliostat_match, int start_id, int end_id,
                                       int heliostat_id, int numberOfSubHeliostat);

    /**
     * 3DDDA
     * */
    __host__ __device__ bool collision(const float3 &origin, const float3 &dir, const RectGrid &rectGrid,
                                       const float3 *d_subheliostat_vertexes,
                                       const HeliostatArgument &heliostatArgument);
}

#endif // SOLARENERGYRAYTRACING_RECTGRIDDDA_CUH