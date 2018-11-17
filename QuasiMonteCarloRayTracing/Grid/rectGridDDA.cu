#include "rectGridDDA.cuh"

__host__ __device__ bool rectGridDDA::intersect(const float3 &orig, const float3 &dir,
                                                const float3 *d_heliostat_vertexs,
                                                const int const *d_grid_heliostat_match, int start_id, int end_id,
                                                int heliostat_id, int numberOfSubHeliostat) {
    float t, u, v;
    for (int i = start_id; i < end_id; ++i) {
        int subHeliostatIndex = d_grid_heliostat_match[i];
        if (subHeliostat_index < heliostat_id || subHeliostat_index >= heliostat_id + numberOfSubHeliostat) {
            if (global_func::rayParallelogramIntersect(orig, dir, d_heliostat_vertexs[3 * subHeliostatIndex],
                                                       d_heliostat_vertexs[3 * subHeliostatIndex + 1],
                                                       d_heliostat_vertexs[3 * subHeliostatIndex + 2], t, u, v)) {
                return true;
            }
        }
    }
    return false;
}

__host__ __device__ bool rectGridDDA::collision(const float3 &origin, const float3 &dir, const RectGrid &rectGrid,
                                                const float3 *d_subheliostat_vertexes,
                                                const HeliostatArgument &heliostatArgument) {
    // Step 1 - Initialization
    //	Step 1.1 Initial current position of origin in the scene
    int3 pos = make_int3((d_orig - rectGrid.getPosition()) / rectGrid.getInterval());

    //	Step 1.2 StepX, StepY, StepZ
    int3 Step;
    Step.x = (d_dir.x >= 0) ? 1 : -1;
    Step.y = (d_dir.y >= 0) ? 1 : -1;
    Step.z = (d_dir.z >= 0) ? 1 : -1;

    //	Step 1.3 Initial tmaxX, tmaxY, tmaxZ
    float3 tMax;    // avoid divide 0
    tMax.x = absDivide(calTMax(d_dir.x, rectGrid.getInterval().x, pos.x, d_orig.x - rectGrid.pos_.x), d_dir.x);
    tMax.y = absDivide(calTMax(d_dir.y, rectGrid.getInterval().y, pos.y, d_orig.y - rectGrid.pos_.y), d_dir.y);
    tMax.z = absDivide(calTMax(d_dir.z, rectGrid.getInterval().z, pos.z, d_orig.z - rectGrid.pos_.z), d_dir.z);

    //	Step 1.4 Initial tDeltaX, tDeltaY, tDeltaZ
    float3 tDelta;  // avoid divide 0
    tDelta.x = absDivide(rectGrid.getInterval().x, d_dir.x);
    tDelta.y = absDivide(rectGrid.getInterval().y, d_dir.y);
    tDelta.z = absDivide(rectGrid.getInterval().z, d_dir.z);

    // Step 2 - Intersection
    int3 grid_index = pos;
    int grid_address;
    while (grid_index.x >= 0 && grid_index.x < rectGrid.getGridNumber().x &&
           grid_index.y >= 0 && grid_index.y < rectGrid.getGridNumber().y &&
           grid_index.z >= 0 && grid_index.z < rectGrid.getGridNumber().z) {
        grid_address = global_func::unroll_index(grid_index, rectGrid.getGridNumber());
        if (Intersect(orig, dir, d_helio_vertexs, rectGrid.getDeviceGridHeliostatMatch(),
                      rectGrid.getDeviceGridHelioIndex()[grid_address],
                      rectGrid.getDeviceGridHelioIndex()[grid_address + 1],
                      heliostatArgument.heliostat_id, heliostatArgument.numberOfSubHeliostats)) {
            return true;
        }

        // next cell location
        if (tMax.x < tMax.y) {
            if (tMax.x <= tMax.z) {
                grid_index.x += Step.x;
                tMax.x += tDelta.x;
            }
            if (tMax.x >= tMax.z) {
                grid_index.z += Step.z;
                tMax.z += tDelta.z;
            }
        }
        if (tMax.x > tMax.y) {
            if (tMax.y <= tMax.z) {
                grid_index.y += Step.y;
                tMax.y += tDelta.y;
            }
            if (tMax.y >= tMax.z) {
                grid_index.z += Step.z;
                tMax.z += tDelta.z;
            }
        }
    }
    return false;
}