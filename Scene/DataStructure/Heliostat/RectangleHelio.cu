#include "RectangleHelio.cuh"
#include "global_function.cuh"

void RectangleHelio::setSize(float3 size) {
    size_ = size;
}

namespace rectange_heliostat {
    // Step 1: Generate local micro-heliostats' centers
    __global__ void map_microhelio_centers(float3 *d_microhelio_centers, float3 helio_size,
                                           const int2 row_col, const int2 sub_row_col, const float2 gap,
                                           const float pixel_length, const size_t size) {
        int myId = global_func::getThreadId();
        if (myId >= size)
            return;

        int row = myId / (row_col.y * sub_row_col.y);
        int col = myId % (row_col.y * sub_row_col.y);

        int block_row = row / sub_row_col.x;
        int block_col = col / sub_row_col.y;

        d_microhelio_centers[myId].x = col * pixel_length + block_col * gap.x + pixel_length / 2 - helio_size.x / 2;
        d_microhelio_centers[myId].y = helio_size.y / 2;
        d_microhelio_centers[myId].z = row * pixel_length + block_row * gap.y + pixel_length / 2 - helio_size.z / 2;
    }

    // Step 2: Generate micro-heliostats' normals
    __global__ void map_microhelio_normals(float3 *d_microhelio_normals, float3 normal, const size_t size) {
        int myId = global_func::getThreadId();
        if (myId >= size)
            return;
        d_microhelio_normals[myId] = normal;
    }

    // Step 3: Transform local micro-helio center to world postion
    __global__ void map_microhelio_center2world(float3 *d_microhelio_world_centers, float3 *d_microhelio_local_centers,
                                                const float3 normal, const float3 world_pos, const size_t size) {
        int myId = global_func::getThreadId();
        if (myId >= size)
            return;

        float3 local = d_microhelio_local_centers[myId];
        local = global_func::local2world(local, normal);    // Then Rotate
        local = global_func::transform(local, world_pos);   // Translation to the world system
        d_microhelio_world_centers[myId] = local;
    }
}

int
RectangleHelio::CGetDiscreteMicroHelioOriginsAndNormals(float3 *&d_microhelio_vertexs, float3 *&d_microhelio_normals) {
    float2 subhelio_row_col_length;
    subhelio_row_col_length.x = (size_.z - gap_.y * (row_col_.x - 1)) / float(row_col_.x);
    subhelio_row_col_length.y = (size_.x - gap_.x * (row_col_.y - 1)) / float(row_col_.y);

    int2 sub_row_col;
    sub_row_col.x = subhelio_row_col_length.x / pixel_length_;
    sub_row_col.y = subhelio_row_col_length.y / pixel_length_;

    int map_size = sub_row_col.x * sub_row_col.y * row_col_.x * row_col_.y;

    int nThreads;
    dim3 nBlocks;
    global_func::setThreadsBlocks(nBlocks, nThreads, map_size);

    // 1. local center position
    if (d_microhelio_vertexs == nullptr)
        checkCudaErrors(cudaMalloc((void **) &d_microhelio_vertexs, sizeof(float3) * map_size));
    rectange_heliostat::map_microhelio_centers << < nBlocks, nThreads >> >
        (d_microhelio_vertexs, size_, row_col_, sub_row_col, gap_, pixel_length_, map_size);

    // 2. normal
    if (d_microhelio_normals == nullptr)
        checkCudaErrors(cudaMalloc((void **) &d_microhelio_normals, sizeof(float3) * map_size));
    rectange_heliostat::map_microhelio_normals << < nBlocks, nThreads >> > (d_microhelio_normals, normal_, map_size);

    // 3. world center position
    rectange_heliostat::map_microhelio_center2world << < nBlocks, nThreads >> >
        (d_microhelio_vertexs, d_microhelio_vertexs, normal_, pos_, map_size);

    return map_size;
}

void RectangleHelio::CGetSubHeliostatVertexes(std::vector<float3> &subHeliostatVertexes) {
    subHeliostatVertexes.push_back(vertex_[0]);
    subHeliostatVertexes.push_back(vertex_[1]);
    subHeliostatVertexes.push_back(vertex_[2]);
}
