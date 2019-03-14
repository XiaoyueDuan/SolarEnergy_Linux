#include "FocusFlatRectangleHelio.cuh"

void FocusFlatRectangleHelio::setSize(float3 size) {
    size_ = size;
}

namespace focusFlatRectangle_heliostat {
    // Step 1: Generate local micro-heliostats' centers and normals
    __global__ void mapMicrohelioCentersAndNormal(float3 *d_microhelio_centers, float3 *d_microhelio_normals,
                                                  float3 *d_local_normal, float3 *d_local_centers,
                                                  float3 subhelio_size,
                                                  const int2 row_col, const int2 sub_row_col,
                                                  const float pixel_length, const size_t size) {
        int myId = global_func::getThreadId();
        if (myId >= size)
            return;

        int row = myId / (row_col.y * sub_row_col.y);
        int col = myId % (row_col.y * sub_row_col.y);
        int block_id = global_func::unroll_index(make_int2(row / sub_row_col.x, col / sub_row_col.y), row_col);
        // 1. Normal
        d_microhelio_normals[myId] = d_local_normal[block_id];

        // 2. Centers
        row %= sub_row_col.x;
        col %= sub_row_col.y;
        // 2.1 Rotate
        float3 local_pos = make_float3((col + 0.5f) * pixel_length - subhelio_size.x / 2,
                                       0.0f,
                                       (row + 0.5f) * pixel_length - subhelio_size.z / 2);
        local_pos = focusFlatRectangleHeliostatLocal2World(local_pos, d_local_normal[block_id]);

        // 2.2 Transform
        d_microhelio_centers[myId] = global_func::transform(local_pos, d_local_centers[block_id]);
    }

    // Step 2: Generate micro-heliostats' normals
    __global__ void map_microhelio_normals(float3 *d_microhelio_world_normals,
                                           float3 *d_microhelio_local_normals, float3 normal, const size_t size) {
        int myId = global_func::getThreadId();
        if (myId >= size)
            return;

        d_microhelio_world_normals[myId] = global_func::local2world(d_microhelio_local_normals[myId], normal);

    }

    // Step 3: Transform local micro-helio center to world postion
    __global__ void map_microhelio_center2world(float3 *d_microhelio_world_centers, float3 *d_microhelio_local_centers,
                                                const float3 normal, const float3 world_pos, const size_t size) {
        int myId = global_func::getThreadId();
        if (myId >= size)
            return;

        float3 local = d_microhelio_local_centers[myId];
        local = focusFlatRectangleHeliostatLocal2World(local, normal);    // Then Rotate
        local = global_func::transform(local, world_pos);   // Translation to the world system
        d_microhelio_world_centers[myId] = local;
    }
}

int FocusFlatRectangleHelio::CGetDiscreteMicroHelioOriginsAndNormals(float3 *&d_microhelio_centers,
                                                                     float3 *&d_microhelio_normals) {
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
    if (d_microhelio_centers == nullptr)
        checkCudaErrors(cudaMalloc((void **) &d_microhelio_centers, sizeof(float3) * map_size));
    if (d_microhelio_normals == nullptr)
        checkCudaErrors(cudaMalloc((void **) &d_microhelio_normals, sizeof(float3) * map_size));
    focusFlatRectangle_heliostat::
    mapMicrohelioCentersAndNormal << < nBlocks, nThreads >> > (d_microhelio_centers, d_microhelio_normals,
            d_local_normals, d_local_centers,
            make_float3(subhelio_row_col_length.y, size_.y, subhelio_row_col_length.x),
            row_col_, sub_row_col, pixel_length_, map_size);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // 2. normal
    focusFlatRectangle_heliostat::
    map_microhelio_normals << < nBlocks, nThreads >> > (d_microhelio_normals, d_microhelio_normals, normal_, map_size);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // 3. world center position
    focusFlatRectangle_heliostat::
        map_microhelio_center2world << < nBlocks, nThreads >> >(d_microhelio_centers, d_microhelio_centers, normal_, pos_, map_size);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    return map_size;
}

void FocusFlatRectangleHelio::CGetSubHeliostatVertexes(std::vector<float3> &subHeliostatVertexes) {
    float2 subhelio_row_col_length;
    subhelio_row_col_length.x = (size_.z - gap_.y * (row_col_.x - 1)) / float(row_col_.x);
    subhelio_row_col_length.y = (size_.x - gap_.x * (row_col_.y - 1)) / float(row_col_.y);

    std::vector<float3> localSubVertexes;
    localSubVertexes.push_back(make_float3(-subhelio_row_col_length.y / 2, 0.0f, subhelio_row_col_length.x / 2));
    localSubVertexes.push_back(make_float3(-subhelio_row_col_length.y / 2, 0.0f, -subhelio_row_col_length.x / 2));
    localSubVertexes.push_back(make_float3(subhelio_row_col_length.y / 2, 0.0f, -subhelio_row_col_length.x / 2));

    float3 *h_local_centers = new float3[row_col_.x * row_col_.y];
    float3 *h_local_normals = new float3[row_col_.x * row_col_.y];

    for (int r = 0; r < row_col_.x; ++r) {
        for (int c = 0; c < row_col_.y; ++c) {
            int id = global_func::unroll_index(make_int2(r, c), row_col_);

            // local centers
            h_local_centers[id] = make_float3(
                    c * (gap_.x + subhelio_row_col_length.y) - size_.x / 2 + subhelio_row_col_length.y / 2,
                    size_.y / 2,
                    r * (gap_.y + subhelio_row_col_length.x) - size_.z / 2 + subhelio_row_col_length.x / 2);
            h_local_centers[id].y += (h_local_centers[id].x * h_local_centers[id].x +
                                      h_local_centers[id].z * h_local_centers[id].z) / (4 * focus_length_);

            // local normals
            h_local_normals[id] = make_float3(-1 / (2 * focus_length_) * h_local_centers[id].x,
                                              1.0f,
                                              -1 / (2 * focus_length_) * h_local_centers[id].z);
            h_local_normals[id] = normalize(h_local_normals[id]);

            for (float3 subHeliostatVertex : localSubVertexes) {
                subHeliostatVertex = focusFlatRectangle_heliostat::focusFlatRectangleHeliostatLocal2World(
                        subHeliostatVertex, h_local_normals[id]);
                subHeliostatVertex = global_func::transform(subHeliostatVertex, h_local_centers[id]);

                subHeliostatVertex = focusFlatRectangle_heliostat::focusFlatRectangleHeliostatLocal2World(
                        subHeliostatVertex, normal_);
                subHeliostatVertex = global_func::transform(subHeliostatVertex, pos_);

                subHeliostatVertexes.push_back(subHeliostatVertex);
            }
        }
    }

    // Copy the local centers and normals from CPU to GPU
    global_func::cpu2gpu(d_local_centers, h_local_centers, row_col_.x * row_col_.y);
    global_func::cpu2gpu(d_local_normals, h_local_normals, row_col_.x * row_col_.y);

    // Clear
    delete[] h_local_centers;
    delete[] h_local_normals;
    h_local_centers = nullptr;
    h_local_normals = nullptr;
}

float FocusFlatRectangleHelio::getFocusLength() const {
    return focus_length_;
}

void FocusFlatRectangleHelio::setFocusLength(float focus_length) {
    focus_length_ = focus_length;
}

std::vector<float3> FocusFlatRectangleHelio::getGPULocalNormals() {
    float3 *h_local_normals = new float3[row_col_.x * row_col_.y];
    global_func::gpu2cpu(h_local_normals, d_local_normals, row_col_.x * row_col_.y);

    std::vector<float3> cpu_local_normals;
    for (int i = 0; i < row_col_.x * row_col_.y; ++i) {
        cpu_local_normals.push_back(h_local_normals[i]);
    }

    // Clear
    delete[] h_local_normals;
    h_local_normals = nullptr;

    return cpu_local_normals;
}

void FocusFlatRectangleHelio::setGPULocalNormals(float3 *h_local_normals) {
    global_func::cpu2gpu(d_local_normals, h_local_normals, row_col_.x * row_col_.y);
}

void FocusFlatRectangleHelio::setGPULocalNormals(std::vector<float3> local_normals) {
    float3 *h_local_normals = new float3[row_col_.x * row_col_.y];
    for (int i = 0; i < row_col_.x * row_col_.y; ++i) {
        h_local_normals[i] = local_normals[i];
    }
    global_func::cpu2gpu(d_local_normals, h_local_normals, row_col_.x * row_col_.y);

    delete[] h_local_normals;
    h_local_normals = nullptr;
}

std::vector<float3> FocusFlatRectangleHelio::getGPULocalCenters() {
    float3 *h_local_centers = new float3[row_col_.x * row_col_.y];
    global_func::gpu2cpu(h_local_centers, d_local_centers, row_col_.x * row_col_.y);

    std::vector<float3> cpu_local_centers;
    for (int i = 0; i < row_col_.x * row_col_.y; ++i) {
        cpu_local_centers.push_back(h_local_centers[i]);
    }

    // Clear
    delete[] h_local_centers;
    h_local_centers = nullptr;

    return cpu_local_centers;
}

void FocusFlatRectangleHelio::setGPUocalCenters(float3 *h_local_centers) {
    global_func::cpu2gpu(d_local_centers, h_local_centers, row_col_.x * row_col_.y);
}

void FocusFlatRectangleHelio::setGPUocalCenters(std::vector<float3> local_centers) {
    float3 *h_local_centers = new float3[row_col_.x * row_col_.y];
    for (int i = 0; i < row_col_.x * row_col_.y; ++i) {
        h_local_centers[i] = local_centers[i];
    }
    global_func::cpu2gpu(d_local_centers, h_local_centers, row_col_.x * row_col_.y);

    delete[] h_local_centers;
    h_local_centers = nullptr;
}

void FocusFlatRectangleHelio::setSurfaceProperty(const std::vector<float> &surface_property) {
    if (surface_property.empty()) {
        return;
    }

    focus_length_ = surface_property[0];
}

std::vector<float> FocusFlatRectangleHelio::getSurfaceProperty() {
    std::vector<float> ans(6, -1.0f);
    ans[0] = focus_length_;
    return ans;
}

void FocusFlatRectangleHelio::CSetNormalAndRotate(const float3 &focus_center, const float3 &sunray_dir) {
    if (focus_length_ < 0.0f) {
        focus_length_ = length(focus_center - pos_);
    }
    Heliostat::CSetNormalAndRotate(focus_center, sunray_dir);
}
