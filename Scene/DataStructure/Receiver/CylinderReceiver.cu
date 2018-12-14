#include "CylinderReceiver.cuh"

void CylinderReceiver::CInit(int geometry_info) {
    pixel_length_ = 1.0f / float(geometry_info);
    Cset_resolution(geometry_info);
    Calloc_image();
    Cclean_image_content();
}

void CylinderReceiver::Cset_resolution(int geometry_info) {
    resolution_.x = int(ceil(2 * size_.x * M_PI * float(geometry_info)));
    resolution_.y = int(size_.y * float(geometry_info));
}

float3 CylinderReceiver::getFocusCenter(const float3 &heliostat_position) {
    if(innerToCylinder(heliostat_position)) {
        // If the heliostat position in the cylinder, then return the receiver position
        return pos_;
    }

    float3 dir = heliostat_position - pos_;
    dir = normalize(dir);
    float radius = size_.x/(length(make_float2(dir.x, dir.z)));

    float x = pos_.x + dir.x * radius;
    float z = pos_.z + dir.z * radius;
    return make_float3(x, pos_.y, z);
}
