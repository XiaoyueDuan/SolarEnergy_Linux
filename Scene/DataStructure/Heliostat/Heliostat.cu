#include "Heliostat.cuh"
#include "vector_arithmetic.cuh"
#include "global_function.cuh"

void Heliostat::CSetNormalAndRotate(const float3 &focus_center, const float3 &sunray_dir) {
    CSetNormal(focus_center, sunray_dir);
    CSetWorldVertex();
}

float3 Heliostat::getPosition() const {
    return pos_;
}

void Heliostat::setPosition(float3 pos) {
    pos_ = pos;
}

float3 Heliostat::getSize() const {
    return size_;
}

void Heliostat::setSize(float3 size) {
    size_ = size;
}

float3 Heliostat::getNormal() const {
    return normal_;
}

void Heliostat::setNormal(float3 normal) {
    normal_ = normal;
}

int2 Heliostat::getRowAndColumn() const {
    return row_col_;
}

void Heliostat::setRowAndColumn(int2 row_col) {
    row_col_ = row_col;
}

float2 Heliostat::getGap() const {
    return gap_;
}

void Heliostat::setGap(float2 gap) {
    gap_ = gap;
}

SubCenterType Heliostat::getSubCenterType() const {
    return subCenterType_;
}

void Heliostat::setSubCenterType(SubCenterType type) {
    subCenterType_ = type;
}

float Heliostat::getPixelLength() const {
    return pixel_length_;
}

void Heliostat::setPixelLength(float pixel_length) {
    pixel_length_ = pixel_length;
}

void Heliostat::Cget_vertex(float3 &v0, float3 &v1, float3 &v3) {
    v0 = vertex_[0];
    v1 = vertex_[1];
    v3 = vertex_[3];
}

void Heliostat::CSetWorldVertex() {
    vertex_[0] = make_float3(-size_.x / 2, size_.y / 2, -size_.z / 2);
    vertex_[1] = vertex_[0] + make_float3(0, 0, size_.z);
    vertex_[2] = vertex_[0] + make_float3(size_.x, 0, size_.z);
    vertex_[3] = vertex_[0] + make_float3(size_.x, 0, 0);

    vertex_[0] = global_func::local2world(vertex_[0], normal_);
    vertex_[1] = global_func::local2world(vertex_[1], normal_);
    vertex_[2] = global_func::local2world(vertex_[2], normal_);
    vertex_[3] = global_func::local2world(vertex_[3], normal_);

    vertex_[0] = global_func::transform(vertex_[0], pos_);
    vertex_[1] = global_func::transform(vertex_[1], pos_);
    vertex_[2] = global_func::transform(vertex_[2], pos_);
    vertex_[3] = global_func::transform(vertex_[3], pos_);
}

void Heliostat::CSetNormal(const float3 &focus_center, const float3 &sunray_dir) {
    float3 local_center = make_float3(pos_.x, pos_.y, pos_.z);
    float3 reflect_dir = focus_center - local_center;
    reflect_dir = normalize(reflect_dir);
    float3 dir = reflect_dir - sunray_dir;
    normal_ = normalize(dir);
}
