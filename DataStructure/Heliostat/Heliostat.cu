#include "Heliostat.cuh"

const float3 &Heliostat::getPosition() const {
    return pos_;
}

void Heliostat::setPosition(float3 pos_) {
    Heliostat::pos_ = pos_;
}

const float3 &Heliostat::getSize() const {
    return size_;
}

void Heliostat::setSize(float3 size_) {
    Heliostat::size_ = size_;
}

const float3 &Heliostat::getNormal() const {
    return normal_;
}

void Heliostat::setNormal(float3 normal_) {
    Heliostat::normal_ = normal_;
}

const int2 &Heliostat::getRowAndColumn() const {
    return row_col_;
}

void Heliostat::setRowAndColumn(int2 row_col_) {
    Heliostat::row_col_ = row_col_;
}

const float2 &Heliostat::getGap() const {
    return gap_;
}

void Heliostat::setGap(float2 gap_) {
    Heliostat::gap_ = gap_;
}

SubCenterType Heliostat::getType() const {
    return type_;
}

void Heliostat::setType(SubCenterType type_) {
    Heliostat::type_ = type_;
}

float Heliostat::getPixelLength() const {
    return pixel_length_;
}

void Heliostat::setPixelLength(float pixel_length_) {
    Heliostat::pixel_length_ = pixel_length_;
}