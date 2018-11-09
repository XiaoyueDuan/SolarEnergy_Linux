#include "Heliostat.cuh"

const float3 &Heliostat::getPosition() const {
    return pos_;
}

void Heliostat::setPosition(float3 pos) {
    pos_ = pos;
}

const float3 &Heliostat::getSize() const {
    return size_;
}

void Heliostat::setSize(float3 size) {
    size_ = size;
}

const float3 &Heliostat::getNormal() const {
    return normal_;
}

void Heliostat::setNormal(float3 normal) {
    normal_ = normal;
}

const int2 &Heliostat::getRowAndColumn() const {
    return row_col_;
}

void Heliostat::setRowAndColumn(int2 row_col) {
    row_col_ = row_col;
}

const float2 &Heliostat::getGap() const {
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