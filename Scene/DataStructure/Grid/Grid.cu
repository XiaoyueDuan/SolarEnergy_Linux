#include "Grid.cuh"

int Grid::getGridType() const {
    return type_;
}

void Grid::setGridType(int type) {
    type_ = type;
}

void Grid::setPosition(float3 pos) {
    pos_ = pos;
}

float3 Grid::getSize() const {
    return size_;
}

void Grid::setSize(float3 size) {
    size_ = size;
}

void Grid::setInterval(float3 interval) {
    interval_ = interval;
}

int Grid::getHeliostatType() const {
    return helio_type_;
}

void Grid::setHeliostatType(int helio_type) {
    helio_type_ = helio_type;
}

int Grid::getStartHeliostatPosition() const {
    return start_helio_pos_;
}

void Grid::setStartHeliostatPosition(int start_helio_pos) {
    start_helio_pos_ = start_helio_pos;
}

int Grid::getNumberOfHeliostats() const {
    return num_helios_;
}

void Grid::setNumberOfHeliostats(int num_helios) {
    num_helios_ = num_helios;
}

int Grid::getBelongingReceiverIndex() const {
    return belonging_receiver_index_;
}

void Grid::setBelongingReceiverIndex(int belonging_receiver_index) {
    belonging_receiver_index_ = belonging_receiver_index;
}
