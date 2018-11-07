#include "Grid.cuh"

int Grid::getGridType() const {
    return type_;
}

void Grid::setGridType(int type_) {
    Grid::type_ = type_;
}

int Grid::getHeliostatType() const {
    return helio_type_;
}

void Grid::setHeliostatType(int helio_type_) {
    Grid::helio_type_ = helio_type_;
}

int Grid::getStartHeliostatPosition() const {
    return start_helio_pos_;
}

void Grid::setStartHeliostatPosition(int start_helio_pos_) {
    Grid::start_helio_pos_ = start_helio_pos_;
}

int Grid::getNumberOfHeliostats() const {
    return num_helios_;
}

void Grid::setNumberOfHeliostats(int num_helios_) {
    Grid::num_helios_ = num_helios_;
}
