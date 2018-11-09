//
// Created by dxt on 18-11-6.
//

#include <fstream>
#include <iostream>

#include "SceneLoader.h"

#include "RectangleHelio.cuh"
#include "RectangleReceiver.cuh"
#include "RectGrid.cuh"

float3 setFloat3Field(std::string field_name, std::istream &stringstream) {
    string head;
    stringstream >> head;

    if (field_name != head) {
        throw std::runtime_error("Miss '" + field_name + "' property.\n");
    }
    float3 ans;
    stringstream >> ans.x >> ans.y >> ans.z;
    return ans;
}

int setIntField(std::string field_name, std::istream &stringstream) {
    string head;
    stringstream >> head;

    if (field_name != head) {
        throw std::runtime_error("Miss '" + field_name + "' property.\n");
    }
    int ans;
    stringstream >> ans;
    return ans;
}

void checkIsolatedField(std::string field_name, std::istream &stringstream) {
    string head;
    stringstream >> head;

    if (field_name != head) {
        throw std::runtime_error("Miss '" + field_name + "' property.\n");
    }
}

void SceneLoader::add_ground(SolarScene *solarScene, std::istream &stringstream) {
    float ground_length, ground_width;
    stringstream >> ground_length >> ground_width;
    solarScene->setGroundLength(ground_length);
    solarScene->setGroundWidth(ground_width);

    solarScene->setNumberOfGrid(setIntField("ngrid", stringstream));
}

void SceneLoader::add_receiver(SolarScene *solarScene, std::istream &stringstream) {
    int type;
    stringstream >> type;

    Receiver *receiver = nullptr;
    switch (type) {
        case 0:
            receiver = new RectangleReceiver();
            break;
        default:
            throw std::runtime_error("Receiver type are not defined.\n");
    }

    receiver->setType(type);
    receiver->setPosition(setFloat3Field("pos", stringstream));
    receiver->setSize(setFloat3Field("size", stringstream));
    receiver->setNormal(setFloat3Field("norm", stringstream));
    receiver->setFaceIndex(setIntField("face", stringstream));
    checkIsolatedField("end", stringstream);

    solarScene->addReceiver(receiver);
}

int SceneLoader::add_grid(SolarScene *solarScene,
                          std::istream &stringstream, int receiver_index, int heliostat_start_index) {
    int type;
    stringstream >> type;

    Grid *grid = nullptr;
    switch (type) {
        case 0:
            grid = new RectGrid();
            break;
        default:
            throw std::runtime_error("Grid type are not defined.\n");
    }

    grid->setGridType(type);
    grid->setBelongingReceiverIndex(receiver_index);
    grid->setStartHeliostatPosition(heliostat_start_index);

    grid->setPosition(setFloat3Field("pos", stringstream));
    grid->setSize(setFloat3Field("size", stringstream));
    grid->setInterval(setFloat3Field("inter", stringstream));
    grid->setNumberOfHeliostats(setIntField("n", stringstream));
    int heliostat_type = setIntField("type", stringstream);
    grid->setHeliostatType(heliostat_type);
    checkIsolatedField("end", stringstream);

    solarScene->addGrid(grid);
    return heliostat_type;
}

void SceneLoader::add_heliostat(SolarScene *solarScene, std::istream &stringstream,
                                int type, float2 gap, int2 matrix) {
    Heliostat *heliostat = nullptr;
    switch (type) {
        case 0:
            heliostat = new RectangleHelio();
            break;
        default:
            throw std::runtime_error("Heliostat type are not defined.\n");
    }
    heliostat->setGap(gap);
    heliostat->setRowAndColumn(matrix);

    float3 position, size;
    stringstream >> position.x >> position.y >> position.z;
    stringstream >> size.x >> size.y >> size.z;
    heliostat->setPosition(position);
    heliostat->setSize(size);

    solarScene->addHeliostat(heliostat);
}

bool SceneLoader::SceneFileRead(SolarScene *solarScene, std::string filepath) {
    int2 matrix = make_int2(1, 1);
    float2 gap = make_float2(0.0f, 0.0f);
    std::string head;
    int receiver_id = -1;
    int current_total_heliostat = 0;
    int heliostat_type = -1;
    int line_id = 0;

    try {
        std::ifstream scene_file(filepath);
        if (scene_file.fail()) {
            throw std::runtime_error("Cannot open the file.");
        }
        stringstream scene_stream;
        scene_stream << scene_file.rdbuf();
        scene_file.close();

        current_status = sceneRETree_.getRoot();
        while (scene_stream >> head) {
            ++line_id;

            if (head[0] == '#') {
                std::string comment;
                getline(scene_stream, comment);
                continue;
            }

            if (head == "gap") {
                scene_stream >> gap.x >> gap.y;
            } else if (head == "matrix") {
                scene_stream >> matrix.x >> matrix.y;
            } else if (head == "ground") {
                current_status = sceneRETree_.step_forward(current_status, 'D');
                add_ground(solarScene, scene_stream);
            } else if (head == "Recv") {
                current_status = sceneRETree_.step_forward(current_status, 'R');
                add_receiver(solarScene, scene_stream);
                ++receiver_id;
            } else if (head == "Grid") {
                current_status = sceneRETree_.step_forward(current_status, 'G');
                heliostat_type = add_grid(solarScene, scene_stream, receiver_id, current_total_heliostat);
            } else if (head == "helio") {
                current_status = sceneRETree_.step_forward(current_status, 'H');
                add_heliostat(solarScene, scene_stream, heliostat_type, gap, matrix);
                ++current_total_heliostat;
            } else {
                current_status = sceneRETree_.step_forward(current_status, '?');
            }
        }

        return true;
    } catch (std::runtime_error runtime_error1) {
        std::cerr << "Error occurs at line " << line_id << " with head '" << head << "'.\nThis is caused by '"
                  << runtime_error1.what() << "'" << std::endl;
        return false;
    }
}
