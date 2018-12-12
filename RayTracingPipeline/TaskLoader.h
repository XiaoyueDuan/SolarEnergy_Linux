//
// Created by dxt on 18-12-12.
//

#ifndef SOLARENERGYRAYTRACING_TASKLOADER_H
#define SOLARENERGYRAYTRACING_TASKLOADER_H

#include <vector>
#include <unordered_set>
#include <string>

#include <SolarScene.h>
#include <bits/unordered_set.h>

class TaskLoader {
public:
    TaskLoader(){}

    bool loadRayTracingHeliostatIndex(std::string task_path, SolarScene &solarScene);

    const std::unordered_set<int> &getHeliostatIndexesArray() const;
    const std::unordered_set<int> &getReceiverIndexesArray() const;

private:
    std::unordered_set<int> heliostat_indexes;
    std::unordered_set<int> receiver_indexes;
};


#endif //SOLARENERGYRAYTRACING_TASKLOADER_H
