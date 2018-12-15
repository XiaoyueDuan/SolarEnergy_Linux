#!/bin/bash

input_path=InputFiles/CompleteInputFilesExample
configure_path=$input_path/twoReceiverConfiguration.json
scene_path=$input_path/twoReceiverScene.scn
heliostat_index=$input_path/heliostat_index.txt

output_path=OutputFiles/CompleteOutputFilesExample/

# Generate makefile and Build
cd cmake-build-debug
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
make SolarEnergyRayTracing -j 4

# Run
cd ..
./cmake-build-debug/SolarEnergyRayTracing -c $configure_path -s $scene_path -h $heliostat_index -o $output_path