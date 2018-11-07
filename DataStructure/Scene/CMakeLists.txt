message("Scene construction begins...")

# required cmake version
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -O3;-G;-g)

add_library(sceneConfigurator STATIC SceneConfiguration.h SceneConfiguration.cpp)

# Configure Json
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(json-develop)
target_include_directories(sceneConfigurator PUBLIC json-develop/single_include)
target_link_libraries(sceneConfigurator PRIVATE nlohmann_json::nlohmann_json)

message("Scene construction ends...")