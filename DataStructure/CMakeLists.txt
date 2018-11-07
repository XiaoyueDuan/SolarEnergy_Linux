#project(datastructure)
message("---------- DataStructure cmake begins ----------")

# required cmake version
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# headers location
include_directories("${CMAKE_SOURCE_DIR}/Util")
include_directories("${CMAKE_SOURCE_DIR}/Tool")
include_directories("${CMAKE_SOURCE_DIR}/DataStructure/Heliostat"
        "${CMAKE_SOURCE_DIR}/DataStructure/Grid"
        "${CMAKE_SOURCE_DIR}/DataStructure/Receiver"
        "${CMAKE_SOURCE_DIR}/DataStructure/Sunray")
#add_subdirectory("${CMAKE_SOURCE_DIR}/Util")

# packages
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -O3;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
list(FILTER CURRENT_HEADERS EXCLUDE REGEX "Scene/.*") # exclude files under Scene directory
file(GLOB_RECURSE CURRENT_SOURCES  *.cpp *.cu)
list(FILTER CURRENT_SOURCES EXCLUDE REGEX "Scene/.*") # exclude files under Scene directory

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

cuda_add_library(datastructure STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

# add Scene
add_subdirectory(Scene)
target_link_libraries(datastructure sceneConfigurator)

message("----------  DataStructure cmake ends ---------- ")