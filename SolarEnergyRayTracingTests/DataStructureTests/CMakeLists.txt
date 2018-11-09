message("---------- Data Structure Test cmake begins ---------- ")

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -O3;-G;-g)

# Test Target
file(GLOB CURRENT_SOURCES *.cpp *.cu)
message("Test files name: ${CURRENT_SOURCES}")
cuda_add_executable(DataStructureTests ${CURRENT_SOURCES})
target_link_libraries(DataStructureTests gtest gtest_main)

# Add Google Tests
include_directories("${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}")

# Add the files to be tested
# 1. scene configuration
include_directories("${CMAKE_SOURCE_DIR}/Scene")
#target_link_libraries(DataStructureTests sceneConfigurator)
#target_link_libraries(DataStructureTests solarScene)
#target_link_libraries(DataStructureTests sceneLoader)

target_link_libraries(DataStructureTests datastructure)

message("---------- Data Structure Test cmake ends ---------- ")
