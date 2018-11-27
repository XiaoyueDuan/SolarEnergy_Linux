message("---------- QMCRT Test cmake begins ---------- ")

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)
#set(CUDA_NVCC_FLAGS -O3;-G;-g)

# Test Target
file(GLOB CURRENT_SOURCES *.cpp *.cu)
cuda_add_executable(QMCRT_Tests ${CURRENT_SOURCES})
target_link_libraries(QMCRT_Tests gtest gtest_main)
target_link_libraries(QMCRT_Tests gmock gmock_main)

# Add the files to be tested
include_directories("${CMAKE_SOURCE_DIR}/QuasiMonteCarloRayTracing")
target_link_libraries(QMCRT_Tests quasiMonteCarloRayTracing)

message("---------- QMCRT Test cmake ends ---------- ")
