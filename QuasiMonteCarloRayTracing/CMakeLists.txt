message("QMCRT construction begins...")

# required cmake version
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -O3;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB_RECURSE CURRENT_SOURCES  *.cpp *.cu)
#set(CURRENT_HEADERS QuasiMonteCarloRayTracer.h RectangleReceiverRectGridRayTracing.cuh)
#set(CURRENT_SOURCES QuasiMonteCarloRayTracer.cpp RectangleReceiverRectGridRayTracing.cu)

cuda_add_library(quasiMonteCarloRayTracing STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_include_directories(datastructure PUBLIC "${CMAKE_SOURCE_DIR}/Util")
target_include_directories(datastructure PUBLIC "${CMAKE_SOURCE_DIR}/Scene")

target_link_libraries(quasiMonteCarloRayTracing datastructure)

message("QMCRT construction ends...")