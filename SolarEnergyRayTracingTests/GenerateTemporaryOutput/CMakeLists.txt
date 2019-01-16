cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)

# Test Target
file(GLOB CURRENT_SOURCES *.cpp)
cuda_add_executable(GenerateTemporaryOutput ${CURRENT_SOURCES})

target_link_libraries(GenerateTemporaryOutput gtest gtest_main)
target_link_libraries(GenerateTemporaryOutput gmock gmock_main)

# Add the files will be used
include_directories("${CMAKE_SOURCE_DIR}/Util")
include_directories("${CMAKE_SOURCE_DIR}/Tool/ImageSaver")
include_directories("${CMAKE_SOURCE_DIR}/Scene")
include_directories("${CMAKE_SOURCE_DIR}/QuasiMonteCarloRayTracing")

target_link_libraries(GenerateTemporaryOutput datastructure)
target_link_libraries(GenerateTemporaryOutput image_saver)
target_link_libraries(GenerateTemporaryOutput quasiMonteCarloRayTracing)
