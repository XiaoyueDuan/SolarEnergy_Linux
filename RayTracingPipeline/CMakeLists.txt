message("Ray Tracing Pipeline begins...")

# General Configuration
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

file(GLOB_RECURSE CURRENT_HEADERS  *.h)
file(GLOB_RECURSE CURRENT_SOURCES  *.cpp)
cuda_add_library(rayTracingPipeline STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(rayTracingPipeline datastructure)
target_include_directories(rayTracingPipeline PUBLIC "${CMAKE_SOURCE_DIR}/Util")
target_include_directories(rayTracingPipeline PUBLIC "${CMAKE_SOURCE_DIR}/Scene")

message("Ray Tracing Pipeline ends...")