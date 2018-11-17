# General Configuration
cmake_minimum_required(VERSION 3.12)
project(SolarEnergyRayTracing)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

message("main-${CMAKE_CURRENT_SOURCE_DIR}")

# Tool
add_subdirectory(Tool/ArgumentParser)
add_subdirectory(Tool/RandomNumberGenerator)
add_subdirectory(Tool/FileLoader)

# Data Structure
add_subdirectory(Scene)

# Ray Tracing
add_subdirectory(QuasiMonteCarloRayTracing)

# Tests
set(GOOGLE_TEST_PATH SolarEnergyRayTracingTests)
add_subdirectory(${GOOGLE_TEST_PATH})

# Set Libs
#set (EXTRA_LIBS ${EXTRA_LIBS} datastructure randomGenerator)

# Finial Project
add_executable(SolarEnergyRayTracing main.cpp)
#target_link_libraries (SolarEnergyRayTracing ${EXTRA_LIBS})
