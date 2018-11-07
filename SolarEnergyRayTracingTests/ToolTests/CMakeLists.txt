message("---------- Tool Test cmake begins ---------- ")

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -O3;-G;-g)

# Test Target
file(GLOB CURRENT_SOURCES *.cpp *.cu)
message("Test files name: ${CURRENT_SOURCES}")
cuda_add_executable(ToolTests ${CURRENT_SOURCES})
target_link_libraries(ToolTests gtest gtest_main)

# Add Google Tests
include_directories("${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}")
#set(GTEST_DIR "../lib/googletest-master"
#        CACHE PATH "The path to the GoogleMock test framework.")
#add_subdirectory(${GTEST_DIR} ${CMAKE_BINARY_DIR}/gtest)

# Add Util
include_directories("${CMAKE_SOURCE_DIR}/Util")

# Add the files to be tested
include_directories("${CMAKE_SOURCE_DIR}/Tool/ArgumentParser")
target_link_libraries(ToolTests argumentParser)

include_directories("${CMAKE_SOURCE_DIR}/Tool/RandomNumberGenerator")
#add_subdirectory(${CMAKE_SOURCE_DIR}/Tool/RandomNumberGenerator ${CMAKE_BINARY_DIR}/RandomNumberGenerator)
target_link_libraries(ToolTests randomGenerator)

include_directories("${CMAKE_SOURCE_DIR}/Tool/FileLoader")
target_link_libraries(ToolTests regularExpressionTree)

message("---------- Tool Test cmake ends ---------- ")
