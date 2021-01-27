cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(Gaussian-Mixture-Model)
option(LIB_OPT "Compile shared libraries (ON) or static (OFF)" OFF)
option(BUILD_SAMPLES "Build the samples showing how to use the package" ON)
set(EIGEN3_FOLDER "C:\\Program Files\\Eigen3" CACHE STRING "C:\\Program Files\\Eigen3")

add_subdirectory(Lib)
if(BUILD_SAMPLES)
    add_subdirectory(Samples)
endif()
