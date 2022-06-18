cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

option(LIB_OPT "Compile shared libraries (ON) or static (OFF)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_SOURCE_DIR}/cmake/MakeLibrary.cmake)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

project(Gaussian-Mixture-Model)
add_subdirectory(src)

option(BUILD_GaussianMixtureModels_SAMPLES "Build the samples showing how to use the package" ON)
if(BUILD_GaussianMixtureModels_SAMPLES)
	project(GaussianMixtureModels-Samples)
    add_subdirectory(samples)
endif()

option(BUILD_GaussianMixtureModels_TESTS "" OFF)
if(BUILD_GaussianMixtureModels_TESTS)
	project(GaussianMixtureModels-Tests)
    add_subdirectory(tests)
endif()
