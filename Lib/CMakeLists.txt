set(PROJECT_SHORTNAME "GMM")
project(${PROJECT_SHORTNAME} VERSION ${VERSION} LANGUAGES CXX)
string(REPLACE - _ COMPONENT_NAME ${PROJECT_NAME})

list(APPEND GMM_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/header/GMM.h
	${CMAKE_CURRENT_SOURCE_DIR}/header/Error.h
	${CMAKE_CURRENT_SOURCE_DIR}/header/TrainSet.h
	${CMAKE_CURRENT_SOURCE_DIR}/header/KMeans.h
	${CMAKE_CURRENT_SOURCE_DIR}/source/GMM.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/source/GMMDivergence.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/source/GMMExpectationMaximization.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/source/GMMSampler.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/source/TrainSet.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/source/KMeans.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/source/Commons.h
	${CMAKE_CURRENT_SOURCE_DIR}/source/Commons.cpp
)

if(LIB_OPT)
	add_library(${PROJECT_NAME} SHARED ${GMM_SOURCES})
else()
	add_library(${PROJECT_NAME} STATIC ${GMM_SOURCES})
endif()
add_library(GMM::${PROJECT_SHORTNAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME}
    PUBLIC cxx_auto_type
    PRIVATE cxx_variadic_templates
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/header ${EIGEN3_FOLDER}>
        $<INSTALL_INTERFACE:header ${EIGEN3_FOLDER}>
)