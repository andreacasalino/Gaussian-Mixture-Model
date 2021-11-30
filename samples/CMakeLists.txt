option(BUILD_SAMPLE_01 "Build Sample_01" ON)
if(BUILD_SAMPLE_01)
    add_executable(Sample01 01-Kmeans.cpp Utils.h)
    target_link_libraries(Sample01 PUBLIC GaussianMixtureModels)	

    add_custom_command(TARGET Sample01 PRE_BUILD
	    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Visualize01.py ${CMAKE_CURRENT_BINARY_DIR}/Visualize01.py)
endif()

option(BUILD_SAMPLE_02 "Build Sample_02" ON)
if(BUILD_SAMPLE_02)
    add_executable(Sample02 02-GMM-2d.cpp Utils.h)
    target_link_libraries(Sample02 PUBLIC GaussianMixtureModels)	

    add_custom_command(TARGET Sample02 PRE_BUILD
	    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Visualize02.py ${CMAKE_CURRENT_BINARY_DIR}/Visualize02.py)
endif()

option(BUILD_SAMPLE_03 "Build Sample_03" ON)
if(BUILD_SAMPLE_03)
    add_executable(Sample03 03-GMM-3d.cpp Utils.h)
    target_link_libraries(Sample03 PUBLIC GaussianMixtureModels)	
	
    add_custom_command(TARGET Sample03 PRE_BUILD
	    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Visualize03.py ${CMAKE_CURRENT_BINARY_DIR}/Visualize03.py)
endif()

option(BUILD_SAMPLE_04 "Build Sample_04" ON)
if(BUILD_SAMPLE_04)
    add_executable(Sample04 04-GMM-divergence.cpp Utils.h)
    target_link_libraries(Sample04 PUBLIC GaussianMixtureModels)	
endif()
