cmake_minimum_required(VERSION 2.8)

project(DepthRF)

set(HOME ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_INSTALL_PREFIX ${HOME}/bin)

if(WIN32)
    set(DEV_BASE "C:/Data/Development/CPP/")
else()
    set(DEV_BASE "/home/kuznetso/Projects/CPP/")
endif()

set(Forest_DIR "${DEV_BASE}MSRFlib/lib/")

if(WIN32)
    set(OpenCV_DIR "${DEV_BASE}custom_build/")
endif()

find_package(OpenCV REQUIRED)

include_directories(${HOME}/src)
include_directories(${Forest_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIB_DIR})

#get_cmake_property(_variableNames VARIABLES)
#foreach(_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

#set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} /Od")

add_subdirectory(${HOME}/src)
