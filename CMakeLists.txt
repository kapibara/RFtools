cmake_minimum_required(VERSION 2.8)

project(DepthRF)

set(HOME ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_INSTALL_PREFIX ${HOME}/bin)

set(DEV_BASE "/home/kuznetso/Projects/CPP/")

set(Forest_DIR "${DEV_BASE}Sherwood/cpp/lib/")

if(WIN32)
    set(OpenCV_DIR "${DEV_BASE}custom_build/")
endif()

find_package(OpenCV REQUIRED)

include_directories(${HOME}/include)
include_directories(${Forest_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OPENNI_LIBS})


#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

#set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} /Od")

add_subdirectory(${HOME}/src)
