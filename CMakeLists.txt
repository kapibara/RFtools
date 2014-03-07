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
#   set(OpenCV_DIR "C:/openCV/custom_build")
    set(BOOST_ROOT "D:/boost_1_54_0")
endif()

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${HOME}/src)
include_directories(${Forest_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

link_directories(${OpenCV_LIB_DIR})
link_directories(${Boost_LIBRARY_DIRS})

#message("Boost_LIBRARY_DIRS = ${Boost_LIBRARY_DIRS}")

set(CMAKE_CXX_FLAGS "-pg ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-pg ${CMAKE_EXE_LINKER_FLAGS}")

if(WIN32)
    set(Boost_LIBS boost_filesystem-vc100-mt-1_54)
    link_directories(${Boost_LIBRARY_DIRS})
else()
    set(Boost_LIBS boost_filesystem boost_system)
    link_directories("/usr/lib64/")
endif()
#get_cmake_property(_variableNames VARIABLES)
#foreach(_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

#set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} /Od")

add_subdirectory(${HOME}/src)
