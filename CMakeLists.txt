cmake_minimum_required(VERSION 2.8)

project(DepthRF)

set(HOME ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_INSTALL_PREFIX ${HOME}/bin)

if(WIN32)
    set(DEV_BASE "C:/Data/Development/CPP/")
else()
    set(DEV_BASE "/home/kuznetso/Projects/CPP/")
    set(FLANN_ROOT "/home/kuznetso/bin/flann/")
endif()

set(Forest_DIR "${DEV_BASE}MSRFlib/lib/")
set(MEANSHIFT_DIR "${DEV_BASE}MeanShift")

set(MEANSHIFT_LIBS meanshift)

if(WIN32)
#   set(OpenCV_DIR "C:/openCV/custom_build")
    set(BOOST_ROOT "D:/boost_1_54_0")
endif()

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
if(WIN32)
    find_package(STXXL REQUIRED)
endif()

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

include_directories(${HOME}/src)
include_directories(${Forest_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${MEANSHIFT_DIR}/src)
if(UNIX)
    include_directories(${FLANN_ROOT}/include)
endif()

link_directories(${OpenCV_LIB_DIR})
link_directories(${Boost_LIBRARY_DIRS})
link_directories(${MEANSHIFT_DIR}/lib)

#message("Boost_LIBRARY_DIRS = ${Boost_LIBRARY_DIRS}")

set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS " ${CMAKE_EXE_LINKER_FLAGS}")

#message(STATUS "${CMAKE_CXX_FLAGS}=${${CMAKE_CXX_FLAGS}}")

if(WIN32)
    set(Boost_LIBS libboost_filesystem-vc100-mt-1_54)
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
