project (ICTGV CXX)

ENABLE_LANGUAGE(C)

cmake_policy(SET CMP0074 NEW)
list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS c)


cmake_minimum_required (VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)

set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
message (STATUS "Library destination directory: ${CMAKE_SOURCE_DIR}/lib")

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  message (STATUS "Switching to default configuration 'Release'")
  set (CMAKE_BUILD_TYPE "Release")
endif()

# Dependencies 
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

# CUDA 
find_package (CUDA 5.0 REQUIRED)
include_directories (${CUDA_INCLUDE_DIRS})
link_libraries (${CUDA_LIBRARIES})
link_libraries (${CUDA_CUBLAS_LIBRARIES})

# AGILE
find_package (AGILE REQUIRED)
include_directories(${AGILE_INCLUDE_DIRS})
link_libraries (${AGILE_LIBRARIES})

# DCMTK
# for DCMTK - Dicom ToolKit¬
#SET(DCMTK_DIR /usr/local/include/dcmtk CACHE STRING "Path to DCMTK root directory")

SET(DCMTK_DIR /usr/local/include/dcmtk CACHE STRING "Path to DCMTK root directory")
# 
ADD_DEFINITIONS(-DHAVE_CONFIG_H)

SET(DCMTK_config_INCLUDE_DIR ${DCMTK_DIR}/config)
SET(DCMTK_ofstd_INCLUDE_DIR ${DCMTK_DIR}/ofstd)
SET(DCMTK_dcmdata_INCLUDE_DIR ${DCMTK_DIR}/dcmdata)
SET(DCMTK_dcmimgle_INCLUDE_DIR ${DCMTK_DIR}/dcmimgle)

find_package (DCMTK REQUIRED)
if (DCMTK_FOUND)
  MESSAGE(STATUS "DCMTK_FOUND")
endif()

link_libraries(${DCMTK_LIBRARIES} oflog ofstd pthread z)

# gpuNUFFT
find_package (GPUNUFFT REQUIRED)
include_directories(${GPUNUFFT_INCLUDE_DIRS})
link_libraries (${GPUNUFFT_LIBRARIES})

# Boost
find_package(Boost 1.58.0 REQUIRED system program_options regex filesystem)
include_directories(${Boost_INCLUDE_DIR})

# ISMRMRD
SET(ISMRMRD_HOME /usr/local CACHE STRING "Path to ISMRMRD install directory")
list(APPEND CMAKE_MODULE_PATH "${ISMRMRD_HOME}/share/ismrmrd/cmake")

find_package(ISMRMRD REQUIRED)
include_directories(${ISMRMRD_INCLUDE_DIR})
link_libraries(${ISMRMRD_LIBRARIES})

set(CUDA_NVCC_FLAGS "-arch;sm_35")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

set(SOURCE_WILDCARDS *.h *.hpp *.hxx *.c *.cpp *.cc *.cxx)

add_subdirectory (include)
add_subdirectory (src)
add_subdirectory (doc)

set(OPTION_ENABLE_TESTS)
