file(GLOB_RECURSE LIB_SOURCES ${SOURCE_WILDCARDS})
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
add_library(ICTGV ${LIB_SOURCES})
cuda_add_cufft_to_target(ICTGV)

#main program
add_executable(avionic_tgv3d main.cpp)
target_link_libraries(avionic_tgv3d ICTGV ${Boost_LIBRARIES})

set_target_properties(avionic_tgv3d
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

