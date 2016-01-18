file(GLOB_RECURSE TEST_SOURCES ${SOURCE_WILDCARDS})

add_executable(runUnitTests ${TEST_SOURCES})

target_link_libraries(runUnitTests ${GTEST_LIB} ${GTEST_MAIN_LIB} pthread ICTGV ${Boost_LIBRARIES} ${ISMRMRD_LIBRARIES})

cuda_add_cufft_to_target(runUnitTests)
