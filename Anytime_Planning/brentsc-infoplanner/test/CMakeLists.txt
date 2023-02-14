## Build Tests for Information Gathering Library

CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(information_gathering_library_test)

SET(CMAKE_CXX_STANDARD 11)


# Test Dependencies
find_package(PkgConfig)
find_package(Eigen3 REQUIRED eigen3)
PKG_CHECK_MODULES(YAMLCPP REQUIRED yaml-cpp)
FIND_PACKAGE(Boost COMPONENTS system filesystem REQUIRED)
find_package(CGAL REQUIRED)

#################################################################################
# Unit Tests
#add_executable(basic_test ${CMAKE_CURRENT_SOURCE_DIR}/unit/basic_test.cpp)
#add_test(NAME basic_test
#        COMMAND ${CMAKE_SOURCE_DIR}/bin/basic_test
#        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
#
#add_executable(target_test ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_target_model.cpp)
#target_link_libraries(target_test ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES})
#add_test(NAME target_test COMMAND ${CMAKE_SOURCE_DIR}/bin/target_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
#
#add_executable(info_target_test ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_info_target_model.cpp)
#target_link_libraries(info_target_test ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES})
#add_test(NAME info_target_test COMMAND ${CMAKE_SOURCE_DIR}/bin/target_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

#################################################################################
# Integration Tests (Multi-Robot Scenarios)
add_test(NAME test_one_robot_ARVI
        COMMAND ${CMAKE_SOURCE_DIR}/bin/test_info_gathering_ARVI ${CMAKE_SOURCE_DIR}/test/integration/config/test_one_robot_ARVI.yaml
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_two_robot_ARVI
        COMMAND ${CMAKE_SOURCE_DIR}/bin/test_info_gathering_ARVI ${CMAKE_SOURCE_DIR}/test/integration/config/test_two_robot_ARVI.yaml
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

