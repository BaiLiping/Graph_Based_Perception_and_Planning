CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project(igl_python)

FIND_PACKAGE(pybind11 REQUIRED)
if (pybind11_FOUND)
    pybind11_add_module(pyInfoGathering MODULE bindings.cpp ${CMAKE_SOURCE_DIR}/src/robot.cpp
            ${CMAKE_SOURCE_DIR}/src/infoplanner.cpp)
    set_target_properties(pyInfoGathering PROPERTIES PREFIX "" SUFFIX ".so")
    target_link_libraries(pyInfoGathering PRIVATE ${Boost_LIBRARIES} ${YAMLCPP_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY})
endif()