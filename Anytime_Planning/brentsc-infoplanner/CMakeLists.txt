CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(information_gathering_library)

# Select a default build configuration if none was chosen
IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF ()
SET(CMAKE_CXX_STANDARD 11)

# Flag for Building Tests.
option(BUILD_TESTS "Build Tests" ON)
option(PYTHON_BINDINGS "Build Python Bindings" ON)
option(OLD_YAML "Build with OLD YAML" ON)  # This is used for backwards compatibility with standard yaml-cpp 0.5.2


# Add Dependencies
find_package(PkgConfig)
find_package(Eigen3 REQUIRED eigen3)
PKG_CHECK_MODULES(YAMLCPP REQUIRED yaml-cpp)
FIND_PACKAGE(Boost COMPONENTS  system  filesystem REQUIRED)
find_package(CGAL REQUIRED)


# Set output paths.
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

INCLUDE_DIRECTORIES(
        include
        ${EIGEN3_INCLUDE_DIRS}
        ${YAMLCPP_INCLUDE_DIRS}
        ${CGAL_INCLUDE_DIRS}
)

# Set link libraries
LIST(APPEND LINK_LIBS
        ${YAMLCPP_LIBRARIES}
        ${EIGEN3_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${CGAL_LIBRARY}
        ${Boost_LIBRARIES}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
)

if(OLD_YAML)
    add_definitions(-DOLD_YAML)
endif()

#####################################################################################################
# Build Library.
add_library(informationGathering src/robot.cpp src/infoplanner.cpp)
target_link_libraries(informationGathering ${LINK_LIBS})

# Build Binary
add_executable(test_info_gathering_ARVI src/robot.cpp test/integration/test_info_gathering_ARVI.cpp src/infoplanner.cpp)
target_link_libraries(test_info_gathering_ARVI ${LINK_LIBS})

#####################################################################################################

# Build Python Bindings
if (PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

#####################################################################################################

# Testing
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

#####################################################################################################

# Install
install(FILES "${PROJECT_NAME}Config.cmake"
        DESTINATION "share/${PROJECT_NAME}/cmake")

install(TARGETS informationGathering
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/igl
        DESTINATION include
        )
