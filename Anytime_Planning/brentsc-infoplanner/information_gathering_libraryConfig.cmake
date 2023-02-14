# - Config file for the InformationGathering package
# It defines the following variables
#  Information_Gathering_INCLUDE_DIRS - include directories for Information Gathering
#  Information_Gathering_LIBRARIES    - libraries to link against

# Compute paths
get_filename_component(INFORMATION_GATHERING_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(INFORMATION_GATHERING_INCLUDE_DIRS "${INFORMATION_GATHERING_CMAKE_DIR}/../../../include")

set(INFORMATION_GATHERING_LIBRARIES
        "${INFORMATION_GATHERING_CMAKE_DIR}/../../../lib/libinformationGathering.so")
