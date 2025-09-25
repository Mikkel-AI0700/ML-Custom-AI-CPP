# - Config file for the Armadillo package
# It defines the following variables
#  ARMADILLO_INCLUDE_DIRS - include directories for Armadillo
#  ARMADILLO_LIBRARY_DIRS - library directories for Armadillo (normally not used!)
#  ARMADILLO_LIBRARIES    - libraries to link against

# Tell the user project where to find our headers and libraries
set(ARMADILLO_INCLUDE_DIRS "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/include/armadillo/armadillo-15.0.3/include")
set(ARMADILLO_LIBRARY_DIRS "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/include/armadillo/armadillo-15.0.3/lib")

# Our library dependencies (contains definitions for IMPORTED targets)
include("/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/include/armadillo/armadillo-15.0.3/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake")

# These are IMPORTED targets created by ArmadilloLibraryDepends.cmake
set(ARMADILLO_LIBRARIES armadillo)

