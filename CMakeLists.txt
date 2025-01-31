cmake_minimum_required(VERSION 3.10)
project(lpsolver)

# Find Eigen3 (header-only library)
find_package(Eigen3 REQUIRED)

# Find SuperLU (external library)
find_package(superlu REQUIRED)

include_directories(/usr/include/superlu)

# Add your executable
add_executable(lpsolver main.cpp)

# Link Eigen and SuperLU
target_link_libraries(lpsolver
    Eigen3::Eigen       # Eigen headers
    superlu::superlu    # SuperLU library
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)