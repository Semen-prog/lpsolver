cmake_minimum_required(VERSION 3.10)
project(lpsolver)

add_compile_options(-Wall -Wextra -Wshadow)

option(DEBINFO "Debug info mode" OFF)

if(DEBINFO)
    add_compile_options(-DINFO)
endif()

find_package(superlu REQUIRED)

include_directories(/usr/include/superlu)
include_directories(/usr/include/eigen3)
include_directories(include)
include_directories(include/LBFGSpp)

add_library(libsolve STATIC
    lib/structs.cpp
    lib/central.cpp
    lib/predict.cpp
    lib/solve.cpp
)

target_link_libraries(libsolve
    superlu::superlu    # SuperLU library
)

add_executable(lpsolver bin/lpsolver.cpp)

target_link_libraries(lpsolver
    libsolve
)