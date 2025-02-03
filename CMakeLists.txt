cmake_minimum_required(VERSION 3.10)
project(lpsolver)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 26)

add_compile_options(-Wall -Wextra -Wshadow)

option(INFO "Debug info mode" OFF)

if(INFO)
    add_compile_options(-DINFO)
endif()

find_package(superlu REQUIRED)

include_directories(/usr/include/superlu)
include_directories(/usr/include/eigen3)
include_directories(include)
include_directories(include/LBFGSpp)

add_library(libstruct STATIC
    lib/structs.cpp
)

add_library(libsolve STATIC
    lib/central.cpp
    lib/predict.cpp
    lib/solve.cpp
)

add_library(libgen STATIC
    lib/generate.cpp
)

target_link_libraries(libsolve
    superlu::superlu    # SuperLU library
    libstruct
)

target_link_libraries(libgen
    libstruct
)

add_executable(lpsolver bin/lpsolver.cpp)
add_executable(generate_problem bin/generate_problem.cpp)

target_link_libraries(lpsolver
    libsolve
)

target_link_libraries(generate_problem
    libgen
)
