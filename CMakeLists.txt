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

add_library(struct STATIC
    lib/structs.cpp
)

add_library(solve STATIC
    lib/central.cpp
    lib/predict.cpp
    lib/solve.cpp
)

add_library(gen STATIC
    lib/generate.cpp
)

target_link_libraries(solve
    superlu::superlu    # SuperLU library
    struct
)

target_link_libraries(gen
    struct
)

add_executable(lpsolver bin/lpsolver.cpp)
add_executable(generate_problem bin/generate_problem.cpp)

target_link_libraries(lpsolver
    solve
)

target_link_libraries(generate_problem
    gen
)
