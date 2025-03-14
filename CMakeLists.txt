cmake_minimum_required(VERSION 3.10)
project(lpsolver)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 26)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -mavx2 -msse2 -msse3 -mssse3 -msse4 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_compile_options(-Wall -Wextra -Wshadow)

option(INFO "Debug info mode" OFF)
option(SUPER "SuperLU mode" OFF)
option(SAN "Sanitizers" OFF)

if(SAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,bounds,undefined,leak -g")
endif()

if(INFO)
    add_compile_options(-DINFO)
endif()

if(SUPER)
    find_package(superlu REQUIRED)
    include_directories(/usr/include/superlu)
    add_compile_options(-DSUPER)
endif()

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

add_library(matrices STATIC
    lib/matrices.cpp
)

if(SUPER)
    target_link_libraries(solve
        superlu::superlu
        struct
    )
else()
    target_link_libraries(solve
        struct
    )
endif()

target_link_libraries(gen
    struct
)

add_executable(lpsolver src/lpsolver.cpp)
add_executable(generate_problem src/generate_problem.cpp)

target_link_libraries(lpsolver
    solve
    matrices
)

target_link_libraries(generate_problem
    gen
)
