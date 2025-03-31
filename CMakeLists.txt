cmake_minimum_required(VERSION 3.11)

project(lpsolver)

enable_language(CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(INFO "Debug info" OFF)
if (DAY)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DINFO")
endif()

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(structs STATIC
    lib/structs.cpp
)

add_library(generate
    lib/generate.cpp
)
target_link_libraries(generate
    structs
)

add_executable(generate_problem
    src/generate_problem.cpp
)
target_link_libraries(generate_problem
    generate
)

set(CUDSS_ARCHIVE "cudss.tar.xz")
set(CUDSS_DIR "libcudss-linux-x86_64-0.5.0.16_cuda12-archive")
set(CUDSS_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${CUDSS_DIR}/lib/libcudss.so"
    "${CMAKE_CURRENT_BINARY_DIR}/${CUDSS_DIR}/include/cudss_distributed_interface.h"
    "${CMAKE_CURRENT_BINARY_DIR}/${CUDSS_DIR}/include/cudss.h"
    "${CMAKE_CURRENT_BINARY_DIR}/${CUDSS_DIR}/include/cudss_threading_interface.h"
)

add_custom_command(
    OUTPUT "${CUDSS_ARCHIVE}"
    COMMAND wget "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.5.0.16_cuda12-archive.tar.xz" -O "${CUDSS_ARCHIVE}"
    COMMENT "Downloading cudss..."
)

add_custom_command(
    OUTPUT ${CUDSS_FILES}
    COMMAND ${CMAKE_COMMAND} -E tar xJvf "${CUDSS_ARCHIVE}" "${CUDSS_DIR}/lib/libcudss.so" "${CUDSS_DIR}/include/cudss_distributed_interface.h" "${CUDSS_DIR}/include/cudss.h" "${CUDSS_DIR}/include/cudss_threading_interface.h"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${CUDSS_ARCHIVE}"
    COMMENT "Unpacking cudss..."
)

add_custom_target(cudss_in DEPENDS ${CUDSS_FILES})

include_directories("${CMAKE_CURRENT_BINARY_DIR}/${CUDSS_DIR}/include")

add_library(cudss SHARED IMPORTED)
set_target_properties(lib_cudss PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${CUDSS_DIR}/lib/libcudss.so")
add_dependencies(cudss
    cudss_in
)

add_library(ludec STATIC
    lib/ludec.cu
)
target_link_libraries(ludec
    lib_cudss
)

add_library(solve STATIC
    lib/matrices.cpp
    lib/cpptocu.cpp
    lib/central.cpp
    lib/predict.cpp
    lib/solve.cpp
)
target_link_libraries(solve
    structs
    ludec
)

add_executable(lpsolver
    src/lpsolver.cpp
)
target_link_libraries(lpsolver
    solve
)