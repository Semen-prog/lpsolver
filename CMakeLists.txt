cmake_minimum_required(VERSION 3.11)

project(lpsolver)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

option(INFO "Debug info" OFF)
if (INFO)
    message(STATUS "Debug info ON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DINFO")
else()
    message(STATUS "Debug info OFF")
endif()

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -O3")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/LBFGSpp")

add_library(structs STATIC
    lib/structs.cpp
)
target_link_libraries(structs
    Eigen3::Eigen
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

file(DOWNLOAD
    "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.5.0.16_cuda12-archive.tar.xz"
    "${CUDSS_ARCHIVE}"
    EXPECTED_HASH SHA256=5245d2ba26a590839e2f1dd074f87e39ee5cc201c3b29245b35c7060d59c37a5
    SHOW_PROGRESS
)

file(ARCHIVE_EXTRACT
    INPUT "${CUDSS_ARCHIVE}"
)

include_directories("${CMAKE_CURRENT_BINARY_DIR}/${CUDSS_DIR}/include")

add_library(cudss SHARED IMPORTED)
set_target_properties(cudss PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${CUDSS_DIR}/lib/libcudss.so")

set_target_properties(cudss PROPERTIES CUDA_ARCHITECTURES "86")

add_library(ludec STATIC
    lib/ludec.cu
)
target_link_libraries(ludec
    cudss
    ${CUDA_LIBRARIES}
)

set_target_properties(ludec PROPERTIES CUDA_ARCHITECTURES "86")

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

