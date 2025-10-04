cmake_minimum_required(VERSION 3.20)
project(pycusaxs VERSION 0.1.0 LANGUAGES CXX CUDA)

# Disable post-build stripping so CUDA device registration symbols remain intact
# in the generated Python extension module.
set(CMAKE_STRIP "")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt CONFIG REQUIRED)

if(NOT TARGET CUDA::cudadevrt)
    find_library(CUDA_cudadevrt_LIBRARY
        NAMES cudadevrt
        HINTS
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDAToolkit_ROOT}
            $ENV{CUDA_PATH}
            /usr/local/cuda
            /opt/cuda
        PATH_SUFFIXES
            lib
            lib64
            targets/x86_64-linux/lib
            targets/x86_64-linux/lib/stubs
    )
    if(CUDA_cudadevrt_LIBRARY)
        add_library(CUDA::cudadevrt STATIC IMPORTED)
        set_target_properties(CUDA::cudadevrt PROPERTIES IMPORTED_LOCATION "${CUDA_cudadevrt_LIBRARY}")
    else()
        message(FATAL_ERROR "Failed to locate cudadevrt library required for CUDA device linking.")
    endif()
endif()

add_subdirectory(cpp-src)
