cmake_minimum_required(VERSION 3.18)
#list(APPEND CMAKE_PREFIX_PATH "/opt/anaconda3/envs/openSAXS")
project(cudaSAXS LANGUAGES CXX C CUDA)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)


set(CUDASAXS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDASAXS_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
# CUDA architectures - set explicitly to avoid auto-detection issues
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures to build for" FORCE)
add_subdirectory(Utilities)
add_subdirectory(System)
add_subdirectory(Saxs)
add_subdirectory(Exec)

pybind11_add_module(pycusaxs_cuda
    pybind/cuda_bindings.cpp
    cudaSAXS.cu
)

target_include_directories(pycusaxs_cuda
    PRIVATE
        ${CUDASAXS_PUBLIC_INCLUDE_DIR}
        ${CUDASAXS_SOURCE_DIR}/Exec
)

target_link_libraries(pycusaxs_cuda PRIVATE
    pybind11::module
    -Wl,--whole-archive
    cudaSAXS::exec
    cudaSAXS::saxs
    cudaSAXS::system
    cudaSAXS::utils
    -Wl,--no-whole-archive
    CUDA::cudart
    Threads::Threads
    fmt::fmt
    ${CMAKE_DL_LIBS}
    ${CUDAToolkit_rt_LIBRARY}
)

set_target_properties(pycusaxs_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CXX_VISIBILITY_PRESET default
    CUDA_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF

)

if(DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS pycusaxs_cuda
            LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
            ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}
            RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR})
endif()
