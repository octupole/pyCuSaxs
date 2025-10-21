cmake_minimum_required(VERSION 3.20)

# Check for CUDA before declaring it as a project language
include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  CUDA Compiler Not Found\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "  pyCuSAXS requires NVIDIA CUDA to build.\n"
        "\n"
        "  Please ensure:\n"
        "    1. NVIDIA CUDA Toolkit is installed (version 11.0 or later)\n"
        "    2. nvcc compiler is in your PATH\n"
        "    3. CUDA_HOME or CUDA_PATH environment variable is set\n"
        "\n"
        "  Installation instructions:\n"
        "    - https://developer.nvidia.com/cuda-downloads\n"
        "    - Or use conda: conda install -c nvidia cuda-toolkit\n"
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
endif()

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

# Require a usable NVIDIA driver stack so CUDA kernels can execute.
find_program(CUSAXS_NVIDIA_SMI nvidia-smi)
if(NOT CUSAXS_NVIDIA_SMI)
    message(FATAL_ERROR
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  NVIDIA GPU Driver Not Found\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "  pyCuSAXS requires NVIDIA GPU drivers to be installed.\n"
        "\n"
        "  The 'nvidia-smi' command was not found in your PATH.\n"
        "\n"
        "  Please ensure:\n"
        "    1. NVIDIA GPU drivers are installed\n"
        "    2. Your system has a compatible NVIDIA GPU\n"
        "    3. nvidia-smi is accessible in your PATH\n"
        "\n"
        "  Driver installation:\n"
        "    - https://www.nvidia.com/Download/index.aspx\n"
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
endif()

execute_process(
    COMMAND ${CUSAXS_NVIDIA_SMI} --query-gpu=name --format=csv,noheader
    OUTPUT_VARIABLE CUSAXS_GPU_LIST
    RESULT_VARIABLE CUSAXS_GPU_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT CUSAXS_GPU_RESULT EQUAL 0 OR CUSAXS_GPU_LIST STREQUAL "")
    message(FATAL_ERROR
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  No NVIDIA GPU Detected\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "  pyCuSAXS requires at least one NVIDIA GPU.\n"
        "\n"
        "  nvidia-smi did not report any GPU devices.\n"
        "\n"
        "  Please verify:\n"
        "    1. Your system has an NVIDIA GPU installed\n"
        "    2. The GPU is properly seated and powered\n"
        "    3. The GPU drivers are correctly installed\n"
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
endif()

message(STATUS "Detected NVIDIA GPU(s): ${CUSAXS_GPU_LIST}")

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
