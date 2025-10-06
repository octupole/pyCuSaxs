# FindDependencies.cmake
# Helper module for finding and configuring cudaSAXS dependencies

# Function to find and configure CUDA with proper error messages
function(find_cuda_dependencies)
    find_package(CUDAToolkit 11.0 REQUIRED)
    
    if(NOT TARGET CUDA::cudart)
        message(FATAL_ERROR "CUDA runtime not found. Please ensure CUDA Toolkit 11.0 or later is installed.")
    endif()
    
    if(NOT TARGET CUDA::cufft)
        message(FATAL_ERROR "CUFFT not found. Please ensure CUDA Toolkit with CUFFT is installed.")
    endif()
    
    if(NOT TARGET CUDA::cublas)
        message(FATAL_ERROR "CUBLAS not found. Please ensure CUDA Toolkit with CUBLAS is installed.")
    endif()
    
    # Check if Thrust is available (should be part of CUDA Toolkit 11+)
    if(NOT TARGET CUDA::thrust)
        find_path(THRUST_INCLUDE_DIR thrust/version.h
            HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            PATH_SUFFIXES include
        )
        
        if(THRUST_INCLUDE_DIR)
            add_library(CUDA::thrust INTERFACE IMPORTED)
            target_include_directories(CUDA::thrust INTERFACE ${THRUST_INCLUDE_DIR})
            message(STATUS "Found Thrust: ${THRUST_INCLUDE_DIR}")
        else()
            message(WARNING "Thrust not found. Some functionality may be limited.")
        endif()
    endif()
    
    message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Runtime: ${CUDA_cudart_LIBRARY}")
    message(STATUS "CUFFT: ${CUDA_cufft_LIBRARY}")
    message(STATUS "CUBLAS: ${CUDA_cublas_LIBRARY}")
endfunction()

# Function to find and configure Python dependencies
function(find_python_dependencies)
    find_package(Python3 3.7 REQUIRED COMPONENTS Interpreter Development)
    
    if(Python3_VERSION VERSION_LESS "3.7")
        message(FATAL_ERROR "Python 3.7 or later is required. Found version ${Python3_VERSION}")
    endif()
    
    # Find pybind11
    find_package(pybind11 2.6)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found in system, trying to find manually...")
        
        # Try common conda environments
        set(CONDA_PREFIXES 
            "$ENV{CONDA_PREFIX}"
            "/opt/anaconda3"
            "/opt/miniconda3" 
            "/opt/miniforge3"
            "/usr/local/anaconda3"
            "/usr/local/miniconda3"
        )
        
        foreach(prefix ${CONDA_PREFIXES})
            if(EXISTS "${prefix}/share/cmake/pybind11")
                set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${prefix}/share/cmake/pybind11")
                find_package(pybind11 2.6)
                if(pybind11_FOUND)
                    break()
                endif()
            endif()
        endforeach()
        
        if(NOT pybind11_FOUND)
            message(FATAL_ERROR "pybind11 2.6 or later is required but not found. Please install it via pip or conda.")
        endif()
    endif()
    
    message(STATUS "Python version: ${Python3_VERSION}")
    message(STATUS "Python executable: ${Python3_EXECUTABLE}")
    message(STATUS "pybind11 version: ${pybind11_VERSION}")
endfunction()

# Function to find OpenMP with fallback options
function(find_openmp_dependency)
    find_package(OpenMP)
    
    if(NOT OpenMP_FOUND)
        message(WARNING "OpenMP not found. Performance may be reduced.")
        
        # Create a dummy OpenMP target for compatibility
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        return()
    endif()
    
    if(NOT TARGET OpenMP::OpenMP_CXX)
        message(FATAL_ERROR "OpenMP C++ support not found")
    endif()
    
    message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
endfunction()

# Function to setup fmt library
function(find_fmt_dependency)
    # Check if fmt targets already exist
    if(TARGET fmt::fmt)
        message(STATUS "fmt library already configured")
        return()
    endif()
    
    # The USE_SYSTEM_FMT option is handled in main CMakeLists.txt
    if(USE_SYSTEM_FMT)
        find_package(fmt 9.0 REQUIRED)
        if(fmt_FOUND)
            message(STATUS "Using system fmt library version: ${fmt_VERSION}")
        else()
            message(FATAL_ERROR "System fmt library requested but not found. Please install fmt >= 9.0 or set USE_SYSTEM_FMT=OFF")
        endif()
    else()
        # Bundled fmt will be available after add_subdirectory in main CMakeLists.txt
        message(STATUS "Using bundled fmt library")
    endif()
endfunction()

# Main function to find all dependencies
function(configure_all_dependencies)
    find_cuda_dependencies()
    find_python_dependencies()
    find_openmp_dependency()
    find_fmt_dependency()
    
    message(STATUS "All dependencies configured successfully")
endfunction()