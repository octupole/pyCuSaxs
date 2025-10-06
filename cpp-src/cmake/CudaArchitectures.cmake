# CUDA Architecture Detection
# This module provides functionality to automatically detect supported CUDA architectures

function(cuda_select_nvcc_arch_flags out_variable)
    # Default CUDA architectures if detection fails
    set(default_archs "75;80;86;89")
    
    if(NOT CMAKE_CUDA_COMPILER_LOADED)
        message(WARNING "CUDA not loaded, using default architectures: ${default_archs}")
        set(${out_variable} ${default_archs} PARENT_SCOPE)
        return()
    endif()

    # Try to detect GPU architectures on the current system
    if(EXISTS "${CMAKE_CUDA_COMPILER}")
        # Create a temporary CUDA program to detect device properties
        set(detect_cuda_file "${CMAKE_CURRENT_BINARY_DIR}/detect_cuda_archs.cu")
        file(WRITE ${detect_cuda_file}
"#include <cuda_runtime.h>
#include <iostream>
int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        std::cout << \"No CUDA devices found\" << std::endl;
        return 1;
    }
    
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << prop.major << prop.minor << std::endl;
    }
    return 0;
}")

        # Try to compile and run the detection program
        try_run(run_result compile_result
            ${CMAKE_CURRENT_BINARY_DIR}
            ${detect_cuda_file}
            CMAKE_FLAGS "-DCMAKE_CUDA_ARCHITECTURES=OFF"
            RUN_OUTPUT_VARIABLE detected_archs_output
        )
        
        if(compile_result AND run_result EQUAL 0)
            string(REPLACE "\n" ";" arch_list "${detected_archs_output}")
            list(REMOVE_DUPLICATES arch_list)
            list(REMOVE_ITEM arch_list "")
            
            if(arch_list)
                message(STATUS "Detected CUDA architectures: ${arch_list}")
                set(${out_variable} ${arch_list} PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()
    
    # Fallback to default architectures
    message(STATUS "Using default CUDA architectures: ${default_archs}")
    set(${out_variable} ${default_archs} PARENT_SCOPE)
endfunction()