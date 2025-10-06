#[[
 * @file Saxs/CMakeLists.txt
 * @brief Modern CMake configuration for the CuSAXS SAXS computation library.
 *
 * This library provides the core SAXS calculation functionality including:
 * - SAXS kernel operations
 * - GPU-accelerated form factor calculations  
 * - Spline interpolation
 * - Device kernel wrappers
]]

# Sources for the SAXS library
set(SAXS_SOURCES
    saxsKernel.cu
    Splines.cu
    opsfact.cu
    saxsDeviceKernels.cu
)

set(SAXS_HEADERS
    saxsKernel.h
    Splines.h
    opsfact.h
    saxsDeviceKernels.cuh
)

# Create the SAXS library
add_library(saxs ${SAXS_SOURCES} ${SAXS_HEADERS})

# Create an alias for consistent naming
add_library(cudaSAXS::saxs ALIAS saxs)

# Set target properties
set_target_properties(saxs PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET default
    CUDA_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Target-specific include directories
target_include_directories(saxs
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CUDASAXS_PUBLIC_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CUDASAXS_SOURCE_DIR}/Exec
        ${CUDASAXS_SOURCE_DIR}/Utilities
)

# Link dependencies
target_link_libraries(saxs
    PUBLIC
        cudaSAXS::utils
        cudaSAXS::system
        CUDA::cufft
        CUDA::cublas
        CUDA::cudart
    PRIVATE
        ${FMT_TARGET}
)

# CUDA-specific compile options
target_compile_options(saxs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -diag-suppress=2417,128
    >
)

# Compiler-specific options
target_compile_features(saxs PUBLIC cxx_std_17 cuda_std_17)

# Install targets
install(TARGETS saxs
    EXPORT cudaSAXSTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES ${SAXS_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cudaSAXS/saxs
    COMPONENT Development
)
