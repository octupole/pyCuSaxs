#[[
 * @file System/CMakeLists.txt
 * @brief Modern CMake configuration for the cudaSAXS system library.
 *
 * This library provides system-level functionality including:
 * - Cell and lattice operations
 * - Atom counting and density calculations
 * - Coordinate transformations
]]

# Sources for the system library
set(SYSTEM_SOURCES
    Cell.cpp
    AtomCounter.cpp
)

set(SYSTEM_HEADERS
    Cell.h
    AtomCounter.h
)

# Create the system library
add_library(system ${SYSTEM_SOURCES} ${SYSTEM_HEADERS})

# Create an alias for consistent naming
add_library(cudaSAXS::system ALIAS system)

# Set target properties
set_target_properties(system PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Target-specific include directories
target_include_directories(system
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CUDASAXS_PUBLIC_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link dependencies
target_link_libraries(system
    PUBLIC
        cudaSAXS::utils
    PRIVATE
        fmt::fmt
)

# Target-specific compile definitions
target_compile_definitions(system 
    PRIVATE 
        PY_SOURCE_DIR="${PROJECT_SOURCE_DIR}/pysrc"
)

# Compiler-specific options
target_compile_features(system PUBLIC cxx_std_17)

# Install targets
install(TARGETS system
    EXPORT cudaSAXSTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES ${SYSTEM_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cudaSAXS/system
    COMPONENT Development
)
