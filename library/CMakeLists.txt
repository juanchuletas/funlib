cmake_minimum_required(VERSION 3.15)
project(funlib)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Use manually compiled SYCL Clang++
set(CMAKE_CXX_COMPILER /home/juanchuletas/Documents/Development/sycl_workspace/llvm/build/bin/clang++)

# Add SYCL-specific flags
add_compile_options(-fsycl)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../Matrix
    ${CMAKE_SOURCE_DIR}/../sycl
    ${CMAKE_SOURCE_DIR}/../include
)

# Source files
set(SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/../sycl/sycl_handler.cpp
    ${CMAKE_SOURCE_DIR}/../Set/set.cpp
    ${CMAKE_SOURCE_DIR}/../Set/set_operations.cpp
    ${CMAKE_SOURCE_DIR}/../LinearAlgebra/linal.cpp
    #${CMAKE_SOURCE_DIR}/../ParticleSys/ParticleSystem.cpp
)

# Header files (for proper IDE indexing and installation)
set(HEADER_FILES
    ${CMAKE_SOURCE_DIR}/../sycl/sycl_handler.hpp
    ${CMAKE_SOURCE_DIR}/../ParticleSys/ParticleSystem.hpp
    ${CMAKE_SOURCE_DIR}/../ParticleSet/particle_set.hpp
    ${CMAKE_SOURCE_DIR}/../Set/set.hpp
    ${CMAKE_SOURCE_DIR}/../Set/set_operations.hpp
    ${CMAKE_SOURCE_DIR}/../LinearAlgebra/linal.hpp
)

# Create static library
add_library(funlib STATIC ${SOURCE_FILES} ${HEADER_FILES})

# Install library and headers
install(TARGETS funlib DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION include)