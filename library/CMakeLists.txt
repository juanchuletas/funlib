cmake_minimum_required(VERSION 3.15)
project(funlib VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# SYCL Configuration
set(CMAKE_CXX_COMPILER /home/juanchuletas/Documents/Development/sycl_workspace/llvm/build/bin/clang++)
add_compile_options(-fsycl)
# Find OpenCL package (required for cl_gl.h)
find_package(OpenCL REQUIRED)
# Source folders
set(FUNLIB_SOURCE_DIR ${CMAKE_SOURCE_DIR}/..)

# Source files (now in src/)
set(SOURCE_FILES
    ${FUNLIB_SOURCE_DIR}/source/sycl_handler.cpp
    ${FUNLIB_SOURCE_DIR}/source/tensor.cpp
    ${FUNLIB_SOURCE_DIR}/source/tensor_operations.cpp
    ${FUNLIB_SOURCE_DIR}/source/linal.cpp
   
)

# Header files (now in include/funlib/)
set(HEADER_FILES
    ${FUNLIB_SOURCE_DIR}/include/funlib/sycl/sycl_handler.hpp
    ${FUNLIB_SOURCE_DIR}/include/funlib/Tensor/tensor.hpp
    ${FUNLIB_SOURCE_DIR}/include/funlib/Tensor/tensor_operations.hpp
    ${FUNLIB_SOURCE_DIR}/include/funlib/LinearAlgebra/linal.hpp
  
)

# Create static library
add_library(funlib STATIC ${SOURCE_FILES} ${HEADER_FILES})
# Link OpenCL
target_link_libraries(funlib PUBLIC OpenCL::OpenCL)
# CRITICAL: Set up include directories
# During development: allows #include <funlib/Tensor/tensor.hpp> to work
# After install: users can #include <funlib/Tensor/tensor.hpp>
target_include_directories(funlib PUBLIC 
    $<BUILD_INTERFACE:${FUNLIB_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set installation prefix to funlib/install
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/../install" CACHE PATH "Install path" FORCE)
endif()

# Install the static library
install(TARGETS funlib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers preserving directory structure
# This copies include/funlib/ -> install/include/funlib/
install(DIRECTORY
    ${FUNLIB_SOURCE_DIR}/include/funlib
    DESTINATION include
)

# Create and install the main funlib.hpp umbrella header
set(FUNLIB_MAIN_HEADER "${CMAKE_BINARY_DIR}/funlib.hpp")
file(WRITE ${FUNLIB_MAIN_HEADER}
"#ifndef FUNLIB_HPP
#define FUNLIB_HPP

// Main funlib header - includes all submodules
#include <funlib/Tensor/tensor.hpp>
#include <funlib/Tensor/tensor_operations.hpp>
#include <funlib/sycl/sycl_handler.hpp>
#include <funlib/LinearAlgebra/linal.hpp>
#include <funlib/ParticleSet/particle_set.hpp>
#include <funlib/ParticleSys/ParticleSystem.hpp>

#endif // FUNLIB_HPP
")

install(FILES ${FUNLIB_MAIN_HEADER} DESTINATION include/funlib)