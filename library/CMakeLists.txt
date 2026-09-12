cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_COMPILER /home/juanchuletas/Documents/Development/sycl_workspace/llvm/build/bin/clang++)
project(funlib VERSION 1.0.0)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# SYCL Configuration
message("DPCPP compiler path: ${CMAKE_CXX_COMPILER}")
# Add funlib
set(FUNLIB_SYCL_TARGETS spir64,nvptx64-nvidia-cuda)
# Find OpenCL package (required for cl_gl.h)
find_package(OpenCL REQUIRED)
# Source folders
get_filename_component(FUNLIB_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Source files (now in src/)
set(SOURCE_FILES
    ${FUNLIB_SOURCE_DIR}/source/sycl_handler.cpp
    ${FUNLIB_SOURCE_DIR}/source/tensor.cpp
    ${FUNLIB_SOURCE_DIR}/source/tensor_operations.cpp
    ${FUNLIB_SOURCE_DIR}/source/batched_gemm.cpp
    ${FUNLIB_SOURCE_DIR}/source/elementwise_operations.cpp
    ${FUNLIB_SOURCE_DIR}/source/softmax.cpp
    ${FUNLIB_SOURCE_DIR}/source/linal.cpp
   
)

# Header files (now in include/funlib/)
set(HEADER_FILES
    ${FUNLIB_SOURCE_DIR}/include/funlib/sycl/sycl_handler.hpp
    ${FUNLIB_SOURCE_DIR}/include/funlib/Tensor/tensor.hpp
    ${FUNLIB_SOURCE_DIR}/include/funlib/Tensor/tensor_operations.hpp
    ${FUNLIB_SOURCE_DIR}/include/funlib/LinearAlgebra/linal.hpp
    ${FUNLIB_SOURCE_DIR}/include/funlib/operations/elementwise_operations/elementwise_operations.hpp
    ${FUNLIB_SOURCE_DIR}/include/funlib/operations/softmax/softmax.hpp
  
)

# Create static library
add_library(funlib STATIC ${SOURCE_FILES} ${HEADER_FILES})
add_library(funlib::funlib ALIAS funlib)
target_compile_features(funlib PUBLIC cxx_std_17)
target_compile_options(funlib PUBLIC -fsycl -fsycl-targets=${FUNLIB_SYCL_TARGETS})
target_link_options(funlib PUBLIC -fsycl -fsycl-targets=${FUNLIB_SYCL_TARGETS})
# Link OpenCL
target_link_libraries(funlib PUBLIC OpenCL::OpenCL GL)
# CRITICAL: Set up include directories
# During development: allows #include <funlib/Tensor/tensor.hpp> to work
# After install: users can #include <funlib/Tensor/tensor.hpp>
target_include_directories(funlib PUBLIC 
    $<BUILD_INTERFACE:${FUNLIB_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set installation prefix to funlib/install
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${FUNLIB_SOURCE_DIR}/install" CACHE PATH "Install path" FORCE)
endif()

# Install the static library
install(TARGETS funlib
    EXPORT funlibTargets
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

set(FUNLIB_CMAKE_INSTALL_DIR lib/cmake/funlib)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/funlibConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/funlibConfig.cmake"
    INSTALL_DESTINATION ${FUNLIB_CMAKE_INSTALL_DIR}
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/funlibConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(EXPORT funlibTargets
    NAMESPACE funlib::
    DESTINATION ${FUNLIB_CMAKE_INSTALL_DIR}
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/funlibConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/funlibConfigVersion.cmake"
    DESTINATION ${FUNLIB_CMAKE_INSTALL_DIR}
)
