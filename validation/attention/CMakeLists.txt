cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_COMPILER /home/juanchuletas/Documents/Development/sycl_workspace/llvm/build/bin/clang++)
project(AttentionValidation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(FUNLIB_SYCL_TARGETS spir64,nvptx64-nvidia-cuda)

set(FUNLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../install")
add_library(funlib STATIC IMPORTED GLOBAL)
set_target_properties(funlib PROPERTIES
    IMPORTED_LOCATION "${FUNLIB_DIR}/lib/libfunlib.a"
    INTERFACE_INCLUDE_DIRECTORIES "${FUNLIB_DIR}/include")

find_package(OpenCL REQUIRED)

add_executable(validate_attention validate_attention.cpp)
target_link_libraries(validate_attention PRIVATE funlib OpenCL::OpenCL GL)
target_compile_options(validate_attention PRIVATE -fsycl -fsycl-targets=${FUNLIB_SYCL_TARGETS})
target_link_options(validate_attention PRIVATE -fsycl -fsycl-targets=${FUNLIB_SYCL_TARGETS})
