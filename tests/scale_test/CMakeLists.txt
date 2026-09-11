cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER /home/juanchuletas/Documents/Development/sycl_workspace/llvm/build/bin/clang++)
project(ScaleTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(FUNLIB_SYCL_TARGETS spir64,nvptx64-nvidia-cuda)

set(FUNLIB_DIR ${CMAKE_SOURCE_DIR}/../../install)
add_library(funlib STATIC IMPORTED GLOBAL)
set_target_properties(funlib PROPERTIES
    IMPORTED_LOCATION ${FUNLIB_DIR}/lib/libfunlib.a
    INTERFACE_INCLUDE_DIRECTORIES ${FUNLIB_DIR}/include)

find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message(STATUS "OpenCL found")
    message(STATUS "OpenCL include dirs: ${OpenCL_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCL library not found!")
endif()

add_executable(scale_test scale_test.cpp)

target_link_libraries(scale_test PRIVATE funlib OpenCL::OpenCL GL)
target_compile_options(scale_test PRIVATE -fsycl -fsycl-targets=${FUNLIB_SYCL_TARGETS})
target_link_options(scale_test PRIVATE -fsycl -fsycl-targets=${FUNLIB_SYCL_TARGETS})
