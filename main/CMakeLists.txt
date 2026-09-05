cmake_minimum_required(VERSION 3.10)
project(FunlibProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER /home/juanchuletas/Documents/Development/sycl_workspace/llvm/build/bin/clang++)
set(FUNLIB_SYCL_TARGETS spir64,nvptx64-nvidia-cuda)
include_directories(${CMAKE_SOURCE_DIR}/../include)

# Add the static library

# Add funlib
 set(FUNLIB_DIR ${CMAKE_SOURCE_DIR}/../install)
    add_library(funlib STATIC IMPORTED GLOBAL)
    set_target_properties(funlib PROPERTIES
        IMPORTED_LOCATION ${FUNLIB_DIR}/lib/libfunlib.a
        INTERFACE_INCLUDE_DIRECTORIES ${FUNLIB_DIR}/include)
find_package(OpenCL REQUIRED)
    if (OpenCL_FOUND)
        message(STATUS "OpenCL found")
        message(STATUS "OpenCL include dirs: ${OpenCL_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "OpenCL library not found!")
    endif()
# Add the executable
add_executable(main main.cpp)

# Link the static library
target_link_libraries(main PRIVATE funlib OpenCL::OpenCL GL)
target_compile_options(main PRIVATE -fsycl -fsycl-targets=${FUNLIB_SYCL_TARGETS})
target_link_options(main PRIVATE -fsycl -fsycl-targets=${FUNLIB_SYCL_TARGETS})
