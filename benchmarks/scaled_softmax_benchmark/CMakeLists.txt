cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER /home/juanchuletas/Documents/Development/sycl_workspace/llvm/build/bin/clang++)
project(ScaledSoftmaxBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(FUNLIB_SYCL_TARGETS spir64,nvptx64-nvidia-cuda)

set(FUNLIB_DIR ${CMAKE_SOURCE_DIR}/../../install)
add_library(funlib STATIC IMPORTED GLOBAL)
set_target_properties(funlib PROPERTIES
    IMPORTED_LOCATION ${FUNLIB_DIR}/lib/libfunlib.a
    INTERFACE_INCLUDE_DIRECTORIES ${FUNLIB_DIR}/include)

find_package(OpenCL REQUIRED)

add_executable(scaled_softmax_benchmark scaled_softmax_benchmark.cpp)
target_link_libraries(scaled_softmax_benchmark PRIVATE funlib OpenCL::OpenCL GL)
target_compile_options(scaled_softmax_benchmark PRIVATE
                       -fsycl -fsycl-targets=${FUNLIB_SYCL_TARGETS})
target_link_options(scaled_softmax_benchmark PRIVATE
                    -fsycl -fsycl-targets=${FUNLIB_SYCL_TARGETS})
