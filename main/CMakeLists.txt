cmake_minimum_required(VERSION 3.10)
project(FunlibProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER /home/juanchuletas/Documents/Development/sycl_workspace/llvm/build/bin/clang++)
include_directories(${CMAKE_SOURCE_DIR}/../include)

# Add the static library
add_library(funlib STATIC IMPORTED)
set_target_properties(funlib PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../library/build/libfunlib.a
)

# Add the executable
add_executable(main main.cpp)

# Link the static library
target_link_libraries(main PRIVATE funlib)
target_compile_options(main PRIVATE -fsycl)
target_link_options(main PRIVATE -fsycl)