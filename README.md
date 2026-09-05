# Funlib

Funlib is a C++ library for tensor operations on CPUs and GPUs through SYCL.

The project aims to provide one API for Intel and NVIDIA hardware. Users can select a device and a SYCL backend, create tensors, move data to the selected device, and run tensor operations.

Funlib is in active development. The current focus is fast matrix multiplication for scientific programs and AI models.

## Project goals

Funlib aims to make GPU computing easier for C++ users.

The main goals are:

1. Provide simple tensor classes for CPU and GPU memory.
2. Manage memory inside the library.
3. Let users select Intel or NVIDIA devices through SYCL.
4. Keep tensor data on the GPU between operations.
5. Provide fast matrix and vector operations.
6. Support scientific computing and AI workloads.
7. Use the same public API across supported GPU backends.
8. Select a suitable kernel for each device and matrix size.

## Current features

### Tensor storage

Funlib provides the `Tensor<T>` class with row major storage.

It supports:

1. CPU tensors.
2. Device tensors.
3. Automatic memory release through RAII.
4. Copy and move operations.
5. Data upload with `copy_from`.
6. Data download with `to_host`.
7. `float`, `double`, and `int` tensor types.

The following short names are available:

```cpp
flib::tensor
flib::ftensor
flib::itensor
```

### SYCL device selection

The `sycl_handler` class can register named queues for different devices and backends.

The current device choices include:

1. Intel GPU through OpenCL.
2. Intel GPU through Level Zero.
3. NVIDIA GPU through CUDA.
4. CPU devices supported by the active SYCL installation.

Queue profiling can be enabled when a queue is registered. Funlib can also print device, platform, driver, and backend information.

### Tensor operations

Funlib currently provides:

1. Matrix multiplication.
2. Matrix and vector multiplication.
3. Dot product.
4. Reduction.
5. Tiled GEMM with local memory.
6. GEMM with 2x2 register blocking.
7. GEMM that combines local memory tiling and 2x2 register blocking.

The optimized GEMM methods are being tested on Intel and NVIDIA GPUs. Different devices can prefer different tile and register block settings.

### Linear algebra

Funlib includes a conjugate gradient solver for linear systems.

This part of the library is still under development and needs more tests and device tensor support.

### OpenGL support

The SYCL handler contains support for creating an OpenCL and OpenGL interop context. This is useful for programs that want to share GPU data with graphics code.

## AI goal

One important goal is to support the main operations used by transformer models.

A transformer block uses matrix multiplication in several places:

```text
Input tensors
Layer normalization
Q, K, and V projections
Attention score calculation
Attention output calculation
Output projection
Feed forward layers
Output tensors
```

Funlib does not provide a complete transformer model yet. The current GEMM work is the base needed for attention and feed forward operations.

The planned AI work includes:

1. Fast GEMM selection for Intel and NVIDIA GPUs.
2. Batched matrix multiplication.
3. Tensor shapes with more than two dimensions.
4. Tensor transpose and data packing.
5. Layer normalization.
6. Softmax.
7. GELU and other activation functions.
8. Attention operations.
9. Mixed precision tensor operations.
10. Kernel fusion to reduce memory traffic.
11. Automatic differentiation for training.

## GEMM development

Funlib contains several GEMM methods because one method does not give the best result on every GPU and matrix size.

The current methods are:

1. `gemm`

   This is the basic matrix multiplication method.

2. `gemmTiled`

   This method loads parts of A and B into local memory. Work items in the same work group reuse this data.

3. `gemm_blocked2x2`

   Each work item calculates a 2x2 part of the output and keeps four partial results in registers.

4. `gemm_tiled_blocked2x2`

   This method combines local memory tiles with 2x2 register blocking.

The benchmark results show that Intel and NVIDIA GPUs do not always prefer the same method. Future work will add automatic selection based on the device and matrix dimensions.


## Basic device tensor example

```cpp
#include <funlib/funlib.hpp>

#include <iostream>
#include <vector>

int main()
{
    flib::sycl_handler::register_queue(
        "cuda",
        flib::device::GPU,
        flib::vendor::NVIDIA,
        flib::backend::CUDA,
        true);

    sycl::queue queue = flib::sycl_handler::get_queue("cuda");

    const std::size_t M = 512;
    const std::size_t K = 512;
    const std::size_t N = 512;

    std::vector<float> hostA(M * K, 1.0f);
    std::vector<float> hostB(K * N, 2.0f);

    flib::ftensor A(M, K, queue);
    flib::ftensor B(K, N, queue);

    A.copy_from(hostA.data(), queue).wait();
    B.copy_from(hostB.data(), queue).wait();
    //Uses gpu acceleration
    flib::ftensor C =
        flib::tensor_operations::gemm_tiled_blocked2x2(A,B,queue); 

    std::vector<float> hostC = C.to_host(queue);

    std::cout<<"C(0, 0): "<<hostC[0]<<std::endl;

    return 0;
}
```

The Tensor objects release their memory when they leave the current scope. Users do not need to call a GPU memory release function.

## Requirements

The current project requires:

1. A compiler with SYCL support.
2. C++17 support.
3. CMake 3.15 or newer.
4. OpenCL development files.
5. OpenGL development files.
6. A working SYCL backend for the selected device.

The current CMake files use a local path to an LLVM SYCL compiler. Change `CMAKE_CXX_COMPILER` in the CMake files if your compiler is installed in another location.

The library is currently compiled for the generic SPIR target and the NVIDIA CUDA target.

## Build

Build and install the library from the `library` directory:

```text
cd library
mkdir build
cd build
cmake ..
make
make install
```

The default installation location is the `install` directory in the project root.

## Tests

The `tests` directory contains tensor operation tests and its own CMake file.

The tests compare GPU results with CPU results and should be expanded as new tensor operations are added.

## Benchmarks

The `benchmarks` directory contains GEMM benchmarks.

The benchmarks measure:

1. Total operation time.
2. Kernel time from SYCL event profiling.
3. GEMM performance in GFLOP per second.
4. Performance differences between GEMM methods.
5. Performance differences between Intel and NVIDIA GPUs.

Performance results depend on the GPU, backend, driver, matrix shape, tile size, and register use.

## Current limits

Funlib is not a complete AI library yet.

Current limits include:

1. Tensor shapes mainly describe matrices and vectors.
2. GEMM kernel selection is manual.
3. The optimized GEMM methods need more correctness tests.
4. Mixed precision operations are not available.
5. Automatic differentiation is not available.
6. Neural network layers are not available.
7. Some linear algebra operations still use CPU tensors.
8. The build configuration contains local compiler paths.

## Roadmap

The next main steps are:

1. Complete correctness tests for all GEMM methods.
2. Compare all GEMM methods under the same benchmark conditions.
3. Test more tile sizes and register block sizes.
4. Add device and matrix based GEMM selection.
5. Add transpose and packed tensor layouts.
6. Add batched GEMM.
7. Add the operations needed for transformer inference.
8. Add training support after inference operations are stable.

## License

Funlib is available under the APACHE License.
