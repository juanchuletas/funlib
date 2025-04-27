📚 funlib

funlib is a lightweight, modular C++ linear algebra library focused on efficient matrix and vector operations. It is GPU-accelerated using SYCL, making it ideal for numerical simulations and scientific computing tasks such as Finite Element Methods (FEM).

✨ Features

    Matrix and Vector operations (row-major storage)

    Matrix-Matrix and Matrix-Vector multiplication

    SYCL-accelerated kernels

    Custom Conjugate Gradient Solver for large sparse systems

    Eigenvalue and Eigenvector computation (under development)

    Simple CPU fallback when no GPU available

    Clean and extensible design

    Designed for FEM and scientific simulations

🛠 Technologies

    C++17

    SYCL (Intel oneAPI DPC++, or LLVM SYCL builds)

    CMake (build system)

🏗️ Build Instructions

Make sure you have:

    A SYCL-capable compiler (e.g., Intel oneAPI DPC++, or LLVM SYCL build)

    CMake version 3.15 or higher

# Clone funlib
git clone https://github.com/yourusername/funlib.git
cd funlib

# Create a build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
make

Notes:

    If you are using a manually built SYCL Clang, ensure your CMakeLists.txt points to the correct clang++ binary.

    You may need to set the environment for oneAPI compilers (source /path/to/oneapi/setvars.sh).

🚀 Quick Start Example

#include "funlib.hpp"

int main()
{
    flib::Set<double> A(3, 3);
    flib::Set<double> x(3, 1);

    // Fill matrix A and vector x
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 2;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 0;
    A(2,0) = 2; A(2,1) = 0; A(2,2) = 1;
    
    x(0,0) = 1;
    x(1,0) = 2;
    x(2,0) = 3;

    auto b = prod(A, x); // Matrix-vector multiplication

    b.print(); // Display result
}

📚 Library Structure

funlib/
├── include/
│   └── funlib.hpp        # Main include file
├── sycl/
│   ├── sycl_handler.hpp  # SYCL device and queue management
│   └── sycl_handler.cpp
├── Matrix/
│   ├── matrix.hpp        # Matrix Set class
│   ├── matrix_impl.hpp   # Matrix Set implementation
│   ├── matrix_op.hpp     # Operations declarations
│   └── matrix_op_impl.hpp# Operations implementation
├── CMakeLists.txt
└── README.md

📈 Roadmap

Matrix and vector classes

Matrix-matrix and matrix-vector product

SYCL GPU acceleration

Conjugate Gradient solver

Eigenvalue computation

Sparse matrix support (CSR format)

    Preconditioners for iterative solvers

📜 License

funlib is open-source under the MIT License.
