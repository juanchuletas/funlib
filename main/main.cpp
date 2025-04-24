#include "../include/funlib.hpp"
#include <iostream>


int main() {
    
    
    flib::sycl_handler::sys_info(); // Prints system info
    flib::sycl_handler::select_device("Intel"); // Selects a vendor for your computations
    flib::sycl_handler::get_device_info(); // Prints current device info

    int N = 5;
    std::vector<float> A = {
        4, 1, 0, 0, 0,
        1, 4, 1, 0, 0,
        0, 1, 4, 1, 0,
        0, 0, 1, 4, 1,
        0, 0, 0, 1, 3
    };

    std::vector<float> b = {1, 2, 0, 1, 3};
    flib::fset  Amat(N, N, A.data());
    flib::fset  bvec(N, b.data());
    flib::fset  xvec(N);

    std::cout << "\n Conjugate Gradient" << "\n";
    flib::linal::conjugate_grad(Amat, bvec, xvec, 100);
  
    std::cout << "\n Solution: " << "\n";
    xvec.print();


    return 0;
}