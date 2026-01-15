#include <funlib/funlib.hpp>
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
    flib::ftensor  Amat(N, N, A.data()); //ftsensor is a float tensor
    flib::ftensor  bvec(N, b.data());
    flib::ftensor  xvec(N);

    std::cout << "\n Conjugate Gradient" << "\n";
    flib::linal::conjugate_grad(Amat, bvec, xvec, 100);
  
    std::cout << "\n Solution: " << "\n";
    xvec.print();


    std::cout << "\n Reduction  Algorithm" << "\n";
    std::cout << "\n 1M vector" << "\n";
    std::size_t Nitems = 100000;
    flib::ftensor  redvec(Nitems);
    redvec.fill(1.0f); // Fill the vector with 1.0f

    float result = flib::tensor_operations::reduction(redvec); //Reduction of a first order tensor (vector)
    std::cout << "Result: " << result << std::endl;

    flib::sycl_handler::get_platform_info();


    return 0;
}