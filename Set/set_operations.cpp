#include "set_operations.hpp"
#include "../sycl/sycl_handler.hpp"
namespace flib
{
    template <typename T>
    Set<T> set_operations::gemm(const Set<T> &A, const Set<T> &B)
    {
         // Assuming A is m x n
         size_t colsA = static_cast<size_t>(A.getCols());
         size_t rowsA = static_cast<size_t>(A.getRows());
         size_t colsB = static_cast<size_t>(B.getCols());
         size_t rowsB = static_cast<size_t>(B.getRows());
         size_t colsC = colsB; 
         size_t rowsC = rowsA;
         if (colsA != rowsB)
         {
             //for matrix multiplication, the number of columns in A must be equal to the number of rows in B
             //because tjhe result matrix will have the same number of rows as A and the same number of columns as B
             std::cout<<"Cols of A : "<<colsA<<std::endl;
             std::cout<<"Cols of B : "<<rowsB<<std::endl;
             throw std::invalid_argument("Set dimensions do not match for multiplication");
         }
 
         Set<T> C(rowsC, colsC); //output matrix
         sycl::queue Q = flib::sycl_handler::get_queue();
            
         { //Sycl scope
             sycl::buffer<T, 1> buffc  = C.to_sycl_buffer();
             sycl::buffer<T, 1> buffa  = A.to_sycl_buffer();
             sycl::buffer<T, 1> buffb  = B.to_sycl_buffer();
             Q.submit([&](sycl::handler &cgh){
                 
                 auto acc_matC = buffc.template get_access<sycl::access::mode::write>(cgh);
                 auto acc_matA = buffa.template get_access<sycl::access::mode::read>(cgh);
                 auto acc_matB = buffb.template get_access<sycl::access::mode::read>(cgh);
                 cgh.parallel_for(sycl::range<2>(sycl::range<2> {static_cast<size_t>(rowsC),static_cast<size_t>(colsC)}),[=](sycl::item<2> item){
                     const int i = item.get_id(0); // is like: for (int i = 0; i < rowsA; i++)
                     const int j = item.get_id(1); // is like: for (int j = 0; j < colsB; j++)
                     T sum = 0.0; 
                     for (int k = 0; k < colsA; k++)
                     {   
                         sum += acc_matA[i*colsA + k]* acc_matB[j + colsB * k];
                     }
                     acc_matC[i*colsB+j] = sum;
                 });
             });
         }
        
        return  C;
        
    }

    template <typename T>
    Set<T> set_operations::matXvec(const Set<T> &A, const Set<T> &B)
    {
            // Assuming A is m x n
            size_t colsA = static_cast<size_t>(A.getCols());
            size_t rowsA = static_cast<size_t>(A.getRows());
            size_t colsB = static_cast<size_t>(B.getCols());
            size_t rowsB = static_cast<size_t>(B.getRows());
            if (colsA != rowsB)
            {
                //for matrix multiplication, the number of columns in A must be equal to the number of rows in B
                //because tjhe result matrix will have the same number of rows as A and the same number of columns as B
                throw std::invalid_argument("Set dimensions do not match for multiplication");
            }
    
            Set<T> C(rowsA, 1); //output matrix
            sycl::queue Q = flib::sycl_handler::get_queue();
                
            { //Sycl scope
                sycl::buffer<T, 1> buffc  = C.to_sycl_buffer();
                sycl::buffer<T, 1> buffa  = A.to_sycl_buffer();
                sycl::buffer<T, 1> buffb  = B.to_sycl_buffer();
                Q.submit([&](sycl::handler &cgh){
                    
                    auto acc_matC = buffc.template get_access<sycl::access::mode::write>(cgh);
                    auto acc_matA = buffa.template get_access<sycl::access::mode::read>(cgh);
                    auto acc_matB = buffb.template get_access<sycl::access::mode::read>(cgh);
                    cgh.parallel_for(sycl::range<2>(sycl::range<2> {static_cast<size_t>(rowsA),static_cast<size_t>(1)}),[=](sycl::item<2> item){
                        const int i = item.get_id(0); // is like: for (int i = 0; i < rowsA; i++)
                        T sum = 0.0; 
                        for (int k = 0; k < colsA; k++)
                        {   
                            sum += acc_matA[i*colsA + k]* acc_matB[k];
                        }
                        acc_matC[i] = sum;
                    });
                });
            }
            
            return  C;
    }

    //class sycl_handler;
    // Implementations of the member functions can be done in a separate .cpp file or inline as shown above.
    // The print function can be implemented to display the vector elements.
    template<typename T>
    Set<T> set_operations::prod(const Set<T>& A, const Set<T>& B)
    {   
    
        if(A.getCols() == 1 || B.getCols() == 1){
            //If one is a vector and the other is a matrix, do a matrix-vector multiplication
            return matXvec(A, B);
        }
        else{
            //Otherwise, do a matrix-matrix multiplication
            return gemm(A, B);
        }
    }


    template<typename T>
    T set_operations::dot(const Set<T>& A, const Set<T>& B){
        size_t colsA = static_cast<size_t>(A.getCols());
        size_t rowsA = static_cast<size_t>(A.getRows());
        size_t colsB = static_cast<size_t>(B.getCols());
        size_t rowsB = static_cast<size_t>(B.getRows());
        if(colsA != 1 || colsB != 1){
            throw std::invalid_argument("Dot product is only defined for vectors");

        }
        if(rowsA != rowsB){
            throw std::invalid_argument("Vectors must be of the same size for dot product");
        }
        T result = 0;
        for (std::size_t i = 0; i < rowsA; ++i)
        {
            result += A[i] * B[i];
        }
        return result;
    }
    //Explicit instantiations (VERY IMPORTANT)
    template Set<double>    set_operations::prod(const Set<double>&, const Set<double>&);
    template Set<float>     set_operations::prod(const Set<float>&, const Set<float>&);
    template Set<int>       set_operations::prod(const Set<int>&, const Set<int>&);

    template double         set_operations::dot(const Set<double>&, const Set<double>&);
    template float          set_operations::dot(const Set<float>&, const Set<float>&);
    template int            set_operations::dot(const Set<int>&, const Set<int>&);

    template Set<double>    set_operations::gemm(const Set<double>&, const Set<double>&);
    template Set<float>     set_operations::gemm(const Set<float>&, const Set<float>&);
    template Set<int>       set_operations::gemm(const Set<int>&, const Set<int>&);
    template Set<double>    set_operations::matXvec(const Set<double>&, const Set<double>&);
    template Set<float>     set_operations::matXvec(const Set<float>&, const Set<float>&);
    template Set<int>       set_operations::matXvec(const Set<int>&, const Set<int>&);
}