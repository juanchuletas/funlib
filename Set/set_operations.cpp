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
    template <typename T>
    T set_operations::reduction(const Set<T> &A)
    {
        // Assuming A is a 1D vector or 1D list.
        if(A.getCols() != 1 && A.getRows() != 1){
            // If the input is not a 1D vector or list, throw an error.
            std::cout<<"Cols: "<<A.getCols()<<std::endl;
            std::cout<<"Rows: "<<A.getRows()<<std::endl;
            throw std::invalid_argument("Reduction is only defined for 1D vectors, arrays or lists");
        }
        std::size_t N = A.getRows() * A.getCols();
        sycl::queue Q = flib::sycl_handler::get_queue();

        /*  Creates the size of the workgroups */
        std::size_t work_group_size = 64; //desired work group size
        std::size_t n_work_groups;
        std::size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;
        n_work_groups = global_size/work_group_size; //number of work groups
        std::cout<<"Number of work groups: "<<n_work_groups<<std::endl;
       
        Set<T> partial(n_work_groups, 1); //partial sum of the work groups
        { //sycl scope: first reduction to get the partial sums
            sycl::buffer<T,1> buffA = A.to_sycl_buffer();
            sycl::buffer<T,1> partial_sum = partial.to_sycl_buffer();
        
            Q.submit([&](sycl::handler &cgh){

                auto acc_global      = buffA.template get_access<sycl::access::mode::read>(cgh);
                auto acc_partial_sum = partial_sum.template get_access<sycl::access::mode::write>(cgh);

                //local memory
                sycl::local_accessor<T, 1> local_sum(sycl::range<1>(work_group_size), cgh);

                cgh.parallel_for(sycl::nd_range<1>{global_size,work_group_size},[=](sycl::nd_item<1> ndItem){

                    std::size_t l_id    = ndItem.get_local_id(0); //Id within a workgroup
                    std::size_t wg_id   = ndItem.get_group(0); //Id of the workgroup
                    std::size_t wg_size = ndItem.get_local_range(0); //Size of the workgroup
                    std::size_t index   = wg_id*(wg_size*2) + l_id;

                    acc_partial_sum[wg_id] = 0;
                    //load the data into local memory
                    if(index<N){

                        local_sum[l_id] = acc_global[index] + acc_global[index + wg_size];
                    }
                    else{
                        local_sum[l_id] = 0;
                    }
                    ndItem.barrier(sycl::access::fence_space::local_space);


                    for(std::size_t i = wg_size/2; i>0; i>>=1){
                        if(l_id<i){
                            local_sum[l_id] = local_sum[l_id] + local_sum[l_id + i];
                        }
                        ndItem.barrier(sycl::access::fence_space::local_space);

                    }

                    if(l_id == 0){
                        acc_partial_sum[wg_id] = local_sum[0];
                    }

                });
                
            }).wait();

        }// end of the first reduction
        
        // Now we have the partial sums in the partial array
        // TODO: We need to do a second reduction to get the final result
        T result = 0;
       
        for(std::size_t i = 0; i <= n_work_groups/2; ++i){
            //std::cout<<"Partial sum : "<<partial[i]<<" at index : "<<i<<std::endl;
            result += partial[i];
        }


        
        return result;






    }
    // Explicit instantiations (VERY IMPORTANT)
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

    template double         set_operations::reduction(const Set<double>&);
    template float          set_operations::reduction(const Set<float>&);
    template int            set_operations::reduction(const Set<int>&);
}