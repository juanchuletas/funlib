
#include <funlib/Tensor/tensor_operations.hpp>
#include <funlib/sycl/sycl_handler.hpp>
#include <array>

namespace flib
{
    template <typename T>
    std::vector<std::size_t> gemm_output_shape(const Tensor<T>& A, const Tensor<T>& B)
    {
        if(A.getRank() < 2){
            throw std::invalid_argument("The first GEMM tensor must have at least two dimensions");
        }
        if(B.getRank() != 2){
            throw std::invalid_argument("The second GEMM tensor must have two dimensions");
        }

        std::vector<std::size_t> shape = A.getShape();
        shape.back() = B.getCols();
        return shape;
    }

    template <typename T>
    Tensor<T> tensor_operations::gemm_naive(const Tensor<T> &A, const Tensor<T> &B, sycl::queue Q,
                                            sycl::event* kernel_event)
    {
        std::vector<std::size_t> output_shape = gemm_output_shape(A, B);
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
            throw std::invalid_argument("Tensor dimensions do not match for multiplication");
        }

        if(A.is_device() || B.is_device())
        {
            if(!A.is_device() || !B.is_device()){
                throw std::invalid_argument("GEMM inputs must use the same memory location");
            }
            if(!A.is_accessible_from(Q) || !B.is_accessible_from(Q))
            {
                throw std::invalid_argument("GEMM queue cannot access the input tensors");
            }

            const T* dataA = A.device_data();
            const T* dataB = B.device_data();
            Tensor<T> C(output_shape, Q);
            T* dataC = C.device_data();

            sycl::event event = Q.submit([&](sycl::handler &cgh){
                cgh.parallel_for(
                    sycl::range<2>{rowsC, colsC},
                    [=](sycl::item<2> item){
                        size_t i = item.get_id(0);
                        size_t j = item.get_id(1);
                        T sum = T(0);
                        for(size_t k = 0; k < colsA; k++)
                        {
                            sum += dataA[i * colsA + k] * dataB[k * colsB + j];
                        }
                        dataC[i * colsC + j] = sum;
                    });
            });

            if(kernel_event != nullptr){
                *kernel_event = event;
            }
            event.wait();
            return C;
        }

        Tensor<T> C(output_shape); //output tensor
        { //Sycl scope
            sycl::buffer<T, 1> buffc  = C.to_sycl_buffer();
            sycl::buffer<T, 1> buffa  = A.to_sycl_buffer();
            sycl::buffer<T, 1> buffb  = B.to_sycl_buffer();
            sycl::event event = Q.submit([&](sycl::handler &cgh){

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
            if(kernel_event != nullptr){
                *kernel_event = event;
            }
        }

        return  C;
        
    }

    template <typename T>
    Tensor<T> tensor_operations::gemmTiled(const Tensor<T> &A, const Tensor<T> &B, sycl::queue Q,
                                           sycl::event* kernel_event)
    {
        std::vector<std::size_t> output_shape = gemm_output_shape(A, B);
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
            //because the result matrix will have the same number of rows as A and the same number of columns as B
            std::cout<<"Cols of A : "<<colsA<<std::endl;
            std::cout<<"Rows of B : "<<rowsB<<std::endl;
            throw std::invalid_argument("Tensor dimensions do not match for multiplication");
        }
        constexpr size_t tile_size = 16;
        size_t global_rows = ((rowsC + tile_size - 1) / tile_size) * tile_size;
        size_t global_cols = ((colsC + tile_size - 1) / tile_size) * tile_size;
        if(A.is_device() || B.is_device())
        {
            if(!A.is_device() || !B.is_device()){
                throw std::invalid_argument("GEMM inputs must use the same memory location");
            }
            if(!A.is_accessible_from(Q) || !B.is_accessible_from(Q))
            {
                throw std::invalid_argument("GEMM queue cannot access the input tensors");
            }
            const T* dataA = A.device_data();
            const T* dataB = B.device_data();
            Tensor<T> C(output_shape, Q); //device output tensor
            T* dataC = C.device_data();
            //
            sycl::range<2> global_range{global_rows, global_cols};
            sycl::range<2> local_range{tile_size, tile_size};
            sycl::event event = Q.submit([&](sycl::handler &cgh){
                sycl::local_accessor<T, 1> tileA(sycl::range<1>{tile_size * tile_size}, cgh);
                sycl::local_accessor<T, 1> tileB(sycl::range<1>{tile_size * tile_size}, cgh);
                cgh.parallel_for(
                    sycl::nd_range<2>{global_range, local_range},[=](sycl::nd_item<2> item){
                        size_t i = item.get_global_id(0);
                        size_t j = item.get_global_id(1);
                        size_t local_i = item.get_local_id(0);
                        size_t local_j = item.get_local_id(1);
                        size_t local_index = local_i * tile_size + local_j;
                        size_t number_of_tiles = (colsA + tile_size - 1) / tile_size;
                        T sum = T(0);
                        for (size_t tile = 0; tile < number_of_tiles; tile++)
                        {
                            size_t colA = tile * tile_size + local_j;
                            size_t rowB = tile * tile_size + local_i;
                            tileA[local_index] =
                                (i < rowsA && colA < colsA) ? dataA[i * colsA + colA] : T(0);
                            tileB[local_index] =
                                (rowB < rowsB && j < colsB) ? dataB[rowB * colsB + j] : T(0);
                            item.barrier(sycl::access::fence_space::local_space);
                            for (size_t k = 0; k < tile_size; k++)
                            {
                                sum += tileA[local_i * tile_size + k] *
                                       tileB[k * tile_size + local_j];
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }
                        if (i < rowsC && j < colsC)
                        {
                            dataC[i * colsC + j] = sum;
                        }
                    });
            });
            if(kernel_event != nullptr){
                *kernel_event = event;
            }
            event.wait();
            return C;
        }
        Tensor<T> C(output_shape); //output tensor
        { //Sycl scope
            sycl::buffer<T, 1> buffc = C.to_sycl_buffer();
            sycl::buffer<T, 1> buffa = A.to_sycl_buffer();
            sycl::buffer<T, 1> buffb = B.to_sycl_buffer();
            sycl::event event = Q.submit([&](sycl::handler &cgh){
                auto acc_matC = buffc.template get_access<sycl::access::mode::write>(cgh);
                auto acc_matA = buffa.template get_access<sycl::access::mode::read>(cgh);
                auto acc_matB = buffb.template get_access<sycl::access::mode::read>(cgh);
                sycl::local_accessor<T, 1> tileA(sycl::range<1>{tile_size * tile_size}, cgh);
                sycl::local_accessor<T, 1> tileB(sycl::range<1>{tile_size * tile_size}, cgh);
                cgh.parallel_for(
                    sycl::nd_range<2>{
                        sycl::range<2>{global_rows, global_cols},
                        sycl::range<2>{tile_size, tile_size}},
                    [=](sycl::nd_item<2> item){
                        size_t i = item.get_global_id(0);
                        size_t j = item.get_global_id(1);
                        size_t local_i = item.get_local_id(0);
                        size_t local_j = item.get_local_id(1);
                        size_t local_index = local_i * tile_size + local_j;
                        size_t number_of_tiles = (colsA + tile_size - 1) / tile_size;
                        T sum = T(0);
                        for (size_t tile = 0; tile < number_of_tiles; tile++)
                        {
                            size_t colA = tile * tile_size + local_j;
                            size_t rowB = tile * tile_size + local_i;
                            tileA[local_index] =
                                (i < rowsA && colA < colsA) ? acc_matA[i * colsA + colA] : T(0);
                            tileB[local_index] =
                                (rowB < rowsB && j < colsB) ? acc_matB[rowB * colsB + j] : T(0);
                            item.barrier(sycl::access::fence_space::local_space);
                            for (size_t k = 0; k < tile_size; k++)
                            {
                                sum += tileA[local_i * tile_size + k] *
                                       tileB[k * tile_size + local_j];
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }
                        if (i < rowsC && j < colsC)
                        {
                            acc_matC[i * colsC + j] = sum;
                        }
                    });
            });
            if(kernel_event != nullptr){
                *kernel_event = event;
            }
        }
        return C;
    }

    template <typename T>
    Tensor<T> tensor_operations::gemm_blocked2x2(const Tensor<T> &A, const Tensor<T> &B, sycl::queue Q,
                                                 sycl::event *kernel_event)
    {
        std::vector<std::size_t> output_shape = gemm_output_shape(A, B);
        /*
            This GEMM uses register blocking to compute 2x2 blocks of the output matrix at a time.
             It is designed for small matrices that fit in the cache. The kernel computes a 2x2 block of the output matrix C at a time, using registers to hold the intermediate sums.
             This can improve performance by reducing memory accesses and taking advantage of data locality.
         */
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
            throw std::invalid_argument("Tensor dimensions do not match for multiplication");
        }
        if(!A.is_device() || !B.is_device())
        {
            throw std::invalid_argument("Blocked GEMM requires device tensors");
        }
        if(!A.is_accessible_from(Q) || !B.is_accessible_from(Q))
        {
            throw std::invalid_argument("GEMM queue cannot access the input tensors");
        }
        const T* dataA = A.device_data(); //get the device data pointer for A
        const T* dataB = B.device_data();

        constexpr size_t block_size = 2;
        constexpr size_t work_group_size = 16;
        size_t row_blocks = (rowsC + block_size - 1) / block_size;
        size_t col_blocks = (colsC + block_size - 1) / block_size;
        size_t global_rows = ((row_blocks + work_group_size - 1) / work_group_size) * work_group_size;
        size_t global_cols = ((col_blocks + work_group_size - 1) / work_group_size) * work_group_size;

        sycl::range<2> global_size{global_rows, global_cols};
        sycl::range<2> local_size{work_group_size, work_group_size};

        Tensor<T> C(output_shape, Q); //device output tensor
        T* dataC = C.device_data();

        sycl::event event = Q.submit([&](sycl::handler &cgh){
            cgh.parallel_for(
                sycl::nd_range<2>{global_size, local_size},
                [=](sycl::nd_item<2> item) {
                    size_t row = item.get_global_id(0) * block_size;
                    size_t col = item.get_global_id(1) * block_size;

                    //Register performance. Each work item will have 4 values:

                    T c00 = T(0);
                    T c01 = T(0);
                    T c10 = T(0);
                    T c11 = T(0);

                    for (size_t k = 0; k < colsA; k++)
                    {
                        T a0 = row < rowsA ? dataA[row * colsA + k] : T(0);
                        T a1 = row + 1 < rowsA ? dataA[(row + 1) * colsA + k] : T(0);
                        T b0 = col < colsB ? dataB[k * colsB + col] : T(0);
                        T b1 = col + 1 < colsB ? dataB[k * colsB + col + 1] : T(0);

                        c00 += a0 * b0;
                        c01 += a0 * b1;
                        c10 += a1 * b0;
                        c11 += a1 * b1;
                    }

                    if(row < rowsC && col < colsC){
                        dataC[row * colsC + col] = c00;
                    }
                    if(row < rowsC && col + 1 < colsC){
                        dataC[row * colsC + col + 1] = c01;
                    }
                    if(row + 1 < rowsC && col < colsC){
                        dataC[(row + 1) * colsC + col] = c10;
                    }
                    if(row + 1 < rowsC && col + 1 < colsC){
                        dataC[(row + 1) * colsC + col + 1] = c11;
                    }
                });
        });

        if(kernel_event != nullptr){
            *kernel_event = event;
        }
        event.wait();
        return C;
    }

    template <typename T>
    Tensor<T> tensor_operations::gemm_tiled_blocked2x2(const Tensor<T> &A, const Tensor<T> &B, sycl::queue Q, sycl::event *kernel_event)
    {
        std::vector<std::size_t> output_shape = gemm_output_shape(A, B);
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
            //because the result matrix will have the same number of rows as A and the same number of columns as B
            std::cout<<"Cols of A : "<<colsA<<std::endl;
            std::cout<<"Rows of B : "<<rowsB<<std::endl;
            throw std::invalid_argument("Tensor dimensions do not match for multiplication");
        }
        if(!A.is_device() || !B.is_device())
        {
            throw std::invalid_argument("Tiled blocked GEMM requires device tensors");
        }
        if(!A.is_accessible_from(Q) || !B.is_accessible_from(Q))
        {
            throw std::invalid_argument("GEMM queue cannot access the input tensors");
        }

        constexpr size_t tile_size = 16;
        constexpr size_t block_size = 2;
        constexpr size_t output_tile_size = tile_size * block_size;
        size_t row_blocks = (rowsC + block_size - 1) / block_size;
        size_t col_blocks = (colsC + block_size - 1) / block_size;
        size_t global_rows = ((row_blocks + tile_size - 1) / tile_size) * tile_size;
        size_t global_cols = ((col_blocks + tile_size - 1) / tile_size) * tile_size;
        size_t number_of_tiles = (colsA + tile_size - 1) / tile_size;

        const T* dataA = A.device_data();
        const T* dataB = B.device_data();
        Tensor<T> C(output_shape, Q); //device output tensor
        T* dataC = C.device_data();
        sycl::range<2> global_range{global_rows, global_cols};
        sycl::range<2> local_range{tile_size, tile_size};

        sycl::event event = Q.submit([&](sycl::handler &cgh){
            sycl::local_accessor<T, 1> tileA(sycl::range<1>{output_tile_size * tile_size}, cgh);
            sycl::local_accessor<T, 1> tileB(sycl::range<1>{tile_size * output_tile_size}, cgh);

            cgh.parallel_for(sycl::nd_range<2>{global_range, local_range},[=](sycl::nd_item<2> item){
                    size_t local_i = item.get_local_id(0);
                    size_t local_j = item.get_local_id(1);
                    size_t row = item.get_global_id(0) * block_size;
                    size_t col = item.get_global_id(1) * block_size;
                    size_t group_row = item.get_group(0) * output_tile_size;
                    size_t group_col = item.get_group(1) * output_tile_size;

                    T c00 = T(0);
                    T c01 = T(0);
                    T c10 = T(0);
                    T c11 = T(0);

                    for(size_t tile = 0; tile < number_of_tiles; tile++)
                    {
                        size_t colA = tile * tile_size + local_j;
                        size_t rowB = tile * tile_size + local_i;
                        size_t tile_row = local_i * block_size;
                        size_t tile_col = local_j * block_size;

                        tileA[tile_row * tile_size + local_j] =
                            (group_row + tile_row < rowsA && colA < colsA) ?
                            dataA[(group_row + tile_row) * colsA + colA] : T(0);
                        tileA[(tile_row + 1) * tile_size + local_j] =
                            (group_row + tile_row + 1 < rowsA && colA < colsA) ?
                            dataA[(group_row + tile_row + 1) * colsA + colA] : T(0);

                        tileB[local_i * output_tile_size + tile_col] =
                            (rowB < rowsB && group_col + tile_col < colsB) ?
                            dataB[rowB * colsB + group_col + tile_col] : T(0);
                        tileB[local_i * output_tile_size + tile_col + 1] =
                            (rowB < rowsB && group_col + tile_col + 1 < colsB) ?
                            dataB[rowB * colsB + group_col + tile_col + 1] : T(0);

                        item.barrier(sycl::access::fence_space::local_space);

                        for(size_t k = 0; k < tile_size; k++)
                        {
                            T a0 = tileA[tile_row * tile_size + k];
                            T a1 = tileA[(tile_row + 1) * tile_size + k];
                            T b0 = tileB[k * output_tile_size + tile_col];
                            T b1 = tileB[k * output_tile_size + tile_col + 1];

                            c00 += a0 * b0;
                            c01 += a0 * b1;
                            c10 += a1 * b0;
                            c11 += a1 * b1;
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if(row < rowsC && col < colsC){
                        dataC[row * colsC + col] = c00;
                    }
                    if(row < rowsC && col + 1 < colsC){
                        dataC[row * colsC + col + 1] = c01;
                    }
                    if(row + 1 < rowsC && col < colsC){
                        dataC[(row + 1) * colsC + col] = c10;
                    }
                    if(row + 1 < rowsC && col + 1 < colsC){
                        dataC[(row + 1) * colsC + col + 1] = c11;
                    }
                });
        });

        if(kernel_event != nullptr){
            *kernel_event = event;
        }
        event.wait();
        return C;
    }

    template <typename T>
    Tensor<T> tensor_operations::matxvec(const Tensor<T> &A, const Tensor<T> &B, sycl::queue Q,
                                         sycl::event *kernel_event)
    {
        std::vector<std::size_t> output_shape = gemm_output_shape(A, B);
        size_t colsA = static_cast<size_t>(A.getCols());
        size_t rowsA = static_cast<size_t>(A.getRows());
        size_t colsB = static_cast<size_t>(B.getCols());
        size_t rowsB = static_cast<size_t>(B.getRows());
        if(colsA != rowsB)
        {
            throw std::invalid_argument("Tensor dimensions do not match for multiplication");
        }
        if(colsB != 1)
        {
            throw std::invalid_argument("The second tensor must be a vector");
        }

        if(A.is_device() || B.is_device())
        {
            if(!A.is_device() || !B.is_device()){
                throw std::invalid_argument("Matrix vector inputs must use the same memory location");
            }
            if(!A.is_accessible_from(Q) || !B.is_accessible_from(Q))
            {
                throw std::invalid_argument("Matrix vector queue cannot access the input tensors");
            }

            const T* dataA = A.device_data();
            const T* dataB = B.device_data();
            Tensor<T> C(output_shape, Q);
            T* dataC = C.device_data();

            sycl::event event = Q.submit([&](sycl::handler &cgh){
                cgh.parallel_for(
                    sycl::range<1>{rowsA},
                    [=](sycl::item<1> item){
                        size_t i = item.get_id(0);
                        T sum = T(0);
                        for(size_t k = 0; k < colsA; k++)
                        {
                            sum += dataA[i * colsA + k] * dataB[k];
                        }
                        dataC[i] = sum;
                    });
            });

            if(kernel_event != nullptr){
                *kernel_event = event;
            }
            event.wait();
            return C;
        }

        Tensor<T> C(output_shape);
        {
            sycl::buffer<T, 1> buffc = C.to_sycl_buffer();
            sycl::buffer<T, 1> buffa = A.to_sycl_buffer();
            sycl::buffer<T, 1> buffb = B.to_sycl_buffer();
            sycl::event event = Q.submit([&](sycl::handler &cgh){
                auto acc_matC = buffc.template get_access<sycl::access::mode::write>(cgh);
                auto acc_matA = buffa.template get_access<sycl::access::mode::read>(cgh);
                auto acc_matB = buffb.template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for(
                    sycl::range<1>{rowsA},
                    [=](sycl::item<1> item){
                        size_t i = item.get_id(0);
                        T sum = T(0);
                        for(size_t k = 0; k < colsA; k++)
                        {
                            sum += acc_matA[i * colsA + k] * acc_matB[k];
                        }
                        acc_matC[i] = sum;
                    });
            });
            if(kernel_event != nullptr){
                *kernel_event = event;
            }
        }

        return C;
    }

    template<typename T>
    Tensor<T> tensor_operations::gemm(const Tensor<T>& A, const Tensor<T>& B, sycl::queue Q,
                                      sycl::event* kernel_event)
    {
        if(A.is_device() != B.is_device()){
            throw std::invalid_argument("GEMM inputs must use the same memory location");
        }

        if(A.is_host()){
            return gemm_naive(A, B, Q, kernel_event);
        }

        if(B.getCols() == 1){
            return matxvec(A, B, Q, kernel_event);
        }

        sycl::device selected_device = Q.get_device();
        if(!selected_device.is_gpu()){
            return gemm_naive(A, B, Q, kernel_event);
        }

        sycl::backend selected_backend = Q.get_backend();
        if(selected_backend == sycl::backend::ext_oneapi_cuda){
            return gemm_blocked2x2(A, B, Q, kernel_event);
        }
        if(selected_backend == sycl::backend::ext_oneapi_level_zero){
            if(A.getCols() <= 64){
                return gemmTiled(A, B, Q, kernel_event);
            }
            return gemm_tiled_blocked2x2(A, B, Q, kernel_event);
        }
        if(selected_backend == sycl::backend::opencl){
            return gemm_tiled_blocked2x2(A, B, Q, kernel_event);
        }

        return gemmTiled(A, B, Q, kernel_event);
    }

    template<typename T>
    Tensor<T> tensor_operations::permute(const Tensor<T>& input, const std::vector<std::size_t>& order,
                                         sycl::queue Q, sycl::event* kernel_event)
    {
        constexpr std::size_t maximum_rank = 16;
        std::size_t rank = input.getRank();
        if(rank == 0 || rank > maximum_rank){
            throw std::invalid_argument("Permute supports tensors with 1 to 16 dimensions");
        }
        if(order.size() != rank){
            throw std::invalid_argument("Permutation order must contain one entry for every dimension");
        }

        std::array<std::size_t, maximum_rank> used{};
        std::array<std::size_t, maximum_rank> input_strides{};
        std::array<std::size_t, maximum_rank> output_strides{};
        std::array<std::size_t, maximum_rank> permutation{};
        std::vector<std::size_t> output_shape(rank);
        const std::vector<std::size_t>& input_shape = input.getShape();

        for(std::size_t axis = 0; axis < rank; axis++){
            if(order[axis] >= rank || used[order[axis]] != 0){
                throw std::invalid_argument("Permutation order must contain every dimension exactly once");
            }
            used[order[axis]] = 1;
            permutation[axis] = order[axis];
            output_shape[axis] = input_shape[order[axis]];
        }

        input_strides[rank - 1] = 1;
        output_strides[rank - 1] = 1;
        for(std::size_t axis = rank - 1; axis > 0; axis--){
            input_strides[axis - 1] = input_strides[axis] * input_shape[axis];
            output_strides[axis - 1] = output_strides[axis] * output_shape[axis];
        }

        if(input.is_host()){
            Tensor<T> output(output_shape);
            for(std::size_t output_index = 0; output_index < input.getSize(); output_index++){
                std::size_t remaining = output_index;
                std::size_t input_index = 0;
                for(std::size_t axis = 0; axis < rank; axis++){
                    std::size_t coordinate = remaining / output_strides[axis];
                    remaining %= output_strides[axis];
                    input_index += coordinate * input_strides[permutation[axis]];
                }
                output[output_index] = input[input_index];
            }
            return output;
        }

        if(!input.is_accessible_from(Q)){
            throw std::invalid_argument("Permute queue cannot access the input tensor");
        }

        Tensor<T> output(output_shape, Q);
        const T* input_data = input.device_data();
        T* output_data = output.device_data();
        std::size_t total_size = input.getSize();

        if(total_size == 0){
            return output;
        }

        sycl::event event = Q.submit([&](sycl::handler& cgh){
            cgh.parallel_for(sycl::range<1>{total_size}, [=](sycl::item<1> item){
                std::size_t output_index = item.get_id(0);
                std::size_t remaining = output_index;
                std::size_t input_index = 0;
                for(std::size_t axis = 0; axis < rank; axis++){
                    std::size_t coordinate = remaining / output_strides[axis];
                    remaining %= output_strides[axis];
                    input_index += coordinate * input_strides[permutation[axis]];
                }
                output_data[output_index] = input_data[input_index];
            });
        });

        if(kernel_event != nullptr){
            *kernel_event = event;
        }
        event.wait();
        return output;
    }


    template<typename T>
    T tensor_operations::dot(const Tensor<T>& A, const Tensor<T>& B, sycl::queue Q){
        (void)Q;
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
    T tensor_operations::reduction(const Tensor<T> &A, sycl::queue Q)
    {
        // Assuming A is a 1D vector or 1D list.
        if(A.getCols() != 1 && A.getRows() != 1){
            // If the input is not a 1D vector or list, throw an error.
            std::cout<<"Cols: "<<A.getCols()<<std::endl;
            std::cout<<"Rows: "<<A.getRows()<<std::endl;
            throw std::invalid_argument("Reduction is only defined for 1D vectors, arrays or lists");
        }
        std::size_t N = A.getRows() * A.getCols();
        /*  Creates the size of the workgroups */
        std::size_t work_group_size = 64; //desired work group size
        std::size_t n_work_groups;
        std::size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;
        n_work_groups = global_size/work_group_size; //number of work groups
        std::cout<<"Number of work groups: "<<n_work_groups<<std::endl;
       
        Tensor<T> partial(n_work_groups, 1); //partial sum of the work groups
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
    template Tensor<double>    tensor_operations::gemm(const Tensor<double>&, const Tensor<double>&, sycl::queue, sycl::event*);
    template Tensor<float>     tensor_operations::gemm(const Tensor<float>&, const Tensor<float>&, sycl::queue, sycl::event*);
    template Tensor<int>       tensor_operations::gemm(const Tensor<int>&, const Tensor<int>&, sycl::queue, sycl::event*);
    template Tensor<double>    tensor_operations::gemm_naive(const Tensor<double>&, const Tensor<double>&, sycl::queue, sycl::event*);
    template Tensor<float>     tensor_operations::gemm_naive(const Tensor<float>&, const Tensor<float>&, sycl::queue, sycl::event*);
    template Tensor<int>       tensor_operations::gemm_naive(const Tensor<int>&, const Tensor<int>&, sycl::queue, sycl::event*);
    template Tensor<double>    tensor_operations::matxvec(const Tensor<double>&, const Tensor<double>&, sycl::queue, sycl::event*);
    template Tensor<float>     tensor_operations::matxvec(const Tensor<float>&, const Tensor<float>&, sycl::queue, sycl::event*);
    template Tensor<int>       tensor_operations::matxvec(const Tensor<int>&, const Tensor<int>&, sycl::queue, sycl::event*);
    template double         tensor_operations::dot(const Tensor<double>&, const Tensor<double>&, sycl::queue);
    template float          tensor_operations::dot(const Tensor<float>&, const Tensor<float>&, sycl::queue);
    template int            tensor_operations::dot(const Tensor<int>&, const Tensor<int>&, sycl::queue);

    template Tensor<double>    tensor_operations::gemmTiled(const Tensor<double>&, const Tensor<double>&, sycl::queue, sycl::event*);
    template Tensor<float>     tensor_operations::gemmTiled(const Tensor<float>&, const Tensor<float>&, sycl::queue, sycl::event*);
    template Tensor<int>       tensor_operations::gemmTiled(const Tensor<int>&, const Tensor<int>&, sycl::queue, sycl::event*);
    template Tensor<double>    tensor_operations::gemm_blocked2x2(const Tensor<double>&, const Tensor<double>&, sycl::queue, sycl::event*);
    template Tensor<float>     tensor_operations::gemm_blocked2x2(const Tensor<float>&, const Tensor<float>&, sycl::queue, sycl::event*);
    template Tensor<int>       tensor_operations::gemm_blocked2x2(const Tensor<int>&, const Tensor<int>&, sycl::queue, sycl::event*);
    template Tensor<double>    tensor_operations::gemm_tiled_blocked2x2(const Tensor<double>&, const Tensor<double>&, sycl::queue, sycl::event*);
    template Tensor<float>     tensor_operations::gemm_tiled_blocked2x2(const Tensor<float>&, const Tensor<float>&, sycl::queue, sycl::event*);
    template Tensor<int>       tensor_operations::gemm_tiled_blocked2x2(const Tensor<int>&, const Tensor<int>&, sycl::queue, sycl::event*);
    template Tensor<double>    tensor_operations::permute(const Tensor<double>&, const std::vector<std::size_t>&, sycl::queue, sycl::event*);
    template Tensor<float>     tensor_operations::permute(const Tensor<float>&, const std::vector<std::size_t>&, sycl::queue, sycl::event*);
    template Tensor<int>       tensor_operations::permute(const Tensor<int>&, const std::vector<std::size_t>&, sycl::queue, sycl::event*);
    template double         tensor_operations::reduction(const Tensor<double>&, sycl::queue);
    template float          tensor_operations::reduction(const Tensor<float>&, sycl::queue);
    template int            tensor_operations::reduction(const Tensor<int>&, sycl::queue);
}
