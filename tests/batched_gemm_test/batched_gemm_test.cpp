#include <funlib/funlib.hpp>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    struct ReferenceResult
    {
        std::vector<std::size_t> shape;
        std::vector<float> data;
    };

    void fillTensor(flib::Tensor<float>& tensor, int seed)
    {
        for(std::size_t i = 0; i < tensor.getSize(); i++){
            tensor[i] = static_cast<float>(static_cast<int>((i * 7 + seed) % 9) - 4);
        }
    }

    ReferenceResult referenceBatchedGemm(const flib::Tensor<float>& A,
                                         const flib::Tensor<float>& B,
                                         bool transpose_A,
                                         bool transpose_B)
    {
        const std::vector<std::size_t>& shapeA = A.getShape();
        const std::vector<std::size_t>& shapeB = B.getShape();
        std::size_t rank = shapeA.size();
        std::size_t stored_rowsA = shapeA[rank - 2];
        std::size_t stored_colsA = shapeA[rank - 1];
        std::size_t stored_rowsB = shapeB[rank - 2];
        std::size_t stored_colsB = shapeB[rank - 1];
        std::size_t rowsC = transpose_A ? stored_colsA : stored_rowsA;
        std::size_t innerA = transpose_A ? stored_rowsA : stored_colsA;
        std::size_t colsC = transpose_B ? stored_rowsB : stored_colsB;
        std::size_t batch_count = 1;

        ReferenceResult result;
        for(std::size_t axis = 0; axis + 2 < rank; axis++){
            batch_count *= shapeA[axis];
            result.shape.push_back(shapeA[axis]);
        }
        result.shape.push_back(rowsC);
        result.shape.push_back(colsC);
        result.data.assign(batch_count * rowsC * colsC, 0.0f);

        std::size_t matrix_sizeA = stored_rowsA * stored_colsA;
        std::size_t matrix_sizeB = stored_rowsB * stored_colsB;
        for(std::size_t batch = 0; batch < batch_count; batch++){
            for(std::size_t row = 0; row < rowsC; row++){
                for(std::size_t col = 0; col < colsC; col++){
                    float sum = 0.0f;
                    for(std::size_t k = 0; k < innerA; k++){
                        std::size_t indexA = transpose_A ? k * stored_colsA + row : row * stored_colsA + k;
                        std::size_t indexB = transpose_B ? col * stored_colsB + k : k * stored_colsB + col;
                        sum += A[batch * matrix_sizeA + indexA] * B[batch * matrix_sizeB + indexB];
                    }
                    result.data[batch * rowsC * colsC + row * colsC + col] = sum;
                }
            }
        }
        return result;
    }

    bool compareResult(const std::string& name,
                       const flib::Tensor<float>& result,
                       const ReferenceResult& expected,
                       sycl::queue Q)
    {
        if(result.getShape() != expected.shape){
            std::cerr<<name<<" produced the wrong output shape"<<std::endl;
            return false;
        }

        std::vector<float> actual = result.to_host(Q);
        for(std::size_t i = 0; i < actual.size(); i++){
            if(actual[i] != expected.data[i]){
                std::cerr<<name<<" mismatch at linear index "<<i<<std::endl;
                std::cerr<<"Expected: "<<expected.data[i]<<std::endl;
                std::cerr<<"Actual: "<<actual[i]<<std::endl;
                return false;
            }
        }
        return true;
    }

    bool checkCase(const std::string& name,
                   const std::vector<std::size_t>& shapeA,
                   const std::vector<std::size_t>& shapeB,
                   bool transpose_A,
                   bool transpose_B,
                   sycl::queue Q)
    {
        flib::Tensor<float> hostA(shapeA);
        flib::Tensor<float> hostB(shapeB);
        fillTensor(hostA, 2);
        fillTensor(hostB, 5);
        ReferenceResult expected = referenceBatchedGemm(hostA, hostB, transpose_A, transpose_B);

        flib::Tensor<float> hostC =
            flib::tensor_operations::gemm_batched(hostA, hostB, Q, transpose_A, transpose_B);
        if(!compareResult(name + " host", hostC, expected, Q)){
            return false;
        }

        std::vector<float> dataA = hostA.to_host(Q);
        std::vector<float> dataB = hostB.to_host(Q);
        flib::Tensor<float> deviceA(shapeA, Q);
        flib::Tensor<float> deviceB(shapeB, Q);
        deviceA.copy_from(dataA.data(), Q).wait();
        deviceB.copy_from(dataB.data(), Q).wait();
        flib::Tensor<float> deviceC =
            flib::tensor_operations::gemm_batched(deviceA, deviceB, Q, transpose_A, transpose_B);
        if(!compareResult(name + " device", deviceC, expected, Q)){
            return false;
        }

        std::cout<<"Passed "<<name<<std::endl;
        return true;
    }

    bool checkInvalidInputs(sycl::queue Q)
    {
        flib::Tensor<float> A({2, 3, 4});
        flib::Tensor<float> wrong_batches({3, 4, 5});
        flib::Tensor<float> wrong_matrix({2, 6, 5});

        try{
            flib::tensor_operations::gemm_batched(A, wrong_batches, Q);
        }
        catch(const std::invalid_argument&){
            try{
                flib::tensor_operations::gemm_batched(A, wrong_matrix, Q);
            }
            catch(const std::invalid_argument&){
                std::cout<<"Passed invalid batched GEMM inputs"<<std::endl;
                return true;
            }
        }

        std::cerr<<"Batched GEMM accepted invalid dimensions"<<std::endl;
        return false;
    }
}

int main()
{
    flib::sycl_handler::register_queue(
        "cuda",
        flib::device::GPU,
        flib::vendor::NVIDIA,
        flib::backend::CUDA,
        true);
    sycl::queue Q = flib::sycl_handler::get_queue("cuda");
    flib::sycl_handler::get_device_info("cuda");
    bool passed = true;

    passed = checkCase("normal matrices", {2, 2, 3}, {2, 3, 4}, false, false, Q) && passed;
    passed = checkCase("transposed A", {2, 4, 2}, {2, 4, 3}, true, false, Q) && passed;
    passed = checkCase("transposed B", {2, 3, 2, 4}, {2, 3, 5, 4}, false, true, Q) && passed;
    passed = checkCase("both transposed", {2, 4, 2}, {2, 3, 4}, true, true, Q) && passed;
    passed = checkInvalidInputs(Q) && passed;

    if(!passed){
        return 1;
    }

    std::cout<<"All batched GEMM tests passed"<<std::endl;
    return 0;
}
