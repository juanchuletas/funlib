#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    template<typename T>
    flib::Tensor<T> gemmReference(const flib::Tensor<T>& A, const flib::Tensor<T>& B)
    {
        if(A.getCols() != B.getRows())
        {
            throw std::invalid_argument("Tensor dimensions do not match for multiplication");
        }

        flib::Tensor<T> C(A.getRows(), B.getCols());
        for(std::size_t i = 0; i < A.getRows(); i++)
        {
            for(std::size_t j = 0; j < B.getCols(); j++)
            {
                T sum = T(0);
                for(std::size_t k = 0; k < A.getCols(); k++)
                {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }

        return C;
    }

    template<typename T>
    void fillTensor(flib::Tensor<T>& tensor, int seed)
    {
        for(std::size_t i = 0; i < tensor.getRows(); i++)
        {
            for(std::size_t j = 0; j < tensor.getCols(); j++)
            {
                int value = static_cast<int>((i * 17 + j * 13 + seed) % 11) - 5;
                tensor(i, j) = static_cast<T>(value) / static_cast<T>(5);
            }
        }
    }

    template<typename T>
    bool almostEqual(T expected, T actual, std::size_t inner_dimension)
    {
        T absolute_tolerance = static_cast<T>(1e-5) * static_cast<T>(inner_dimension);
        T relative_tolerance = static_cast<T>(1e-4);
        T difference = std::abs(expected - actual);
        T scale = std::max(std::abs(expected), std::abs(actual));
        return difference <= absolute_tolerance + relative_tolerance * scale;
    }

    template<>
    bool almostEqual<int>(int expected, int actual, std::size_t)
    {
        return expected == actual;
    }

    template<typename T, typename Operation>
    bool checkMethod(
        const std::string& method,
        const flib::Tensor<T>& expected,
        std::size_t rowsA,
        std::size_t colsA,
        std::size_t colsB,
        sycl::queue Q,
        Operation operation)
    {
        try
        {
            flib::Tensor<T> actual = operation();

            if(actual.getRows() != rowsA || actual.getCols() != colsB)
            {
                std::cerr<<method<<" returned shape "
                         <<actual.getRows()<<"x"<<actual.getCols()
                         <<" instead of "<<rowsA<<"x"<<colsB<<std::endl;
                return false;
            }

            std::vector<T> actual_data = actual.to_host(Q);
            for(std::size_t i = 0; i < rowsA; i++)
            {
                for(std::size_t j = 0; j < colsB; j++)
                {
                    T expected_value = expected(i, j);
                    T actual_value = actual_data[i * colsB + j];
                    if(!almostEqual(expected_value, actual_value, colsA))
                    {
                        std::cerr<<method<<" mismatch at C("<<i<<", "<<j<<")"
                                 <<" for "<<rowsA<<"x"<<colsA
                                 <<" * "<<colsA<<"x"<<colsB<<std::endl;
                        std::cerr<<"Expected: "<<expected_value<<std::endl;
                        std::cerr<<"Actual: "<<actual_value<<std::endl;
                        return false;
                    }
                }
            }
        }
        catch(const std::exception& error)
        {
            std::cerr<<method<<" threw an exception for "
                     <<rowsA<<"x"<<colsA<<" * "<<colsA<<"x"<<colsB
                     <<": "<<error.what()<<std::endl;
            return false;
        }

        std::cout<<"Passed "<<method<<" for "
                 <<rowsA<<"x"<<colsA<<" * "<<colsA<<"x"<<colsB<<std::endl;
        return true;
    }

    template<typename T>
    bool checkShape(
        std::size_t rowsA,
        std::size_t colsA,
        std::size_t colsB,
        sycl::queue Q)
    {
        flib::Tensor<T> A(rowsA, colsA);
        flib::Tensor<T> B(colsA, colsB);
        fillTensor(A, 3);
        fillTensor(B, 7);

        flib::Tensor<T> expected = gemmReference(A, B);
        std::vector<T> hostA = A.to_host(Q);
        std::vector<T> hostB = B.to_host(Q);

        flib::Tensor<T> deviceA(rowsA, colsA, Q);
        flib::Tensor<T> deviceB(colsA, colsB, Q);
        deviceA.copy_from(hostA.data(), Q).wait();
        deviceB.copy_from(hostB.data(), Q).wait();

        bool passed = true;

        passed = checkMethod<T>(
            "gemm_naive",
            expected,
            rowsA,
            colsA,
            colsB,
            Q,
            [&](){
                return flib::tensor_operations::gemm_naive(deviceA, deviceB, Q);
            }) && passed;

        passed = checkMethod<T>(
            "gemm dispatcher",
            expected,
            rowsA,
            colsA,
            colsB,
            Q,
            [&](){
                return flib::tensor_operations::gemm(deviceA, deviceB, Q);
            }) && passed;

        passed = checkMethod<T>(
            "gemmTiled",
            expected,
            rowsA,
            colsA,
            colsB,
            Q,
            [&](){
                return flib::tensor_operations::gemmTiled(deviceA, deviceB, Q);
            }) && passed;

        passed = checkMethod<T>(
            "gemm_blocked2x2",
            expected,
            rowsA,
            colsA,
            colsB,
            Q,
            [&](){
                return flib::tensor_operations::gemm_blocked2x2(deviceA, deviceB, Q);
            }) && passed;

        passed = checkMethod<T>(
            "gemm_tiled_blocked2x2",
            expected,
            rowsA,
            colsA,
            colsB,
            Q,
            [&](){
                return flib::tensor_operations::gemm_tiled_blocked2x2(deviceA, deviceB, Q);
            }) && passed;

        return passed;
    }

    template<typename Operation>
    bool checkInvalidMethod(const std::string& method, Operation operation)
    {
        try
        {
            operation();
        }
        catch(const std::invalid_argument&)
        {
            std::cout<<"Passed invalid shape for "<<method<<std::endl;
            return true;
        }
        catch(const std::exception& error)
        {
            std::cerr<<method<<" threw the wrong exception: "<<error.what()<<std::endl;
            return false;
        }

        std::cerr<<method<<" did not reject an invalid shape"<<std::endl;
        return false;
    }

    bool checkInvalidShapes(sycl::queue Q)
    {
        flib::Tensor<float> A(4, 3);
        flib::Tensor<float> B(2, 5);
        fillTensor(A, 3);
        fillTensor(B, 7);

        std::vector<float> hostA = A.to_host(Q);
        std::vector<float> hostB = B.to_host(Q);
        flib::Tensor<float> deviceA(4, 3, Q);
        flib::Tensor<float> deviceB(2, 5, Q);
        deviceA.copy_from(hostA.data(), Q).wait();
        deviceB.copy_from(hostB.data(), Q).wait();

        bool passed = true;

        passed = checkInvalidMethod(
            "gemm_naive",
            [&](){
                flib::tensor_operations::gemm_naive(deviceA, deviceB, Q);
            }) && passed;

        passed = checkInvalidMethod(
            "gemm dispatcher",
            [&](){
                flib::tensor_operations::gemm(deviceA, deviceB, Q);
            }) && passed;

        passed = checkInvalidMethod(
            "gemmTiled",
            [&](){
                flib::tensor_operations::gemmTiled(deviceA, deviceB, Q);
            }) && passed;

        passed = checkInvalidMethod(
            "gemm_blocked2x2",
            [&](){
                flib::tensor_operations::gemm_blocked2x2(deviceA, deviceB, Q);
            }) && passed;

        passed = checkInvalidMethod(
            "gemm_tiled_blocked2x2",
            [&](){
                flib::tensor_operations::gemm_tiled_blocked2x2(deviceA, deviceB, Q);
            }) && passed;

        return passed;
    }

    bool checkNDimensionalShape(sycl::queue Q)
    {
        flib::Tensor<float> hostA({2, 3, 5});
        flib::Tensor<float> hostB({5, 7});
        fillTensor(hostA, 3);
        fillTensor(hostB, 7);

        std::vector<float> dataA = hostA.to_host(Q);
        std::vector<float> dataB = hostB.to_host(Q);
        flib::Tensor<float> deviceA({2, 3, 5}, Q);
        flib::Tensor<float> deviceB({5, 7}, Q);
        deviceA.copy_from(dataA.data(), Q).wait();
        deviceB.copy_from(dataB.data(), Q).wait();

        flib::Tensor<float> result = flib::tensor_operations::gemm(deviceA, deviceB, Q);
        const std::vector<std::size_t> expected_shape{2, 3, 7};
        if(result.getShape() != expected_shape)
        {
            std::cerr<<"GEMM did not preserve the leading tensor dimensions"<<std::endl;
            return false;
        }

        std::cout<<"Passed GEMM shape preservation for [2, 3, 5] * [5, 7]"<<std::endl;
        return true;
    }

    bool checkReshape(sycl::queue Q)
    {
        flib::Tensor<float> hostTensor({2, 3, 4});
        for(std::size_t i = 0; i < 24; i++){
            hostTensor[i] = static_cast<float>(i);
        }

        hostTensor.reshape({2, 2, 2, 3});
        const std::vector<std::size_t> expected_shape{2, 2, 2, 3};
        if(hostTensor.getShape() != expected_shape || hostTensor[17] != 17.0f)
        {
            std::cerr<<"Host tensor reshape changed its data or produced the wrong shape"<<std::endl;
            return false;
        }

        flib::Tensor<float> deviceTensor({2, 3, 4}, Q);
        float* original_pointer = deviceTensor.device_data();
        deviceTensor.reshape({6, 4});
        if(deviceTensor.device_data() != original_pointer || deviceTensor.getShape() != std::vector<std::size_t>{6, 4})
        {
            std::cerr<<"Device tensor reshape allocated memory or produced the wrong shape"<<std::endl;
            return false;
        }

        try
        {
            hostTensor.reshape({2, 2});
        }
        catch(const std::invalid_argument&)
        {
            std::cout<<"Passed tensor reshape tests"<<std::endl;
            return true;
        }

        std::cerr<<"Tensor reshape accepted a different number of elements"<<std::endl;
        return false;
    }

    bool checkPermute(sycl::queue Q)
    {
        flib::Tensor<float> hostInput({2, 3, 4});
        for(std::size_t i = 0; i < hostInput.getSize(); i++){
            hostInput[i] = static_cast<float>(i);
        }

        flib::Tensor<float> hostOutput =
            flib::tensor_operations::permute(hostInput, {0, 2, 1}, Q);
        const std::vector<std::size_t> expected_shape{2, 4, 3};
        if(hostOutput.getShape() != expected_shape){
            std::cerr<<"Host permute produced the wrong shape"<<std::endl;
            return false;
        }

        for(std::size_t batch = 0; batch < 2; batch++){
            for(std::size_t feature = 0; feature < 4; feature++){
                for(std::size_t token = 0; token < 3; token++){
                    std::size_t input_index = (batch * 3 + token) * 4 + feature;
                    std::size_t output_index = (batch * 4 + feature) * 3 + token;
                    if(hostOutput[output_index] != hostInput[input_index]){
                        std::cerr<<"Host permute produced incorrect data"<<std::endl;
                        return false;
                    }
                }
            }
        }

        std::vector<float> input_data = hostInput.to_host(Q);
        flib::Tensor<float> deviceInput({2, 3, 4}, Q);
        deviceInput.copy_from(input_data.data(), Q).wait();
        flib::Tensor<float> deviceOutput =
            flib::tensor_operations::permute(deviceInput, {0, 2, 1}, Q);
        std::vector<float> output_data = deviceOutput.to_host(Q);
        if(deviceOutput.getShape() != expected_shape){
            std::cerr<<"Device permute produced the wrong shape"<<std::endl;
            return false;
        }

        for(std::size_t i = 0; i < output_data.size(); i++){
            if(output_data[i] != hostOutput[i]){
                std::cerr<<"Device permute does not match host permute"<<std::endl;
                return false;
            }
        }

        try
        {
            flib::tensor_operations::permute(hostInput, {0, 0, 1}, Q);
        }
        catch(const std::invalid_argument&)
        {
            std::cout<<"Passed tensor permute tests"<<std::endl;
            return true;
        }

        std::cerr<<"Permute accepted a repeated dimension"<<std::endl;
        return false;
    }

    bool checkBatchedGemm(sycl::queue Q)
    {
        flib::Tensor<float> hostA({2, 3, 2, 4});
        flib::Tensor<float> hostB({2, 3, 5, 4});
        for(std::size_t i = 0; i < hostA.getSize(); i++){
            hostA[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
        }
        for(std::size_t i = 0; i < hostB.getSize(); i++){
            hostB[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
        }

        std::vector<float> expected(2 * 3 * 2 * 5, 0.0f);
        for(std::size_t batch = 0; batch < 6; batch++){
            for(std::size_t row = 0; row < 2; row++){
                for(std::size_t col = 0; col < 5; col++){
                    for(std::size_t k = 0; k < 4; k++){
                        expected[batch * 10 + row * 5 + col] +=
                            hostA[batch * 8 + row * 4 + k] *
                            hostB[batch * 20 + col * 4 + k];
                    }
                }
            }
        }

        flib::Tensor<float> hostC =
            flib::tensor_operations::gemm_batched(hostA, hostB, Q, false, true);
        const std::vector<std::size_t> expected_shape{2, 3, 2, 5};
        if(hostC.getShape() != expected_shape){
            std::cerr<<"Host batched GEMM produced the wrong shape"<<std::endl;
            return false;
        }
        for(std::size_t i = 0; i < expected.size(); i++){
            if(hostC[i] != expected[i]){
                std::cerr<<"Host batched GEMM produced incorrect data"<<std::endl;
                return false;
            }
        }

        std::vector<float> dataA = hostA.to_host(Q);
        std::vector<float> dataB = hostB.to_host(Q);
        flib::Tensor<float> deviceA({2, 3, 2, 4}, Q);
        flib::Tensor<float> deviceB({2, 3, 5, 4}, Q);
        deviceA.copy_from(dataA.data(), Q).wait();
        deviceB.copy_from(dataB.data(), Q).wait();
        flib::Tensor<float> deviceC =
            flib::tensor_operations::gemm_batched(deviceA, deviceB, Q, false, true);
        std::vector<float> actual = deviceC.to_host(Q);
        if(deviceC.getShape() != expected_shape){
            std::cerr<<"Device batched GEMM produced the wrong shape"<<std::endl;
            return false;
        }
        for(std::size_t i = 0; i < expected.size(); i++){
            if(actual[i] != expected[i]){
                std::cerr<<"Device batched GEMM produced incorrect data"<<std::endl;
                return false;
            }
        }

        std::cout<<"Passed batched GEMM for [2, 3, 2, 4] * [2, 3, 5, 4] transpose"<<std::endl;
        return true;
    }

}

int main()
{
    sycl::queue Q = flib::sycl_handler::get_queue();
    bool passed = true;

    passed = checkShape<float>(1, 1, 1, Q) && passed;
    passed = checkShape<float>(2, 2, 2, Q) && passed;
    passed = checkShape<float>(3, 5, 7, Q) && passed;
    passed = checkShape<float>(5, 3, 1, Q) && passed;
    passed = checkShape<float>(15, 17, 13, Q) && passed;
    passed = checkShape<float>(16, 16, 16, Q) && passed;
    passed = checkShape<float>(17, 16, 31, Q) && passed;
    passed = checkShape<float>(31, 33, 29, Q) && passed;
    passed = checkShape<float>(127, 512, 511, Q) && passed;
    passed = checkShape<double>(17, 31, 9, Q) && passed;
    passed = checkShape<int>(17, 31, 9, Q) && passed;
    passed = checkInvalidShapes(Q) && passed;
    passed = checkNDimensionalShape(Q) && passed;
    passed = checkReshape(Q) && passed;
    passed = checkPermute(Q) && passed;
    passed = checkBatchedGemm(Q) && passed;

    if(!passed)
    {
        return 1;
    }

    std::cout<<"All GEMM correctness tests passed"<<std::endl;
    return 0;
}
