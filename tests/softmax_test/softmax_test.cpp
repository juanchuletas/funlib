#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

    std::vector<float> softmaxReference(const std::vector<float>& input,
                                        std::size_t row_size)
    {
        std::vector<float> output(input.size());
        std::size_t row_count = input.size() / row_size;
        for(std::size_t row = 0; row < row_count; row++){
            std::size_t row_offset = row * row_size;
            float maximum = input[row_offset];
            for(std::size_t column = 1; column < row_size; column++){
                maximum = std::max(maximum, input[row_offset + column]);
            }

            double sum = 0.0;
            for(std::size_t column = 0; column < row_size; column++){
                float exponential = std::exp(input[row_offset + column] - maximum);
                output[row_offset + column] = exponential;
                sum += exponential;
            }
            for(std::size_t column = 0; column < row_size; column++){
                output[row_offset + column] /= static_cast<float>(sum);
            }
        }
        return output;
    }

    bool checkResult(const std::string& name,
                     const flib::Tensor<float>& result,
                     const std::vector<std::size_t>& expected_shape,
                     const std::vector<float>& expected,
                     sycl::queue Q)
    {
        if(result.getShape() != expected_shape){
            std::cerr<<name<<" produced the wrong shape"<<std::endl;
            return false;
        }

        std::vector<float> actual = result.to_host(Q);
        std::size_t row_size = expected_shape.back();
        for(std::size_t i = 0; i < actual.size(); i++){
            float tolerance = 1.0e-5f + 1.0e-4f * std::abs(expected[i]);
            if(!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance){
                std::cerr<<name<<" mismatch at index "<<i<<std::endl;
                std::cerr<<"Expected: "<<expected[i]<<std::endl;
                std::cerr<<"Actual: "<<actual[i]<<std::endl;
                return false;
            }
        }

        for(std::size_t row = 0; row < actual.size() / row_size; row++){
            float sum = 0.0f;
            for(std::size_t column = 0; column < row_size; column++){
                sum += actual[row * row_size + column];
            }
            if(std::abs(sum - 1.0f) > 2.0e-4f){
                std::cerr<<name<<" row "<<row<<" does not sum to one"<<std::endl;
                return false;
            }
        }
        return true;
    }

    bool checkCase(const std::string& name,
                   const std::vector<std::size_t>& shape,
                   const std::vector<float>& values,
                   sycl::queue Q)
    {
        flib::Tensor<float> hostInput(shape);
        for(std::size_t i = 0; i < values.size(); i++){
            hostInput[i] = values[i];
        }
        std::vector<float> expected = softmaxReference(values, shape.back());
        flib::Tensor<float> hostOutput = flib::operations::softmax(hostInput, Q);
        if(!checkResult(name + " host", hostOutput, shape, expected, Q)){
            return false;
        }

        flib::Tensor<float> deviceInput(shape, Q);
        deviceInput.copy_from(values.data(), Q).wait();
        sycl::event kernel_event;
        flib::Tensor<float> deviceOutput =
            flib::operations::softmax(deviceInput, Q, &kernel_event);
        kernel_event.wait();
        if(!checkResult(name + " device", deviceOutput, shape, expected, Q)){
            return false;
        }

        std::cout<<"Passed "<<name<<std::endl;
        return true;
    }

int main()
{
    flib::sycl_handler::register_queue(
        "intel",
        flib::device::GPU,
        flib::vendor::INTEL,
        flib::backend::OPENCL,
        true);
    sycl::queue Q = flib::sycl_handler::get_queue("intel");
    flib::sycl_handler::get_device_info("intel");

    bool passed = true;
    passed = checkCase("known values", {1, 3}, {1.0f, 2.0f, 3.0f}, Q) && passed;
    passed = checkCase(
        "large stable values",
        {2, 3},
        {1000.0f, 1001.0f, 1002.0f, -1000.0f, -999.0f, -998.0f},
        Q) && passed;
    passed = checkCase(
        "multiple tensor rows",
        {2, 2, 5},
        {1.0f, -2.0f, 3.0f, 0.0f, 2.0f,
         -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
         4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
         8.0f, 2.0f, -6.0f, 1.0f, 0.0f},
        Q) && passed;

    std::vector<float> long_row(2 * 513);
    for(std::size_t i = 0; i < long_row.size(); i++){
        long_row[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 4.0f;
    }
    passed = checkCase("rows longer than one work group", {2, 513}, long_row, Q) && passed;

    try{
        flib::Tensor<float> invalid({2, 0});
        flib::operations::softmax(invalid, Q);
        std::cerr<<"Softmax accepted a zero final dimension"<<std::endl;
        passed = false;
    }
    catch(const std::invalid_argument&){
        std::cout<<"Passed invalid Softmax shape"<<std::endl;
    }

    if(!passed){
        return 1;
    }

    std::cout<<"All Softmax tests passed"<<std::endl;
    return 0;
}
