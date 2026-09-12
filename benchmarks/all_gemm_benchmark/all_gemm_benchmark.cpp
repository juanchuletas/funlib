#include <funlib/funlib.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Measurements
{
    double total_time;
    double kernel_time;
    double gflops;
};

void fillTensor(flib::Tensor<float>& tensor, int seed)
{
    for(std::size_t i = 0; i < tensor.getRows(); i++)
    {
        for(std::size_t j = 0; j < tensor.getCols(); j++)
        {
            int value = static_cast<int>((i * 17 + j * 13 + seed) % 11) - 5;
            tensor(i, j) = static_cast<float>(value) / 5.0f;
        }
    }
}

template<typename Operation>
double medianTotalTime(Operation operation, std::size_t warmups, std::size_t repetitions)
{
    for(std::size_t i = 0; i < warmups; i++)
    {
        operation();
    }

    std::vector<double> times;
    times.reserve(repetitions);
    for(std::size_t i = 0; i < repetitions; i++)
    {
        auto start = std::chrono::steady_clock::now();
        operation();
        auto end = std::chrono::steady_clock::now();

        times.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

template<typename Operation>
double medianKernelTime(Operation operation, std::size_t warmups, std::size_t repetitions)
{
    for(std::size_t i = 0; i < warmups; i++)
    {
        sycl::event event;
        operation(event);
        event.wait();
    }

    std::vector<double> times;
    times.reserve(repetitions);
    for(std::size_t i = 0; i < repetitions; i++)
    {
        sycl::event event;
        operation(event);
        event.wait();

        std::uint64_t start = event.get_profiling_info<
            sycl::info::event_profiling::command_start>();
        std::uint64_t end = event.get_profiling_info<
            sycl::info::event_profiling::command_end>();
        times.push_back(static_cast<double>(end - start) / 1.0e6);
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double calculateGflops(
    std::size_t M,
    std::size_t K,
    std::size_t N,
    double milliseconds)
{
    double operations = 2.0 * static_cast<double>(M) *
                        static_cast<double>(K) *
                        static_cast<double>(N);
    return operations / (milliseconds / 1000.0) / 1.0e9;
}

template<typename TotalOperation, typename KernelOperation>
Measurements measure(
    std::size_t M,
    std::size_t K,
    std::size_t N,
    TotalOperation total_operation,
    KernelOperation kernel_operation)
{
    constexpr std::size_t warmups = 5;
    constexpr std::size_t repetitions = 21;

    double total_time = medianTotalTime(
        total_operation,
        warmups,
        repetitions);
    double kernel_time = medianKernelTime(
        kernel_operation,
        warmups,
        repetitions);

    return Measurements{
        total_time,
        kernel_time,
        calculateGflops(M, K, N, kernel_time)};
}

void printMeasurements(
    std::size_t M,
    std::size_t K,
    std::size_t N,
    const std::string& method,
    const Measurements& measurements,
    double original_kernel_time)
{
    std::cout<<std::setw(5)<<M<<"x"<<std::setw(5)<<K<<"x"<<std::setw(5)<<N
                <<std::setw(28)<<method
                <<std::setw(18)<<measurements.total_time
                <<std::setw(18)<<measurements.kernel_time
                <<std::setw(18)<<measurements.gflops
                <<std::setw(18)<<original_kernel_time / measurements.kernel_time
                <<std::endl;
}

void benchmarkShape(std::size_t M, std::size_t K, std::size_t N, sycl::queue Q)
{
    flib::Tensor<float> A(M, K);
    flib::Tensor<float> B(K, N);
    fillTensor(A, 3);
    fillTensor(B, 7);

    std::vector<float> hostA = A.to_host(Q);
    std::vector<float> hostB = B.to_host(Q);

    flib::Tensor<float> deviceA(M, K, Q);
    flib::Tensor<float> deviceB(K, N, Q);
    deviceA.copy_from(hostA.data(), Q).wait();
    deviceB.copy_from(hostB.data(), Q).wait();

    Measurements original = measure(
        M,
        K,
        N,
        [&](){
            auto C = flib::tensor_operations::gemm_naive(deviceA, deviceB, Q);
            (void)C;
        },
        [&](sycl::event& event){
            auto C = flib::tensor_operations::gemm_naive(deviceA, deviceB, Q, &event);
            (void)C;
        });

    Measurements tiled = measure(
        M,
        K,
        N,
        [&](){
            auto C = flib::tensor_operations::gemmTiled(deviceA, deviceB, Q);
            (void)C;
        },
        [&](sycl::event& event){
            auto C = flib::tensor_operations::gemmTiled(deviceA, deviceB, Q, &event);
            (void)C;
        });

    Measurements blocked = measure(M,K,N,
        [&](){
            auto C = flib::tensor_operations::gemm_blocked2x2(deviceA, deviceB, Q);
            (void)C;
        },
        [&](sycl::event& event){
            auto C = flib::tensor_operations::gemm_blocked2x2(deviceA, deviceB, Q, &event);
            (void)C;
        });

    Measurements tiled_blocked = measure(M,K,N,[&](){
            auto C = flib::tensor_operations::gemm_tiled_blocked2x2(deviceA, deviceB, Q);
            (void)C;
        },
        [&](sycl::event& event){
            auto C = flib::tensor_operations::gemm_tiled_blocked2x2(deviceA, deviceB, Q, &event);
            (void)C;
        });

    printMeasurements(M, K, N, "gemm_naive", original, original.kernel_time);
    printMeasurements(M, K, N, "gemmTiled", tiled, original.kernel_time);
    printMeasurements(M, K, N, "gemm_blocked2x2", blocked, original.kernel_time);
    printMeasurements(
        M,
        K,
        N,
        "gemm_tiled_blocked2x2",
        tiled_blocked,
        original.kernel_time);
    std::cout<<std::endl;
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

    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<<"    M x    K x    N"
             <<std::setw(28)<<"Method"
             <<std::setw(18)<<"Total ms"
             <<std::setw(18)<<"Kernel ms"
             <<std::setw(18)<<"Kernel GFLOPS"
             <<std::setw(18)<<"GEMM speedup"
             <<std::endl;

    benchmarkShape(128, 128, 128, Q);
    benchmarkShape(256, 256, 256, Q);
    benchmarkShape(512, 512, 512, Q);
    benchmarkShape(127, 512, 512, Q);
    benchmarkShape(512, 64, 512, Q);
    benchmarkShape(512, 512, 2048, Q);
    benchmarkShape(512, 2048, 512, Q);

    return 0;
}
