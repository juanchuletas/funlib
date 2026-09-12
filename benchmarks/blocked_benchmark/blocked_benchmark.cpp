#include <funlib/funlib.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

void fillTensor(flib::Tensor<float> &tensor, int seed) {
  for (std::size_t i = 0; i < tensor.getRows(); i++) {
    for (std::size_t j = 0; j < tensor.getCols(); j++) {
      int value = static_cast<int>((i * 17 + j * 13 + seed) % 11) - 5;
      tensor(i, j) = static_cast<float>(value) / 5.0f;
    }
  }
}

template <typename Operation>
double medianTime(Operation operation, std::size_t warmups,
                  std::size_t repetitions) {
  for (std::size_t i = 0; i < warmups; i++) {
    operation();
  }

  std::vector<double> times;
  times.reserve(repetitions);
  for (std::size_t i = 0; i < repetitions; i++) {
    auto start = std::chrono::steady_clock::now();
    operation();
    auto end = std::chrono::steady_clock::now();

    times.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }

  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

template <typename Operation>
double medianKernelTime(Operation operation, std::size_t warmups,
                        std::size_t repetitions) {
  for (std::size_t i = 0; i < warmups; i++) {
    sycl::event event;
    operation(event);
    event.wait();
  }

  std::vector<double> times;
  times.reserve(repetitions);
  for (std::size_t i = 0; i < repetitions; i++) {
    sycl::event event;
    operation(event);
    event.wait();

    std::uint64_t start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::uint64_t end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    times.push_back(static_cast<double>(end - start) / 1.0e6);
  }

  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

double calculateGflops(std::size_t M, std::size_t K, std::size_t N,
                       double milliseconds) {
  double operations = 2.0 * static_cast<double>(M) * static_cast<double>(K) *
                      static_cast<double>(N);
  return operations / (milliseconds / 1000.0) / 1.0e9;
}

void benchmarkShape(std::size_t M, std::size_t K, std::size_t N,
                    sycl::queue m_queue) {
  constexpr std::size_t warmups = 5;
  constexpr std::size_t repetitions = 21;

  flib::Tensor<float> A(M, K);
  flib::Tensor<float> B(K, N);
  fillTensor(A, 3);
  fillTensor(B, 7);

  std::vector<float> hostA = A.to_host(m_queue);
  std::vector<float> hostB = B.to_host(m_queue);

  flib::Tensor<float> deviceA(M, K, m_queue);
  flib::Tensor<float> deviceB(K, N, m_queue);
  deviceA.copy_from(hostA.data(), m_queue).wait();
  deviceB.copy_from(hostB.data(), m_queue).wait();

  double blocked_total_time = medianTime(
      [&]() {
        auto C =
            flib::tensor_operations::gemm_blocked2x2(deviceA, deviceB, m_queue);
        (void)C;
      },
      warmups, repetitions);

  double blocked_kernel_time = medianKernelTime(
      [&](sycl::event &event) {
        auto C = flib::tensor_operations::gemm_blocked2x2(deviceA, deviceB,
                                                          m_queue, &event);
        (void)C;
      },
      warmups, repetitions);

  double tiled_total_time = medianTime(
      [&]() {
        auto C = flib::tensor_operations::gemm_tiled_blocked2x2(
            deviceA, deviceB, m_queue);
        (void)C;
      },
      warmups, repetitions);

  double tiled_kernel_time = medianKernelTime(
      [&](sycl::event &event) {
        auto C = flib::tensor_operations::gemm_tiled_blocked2x2(
            deviceA, deviceB, m_queue, &event);
        (void)C;
      },
      warmups, repetitions);

  std::cout << std::setw(5) << M << "x" << std::setw(5) << K << "x"
            << std::setw(5) << N << std::setw(18) << blocked_total_time
            << std::setw(18) << blocked_kernel_time << std::setw(18)
            << calculateGflops(M, K, N, blocked_kernel_time) << std::setw(18)
            << tiled_total_time << std::setw(18) << tiled_kernel_time
            << std::setw(18) << calculateGflops(M, K, N, tiled_kernel_time)
            << std::setw(16) << blocked_kernel_time / tiled_kernel_time
            << std::endl;
}

int main() {
  flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                     flib::vendor::NVIDIA, flib::backend::CUDA,
                                     true);
  sycl::queue m_queue = flib::sycl_handler::get_queue("cuda");
  flib::sycl_handler::get_device_info("cuda");

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "    M x    K x    N" << std::setw(18) << "Blocked total ms"
            << std::setw(18) << "Blocked kernel ms" << std::setw(18)
            << "Blocked GFLOPS" << std::setw(18) << "Tiled total ms"
            << std::setw(18) << "Tiled kernel ms" << std::setw(18)
            << "Tiled GFLOPS" << std::setw(16) << "Tiled speedup" << std::endl;

  benchmarkShape(128, 128, 128, m_queue);
  benchmarkShape(256, 256, 256, m_queue);
  benchmarkShape(512, 512, 512, m_queue);
  benchmarkShape(127, 512, 512, m_queue);
  benchmarkShape(512, 64, 512, m_queue);

  return 0;
}
