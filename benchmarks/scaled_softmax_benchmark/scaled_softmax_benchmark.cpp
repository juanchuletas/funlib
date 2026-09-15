#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

struct SoftmaxShape {
  std::size_t batch_size;
  std::size_t token_count;
  std::size_t head_count;
  std::size_t head_size;
};

struct SoftmaxMeasurements {
  double separate;
  double fused;
};

double kernelMilliseconds(sycl::event event) {
  event.wait();
  std::uint64_t start =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  std::uint64_t end =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

SoftmaxMeasurements measure(const flib::Tensor<float> &scores, float factor,
                            sycl::queue Q) {
  sycl::event scale_event;
  sycl::event softmax_event;
  sycl::event fused_event;
  flib::Tensor<float> scaled =
      flib::operations::scale(scores, factor, Q, &scale_event);
  flib::Tensor<float> separate =
      flib::operations::softmax(scaled, Q, &softmax_event);
  flib::Tensor<float> fused =
      flib::operations::scaled_softmax(scores, factor, Q, &fused_event);
  (void)separate;
  (void)fused;
  return {kernelMilliseconds(scale_event) + kernelMilliseconds(softmax_event),
          kernelMilliseconds(fused_event)};
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void benchmark(const SoftmaxShape &shape, sycl::queue Q) {
  constexpr std::size_t warmups = 2;
  constexpr std::size_t repetitions = 7;
  std::size_t size = shape.batch_size * shape.head_count * shape.token_count *
                     shape.token_count;
  std::vector<float> values(size);
  for (std::size_t i = 0; i < size; i++) {
    values[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 8.0f;
  }
  flib::Tensor<float> scores({shape.batch_size, shape.head_count,
                              shape.token_count, shape.token_count},
                             Q);
  scores.copy_from(values.data(), Q).wait();
  float factor = 1.0f / std::sqrt(static_cast<float>(shape.head_size));

  for (std::size_t i = 0; i < warmups; i++) {
    measure(scores, factor, Q);
  }
  std::vector<double> separate;
  std::vector<double> fused;
  for (std::size_t i = 0; i < repetitions; i++) {
    SoftmaxMeasurements result = measure(scores, factor, Q);
    separate.push_back(result.separate);
    fused.push_back(result.fused);
  }

  double separate_median = median(separate);
  double fused_median = median(fused);
  std::cout << std::setw(5) << shape.batch_size << std::setw(9)
            << shape.token_count << std::setw(8) << shape.head_count
            << std::setw(16) << separate_median << std::setw(14) << fused_median
            << std::setw(12) << separate_median / fused_median << std::endl;
}

int main() {
  // flib::sycl_handler::register_queue("cuda", flib::device::GPU,
  //                                    flib::vendor::NVIDIA, flib::backend::CUDA,
  //                                    true);
  // sycl::queue Q = flib::sycl_handler::get_queue("cuda");
  // flib::sycl_handler::get_device_info("cuda");
  flib::sycl_handler::register_queue("intel", flib::device::GPU,
                                     flib::vendor::INTEL, flib::backend::OPENCL,
                                     true);
  sycl::queue Q = flib::sycl_handler::get_queue("intel");
  flib::sycl_handler::get_device_info("intel");
  const std::vector<SoftmaxShape> shapes{
      {1, 196, 8, 64},
      {1, 512, 8, 64},
      {2, 196, 8, 64},
  };
  std::cout << std::fixed << std::setprecision(3);
  std::cout << std::setw(5) << "B" << std::setw(9) << "Tokens" << std::setw(8)
            << "Heads" << std::setw(16) << "Separate ms" << std::setw(14)
            << "Fused ms" << std::setw(12) << "Speedup" << std::endl;
  for (const SoftmaxShape &shape : shapes) {
    benchmark(shape, Q);
  }
  return 0;
}
