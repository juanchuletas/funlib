#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

std::vector<float> scaledSoftmaxReference(const std::vector<float> &input,
                                          std::size_t row_size, float scale) {
  std::vector<float> output(input.size());
  std::size_t row_count = input.size() / row_size;
  for (std::size_t row = 0; row < row_count; row++) {
    std::size_t offset = row * row_size;
    float maximum = input[offset] * scale;
    for (std::size_t column = 1; column < row_size; column++) {
      maximum = std::max(maximum, input[offset + column] * scale);
    }
    float sum = 0.0f;
    for (std::size_t column = 0; column < row_size; column++) {
      output[offset + column] =
          std::exp(input[offset + column] * scale - maximum);
      sum += output[offset + column];
    }
    for (std::size_t column = 0; column < row_size; column++) {
      output[offset + column] /= sum;
    }
  }
  return output;
}

bool checkCase(const std::vector<std::size_t> &shape,
               const std::vector<float> &values, float scale, sycl::queue Q) {
  std::vector<float> expected =
      scaledSoftmaxReference(values, shape.back(), scale);

  flib::Tensor<float> host_input(shape);
  for (std::size_t i = 0; i < values.size(); i++) {
    host_input[i] = values[i];
  }
  flib::Tensor<float> host_output =
      flib::operations::scaled_softmax(host_input, scale, Q);
  std::vector<float> host_actual = host_output.to_host(Q);
  for (std::size_t i = 0; i < host_actual.size(); i++) {
    float tolerance = 1.0e-5f + 1.0e-4f * std::abs(expected[i]);
    if (!std::isfinite(host_actual[i]) ||
        std::abs(host_actual[i] - expected[i]) > tolerance) {
      std::cerr << "Host scaled Softmax mismatch at index " << i << std::endl;
      return false;
    }
  }

  flib::Tensor<float> input(shape, Q);
  input.copy_from(values.data(), Q).wait();
  sycl::event kernel_event;
  flib::Tensor<float> output =
      flib::operations::scaled_softmax(input, scale, Q, &kernel_event);
  kernel_event.wait();
  std::vector<float> actual = output.to_host(Q);

  for (std::size_t i = 0; i < actual.size(); i++) {
    float tolerance = 1.0e-5f + 1.0e-4f * std::abs(expected[i]);
    if (!std::isfinite(actual[i]) ||
        std::abs(actual[i] - expected[i]) > tolerance) {
      std::cerr << "Device scaled Softmax mismatch at index " << i << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                     flib::vendor::NVIDIA, flib::backend::CUDA,
                                     true);
  sycl::queue Q = flib::sycl_handler::get_queue("cuda");
  flib::sycl_handler::get_device_info("cuda");

  std::vector<float> values{1.0f,  2.0f, 3.0f, -4.0f, 5.0f,  8.0f,
                            -2.0f, 0.0f, 4.0f, 1.0f,  -3.0f, 2.0f};
  bool passed = checkCase({2, 2, 3}, values, 0.125f, Q);
  passed = checkCase({2, 2, 3}, values, -0.5f, Q) && passed;

  if (!passed) {
    return 1;
  }
  std::cout << "All scaled Softmax tests passed" << std::endl;
  return 0;
}
