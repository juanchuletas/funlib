#include <funlib/funlib.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

bool compareValues(const std::string &name, const flib::Tensor<float> &output,
                   const std::vector<float> &expected, sycl::queue Q) {
  std::vector<float> actual = output.to_host(Q);
  for (std::size_t i = 0; i < actual.size(); i++) {
    float tolerance = 1.0e-6f + 1.0e-5f * std::abs(expected[i]);
    if (!std::isfinite(actual[i]) ||
        std::abs(actual[i] - expected[i]) > tolerance) {
      std::cerr << name << " mismatch at index " << i << std::endl;
      std::cerr << "Expected: " << expected[i] << std::endl;
      std::cerr << "Actual: " << actual[i] << std::endl;
      return false;
    }
  }
  return true;
}

float geluReference(float value) {
  return 0.5f * value * (1.0f + std::erf(value * 0.70710678118654752440f));
}

bool checkHost(sycl::queue Q) {
  flib::Tensor<float> left({2, 2, 3});
  flib::Tensor<float> right({2, 2, 3});
  std::vector<float> added(left.getSize());
  std::vector<float> activated(left.getSize());
  for (std::size_t i = 0; i < left.getSize(); i++) {
    left[i] = static_cast<float>(static_cast<int>(i) - 6) / 2.0f;
    right[i] = static_cast<float>(i % 4) / 3.0f;
    added[i] = left[i] + right[i];
    activated[i] = geluReference(left[i]);
  }

  flib::Tensor<float> add_output = flib::operations::add(left, right, Q);
  flib::Tensor<float> gelu_output = flib::operations::gelu(left, Q);
  return compareValues("Host add", add_output, added, Q) &&
         compareValues("Host GELU", gelu_output, activated, Q);
}

bool checkDevice(sycl::queue Q) {
  std::vector<float> left_values(24);
  std::vector<float> right_values(24);
  std::vector<float> added(24);
  std::vector<float> activated(24);
  for (std::size_t i = 0; i < left_values.size(); i++) {
    left_values[i] = static_cast<float>(static_cast<int>(i) - 12) / 4.0f;
    right_values[i] = static_cast<float>(i % 5) / 5.0f;
    added[i] = left_values[i] + right_values[i];
    activated[i] = geluReference(left_values[i]);
  }

  flib::Tensor<float> left({2, 3, 4}, Q);
  flib::Tensor<float> right({2, 3, 4}, Q);
  left.copy_from(left_values.data(), Q).wait();
  right.copy_from(right_values.data(), Q).wait();
  sycl::event add_event;
  sycl::event gelu_event;
  flib::Tensor<float> add_output =
      flib::operations::add(left, right, Q, &add_event);
  flib::Tensor<float> gelu_output =
      flib::operations::gelu(left, Q, &gelu_event);
  add_event.wait();
  gelu_event.wait();
  return compareValues("Device add", add_output, added, Q) &&
         compareValues("Device GELU", gelu_output, activated, Q);
}

bool checkInvalidShape(sycl::queue Q) {
  try {
    flib::Tensor<float> left({2, 3});
    flib::Tensor<float> right({3, 2});
    flib::operations::add(left, right, Q);
    std::cerr << "Add accepted tensors with different shapes" << std::endl;
    return false;
  } catch (const std::invalid_argument &) {
    return true;
  }
}

int main() {
  flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                     flib::vendor::NVIDIA, flib::backend::CUDA,
                                     true);
  sycl::queue Q = flib::sycl_handler::get_queue("cuda");
  flib::sycl_handler::get_device_info("cuda");

  if (!checkHost(Q) || !checkDevice(Q) || !checkInvalidShape(Q)) {
    return 1;
  }
  std::cout << "All elementwise operation tests passed" << std::endl;
  return 0;
}
