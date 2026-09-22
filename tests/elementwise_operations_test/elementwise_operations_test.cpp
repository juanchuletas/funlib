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
  if (actual.size() != expected.size()) {
    std::cerr << name << " produced the wrong number of values" << std::endl;
    return false;
  }
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
  flib::Tensor<float> bias({3});
  flib::Tensor<float> packed({2, 2, 6});
  std::vector<float> added(left.getSize());
  std::vector<float> biased(left.getSize());
  std::vector<float> activated(left.getSize());
  std::vector<float> gated(left.getSize());
  for (std::size_t feature = 0; feature < bias.getSize(); feature++) {
    bias[feature] = static_cast<float>(feature + 1) / 4.0f;
  }
  for (std::size_t i = 0; i < left.getSize(); i++) {
    left[i] = static_cast<float>(static_cast<int>(i) - 6) / 2.0f;
    right[i] = static_cast<float>(i % 4) / 3.0f;
    added[i] = left[i] + right[i];
    biased[i] = left[i] + bias[i % 3];
    activated[i] = geluReference(left[i]);
    std::size_t row = i / 3;
    std::size_t column = i % 3;
    float value = left[i];
    float gate = right[i];
    packed[row * 6 + column] = value;
    packed[row * 6 + 3 + column] = gate;
    gated[i] = value * geluReference(gate);
  }

  flib::Tensor<float> add_output = flib::operations::add(left, right, Q);
  flib::Tensor<float> bias_output = flib::operations::add_bias(left, bias, Q);
  flib::Tensor<float> gelu_output = flib::operations::gelu(left, Q);
  flib::Tensor<float> geglu_output = flib::operations::geglu(packed, Q);
  if (bias_output.getShape() != left.getShape() ||
      geglu_output.getShape() != std::vector<std::size_t>{2, 2, 3}) {
    std::cerr << "Host elementwise operation produced the wrong shape"
              << std::endl;
    return false;
  }
  return compareValues("Host add", add_output, added, Q) &&
         compareValues("Host add bias", bias_output, biased, Q) &&
         compareValues("Host GELU", gelu_output, activated, Q) &&
         compareValues("Host GEGLU", geglu_output, gated, Q);
}

bool checkDevice(sycl::queue Q) {
  std::vector<float> left_values(24);
  std::vector<float> right_values(24);
  std::vector<float> added(24);
  std::vector<float> biased(24);
  std::vector<float> activated(24);
  std::vector<float> bias_values{0.25f, -0.5f, 0.75f, -1.0f};
  std::vector<float> packed_values(48);
  std::vector<float> gated(24);
  for (std::size_t i = 0; i < left_values.size(); i++) {
    left_values[i] = static_cast<float>(static_cast<int>(i) - 12) / 4.0f;
    right_values[i] = static_cast<float>(i % 5) / 5.0f;
    added[i] = left_values[i] + right_values[i];
    biased[i] = left_values[i] + bias_values[i % 4];
    activated[i] = geluReference(left_values[i]);
    std::size_t row = i / 4;
    std::size_t column = i % 4;
    packed_values[row * 8 + column] = left_values[i];
    packed_values[row * 8 + 4 + column] = right_values[i];
    gated[i] = left_values[i] * geluReference(right_values[i]);
  }

  flib::Tensor<float> left({2, 3, 4}, Q);
  flib::Tensor<float> right({2, 3, 4}, Q);
  flib::Tensor<float> bias({4}, Q);
  flib::Tensor<float> packed({2, 3, 8}, Q);
  left.copy_from(left_values.data(), Q).wait();
  right.copy_from(right_values.data(), Q).wait();
  bias.copy_from(bias_values.data(), Q).wait();
  packed.copy_from(packed_values.data(), Q).wait();
  sycl::event add_event;
  sycl::event bias_event;
  sycl::event gelu_event;
  sycl::event geglu_event;
  flib::Tensor<float> add_output =
      flib::operations::add(left, right, Q, &add_event);
  flib::Tensor<float> bias_output =
      flib::operations::add_bias(left, bias, Q, &bias_event);
  flib::Tensor<float> gelu_output =
      flib::operations::gelu(left, Q, &gelu_event);
  flib::Tensor<float> geglu_output =
      flib::operations::geglu(packed, Q, &geglu_event);
  add_event.wait();
  bias_event.wait();
  gelu_event.wait();
  geglu_event.wait();
  if (bias_output.getShape() != left.getShape() ||
      geglu_output.getShape() != std::vector<std::size_t>{2, 3, 4}) {
    std::cerr << "Device elementwise operation produced the wrong shape"
              << std::endl;
    return false;
  }
  return compareValues("Device add", add_output, added, Q) &&
         compareValues("Device add bias", bias_output, biased, Q) &&
         compareValues("Device GELU", gelu_output, activated, Q) &&
         compareValues("Device GEGLU", geglu_output, gated, Q);
}

bool checkInvalidShape(sycl::queue Q) {
  try {
    flib::Tensor<float> left({2, 3});
    flib::Tensor<float> right({3, 2});
    flib::operations::add(left, right, Q);
    std::cerr << "Add accepted tensors with different shapes" << std::endl;
    return false;
  } catch (const std::invalid_argument &) {
  }

  try {
    flib::Tensor<float> input({2, 3, 4});
    flib::Tensor<float> bias({3});
    flib::operations::add_bias(input, bias, Q);
    std::cerr << "Add bias accepted the wrong bias size" << std::endl;
    return false;
  } catch (const std::invalid_argument &) {
  }

  try {
    flib::Tensor<float> input({2, 3, 7});
    flib::operations::geglu(input, Q);
    std::cerr << "GEGLU accepted an odd final dimension" << std::endl;
    return false;
  } catch (const std::invalid_argument &) {
  }
  return true;
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
