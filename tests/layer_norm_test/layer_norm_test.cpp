#include <funlib/funlib.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<float> layerNormReference(const std::vector<float> &input,
                                      const std::vector<float> &gamma,
                                      const std::vector<float> &beta,
                                      float epsilon) {
  std::size_t row_size = gamma.size();
  std::size_t row_count = input.size() / row_size;
  std::vector<float> output(input.size());
  for (std::size_t row = 0; row < row_count; row++) {
    std::size_t offset = row * row_size;
    double mean = 0.0;
    for (std::size_t column = 0; column < row_size; column++) {
      mean += input[offset + column];
    }
    mean /= static_cast<double>(row_size);

    double variance = 0.0;
    for (std::size_t column = 0; column < row_size; column++) {
      double difference = input[offset + column] - mean;
      variance += difference * difference;
    }
    variance /= static_cast<double>(row_size);
    double inverse_standard_deviation = 1.0 / std::sqrt(variance + epsilon);
    for (std::size_t column = 0; column < row_size; column++) {
      output[offset + column] =
          static_cast<float>((input[offset + column] - mean) *
                             inverse_standard_deviation) *
              gamma[column] +
          beta[column];
    }
  }
  return output;
}

bool checkOutput(const std::string &name, const flib::Tensor<float> &output,
                 const std::vector<std::size_t> &shape,
                 const std::vector<float> &expected, sycl::queue Q) {
  if (output.getShape() != shape) {
    std::cerr << name << " produced the wrong shape" << std::endl;
    return false;
  }
  std::vector<float> actual = output.to_host(Q);
  for (std::size_t i = 0; i < actual.size(); i++) {
    float tolerance = 2.0e-4f + 2.0e-4f * std::abs(expected[i]);
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

bool checkCase(const std::string &name, const std::vector<std::size_t> &shape,
               const std::vector<float> &input_values,
               const std::vector<float> &gamma_values,
               const std::vector<float> &beta_values, sycl::queue Q) {
  constexpr float epsilon = 1.0e-5f;
  std::vector<float> expected =
      layerNormReference(input_values, gamma_values, beta_values, epsilon);

  flib::Tensor<float> host_input(shape);
  flib::Tensor<float> host_gamma({gamma_values.size()});
  flib::Tensor<float> host_beta({beta_values.size()});
  for (std::size_t i = 0; i < input_values.size(); i++) {
    host_input[i] = input_values[i];
  }
  for (std::size_t i = 0; i < gamma_values.size(); i++) {
    host_gamma[i] = gamma_values[i];
    host_beta[i] = beta_values[i];
  }
  flib::Tensor<float> host_output = flib::operations::layer_norm(
      host_input, host_gamma, host_beta, epsilon, Q);
  if (!checkOutput(name + " host", host_output, shape, expected, Q)) {
    return false;
  }

  flib::Tensor<float> device_input(shape, Q);
  flib::Tensor<float> device_gamma({gamma_values.size()}, Q);
  flib::Tensor<float> device_beta({beta_values.size()}, Q);
  device_input.copy_from(input_values.data(), Q).wait();
  device_gamma.copy_from(gamma_values.data(), Q).wait();
  device_beta.copy_from(beta_values.data(), Q).wait();
  sycl::event kernel_event;
  flib::Tensor<float> device_output = flib::operations::layer_norm(
      device_input, device_gamma, device_beta, epsilon, Q, &kernel_event);
  kernel_event.wait();
  if (!checkOutput(name + " device", device_output, shape, expected, Q)) {
    return false;
  }

  std::cout << "Passed " << name << std::endl;
  return true;
}

int main() {
  flib::sycl_handler::register_queue("intel", flib::device::GPU,
                                     flib::vendor::INTEL, flib::backend::OPENCL,
                                     true);
  sycl::queue Q = flib::sycl_handler::get_queue("intel");
  flib::sycl_handler::get_device_info("intel");

  bool passed = true;
  passed = checkCase("known values", {2, 4},
                     {1.0f, 2.0f, 3.0f, 4.0f, -2.0f, 0.0f, 2.0f, 4.0f},
                     {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, Q) &&
           passed;
  passed =
      checkCase("affine values", {1, 2, 4},
                {3.0f, -1.0f, 2.0f, 7.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                {0.5f, 1.5f, -1.0f, 2.0f}, {1.0f, -2.0f, 0.25f, 3.0f}, Q) &&
      passed;

  std::vector<float> long_input(2 * 513);
  std::vector<float> long_gamma(513);
  std::vector<float> long_beta(513);
  for (std::size_t i = 0; i < long_input.size(); i++) {
    long_input[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 5.0f;
  }
  for (std::size_t i = 0; i < long_gamma.size(); i++) {
    long_gamma[i] = 0.5f + static_cast<float>(i % 7) / 10.0f;
    long_beta[i] = static_cast<float>(static_cast<int>(i % 5) - 2) / 10.0f;
  }
  passed = checkCase("rows longer than one work group", {2, 513}, long_input,
                     long_gamma, long_beta, Q) &&
           passed;

  try {
    flib::Tensor<float> input({2, 4});
    flib::Tensor<float> gamma({3});
    flib::Tensor<float> beta({4});
    flib::operations::layer_norm(input, gamma, beta, 1.0e-5f, Q);
    std::cerr << "LayerNorm accepted an invalid gamma shape" << std::endl;
    passed = false;
  } catch (const std::invalid_argument &) {
    std::cout << "Passed invalid LayerNorm shape" << std::endl;
  }

  if (!passed) {
    return 1;
  }
  std::cout << "All LayerNorm tests passed" << std::endl;
  return 0;
}
