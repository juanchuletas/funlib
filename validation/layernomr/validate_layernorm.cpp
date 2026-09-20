#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t features = 64;
constexpr std::size_t element_count = 1 * 8 * features;
constexpr float epsilon = 1.0e-5f;
constexpr double tolerance = 1.0e-4;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Reference files require IEEE 754 float32");

// NumPy tofile writes raw values in native byte order, without a header.
std::vector<float> loadReference(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Cannot open " + path);
  }
  constexpr std::streamsize bytes = element_count * sizeof(float);
  if (file.tellg() != std::streampos(bytes)) {
    throw std::runtime_error(path + " must contain exactly 512 float32 values (2048 bytes)");
  }
  file.seekg(0);
  std::vector<float> values(element_count);
  if (!file.read(reinterpret_cast<char *>(values.data()), bytes)) {
    throw std::runtime_error("Cannot read " + path);
  }
  for (float value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(path + " contains a non-finite value");
    }
  }
  return values;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 1 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << " [input.bin output.bin]\n";
    return 1;
  }
  try {
    const auto input_values = loadReference(
        argc == 3 ? argv[1] : "ref_layernorm_input.bin");
    const auto expected = loadReference(
        argc == 3 ? argv[2] : "ref_layernorm_output.bin");

    flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                       flib::vendor::NVIDIA, flib::backend::CUDA,
                                       true);
    sycl::queue queue = flib::sycl_handler::get_queue("cuda");
    flib::sycl_handler::get_device_info("cuda");

    flib::Tensor<float> input({1, 8, features}, queue);
    flib::Tensor<float> gamma({features}, queue);
    flib::Tensor<float> beta({features}, queue);
    const std::vector<float> gamma_values(features, 1.0f);
    const std::vector<float> beta_values(features, 0.0f);
    input.copy_from(input_values.data(), queue).wait_and_throw();
    gamma.copy_from(gamma_values.data(), queue).wait_and_throw();
    beta.copy_from(beta_values.data(), queue).wait_and_throw();

    auto output = flib::operations::layer_norm(input, gamma, beta, epsilon, queue);
    queue.wait_and_throw();
    const auto actual = output.to_host(queue);
    queue.wait_and_throw();
    if (output.getShape() != input.getShape() || actual.size() != expected.size()) {
      throw std::runtime_error("LayerNorm returned an unexpected shape or size");
    }

    double max_difference = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (!std::isfinite(actual[i])) {
        max_difference = std::numeric_limits<double>::infinity();
        break;
      }
      max_difference = std::max(
          max_difference, std::abs(static_cast<double>(actual[i]) - expected[i]));
    }
    const bool passed = max_difference < tolerance;
    std::cout << std::setprecision(10)
              << "Max absolute difference: " << max_difference << '\n'
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
