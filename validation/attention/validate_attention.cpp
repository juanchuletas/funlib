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
constexpr std::size_t batch_size = 1;
constexpr std::size_t token_count = 8;
constexpr std::size_t head_count = 1;
constexpr std::size_t head_size = 64;
constexpr std::size_t input_count = batch_size * token_count * head_size;
constexpr std::size_t weight_count = batch_size * token_count * token_count;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Reference files require IEEE 754 float32");

// Raw row-major NumPy float32 values in native byte order.
std::vector<float> loadReference(const std::string &path, std::size_t count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Cannot open " + path);
  }
  const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
  if (file.tellg() != std::streampos(bytes)) {
    throw std::runtime_error(path + " must contain exactly " +
                             std::to_string(count) + " float32 values");
  }
  file.seekg(0);
  std::vector<float> values(count);
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

void saveWeights(const std::vector<float> &values) {
  const char *path = "funlib_attn_weights.bin";
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    throw std::runtime_error(std::string("Cannot open ") + path);
  }
  file.write(reinterpret_cast<const char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  file.close();
  if (!file) {
    throw std::runtime_error(std::string("Cannot write ") + path);
  }
  std::cout << "Saved weights [1,8,8] to " << path << '\n';
}

flib::Tensor<float> deviceTensor(const std::vector<float> &values,
                                 sycl::queue queue) {
  flib::Tensor<float> tensor({batch_size, token_count, head_size}, queue);
  tensor.copy_from(values.data(), queue).wait_and_throw();
  // Match attention_test.cpp: attention expects [B,N,H,D].
  tensor.reshape({batch_size, token_count, head_count, head_size});
  return tensor;
}

std::vector<float> readOutput(const flib::Tensor<float> &tensor,
                              const std::vector<std::size_t> &shape,
                              sycl::queue queue) {
  queue.wait_and_throw();
  if (tensor.getShape() != shape) {
    throw std::runtime_error("Unexpected attention tensor shape");
  }
  auto values = tensor.to_host(queue);
  queue.wait_and_throw();
  return values;
}

void printRow(const char *label, const std::vector<float> &values,
              std::size_t row, std::size_t width) {
  std::cout << label << " [";
  for (std::size_t column = 0; column < width; ++column) {
    if (column != 0) {
      std::cout << ", ";
    }
    std::cout << values[row * width + column];
  }
  std::cout << "]\n";
}

bool compare(const char *name, const std::vector<float> &actual,
             const std::vector<float> &expected, std::size_t width,
             double tolerance) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(name) + " has an unexpected size");
  }
  std::cout << '\n' << name << '\n';
  for (std::size_t row = 0; row < actual.size() / width; ++row) {
    std::cout << "Row " << row << '\n';
    printRow("pytorch:", expected, row, width);
    printRow("funlib: ", actual, row, width);
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
  std::cout << name << " max absolute difference: " << max_difference
            << " (threshold " << tolerance << ") "
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 1 && argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " [Q.bin K.bin V.bin out.bin weights.bin]\n";
    return 1;
  }
  try {
    const auto q_values = loadReference(
        argc == 6 ? argv[1] : "ref_attn_Q.bin", input_count);
    const auto k_values = loadReference(
        argc == 6 ? argv[2] : "ref_attn_K.bin", input_count);
    const auto v_values = loadReference(
        argc == 6 ? argv[3] : "ref_attn_V.bin", input_count);
    const auto expected_output = loadReference(
        argc == 6 ? argv[4] : "ref_attn_out.bin", input_count);
    const auto expected_weights = loadReference(
        argc == 6 ? argv[5] : "ref_attn_weights.bin", weight_count);

    flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                     flib::vendor::NVIDIA, flib::backend::CUDA,
                                     true);
    sycl::queue queue = flib::sycl_handler::get_queue("cuda");
    flib::sycl_handler::get_device_info("cuda");
    auto query = deviceTensor(q_values, queue);
    auto key = deviceTensor(k_values, queue);
    auto value = deviceTensor(v_values, queue);

    // Reproduce the library's attention pipeline to expose its weights.
    auto query_heads = flib::operations::split_heads(query, queue);
    auto key_heads = flib::operations::split_heads(key, queue);
    auto value_heads = flib::operations::split_heads(value, queue);
    auto scores = flib::tensor_operations::gemm_batched(
        query_heads, key_heads, queue, false, true);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    auto weights = flib::operations::scaled_softmax(scores, scale, queue);
    const auto actual_weights = readOutput(
        weights, {batch_size, head_count, token_count, token_count}, queue);
    auto context = flib::tensor_operations::gemm_batched(
        weights, value_heads, queue);
    auto output = flib::operations::join_heads(context, queue);
    output.reshape({batch_size, token_count, head_size});
    const auto actual_output = readOutput(
        output, {batch_size, token_count, head_size}, queue);

    // Also exercise the public entry point used by attention_test.cpp.
    auto attention_output = flib::operations::scaled_dot_product_attention(
        query, key, value, head_count, queue);
    const auto actual_attention = readOutput(
        attention_output, {batch_size, token_count, head_size}, queue);

    std::cout << std::setprecision(10);
    saveWeights(actual_weights);
    const bool weights_passed = compare("Attention weights", actual_weights,
                                        expected_weights, token_count, 1.0e-4);
    const bool output_passed = compare("Pipeline output", actual_output,
                                       expected_output, head_size, 1.0e-3);
    const bool attention_passed = compare("Attention API output", actual_attention,
                                          expected_output, head_size, 1.0e-3);
    const bool passed = weights_passed && output_passed && attention_passed;
    std::cout << '\n' << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
