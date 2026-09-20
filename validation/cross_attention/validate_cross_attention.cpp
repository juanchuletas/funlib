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
constexpr std::size_t query_tokens = 8;
constexpr std::size_t kv_tokens = 16;
constexpr std::size_t head_count = 1;
constexpr std::size_t head_size = 64;
constexpr double tolerance = 1.0e-3;
using Tensor = flib::Tensor<float>;
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

Tensor deviceTensor(const std::vector<float> &values, std::size_t tokens,
                    sycl::queue queue) {
  Tensor tensor({batch_size, tokens, head_size}, queue);
  if (tensor.getSize() != values.size()) {
    throw std::runtime_error("Input size mismatch");
  }
  tensor.copy_from(values.data(), queue).wait_and_throw();
  tensor.reshape({batch_size, tokens, head_count, head_size});
  return tensor;
}

std::vector<float> readOutput(const Tensor &tensor,
                              const std::vector<std::size_t> &shape,
                              sycl::queue queue) {
  queue.wait_and_throw();
  if (tensor.getShape() != shape) {
    throw std::runtime_error("Unexpected cross-attention tensor shape");
  }
  auto values = tensor.to_host(queue);
  queue.wait_and_throw();
  return values;
}

bool compare(const char *label, const std::vector<float> &actual,
             const std::vector<float> &expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("Cross-attention output size mismatch");
  }
  double max_difference = 0.0;
  std::cout << '\n' << label << '\n';
  for (std::size_t row = 0; row < query_tokens; ++row) {
    std::cout << "Row " << row << "\npytorch: [";
    for (std::size_t col = 0; col < head_size; ++col) {
      std::cout << (col ? ", " : "") << expected[row * head_size + col];
    }
    std::cout << "]\nfunlib:  [";
    for (std::size_t col = 0; col < head_size; ++col) {
      const auto i = row * head_size + col;
      std::cout << (col ? ", " : "") << actual[i];
      const double difference =
          std::isfinite(actual[i])
              ? std::abs(static_cast<double>(actual[i]) - expected[i])
              : std::numeric_limits<double>::infinity();
      max_difference = std::max(max_difference, difference);
    }
    std::cout << "]\n";
  }
  const bool passed = max_difference < tolerance;
  std::cout << label << " max absolute difference: " << max_difference
            << " (threshold " << tolerance << ") " << (passed ? "PASS" : "FAIL")
            << '\n';
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 1 && argc != 5) {
    std::cerr << "Usage: " << argv[0] << " [Q.bin K.bin V.bin out.bin]\n";
    return 1;
  }
  try {
    const auto q_values =
        loadReference(argc == 5 ? argv[1] : "ref_cross_attn_Q.bin",
                      batch_size * query_tokens * head_size);
    const auto k_values =
        loadReference(argc == 5 ? argv[2] : "ref_cross_attn_K.bin",
                      batch_size * kv_tokens * head_size);
    const auto v_values =
        loadReference(argc == 5 ? argv[3] : "ref_cross_attn_V.bin",
                      batch_size * kv_tokens * head_size);
    const auto expected =
        loadReference(argc == 5 ? argv[4] : "ref_cross_attn_out.bin",
                      batch_size * query_tokens * head_size);

    flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                       flib::vendor::NVIDIA,
                                       flib::backend::CUDA, true);
    sycl::queue queue = flib::sycl_handler::get_queue("cuda");
    flib::sycl_handler::get_device_info("cuda");
    auto query = deviceTensor(q_values, query_tokens, queue);
    auto key = deviceTensor(k_values, kv_tokens, queue);
    auto value = deviceTensor(v_values, kv_tokens, queue);

    auto query_heads = flib::operations::split_heads(query, queue);
    auto key_heads = flib::operations::split_heads(key, queue);
    auto value_heads = flib::operations::split_heads(value, queue);
    auto scores = flib::tensor_operations::gemm_batched(query_heads, key_heads,
                                                        queue, false, true);
    const std::vector<std::size_t> score_shape{batch_size, head_count,
                                               query_tokens, kv_tokens};
    if (scores.getShape() != score_shape) {
      throw std::runtime_error("Scores must have shape [1,1,8,16]");
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    auto weights = flib::operations::scaled_softmax(scores, scale, queue);
    const auto probabilities = readOutput(weights, score_shape, queue);
    bool rows_passed = true;
    double max_row_error = 0.0;
    for (std::size_t row = 0; row < query_tokens; ++row) {
      double sum = 0.0;
      for (std::size_t col = 0; col < kv_tokens; ++col) {
        sum += probabilities.at(row * kv_tokens + col);
      }
      const double error = std::isfinite(sum)
                               ? std::abs(sum - 1.0)
                               : std::numeric_limits<double>::infinity();
      max_row_error = std::max(max_row_error, error);
      rows_passed = rows_passed && error <= 1.0e-6;
    }
    auto context =
        flib::tensor_operations::gemm_batched(weights, value_heads, queue);
    auto output = flib::operations::join_heads(context, queue);
    if (output.getShape() != std::vector<std::size_t>{batch_size, query_tokens,
                                                      head_count, head_size}) {
      throw std::runtime_error(
          "Joined output must retain the eight query tokens");
    }
    output.reshape({batch_size, query_tokens, head_size});
    const std::vector<std::size_t> output_shape{batch_size, query_tokens,
                                                head_size};
    const auto actual = readOutput(output, output_shape, queue);

    // Validate the public API too, including its returned sequence length.
    auto api_output = flib::operations::scaled_dot_product_attention(
        query, key, value, head_count, queue);
    const auto api_actual = readOutput(api_output, output_shape, queue);
    std::cout << std::setprecision(10)
              << "Output shape: [1,8,64] (matches Q)\n";
    const bool pipeline_passed = compare("Pipeline output", actual, expected);
    const bool api_passed =
        compare("Attention API output", api_actual, expected);
    std::cout << "Max weights row sum error: " << max_row_error << " ("
              << (rows_passed ? "PASS" : "FAIL") << ")\n";
    const bool passed = pipeline_passed && api_passed && rows_passed;
    std::cout << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
