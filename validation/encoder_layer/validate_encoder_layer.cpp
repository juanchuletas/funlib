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
constexpr std::size_t B = 1;
constexpr std::size_t N = 8;
constexpr std::size_t D = 64;
constexpr std::size_t heads = 8;
constexpr std::size_t head_dim = D / heads;
constexpr std::size_t hidden = D * 4;
constexpr float epsilon = 1.0e-5f;
constexpr double tolerance = 1.0e-3;
using Tensor = flib::Tensor<float>;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Reference files require IEEE 754 float32");

// Files contain raw row-major float32 values in native byte order.
std::vector<float> load(const std::string &path, std::size_t count) {
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

Tensor deviceTensor(const std::vector<std::size_t> &shape,
                    const std::vector<float> &values, sycl::queue queue) {
  Tensor tensor(shape, queue);
  if (tensor.getSize() != values.size()) {
    throw std::runtime_error("Weight or input shape mismatch");
  }
  tensor.copy_from(values.data(), queue).wait_and_throw();
  return tensor;
}

struct Linear {
  Tensor weight;
  Tensor bias;
};

Linear makeLinear(const std::vector<float> &weight,
                  const std::vector<float> &bias, std::size_t inputs,
                  std::size_t outputs, sycl::queue queue,
                  std::size_t weight_offset = 0, std::size_t bias_offset = 0) {
  // PyTorch stores [out,in]; funlib GEMM expects [in,out].
  std::vector<float> transposed(inputs * outputs);
  for (std::size_t out = 0; out < outputs; ++out) {
    for (std::size_t in = 0; in < inputs; ++in) {
      transposed[in * outputs + out] = weight.at(weight_offset + out * inputs + in);
    }
  }
  // funlib add requires equal shapes, so expand each bias over the tokens.
  std::vector<float> expanded_bias(B * N * outputs);
  for (std::size_t i = 0; i < expanded_bias.size(); ++i) {
    expanded_bias[i] = bias.at(bias_offset + i % outputs);
  }
  return {deviceTensor({inputs, outputs}, transposed, queue),
          deviceTensor({B, N, outputs}, expanded_bias, queue)};
}

Tensor linear(const Tensor &input, const Linear &parameters, sycl::queue queue) {
  auto projected = flib::tensor_operations::gemm(input, parameters.weight, queue);
  return flib::operations::add(projected, parameters.bias, queue);
}

void printRow(const char *label, const std::vector<float> &values,
              std::size_t row) {
  std::cout << label << " [";
  for (std::size_t column = 0; column < D; ++column) {
    if (column != 0) {
      std::cout << ", ";
    }
    std::cout << values[row * D + column];
  }
  std::cout << "]\n";
}
} // namespace

int main(int argc, char **argv) {
  if (argc > 2) {
    std::cerr << "Usage: " << argv[0] << " [reference_directory]\n";
    return 1;
  }
  try {
    const std::string directory = argc == 2 ? argv[1] : ".";
    const auto input_values = load(directory + "/ref_encoder_layer_input.bin", B * N * D);
    const auto expected = load(directory + "/ref_encoder_layer_output.bin", B * N * D);

    flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                     flib::vendor::NVIDIA, flib::backend::CUDA,
                                     true);
    sycl::queue queue = flib::sycl_handler::get_queue("cuda");
    flib::sycl_handler::get_device_info("cuda");

    // Load original PyTorch layouts on the host; makeLinear handles the
    // model-specific transpose and packed Q/K/V slicing before GPU upload.
    const flib::WeightLoader weights(directory + "/weights.json");
    const auto weight = [&](const std::string &name,
                            const std::vector<std::size_t> &shape) {
      const auto &tensor = weights.at(name);
      if (tensor.getShape() != shape) {
        throw std::runtime_error("Unexpected shape for weight: " + name);
      }
      auto values = tensor.to_host(queue);
      for (float value : values) {
        if (!std::isfinite(value)) {
          throw std::runtime_error("Non-finite value in weight: " + name);
        }
      }
      return values;
    };
    std::cout << "Loaded " << weights.size() << " tensors from weights.json\n";

    // Packed PyTorch attention projections are ordered Q, K, V.
    const auto packed_weight = weight("self_attn.in_proj_weight", {3 * D, D});
    const auto packed_bias = weight("self_attn.in_proj_bias", {3 * D});
    auto q_projection = makeLinear(packed_weight, packed_bias, D, D, queue);
    auto k_projection = makeLinear(packed_weight, packed_bias, D, D, queue, D * D, D);
    auto v_projection = makeLinear(packed_weight, packed_bias, D, D, queue, 2 * D * D, 2 * D);
    auto out_projection = makeLinear(
        weight("self_attn.out_proj.weight", {D, D}),
        weight("self_attn.out_proj.bias", {D}), D, D, queue);
    auto expand = makeLinear(weight("linear1.weight", {hidden, D}),
                             weight("linear1.bias", {hidden}), D, hidden, queue);
    auto reduce = makeLinear(weight("linear2.weight", {D, hidden}),
                             weight("linear2.bias", {D}), hidden, D, queue);
    auto gamma1 = deviceTensor({D}, weight("norm1.weight", {D}), queue);
    auto beta1 = deviceTensor({D}, weight("norm1.bias", {D}), queue);
    auto gamma2 = deviceTensor({D}, weight("norm2.weight", {D}), queue);
    auto beta2 = deviceTensor({D}, weight("norm2.bias", {D}), queue);
    auto input = deviceTensor({B, N, D}, input_values, queue);

    // norm_first=True, evaluation mode: dropout is disabled, with no mask.
    auto normalized1 = flib::operations::layer_norm(input, gamma1, beta1, epsilon, queue);
    auto query = linear(normalized1, q_projection, queue);
    auto key = linear(normalized1, k_projection, queue);
    auto value = linear(normalized1, v_projection, queue);
    // Match transformer_block_test.cpp's [B,N,H,head_dim] layout.
    query.reshape({B, N, heads, head_dim});
    key.reshape({B, N, heads, head_dim});
    value.reshape({B, N, heads, head_dim});
    auto attention = flib::operations::scaled_dot_product_attention(
        query, key, value, heads, queue);
    auto projected_attention = linear(attention, out_projection, queue);
    auto residual1 = flib::operations::add(input, projected_attention, queue);
    auto normalized2 = flib::operations::layer_norm(residual1, gamma2, beta2, epsilon, queue);
    auto expanded = linear(normalized2, expand, queue);
    auto activated = flib::operations::gelu(expanded, queue);
    auto feed_forward = linear(activated, reduce, queue);
    auto output = flib::operations::add(residual1, feed_forward, queue);
    queue.wait_and_throw();
    if (output.getShape() != std::vector<std::size_t>{B, N, D}) {
      throw std::runtime_error("Encoder layer returned an unexpected shape");
    }
    const auto actual = output.to_host(queue);
    queue.wait_and_throw();
    if (actual.size() != expected.size()) {
      throw std::runtime_error("Encoder layer returned an unexpected size");
    }

    std::cout << std::setprecision(10);
    for (std::size_t row = 0; row < B * N; ++row) {
      std::cout << "\nOutput row " << row << '\n';
      printRow("pytorch:", expected, row);
      printRow("funlib: ", actual, row);
    }
    double max_difference = 0.0;
    std::size_t worst_index = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const double difference = std::isfinite(actual[i])
          ? std::abs(static_cast<double>(actual[i]) - expected[i])
          : std::numeric_limits<double>::infinity();
      if (difference > max_difference) {
        max_difference = difference;
        worst_index = i;
      }
    }
    const bool passed = max_difference < tolerance;
    std::cout << "\nMax absolute difference: " << max_difference << '\n'
              << "Worst index: " << worst_index
              << " (pytorch: " << expected[worst_index]
              << ", funlib: " << actual[worst_index] << ")\n"
              << "Threshold: " << tolerance << '\n'
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
