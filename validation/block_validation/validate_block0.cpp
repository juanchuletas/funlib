#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <funlib/funlib.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef BLOCK0_REFERENCE_DIR
#define BLOCK0_REFERENCE_DIR "."
#endif

namespace block_validation {
using Tensor = flib::Tensor<float>;
constexpr std::size_t tokens = 3072, image_tokens = 196;
constexpr std::size_t width = 1024, image_width = 768;
constexpr std::size_t heads = 16, head_dim = 64, hidden = 4096;
constexpr double tolerance = 1.0e-2;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "References require IEEE float32");

class Block {
  std::filesystem::path directory;
  sycl::queue queue;
  static constexpr float epsilon = 1.0e-5f;

  std::vector<float> load(const std::string &file, std::size_t count) const {
    const auto path = directory / file;
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
    if (!stream || stream.tellg() != std::streampos(bytes)) {
      throw std::runtime_error("Missing or incorrectly sized file: " +
                               path.string());
    }
    stream.seekg(0);
    std::vector<float> values(count);
    if (!stream.read(reinterpret_cast<char *>(values.data()), bytes)) {
      throw std::runtime_error("Cannot read " + path.string());
    }
    for (float value : values) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("Non-finite value in " + path.string());
      }
    }
    return values;
  }

  std::vector<float> weight(const std::string &name, std::size_t count) const {
    return load("backbone_transformer_blocks_0_" + name + ".bin", count);
  }

  Tensor upload(const std::vector<float> &values,
                const std::vector<std::size_t> &shape) const {
    Tensor result(shape, queue);
    if (result.getSize() != values.size()) {
      throw std::runtime_error("Tensor upload shape mismatch");
    }
    result.copy_from(values.data(), queue).wait_and_throw();
    return result;
  }

  Tensor norm(const Tensor &input, const std::string &name) const {
    auto gamma = upload(weight(name + "_weight", width), {width});
    auto beta = upload(weight(name + "_bias", width), {width});
    return flib::operations::layer_norm(input, gamma, beta, epsilon, queue);
  }

  Tensor linear(const Tensor &input, const std::string &name,
                std::size_t output_width, bool has_bias) const {
    const auto input_width = input.getShape().back();
    auto raw = weight(name + "_weight", output_width * input_width);
    std::vector<float> transposed(raw.size());
    // PyTorch [out,in] -> funlib [in,out].
    for (std::size_t out = 0; out < output_width; ++out) {
      for (std::size_t in = 0; in < input_width; ++in) {
        transposed[in * output_width + out] = raw[out * input_width + in];
      }
    }
    auto matrix = upload(transposed, {input_width, output_width});
    auto output = flib::tensor_operations::gemm(input, matrix, queue);
    if (has_bias) {
      auto bias = upload(weight(name + "_bias", output_width), {output_width});
      output = flib::operations::add_bias(output, bias, queue);
    }
    return output;
  }

  Tensor attention(const Tensor &query_input, const Tensor &kv_input,
                   const std::string &name) const {
    auto q = linear(query_input, name + "_to_q", width, false);
    auto k = linear(kv_input, name + "_to_k", width, false);
    auto v = linear(kv_input, name + "_to_v", width, false);
    q.reshape({1, query_input.getShape()[1], heads, head_dim});
    k.reshape({1, kv_input.getShape()[1], heads, head_dim});
    v.reshape({1, kv_input.getShape()[1], heads, head_dim});
    auto context =
        flib::operations::scaled_dot_product_attention(q, k, v, heads, queue);
    return linear(context, name + "_to_out_0", width, true);
  }

  Tensor ffn(const Tensor &input) const {
    auto projected = linear(input, "ff_net_0_proj", 2 * hidden, true);
    auto activated = flib::operations::geglu(projected, queue);
    return linear(activated, "ff_net_2", width, true);
  }

public:
  Block(std::filesystem::path path, sycl::queue q)
      : directory(std::move(path)), queue(q) {}

  bool validate() const {
    auto input = upload(load("ref_block0_input.bin", tokens * width),
                        {1, tokens, width});
    auto image =
        upload(load("ref_block0_img_tokens.bin", image_tokens * image_width),
               {1, image_tokens, image_width});
    const auto expected = load("ref_block0_output.bin", tokens * width);
    Tensor residual;
    {
      std::cout << "Self-attention (3072 query/key tokens)" << std::endl;
      auto normalized = norm(input, "norm1");
      auto attended = attention(normalized, normalized, "attn1");
      residual = flib::operations::add(input, attended, queue);
    }
    {
      std::cout << "Cross-attention (3072 queries, 196 image tokens)"
                << std::endl;
      auto normalized = norm(residual, "norm2");
      auto attended = attention(normalized, image, "attn2");
      residual = flib::operations::add(residual, attended, queue);
    }
    {
      std::cout << "GEGLU feed-forward" << std::endl;
      auto normalized = norm(residual, "norm3");
      auto feed_forward = ffn(normalized);
      residual = flib::operations::add(residual, feed_forward, queue);
    }
    if (residual.getShape() != std::vector<std::size_t>{1, tokens, width}) {
      throw std::runtime_error("Output shape must be [1,3072,1024]");
    }
    const auto actual = residual.to_host(queue);
    if (actual.size() != expected.size()) {
      throw std::runtime_error("Output size mismatch");
    }
    double maximum = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const double diff =
          std::isfinite(actual[i])
              ? std::abs(static_cast<double>(actual[i]) - expected[i])
              : std::numeric_limits<double>::infinity();
      if (diff > maximum) {
        maximum = diff;
        worst = i;
      }
    }
    std::cout << std::setprecision(10) << "pytorch: [";
    for (std::size_t i = 0; i < 4; ++i)
      std::cout << (i ? ", " : "") << expected[i];
    std::cout << "]\nfunlib:  [";
    for (std::size_t i = 0; i < 4; ++i)
      std::cout << (i ? ", " : "") << actual[i];
    std::cout << "]\nMax absolute difference: " << maximum
              << "\nWorst index: [0," << worst / width << ',' << worst % width
              << ']' << "\npytorch: " << expected[worst]
              << ", funlib: " << actual[worst] << "\nThreshold: " << tolerance
              << '\n';
    const bool passed = maximum < tolerance;
    std::cout << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
  }
};
} // namespace block_validation

int main(int argc, char **argv) {
  if (argc > 2) {
    std::cerr << "Usage: " << argv[0] << " [reference_directory]\n";
    return 1;
  }
  try {
    flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                       flib::vendor::NVIDIA,
                                       flib::backend::CUDA, true);
    auto queue = flib::sycl_handler::get_queue("cuda");
    flib::sycl_handler::get_device_info("cuda");
    block_validation::Block block(argc == 2 ? argv[1] : BLOCK0_REFERENCE_DIR,
                                  queue);
    return block.validate() ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
