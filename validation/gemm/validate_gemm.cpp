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
constexpr std::size_t M = 8;
constexpr std::size_t K = 64;
constexpr std::size_t N = 32;
constexpr double tolerance = 1.0e-3;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Reference files require IEEE 754 float32");

// Raw row-major NumPy float32 values in native byte order, without a header.
std::vector<float> loadReference(const std::string &path, std::size_t count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Cannot open " + path);
  }
  const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
  if (file.tellg() != std::streampos(bytes)) {
    throw std::runtime_error(path + " must contain exactly " +
                             std::to_string(count) + " float32 values (" +
                             std::to_string(bytes) + " bytes)");
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

void printRow(const char *label, const std::vector<float> &values,
              std::size_t row) {
  std::cout << label << " [";
  for (std::size_t column = 0; column < N; ++column) {
    if (column != 0) {
      std::cout << ", ";
    }
    std::cout << values[row * N + column];
  }
  std::cout << "]\n";
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 1 && argc != 4) {
    std::cerr << "Usage: " << argv[0] << " [A.bin B.bin C.bin]\n";
    return 1;
  }
  try {
    const auto a_values =
        loadReference(argc == 4 ? argv[1] : "ref_gemm_A.bin", M * K);
    const auto b_values =
        loadReference(argc == 4 ? argv[2] : "ref_gemm_B.bin", K * N);
    const auto expected =
        loadReference(argc == 4 ? argv[3] : "ref_gemm_C.bin", M * N);

    flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                       flib::vendor::NVIDIA,
                                       flib::backend::CUDA, true);
    sycl::queue queue = flib::sycl_handler::get_queue("cuda");
    flib::sycl_handler::get_device_info("cuda");
    flib::Tensor<float> A({M, K}, queue);
    flib::Tensor<float> B({K, N}, queue);
    A.copy_from(a_values.data(), queue).wait_and_throw();
    B.copy_from(b_values.data(), queue).wait_and_throw();
    auto C = flib::tensor_operations::gemm(A, B, queue);
    queue.wait_and_throw();
    const auto actual = C.to_host(queue);
    queue.wait_and_throw();
    if (C.getShape() != std::vector<std::size_t>{M, N} ||
        actual.size() != expected.size()) {
      throw std::runtime_error("GEMM returned an unexpected shape or size");
    }

    std::cout << std::setprecision(10);
    for (std::size_t row = 0; row < M; ++row) {
      std::cout << "\nC row " << row << '\n';
      printRow("pytorch:", expected, row);
      printRow("funlib: ", actual, row);
    }

    double max_difference = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (!std::isfinite(actual[i])) {
        max_difference = std::numeric_limits<double>::infinity();
        break;
      }
      max_difference =
          std::max(max_difference,
                   std::abs(static_cast<double>(actual[i]) - expected[i]));
    }
    const bool passed = max_difference < tolerance;
    std::cout << "\nMax absolute difference: " << max_difference << '\n'
              << "Threshold: " << tolerance << '\n'
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
