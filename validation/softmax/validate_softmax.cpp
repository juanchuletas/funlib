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
constexpr std::size_t row_count = 8;
constexpr std::size_t row_size = 8;
constexpr std::size_t element_count = row_count * row_size;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Reference files require IEEE 754 float32");

// Raw NumPy float32 values in native byte order, without a header.
std::vector<float> loadReference(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Cannot open " + path);
  }
  constexpr std::streamsize bytes = element_count * sizeof(float);
  if (file.tellg() != std::streampos(bytes)) {
    throw std::runtime_error(
        path + " must contain exactly 64 float32 values (256 bytes)");
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

void printRow(const char *label, const std::vector<float> &values,
              std::size_t row) {
  double sum = 0.0;
  std::cout << label << " [";
  for (std::size_t column = 0; column < row_size; ++column) {
    const float value = values[row * row_size + column];
    if (column != 0) {
      std::cout << ", ";
    }
    std::cout << value;
    sum += static_cast<double>(value);
  }
  std::cout << "]  sum=" << sum << '\n';
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 1 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << " [input.bin output.bin]\n";
    return 1;
  }
  try {
    const auto values =
        loadReference(argc == 3 ? argv[1] : "ref_softmax_input.bin");
    const auto expected =
        loadReference(argc == 3 ? argv[2] : "ref_softmax_output.bin");

    flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                       flib::vendor::NVIDIA,
                                       flib::backend::CUDA, true);
    sycl::queue queue = flib::sycl_handler::get_queue("cuda");
    flib::sycl_handler::get_device_info("cuda");
    flib::Tensor<float> input({1, row_count, row_size}, queue);
    input.copy_from(values.data(), queue).wait_and_throw();
    // Softmax normalizes along the final dimension.
    auto output = flib::operations::softmax(input, queue);
    queue.wait_and_throw();
    const auto actual = output.to_host(queue);
    queue.wait_and_throw();
    if (output.getShape() != input.getShape() ||
        actual.size() != expected.size()) {
      throw std::runtime_error("Softmax returned an unexpected shape or size");
    }

    double max_difference = 0.0;
    double max_row_sum_error = 0.0;
    bool rows_passed = true;
    std::cout << std::setprecision(10);
    for (std::size_t row = 0; row < row_count; ++row) {
      std::cout << "\nRow " << row << '\n';
      printRow("pytorch:", expected, row);
      printRow("funlib: ", actual, row);
      double sum = 0.0;
      bool finite_row = true;
      for (std::size_t column = 0; column < row_size; ++column) {
        const std::size_t index = row * row_size + column;
        if (!std::isfinite(actual[index])) {
          finite_row = false;
          max_difference = std::numeric_limits<double>::infinity();
          continue;
        }
        sum += static_cast<double>(actual[index]);
        max_difference = std::max(
            max_difference,
            std::abs(static_cast<double>(actual[index]) - expected[index]));
      }
      const double error = finite_row ? std::abs(sum - 1.0)
                                      : std::numeric_limits<double>::infinity();
      max_row_sum_error = std::max(max_row_sum_error, error);
      if (error > 1.0e-6) {
        rows_passed = false;
        if (finite_row) {
          std::cout << "Row " << row << " sum: " << sum << " (FAIL)\n";
        } else {
          std::cout << "Row " << row << " contains non-finite output (FAIL)\n";
        }
      }
    }
    const bool reference_passed = max_difference < 1.0e-4;
    const bool passed = reference_passed && rows_passed;
    std::cout << "\nMax absolute difference: " << max_difference << " ("
              << (reference_passed ? "PASS" : "FAIL") << ")\n"
              << "Max row sum error: " << max_row_sum_error << " ("
              << (rows_passed ? "PASS" : "FAIL") << ")\n"
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
