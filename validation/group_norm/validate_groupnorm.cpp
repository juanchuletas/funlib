#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

constexpr std::size_t batch_count = 2;
constexpr std::size_t channel_count = 4;
constexpr std::size_t position_count = 3;
constexpr int group_count = 2;
constexpr float epsilon = 1.0e-5f;
constexpr double tolerance = 1.0e-4;

std::vector<float> groupNormReference(const std::vector<float> &input,
                                      const std::vector<float> &weight,
                                      const std::vector<float> &bias) {
  std::vector<float> output(input.size());
  std::size_t groups = static_cast<std::size_t>(group_count);
  std::size_t channels_per_group = channel_count / groups;
  std::size_t values_per_group = channels_per_group * position_count;

  for (std::size_t batch = 0; batch < batch_count; batch++) {
    for (std::size_t group = 0; group < groups; group++) {
      std::size_t offset =
          batch * channel_count * position_count + group * values_per_group;
      double sum = 0.0;
      for (std::size_t index = 0; index < values_per_group; index++) {
        sum += input[offset + index];
      }
      double mean = sum / static_cast<double>(values_per_group);

      double variance_sum = 0.0;
      for (std::size_t index = 0; index < values_per_group; index++) {
        double difference = input[offset + index] - mean;
        variance_sum += difference * difference;
      }
      double inverse_standard_deviation =
          1.0 / std::sqrt(variance_sum / static_cast<double>(values_per_group) +
                          epsilon);

      for (std::size_t index = 0; index < values_per_group; index++) {
        std::size_t channel =
            group * channels_per_group + index / position_count;
        double normalized =
            (input[offset + index] - mean) * inverse_standard_deviation;
        output[offset + index] =
            static_cast<float>(normalized * weight[channel] + bias[channel]);
      }
    }
  }
  return output;
}

int main() {
  try {
    flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                       flib::vendor::NVIDIA,
                                       flib::backend::CUDA, true);
    sycl::queue queue = flib::sycl_handler::get_queue("cuda");
    flib::sycl_handler::get_device_info("cuda");

    const std::vector<float> input_values{
        -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f,   2.0f,  3.0f,
        4.0f,  5.0f,  6.0f,  7.0f,  1.5f, -2.5f,  3.5f,  -4.5f,
        5.5f,  -6.5f, 7.5f,  -8.5f, 9.5f, -10.5f, 11.5f, -12.5f};
    const std::vector<float> weight_values{1.0f, 0.5f, -1.0f, 2.0f};
    const std::vector<float> bias_values{0.0f, 1.0f, -0.5f, 0.25f};
    const std::vector<float> expected =
        groupNormReference(input_values, weight_values, bias_values);

    flib::ftensor input({batch_count, channel_count, position_count}, queue);
    flib::ftensor weight({channel_count}, queue);
    flib::ftensor bias({channel_count}, queue);
    input.copy_from(input_values.data(), queue).wait_and_throw();
    weight.copy_from(weight_values.data(), queue).wait_and_throw();
    bias.copy_from(bias_values.data(), queue).wait_and_throw();

    flib::ftensor output = flib::operations::group_norm(
        input, weight, bias, group_count, epsilon, queue);
    const std::vector<float> actual = output.to_host(queue);
    if (output.getShape() != input.getShape() ||
        actual.size() != expected.size()) {
      throw std::runtime_error(
          "GroupNorm returned an unexpected shape or size");
    }

    double max_difference = 0.0;
    for (std::size_t index = 0; index < actual.size(); index++) {
      if (!std::isfinite(actual[index])) {
        max_difference = std::numeric_limits<double>::infinity();
        break;
      }
      max_difference =
          std::max(max_difference, std::abs(static_cast<double>(actual[index]) -
                                            expected[index]));
    }

    std::cout << std::setprecision(8) << "reference: ";
    for (std::size_t index = 0; index < 4; index++) {
      std::cout << expected[index] << ' ';
    }
    std::cout << "\nfunlib:   ";
    for (std::size_t index = 0; index < 4; index++) {
      std::cout << actual[index] << ' ';
    }
    bool passed = max_difference < tolerance;
    std::cout << "\nMax absolute difference: " << max_difference << '\n'
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
