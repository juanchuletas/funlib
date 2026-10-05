#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

constexpr float tolerance = 1.0e-4f;

std::vector<float> convolution2dReference(
    const std::vector<float> &input, const std::vector<float> &weight,
    const std::vector<float> &bias, std::size_t batch_count,
    std::size_t input_channels, std::size_t input_height,
    std::size_t input_width, std::size_t output_channels,
    std::size_t kernel_height, std::size_t kernel_width, std::size_t stride) {
  std::size_t output_height =
      (input_height - kernel_height) / stride + 1;
  std::size_t output_width = (input_width - kernel_width) / stride + 1;
  std::vector<float> output(batch_count * output_channels * output_height *
                            output_width);

  for (std::size_t batch = 0; batch < batch_count; batch++) {
    for (std::size_t output_channel = 0;
         output_channel < output_channels; output_channel++) {
      for (std::size_t output_y = 0; output_y < output_height; output_y++) {
        for (std::size_t output_x = 0; output_x < output_width; output_x++) {
          float sum = bias[output_channel];
          for (std::size_t input_channel = 0;
               input_channel < input_channels; input_channel++) {
            for (std::size_t kernel_y = 0; kernel_y < kernel_height;
                 kernel_y++) {
              for (std::size_t kernel_x = 0; kernel_x < kernel_width;
                   kernel_x++) {
                std::size_t input_y = output_y * stride + kernel_y;
                std::size_t input_x = output_x * stride + kernel_x;
                std::size_t input_index =
                    ((batch * input_channels + input_channel) * input_height +
                     input_y) *
                        input_width +
                    input_x;
                std::size_t weight_index =
                    ((output_channel * input_channels + input_channel) *
                         kernel_height +
                     kernel_y) *
                        kernel_width +
                    kernel_x;
                sum += input[input_index] * weight[weight_index];
              }
            }
          }
          std::size_t output_index =
              ((batch * output_channels + output_channel) * output_height +
               output_y) *
                  output_width +
              output_x;
          output[output_index] = sum;
        }
      }
    }
  }
  return output;
}

std::vector<float> convolution2dTransposeReference(
    const std::vector<float> &input, const std::vector<float> &weight,
    const std::vector<float> &bias, std::size_t batch_count,
    std::size_t input_channels, std::size_t input_height,
    std::size_t input_width, std::size_t output_channels,
    std::size_t kernel_height, std::size_t kernel_width, std::size_t stride,
    std::size_t padding, std::size_t output_padding, std::size_t dilation) {
  std::size_t output_height = (input_height - 1) * stride - 2 * padding +
                              dilation * (kernel_height - 1) +
                              output_padding + 1;
  std::size_t output_width = (input_width - 1) * stride - 2 * padding +
                             dilation * (kernel_width - 1) + output_padding +
                             1;
  std::vector<float> output(batch_count * output_channels * output_height *
                            output_width);

  // Start every output position with its channel bias.
  for (std::size_t batch = 0; batch < batch_count; batch++) {
    for (std::size_t output_channel = 0;
         output_channel < output_channels; output_channel++) {
      for (std::size_t output_y = 0; output_y < output_height; output_y++) {
        for (std::size_t output_x = 0; output_x < output_width; output_x++) {
          std::size_t output_index =
              ((batch * output_channels + output_channel) * output_height +
               output_y) *
                  output_width +
              output_x;
          output[output_index] = bias[output_channel];
        }
      }
    }
  }

  // Scatter each input value through its transposed convolution kernel.
  for (std::size_t batch = 0; batch < batch_count; batch++) {
    for (std::size_t input_channel = 0; input_channel < input_channels;
         input_channel++) {
      for (std::size_t input_y = 0; input_y < input_height; input_y++) {
        for (std::size_t input_x = 0; input_x < input_width; input_x++) {
          std::size_t input_index =
              ((batch * input_channels + input_channel) * input_height +
               input_y) *
                  input_width +
              input_x;
          for (std::size_t output_channel = 0;
               output_channel < output_channels; output_channel++) {
            for (std::size_t kernel_y = 0; kernel_y < kernel_height;
                 kernel_y++) {
              for (std::size_t kernel_x = 0; kernel_x < kernel_width;
                   kernel_x++) {
                std::ptrdiff_t output_y =
                    static_cast<std::ptrdiff_t>(input_y * stride) -
                    static_cast<std::ptrdiff_t>(padding) +
                    static_cast<std::ptrdiff_t>(kernel_y * dilation);
                std::ptrdiff_t output_x =
                    static_cast<std::ptrdiff_t>(input_x * stride) -
                    static_cast<std::ptrdiff_t>(padding) +
                    static_cast<std::ptrdiff_t>(kernel_x * dilation);
                if (output_y < 0 || output_x < 0 ||
                    output_y >= static_cast<std::ptrdiff_t>(output_height) ||
                    output_x >= static_cast<std::ptrdiff_t>(output_width)) {
                  continue;
                }

                std::size_t output_index =
                    ((batch * output_channels + output_channel) *
                         output_height +
                     static_cast<std::size_t>(output_y)) *
                        output_width +
                    static_cast<std::size_t>(output_x);
                std::size_t weight_index =
                    ((input_channel * output_channels + output_channel) *
                         kernel_height +
                     kernel_y) *
                        kernel_width +
                    kernel_x;
                output[output_index] +=
                    input[input_index] * weight[weight_index];
              }
            }
          }
        }
      }
    }
  }
  return output;
}

bool compareValues(const std::vector<float> &actual,
                   const std::vector<float> &expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0; index < actual.size(); index++) {
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) > tolerance) {
      std::cerr << "Mismatch at index " << index << ": expected "
                << expected[index] << ", received " << actual[index]
                << std::endl;
      return false;
    }
  }
  return true;
}

bool testConvolution2d(sycl::queue queue) {
  constexpr std::size_t batch_count = 1;
  constexpr std::size_t input_channels = 2;
  constexpr std::size_t input_height = 4;
  constexpr std::size_t input_width = 4;
  constexpr std::size_t output_channels = 3;
  constexpr std::size_t kernel_height = 2;
  constexpr std::size_t kernel_width = 2;
  constexpr std::size_t stride = 1;
  const std::vector<float> input_values{
      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
      -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15,
      -16};
  const std::vector<float> weight_values{
      1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
      1, 0, 0, 1, 2, 0, 0, 2, -1, 1, 1, -1};
  const std::vector<float> bias_values{0.0f, 1.0f, -2.0f};
  const std::vector<float> expected = convolution2dReference(
      input_values, weight_values, bias_values, batch_count, input_channels,
      input_height, input_width, output_channels, kernel_height, kernel_width,
      stride);

  // These tensors store the input, learned kernels, and learned bias on the GPU.
  flib::ftensor input(
      {batch_count, input_channels, input_height, input_width}, queue);
  flib::ftensor weight(
      {output_channels, input_channels, kernel_height, kernel_width}, queue);
  flib::ftensor bias({output_channels}, queue);
  input.copy_from(input_values.data(), queue).wait_and_throw();
  weight.copy_from(weight_values.data(), queue).wait_and_throw();
  bias.copy_from(bias_values.data(), queue).wait_and_throw();

  flib::ftensor output =
      flib::operations::convolution2d(input, weight, bias, stride, queue);
  std::vector<float> actual = output.to_host(queue);
  std::vector<std::size_t> expected_shape{1, 3, 3, 3};
  bool passed = output.getShape() == expected_shape &&
                compareValues(actual, expected);
  std::cout << (passed ? "Passed convolution2d" : "Failed convolution2d")
            << std::endl;
  return passed;
}

bool testConvolution2dTranspose(sycl::queue queue) {
  constexpr std::size_t batch_count = 1;
  constexpr std::size_t input_channels = 2;
  constexpr std::size_t input_height = 2;
  constexpr std::size_t input_width = 2;
  constexpr std::size_t output_channels = 3;
  constexpr std::size_t kernel_height = 2;
  constexpr std::size_t kernel_width = 2;
  constexpr std::size_t stride = 2;
  constexpr std::size_t padding = 0;
  constexpr std::size_t output_padding = 0;
  constexpr std::size_t dilation = 1;
  const std::vector<float> input_values{1, 2, 3, 4, -1, -2, -3, -4};
  const std::vector<float> weight_values{
      1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
      1, 0, 0, 1, 2, 0, 0, 2, -1, 1, 1, -1};
  const std::vector<float> bias_values{0.0f, 1.0f, -2.0f};
  const std::vector<float> expected = convolution2dTransposeReference(
      input_values, weight_values, bias_values, batch_count, input_channels,
      input_height, input_width, output_channels, kernel_height, kernel_width,
      stride, padding, output_padding, dilation);

  // Transposed convolution weights use [Cin, Cout, KH, KW].
  flib::ftensor input(
      {batch_count, input_channels, input_height, input_width}, queue);
  flib::ftensor weight(
      {input_channels, output_channels, kernel_height, kernel_width}, queue);
  flib::ftensor bias({output_channels}, queue);
  input.copy_from(input_values.data(), queue).wait_and_throw();
  weight.copy_from(weight_values.data(), queue).wait_and_throw();
  bias.copy_from(bias_values.data(), queue).wait_and_throw();

  flib::ftensor output = flib::operations::convolution2dTranspose(
      input, weight, bias, stride, padding, output_padding, dilation, queue);
  std::vector<float> actual = output.to_host(queue);
  std::vector<std::size_t> expected_shape{1, 3, 4, 4};
  bool passed = output.getShape() == expected_shape &&
                compareValues(actual, expected);
  std::cout << (passed ? "Passed convolution2dTranspose"
                       : "Failed convolution2dTranspose")
            << std::endl;
  return passed;
}

int main() {
  try {
    flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                       flib::vendor::NVIDIA,
                                       flib::backend::CUDA, true);
    sycl::queue queue = flib::sycl_handler::get_queue("cuda");
    flib::sycl_handler::get_device_info("cuda");

    bool passed = testConvolution2d(queue);
    passed = testConvolution2dTranspose(queue) && passed;
    std::cout << (passed ? "All convolution tests passed"
                         : "Convolution tests failed")
              << std::endl;
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
