#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

float readPlane(const std::vector<float> &triplane, std::size_t plane,
                std::size_t channel, std::ptrdiff_t y, std::ptrdiff_t x,
                std::size_t channel_count, std::size_t height,
                std::size_t width) {
  if (x < 0 || x >= static_cast<std::ptrdiff_t>(width) || y < 0 ||
      y >= static_cast<std::ptrdiff_t>(height)) {
    return 0.0f;
  }
  std::size_t index =
      ((plane * channel_count + channel) * height +
       static_cast<std::size_t>(y)) *
          width +
      static_cast<std::size_t>(x);
  return triplane[index];
}

std::vector<float> triplaneReference(
    const std::vector<float> &triplane,
    const std::vector<float> &positions, std::size_t point_count,
    std::size_t channel_count, std::size_t height, std::size_t width) {
  std::size_t features_per_point = 3 * channel_count;
  std::vector<float> output(point_count * features_per_point);
  for (std::size_t point = 0; point < point_count; point++) {
    float x = positions[point * 3];
    float y = positions[point * 3 + 1];
    float z = positions[point * 3 + 2];
    for (std::size_t plane = 0; plane < 3; plane++) {
      float horizontal = plane < 2 ? x : y;
      float vertical = plane == 0 ? y : z;
      float image_x =
          ((horizontal + 1.0f) * static_cast<float>(width) - 1.0f) / 2.0f;
      float image_y =
          ((vertical + 1.0f) * static_cast<float>(height) - 1.0f) / 2.0f;
      std::ptrdiff_t x0 = static_cast<std::ptrdiff_t>(std::floor(image_x));
      std::ptrdiff_t y0 = static_cast<std::ptrdiff_t>(std::floor(image_y));
      std::ptrdiff_t x1 = x0 + 1;
      std::ptrdiff_t y1 = y0 + 1;
      float x_weight = image_x - static_cast<float>(x0);
      float y_weight = image_y - static_cast<float>(y0);

      for (std::size_t channel = 0; channel < channel_count; channel++) {
        float top_left = readPlane(triplane, plane, channel, y0, x0,
                                   channel_count, height, width);
        float top_right = readPlane(triplane, plane, channel, y0, x1,
                                    channel_count, height, width);
        float bottom_left = readPlane(triplane, plane, channel, y1, x0,
                                      channel_count, height, width);
        float bottom_right = readPlane(triplane, plane, channel, y1, x1,
                                       channel_count, height, width);
        std::size_t output_index =
            point * features_per_point + plane * channel_count + channel;
        output[output_index] =
            top_left * (1.0f - x_weight) * (1.0f - y_weight) +
            top_right * x_weight * (1.0f - y_weight) +
            bottom_left * (1.0f - x_weight) * y_weight +
            bottom_right * x_weight * y_weight;
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

    constexpr std::size_t point_count = 4;
    constexpr std::size_t channel_count = 2;
    constexpr std::size_t height = 2;
    constexpr std::size_t width = 2;
    const std::vector<float> triplane_values{
        1,   2,   3,   4,   5,   6,   7,   8,
        10,  20,  30,  40,  50,  60,  70,  80,
        100, 200, 300, 400, 500, 600, 700, 800};
    const std::vector<float> position_values{
        0.0f, 0.0f, 0.0f, -0.5f, 0.5f, 0.0f,
        0.5f, -0.5f, 0.5f, 2.0f, 2.0f, 2.0f};
    const std::vector<float> expected =
        triplaneReference(triplane_values, position_values, point_count,
                          channel_count, height, width);

    // The three planes and sample positions remain on the GPU.
    flib::ftensor triplane({3, channel_count, height, width}, queue);
    flib::ftensor positions({point_count, 3}, queue);
    triplane.copy_from(triplane_values.data(), queue).wait_and_throw();
    positions.copy_from(position_values.data(), queue).wait_and_throw();

    flib::ftensor output =
        flib::operations::triplane_sample(triplane, positions, queue);
    std::vector<float> actual = output.to_host(queue);
    if (output.getShape() !=
            std::vector<std::size_t>{point_count, 3 * channel_count} ||
        actual.size() != expected.size()) {
      throw std::runtime_error(
          "Triplane sampler returned an unexpected shape or size");
    }

    float max_difference = 0.0f;
    for (std::size_t index = 0; index < actual.size(); index++) {
      if (!std::isfinite(actual[index])) {
        throw std::runtime_error("Triplane sampler returned a non-finite value");
      }
      max_difference =
          std::max(max_difference, std::abs(actual[index] - expected[index]));
    }
    bool passed = max_difference < 1.0e-5f;
    std::cout << "Max absolute difference: " << max_difference << std::endl;
    std::cout << (passed ? "All triplane sampling tests passed"
                         : "Triplane sampling tests failed")
              << std::endl;
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
