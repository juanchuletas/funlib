#include <funlib/operations/sampling/interpolation.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flib::operations {

template <typename T> T bicubic_kernel_weight(T distance) {
  T absolute_distance = sycl::fabs(distance);
  T coefficient = T(-0.75);
  if (absolute_distance <= T(1)) {
    return ((coefficient + T(2)) * absolute_distance -
            (coefficient + T(3))) *
               absolute_distance * absolute_distance +
           T(1);
  }
  if (absolute_distance < T(2)) {
    return ((coefficient * absolute_distance - T(5) * coefficient) *
                absolute_distance +
            T(8) * coefficient) *
               absolute_distance -
           T(4) * coefficient;
  }
  return T(0);
}

template <typename T>
Tensor<T> interpolate_bicubic(const Tensor<T> &input,
                              std::size_t output_height,
                              std::size_t output_width, sycl::queue Q,
                              sycl::event *kernel_event) {
  if (input.getRank() != 4) {
    throw std::invalid_argument(
        "Bicubic interpolation requires input shape [B, C, H, W]");
  }
  if (!input.is_device()) {
    throw std::invalid_argument(
        "Bicubic interpolation requires a device tensor");
  }
  if (!input.is_accessible_from(Q)) {
    throw std::invalid_argument(
        "Bicubic interpolation queue cannot access the input tensor");
  }
  if (output_height == 0 || output_width == 0) {
    throw std::invalid_argument(
        "Bicubic interpolation output dimensions cannot be zero");
  }

  const std::vector<std::size_t> &input_shape = input.getShape();
  std::size_t batch_count = input_shape[0];
  std::size_t channel_count = input_shape[1];
  std::size_t input_height = input_shape[2];
  std::size_t input_width = input_shape[3];
  if (input_height == 0 || input_width == 0) {
    throw std::invalid_argument(
        "Bicubic interpolation input dimensions cannot be zero");
  }
  if (input_height > static_cast<std::size_t>(
                         std::numeric_limits<std::int32_t>::max()) ||
      input_width > static_cast<std::size_t>(
                        std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument(
        "Bicubic interpolation dimensions must fit in int32");
  }

  std::vector<std::size_t> output_shape{batch_count, channel_count,
                                        output_height, output_width};
  Tensor<T> output(output_shape, Q);
  std::size_t output_size =
      batch_count * channel_count * output_height * output_width;
  if (output_size == 0) {
    return output;
  }

  const T *input_data = input.device_data();
  T *output_data = output.device_data();
  sycl::event event = Q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>{output_size}, [=](sycl::item<1> item) {
      std::size_t output_index = item.get_id(0);
      std::size_t output_x = output_index % output_width;
      std::size_t remaining = output_index / output_width;
      std::size_t output_y = remaining % output_height;
      remaining /= output_height;
      std::size_t channel = remaining % channel_count;
      std::size_t batch = remaining / channel_count;

      // align_corners=false maps output pixel centers to input pixel centers.
      T source_x =
          (static_cast<T>(output_x) + T(0.5)) *
              static_cast<T>(input_width) / static_cast<T>(output_width) -
          T(0.5);
      T source_y =
          (static_cast<T>(output_y) + T(0.5)) *
              static_cast<T>(input_height) / static_cast<T>(output_height) -
          T(0.5);
      std::int32_t base_x =
          static_cast<std::int32_t>(sycl::floor(source_x));
      std::int32_t base_y =
          static_cast<std::int32_t>(sycl::floor(source_y));
      T fraction_x = source_x - static_cast<T>(base_x);
      T fraction_y = source_y - static_cast<T>(base_y);

      T value = T(0);
      for (std::int32_t sample_y = 0; sample_y < 4; sample_y++) {
        std::int32_t input_y = base_y + sample_y - 1;
        input_y = sycl::max(
            std::int32_t(0),
            sycl::min(input_y, static_cast<std::int32_t>(input_height) - 1));
        T weight_y = bicubic_kernel_weight(
            static_cast<T>(sample_y - 1) - fraction_y);

        for (std::int32_t sample_x = 0; sample_x < 4; sample_x++) {
          std::int32_t input_x = base_x + sample_x - 1;
          input_x = sycl::max(
              std::int32_t(0),
              sycl::min(input_x, static_cast<std::int32_t>(input_width) - 1));
          T weight_x = bicubic_kernel_weight(
              static_cast<T>(sample_x - 1) - fraction_x);
          std::size_t input_index =
              ((batch * channel_count + channel) * input_height +
               static_cast<std::size_t>(input_y)) *
                  input_width +
              static_cast<std::size_t>(input_x);
          value += input_data[input_index] * weight_y * weight_x;
        }
      }
      output_data[output_index] = value;
    });
  });
  if (kernel_event != nullptr) {
    *kernel_event = event;
  }
  event.wait();
  return output;
}

template Tensor<double> interpolate_bicubic(const Tensor<double> &,
                                            std::size_t, std::size_t,
                                            sycl::queue, sycl::event *);
template Tensor<float> interpolate_bicubic(const Tensor<float> &, std::size_t,
                                           std::size_t, sycl::queue,
                                           sycl::event *);

} // namespace flib::operations
