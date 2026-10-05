#include <funlib/operations/convolution/convolution.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace flib::operations {

template <typename T>
Tensor<T> convolution2d(const Tensor<T> &input, const Tensor<T> &weight,
                      const Tensor<T> &bias, std::size_t stride,
                      sycl::queue queue, sycl::event *kernel_event) {
  // Check the required NCHW input and OIHW weight layouts.
  if (input.getRank() != 4) {
    throw std::invalid_argument(
        "Convolution input must have shape [B, Cin, Hin, Win]");
  }
  if (weight.getRank() != 4) {
    throw std::invalid_argument(
        "Convolution weight must have shape [Cout, Cin, KH, KW]");
  }
  if (bias.getRank() != 1) {
    throw std::invalid_argument("Convolution bias must have shape [Cout]");
  }
  if (stride == 0) {
    throw std::invalid_argument("Convolution stride must be greater than zero");
  }

  const std::vector<std::size_t> &input_shape = input.getShape();
  const std::vector<std::size_t> &weight_shape = weight.getShape();
  std::size_t batch_count = input_shape[0];
  std::size_t input_channels = input_shape[1];
  std::size_t input_height = input_shape[2];
  std::size_t input_width = input_shape[3];
  std::size_t output_channels = weight_shape[0];
  std::size_t weight_input_channels = weight_shape[1];
  std::size_t kernel_height = weight_shape[2];
  std::size_t kernel_width = weight_shape[3];

  // Check that the tensor dimensions can form a valid convolution.
  if (input_channels == 0 || output_channels == 0 || input_height == 0 ||
      input_width == 0 || kernel_height == 0 || kernel_width == 0) {
    throw std::invalid_argument(
        "Convolution channel and spatial dimensions cannot be zero");
  }
  if (input_channels != weight_input_channels) {
    throw std::invalid_argument(
        "Convolution input channels must match the weight input channels");
  }
  if (bias.getSize() != output_channels) {
    throw std::invalid_argument(
        "Convolution bias must match the output channel count");
  }
  if (kernel_height > input_height || kernel_width > input_width) {
    throw std::invalid_argument(
        "Convolution kernel cannot be larger than the input");
  }
  if (!input.is_device() || !weight.is_device() || !bias.is_device()) {
    throw std::invalid_argument(
        "Convolution requires device tensors");
  }
  if (!input.is_accessible_from(queue) || !weight.is_accessible_from(queue) ||
      !bias.is_accessible_from(queue)) {
    throw std::invalid_argument(
        "Convolution queue cannot access the input tensors");
  }

  // This first version uses no padding and dilation equal to one.
  std::size_t output_height =
      (input_height - kernel_height) / stride + 1;
  std::size_t output_width = (input_width - kernel_width) / stride + 1;
  std::vector<std::size_t> output_shape{batch_count, output_channels,
                                        output_height, output_width};
  std::size_t output_size =
      batch_count * output_channels * output_height * output_width;

  Tensor<T> output(output_shape, queue);
  if (output_size == 0) {
    return output;
  }

  const T *input_data = input.device_data();
  const T *weight_data = weight.device_data();
  const T *bias_data = bias.device_data();
  T *output_data = output.device_data();
  sycl::event event = queue.submit([&](sycl::handler &cgh) {
    // Each work item calculates one output value.
    cgh.parallel_for(sycl::range<1>{output_size}, [=](sycl::item<1> item) {
      std::size_t output_index = item.get_id(0);

      // Convert the linear index into [B, Cout, Hout, Wout].
      std::size_t remaining = output_index;
      std::size_t output_x = remaining % output_width;
      remaining /= output_width;
      std::size_t output_y = remaining % output_height;
      remaining /= output_height;
      std::size_t output_channel = remaining % output_channels;
      std::size_t batch = remaining / output_channels;

      // Start with the learned bias for this output channel.
      T sum = bias_data[output_channel];

      // Apply every input channel and kernel value to this output position.
      for (std::size_t input_channel = 0; input_channel < input_channels;
           input_channel++) {
        for (std::size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
          for (std::size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
            std::size_t input_y = output_y * stride + kernel_y;
            std::size_t input_x = output_x * stride + kernel_x;
            std::size_t input_index = ((batch * input_channels + input_channel) * input_height + input_y) * input_width + input_x;
            std::size_t weight_index =
                ((output_channel * input_channels + input_channel) *
                     kernel_height +
                 kernel_y) *
                    kernel_width +
                kernel_x;
            sum += input_data[input_index] * weight_data[weight_index];
          }
        }
      }
      output_data[output_index] = sum;
    });
  });
  if (kernel_event != nullptr) {
    *kernel_event = event;
  }
  event.wait();
  return output;
}

template <typename T>
Tensor<T> convolution2dTranspose(
    const Tensor<T> &input, const Tensor<T> &weight, const Tensor<T> &bias,
    std::size_t stride, std::size_t padding, std::size_t output_padding,
    std::size_t dilation, sycl::queue queue, sycl::event *kernel_event) {
  // Check the required NCHW input and IOHW weight layouts.
  if (input.getRank() != 4) {
    throw std::invalid_argument(
        "Transposed convolution input must have shape [B, Cin, Hin, Win]");
  }
  if (weight.getRank() != 4) {
    throw std::invalid_argument(
        "Transposed convolution weight must have shape [Cin, Cout, KH, KW]");
  }
  if (bias.getRank() != 1) {
    throw std::invalid_argument(
        "Transposed convolution bias must have shape [Cout]");
  }
  // Stride controls the distance between expanded input positions.
  if (stride == 0) {
    throw std::invalid_argument(
        "Transposed convolution stride must be greater than zero");
  }
  // Dilation controls the distance between kernel values.
  if (dilation == 0) {
    throw std::invalid_argument(
        "Transposed convolution dilation must be greater than zero");
  }
  // Output padding can only select an extra position inside the stride.
  if (output_padding >= stride) {
    throw std::invalid_argument(
        "Transposed convolution output padding must be smaller than stride");
  }

  const std::vector<std::size_t> &input_shape = input.getShape();
  const std::vector<std::size_t> &weight_shape = weight.getShape();
  std::size_t batch_count = input_shape[0];
  std::size_t input_channels = input_shape[1];
  std::size_t input_height = input_shape[2];
  std::size_t input_width = input_shape[3];
  std::size_t weight_input_channels = weight_shape[0];
  std::size_t output_channels = weight_shape[1];
  std::size_t kernel_height = weight_shape[2];
  std::size_t kernel_width = weight_shape[3];

  if (input_channels == 0 || output_channels == 0 || input_height == 0 ||
      input_width == 0 || kernel_height == 0 || kernel_width == 0) {
    throw std::invalid_argument(
        "Transposed convolution dimensions cannot be zero");
  }
  if (input_channels != weight_input_channels) {
    throw std::invalid_argument(
        "Transposed convolution input channels must match the weight input channels");
  }
  if (bias.getSize() != output_channels) {
    throw std::invalid_argument(
        "Transposed convolution bias must match the output channel count");
  }
  if (!input.is_device() || !weight.is_device() || !bias.is_device()) {
    throw std::invalid_argument(
        "Transposed convolution requires device tensors");
  }
  if (!input.is_accessible_from(queue) || !weight.is_accessible_from(queue) ||
      !bias.is_accessible_from(queue)) {
    throw std::invalid_argument(
        "Transposed convolution queue cannot access the input tensors");
  }

  // Padding removes values from both sides of the expanded output.
  std::ptrdiff_t output_height =
      static_cast<std::ptrdiff_t>((input_height - 1) * stride) -
      static_cast<std::ptrdiff_t>(2 * padding) +
      static_cast<std::ptrdiff_t>(dilation * (kernel_height - 1)) +
      static_cast<std::ptrdiff_t>(output_padding) + 1;
  std::ptrdiff_t output_width =
      static_cast<std::ptrdiff_t>((input_width - 1) * stride) -
      static_cast<std::ptrdiff_t>(2 * padding) +
      static_cast<std::ptrdiff_t>(dilation * (kernel_width - 1)) +
      static_cast<std::ptrdiff_t>(output_padding) + 1;
  if (output_height <= 0 || output_width <= 0) {
    throw std::invalid_argument(
        "Transposed convolution output dimensions must be positive");
  }

  std::size_t output_height_size =
      static_cast<std::size_t>(output_height);
  std::size_t output_width_size = static_cast<std::size_t>(output_width);
  std::vector<std::size_t> output_shape{batch_count, output_channels,
                                        output_height_size, output_width_size};
  std::size_t output_size = batch_count * output_channels *
                            output_height_size * output_width_size;

  Tensor<T> output(output_shape, queue);
  if (output_size == 0) {
    return output;
  }

  const T *input_data = input.device_data();
  const T *weight_data = weight.device_data();
  const T *bias_data = bias.device_data();
  T *output_data = output.device_data();
  sycl::event event = queue.submit([&](sycl::handler &cgh) {
    // Each work item collects all contributions for one output value.
    cgh.parallel_for(sycl::range<1>{output_size}, [=](sycl::item<1> item) {
      std::size_t output_index = item.get_id(0);

      // Convert the linear index into [B, Cout, Hout, Wout].
      std::size_t remaining = output_index;
      std::size_t output_x = remaining % output_width_size;
      remaining /= output_width_size;
      std::size_t output_y = remaining % output_height_size;
      remaining /= output_height_size;
      std::size_t output_channel = remaining % output_channels;
      std::size_t batch = remaining / output_channels;

      T sum = bias_data[output_channel];
      for (std::size_t input_channel = 0; input_channel < input_channels;
           input_channel++) {
        for (std::size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
          for (std::size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
            // Find the input coordinate that can produce this output position.
            std::ptrdiff_t input_y_value =
                static_cast<std::ptrdiff_t>(output_y) +
                static_cast<std::ptrdiff_t>(padding) -
                static_cast<std::ptrdiff_t>(kernel_y * dilation);
            std::ptrdiff_t input_x_value =
                static_cast<std::ptrdiff_t>(output_x) +
                static_cast<std::ptrdiff_t>(padding) -
                static_cast<std::ptrdiff_t>(kernel_x * dilation);

            // Stride leaves positions that do not map back to the input.
            if (input_y_value < 0 || input_x_value < 0 ||
                input_y_value % static_cast<std::ptrdiff_t>(stride) != 0 ||
                input_x_value % static_cast<std::ptrdiff_t>(stride) != 0) {
              continue;
            }

            std::size_t input_y =
                static_cast<std::size_t>(input_y_value) / stride;
            std::size_t input_x =
                static_cast<std::size_t>(input_x_value) / stride;
            if (input_y >= input_height || input_x >= input_width) {
              continue;
            }

            std::size_t input_index =
                ((batch * input_channels + input_channel) * input_height +
                 input_y) *
                    input_width +
                input_x;
            std::size_t weight_index =
                ((input_channel * output_channels + output_channel) *
                     kernel_height +
                 kernel_y) *
                    kernel_width +
                kernel_x;
            sum += input_data[input_index] * weight_data[weight_index];
          }
        }
      }
      output_data[output_index] = sum;
    });
  });
  if (kernel_event != nullptr) {
    *kernel_event = event;
  }
  event.wait();
  return output;
}

template Tensor<double> convolution2d(const Tensor<double> &,
                                      const Tensor<double> &,
                                      const Tensor<double> &, std::size_t,
                                      sycl::queue, sycl::event *);
template Tensor<float> convolution2d(const Tensor<float> &,
                                     const Tensor<float> &,
                                     const Tensor<float> &, std::size_t,
                                     sycl::queue, sycl::event *);
template Tensor<int> convolution2d(const Tensor<int> &, const Tensor<int> &,
                                   const Tensor<int> &, std::size_t,
                                   sycl::queue, sycl::event *);

template Tensor<double> convolution2dTranspose(
    const Tensor<double> &, const Tensor<double> &, const Tensor<double> &,
    std::size_t, std::size_t, std::size_t, std::size_t, sycl::queue,
    sycl::event *);
template Tensor<float> convolution2dTranspose(
    const Tensor<float> &, const Tensor<float> &, const Tensor<float> &,
    std::size_t, std::size_t, std::size_t, std::size_t, sycl::queue,
    sycl::event *);
template Tensor<int> convolution2dTranspose(
    const Tensor<int> &, const Tensor<int> &, const Tensor<int> &, std::size_t,
    std::size_t, std::size_t, std::size_t, sycl::queue, sycl::event *);

} // namespace flib::operations
