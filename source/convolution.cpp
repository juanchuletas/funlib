#include <funlib/operations/convolution/convolution.hpp>

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

template Tensor<double> convolution(const Tensor<double> &,
                                    const Tensor<double> &,
                                    const Tensor<double> &, std::size_t,
                                    sycl::queue, sycl::event *);
template Tensor<float> convolution(const Tensor<float> &, const Tensor<float> &,
                                   const Tensor<float> &, std::size_t,
                                   sycl::queue, sycl::event *);
template Tensor<int> convolution(const Tensor<int> &, const Tensor<int> &,
                                 const Tensor<int> &, std::size_t, sycl::queue,
                                 sycl::event *);

} // namespace flib::operations
