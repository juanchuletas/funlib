#include <funlib/Tensor/tensor_operations.hpp>

#include <stdexcept>
#include <vector>

namespace flib {

template <typename T>
Tensor<T> tensor_operations::slice(const Tensor<T> &input,
                                   std::size_t dimension, std::size_t start,
                                   std::size_t end, sycl::queue Q,
                                   sycl::event *kernel_event) {
  if (input.getRank() == 0) {
    throw std::invalid_argument(
        "Slice requires a tensor with at least one dimension");
  }
  if (dimension >= input.getRank()) {
    throw std::invalid_argument("Slice dimension is outside the tensor rank");
  }

  const std::vector<std::size_t> &input_shape = input.getShape();
  if (start > end || end > input_shape[dimension]) {
    throw std::invalid_argument("Slice range is outside the selected dimension");
  }
  if (input.is_device() && !input.is_accessible_from(Q)) {
    throw std::invalid_argument("Slice queue cannot access the input tensor");
  }

  std::vector<std::size_t> output_shape = input_shape;
  output_shape[dimension] = end - start;

  std::size_t inner_size = 1;
  for (std::size_t index = dimension + 1; index < input.getRank(); index++) {
    inner_size *= input_shape[index];
  }

  std::size_t output_size = 1;
  for (std::size_t size : output_shape) {
    output_size *= size;
  }
  std::size_t input_dimension_size = input_shape[dimension];
  std::size_t output_dimension_size = output_shape[dimension];

  if (input.is_device()) {
    Tensor<T> output(output_shape, Q);
    if (output_size == 0) {
      return output;
    }

    const T *input_data = input.device_data();
    T *output_data = output.device_data();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>{output_size}, [=](sycl::item<1> item) {
        std::size_t output_index = item.get_id(0);
        std::size_t output_block = output_dimension_size * inner_size;
        std::size_t outer = output_index / output_block;
        std::size_t block_index = output_index % output_block;
        std::size_t dimension_index = block_index / inner_size;
        std::size_t inner = block_index % inner_size;
        std::size_t input_index =
            (outer * input_dimension_size + start + dimension_index) *
                inner_size +
            inner;
        output_data[output_index] = input_data[input_index];
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
    event.wait();
    return output;
  }

  Tensor<T> output(output_shape);
  if (output_size == 0) {
    return output;
  }

  sycl::buffer<T, 1> input_buffer = input.to_sycl_buffer();
  sycl::buffer<T, 1> output_buffer = output.to_sycl_buffer();
  sycl::event event = Q.submit([&](sycl::handler &cgh) {
    auto input_accessor =
        input_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto output_accessor =
        output_buffer.template get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for(sycl::range<1>{output_size}, [=](sycl::item<1> item) {
      std::size_t output_index = item.get_id(0);
      std::size_t output_block = output_dimension_size * inner_size;
      std::size_t outer = output_index / output_block;
      std::size_t block_index = output_index % output_block;
      std::size_t dimension_index = block_index / inner_size;
      std::size_t inner = block_index % inner_size;
      std::size_t input_index =
          (outer * input_dimension_size + start + dimension_index) *
              inner_size +
          inner;
      output_accessor[output_index] = input_accessor[input_index];
    });
  });
  if (kernel_event != nullptr) {
    *kernel_event = event;
  }
  return output;
}

template Tensor<double>
tensor_operations::slice(const Tensor<double> &, std::size_t, std::size_t,
                         std::size_t, sycl::queue, sycl::event *);
template Tensor<float>
tensor_operations::slice(const Tensor<float> &, std::size_t, std::size_t,
                         std::size_t, sycl::queue, sycl::event *);
template Tensor<int>
tensor_operations::slice(const Tensor<int> &, std::size_t, std::size_t,
                         std::size_t, sycl::queue, sycl::event *);

} // namespace flib
