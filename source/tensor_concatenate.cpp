#include <funlib/Tensor/tensor_operations.hpp>

namespace flib {
template <typename T>
Tensor<T> tensor_operations::concatenate(const Tensor<T> &left,
                                         const Tensor<T> &right,
                                         std::size_t axis, sycl::queue Q,
                                         sycl::event *kernel_event) {
  if (left.getRank() == 0 || right.getRank() == 0) {
    throw std::invalid_argument(
        "Concatenate requires tensors with at least one dimension");
  }
  if (left.getRank() != right.getRank()) {
    throw std::invalid_argument(
        "Concatenate requires tensors with the same rank");
  }
  if (axis >= left.getRank()) {
    throw std::invalid_argument("Concatenate axis is outside the tensor rank");
  }

  const std::vector<std::size_t> &left_shape = left.getShape();
  const std::vector<std::size_t> &right_shape = right.getShape();
  for (std::size_t dimension = 0; dimension < left.getRank(); dimension++) {
    if (dimension != axis && left_shape[dimension] != right_shape[dimension]) {
      throw std::invalid_argument(
          "Concatenate dimensions must match outside the selected axis");
    }
  }
  if (left.is_device() != right.is_device()) {
    throw std::invalid_argument(
        "Concatenate tensors must use the same memory location");
  }
  if (left.is_device() &&
      (!left.is_accessible_from(Q) || !right.is_accessible_from(Q))) {
    throw std::invalid_argument(
        "Concatenate queue cannot access the input tensors");
  }

  std::vector<std::size_t> output_shape = left_shape;
  output_shape[axis] += right_shape[axis];
  std::size_t inner_size = 1;
  for (std::size_t dimension = axis + 1; dimension < left.getRank();
       dimension++) {
    inner_size *= left_shape[dimension];
  }
  std::size_t left_axis_size = left_shape[axis];
  std::size_t right_axis_size = right_shape[axis];
  std::size_t output_axis_size = output_shape[axis];
  std::size_t output_size = 1;
  for (std::size_t dimension : output_shape) {
    output_size *= dimension;
  }

  if (left.is_device()) {
    Tensor<T> output(output_shape, Q);
    if (output_size == 0) {
      return output;
    }
    const T *left_data = left.device_data();
    const T *right_data = right.device_data();
    T *output_data = output.device_data();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>{output_size}, [=](sycl::item<1> item) {
        std::size_t output_index = item.get_id(0);
        std::size_t output_block = output_axis_size * inner_size;
        std::size_t outer = output_index / output_block;
        std::size_t block_index = output_index % output_block;
        std::size_t axis_index = block_index / inner_size;
        std::size_t inner = block_index % inner_size;
        if (axis_index < left_axis_size) {
          std::size_t input_index =
              (outer * left_axis_size + axis_index) * inner_size + inner;
          output_data[output_index] = left_data[input_index];
        } else {
          std::size_t right_axis_index = axis_index - left_axis_size;
          std::size_t input_index =
              (outer * right_axis_size + right_axis_index) * inner_size + inner;
          output_data[output_index] = right_data[input_index];
        }
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
  {
    sycl::buffer<T, 1> left_buffer = left.to_sycl_buffer();
    sycl::buffer<T, 1> right_buffer = right.to_sycl_buffer();
    sycl::buffer<T, 1> output_buffer = output.to_sycl_buffer();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      auto left_accessor =
          left_buffer.template get_access<sycl::access::mode::read>(cgh);
      auto right_accessor =
          right_buffer.template get_access<sycl::access::mode::read>(cgh);
      auto output_accessor =
          output_buffer.template get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>{output_size}, [=](sycl::item<1> item) {
        std::size_t output_index = item.get_id(0);
        std::size_t output_block = output_axis_size * inner_size;
        std::size_t outer = output_index / output_block;
        std::size_t block_index = output_index % output_block;
        std::size_t axis_index = block_index / inner_size;
        std::size_t inner = block_index % inner_size;
        if (axis_index < left_axis_size) {
          std::size_t input_index =
              (outer * left_axis_size + axis_index) * inner_size + inner;
          output_accessor[output_index] = left_accessor[input_index];
        } else {
          std::size_t right_axis_index = axis_index - left_axis_size;
          std::size_t input_index =
              (outer * right_axis_size + right_axis_index) * inner_size + inner;
          output_accessor[output_index] = right_accessor[input_index];
        }
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
  }
  return output;
}

template Tensor<double> tensor_operations::concatenate(const Tensor<double> &,
                                                       const Tensor<double> &,
                                                       std::size_t, sycl::queue,
                                                       sycl::event *);
template Tensor<float> tensor_operations::concatenate(const Tensor<float> &,
                                                      const Tensor<float> &,
                                                      std::size_t, sycl::queue,
                                                      sycl::event *);
template Tensor<int> tensor_operations::concatenate(const Tensor<int> &,
                                                    const Tensor<int> &,
                                                    std::size_t, sycl::queue,
                                                    sycl::event *);
} // namespace flib
