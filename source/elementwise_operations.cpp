#include <funlib/operations/elementwise_operations/elementwise_operations.hpp>

namespace flib::operations {
template <typename T>
Tensor<T> scale(const Tensor<T> &input, T value, sycl::queue Q,
                sycl::event *kernel_event) {
  const std::vector<std::size_t> &shape = input.getShape();
  if (shape.size() == 0) {
    throw std::invalid_argument(
        "Scale requires a tensor with at least one dimension");
  }

  std::size_t size = input.getSize();
  if (input.is_device()) {
    if (!input.is_accessible_from(Q)) {
      throw std::invalid_argument("Scale queue cannot access the input tensor");
    }

    Tensor<T> output(shape, Q);
    if (size == 0) {
      return output;
    }

    const T *input_data = input.device_data();
    T *output_data = output.device_data();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>{size}, [=](sycl::item<1> item) {
        std::size_t index = item.get_id(0);
        output_data[index] = input_data[index] * value;
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
    event.wait();
    return output;
  }

  Tensor<T> output(shape);
  if (size == 0) {
    return output;
  }
  {
    sycl::buffer<T, 1> input_buffer = input.to_sycl_buffer();
    sycl::buffer<T, 1> output_buffer = output.to_sycl_buffer();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      auto input_accessor =
          input_buffer.template get_access<sycl::access::mode::read>(cgh);
      auto output_accessor =
          output_buffer.template get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>{size}, [=](sycl::item<1> item) {
        std::size_t index = item.get_id(0);
        output_accessor[index] = input_accessor[index] * value;
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
  }
  return output;
}

template <typename T>
Tensor<T> add(const Tensor<T> &left, const Tensor<T> &right, sycl::queue Q,
              sycl::event *kernel_event) {
  if (left.getShape() != right.getShape()) {
    throw std::invalid_argument("Add requires tensors with the same shape");
  }
  if (left.getShape().empty()) {
    throw std::invalid_argument(
        "Add requires tensors with at least one dimension");
  }
  if (left.is_device() != right.is_device()) {
    throw std::invalid_argument("Add inputs must use the same memory location");
  }

  const std::vector<std::size_t> &shape = left.getShape();
  std::size_t size = left.getSize();
  if (left.is_device()) {
    if (!left.is_accessible_from(Q) || !right.is_accessible_from(Q)) {
      throw std::invalid_argument("Add queue cannot access the input tensors");
    }
    Tensor<T> output(shape, Q);
    if (size == 0) {
      return output;
    }

    const T *left_data = left.device_data();
    const T *right_data = right.device_data();
    T *output_data = output.device_data();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>{size}, [=](sycl::item<1> item) {
        std::size_t index = item.get_id(0);
        output_data[index] = left_data[index] + right_data[index];
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
    event.wait();
    return output;
  }

  Tensor<T> output(shape);
  if (size == 0) {
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
      cgh.parallel_for(sycl::range<1>{size}, [=](sycl::item<1> item) {
        std::size_t index = item.get_id(0);
        output_accessor[index] = left_accessor[index] + right_accessor[index];
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
  }
  return output;
}

template <typename T>
Tensor<T> add_bias(const Tensor<T> &input, const Tensor<T> &bias, sycl::queue Q,
                   sycl::event *kernel_event) {
  if (input.getShape().empty()) {
    throw std::invalid_argument(
        "Add bias requires an input with at least one dimension");
  }
  std::size_t feature_count = input.getShape().back();
  if (feature_count == 0) {
    throw std::invalid_argument(
        "Add bias final input dimension cannot be zero");
  }
  if (bias.getRank() != 1 || bias.getSize() != feature_count) {
    throw std::invalid_argument(
        "Add bias requires a one dimensional bias matching the final input "
        "dimension");
  }
  if (input.is_device() != bias.is_device()) {
    throw std::invalid_argument(
        "Add bias tensors must use the same memory location");
  }

  const std::vector<std::size_t> &shape = input.getShape();
  std::size_t size = input.getSize();
  if (input.is_device()) {
    if (!input.is_accessible_from(Q) || !bias.is_accessible_from(Q)) {
      throw std::invalid_argument(
          "Add bias queue cannot access the input tensors");
    }
    Tensor<T> output(shape, Q);
    if (size == 0) {
      return output;
    }

    const T *input_data = input.device_data();
    const T *bias_data = bias.device_data();
    T *output_data = output.device_data();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>{size}, [=](sycl::item<1> item) {
        std::size_t index = item.get_id(0);
        output_data[index] =
            input_data[index] + bias_data[index % feature_count];
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
    event.wait();
    return output;
  }

  Tensor<T> output(shape);
  if (size == 0) {
    return output;
  }
  {
    sycl::buffer<T, 1> input_buffer = input.to_sycl_buffer();
    sycl::buffer<T, 1> bias_buffer = bias.to_sycl_buffer();
    sycl::buffer<T, 1> output_buffer = output.to_sycl_buffer();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      auto input_accessor =
          input_buffer.template get_access<sycl::access::mode::read>(cgh);
      auto bias_accessor =
          bias_buffer.template get_access<sycl::access::mode::read>(cgh);
      auto output_accessor =
          output_buffer.template get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>{size}, [=](sycl::item<1> item) {
        std::size_t index = item.get_id(0);
        output_accessor[index] =
            input_accessor[index] + bias_accessor[index % feature_count];
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
  }
  return output;
}

template <typename T>
Tensor<T> gelu(const Tensor<T> &input, sycl::queue Q,
               sycl::event *kernel_event) {
  const std::vector<std::size_t> &shape = input.getShape();
  if (shape.empty()) {
    throw std::invalid_argument(
        "GELU requires a tensor with at least one dimension");
  }

  std::size_t size = input.getSize();
  if (input.is_device()) {
    if (!input.is_accessible_from(Q)) {
      throw std::invalid_argument("GELU queue cannot access the input tensor");
    }
    Tensor<T> output(shape, Q);
    if (size == 0) {
      return output;
    }

    const T *input_data = input.device_data();
    T *output_data = output.device_data();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>{size}, [=](sycl::item<1> item) {
        std::size_t index = item.get_id(0);
        T value = input_data[index];
        output_data[index] =
            T(0.5) * value *
            (T(1) + sycl::erf(value * T(0.70710678118654752440)));
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
    event.wait();
    return output;
  }

  Tensor<T> output(shape);
  if (size == 0) {
    return output;
  }
  {
    sycl::buffer<T, 1> input_buffer = input.to_sycl_buffer();
    sycl::buffer<T, 1> output_buffer = output.to_sycl_buffer();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      auto input_accessor =
          input_buffer.template get_access<sycl::access::mode::read>(cgh);
      auto output_accessor =
          output_buffer.template get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>{size}, [=](sycl::item<1> item) {
        std::size_t index = item.get_id(0);
        T value = input_accessor[index];
        output_accessor[index] =
            T(0.5) * value *
            (T(1) + sycl::erf(value * T(0.70710678118654752440)));
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
  }
  return output;
}

template <typename T>
Tensor<T> geglu(const Tensor<T> &input, sycl::queue Q,
                sycl::event *kernel_event) {
  const std::vector<std::size_t> &input_shape = input.getShape();
  if (input_shape.empty()) {
    throw std::invalid_argument(
        "GEGLU requires a tensor with at least one dimension");
  }
  std::size_t packed_size = input_shape.back();
  if (packed_size == 0 || packed_size % 2 != 0) {
    throw std::invalid_argument(
        "GEGLU final input dimension must be positive and even");
  }

  std::size_t output_width = packed_size / 2;
  std::vector<std::size_t> output_shape = input_shape;
  output_shape.back() = output_width;
  std::size_t output_size = input.getSize() / 2;
  if (input.is_device()) {
    if (!input.is_accessible_from(Q)) {
      throw std::invalid_argument("GEGLU queue cannot access the input tensor");
    }
    Tensor<T> output(output_shape, Q);
    if (output_size == 0) {
      return output;
    }

    const T *input_data = input.device_data();
    T *output_data = output.device_data();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>{output_size}, [=](sycl::item<1> item) {
        std::size_t index = item.get_id(0);
        std::size_t row = index / output_width;
        std::size_t column = index % output_width;
        std::size_t input_offset = row * packed_size + column;
        T value = input_data[input_offset];
        T gate = input_data[input_offset + output_width];
        T activated_gate = T(0.5) * gate *
                           (T(1) + sycl::erf(gate * T(0.70710678118654752440)));
        output_data[index] = value * activated_gate;
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
    sycl::buffer<T, 1> input_buffer = input.to_sycl_buffer();
    sycl::buffer<T, 1> output_buffer = output.to_sycl_buffer();
    sycl::event event = Q.submit([&](sycl::handler &cgh) {
      auto input_accessor =
          input_buffer.template get_access<sycl::access::mode::read>(cgh);
      auto output_accessor =
          output_buffer.template get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>{output_size}, [=](sycl::item<1> item) {
        std::size_t index = item.get_id(0);
        std::size_t row = index / output_width;
        std::size_t column = index % output_width;
        std::size_t input_offset = row * packed_size + column;
        T value = input_accessor[input_offset];
        T gate = input_accessor[input_offset + output_width];
        T activated_gate = T(0.5) * gate *
                           (T(1) + sycl::erf(gate * T(0.70710678118654752440)));
        output_accessor[index] = value * activated_gate;
      });
    });
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
  }
  return output;
}

template Tensor<double> scale(const Tensor<double> &, double, sycl::queue,
                              sycl::event *);
template Tensor<float> scale(const Tensor<float> &, float, sycl::queue,
                             sycl::event *);
template Tensor<int> scale(const Tensor<int> &, int, sycl::queue,
                           sycl::event *);
template Tensor<double> add(const Tensor<double> &, const Tensor<double> &,
                            sycl::queue, sycl::event *);
template Tensor<float> add(const Tensor<float> &, const Tensor<float> &,
                           sycl::queue, sycl::event *);
template Tensor<int> add(const Tensor<int> &, const Tensor<int> &, sycl::queue,
                         sycl::event *);
template Tensor<double> add_bias(const Tensor<double> &, const Tensor<double> &,
                                 sycl::queue, sycl::event *);
template Tensor<float> add_bias(const Tensor<float> &, const Tensor<float> &,
                                sycl::queue, sycl::event *);
template Tensor<int> add_bias(const Tensor<int> &, const Tensor<int> &,
                              sycl::queue, sycl::event *);
template Tensor<double> gelu(const Tensor<double> &, sycl::queue,
                             sycl::event *);
template Tensor<float> gelu(const Tensor<float> &, sycl::queue, sycl::event *);
template Tensor<double> geglu(const Tensor<double> &, sycl::queue,
                              sycl::event *);
template Tensor<float> geglu(const Tensor<float> &, sycl::queue, sycl::event *);
} // namespace flib::operations
