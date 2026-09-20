#include <funlib/Tensor/tensor_operations.hpp>
#include <funlib/operations/attention/attention_operations.hpp>
#include <funlib/operations/softmax/softmax.hpp>

#include <cmath>

namespace flib::operations {
template <typename T>
sycl::event submit_split_heads_usm(const T *input_data, T *output_data,
                                   std::size_t batch_size,
                                   std::size_t token_count,
                                   std::size_t head_count,
                                   std::size_t head_size, sycl::queue Q) {
  std::size_t vector_count = batch_size * token_count * head_count;
  return Q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::range<2>{vector_count, head_size}, [=](sycl::item<2> item) {
          std::size_t input_vector = item.get_id(0);
          std::size_t feature = item.get_id(1);
          std::size_t head = input_vector % head_count;
          std::size_t token = (input_vector / head_count) % token_count;
          std::size_t batch = input_vector / (token_count * head_count);
          std::size_t output_vector =
              (batch * head_count + head) * token_count + token;
          output_data[output_vector * head_size + feature] =
              input_data[input_vector * head_size + feature];
        });
  });
}

template <typename T>
sycl::event submit_split_heads_buffer(sycl::buffer<T, 1> &input_buffer,
                                      sycl::buffer<T, 1> &output_buffer,
                                      std::size_t batch_size,
                                      std::size_t token_count,
                                      std::size_t head_count,
                                      std::size_t head_size, sycl::queue Q) {
  std::size_t vector_count = batch_size * token_count * head_count;
  return Q.submit([&](sycl::handler &cgh) {
    auto input_accessor =
        input_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto output_accessor =
        output_buffer.template get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for(
        sycl::range<2>{vector_count, head_size}, [=](sycl::item<2> item) {
          std::size_t input_vector = item.get_id(0);
          std::size_t feature = item.get_id(1);
          std::size_t head = input_vector % head_count;
          std::size_t token = (input_vector / head_count) % token_count;
          std::size_t batch = input_vector / (token_count * head_count);
          std::size_t output_vector =
              (batch * head_count + head) * token_count + token;
          output_accessor[output_vector * head_size + feature] =
              input_accessor[input_vector * head_size + feature];
        });
  });
}

template <typename T>
sycl::event submit_join_heads_usm(const T *input_data, T *output_data,
                                  std::size_t batch_size,
                                  std::size_t token_count,
                                  std::size_t head_count, std::size_t head_size,
                                  sycl::queue Q) {
  std::size_t vector_count = batch_size * head_count * token_count;
  return Q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::range<2>{vector_count, head_size}, [=](sycl::item<2> item) {
          std::size_t input_vector = item.get_id(0);
          std::size_t feature = item.get_id(1);
          std::size_t token = input_vector % token_count;
          std::size_t head = (input_vector / token_count) % head_count;
          std::size_t batch = input_vector / (head_count * token_count);
          std::size_t output_vector =
              (batch * token_count + token) * head_count + head;
          output_data[output_vector * head_size + feature] =
              input_data[input_vector * head_size + feature];
        });
  });
}

template <typename T>
sycl::event submit_join_heads_buffer(sycl::buffer<T, 1> &input_buffer,
                                     sycl::buffer<T, 1> &output_buffer,
                                     std::size_t batch_size,
                                     std::size_t token_count,
                                     std::size_t head_count,
                                     std::size_t head_size, sycl::queue Q) {
  std::size_t vector_count = batch_size * head_count * token_count;
  return Q.submit([&](sycl::handler &cgh) {
    auto input_accessor =
        input_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto output_accessor =
        output_buffer.template get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for(
        sycl::range<2>{vector_count, head_size}, [=](sycl::item<2> item) {
          std::size_t input_vector = item.get_id(0);
          std::size_t feature = item.get_id(1);
          std::size_t token = input_vector % token_count;
          std::size_t head = (input_vector / token_count) % head_count;
          std::size_t batch = input_vector / (head_count * token_count);
          std::size_t output_vector =
              (batch * token_count + token) * head_count + head;
          output_accessor[output_vector * head_size + feature] =
              input_accessor[input_vector * head_size + feature];
        });
  });
}

template <typename T>
Tensor<T> split_heads(const Tensor<T> &input, sycl::queue Q,
                      sycl::event *kernel_event) {
  if (input.getRank() != 4) {
    throw std::invalid_argument(
        "Split heads requires a tensor with shape [B, N, H, D]");
  }
  const std::vector<std::size_t> &shape = input.getShape();
  std::size_t batch_size = shape[0];
  std::size_t token_count = shape[1];
  std::size_t head_count = shape[2];
  std::size_t head_size = shape[3];
  std::vector<std::size_t> output_shape{batch_size, head_count, token_count,
                                        head_size};

  if (input.is_device()) {
    if (!input.is_accessible_from(Q)) {
      throw std::invalid_argument(
          "Split heads queue cannot access the input tensor");
    }
    Tensor<T> output(output_shape, Q);
    if (input.getSize() == 0) {
      return output;
    }
    sycl::event event = submit_split_heads_usm(
        input.device_data(), output.device_data(), batch_size, token_count,
        head_count, head_size, Q);
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
    event.wait();
    return output;
  }

  Tensor<T> output(output_shape);
  if (input.getSize() == 0) {
    return output;
  }
  {
    sycl::buffer<T, 1> input_buffer = input.to_sycl_buffer();
    sycl::buffer<T, 1> output_buffer = output.to_sycl_buffer();
    sycl::event event =
        submit_split_heads_buffer(input_buffer, output_buffer, batch_size,
                                  token_count, head_count, head_size, Q);
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
  }
  return output;
}

template <typename T>
Tensor<T> join_heads(const Tensor<T> &input, sycl::queue Q,
                     sycl::event *kernel_event) {
  if (input.getRank() != 4) {
    throw std::invalid_argument(
        "Join heads requires a tensor with shape [B, H, N, D]");
  }
  const std::vector<std::size_t> &shape = input.getShape();
  std::size_t batch_size = shape[0];
  std::size_t head_count = shape[1];
  std::size_t token_count = shape[2];
  std::size_t head_size = shape[3];
  std::vector<std::size_t> output_shape{batch_size, token_count, head_count,
                                        head_size};

  if (input.is_device()) {
    if (!input.is_accessible_from(Q)) {
      throw std::invalid_argument(
          "Join heads queue cannot access the input tensor");
    }
    Tensor<T> output(output_shape, Q);
    if (input.getSize() == 0) {
      return output;
    }
    sycl::event event = submit_join_heads_usm(
        input.device_data(), output.device_data(), batch_size, token_count,
        head_count, head_size, Q);
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
    event.wait();
    return output;
  }

  Tensor<T> output(output_shape);
  if (input.getSize() == 0) {
    return output;
  }
  {
    sycl::buffer<T, 1> input_buffer = input.to_sycl_buffer();
    sycl::buffer<T, 1> output_buffer = output.to_sycl_buffer();
    sycl::event event =
        submit_join_heads_buffer(input_buffer, output_buffer, batch_size,
                                 token_count, head_count, head_size, Q);
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
  }
  return output;
}

template <typename T>
Tensor<T> scaled_dot_product_attention(const Tensor<T> &query,
                                       const Tensor<T> &key,
                                       const Tensor<T> &value,
                                       std::size_t head_count, sycl::queue Q) {
  if (query.getRank() != 4 || key.getRank() != 4 || value.getRank() != 4) {
    throw std::invalid_argument(
        "Attention requires Q, K and V with shape [B, N, H, D]");
  }
  if (head_count == 0) {
    throw std::invalid_argument("Attention head count cannot be zero");
  }

  const std::vector<std::size_t> &query_shape = query.getShape();
  const std::vector<std::size_t> &key_shape = key.getShape();
  const std::vector<std::size_t> &value_shape = value.getShape();
  if (query_shape[0] != key_shape[0] || query_shape[0] != value_shape[0]) {
    throw std::invalid_argument("Attention batch dimensions must match");
  }
  if (query_shape[2] != head_count || key_shape[2] != head_count ||
      value_shape[2] != head_count) {
    throw std::invalid_argument("Attention tensor head counts must match");
  }
  if (query_shape[3] != key_shape[3] || query_shape[3] != value_shape[3]) {
    throw std::invalid_argument("Attention head sizes must match");
  }
  if (key_shape[1] != value_shape[1]) {
    throw std::invalid_argument(
        "Attention key and value token counts must match");
  }

  Tensor<T> query_heads = split_heads(query, Q);
  Tensor<T> key_heads = split_heads(key, Q);
  Tensor<T> value_heads = split_heads(value, Q);
  Tensor<T> scores =
      tensor_operations::gemm_batched(query_heads, key_heads, Q, false, true);
  T scale =
      T(1) / static_cast<T>(std::sqrt(static_cast<double>(query_shape[3])));
  Tensor<T> probabilities = scaled_softmax(scores, scale, Q);
  Tensor<T> head_output =
      tensor_operations::gemm_batched(probabilities, value_heads, Q);
  Tensor<T> output = join_heads(head_output, Q);
  output.reshape({query_shape[0], query_shape[1], head_count * query_shape[3]});
  return output;
}

template Tensor<double> split_heads(const Tensor<double> &, sycl::queue,
                                    sycl::event *);
template Tensor<float> split_heads(const Tensor<float> &, sycl::queue,
                                   sycl::event *);
template Tensor<int> split_heads(const Tensor<int> &, sycl::queue,
                                 sycl::event *);
template Tensor<double> join_heads(const Tensor<double> &, sycl::queue,
                                   sycl::event *);
template Tensor<float> join_heads(const Tensor<float> &, sycl::queue,
                                  sycl::event *);
template Tensor<int> join_heads(const Tensor<int> &, sycl::queue,
                                sycl::event *);
template Tensor<double> scaled_dot_product_attention(const Tensor<double> &,
                                                     const Tensor<double> &,
                                                     const Tensor<double> &,
                                                     std::size_t, sycl::queue);
template Tensor<float> scaled_dot_product_attention(const Tensor<float> &,
                                                    const Tensor<float> &,
                                                    const Tensor<float> &,
                                                    std::size_t, sycl::queue);
} // namespace flib::operations
