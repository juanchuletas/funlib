#include <funlib/operations/normalization/normalization.hpp>

#include <algorithm>

namespace flib::operations::details {
std::size_t layer_norm_work_group_size(std::size_t row_size, sycl::queue Q) {
  std::size_t device_maximum =
      Q.get_device().get_info<sycl::info::device::max_work_group_size>();
  std::size_t work_group_limit = std::min<std::size_t>(256, device_maximum);
  std::size_t work_group_size = 1;
  while (work_group_size < row_size &&
         work_group_size * 2 <= work_group_limit) {
    work_group_size *= 2;
  }
  return work_group_size;
}

std::size_t group_norm_work_group_size(std::size_t group_size, sycl::queue Q) {
  std::size_t device_maximum =
      Q.get_device().get_info<sycl::info::device::max_work_group_size>();
  std::size_t work_group_limit = std::min<std::size_t>(256, device_maximum);
  std::size_t work_group_size = 1;
  while (work_group_size < group_size &&
         work_group_size * 2 <= work_group_limit) {
    work_group_size *= 2;
  }
  return work_group_size;
}
} // namespace flib::operations::details

namespace flib::operations {
template <typename T>
sycl::event submit_layer_norm_usm(const T *input_data, const T *gamma_data,
                                  const T *beta_data, T *output_data,
                                  std::size_t row_count, std::size_t row_size,
                                  T epsilon, std::size_t work_group_size,
                                  sycl::queue Q) {
  std::size_t global_size = row_count * work_group_size;
  return Q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<T, 1> local_values(sycl::range<1>{work_group_size},
                                            cgh);
    cgh.parallel_for(
        sycl::nd_range<1>{global_size, work_group_size},
        [=](sycl::nd_item<1> item) {
          std::size_t row = item.get_group(0);
          std::size_t local_id = item.get_local_id(0);
          std::size_t row_offset = row * row_size;
          T local_sum = T(0);
          for (std::size_t column = local_id; column < row_size;
               column += work_group_size) {
            local_sum += input_data[row_offset + column];
          }
          local_values[local_id] = local_sum;
          item.barrier(sycl::access::fence_space::local_space);

          for (std::size_t stride = work_group_size / 2; stride > 0;
               stride /= 2) {
            if (local_id < stride) {
              local_values[local_id] += local_values[local_id + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          T mean = local_values[0] / static_cast<T>(row_size);

          T local_variance = T(0);
          for (std::size_t column = local_id; column < row_size;
               column += work_group_size) {
            T difference = input_data[row_offset + column] - mean;
            local_variance += difference * difference;
          }
          local_values[local_id] = local_variance;
          item.barrier(sycl::access::fence_space::local_space);

          for (std::size_t stride = work_group_size / 2; stride > 0;
               stride /= 2) {
            if (local_id < stride) {
              local_values[local_id] += local_values[local_id + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          T inverse_standard_deviation =
              sycl::rsqrt(local_values[0] / static_cast<T>(row_size) + epsilon);

          for (std::size_t column = local_id; column < row_size;
               column += work_group_size) {
            T normalized = (input_data[row_offset + column] - mean) *
                           inverse_standard_deviation;
            output_data[row_offset + column] =
                normalized * gamma_data[column] + beta_data[column];
          }
        });
  });
}

template <typename T>
sycl::event submit_layer_norm_buffer(
    sycl::buffer<T, 1> &input_buffer, sycl::buffer<T, 1> &gamma_buffer,
    sycl::buffer<T, 1> &beta_buffer, sycl::buffer<T, 1> &output_buffer,
    std::size_t row_count, std::size_t row_size, T epsilon,
    std::size_t work_group_size, sycl::queue Q) {
  std::size_t global_size = row_count * work_group_size;
  return Q.submit([&](sycl::handler &cgh) {
    auto input =
        input_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto gamma =
        gamma_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto beta = beta_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto output =
        output_buffer.template get_access<sycl::access::mode::write>(cgh);
    sycl::local_accessor<T, 1> local_values(sycl::range<1>{work_group_size},
                                            cgh);
    cgh.parallel_for(
        sycl::nd_range<1>{global_size, work_group_size},
        [=](sycl::nd_item<1> item) {
          std::size_t row = item.get_group(0);
          std::size_t local_id = item.get_local_id(0);
          std::size_t row_offset = row * row_size;
          T local_sum = T(0);
          for (std::size_t column = local_id; column < row_size;
               column += work_group_size) {
            local_sum += input[row_offset + column];
          }
          local_values[local_id] = local_sum;
          item.barrier(sycl::access::fence_space::local_space);

          for (std::size_t stride = work_group_size / 2; stride > 0;
               stride /= 2) {
            if (local_id < stride) {
              local_values[local_id] += local_values[local_id + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          T mean = local_values[0] / static_cast<T>(row_size);

          T local_variance = T(0);
          for (std::size_t column = local_id; column < row_size;
               column += work_group_size) {
            T difference = input[row_offset + column] - mean;
            local_variance += difference * difference;
          }
          local_values[local_id] = local_variance;
          item.barrier(sycl::access::fence_space::local_space);

          for (std::size_t stride = work_group_size / 2; stride > 0;
               stride /= 2) {
            if (local_id < stride) {
              local_values[local_id] += local_values[local_id + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          T inverse_standard_deviation =
              sycl::rsqrt(local_values[0] / static_cast<T>(row_size) + epsilon);

          for (std::size_t column = local_id; column < row_size;
               column += work_group_size) {
            T normalized = (input[row_offset + column] - mean) *
                           inverse_standard_deviation;
            output[row_offset + column] =
                normalized * gamma[column] + beta[column];
          }
        });
  });
}

sycl::event submit_group_norm_usm(
    const float *input_data, const float *weight_data, const float *bias_data,
    float *output_data, std::size_t channel_count, std::size_t position_count,
    std::size_t num_groups, std::size_t channels_per_group,
    std::size_t values_per_group, float epsilon, std::size_t work_group_size,
    std::size_t group_count, sycl::queue &Q) {
  std::size_t global_size = group_count * work_group_size;
  return Q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> local_values(sycl::range<1>{work_group_size},
                                                cgh);
    cgh.parallel_for(
        sycl::nd_range<1>{global_size, work_group_size},
        [=](sycl::nd_item<1> item) {
          std::size_t group_id = item.get_group(0);
          std::size_t local_id = item.get_local_id(0);
          std::size_t batch = group_id / num_groups;
          std::size_t group = group_id % num_groups;
          std::size_t group_offset =
              batch * channel_count * position_count + group * values_per_group;

          float local_sum = 0.0f;
          for (std::size_t index = local_id; index < values_per_group;
               index += work_group_size) {
            local_sum += input_data[group_offset + index];
          }
          local_values[local_id] = local_sum;
          item.barrier(sycl::access::fence_space::local_space);

          for (std::size_t stride = work_group_size / 2; stride > 0;
               stride /= 2) {
            if (local_id < stride) {
              local_values[local_id] += local_values[local_id + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          float mean = local_values[0] / static_cast<float>(values_per_group);
          item.barrier(sycl::access::fence_space::local_space);

          float local_variance = 0.0f;
          for (std::size_t index = local_id; index < values_per_group;
               index += work_group_size) {
            float difference = input_data[group_offset + index] - mean;
            local_variance += difference * difference;
          }
          local_values[local_id] = local_variance;
          item.barrier(sycl::access::fence_space::local_space);

          for (std::size_t stride = work_group_size / 2; stride > 0;
               stride /= 2) {
            if (local_id < stride) {
              local_values[local_id] += local_values[local_id + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          float inverse_standard_deviation = sycl::rsqrt(
              local_values[0] / static_cast<float>(values_per_group) + epsilon);

          for (std::size_t index = local_id; index < values_per_group;
               index += work_group_size) {
            std::size_t channel =
                group * channels_per_group + index / position_count;
            float normalized = (input_data[group_offset + index] - mean) *
                               inverse_standard_deviation;
            output_data[group_offset + index] =
                normalized * weight_data[channel] + bias_data[channel];
          }
        });
  });
}

sycl::event submit_group_norm_buffer(
    sycl::buffer<float, 1> &input_buffer, sycl::buffer<float, 1> &weight_buffer,
    sycl::buffer<float, 1> &bias_buffer, sycl::buffer<float, 1> &output_buffer,
    std::size_t channel_count, std::size_t position_count,
    std::size_t num_groups, std::size_t channels_per_group,
    std::size_t values_per_group, float epsilon, std::size_t work_group_size,
    std::size_t group_count, sycl::queue &Q) {
  std::size_t global_size = group_count * work_group_size;
  return Q.submit([&](sycl::handler &cgh) {
    auto input =
        input_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto weight =
        weight_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto bias = bias_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto output =
        output_buffer.template get_access<sycl::access::mode::write>(cgh);
    sycl::local_accessor<float, 1> local_values(sycl::range<1>{work_group_size},
                                                cgh);
    cgh.parallel_for(
        sycl::nd_range<1>{global_size, work_group_size},
        [=](sycl::nd_item<1> item) {
          std::size_t group_id = item.get_group(0);
          std::size_t local_id = item.get_local_id(0);
          std::size_t batch = group_id / num_groups;
          std::size_t group = group_id % num_groups;
          std::size_t group_offset =
              batch * channel_count * position_count + group * values_per_group;

          float local_sum = 0.0f;
          for (std::size_t index = local_id; index < values_per_group;
               index += work_group_size) {
            local_sum += input[group_offset + index];
          }
          local_values[local_id] = local_sum;
          item.barrier(sycl::access::fence_space::local_space);

          for (std::size_t stride = work_group_size / 2; stride > 0;
               stride /= 2) {
            if (local_id < stride) {
              local_values[local_id] += local_values[local_id + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          float mean = local_values[0] / static_cast<float>(values_per_group);
          item.barrier(sycl::access::fence_space::local_space);

          float local_variance = 0.0f;
          for (std::size_t index = local_id; index < values_per_group;
               index += work_group_size) {
            float difference = input[group_offset + index] - mean;
            local_variance += difference * difference;
          }
          local_values[local_id] = local_variance;
          item.barrier(sycl::access::fence_space::local_space);

          for (std::size_t stride = work_group_size / 2; stride > 0;
               stride /= 2) {
            if (local_id < stride) {
              local_values[local_id] += local_values[local_id + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          float inverse_standard_deviation = sycl::rsqrt(
              local_values[0] / static_cast<float>(values_per_group) + epsilon);

          for (std::size_t index = local_id; index < values_per_group;
               index += work_group_size) {
            std::size_t channel =
                group * channels_per_group + index / position_count;
            float normalized = (input[group_offset + index] - mean) *
                               inverse_standard_deviation;
            output[group_offset + index] =
                normalized * weight[channel] + bias[channel];
          }
        });
  });
}

template <typename T>
Tensor<T> layer_norm(const Tensor<T> &input, const Tensor<T> &gamma,
                     const Tensor<T> &beta, T epsilon, sycl::queue Q,
                     sycl::event *kernel_event) {
  if (input.getRank() == 0) {
    throw std::invalid_argument(
        "LayerNorm requires an input with at least one dimension");
  }
  std::size_t row_size = input.getShape().back();
  if (row_size == 0) {
    throw std::invalid_argument("LayerNorm final dimension cannot be zero");
  }
  if (gamma.getRank() != 1 || beta.getRank() != 1 ||
      gamma.getSize() != row_size || beta.getSize() != row_size) {
    throw std::invalid_argument(
        "LayerNorm gamma and beta must match the final input dimension");
  }
  if (epsilon <= T(0)) {
    throw std::invalid_argument("LayerNorm epsilon must be positive");
  }
  if (input.is_device() != gamma.is_device() ||
      input.is_device() != beta.is_device()) {
    throw std::invalid_argument(
        "LayerNorm tensors must use the same memory location");
  }
  if (input.is_device() &&
      (!input.is_accessible_from(Q) || !gamma.is_accessible_from(Q) ||
       !beta.is_accessible_from(Q))) {
    throw std::invalid_argument(
        "LayerNorm queue cannot access the input tensors");
  }

  const std::vector<std::size_t> &shape = input.getShape();
  std::size_t row_count = input.getSize() / row_size;
  std::size_t work_group_size =
      details::layer_norm_work_group_size(row_size, Q);
  if (input.is_device()) {
    Tensor<T> output(shape, Q);
    if (row_count == 0) {
      return output;
    }
    sycl::event event = submit_layer_norm_usm(
        input.device_data(), gamma.device_data(), beta.device_data(),
        output.device_data(), row_count, row_size, epsilon, work_group_size, Q);
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
    event.wait();
    return output;
  }

  Tensor<T> output(shape);
  if (row_count == 0) {
    return output;
  }
  {
    sycl::buffer<T, 1> input_buffer = input.to_sycl_buffer();
    sycl::buffer<T, 1> gamma_buffer = gamma.to_sycl_buffer();
    sycl::buffer<T, 1> beta_buffer = beta.to_sycl_buffer();
    sycl::buffer<T, 1> output_buffer = output.to_sycl_buffer();
    sycl::event event = submit_layer_norm_buffer(
        input_buffer, gamma_buffer, beta_buffer, output_buffer, row_count,
        row_size, epsilon, work_group_size, Q);
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
  }
  return output;
}

ftensor group_norm(const ftensor &input, const ftensor &weight,
                   const ftensor &bias, int num_groups, float epsilon,
                   sycl::queue &Q) {
  if (input.getRank() != 3) {
    throw std::invalid_argument("GroupNorm input must have shape [B, C, N]");
  }
  if (num_groups <= 0) {
    throw std::invalid_argument("GroupNorm group count must be positive");
  }
  if (epsilon <= 0.0f) {
    throw std::invalid_argument("GroupNorm epsilon must be positive");
  }

  const std::vector<std::size_t> &shape = input.getShape();
  std::size_t batch_count = shape[0];
  std::size_t channel_count = shape[1];
  std::size_t position_count = shape[2];
  std::size_t group_number = static_cast<std::size_t>(num_groups);
  if (channel_count == 0 || position_count == 0) {
    throw std::invalid_argument(
        "GroupNorm channel and position dimensions cannot be zero");
  }
  if (channel_count % group_number != 0) {
    throw std::invalid_argument(
        "GroupNorm channel count must be divisible by the group count");
  }
  if (weight.getRank() != 1 || bias.getRank() != 1 ||
      weight.getSize() != channel_count || bias.getSize() != channel_count) {
    throw std::invalid_argument(
        "GroupNorm weight and bias must match the channel dimension");
  }
  if (input.is_device() != weight.is_device() ||
      input.is_device() != bias.is_device()) {
    throw std::invalid_argument(
        "GroupNorm tensors must use the same memory location");
  }
  if (input.is_device() &&
      (!input.is_accessible_from(Q) || !weight.is_accessible_from(Q) ||
       !bias.is_accessible_from(Q))) {
    throw std::invalid_argument(
        "GroupNorm queue cannot access the input tensors");
  }

  std::size_t channels_per_group = channel_count / group_number;
  std::size_t values_per_group = channels_per_group * position_count;
  std::size_t group_count = batch_count * group_number;
  std::size_t work_group_size =
      details::group_norm_work_group_size(values_per_group, Q);

  if (input.is_device()) {
    ftensor output(shape, Q);
    if (group_count == 0) {
      return output;
    }
    sycl::event event = submit_group_norm_usm(
        input.device_data(), weight.device_data(), bias.device_data(),
        output.device_data(), channel_count, position_count, group_number,
        channels_per_group, values_per_group, epsilon, work_group_size,
        group_count, Q);
    event.wait();
    return output;
  }

  ftensor output(shape);
  if (group_count == 0) {
    return output;
  }
  {
    sycl::buffer<float, 1> input_buffer = input.to_sycl_buffer();
    sycl::buffer<float, 1> weight_buffer = weight.to_sycl_buffer();
    sycl::buffer<float, 1> bias_buffer = bias.to_sycl_buffer();
    sycl::buffer<float, 1> output_buffer = output.to_sycl_buffer();
    submit_group_norm_buffer(input_buffer, weight_buffer, bias_buffer,
                             output_buffer, channel_count, position_count,
                             group_number, channels_per_group, values_per_group,
                             epsilon, work_group_size, group_count, Q);
  }
  return output;
}

template Tensor<double> layer_norm(const Tensor<double> &,
                                   const Tensor<double> &,
                                   const Tensor<double> &, double, sycl::queue,
                                   sycl::event *);
template Tensor<float> layer_norm(const Tensor<float> &, const Tensor<float> &,
                                  const Tensor<float> &, float, sycl::queue,
                                  sycl::event *);
} // namespace flib::operations
