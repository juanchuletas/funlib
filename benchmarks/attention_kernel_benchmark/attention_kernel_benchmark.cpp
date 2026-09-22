#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

struct AttentionShape {
  std::size_t batch_size;
  std::size_t token_count;
  std::size_t head_count;
  std::size_t head_size;
};

struct KernelMeasurements {
  double projections;
  double head_layout;
  double score_gemm;
  double scaled_softmax;
  double value_gemm;
  double join_heads;
  double output_gemm;
  double total;
};

double kernelMilliseconds(sycl::event event) {
  event.wait();
  std::uint64_t start =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  std::uint64_t end =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

flib::Tensor<float> deviceTensor(const std::vector<std::size_t> &shape,
                                 const std::vector<float> &values,
                                 sycl::queue queue) {
  flib::Tensor<float> tensor(shape, queue);
  tensor.copy_from(values.data(), queue).wait();
  return tensor;
}

KernelMeasurements measureKernels(const flib::Tensor<float> &input,
                                  const flib::Tensor<float> &weightsQ,
                                  const flib::Tensor<float> &weightsK,
                                  const flib::Tensor<float> &weightsV,
                                  const flib::Tensor<float> &weightsO,
                                  const AttentionShape &shape,
                                  sycl::queue queue) {
  sycl::event query_event;
  sycl::event key_event;
  sycl::event value_event;
  sycl::event query_layout_event;
  sycl::event key_layout_event;
  sycl::event value_layout_event;
  sycl::event score_event;
  sycl::event scaled_softmax_event;
  sycl::event value_gemm_event;
  sycl::event join_event;
  sycl::event output_event;
  std::size_t model_size = shape.head_count * shape.head_size;

  flib::Tensor<float> query =
      flib::tensor_operations::gemm(input, weightsQ, queue, &query_event);
  flib::Tensor<float> key =
      flib::tensor_operations::gemm(input, weightsK, queue, &key_event);
  flib::Tensor<float> value =
      flib::tensor_operations::gemm(input, weightsV, queue, &value_event);

  query.reshape(
      {shape.batch_size, shape.token_count, shape.head_count, shape.head_size});
  key.reshape(
      {shape.batch_size, shape.token_count, shape.head_count, shape.head_size});
  value.reshape(
      {shape.batch_size, shape.token_count, shape.head_count, shape.head_size});

  flib::Tensor<float> query_heads =
      flib::operations::split_heads(query, queue, &query_layout_event);
  flib::Tensor<float> key_heads =
      flib::operations::split_heads(key, queue, &key_layout_event);
  flib::Tensor<float> value_heads =
      flib::operations::split_heads(value, queue, &value_layout_event);

  flib::Tensor<float> scores = flib::tensor_operations::gemm_batched(
      query_heads, key_heads, queue, false, true, &score_event);
  flib::Tensor<float> probabilities = flib::operations::scaled_softmax(
      scores, 1.0f / std::sqrt(static_cast<float>(shape.head_size)), queue,
      &scaled_softmax_event);
  flib::Tensor<float> head_output = flib::tensor_operations::gemm_batched(
      probabilities, value_heads, queue, false, false, &value_gemm_event);
  flib::Tensor<float> joined_output =
      flib::operations::join_heads(head_output, queue, &join_event);
  joined_output.reshape({shape.batch_size, shape.token_count, model_size});
  flib::Tensor<float> output = flib::tensor_operations::gemm(
      joined_output, weightsO, queue, &output_event);
  (void)output;

  KernelMeasurements result{};
  result.projections = kernelMilliseconds(query_event) +
                       kernelMilliseconds(key_event) +
                       kernelMilliseconds(value_event);
  result.head_layout = kernelMilliseconds(query_layout_event) +
                       kernelMilliseconds(key_layout_event) +
                       kernelMilliseconds(value_layout_event);
  result.score_gemm = kernelMilliseconds(score_event);
  result.scaled_softmax = kernelMilliseconds(scaled_softmax_event);
  result.value_gemm = kernelMilliseconds(value_gemm_event);
  result.join_heads = kernelMilliseconds(join_event);
  result.output_gemm = kernelMilliseconds(output_event);
  result.total = result.projections + result.head_layout + result.score_gemm +
                 result.scaled_softmax + result.value_gemm + result.join_heads +
                 result.output_gemm;
  return result;
}

double median(const std::vector<KernelMeasurements> &measurements,
              double KernelMeasurements::*field) {
  std::vector<double> values;
  values.reserve(measurements.size());
  for (const KernelMeasurements &measurement : measurements) {
    values.push_back(measurement.*field);
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

KernelMeasurements benchmarkKernels(const AttentionShape &shape,
                                    sycl::queue queue) {
  constexpr std::size_t warmups = 2;
  constexpr std::size_t repetitions = 7;
  std::size_t model_size = shape.head_count * shape.head_size;
  std::vector<float> input_values(shape.batch_size * shape.token_count *
                                  model_size);
  std::vector<float> weight_values(model_size * model_size);

  for (std::size_t i = 0; i < input_values.size(); i++) {
    input_values[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f;
  }
  for (std::size_t i = 0; i < weight_values.size(); i++) {
    weight_values[i] =
        static_cast<float>(static_cast<int>((i * 3) % 19) - 9) / 32.0f;
  }

  flib::Tensor<float> input = deviceTensor(
      {shape.batch_size, shape.token_count, model_size}, input_values, queue);
  flib::Tensor<float> weightsQ =
      deviceTensor({model_size, model_size}, weight_values, queue);
  flib::Tensor<float> weightsK =
      deviceTensor({model_size, model_size}, weight_values, queue);
  flib::Tensor<float> weightsV =
      deviceTensor({model_size, model_size}, weight_values, queue);
  flib::Tensor<float> weightsO =
      deviceTensor({model_size, model_size}, weight_values, queue);

  for (std::size_t i = 0; i < warmups; i++) {
    measureKernels(input, weightsQ, weightsK, weightsV, weightsO, shape, queue);
  }

  std::vector<KernelMeasurements> measurements;
  measurements.reserve(repetitions);
  for (std::size_t i = 0; i < repetitions; i++) {
    measurements.push_back(measureKernels(input, weightsQ, weightsK, weightsV,
                                          weightsO, shape, queue));
  }

  return KernelMeasurements{
      median(measurements, &KernelMeasurements::projections),
      median(measurements, &KernelMeasurements::head_layout),
      median(measurements, &KernelMeasurements::score_gemm),
      median(measurements, &KernelMeasurements::scaled_softmax),
      median(measurements, &KernelMeasurements::value_gemm),
      median(measurements, &KernelMeasurements::join_heads),
      median(measurements, &KernelMeasurements::output_gemm),
      median(measurements, &KernelMeasurements::total)};
}

void printResult(const AttentionShape &shape,
                 const KernelMeasurements &measurements) {
  double tokens = static_cast<double>(shape.batch_size * shape.token_count);
  double kernel_tokens_per_second = tokens / (measurements.total / 1000.0);
  std::cout << std::setw(5) << shape.batch_size << std::setw(8)
            << shape.token_count << std::setw(8) << shape.head_count
            << std::setw(13) << measurements.projections << std::setw(13)
            << measurements.head_layout << std::setw(13)
            << measurements.score_gemm << std::setw(16)
            << measurements.scaled_softmax << std::setw(13)
            << measurements.value_gemm << std::setw(13)
            << measurements.join_heads << std::setw(13)
            << measurements.output_gemm << std::setw(13) << measurements.total
            << std::setw(15) << kernel_tokens_per_second << std::endl;
}

int main() {
  // flib::sycl_handler::register_queue("cuda", flib::device::GPU,
  //                                    flib::vendor::NVIDIA,
  //                                    flib::backend::CUDA, true);
  // sycl::queue queue = flib::sycl_handler::get_queue("cuda");
  // flib::sycl_handler::get_device_info("cuda");
  flib::sycl_handler::register_queue("intel", flib::device::GPU,
                                     flib::vendor::INTEL, flib::backend::OPENCL,
                                     true);
  sycl::queue queue = flib::sycl_handler::get_queue("intel");
  flib::sycl_handler::get_device_info("intel");
  const std::vector<AttentionShape> shapes{
      {1, 196, 8, 64},
      {1, 512, 8, 64},
      {2, 196, 8, 64},
  };

  std::cout << std::fixed << std::setprecision(3);
  std::cout << std::setw(5) << "B" << std::setw(8) << "Tokens" << std::setw(8)
            << "Heads" << std::setw(13) << "QKV GEMM" << std::setw(13)
            << "Head layout" << std::setw(13) << "QK GEMM" << std::setw(16)
            << "Scaled Softmax" << std::setw(13) << "PV GEMM" << std::setw(13)
            << "Join" << std::setw(13) << "Output GEMM" << std::setw(13)
            << "Kernel ms" << std::setw(15) << "Kernel tokens/s" << std::endl;

  for (const AttentionShape &shape : shapes) {
    printResult(shape, benchmarkKernels(shape, queue));
  }
  return 0;
}
