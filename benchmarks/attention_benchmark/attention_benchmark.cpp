#include <funlib/funlib.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

struct AttentionShape {
  std::size_t batch_size;
  std::size_t token_count;
  std::size_t head_count;
  std::size_t head_size;
};

struct AttentionMeasurements {
  double projections;
  double head_permutations;
  double score_gemm;
  double scale;
  double softmax;
  double value_gemm;
  double join_permutation;
  double output_gemm;
  double total;
};

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

flib::Tensor<float> deviceTensor(const std::vector<std::size_t> &shape,
                                 const std::vector<float> &values,
                                 sycl::queue queue) {
  flib::Tensor<float> tensor(shape, queue);
  tensor.copy_from(values.data(), queue).wait();
  return tensor;
}

AttentionMeasurements measureAttention(const flib::Tensor<float> &input,
                                       const flib::Tensor<float> &weightsQ,
                                       const flib::Tensor<float> &weightsK,
                                       const flib::Tensor<float> &weightsV,
                                       const flib::Tensor<float> &weightsO,
                                       const AttentionShape &shape,
                                       sycl::queue queue) {
  AttentionMeasurements result{};
  std::size_t model_size = shape.head_count * shape.head_size;
  Clock::time_point total_start = Clock::now();
  {
    Clock::time_point start = Clock::now();
    flib::Tensor<float> query =
        flib::tensor_operations::gemm(input, weightsQ, queue);
    flib::Tensor<float> key =
        flib::tensor_operations::gemm(input, weightsK, queue);
    flib::Tensor<float> value =
        flib::tensor_operations::gemm(input, weightsV, queue);
    result.projections = milliseconds(start, Clock::now());

    query.reshape({shape.batch_size, shape.token_count, shape.head_count,
                   shape.head_size});
    key.reshape({shape.batch_size, shape.token_count, shape.head_count,
                 shape.head_size});
    value.reshape({shape.batch_size, shape.token_count, shape.head_count,
                   shape.head_size});

    start = Clock::now();
    flib::Tensor<float> query_heads =
        flib::tensor_operations::permute(query, {0, 2, 1, 3}, queue);
    flib::Tensor<float> key_heads =
        flib::tensor_operations::permute(key, {0, 2, 1, 3}, queue);
    flib::Tensor<float> value_heads =
        flib::tensor_operations::permute(value, {0, 2, 1, 3}, queue);
    result.head_permutations = milliseconds(start, Clock::now());

    start = Clock::now();
    flib::Tensor<float> scores = flib::tensor_operations::gemm_batched(
        query_heads, key_heads, queue, false, true);
    result.score_gemm = milliseconds(start, Clock::now());

    start = Clock::now();
    flib::Tensor<float> scaled_scores = flib::operations::scale(
        scores, 1.0f / std::sqrt(static_cast<float>(shape.head_size)), queue);
    result.scale = milliseconds(start, Clock::now());

    start = Clock::now();
    flib::Tensor<float> probabilities =
        flib::operations::softmax(scaled_scores, queue);
    result.softmax = milliseconds(start, Clock::now());

    start = Clock::now();
    flib::Tensor<float> head_output = flib::tensor_operations::gemm_batched(
        probabilities, value_heads, queue);
    result.value_gemm = milliseconds(start, Clock::now());

    start = Clock::now();
    flib::Tensor<float> joined_output =
        flib::tensor_operations::permute(head_output, {0, 2, 1, 3}, queue);
    joined_output.reshape({shape.batch_size, shape.token_count, model_size});
    result.join_permutation = milliseconds(start, Clock::now());

    start = Clock::now();
    flib::Tensor<float> output =
        flib::tensor_operations::gemm(joined_output, weightsO, queue);
    result.output_gemm = milliseconds(start, Clock::now());
    (void)output;
  }
  result.total = milliseconds(total_start, Clock::now());
  return result;
}

double median(const std::vector<AttentionMeasurements> &measurements,
              double AttentionMeasurements::*field) {
  std::vector<double> values;
  values.reserve(measurements.size());
  for (const AttentionMeasurements &measurement : measurements) {
    values.push_back(measurement.*field);
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

AttentionMeasurements benchmarkAttention(const AttentionShape &shape,
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
    measureAttention(input, weightsQ, weightsK, weightsV, weightsO, shape,
                     queue);
  }

  std::vector<AttentionMeasurements> measurements;
  measurements.reserve(repetitions);
  for (std::size_t i = 0; i < repetitions; i++) {
    measurements.push_back(measureAttention(input, weightsQ, weightsK, weightsV,
                                            weightsO, shape, queue));
  }

  return AttentionMeasurements{
      median(measurements, &AttentionMeasurements::projections),
      median(measurements, &AttentionMeasurements::head_permutations),
      median(measurements, &AttentionMeasurements::score_gemm),
      median(measurements, &AttentionMeasurements::scale),
      median(measurements, &AttentionMeasurements::softmax),
      median(measurements, &AttentionMeasurements::value_gemm),
      median(measurements, &AttentionMeasurements::join_permutation),
      median(measurements, &AttentionMeasurements::output_gemm),
      median(measurements, &AttentionMeasurements::total)};
}

void printResult(const AttentionShape &shape,
                 const AttentionMeasurements &measurements) {
  double tokens = static_cast<double>(shape.batch_size * shape.token_count);
  double tokens_per_second = tokens / (measurements.total / 1000.0);
  std::cout << std::setw(5) << shape.batch_size << std::setw(8)
            << shape.token_count << std::setw(8) << shape.head_count
            << std::setw(13) << measurements.projections << std::setw(13)
            << measurements.head_permutations << std::setw(13)
            << measurements.score_gemm << std::setw(11) << measurements.scale
            << std::setw(11) << measurements.softmax << std::setw(13)
            << measurements.value_gemm << std::setw(13)
            << measurements.join_permutation << std::setw(13)
            << measurements.output_gemm << std::setw(13) << measurements.total
            << std::setw(15) << tokens_per_second << std::endl;
}

int main() {
  flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                     flib::vendor::NVIDIA, flib::backend::CUDA,
                                     true);
  sycl::queue queue = flib::sycl_handler::get_queue("cuda");
  flib::sycl_handler::get_device_info("cuda");

//    flib::sycl_handler::register_queue("intel", flib::device::GPU,
//                                      flib::vendor::INTEL, flib::backend::OPENCL,
//                                      true);
//   sycl::queue queue = flib::sycl_handler::get_queue("intel");
//   flib::sycl_handler::get_device_info("intel");
  const std::vector<AttentionShape> shapes{
      {1, 196, 8, 64},
      {1, 512, 8, 64},
      {2, 196, 8, 64},
  };

  std::cout << std::fixed << std::setprecision(3);
  std::cout << std::setw(5) << "B" << std::setw(8) << "Tokens" << std::setw(8)
            << "Heads" << std::setw(13) << "QKV GEMM" << std::setw(13)
            << "QKV permute" << std::setw(13) << "QK GEMM" << std::setw(11)
            << "Scale" << std::setw(11) << "Softmax" << std::setw(13)
            << "PV GEMM" << std::setw(13) << "Join" << std::setw(13)
            << "Output GEMM" << std::setw(13) << "Total ms" << std::setw(15)
            << "Tokens/s" << std::endl;

  for (const AttentionShape &shape : shapes) {
    printResult(shape, benchmarkAttention(shape, queue));
  }
  return 0;
}
