#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

constexpr std::size_t batch_size = 2;
constexpr std::size_t token_count = 3;
constexpr std::size_t head_count = 2;
constexpr std::size_t head_size = 2;
constexpr std::size_t model_size = head_count * head_size;

std::vector<float> projection(const std::vector<float> &input,
                              const std::vector<float> &weights) {
  std::vector<float> output(batch_size * token_count * model_size, 0.0f);
  for (std::size_t batch = 0; batch < batch_size; batch++) {
    for (std::size_t token = 0; token < token_count; token++) {
      for (std::size_t output_feature = 0; output_feature < model_size;
           output_feature++) {
        for (std::size_t input_feature = 0; input_feature < model_size;
             input_feature++) {
          output[(batch * token_count + token) * model_size + output_feature] +=
              input[(batch * token_count + token) * model_size +
                    input_feature] *
              weights[input_feature * model_size + output_feature];
        }
      }
    }
  }
  return output;
}

std::size_t projectedIndex(std::size_t batch, std::size_t token,
                           std::size_t head, std::size_t feature) {
  return (batch * token_count + token) * model_size + head * head_size +
         feature;
}

std::vector<float> attentionReference(const std::vector<float> &input,
                                      const std::vector<float> &weightsQ,
                                      const std::vector<float> &weightsK,
                                      const std::vector<float> &weightsV,
                                      const std::vector<float> &weightsO) {
  std::vector<float> Q = projection(input, weightsQ);
  std::vector<float> K = projection(input, weightsK);
  std::vector<float> V = projection(input, weightsV);
  std::vector<float> context(batch_size * token_count * model_size, 0.0f);
  float factor = 1.0f / std::sqrt(static_cast<float>(head_size));

  for (std::size_t batch = 0; batch < batch_size; batch++) {
    for (std::size_t head = 0; head < head_count; head++) {
      for (std::size_t query_token = 0; query_token < token_count;
           query_token++) {
        std::vector<float> scores(token_count, 0.0f);
        for (std::size_t key_token = 0; key_token < token_count; key_token++) {
          for (std::size_t feature = 0; feature < head_size; feature++) {
            scores[key_token] +=
                Q[projectedIndex(batch, query_token, head, feature)] *
                K[projectedIndex(batch, key_token, head, feature)];
          }
          scores[key_token] *= factor;
        }

        float maximum = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        for (float &score : scores) {
          score = std::exp(score - maximum);
          sum += score;
        }
        for (float &score : scores) {
          score /= sum;
        }

        for (std::size_t feature = 0; feature < head_size; feature++) {
          float value = 0.0f;
          for (std::size_t key_token = 0; key_token < token_count;
               key_token++) {
            value += scores[key_token] *
                     V[projectedIndex(batch, key_token, head, feature)];
          }
          context[projectedIndex(batch, query_token, head, feature)] = value;
        }
      }
    }
  }

  return projection(context, weightsO);
}

flib::Tensor<float> deviceTensor(const std::vector<std::size_t> &shape,
                                 const std::vector<float> &values,
                                 sycl::queue Q) {
  flib::Tensor<float> tensor(shape, Q);
  tensor.copy_from(values.data(), Q).wait();
  return tensor;
}

bool compareOutput(const flib::Tensor<float> &output,
                   const std::vector<float> &expected, sycl::queue Q) {
  const std::vector<std::size_t> expected_shape{batch_size, token_count,
                                                model_size};
  if (output.getShape() != expected_shape) {
    std::cerr << "Attention produced the wrong output shape" << std::endl;
    return false;
  }

  std::vector<float> actual = output.to_host(Q);
  for (std::size_t i = 0; i < actual.size(); i++) {
    float tolerance = 2.0e-4f + 2.0e-4f * std::abs(expected[i]);
    if (!std::isfinite(actual[i]) ||
        std::abs(actual[i] - expected[i]) > tolerance) {
      std::cerr << "Attention mismatch at index " << i << std::endl;
      std::cerr << "Expected: " << expected[i] << std::endl;
      std::cerr << "Actual: " << actual[i] << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                     flib::vendor::NVIDIA, flib::backend::CUDA,
                                     true);
  sycl::queue Q = flib::sycl_handler::get_queue("cuda");
  flib::sycl_handler::get_device_info("cuda");

  std::vector<float> input(batch_size * token_count * model_size);
  std::vector<float> weightsQ(model_size * model_size);
  std::vector<float> weightsK(model_size * model_size);
  std::vector<float> weightsV(model_size * model_size);
  std::vector<float> weightsO(model_size * model_size);
  for (std::size_t i = 0; i < input.size(); i++) {
    input[i] = static_cast<float>(static_cast<int>(i % 9) - 4) / 4.0f;
  }
  for (std::size_t i = 0; i < weightsQ.size(); i++) {
    weightsQ[i] =
        static_cast<float>(static_cast<int>((i * 3 + 1) % 7) - 3) / 5.0f;
    weightsK[i] =
        static_cast<float>(static_cast<int>((i * 5 + 2) % 9) - 4) / 6.0f;
    weightsV[i] =
        static_cast<float>(static_cast<int>((i * 7 + 3) % 11) - 5) / 7.0f;
    weightsO[i] =
        static_cast<float>(static_cast<int>((i * 2 + 4) % 8) - 4) / 5.0f;
  }

  std::vector<float> expected =
      attentionReference(input, weightsQ, weightsK, weightsV, weightsO);
  flib::Tensor<float> X =
      deviceTensor({batch_size, token_count, model_size}, input, Q);
  flib::Tensor<float> Wq = deviceTensor({model_size, model_size}, weightsQ, Q);
  flib::Tensor<float> Wk = deviceTensor({model_size, model_size}, weightsK, Q);
  flib::Tensor<float> Wv = deviceTensor({model_size, model_size}, weightsV, Q);
  flib::Tensor<float> Wo = deviceTensor({model_size, model_size}, weightsO, Q);

  flib::Tensor<float> projectedQ = flib::tensor_operations::gemm(X, Wq, Q);
  flib::Tensor<float> projectedK = flib::tensor_operations::gemm(X, Wk, Q);
  flib::Tensor<float> projectedV = flib::tensor_operations::gemm(X, Wv, Q);

  projectedQ.reshape({batch_size, token_count, head_count, head_size});
  projectedK.reshape({batch_size, token_count, head_count, head_size});
  projectedV.reshape({batch_size, token_count, head_count, head_size});

  flib::Tensor<float> Qheads =
      flib::tensor_operations::permute(projectedQ, {0, 2, 1, 3}, Q);
  flib::Tensor<float> Kheads =
      flib::tensor_operations::permute(projectedK, {0, 2, 1, 3}, Q);
  flib::Tensor<float> Vheads =
      flib::tensor_operations::permute(projectedV, {0, 2, 1, 3}, Q);

  flib::Tensor<float> scores =
      flib::tensor_operations::gemm_batched(Qheads, Kheads, Q, false, true);
  flib::Tensor<float> scaled_scores = flib::operations::scale(
      scores, 1.0f / std::sqrt(static_cast<float>(head_size)), Q);
  flib::Tensor<float> probabilities =
      flib::operations::softmax(scaled_scores, Q);
  flib::Tensor<float> head_output =
      flib::tensor_operations::gemm_batched(probabilities, Vheads, Q);
  flib::Tensor<float> joined_output =
      flib::tensor_operations::permute(head_output, {0, 2, 1, 3}, Q);
  joined_output.reshape({batch_size, token_count, model_size});
  flib::Tensor<float> output =
      flib::tensor_operations::gemm(joined_output, Wo, Q);

  if (!compareOutput(output, expected, Q)) {
    return 1;
  }

  std::cout << "Attention integration test passed" << std::endl;
  return 0;
}
