#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

constexpr std::size_t batch_size = 1;
constexpr std::size_t token_count = 4;
constexpr std::size_t model_size = 8;
constexpr std::size_t head_count = 2;
constexpr std::size_t head_size = model_size / head_count;
constexpr std::size_t hidden_size = 16;
constexpr float epsilon = 1.0e-5f;

std::vector<float> linearReference(const std::vector<float> &input,
                                   const std::vector<float> &weights,
                                   std::size_t input_size,
                                   std::size_t output_size) {
  std::size_t row_count = input.size() / input_size;
  std::vector<float> output(row_count * output_size, 0.0f);
  for (std::size_t row = 0; row < row_count; row++) {
    for (std::size_t column = 0; column < output_size; column++) {
      for (std::size_t inner = 0; inner < input_size; inner++) {
        output[row * output_size + column] +=
            input[row * input_size + inner] *
            weights[inner * output_size + column];
      }
    }
  }
  return output;
}

std::vector<float> layerNormReference(const std::vector<float> &input,
                                      const std::vector<float> &gamma,
                                      const std::vector<float> &beta) {
  std::size_t row_count = input.size() / model_size;
  std::vector<float> output(input.size());
  for (std::size_t row = 0; row < row_count; row++) {
    std::size_t offset = row * model_size;
    float mean = 0.0f;
    for (std::size_t feature = 0; feature < model_size; feature++) {
      mean += input[offset + feature];
    }
    mean /= static_cast<float>(model_size);

    float variance = 0.0f;
    for (std::size_t feature = 0; feature < model_size; feature++) {
      float difference = input[offset + feature] - mean;
      variance += difference * difference;
    }
    variance /= static_cast<float>(model_size);
    float inverse_standard_deviation = 1.0f / std::sqrt(variance + epsilon);
    for (std::size_t feature = 0; feature < model_size; feature++) {
      output[offset + feature] = (input[offset + feature] - mean) *
                                     inverse_standard_deviation *
                                     gamma[feature] +
                                 beta[feature];
    }
  }
  return output;
}

std::size_t tokenIndex(std::size_t batch, std::size_t token, std::size_t head,
                       std::size_t feature) {
  return (batch * token_count + token) * model_size + head * head_size +
         feature;
}

std::vector<float> attentionReference(const std::vector<float> &query,
                                      const std::vector<float> &key,
                                      const std::vector<float> &value) {
  std::vector<float> output(query.size(), 0.0f);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  for (std::size_t batch = 0; batch < batch_size; batch++) {
    for (std::size_t head = 0; head < head_count; head++) {
      for (std::size_t query_token = 0; query_token < token_count;
           query_token++) {
        std::vector<float> scores(token_count, 0.0f);
        for (std::size_t key_token = 0; key_token < token_count; key_token++) {
          for (std::size_t feature = 0; feature < head_size; feature++) {
            scores[key_token] +=
                query[tokenIndex(batch, query_token, head, feature)] *
                key[tokenIndex(batch, key_token, head, feature)];
          }
          scores[key_token] *= scale;
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
          for (std::size_t key_token = 0; key_token < token_count;
               key_token++) {
            output[tokenIndex(batch, query_token, head, feature)] +=
                scores[key_token] *
                value[tokenIndex(batch, key_token, head, feature)];
          }
        }
      }
    }
  }
  return output;
}

std::vector<float> addReference(const std::vector<float> &left,
                                const std::vector<float> &right) {
  std::vector<float> output(left.size());
  for (std::size_t i = 0; i < output.size(); i++) {
    output[i] = left[i] + right[i];
  }
  return output;
}

std::vector<float> geluReference(const std::vector<float> &input) {
  std::vector<float> output(input.size());
  for (std::size_t i = 0; i < output.size(); i++) {
    output[i] =
        0.5f * input[i] * (1.0f + std::erf(input[i] * 0.70710678118654752440f));
  }
  return output;
}

std::vector<float> transformerReference(
    const std::vector<float> &input, const std::vector<float> &gamma1,
    const std::vector<float> &beta1, const std::vector<float> &weightsQ,
    const std::vector<float> &weightsK, const std::vector<float> &weightsV,
    const std::vector<float> &weightsO, const std::vector<float> &gamma2,
    const std::vector<float> &beta2, const std::vector<float> &weights1,
    const std::vector<float> &weights2) {
  std::vector<float> normalized1 = layerNormReference(input, gamma1, beta1);
  std::vector<float> query =
      linearReference(normalized1, weightsQ, model_size, model_size);
  std::vector<float> key =
      linearReference(normalized1, weightsK, model_size, model_size);
  std::vector<float> value =
      linearReference(normalized1, weightsV, model_size, model_size);
  std::vector<float> attention = attentionReference(query, key, value);
  std::vector<float> projected_attention =
      linearReference(attention, weightsO, model_size, model_size);
  std::vector<float> residual1 = addReference(input, projected_attention);
  std::vector<float> normalized2 = layerNormReference(residual1, gamma2, beta2);
  std::vector<float> hidden =
      linearReference(normalized2, weights1, model_size, hidden_size);
  hidden = geluReference(hidden);
  std::vector<float> feed_forward =
      linearReference(hidden, weights2, hidden_size, model_size);
  return addReference(residual1, feed_forward);
}

flib::Tensor<float> deviceTensor(const std::vector<std::size_t> &shape,
                                 const std::vector<float> &values,
                                 sycl::queue Q) {
  // This creates GPU memory and copies the CPU values into it.
  flib::Tensor<float> tensor(shape, Q);
  tensor.copy_from(values.data(), Q).wait();
  return tensor;
}

std::vector<float> makeValues(std::size_t size, std::size_t multiplier,
                              std::size_t modulus, float divisor) {
  // This creates fixed test values. These are not trained model weights.
  // The arguments change the pattern and keep the numbers small.
  std::vector<float> values(size);
  for (std::size_t i = 0; i < size; i++) {
    values[i] =
        static_cast<float>(static_cast<int>((i * multiplier + 1) % modulus) -
                           static_cast<int>(modulus / 2)) /
        divisor;
  }
  return values;
}

bool compareOutput(const flib::Tensor<float> &output,
                   const std::vector<float> &expected, sycl::queue Q) {
  const std::vector<std::size_t> expected_shape{batch_size, token_count,
                                                model_size};
  if (output.getShape() != expected_shape) {
    std::cerr << "Transformer block produced the wrong shape" << std::endl;
    return false;
  }

  std::vector<float> actual = output.to_host(Q);
  for (std::size_t i = 0; i < actual.size(); i++) {
    float tolerance = 5.0e-4f + 5.0e-4f * std::abs(expected[i]);
    if (!std::isfinite(actual[i]) ||
        std::abs(actual[i] - expected[i]) > tolerance) {
      std::cerr << "Transformer block mismatch at index " << i << std::endl;
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

  // Input contains four tokens. Each token has eight model features.
  std::vector<float> input =
      makeValues(batch_size * token_count * model_size, 3, 17, 12.0f);

  // Gamma changes the size of every normalized feature.
  // Beta moves every normalized feature by a learned amount.
  // Each LayerNorm has its own gamma and beta values.
  std::vector<float> gamma1 = makeValues(model_size, 2, 9, 12.0f);
  std::vector<float> beta1 = makeValues(model_size, 3, 11, 20.0f);
  std::vector<float> gamma2 = makeValues(model_size, 4, 13, 15.0f);
  std::vector<float> beta2 = makeValues(model_size, 5, 9, 24.0f);
  for (float &value : gamma1) {
    value += 1.0f;
  }
  for (float &value : gamma2) {
    value += 1.0f;
  }

  // These weights create query, key and value from the normalized input.
  std::vector<float> weightsQ =
      makeValues(model_size * model_size, 3, 13, 20.0f);
  std::vector<float> weightsK =
      makeValues(model_size * model_size, 5, 17, 22.0f);
  std::vector<float> weightsV =
      makeValues(model_size * model_size, 7, 19, 24.0f);

  // These weights move the attention result back to the model feature space.
  std::vector<float> weightsO =
      makeValues(model_size * model_size, 9, 23, 28.0f);

  // W1 expands each token from 8 to 16 features.
  // W2 reduces each token from 16 back to 8 features.
  std::vector<float> weights1 =
      makeValues(model_size * hidden_size, 11, 29, 32.0f);
  std::vector<float> weights2 =
      makeValues(hidden_size * model_size, 13, 31, 36.0f);

  // This runs the same transformer block on the CPU for comparison.
  std::vector<float> expected =
      transformerReference(input, gamma1, beta1, weightsQ, weightsK, weightsV,
                           weightsO, gamma2, beta2, weights1, weights2);

  // These tensors store the input and LayerNorm parameters on the GPU.
  flib::Tensor<float> input_tensor =
      deviceTensor({batch_size, token_count, model_size}, input, Q);
  flib::Tensor<float> gamma1_tensor = deviceTensor({model_size}, gamma1, Q);
  flib::Tensor<float> beta1_tensor = deviceTensor({model_size}, beta1, Q);
  flib::Tensor<float> gamma2_tensor = deviceTensor({model_size}, gamma2, Q);
  flib::Tensor<float> beta2_tensor = deviceTensor({model_size}, beta2, Q);

  // These tensors store the matrix weights on the GPU.
  flib::Tensor<float> query_weights =
      deviceTensor({model_size, model_size}, weightsQ, Q);
  flib::Tensor<float> key_weights =
      deviceTensor({model_size, model_size}, weightsK, Q);
  flib::Tensor<float> value_weights =
      deviceTensor({model_size, model_size}, weightsV, Q);
  flib::Tensor<float> output_weights =
      deviceTensor({model_size, model_size}, weightsO, Q);
  flib::Tensor<float> ffn_expand_weights =
      deviceTensor({model_size, hidden_size}, weights1, Q);
  flib::Tensor<float> ffn_reduce_weights =
      deviceTensor({hidden_size, model_size}, weights2, Q);

  // This normalizes every token. The tensor shape stays [1, 4, 8].
  flib::Tensor<float> normalized1 = flib::operations::layer_norm(
      input_tensor, gamma1_tensor, beta1_tensor, epsilon, Q);

  // These projections create Q, K and V. Each shape is [1, 4, 8].
  flib::Tensor<float> query =
      flib::tensor_operations::gemm(normalized1, query_weights, Q);
  flib::Tensor<float> key =
      flib::tensor_operations::gemm(normalized1, key_weights, Q);
  flib::Tensor<float> value =
      flib::tensor_operations::gemm(normalized1, value_weights, Q);

  // This separates eight model features into two heads of four features.
  query.reshape({batch_size, token_count, head_count, head_size});
  key.reshape({batch_size, token_count, head_count, head_size});
  value.reshape({batch_size, token_count, head_count, head_size});

  // Attention lets every token collect information from the other tokens.
  // It joins the heads and returns the shape [1, 4, 8].
  flib::Tensor<float> attention =
      flib::operations::scaled_dot_product_attention(query, key, value,
                                                     head_count, Q);
  flib::Tensor<float> projected_attention =
      flib::tensor_operations::gemm(attention, output_weights, Q);

  // This residual connection keeps the original input information.
  flib::Tensor<float> residual1 =
      flib::operations::add(input_tensor, projected_attention, Q);

  // This prepares the attention result for the feed forward network.
  flib::Tensor<float> normalized2 = flib::operations::layer_norm(
      residual1, gamma2_tensor, beta2_tensor, epsilon, Q);

  // This expands every token to 16 features and applies GELU.
  flib::Tensor<float> hidden =
      flib::tensor_operations::gemm(normalized2, ffn_expand_weights, Q);
  hidden = flib::operations::gelu(hidden, Q);

  // This reduces every token from 16 features back to 8 features.
  flib::Tensor<float> feed_forward =
      flib::tensor_operations::gemm(hidden, ffn_reduce_weights, Q);

  // This is the final residual connection of the transformer block.
  flib::Tensor<float> output =
      flib::operations::add(residual1, feed_forward, Q);

  if (!compareOutput(output, expected, Q)) {
    return 1;
  }
  std::cout << "Transformer block test passed" << std::endl;
  return 0;
}
