#include <funlib/funlib.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

bool compare_values(const std::vector<float> &actual,
                    const std::vector<float> &expected, float tolerance) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0; index < actual.size(); index++) {
    if (std::fabs(actual[index] - expected[index]) > tolerance) {
      std::cerr << "Value " << index << " differs: " << actual[index]
                << " expected " << expected[index] << std::endl;
      return false;
    }
  }
  return true;
}

bool check_position_slices(sycl::queue Q) {
  // The first token is the CLS position. The remaining tokens form the grid.
  const std::vector<float> positions{100.0f, 101.0f, 1.0f, 2.0f,
                                     3.0f,   4.0f,   5.0f, 6.0f};
  flib::ftensor input({1, 4, 2}, Q);
  input.copy_from(positions.data(), Q).wait();

  auto cls = flib::tensor_operations::slice(input, 1, 0, 1, Q);
  auto patches = flib::tensor_operations::slice(input, 1, 1, 4, Q);

  if (cls.getShape() != std::vector<std::size_t>({1, 1, 2}) ||
      patches.getShape() != std::vector<std::size_t>({1, 3, 2})) {
    std::cerr << "Slice produced an incorrect shape" << std::endl;
    return false;
  }
  return compare_values(cls.to_host(Q), {100.0f, 101.0f}, 0.0f) &&
         compare_values(patches.to_host(Q),
                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 0.0f);
}

bool check_bicubic_reference(sycl::queue Q) {
  // This reference uses bicubic interpolation with align_corners=false.
  const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> expected{
      0.68359375f, 1.015625f,  1.5625f,     1.89453125f,
      1.34765625f, 1.6796875f, 2.2265625f,  2.55859375f,
      2.44140625f, 2.7734375f, 3.3203125f,  3.65234375f,
      3.10546875f, 3.4375f,    3.984375f,   4.31640625f};

  flib::ftensor input({1, 1, 2, 2}, Q);
  input.copy_from(input_values.data(), Q).wait();
  auto output = flib::operations::interpolate_bicubic(input, 4, 4, Q);

  if (output.getShape() != std::vector<std::size_t>({1, 1, 4, 4})) {
    std::cerr << "Bicubic interpolation produced an incorrect shape"
              << std::endl;
    return false;
  }
  return compare_values(output.to_host(Q), expected, 1e-5f);
}

bool check_position_grid_shape(sycl::queue Q) {
  // A 14 by 14 learned grid becomes the 32 by 32 grid used by a 512 image.
  std::vector<float> values(768 * 14 * 14, 2.0f);
  flib::ftensor input({1, 768, 14, 14}, Q);
  input.copy_from(values.data(), Q).wait();
  auto output = flib::operations::interpolate_bicubic(input, 32, 32, Q);

  if (output.getShape() != std::vector<std::size_t>({1, 768, 32, 32})) {
    std::cerr << "Position grid produced an incorrect shape" << std::endl;
    return false;
  }
  std::vector<float> result = output.to_host(Q);
  for (float value : result) {
    if (std::fabs(value - 2.0f) > 1e-5f) {
      std::cerr << "Bicubic interpolation changed a constant grid"
                << std::endl;
      return false;
    }
  }
  return true;
}

bool check_invalid_inputs(sycl::queue Q) {
  try {
    flib::ftensor input({1, 4, 2}, Q);
    flib::tensor_operations::slice(input, 1, 3, 5, Q);
    std::cerr << "Slice accepted an invalid range" << std::endl;
    return false;
  } catch (const std::invalid_argument &) {
  }

  try {
    flib::ftensor input({2, 2}, Q);
    flib::operations::interpolate_bicubic(input, 4, 4, Q);
    std::cerr << "Bicubic interpolation accepted an invalid rank"
              << std::endl;
    return false;
  } catch (const std::invalid_argument &) {
  }
  return true;
}

int main() {
  flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                     flib::vendor::NVIDIA, flib::backend::CUDA,
                                     true);
  sycl::queue Q = flib::sycl_handler::get_queue("cuda");
  flib::sycl_handler::get_device_info("cuda");

  if (!check_position_slices(Q) || !check_bicubic_reference(Q) ||
      !check_position_grid_shape(Q) || !check_invalid_inputs(Q)) {
    return 1;
  }

  std::cout << "All image interpolation tests passed" << std::endl;
  return 0;
}
