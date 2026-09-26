#include <funlib/funlib.hpp>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

bool compare(const std::string &name, const flib::Tensor<float> &output,
             const std::vector<std::size_t> &shape,
             const std::vector<float> &expected, sycl::queue Q) {
  if (output.getShape() != shape) {
    std::cerr << name << " produced the wrong shape" << std::endl;
    return false;
  }
  std::vector<float> actual = output.to_host(Q);
  if (actual != expected) {
    std::cerr << name << " produced incorrect values" << std::endl;
    return false;
  }
  return true;
}

bool checkHost(sycl::queue Q) {
  flib::Tensor<float> left({2, 1, 2});
  flib::Tensor<float> right({2, 2, 2});
  for (std::size_t i = 0; i < left.getSize(); i++) {
    left[i] = static_cast<float>(i + 1);
  }
  for (std::size_t i = 0; i < right.getSize(); i++) {
    right[i] = static_cast<float>(i + 10);
  }
  auto output = flib::tensor_operations::concatenate(left, right, 1, Q);
  return compare("Host concatenate", output, {2, 3, 2},
                 {1.0f, 2.0f, 10.0f, 11.0f, 12.0f, 13.0f, 3.0f, 4.0f, 14.0f,
                  15.0f, 16.0f, 17.0f},
                 Q);
}

bool checkDevice(sycl::queue Q) {
  const std::vector<float> class_token{100.0f, 101.0f, 102.0f, 103.0f};
  const std::vector<float> patch_tokens{1.0f, 2.0f, 3.0f, 4.0f,
                                        5.0f, 6.0f, 7.0f, 8.0f};
  flib::Tensor<float> left({1, 4}, Q);
  flib::Tensor<float> right({2, 4}, Q);
  left.copy_from(class_token.data(), Q).wait();
  right.copy_from(patch_tokens.data(), Q).wait();
  sycl::event kernel_event;
  auto output =
      flib::tensor_operations::concatenate(left, right, 0, Q, &kernel_event);
  kernel_event.wait();
  return compare("Device concatenate", output, {3, 4},
                 {100.0f, 101.0f, 102.0f, 103.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                  6.0f, 7.0f, 8.0f},
                 Q);
}

bool checkInvalidShape(sycl::queue Q) {
  try {
    flib::Tensor<float> left({1, 4});
    flib::Tensor<float> right({2, 5});
    flib::tensor_operations::concatenate(left, right, 0, Q);
    std::cerr << "Concatenate accepted incompatible shapes" << std::endl;
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

  if (!checkHost(Q) || !checkDevice(Q) || !checkInvalidShape(Q)) {
    return 1;
  }
  std::cout << "All concatenate tests passed" << std::endl;
  return 0;
}
