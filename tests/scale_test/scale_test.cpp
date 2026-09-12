#include <funlib/funlib.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

bool checkValues(const flib::Tensor<float> &tensor,
                 const std::vector<float> &expected, sycl::queue Q) {
  const std::vector<std::size_t> expected_shape{2, 3, 4};
  if (tensor.getShape() != expected_shape) {
    std::cerr << "Scale produced the wrong tensor shape" << std::endl;
    return false;
  }

  std::vector<float> actual = tensor.to_host(Q);
  for (std::size_t i = 0; i < actual.size(); i++) {
    if (actual[i] != expected[i]) {
      std::cerr << "Scale mismatch at index " << i << std::endl;
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

  flib::Tensor<float> hostInput({2, 3, 4});
  std::vector<float> expected(hostInput.getSize());
  for (std::size_t i = 0; i < hostInput.getSize(); i++) {
    hostInput[i] = static_cast<float>(static_cast<int>(i) - 12);
    expected[i] = hostInput[i] * 0.125f;
  }

  flib::Tensor<float> hostOutput =
      flib::operations::scale(hostInput, 0.125f, Q);
  if (!checkValues(hostOutput, expected, Q)) {
    return 1;
  }

  std::vector<float> input_data = hostInput.to_host(Q);
  flib::Tensor<float> deviceInput({2, 3, 4}, Q);
  deviceInput.copy_from(input_data.data(), Q).wait();
  sycl::event kernel_event;
  flib::Tensor<float> deviceOutput =
      flib::operations::scale(deviceInput, 0.125f, Q, &kernel_event);
  kernel_event.wait();
  if (!checkValues(deviceOutput, expected, Q)) {
    return 1;
  }

  std::cout << "All scale tests passed" << std::endl;
  return 0;
}
