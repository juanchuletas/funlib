#include <funlib/operations/sampling/triplane_sampling.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace flib::operations {

template <typename T>
Tensor<T> triplane_sample(const Tensor<T> &triplane,
                          const Tensor<T> &positions, sycl::queue Q,
                          sycl::event *kernel_event) {
  if (triplane.getRank() != 4 || triplane.getShape()[0] != 3) {
    throw std::invalid_argument(
        "Triplane sampler requires triplane shape [3, C, H, W]");
  }
  if (positions.getRank() != 2 || positions.getShape()[1] != 3) {
    throw std::invalid_argument(
        "Triplane sampler requires position shape [N, 3]");
  }

  const std::vector<std::size_t> &triplane_shape = triplane.getShape();
  const std::vector<std::size_t> &position_shape = positions.getShape();
  std::size_t channel_count = triplane_shape[1];
  std::size_t height = triplane_shape[2];
  std::size_t width = triplane_shape[3];
  std::size_t point_count = position_shape[0];
  if (channel_count == 0 || height == 0 || width == 0) {
    throw std::invalid_argument(
        "Triplane sampler channel and spatial dimensions cannot be zero");
  }
  if (!triplane.is_device() || !positions.is_device()) {
    throw std::invalid_argument("Triplane sampler requires device tensors");
  }
  if (!triplane.is_accessible_from(Q) || !positions.is_accessible_from(Q)) {
    throw std::invalid_argument(
        "Triplane sampler queue cannot access the input tensors");
  }

  std::size_t features_per_point = 3 * channel_count;
  std::size_t output_size = point_count * features_per_point;
  Tensor<T> output({point_count, features_per_point}, Q);
  if (output_size == 0) {
    return output;
  }

  const T *triplane_data = triplane.device_data();
  const T *position_data = positions.device_data();
  T *output_data = output.device_data();
  sycl::event event = Q.submit([&](sycl::handler &cgh) {
    // One work item interpolates one channel from one plane for one point.
    cgh.parallel_for(sycl::range<1>{output_size}, [=](sycl::item<1> item) {
      std::size_t output_index = item.get_id(0);
      std::size_t point = output_index / features_per_point;
      std::size_t feature = output_index % features_per_point;
      std::size_t plane = feature / channel_count;
      std::size_t channel = feature % channel_count;

      T x = position_data[point * 3];
      T y = position_data[point * 3 + 1];
      T z = position_data[point * 3 + 2];
      T horizontal;
      T vertical;
      if (plane == 0) {
        // The first plane samples the XY coordinates.
        horizontal = x;
        vertical = y;
      } else if (plane == 1) {
        // The second plane samples the XZ coordinates.
        horizontal = x;
        vertical = z;
      } else {
        // The third plane samples the YZ coordinates.
        horizontal = y;
        vertical = z;
      }

      // Convert normalized coordinates using align_corners=false.
      T image_x =
          ((horizontal + T(1)) * static_cast<T>(width) - T(1)) / T(2);
      T image_y =
          ((vertical + T(1)) * static_cast<T>(height) - T(1)) / T(2);
      std::ptrdiff_t x0 =
          static_cast<std::ptrdiff_t>(sycl::floor(image_x));
      std::ptrdiff_t y0 =
          static_cast<std::ptrdiff_t>(sycl::floor(image_y));
      std::ptrdiff_t x1 = x0 + 1;
      std::ptrdiff_t y1 = y0 + 1;
      T x_weight = image_x - static_cast<T>(x0);
      T y_weight = image_y - static_cast<T>(y0);

      // Neighbors outside the plane use zero padding.
      T top_left = T(0);
      T top_right = T(0);
      T bottom_left = T(0);
      T bottom_right = T(0);
      if (x0 >= 0 && x0 < static_cast<std::ptrdiff_t>(width) && y0 >= 0 &&
          y0 < static_cast<std::ptrdiff_t>(height)) {
        std::size_t index =
            ((plane * channel_count + channel) * height +
             static_cast<std::size_t>(y0)) *
                width +
            static_cast<std::size_t>(x0);
        top_left = triplane_data[index];
      }
      if (x1 >= 0 && x1 < static_cast<std::ptrdiff_t>(width) && y0 >= 0 &&
          y0 < static_cast<std::ptrdiff_t>(height)) {
        std::size_t index =
            ((plane * channel_count + channel) * height +
             static_cast<std::size_t>(y0)) *
                width +
            static_cast<std::size_t>(x1);
        top_right = triplane_data[index];
      }
      if (x0 >= 0 && x0 < static_cast<std::ptrdiff_t>(width) && y1 >= 0 &&
          y1 < static_cast<std::ptrdiff_t>(height)) {
        std::size_t index =
            ((plane * channel_count + channel) * height +
             static_cast<std::size_t>(y1)) *
                width +
            static_cast<std::size_t>(x0);
        bottom_left = triplane_data[index];
      }
      if (x1 >= 0 && x1 < static_cast<std::ptrdiff_t>(width) && y1 >= 0 &&
          y1 < static_cast<std::ptrdiff_t>(height)) {
        std::size_t index =
            ((plane * channel_count + channel) * height +
             static_cast<std::size_t>(y1)) *
                width +
            static_cast<std::size_t>(x1);
        bottom_right = triplane_data[index];
      }

      output_data[output_index] =
          top_left * (T(1) - x_weight) * (T(1) - y_weight) +
          top_right * x_weight * (T(1) - y_weight) +
          bottom_left * (T(1) - x_weight) * y_weight +
          bottom_right * x_weight * y_weight;
    });
  });
  if (kernel_event != nullptr) {
    *kernel_event = event;
  }
  event.wait();
  return output;
}

template Tensor<double> triplane_sample(const Tensor<double> &,
                                        const Tensor<double> &, sycl::queue,
                                        sycl::event *);
template Tensor<float> triplane_sample(const Tensor<float> &,
                                       const Tensor<float> &, sycl::queue,
                                       sycl::event *);

} // namespace flib::operations
