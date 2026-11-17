#include <cassert>
#include <funlib/graphics/marching_cubes.hpp>

int main() {
  sycl::queue queue;
  constexpr std::size_t n = 24;
  flib::ftensor density({n, n, n}, queue);
  float *values = density.device_data();
  queue
      .submit([=](sycl::handler &handler) {
        handler.parallel_for(sycl::range<1>(density.getSize()),
                             [=](sycl::item<1> item) {
                               const std::size_t id = item.get_id(0);
                               const std::size_t z = id / (n * n);
                               const std::size_t y = (id / n) % n;
                               const std::size_t x = id % n;
                               const float px = -1.0f + 2.0f * x / (n - 1);
                               const float py = -1.0f + 2.0f * y / (n - 1);
                               const float pz = -1.0f + 2.0f * z / (n - 1);
                               values[id] = px * px + py * py + pz * pz - 0.25f;
                             });
      })
      .wait();
  flib::graphics::marching_cubes::Settings settings{
      0.0f, {-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}};
  auto mesh =
      flib::graphics::marching_cubes::generate_mesh(density, settings, queue);
  assert(mesh.positions.getRank() == 2);
  assert(mesh.positions.getShape()[1] == 3);
  assert(mesh.normals.getShape() == mesh.positions.getShape());
  assert(mesh.positions.getShape()[0] > 0);
}
