#if !defined(_MARCHING_CUBES_FLIB_H)
#define _MARCHING_CUBES_FLIB_H
#include <funlib/Tensor/tensor.hpp>

namespace flib {
namespace graphics {

namespace marching_cubes {

struct Settings {
  float isovalue;
  std::array<float, 3> bounds_min;
  std::array<float, 3> bounds_max;
};
struct Mesh {

  flib::ftensor positions;
  flib::ftensor normals;
  flib::ftensor colors;
};

Mesh generate_mesh(const flib::ftensor &density,
                   const marching_cubes::Settings &settings,
                   sycl::queue &queue);

} // namespace marching_cubes

} // namespace graphics

} // namespace flib

#endif // _MARCHING_CUBES_FLIB_H
