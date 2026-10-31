#include "internal.hpp"

namespace flib::graphics::marching_cubes::detail {

    flib::graphics::marching_cubes::Mesh allocate_mesh(
        std::size_t triangle_count, sycl::queue& queue) {
        const std::size_t vertex_count = triangle_count * 3;
        return {
            flib::ftensor({vertex_count, 3}, queue),
            flib::ftensor({vertex_count, 3}, queue)};
    }

} // namespace flib::graphics::marching_cubes::detail
