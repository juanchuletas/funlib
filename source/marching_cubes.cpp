#include<funlib/graphics/marching_cubes.hpp>
#include "graphics/internal.hpp"
#include <stdexcept>

namespace flib::graphics::marching_cubes {

    Mesh generate_mesh(const flib::ftensor& density,
                       const Settings& settings,
                       sycl::queue& queue)
    {
        if (density.getRank() != 3 || !density.is_device() ||
            !density.is_accessible_from(queue)) {
            throw std::invalid_argument(
                "Marching cubes requires a 3D density tensor accessible from the queue");
        }
        const std::vector<std::size_t>& shape = density.getShape();
        if (shape[0] < 2 || shape[1] < 2 || shape[2] < 2) {
            throw std::invalid_argument(
                "Marching cubes requires at least two samples per dimension");
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (!(settings.bounds_min[axis] < settings.bounds_max[axis])) {
                throw std::invalid_argument(
                    "Marching cubes bounds must be strictly increasing");
            }
        }
        detail::Grid grid{
            shape[2], shape[1], shape[0], settings.isovalue,
            settings.bounds_min, settings.bounds_max};
        detail::CellData cells = detail::classify_cells(density, grid, queue);
        detail::ScanResult scan = detail::exclusive_scan(cells.triangle_counts, queue);
        Mesh mesh = detail::allocate_mesh(scan.total_triangles, queue);
        detail::generate_triangles(density, grid, cells.cube_indices,
                                   scan.offsets, mesh, queue);
        return mesh;
    }

} // namespace flib::graphics::marching_cubes
