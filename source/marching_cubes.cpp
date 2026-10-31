#include<funlib/graphics/marching_cubes.hpp>
#include "graphics/internal.hpp"

namespace flib::graphics::marching_cubes {

    Mesh generate_mesh(const flib::ftensor& density,
                       const Settings& settings,
                       sycl::queue& queue)
    {
        const std::vector<std::size_t>& shape = density.getShape();
        Grid grid{
            shape[2], shape[1], shape[0], settings.isovalue,
            settings.bounds_min, settings.bounds_max};
        CellData cells = detail::classify_cells(density, grid, queue);
        ScanResult scan = detail::exclusive_scan(cells.triangle_counts, queue);
        Mesh mesh = detail::allocate_mesh(scan.total_triangles, queue);
        detail::generate_triangles(density, grid, cells.cube_indices,
                                   scan.offsets, mesh, queue);
        return mesh;
    }

} // namespace flib::graphics::marching_cubes
