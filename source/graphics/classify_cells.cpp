#include "internal.hpp"
#include <funlib/graphics/mc_tables.hpp>
#include <cstdint>
#include <stdexcept>

namespace flib::graphics::marching_cubes::detail {

    CellData classify_cells(const flib::ftensor& density,
                            const Grid& grid,
                            sycl::queue& queue) {
        if (density.getRank() != 3 || density.getShape().size() != 3) {
            throw std::invalid_argument("Marching cubes requires a 3D density tensor");
        }
        if (!density.is_device() || !density.is_accessible_from(queue)) {
            throw std::invalid_argument(
                "Marching cubes requires density data accessible from the queue");
        }
        const std::vector<std::size_t>& shape = density.getShape();
        if (grid.nx < 2 || grid.ny < 2 || grid.nz < 2 ||
            shape[0] != grid.nz || shape[1] != grid.ny || shape[2] != grid.nx) {
            throw std::invalid_argument(
                "Marching cubes grid dimensions do not match the density tensor");
        }

        const std::size_t cell_count =
            (grid.nx - 1) * (grid.ny - 1) * (grid.nz - 1);
        CellData cells{
            flib::ui8tensor({cell_count}, queue),
            flib::ui32tensor({cell_count}, queue)};

        const float* density_data = density.device_data();
        std::uint8_t* cube_indices = cells.cube_indices.device_data();
        std::uint32_t* triangle_counts = cells.triangle_counts.device_data();
        const std::size_t nx = grid.nx;
        const std::size_t ny = grid.ny;
        const float isovalue = grid.isovalue;

        queue.submit([=](sycl::handler& handler) {
            handler.parallel_for(sycl::range<1>(cell_count), [=](sycl::item<1> item) {
                const std::size_t cell_id = item.get_id(0);
                const std::size_t ix = cell_id % (nx - 1);
                const std::size_t iy = (cell_id / (nx - 1)) % (ny - 1);
                const std::size_t iz = cell_id / ((nx - 1) * (ny - 1));

                const std::size_t base = ix + nx * (iy + ny * iz);
                const std::size_t row = nx * ny;
                const float values[8] = {
                    density_data[base], density_data[base + 1],
                    density_data[base + 1 + row], density_data[base + row],
                    density_data[base + nx], density_data[base + nx + 1],
                    density_data[base + nx + 1 + row], density_data[base + nx + row]};

                std::uint8_t cube_index = 0;
                for (int corner = 0; corner < 8; ++corner) {
                    if (values[corner] < isovalue) {
                        cube_index |= static_cast<std::uint8_t>(1u << corner);
                    }
                }

                std::uint32_t triangle_count = 0;
                for (int index = 0; triTable[cube_index][index] != -1; index += 3) {
                    ++triangle_count;
                }
                cube_indices[cell_id] = cube_index;
                triangle_counts[cell_id] = triangle_count;
            });
        }).wait();

        return cells;
    }

} // namespace flib::graphics::marching_cubes::detail
