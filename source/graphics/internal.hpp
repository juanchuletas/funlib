#if !defined(_FUNLIB_GRAPHICS_INTERNAL_HPP)
#define _FUNLIB_GRAPHICS_INTERNAL_HPP
#include <funlib/Tensor/tensor.hpp>
#include <funlib/graphics/marching_cubes.hpp>
#include <array>
#include <cstddef>

namespace flib::graphics::marching_cubes::detail {

    struct Grid {
        std::size_t nx;
        std::size_t ny;
        std::size_t nz;
        float isovalue;
        std::array<float, 3> bounds_min;
        std::array<float, 3> bounds_max;
    };

    struct CellData {
        flib::ui8tensor cube_indices;
        flib::ui32tensor triangle_counts;
    };

    CellData classify_cells(const flib::ftensor& density,
                            const Grid& grid,
                            sycl::queue& queue);
    struct ScanResult {
        flib::ui32tensor offsets;
        std::size_t total_triangles;
    };

    ScanResult exclusive_scan(const flib::ui32tensor& counts,
                              sycl::queue& queue);
    flib::graphics::marching_cubes::Mesh allocate_mesh(
        std::size_t triangle_count, sycl::queue& queue);
    void generate_triangles(const flib::ftensor& density, const Grid& grid,
                            const flib::ui8tensor& cube_indices,
                            const flib::ui32tensor& offsets,
                            flib::graphics::marching_cubes::Mesh& mesh,
                            sycl::queue& queue);

} // namespace flib::graphics::marching_cubes::detail

#endif // _FUNLIB_GRAPHICS_INTERNAL_HPP
