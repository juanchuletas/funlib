#include "internal.hpp"
#include <funlib/graphics/mc_tables.hpp>
#include <cmath>

namespace flib::graphics::marching_cubes::detail {

    void generate_triangles(const flib::ftensor& density, const Grid& grid,
                            const flib::ui8tensor& cube_indices,
                            const flib::ui32tensor& offsets,
                            flib::graphics::marching_cubes::Mesh& mesh,
                            sycl::queue& queue) {
        const float* values = density.device_data();
        const std::uint8_t* indices = cube_indices.device_data();
        const std::uint32_t* output_offsets = offsets.device_data();
        float* positions = mesh.positions.device_data();
        float* normals = mesh.normals.device_data();
        const std::size_t nx = grid.nx;
        const std::size_t ny = grid.ny;
        const float iso = grid.isovalue;
        const float sx = (grid.bounds_max[0] - grid.bounds_min[0]) / (nx - 1);
        const float sy = (grid.bounds_max[1] - grid.bounds_min[1]) / (ny - 1);
        const float sz = (grid.bounds_max[2] - grid.bounds_min[2]) / (grid.nz - 1);

        queue.submit([=](sycl::handler& handler) {
            handler.parallel_for(sycl::range<1>(cube_indices.getSize()), [=](sycl::item<1> item) {
                const std::size_t cell = item.get_id(0);
                const int ix = cell % (nx - 1);
                const int iy = (cell / (nx - 1)) % (ny - 1);
                const int iz = cell / ((nx - 1) * (ny - 1));
                const std::size_t base = ix + nx * (iy + ny * iz);
                const std::size_t plane = nx * ny;
                const float v[8] = {values[base], values[base+1], values[base+1+plane], values[base+plane],
                    values[base+nx], values[base+nx+1], values[base+nx+1+plane], values[base+nx+plane]};
                const float p[8][3] = {
                    {grid.bounds_min[0]+ix*sx, grid.bounds_min[1]+iy*sy, grid.bounds_min[2]+iz*sz},
                    {grid.bounds_min[0]+(ix+1)*sx, grid.bounds_min[1]+iy*sy, grid.bounds_min[2]+iz*sz},
                    {grid.bounds_min[0]+(ix+1)*sx, grid.bounds_min[1]+iy*sy, grid.bounds_min[2]+(iz+1)*sz},
                    {grid.bounds_min[0]+ix*sx, grid.bounds_min[1]+iy*sy, grid.bounds_min[2]+(iz+1)*sz},
                    {grid.bounds_min[0]+ix*sx, grid.bounds_min[1]+(iy+1)*sy, grid.bounds_min[2]+iz*sz},
                    {grid.bounds_min[0]+(ix+1)*sx, grid.bounds_min[1]+(iy+1)*sy, grid.bounds_min[2]+iz*sz},
                    {grid.bounds_min[0]+(ix+1)*sx, grid.bounds_min[1]+(iy+1)*sy, grid.bounds_min[2]+(iz+1)*sz},
                    {grid.bounds_min[0]+ix*sx, grid.bounds_min[1]+(iy+1)*sy, grid.bounds_min[2]+(iz+1)*sz}};
                constexpr int edges[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
                float e[12][3];
                const int mask = edgeTable[indices[cell]];
                for (int k=0;k<12;++k) if (mask & (1<<k)) { const int a=edges[k][0], b=edges[k][1]; const float t=(iso-v[a])/(v[b]-v[a]); for(int d=0;d<3;++d)e[k][d]=p[a][d]+t*(p[b][d]-p[a][d]); }
                std::size_t out = static_cast<std::size_t>(output_offsets[cell]) * 3;
                for (int t=0; triTable[indices[cell]][t] != -1; t += 3) {
                    const float* a=e[triTable[indices[cell]][t]], *b=e[triTable[indices[cell]][t+1]], *c=e[triTable[indices[cell]][t+2]];
                    const float ux=b[0]-a[0], uy=b[1]-a[1], uz=b[2]-a[2], vx=c[0]-a[0], vy=c[1]-a[1], vz=c[2]-a[2];
                    float n[3]={uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx}; const float len=sycl::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); if(len>0)for(float& x:n)x/=len;
                    for(const float* q:{a,b,c}) { for(int d=0;d<3;++d){positions[out*3+d]=q[d];normals[out*3+d]=n[d];} ++out; }
                }
            });
        }).wait();
    }

} // namespace flib::graphics::marching_cubes::detail
