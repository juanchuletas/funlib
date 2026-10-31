#include "internal.hpp"
#include <cstdint>
#include <stdexcept>

namespace flib::graphics::marching_cubes::detail {

    namespace {

        void scan_device(sycl::queue& queue, std::uint32_t* values,
                         std::size_t count) {
            if (count == 0) {
                return;
            }

            constexpr std::size_t work_group_size = 64;
            const std::size_t global_size =
                ((count + work_group_size - 1) / work_group_size) *
                work_group_size;
            const std::size_t block_count = global_size / work_group_size;
            std::uint32_t* block_sums =
                sycl::malloc_device<std::uint32_t>(block_count, queue);

            queue.submit([=](sycl::handler& handler) {
                sycl::local_accessor<std::uint32_t, 1> local(
                    sycl::range<1>(work_group_size), handler);
                handler.parallel_for(
                    sycl::nd_range<1>(global_size, work_group_size),
                    [=](sycl::nd_item<1> item) {
                        const std::size_t local_id = item.get_local_id(0);
                        const std::size_t index = item.get_global_id(0);
                        local[local_id] = index < count ? values[index] : 0;

                        for (std::size_t stride = 1; stride < work_group_size;
                             stride *= 2) {
                            item.barrier(sycl::access::fence_space::local_space);
                            const std::size_t target =
                                (local_id + 1) * stride * 2 - 1;
                            if (target < work_group_size) {
                                local[target] += local[target - stride];
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                        if (local_id == 0) {
                            block_sums[item.get_group(0)] =
                                local[work_group_size - 1];
                            local[work_group_size - 1] = 0;
                        }

                        for (std::size_t stride = work_group_size / 2;
                             stride > 0; stride /= 2) {
                            item.barrier(sycl::access::fence_space::local_space);
                            const std::size_t target =
                                (local_id + 1) * stride * 2 - 1;
                            if (target < work_group_size) {
                                const std::uint32_t value =
                                    local[target - stride];
                                local[target - stride] = local[target];
                                local[target] += value;
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                        if (index < count) {
                            values[index] = local[local_id];
                        }
                    });
            }).wait();

            if (block_count > 1) {
                scan_device(queue, block_sums, block_count);
                queue.submit([=](sycl::handler& handler) {
                    handler.parallel_for(sycl::range<1>(global_size),
                        [=](sycl::item<1> item) {
                            const std::size_t index = item.get_id(0);
                            if (index < count) {
                                values[index] +=
                                    block_sums[index / work_group_size];
                            }
                        });
                }).wait();
            }

            sycl::free(block_sums, queue);
        }

    }

    ScanResult exclusive_scan(const flib::ui32tensor& counts,
                              sycl::queue& queue) {
        if (!counts.is_device() || !counts.is_accessible_from(queue)) {
            throw std::invalid_argument(
                "Marching cubes scan requires device-accessible counts");
        }

        ScanResult result{flib::ui32tensor({counts.getSize()}, queue), 0};
        if (counts.getSize() == 0) {
            return result;
        }

        queue.memcpy(result.offsets.device_data(), counts.device_data(),
                     counts.getSize() * sizeof(std::uint32_t)).wait();
        scan_device(queue, result.offsets.device_data(), counts.getSize());

        std::uint32_t last_offset = 0;
        std::uint32_t last_count = 0;
        queue.memcpy(&last_offset,
                     result.offsets.device_data() + counts.getSize() - 1,
                     sizeof(std::uint32_t));
        queue.memcpy(&last_count,
                     counts.device_data() + counts.getSize() - 1,
                     sizeof(std::uint32_t)).wait();
        result.total_triangles = last_offset + last_count;
        return result;
    }

} // namespace flib::graphics::marching_cubes::detail
