#include <funlib/operations/softmax/softmax.hpp>

#include <algorithm>
#include <limits>

namespace flib::operations::details
{
    std::size_t softmax_work_group_size(std::size_t row_size, sycl::queue Q)
    {
        std::size_t device_maximum =
            Q.get_device().get_info<sycl::info::device::max_work_group_size>();
        std::size_t work_group_limit = std::min<std::size_t>(256, device_maximum);
        std::size_t work_group_size = 1;
        while(work_group_size < row_size && work_group_size * 2 <= work_group_limit){
            work_group_size *= 2;
        }
        return work_group_size;
    }
}

namespace flib::operations
{
    template<typename T>
    Tensor<T> softmax(const Tensor<T>& input, sycl::queue Q, sycl::event* kernel_event)
    {
        const std::vector<std::size_t>& shape = input.getShape();
        if(shape.size() == 0){
            throw std::invalid_argument("Softmax requires a tensor with at least one dimension");
        }

        std::size_t row_size = shape.back();
        if(row_size == 0){
            throw std::invalid_argument("Softmax final dimension cannot be zero");
        }

        std::size_t row_count = input.getSize() / row_size;
        std::size_t work_group_size = details::softmax_work_group_size(row_size, Q);
        std::size_t global_size = row_count * work_group_size;

        if(input.is_device()){
            if(!input.is_accessible_from(Q)){
                throw std::invalid_argument("Softmax queue cannot access the input tensor");
            }

            Tensor<T> output(shape, Q);
            if(row_count == 0){
                return output;
            }

            const T* input_data = input.device_data();
            T* output_data = output.device_data();
            sycl::event event = Q.submit([&](sycl::handler& cgh){
                sycl::local_accessor<T, 1> local_values(
                    sycl::range<1>{work_group_size}, cgh);
                cgh.parallel_for(
                    sycl::nd_range<1>{global_size, work_group_size},
                    [=](sycl::nd_item<1> item){
                        std::size_t row = item.get_group(0);
                        std::size_t local_id = item.get_local_id(0);
                        std::size_t row_offset = row * row_size;
                        T local_maximum = std::numeric_limits<T>::lowest();

                        for(std::size_t column = local_id; column < row_size;
                            column += work_group_size){
                            local_maximum = sycl::max(
                                local_maximum, input_data[row_offset + column]);
                        }
                        local_values[local_id] = local_maximum;
                        item.barrier(sycl::access::fence_space::local_space);

                        for(std::size_t stride = work_group_size / 2; stride > 0; stride /= 2){
                            if(local_id < stride){
                                local_values[local_id] = sycl::max(
                                    local_values[local_id], local_values[local_id + stride]);
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }

                        T maximum = local_values[0];
                        item.barrier(sycl::access::fence_space::local_space);
                        T local_sum = T(0);
                        for(std::size_t column = local_id; column < row_size; column += work_group_size){
                            T exponential = sycl::exp(input_data[row_offset + column] - maximum);
                            output_data[row_offset + column] = exponential;
                            local_sum += exponential;
                        }
                        local_values[local_id] = local_sum;
                        item.barrier(sycl::access::fence_space::local_space);

                        for(std::size_t stride = work_group_size / 2; stride > 0; stride /= 2){
                            if(local_id < stride){
                                local_values[local_id] += local_values[local_id + stride];
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }

                        T sum = local_values[0];
                        for(std::size_t column = local_id; column < row_size;
                            column += work_group_size){
                            output_data[row_offset + column] /= sum;
                        }
                    });
            });
            if(kernel_event != nullptr){
                *kernel_event = event;
            }
            event.wait();
            return output;
        }

        Tensor<T> output(shape);
        if(row_count == 0){
            return output;
        }
        {
            sycl::buffer<T, 1> input_buffer = input.to_sycl_buffer();
            sycl::buffer<T, 1> output_buffer = output.to_sycl_buffer();
            sycl::event event = Q.submit([&](sycl::handler& cgh){
                auto input_accessor =
                    input_buffer.template get_access<sycl::access::mode::read>(cgh);
                auto output_accessor =
                    output_buffer.template get_access<sycl::access::mode::write>(cgh);
                sycl::local_accessor<T, 1> local_values(
                    sycl::range<1>{work_group_size}, cgh);
                cgh.parallel_for(
                    sycl::nd_range<1>{global_size, work_group_size},
                    [=](sycl::nd_item<1> item){
                        std::size_t row = item.get_group(0);
                        std::size_t local_id = item.get_local_id(0);
                        std::size_t row_offset = row * row_size;
                        T local_maximum = std::numeric_limits<T>::lowest();

                        for(std::size_t column = local_id; column < row_size;
                            column += work_group_size){
                            local_maximum = sycl::max(
                                local_maximum, input_accessor[row_offset + column]);
                        }
                        local_values[local_id] = local_maximum;
                        item.barrier(sycl::access::fence_space::local_space);

                        for(std::size_t stride = work_group_size / 2; stride > 0; stride /= 2){
                            if(local_id < stride){
                                local_values[local_id] = sycl::max(
                                    local_values[local_id], local_values[local_id + stride]);
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }

                        T maximum = local_values[0];
                        item.barrier(sycl::access::fence_space::local_space);
                        T local_sum = T(0);
                        for(std::size_t column = local_id; column < row_size;
                            column += work_group_size){
                            T exponential = sycl::exp(input_accessor[row_offset + column] - maximum);
                            output_accessor[row_offset + column] = exponential;
                            local_sum += exponential;
                        }
                        local_values[local_id] = local_sum;
                        item.barrier(sycl::access::fence_space::local_space);

                        for(std::size_t stride = work_group_size / 2; stride > 0; stride /= 2){
                            if(local_id < stride){
                                local_values[local_id] += local_values[local_id + stride];
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }

                        T sum = local_values[0];
                        for(std::size_t column = local_id; column < row_size;
                            column += work_group_size){
                            output_accessor[row_offset + column] /= sum;
                        }
                    });
            });
            if(kernel_event != nullptr){
                *kernel_event = event;
            }
        }
        return output;
    }

    template Tensor<float> softmax(const Tensor<float>&, sycl::queue, sycl::event*);
    template Tensor<double> softmax(const Tensor<double>&, sycl::queue, sycl::event*);
}
