#include <funlib/Tensor/tensor_operations.hpp>

namespace flib
{
    template<typename T>
    Tensor<T> tensor_operations::scale(const Tensor<T>& input, T value, sycl::queue Q,
                                       sycl::event* kernel_event)
    {
        const std::vector<std::size_t>& shape = input.getShape();
        if(shape.size() == 0){
            throw std::invalid_argument("Scale requires a tensor with at least one dimension");
        }

        std::size_t size = input.getSize();
        if(input.is_device()){
            if(!input.is_accessible_from(Q)){
                throw std::invalid_argument("Scale queue cannot access the input tensor");
            }

            Tensor<T> output(shape, Q);
            if(size == 0){
                return output;
            }

            const T* input_data = input.device_data();
            T* output_data = output.device_data();
            sycl::event event = Q.submit([&](sycl::handler& cgh){
                cgh.parallel_for(sycl::range<1>{size}, [=](sycl::item<1> item){
                    std::size_t index = item.get_id(0);
                    output_data[index] = input_data[index] * value;
                });
            });
            if(kernel_event != nullptr){
                *kernel_event = event;
            }
            event.wait();
            return output;
        }

        Tensor<T> output(shape);
        if(size == 0){
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
                cgh.parallel_for(sycl::range<1>{size}, [=](sycl::item<1> item){
                    std::size_t index = item.get_id(0);
                    output_accessor[index] = input_accessor[index] * value;
                });
            });
            if(kernel_event != nullptr){
                *kernel_event = event;
            }
        }
        return output;
    }

    template Tensor<double> tensor_operations::scale(
        const Tensor<double>&, double, sycl::queue, sycl::event*);
    template Tensor<float> tensor_operations::scale(
        const Tensor<float>&, float, sycl::queue, sycl::event*);
    template Tensor<int> tensor_operations::scale(
        const Tensor<int>&, int, sycl::queue, sycl::event*);
}
