#if !defined(_CONVOLUTION_HPP_)
#define _CONVOLUTION_HPP_
#include <funlib/Tensor/tensor.hpp>


namespace flib::operations {
    template <typename T>
    Tensor<T> convolution2d(const Tensor<T> &input, const Tensor<T> &weight,
                         const Tensor<T>& bias,   std::size_t stride, sycl::queue queue,
                         sycl::event *kernel_event = nullptr);







}


#endif // _CONVOLUTION_HPP_
