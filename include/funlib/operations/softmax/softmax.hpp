#if !defined(_SOFTMAX_HPP_)
#define _SOFTMAX_HPP_
#include <funlib/Tensor/tensor.hpp>
namespace flib::operations{

    template<typename T>
    Tensor<T> softmax(const Tensor<T>& input, sycl::queue Q,
                      sycl::event* kernel_event = nullptr);


}

#endif // _SOFTMAX_HPP_
