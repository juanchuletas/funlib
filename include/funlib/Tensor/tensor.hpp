#if !defined(_TENSOR_HPP_)
#define _TENSOR_HPP_
#include <memory>
#include <iostream>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <sycl/sycl.hpp>
namespace flib{
    template <typename T>
    class Tensor {
        //Tensor is the basic data structure for the funlib library
        std::size_t m_rows;
        std::size_t m_cols;
        std::size_t m_gsize;
        std::unique_ptr<T[]> m_data;
        T* m_device_data;
        std::optional<sycl::context> m_context;
        std::optional<sycl::device> m_device;

        void release_device_data();

    public:
        Tensor();
        //2D set of values (matrix) if cols = 1 is a vector
        Tensor(std::size_t rows, std::size_t cols); //This is the default constructor for a 2D set
        Tensor(std::size_t rows, std::size_t cols, T* value);
        Tensor(std::size_t rows);
        Tensor(std::size_t rows, T* value);
        Tensor(std::size_t rows, std::size_t cols, sycl::queue queue);
        Tensor(const Tensor<T>& other);
        Tensor(Tensor<T>&& other) noexcept;
        ~Tensor();
        

        T& operator()(int row, int col);
        const T& operator()(int row, int col) const;
        const T& operator[](int index) const;
        T& operator[](int index);

        Tensor<T>& operator=(const Tensor<T>& other);
        Tensor<T>& operator=(Tensor<T>&& other) noexcept;

        //to sycl buffer
        sycl::buffer<T, 1> to_sycl_buffer() const;
        sycl::event copy_from(const T* host_data, sycl::queue queue);
        std::vector<T> to_host(sycl::queue queue) const;

        T* device_data() { return m_device_data; }
        const T* device_data() const { return m_device_data; }
        bool is_device() const { return m_context.has_value(); }
        bool is_host() const { return !m_context.has_value(); }
        bool is_accessible_from(sycl::queue queue) const {
            return is_host() || queue.get_context() == *m_context;
        }

        //getters
        std::size_t getRows() const { return m_rows; }
        std::size_t getCols() const { return m_cols; }

        void print() const;
        void fill(T value) {
            if(is_device()) {
                throw std::runtime_error("Cannot fill a device tensor from the host");
            }
            for (std::size_t i = 0; i < m_rows * m_cols; ++i) {
                m_data[i] = value;
            }
        }
    };
    // Typedefs
    using tensor = Tensor<double>;
    using ftensor = Tensor<float>;
    using itensor = Tensor<int>;
   
};

#endif // _SET_HPP_
