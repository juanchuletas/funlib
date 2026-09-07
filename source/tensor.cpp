
#include <funlib/Tensor/tensor.hpp>
namespace flib{


    template <typename T>
    Tensor<T>::Tensor()
    {
        m_rows = 0;
        m_cols = 0;
        m_gsize = 0;
        m_data = nullptr;
        m_device_data = nullptr;
        m_context = std::nullopt;
        m_device = std::nullopt;
    }
    template <typename T>
    Tensor<T>::Tensor(std::size_t rows, std::size_t cols) : Tensor({rows, cols}) {}

    template <typename T>
    Tensor<T>::Tensor(std::size_t rows, std::size_t cols, T* value) : Tensor({rows, cols}, value) {}

    template <typename T>
    Tensor<T>::Tensor(std::size_t rows) : Tensor({rows, 1}) {}

    template <typename T>
    Tensor<T>::Tensor(std::size_t rows, T *value) : Tensor({rows, 1}, value) {}

    template <typename T>
    Tensor<T>::Tensor(std::size_t rows, std::size_t cols, sycl::queue queue) : Tensor({rows, cols}, queue) {}

    template <typename T>
    Tensor<T>::Tensor(std::initializer_list<std::size_t> shape, sycl::queue queue) :
        Tensor(std::vector<std::size_t>(shape), queue) {}

    template <typename T>
    Tensor<T>::Tensor(std::initializer_list<std::size_t> shape) :
        Tensor(std::vector<std::size_t>(shape)) {}

    template <typename T>
    Tensor<T>::Tensor(std::initializer_list<std::size_t> shape, T* value) :
        Tensor(std::vector<std::size_t>(shape), value) {}

    template <typename T>
    Tensor<T>::Tensor(const std::vector<std::size_t>& shape, sycl::queue queue)
    {
        if(shape.size() == 0){
            throw std::invalid_argument("Tensor shape cannot be empty");
        }

        m_shape = shape;
        m_gsize = 1;
        for(std::size_t dimension : m_shape){
            m_gsize *= dimension;
        }

        m_cols = m_shape.back();
        m_rows = m_cols == 0 ? 0 : m_gsize / m_cols;
        m_data = nullptr;
        m_context = queue.get_context();
        m_device = queue.get_device();
        m_device_data = sycl::malloc_device<T>(m_gsize, queue);
        if(m_gsize != 0 && m_device_data == nullptr){
            throw std::bad_alloc();
        }
    }
    template <typename T>
    Tensor<T>::Tensor(const std::vector<std::size_t>& shape)
    {
        if(shape.size() == 0){
            throw std::invalid_argument("Tensor shape cannot be empty");
        }

        m_shape = shape;
        m_gsize = 1;
        for(std::size_t dimension : m_shape){
            m_gsize *= dimension;
        }

        m_cols = m_shape.back();
        m_rows = m_cols == 0 ? 0 : m_gsize / m_cols;
        m_device_data = nullptr;
        m_context = std::nullopt;
        m_device = std::nullopt;
        m_data = std::make_unique<T[]>(m_gsize);
        for (std::size_t i = 0; i < m_gsize; ++i) {
            m_data[i] = T(0); // Initialize with default value
        }
    }
    template <typename T>
    Tensor<T>::Tensor(const std::vector<std::size_t>& shape, T* value)
    {
        if(shape.size() == 0){
            throw std::invalid_argument("Tensor shape cannot be empty");
        }

        m_shape = shape;
        m_gsize = 1;
        for(std::size_t dimension : m_shape){
            m_gsize *= dimension;
        }

        m_cols = m_shape.back();
        m_rows = m_cols == 0 ? 0 : m_gsize / m_cols;
        m_device_data = nullptr;
        m_context = std::nullopt;
        m_device = std::nullopt;
        m_data = std::make_unique<T[]>(m_gsize);
        for(std::size_t i = 0; i < m_gsize; i++){
            m_data[i] = value[i];
        }
    }
    template <typename T>
    Tensor<T>::Tensor(const Tensor<T> &other)
    {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_gsize = other.m_gsize;
        m_shape = other.m_shape;
        m_device_data = nullptr;
        m_context = other.m_context;
        m_device = other.m_device;

        if(other.is_device()){
            sycl::queue queue(*m_context, *m_device);
            m_device_data = sycl::malloc_device<T>(m_gsize, queue);
            if(m_gsize != 0 && m_device_data == nullptr){
                throw std::bad_alloc();
            }
            queue.memcpy(m_device_data, other.m_device_data, m_gsize * sizeof(T)).wait();
        }
        else{
            m_data = std::make_unique<T[]>(m_gsize);
            for (std::size_t i = 0; i < m_gsize; ++i) {
                m_data[i] = other.m_data[i];
            }
        }
    }
    template <typename T>
    Tensor<T>::Tensor(Tensor<T> &&other) noexcept
    {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_gsize = other.m_gsize;
        m_shape = std::move(other.m_shape);
        m_data = std::move(other.m_data);
        m_device_data = other.m_device_data;
        m_context = std::move(other.m_context);
        m_device = std::move(other.m_device);

        other.m_rows = 0;
        other.m_cols = 0;
        other.m_gsize = 0;
        other.m_shape.clear();
        other.m_device_data = nullptr;
        other.m_context = std::nullopt;
        other.m_device = std::nullopt;
    }
    template <typename T>
    Tensor<T>::~Tensor()
    {
        release_device_data();
    }
    template <typename T>
    void Tensor<T>::release_device_data()
    {
        if(m_device_data != nullptr && m_context.has_value()){
            sycl::free(m_device_data, *m_context);
            m_device_data = nullptr;
        }
    }
    template <typename T>
    T &Tensor<T>::operator()(int row, int col)
    {
        if(is_device()) {
            throw std::runtime_error("Device tensor data is not directly accessible from the host");
        }
        //used to modify the matrix like: mat(i,j) = 5;
        if(row >= m_rows || col >= m_cols) {
            throw std::out_of_range("Index out of range");
        }
        return m_data[row * m_cols + col];
    }
    template <typename T>
    const T& Tensor<T>::operator()(int row, int col) const
    {
        if(is_device()) {
            throw std::runtime_error("Device tensor data is not directly accessible from the host");
        }
        //used to read the matrix like: val = mat(i,j);
        if(row >= m_rows || col >= m_cols) {
            throw std::out_of_range("Index out of range");
        }
        return m_data[row * m_cols + col];
    }
    template <typename T>
    const T& Tensor<T>::operator[](int index) const
    {
        if(is_device()) {
            throw std::runtime_error("Device tensor data is not directly accessible from the host");
        }
        //used to read the matrix like: val = mat[i];
        if(index >= m_rows * m_cols) {
            throw std::out_of_range("Index out of range");
        }
        return m_data[index];
    }
    template <typename T>
    T& Tensor<T>::operator[](int index)
    {
       if(is_device()) {
           throw std::runtime_error("Device tensor data is not directly accessible from the host");
       }
       //used to modify the matrix like: mat[i] = 5;
        if(index >= m_rows * m_cols) {
            throw std::out_of_range("Index out of range");
        }
        return m_data[index];
    }
    template <typename T>
    Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other)
    {
        if (this != &other) {
            release_device_data();
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_gsize = other.m_gsize;
            m_shape = other.m_shape;
            m_context = other.m_context;
            m_device = other.m_device;

            if(other.is_device()){
                m_data = nullptr;
                sycl::queue queue(*m_context, *m_device);
                m_device_data = sycl::malloc_device<T>(m_gsize, queue);
                if(m_gsize != 0 && m_device_data == nullptr){
                    throw std::bad_alloc();
                }
                queue.memcpy(m_device_data, other.m_device_data, m_gsize * sizeof(T)).wait();
            }
            else{
                m_device_data = nullptr;
                m_data = std::make_unique<T[]>(m_gsize);
                for (std::size_t i = 0; i < m_gsize; ++i) {
                    m_data[i] = other.m_data[i];
                }
            }
        }
        return *this;
    }
    template <typename T>
    Tensor<T> &Tensor<T>::operator=(Tensor<T> &&other) noexcept
    {
        if(this != &other){
            release_device_data();
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_gsize = other.m_gsize;
            m_shape = std::move(other.m_shape);
            m_data = std::move(other.m_data);
            m_device_data = other.m_device_data;
            m_context = std::move(other.m_context);
            m_device = std::move(other.m_device);

            other.m_rows = 0;
            other.m_cols = 0;
            other.m_gsize = 0;
            other.m_shape.clear();
            other.m_device_data = nullptr;
            other.m_context = std::nullopt;
            other.m_device = std::nullopt;
        }
        return *this;
    }
    template <typename T>
    sycl::buffer<T, 1> Tensor<T>::to_sycl_buffer() const
    {
        if(is_device()){
            throw std::runtime_error("A device tensor cannot be wrapped in a host-backed SYCL buffer");
        }
        return sycl::buffer<T, 1>(m_data.get(), sycl::range<1>(m_rows * m_cols));
    }

    template <typename T>
    sycl::event Tensor<T>::copy_from(const T* host_data, sycl::queue queue)
    {
        if(!is_device()){
            throw std::runtime_error("copy_from requires a device tensor");
        }
        if(queue.get_context() != *m_context){
            throw std::invalid_argument("Queue context does not match the tensor context");
        }
        return queue.memcpy(m_device_data, host_data, m_gsize * sizeof(T));
    }

    template <typename T>
    std::vector<T> Tensor<T>::to_host(sycl::queue queue) const
    {
        std::vector<T> result(m_gsize);
        if(is_device()){
            if(queue.get_context() != *m_context){
                throw std::invalid_argument("Queue context does not match the tensor context");
            }
            queue.memcpy(result.data(), m_device_data, m_gsize * sizeof(T)).wait();
        }
        else{
            for(std::size_t i = 0; i < m_gsize; i++){
                result[i] = m_data[i];
            }
        }
        return result;
    }

    template <typename T>
    void Tensor<T>::print() const
    {
        if(is_device()){
            throw std::runtime_error("Use to_host before printing a device tensor");
        }
        for (std::size_t i = 0; i < m_rows; ++i) {
            for (std::size_t j = 0; j < m_cols; ++j) {
                std::cout << std::fixed << std::setprecision(6) << m_data[i * m_cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // ---- Explicit Instantiations ----

    template class Tensor<double>;
    template class Tensor<float>;
    template class Tensor<int>;
    
};
