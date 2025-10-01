
#include <funlib/Tensor/tensor.hpp>
namespace flib{


    template <typename T>
    Tensor<T>::Tensor()
    {
        m_rows = 0;
        m_cols = 0;
        m_gsize = 0;
        m_data = nullptr;
    }
    template <typename T>
    Tensor<T>::Tensor(std::size_t rows, std::size_t cols)
    {
        this->m_rows = rows;
        this->m_cols = cols;
        m_gsize = rows * cols;
        m_data = std::make_unique<T[]>(m_rows * m_cols);
        for (std::size_t i = 0; i < m_rows * m_cols; ++i) {
            m_data[i] = T(0); // Initialize with default value
        }
    }
    template <typename T>
    Tensor<T>::Tensor(std::size_t rows, std::size_t cols, T* value)
    {
        this->m_rows = rows;
        this->m_cols = cols;
        m_gsize = rows * cols;
        m_data = std::make_unique<T[]>(m_rows * m_cols);
        for (std::size_t i = 0; i < m_rows * m_cols; ++i) {
            m_data[i] = value[i]; // Initialize with provided values
        }
    }
    template <typename T>
    Tensor<T>::Tensor(std::size_t rows)
    {
        this->m_rows = rows;
        this->m_cols = 1;
        m_gsize = rows;
        m_data = std::make_unique<T[]>(m_rows * m_cols);
        for (std::size_t i = 0; i < m_rows * m_cols; ++i) {
            m_data[i] = T(0); // Initialize with default value
        }
    }
    template <typename T>
    Tensor<T>::Tensor(std::size_t rows, T *value)
    {
        this->m_rows = rows;
        this->m_cols = 1;
        m_gsize = rows;
        m_data = std::make_unique<T[]>(m_rows * m_cols);
        for (std::size_t i = 0; i < m_rows * m_cols; ++i) {
            m_data[i] = value[i]; // Initialize with provided values
        }
    }
    template <typename T>
    Tensor<T>::Tensor(const Tensor<T> &other)
    {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_gsize = other.m_gsize;
        m_data = std::make_unique<T[]>(m_rows * m_cols);
        for (std::size_t i = 0; i < m_rows * m_cols; ++i) {
            m_data[i] = other.m_data[i];
        }
    }
    template <typename T>
    T &Tensor<T>::operator()(int row, int col)
    {
        //used to modify the matrix like: mat(i,j) = 5;
        if(row >= m_rows || col >= m_cols) {
            throw std::out_of_range("Index out of range");
        }
        return m_data[row * m_cols + col];
    }
    template <typename T>
    const T& Tensor<T>::operator()(int row, int col) const
    {
        //used to read the matrix like: val = mat(i,j);
        if(row >= m_rows || col >= m_cols) {
            throw std::out_of_range("Index out of range");
        }
        return m_data[row * m_cols + col];
    }
    template <typename T>
    const T& Tensor<T>::operator[](int index) const
    {
        //used to read the matrix like: val = mat[i];
        if(index >= m_rows * m_cols) {
            throw std::out_of_range("Index out of range");
        }
        return m_data[index];
    }
    template <typename T>
    T& Tensor<T>::operator[](int index)
    {
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
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_gsize = other.m_gsize;
            m_data = std::make_unique<T[]>(m_rows * m_cols);
            for (std::size_t i = 0; i < m_rows * m_cols; ++i) {
                m_data[i] = other.m_data[i];
            }
        }
        return *this;
    }
    template <typename T>
    sycl::buffer<T, 1> Tensor<T>::to_sycl_buffer() const
    {
        return sycl::buffer<T, 1>(m_data.get(), sycl::range<1>(m_rows * m_cols));
    }

    template <typename T>
    void Tensor<T>::print() const
    {
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