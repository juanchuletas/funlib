#if !defined(_SET_HPP_)
#define _SET_HPP_
#include <memory>
#include <iostream>
#include <iomanip>
#include <sycl/sycl.hpp>
namespace flib{
    template <typename T>
    class Set {
        //Set is the basic data structure for the funlib library
        std::size_t m_rows;
        std::size_t m_cols;
        std::size_t m_gsize;
        std::unique_ptr<T[]> m_data;

    public:
        Set();
        //2D set of values (matrix) if cols = 1 is a vector
        Set(std::size_t rows, std::size_t cols); //This is the default constructor for a 2D set
        Set(std::size_t rows, std::size_t cols, T* value);
        Set(std::size_t rows);
        Set(std::size_t rows, T* value);
        Set(const Set<T>& other);
        

        T& operator()(int row, int col);
        const T& operator()(int row, int col) const;
        const T& operator[](int index) const;
        T& operator[](int index);

        Set<T>& operator=(const Set<T>& other);

        //to sycl buffer
        sycl::buffer<T, 1> to_sycl_buffer() const;

        //getters
        std::size_t getRows() const { return m_rows; }
        std::size_t getCols() const { return m_cols; }

        void print() const;
        void fill(T value) {
            for (std::size_t i = 0; i < m_rows * m_cols; ++i) {
                m_data[i] = value;
            }
        }
    };
    // Typedefs
    using set = Set<double>;
    using fset = Set<float>;
    using iset = Set<int>;
   
};

#endif // _SET_HPP_
