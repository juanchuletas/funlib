#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    template<typename T>
    flib::Tensor<T> gemmReference(const flib::Tensor<T>& A, const flib::Tensor<T>& B)
    {
        if (A.getCols() != B.getRows())
        {
            throw std::invalid_argument("Tensor dimensions do not match for multiplication");
        }

        flib::Tensor<T> C(A.getRows(), B.getCols());
        for (std::size_t i = 0; i < A.getRows(); i++)
        {
            for (std::size_t j = 0; j < B.getCols(); j++)
            {
                T sum = T(0);
                for (std::size_t k = 0; k < A.getCols(); k++)
                {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }

        return C;
    }

    template<typename T>
    void fillTensor(flib::Tensor<T>& tensor, int seed)
    {
        for (std::size_t i = 0; i < tensor.getRows(); i++)
        {
            for (std::size_t j = 0; j < tensor.getCols(); j++)
            {
                int value = static_cast<int>((i * 17 + j * 13 + seed) % 11) - 5;
                tensor(i, j) = static_cast<T>(value) / static_cast<T>(5);
            }
        }
    }

    template<typename T>
    bool almostEqual(T expected, T actual, std::size_t inner_dimension)
    {
        T absolute_tolerance = static_cast<T>(1e-5) * static_cast<T>(inner_dimension);
        T relative_tolerance = static_cast<T>(1e-4);
        T difference = std::abs(expected - actual);
        T scale = std::max(std::abs(expected), std::abs(actual));
        return difference <= absolute_tolerance + relative_tolerance * scale;
    }

    template<>
    bool almostEqual<int>(int expected, int actual, std::size_t)
    {
        return expected == actual;
    }

    template<typename T>
    bool checkShape(std::size_t rowsA, std::size_t colsA, std::size_t colsB)
    {
        flib::Tensor<T> A(rowsA, colsA);
        flib::Tensor<T> B(colsA, colsB);
        fillTensor(A, 3);
        fillTensor(B, 7);

        flib::Tensor<T> expected = gemmReference(A, B);
        sycl::queue Q = flib::sycl_handler::get_queue();
        flib::Tensor<T> actual = flib::tensor_operations::gemmTiled(A, B, Q);

        for (std::size_t i = 0; i < rowsA; i++)
        {
            for (std::size_t j = 0; j < colsB; j++)
            {
                if (!almostEqual(expected(i, j), actual(i, j), colsA))
                {
                    std::cerr<<"Mismatch at C("<<i<<", "<<j<<")"
                             <<" for shape "<<rowsA<<"x"<<colsA
                             <<" * "<<colsA<<"x"<<colsB<<std::endl;
                    std::cerr<<"Expected: "<<expected(i, j)<<std::endl;
                    std::cerr<<"Actual: "<<actual(i, j)<<std::endl;
                    return false;
                }
            }
        }

        std::cout<<"Passed "<<rowsA<<"x"<<colsA
                 <<" * "<<colsA<<"x"<<colsB<<std::endl;
        return true;
    }

    bool checkInvalidShape()
    {
        flib::Tensor<float> A(4, 3);
        flib::Tensor<float> B(2, 5);

        try
        {
            sycl::queue Q = flib::sycl_handler::get_queue();
            flib::tensor_operations::gemmTiled(A, B, Q);
        }
        catch (const std::invalid_argument&)
        {
            std::cout<<"Passed invalid shape"<<std::endl;
            return true;
        }

        std::cerr<<"Expected invalid shape to throw"<<std::endl;
        return false;
    }
}

int main()
{
    bool passed = true;

    passed = checkShape<float>(1, 1, 1) && passed;
    passed = checkShape<float>(16, 16, 16) && passed;
    passed = checkShape<float>(17, 31, 9) && passed;
    passed = checkShape<float>(64, 64, 64) && passed;
    passed = checkShape<double>(17, 31, 9) && passed;
    passed = checkShape<int>(17, 31, 9) && passed;
    passed = checkInvalidShape() && passed;

    if (!passed)
    {
        return 1;
    }

    std::cout<<"All tensor operation tests passed"<<std::endl;
    return 0;
}
