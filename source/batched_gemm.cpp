#include <funlib/Tensor/tensor_operations.hpp>

namespace flib {
constexpr std::size_t batched_gemm_tile_size = 16;

struct BatchedGemmSizes {
  std::size_t batch_count;
  std::size_t stored_rowsA;
  std::size_t stored_colsA;
  std::size_t stored_rowsB;
  std::size_t stored_colsB;
  std::size_t rowsC;
  std::size_t colsC;
  std::size_t inner_size;
};

template <bool transpose_A, bool transpose_B>
void set_strides(const BatchedGemmSizes &sizes, std::size_t &row_strideA,
                 std::size_t &inner_strideA, std::size_t &inner_strideB,
                 std::size_t &col_strideB) {
  if constexpr (transpose_A) {
    row_strideA = 1;
    inner_strideA = sizes.stored_colsA;
  } else {
    row_strideA = sizes.stored_colsA;
    inner_strideA = 1;
  }

  if constexpr (transpose_B) {
    inner_strideB = 1;
    col_strideB = sizes.stored_colsB;
  } else {
    inner_strideB = sizes.stored_colsB;
    col_strideB = 1;
  }
}

template <typename T, bool transpose_A, bool transpose_B>
sycl::event submit_usm_kernel(const T *dataA, const T *dataB, T *dataC,
                              const BatchedGemmSizes &sizes, sycl::queue Q) {
  std::size_t row_strideA;
  std::size_t inner_strideA;
  std::size_t inner_strideB;
  std::size_t col_strideB;
  set_strides<transpose_A, transpose_B>(sizes, row_strideA, inner_strideA,
                                        inner_strideB, col_strideB);

  std::size_t matrix_sizeA = sizes.stored_rowsA * sizes.stored_colsA;
  std::size_t matrix_sizeB = sizes.stored_rowsB * sizes.stored_colsB;
  std::size_t matrix_sizeC = sizes.rowsC * sizes.colsC;

  return Q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::range<3>{sizes.batch_count, sizes.rowsC, sizes.colsC},
        [=](sycl::item<3> item) {
          std::size_t batch = item.get_id(0);
          std::size_t row = item.get_id(1);
          std::size_t col = item.get_id(2);
          std::size_t offsetA = batch * matrix_sizeA;
          std::size_t offsetB = batch * matrix_sizeB;
          T sum = T(0);
          for (std::size_t k = 0; k < sizes.inner_size; k++) {
            std::size_t indexA = row * row_strideA + k * inner_strideA;
            std::size_t indexB = k * inner_strideB + col * col_strideB;
            sum += dataA[offsetA + indexA] * dataB[offsetB + indexB];
          }
          dataC[batch * matrix_sizeC + row * sizes.colsC + col] = sum;
        });
  });
}

template <typename T>
sycl::event submit_usm_normal_transposed_tiled_kernel(
    const T *dataA, const T *dataB, T *dataC,
    const BatchedGemmSizes &sizes, sycl::queue Q) {
  constexpr std::size_t tile_size = batched_gemm_tile_size;
  std::size_t matrix_sizeA = sizes.stored_rowsA * sizes.stored_colsA;
  std::size_t matrix_sizeB = sizes.stored_rowsB * sizes.stored_colsB;
  std::size_t matrix_sizeC = sizes.rowsC * sizes.colsC;
  std::size_t global_rows =
      ((sizes.rowsC + tile_size - 1) / tile_size) * tile_size;
  std::size_t global_cols =
      ((sizes.colsC + tile_size - 1) / tile_size) * tile_size;

  return Q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<T, 1> tileA(tile_size * tile_size, cgh);
    sycl::local_accessor<T, 1> tileB(tile_size * tile_size, cgh);

    cgh.parallel_for(
        sycl::nd_range<3>{{sizes.batch_count, global_rows, global_cols},
                          {1, tile_size, tile_size}},
        [=](sycl::nd_item<3> item) {
          std::size_t batch = item.get_global_id(0);
          std::size_t row = item.get_global_id(1);
          std::size_t col = item.get_global_id(2);
          std::size_t local_row = item.get_local_id(1);
          std::size_t local_col = item.get_local_id(2);
          std::size_t offsetA = batch * matrix_sizeA;
          std::size_t offsetB = batch * matrix_sizeB;
          T sum = T(0);

          for (std::size_t tile = 0; tile < sizes.inner_size;
               tile += tile_size) {
            std::size_t k = tile + local_col;
            std::size_t key_row = item.get_group(2) * tile_size + local_row;

            tileA[local_row * tile_size + local_col] =
                row < sizes.rowsC && k < sizes.inner_size
                    ? dataA[offsetA + row * sizes.stored_colsA + k]
                    : T(0);
            tileB[local_row * tile_size + local_col] =
                key_row < sizes.colsC && k < sizes.inner_size
                    ? dataB[offsetB + key_row * sizes.stored_colsB + k]
                    : T(0);

            item.barrier(sycl::access::fence_space::local_space);

            for (std::size_t k = 0; k < tile_size; k++) {
              sum += tileA[local_row * tile_size + k] *
                     tileB[local_col * tile_size + k];
            }

            item.barrier(sycl::access::fence_space::local_space);
          }

          if (row < sizes.rowsC && col < sizes.colsC) {
            dataC[batch * matrix_sizeC + row * sizes.colsC + col] = sum;
          }
        });
  });
}

template <typename T, bool transpose_A, bool transpose_B>
sycl::event submit_buffer_kernel(sycl::buffer<T, 1> &buffA,
                                 sycl::buffer<T, 1> &buffB,
                                 sycl::buffer<T, 1> &buffC,
                                 const BatchedGemmSizes &sizes, sycl::queue Q) {
  std::size_t row_strideA;
  std::size_t inner_strideA;
  std::size_t inner_strideB;
  std::size_t col_strideB;
  set_strides<transpose_A, transpose_B>(sizes, row_strideA, inner_strideA,
                                        inner_strideB, col_strideB);

  std::size_t matrix_sizeA = sizes.stored_rowsA * sizes.stored_colsA;
  std::size_t matrix_sizeB = sizes.stored_rowsB * sizes.stored_colsB;
  std::size_t matrix_sizeC = sizes.rowsC * sizes.colsC;

  return Q.submit([&](sycl::handler &cgh) {
    auto accA = buffA.template get_access<sycl::access::mode::read>(cgh);
    auto accB = buffB.template get_access<sycl::access::mode::read>(cgh);
    auto accC = buffC.template get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for(
        sycl::range<3>{sizes.batch_count, sizes.rowsC, sizes.colsC},
        [=](sycl::item<3> item) {
          std::size_t batch = item.get_id(0);
          std::size_t row = item.get_id(1);
          std::size_t col = item.get_id(2);
          std::size_t offsetA = batch * matrix_sizeA;
          std::size_t offsetB = batch * matrix_sizeB;
          T sum = T(0);
          for (std::size_t k = 0; k < sizes.inner_size; k++) {
            std::size_t indexA = row * row_strideA + k * inner_strideA;
            std::size_t indexB = k * inner_strideB + col * col_strideB;
            sum += accA[offsetA + indexA] * accB[offsetB + indexB];
          }
          accC[batch * matrix_sizeC + row * sizes.colsC + col] = sum;
        });
  });
}

template <typename T>
sycl::event submit_buffer_normal_transposed_tiled_kernel(
    sycl::buffer<T, 1> &buffA, sycl::buffer<T, 1> &buffB,
    sycl::buffer<T, 1> &buffC, const BatchedGemmSizes &sizes, sycl::queue Q) {
  constexpr std::size_t tile_size = batched_gemm_tile_size;
  std::size_t matrix_sizeA = sizes.stored_rowsA * sizes.stored_colsA;
  std::size_t matrix_sizeB = sizes.stored_rowsB * sizes.stored_colsB;
  std::size_t matrix_sizeC = sizes.rowsC * sizes.colsC;
  std::size_t global_rows =
      ((sizes.rowsC + tile_size - 1) / tile_size) * tile_size;
  std::size_t global_cols =
      ((sizes.colsC + tile_size - 1) / tile_size) * tile_size;

  return Q.submit([&](sycl::handler &cgh) {
    auto accA = buffA.template get_access<sycl::access::mode::read>(cgh);
    auto accB = buffB.template get_access<sycl::access::mode::read>(cgh);
    auto accC = buffC.template get_access<sycl::access::mode::write>(cgh);
    sycl::local_accessor<T, 1> tileA(tile_size * tile_size, cgh);
    sycl::local_accessor<T, 1> tileB(tile_size * tile_size, cgh);

    cgh.parallel_for(
        sycl::nd_range<3>{{sizes.batch_count, global_rows, global_cols},
                          {1, tile_size, tile_size}},
        [=](sycl::nd_item<3> item) {
          std::size_t batch = item.get_global_id(0);
          std::size_t row = item.get_global_id(1);
          std::size_t col = item.get_global_id(2);
          std::size_t local_row = item.get_local_id(1);
          std::size_t local_col = item.get_local_id(2);
          std::size_t offsetA = batch * matrix_sizeA;
          std::size_t offsetB = batch * matrix_sizeB;
          T sum = T(0);

          for (std::size_t tile = 0; tile < sizes.inner_size;
               tile += tile_size) {
            std::size_t k = tile + local_col;
            std::size_t key_row = item.get_group(2) * tile_size + local_row;

            tileA[local_row * tile_size + local_col] =
                row < sizes.rowsC && k < sizes.inner_size
                    ? accA[offsetA + row * sizes.stored_colsA + k]
                    : T(0);
            tileB[local_row * tile_size + local_col] =
                key_row < sizes.colsC && k < sizes.inner_size
                    ? accB[offsetB + key_row * sizes.stored_colsB + k]
                    : T(0);

            item.barrier(sycl::access::fence_space::local_space);

            for (std::size_t inner = 0; inner < tile_size; inner++) {
              sum += tileA[local_row * tile_size + inner] *
                     tileB[local_col * tile_size + inner];
            }

            item.barrier(sycl::access::fence_space::local_space);
          }

          if (row < sizes.rowsC && col < sizes.colsC) {
            accC[batch * matrix_sizeC + row * sizes.colsC + col] = sum;
          }
        });
  });
}

template <typename T>
sycl::event submit_usm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C,
                       const BatchedGemmSizes &sizes, bool transpose_A,
                       bool transpose_B, sycl::queue Q) {
  if (transpose_A) {
    if (transpose_B) {
      return submit_usm_kernel<T, true, true>(
          A.device_data(), B.device_data(), C.device_data(), sizes, Q);
    }
    return submit_usm_kernel<T, true, false>(
        A.device_data(), B.device_data(), C.device_data(), sizes, Q);
  }
  if (transpose_B) {
    return submit_usm_normal_transposed_tiled_kernel<T>(
        A.device_data(), B.device_data(), C.device_data(), sizes, Q);
  }
  return submit_usm_kernel<T, false, false>(A.device_data(), B.device_data(),
                                            C.device_data(), sizes, Q);
}

template <typename T>
sycl::event submit_buffer(sycl::buffer<T, 1> &buffA, sycl::buffer<T, 1> &buffB,
                          sycl::buffer<T, 1> &buffC,
                          const BatchedGemmSizes &sizes, bool transpose_A,
                          bool transpose_B, sycl::queue Q) {
  if (transpose_A) {
    if (transpose_B) {
      return submit_buffer_kernel<T, true, true>(buffA, buffB, buffC, sizes, Q);
    }
    return submit_buffer_kernel<T, true, false>(buffA, buffB, buffC, sizes, Q);
  }
  if (transpose_B) {
    return submit_buffer_normal_transposed_tiled_kernel<T>(buffA, buffB, buffC,
                                                           sizes, Q);
  }
  return submit_buffer_kernel<T, false, false>(buffA, buffB, buffC, sizes, Q);
}

template <typename T>
Tensor<T> tensor_operations::gemm_batched(const Tensor<T> &A,
                                          const Tensor<T> &B, sycl::queue Q,
                                          bool transpose_A, bool transpose_B,
                                          sycl::event *kernel_event) {
  if (A.getRank() < 2 || B.getRank() < 2) {
    throw std::invalid_argument(
        "Batched GEMM tensors must have at least two dimensions");
  }
  if (A.getRank() != B.getRank()) {
    throw std::invalid_argument("Batched GEMM tensors must have the same rank");
  }

  const std::vector<std::size_t> &shapeA = A.getShape();
  const std::vector<std::size_t> &shapeB = B.getShape();
  std::size_t rank = A.getRank();
  std::size_t batch_count = 1;
  for (std::size_t axis = 0; axis + 2 < rank; axis++) {
    if (shapeA[axis] != shapeB[axis]) {
      throw std::invalid_argument("Batched GEMM batch dimensions must match");
    }
    batch_count *= shapeA[axis];
  }

  BatchedGemmSizes sizes;
  sizes.batch_count = batch_count;
  sizes.stored_rowsA = shapeA[rank - 2];
  sizes.stored_colsA = shapeA[rank - 1];
  sizes.stored_rowsB = shapeB[rank - 2];
  sizes.stored_colsB = shapeB[rank - 1];
  sizes.rowsC = transpose_A ? sizes.stored_colsA : sizes.stored_rowsA;
  sizes.inner_size = transpose_A ? sizes.stored_rowsA : sizes.stored_colsA;
  std::size_t innerB = transpose_B ? sizes.stored_colsB : sizes.stored_rowsB;
  sizes.colsC = transpose_B ? sizes.stored_rowsB : sizes.stored_colsB;
  if (sizes.inner_size != innerB) {
    throw std::invalid_argument("Batched GEMM matrix dimensions do not match");
  }
  if (A.is_device() != B.is_device()) {
    throw std::invalid_argument(
        "Batched GEMM inputs must use the same memory location");
  }
  if (A.is_device() && (!A.is_accessible_from(Q) || !B.is_accessible_from(Q))) {
    throw std::invalid_argument(
        "Batched GEMM queue cannot access the input tensors");
  }

  std::vector<std::size_t> output_shape(shapeA.begin(), shapeA.end() - 2);
  output_shape.push_back(sizes.rowsC);
  output_shape.push_back(sizes.colsC);
  std::size_t total_size = sizes.batch_count * sizes.rowsC * sizes.colsC;

  if (A.is_device()) {
    Tensor<T> C(output_shape, Q);
    if (total_size == 0) {
      return C;
    }
    if (sizes.inner_size == 0) {
      sycl::event event = Q.fill(C.device_data(), T(0), total_size);
      if (kernel_event != nullptr) {
        *kernel_event = event;
      }
      event.wait();
      return C;
    }

    sycl::event event =
        submit_usm(A, B, C, sizes, transpose_A, transpose_B, Q);
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
    event.wait();
    return C;
  }

  Tensor<T> C(output_shape);
  if (total_size == 0 || sizes.inner_size == 0) {
    return C;
  }
  {
    sycl::buffer<T, 1> buffA = A.to_sycl_buffer();
    sycl::buffer<T, 1> buffB = B.to_sycl_buffer();
    sycl::buffer<T, 1> buffC = C.to_sycl_buffer();
    sycl::event event =
        submit_buffer(buffA, buffB, buffC, sizes, transpose_A, transpose_B, Q);
    if (kernel_event != nullptr) {
      *kernel_event = event;
    }
  }
  return C;
}

template Tensor<double> tensor_operations::gemm_batched(const Tensor<double> &,
                                                        const Tensor<double> &,
                                                        sycl::queue, bool, bool,
                                                        sycl::event *);
template Tensor<float> tensor_operations::gemm_batched(const Tensor<float> &,
                                                       const Tensor<float> &,
                                                       sycl::queue, bool, bool,
                                                       sycl::event *);
template Tensor<int> tensor_operations::gemm_batched(const Tensor<int> &,
                                                     const Tensor<int> &,
                                                     sycl::queue, bool, bool,
                                                     sycl::event *);
} // namespace flib
