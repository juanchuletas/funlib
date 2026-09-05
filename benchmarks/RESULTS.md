# Funlib GEMM benchmark results

This document records the current FP32 GEMM results for all four Funlib implementations:

- `gemm`: one work-item computes one output element directly from global memory.
- `gemmTiled`: one work-item computes one output element using work-group local-memory tiles.
- `gemm_blocked2x2`: one work-item computes a 2x2 output block directly from global memory.
- `gemm_tiled_blocked2x2`: one work-item computes a 2x2 output block using local-memory tiles.

Kernel GFLOP/s is calculated as:

```text
GFLOP/s = (2 * M * K * N) / kernel_time
```

The benchmark uses 5 warmup executions and reports the median of 21 measured executions.

## NVIDIA GeForce RTX 4060 Laptop GPU — CUDA

Environment:

```text
Device:   NVIDIA GeForce RTX 4060 Laptop GPU
Backend:  CUDA
Driver:   CUDA 12.4
Data type: FP32
```

### `gemm` compared with `gemmTiled`

These are representative kernel-only medians from the three reported runs.

| M | K | N | `gemm` kernel (ms) | `gemm` GFLOP/s | `gemmTiled` kernel (ms) | `gemmTiled` GFLOP/s | Tiled speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 128 | 0.017 | 246.724 | 0.017 | 246.724 | 1.000x |
| 256 | 256 | 256 | 0.074 | 453.438 | 0.079 | 424.740 | 0.937x |
| 512 | 512 | 512 | 0.522 | 514.244 | 0.541 | 496.184 | 0.965x |
| 127 | 512 | 512 | 0.133 | 500.636 | 0.135 | 493.219 | 0.985x |
| 512 | 64 | 512 | 0.061 | 550.073 | 0.067 | 500.812 | 0.910x |

For these CUDA measurements, the simple and tiled one-output-per-work-item kernels perform similarly. The explicit local-memory traffic and synchronization in `gemmTiled` do not produce a speedup.

### `gemmTiled` compared with `gemm_blocked2x2`

| M | K | N | `gemmTiled` kernel (ms) | `gemmTiled` GFLOP/s | `gemm_blocked2x2` kernel (ms) | `gemm_blocked2x2` GFLOP/s | Blocked speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 128 | 0.017 | 241.121 | 0.016 | 255.938 | 1.061x |
| 256 | 256 | 256 | 0.079 | 425.504 | 0.046 | 728.162 | 1.711x |
| 512 | 512 | 512 | 0.563 | 476.624 | 0.273 | 981.816 | 2.060x |
| 127 | 512 | 512 | 0.156 | 427.729 | 0.086 | 774.248 | 1.810x |
| 512 | 64 | 512 | 0.057 | 584.847 | 0.030 | 1131.188 | 1.934x |

Register blocking is the major CUDA improvement. Computing four outputs per work-item reuses each loaded A and B value across multiple multiply-add operations.

### `gemm_blocked2x2` compared with `gemm_tiled_blocked2x2`

| M | K | N | Blocked total (ms) | Blocked kernel (ms) | Blocked GFLOP/s | Tiled-blocked total (ms) | Tiled-blocked kernel (ms) | Tiled-blocked GFLOP/s | Tiled-blocked speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 128 | 0.040 | 0.016 | 256.000 | 0.040 | 0.015 | 273.031 | 1.067x |
| 256 | 256 | 256 | 0.070 | 0.046 | 728.162 | 0.071 | 0.049 | 682.667 | 0.938x |
| 512 | 512 | 512 | 0.390 | 0.276 | 970.897 | 0.419 | 0.305 | 879.678 | 0.906x |
| 127 | 512 | 512 | 0.110 | 0.087 | 765.023 | 0.116 | 0.090 | 738.852 | 0.966x |
| 512 | 64 | 512 | 0.062 | 0.039 | 862.360 | 0.067 | 0.042 | 799.048 | 0.927x |

The non-tiled 2x2 kernel wins for four of the five CUDA shapes. NVIDIA already obtains useful coalescing and cache reuse from this access pattern. The current tiled-blocked kernel adds local-memory operations and two barriers per K tile, so its overhead is slightly greater than its traffic reduction.

### Current CUDA conclusion

| Implementation | Observed behavior |
|---|---|
| `gemm` | Baseline |
| `gemmTiled` | Similar to, or slightly slower than, `gemm` |
| `gemm_blocked2x2` | Best current CUDA implementation |
| `gemm_tiled_blocked2x2` | Close to blocked, but usually slower with the current tile configuration |

The best reported CUDA result is `1131.188 GFLOP/s` from `gemm_blocked2x2` for 512x64x512. For the square 512x512x512 case, the best reported result is `981.816 GFLOP/s`.

## Intel Arc Graphics — Level Zero

Environment:

```text
Device:   Intel(R) Arc(TM) Graphics
Backend:  Level Zero
Driver:   1.15.38308+1
Data type: FP32
```

### `gemmTiled` compared with `gemm_blocked2x2`

| M | K | N | `gemmTiled` kernel (ms) | `gemmTiled` GFLOP/s | `gemm_blocked2x2` kernel (ms) | `gemm_blocked2x2` GFLOP/s | Blocked speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 128 | 0.085 | 49.527 | 0.130 | 32.316 | 0.652x |
| 256 | 256 | 256 | 0.264 | 127.271 | 0.322 | 104.146 | 0.818x |
| 512 | 512 | 512 | 1.312 | 204.571 | 1.111 | 241.607 | 1.181x |
| 127 | 512 | 512 | 0.412 | 161.499 | 0.709 | 93.919 | 0.582x |
| 512 | 64 | 512 | 0.278 | 120.735 | 0.228 | 147.357 | 1.221x |

On Level Zero, plain tiling wins for the smaller square shapes and the odd-row shape. Register blocking wins for 512x512x512 and 512x64x512. This indicates that the preferred kernel depends on shape even before introducing device-specific dispatch.

No Level Zero measurements for `gemm` or `gemm_tiled_blocked2x2` have been recorded in the supplied results yet.

## Intel Arc Graphics — OpenCL

Environment:

```text
Device:   Intel(R) Arc(TM) Graphics
Backend:  OpenCL
Driver:   26.18.38308.1
Data type: FP32
```

### `gemm_blocked2x2` compared with `gemm_tiled_blocked2x2`

| M | K | N | Blocked total (ms) | Blocked kernel (ms) | Blocked GFLOP/s | Tiled-blocked total (ms) | Tiled-blocked kernel (ms) | Tiled-blocked GFLOP/s | Tiled-blocked speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 128 | 0.275 | 0.161 | 26.045 | 0.192 | 0.077 | 54.120 | 2.078x |
| 256 | 256 | 256 | 0.342 | 0.277 | 121.190 | 0.270 | 0.147 | 227.971 | 1.881x |
| 512 | 512 | 512 | 1.164 | 1.022 | 262.662 | 0.733 | 0.585 | 459.028 | 1.748x |
| 127 | 512 | 512 | 0.768 | 0.640 | 104.038 | 0.366 | 0.242 | 275.049 | 2.644x |
| 512 | 64 | 512 | 0.286 | 0.153 | 219.879 | 0.221 | 0.120 | 280.352 | 1.275x |

The tiled-blocked implementation wins for every measured OpenCL shape. Explicit work-group local-memory reuse is especially valuable on this Intel configuration.

No OpenCL measurements for `gemm` or `gemmTiled` have been recorded in the supplied four-kernel comparison yet.

## Overall findings

1. Register blocking provides the largest improvement currently observed on NVIDIA CUDA.
2. Combining local-memory tiling with 2x2 register blocking provides the largest improvement currently observed on Intel OpenCL.
3. The current local-memory configuration adds overhead on NVIDIA but removes substantial memory cost on Intel OpenCL.
4. Odd and narrow matrix shapes can change which implementation wins.
5. Runtime dispatch should consider the backend and matrix dimensions instead of selecting one universal GEMM kernel.

## Measurement limitations

- Results from CUDA, Level Zero, and OpenCL must not be compared as if only the kernel changed; the backend and driver also changed.
- Some tables came from separate benchmark executions, so small timing differences can be caused by GPU clocks, temperature, power state, and submission overhead.
- Kernel time measures device execution. Total time also contains allocation, submission, synchronization, and tensor destruction overhead.
- The queue registry name is user-defined. The device information and reported SYCL backend determine where the kernel actually executed.
- A single head-to-head benchmark containing all four implementations is still needed for a strictly controlled comparison.
- Correctness must be verified for regular, odd, small, and non-multiple-of-16 dimensions before performance-based dispatch is enabled.

## Next measurement

Run all four implementations in the same executable, with the same resident inputs, warmup policy, repetitions, queue, and execution order:

```text
gemm
gemmTiled
gemm_blocked2x2
gemm_tiled_blocked2x2
```

That will produce a definitive four-way table for each backend and remove ambiguity caused by combining measurements from different runs.
