# High-Performance GEMM Operator Example

## Overview

This example demonstrates how to implement a high-performance GEMM operator using PTO and common optimization techniques (core partitioning, base-block selection, L1 caching, and double buffering).

## Supported AI Processors

- A2/A3

## Directory Layout

```
kernels/gemm_performance/
├── scripts/
│   └── gen_data.py                  # Generates input and golden output
├── CMakeLists.txt                   # Build configuration
├── gemm_performance_kernel.cpp      # Kernel implementation
├── main.cpp                         # Host-side entry point
└── run.sh                           # Convenience script
```

## Operator Description

### Function

This example implements GEMM with fixed dimensions `[m, k, n] = [6144, 6144, 6144]`:

$$
C = A \times B
$$

Where `A`, `B`, and `C` are all `6144 × 6144`.

### Specification

| Item        | Value |
| ----------- | ----- |
| OpType      | `GEMM` |
| Inputs      | `a`: `m×k`, `float16`, `ND`; `b`: `n×k`, `float16`, `ND` |
| Output      | `c`: `m×n`, `float`, `ND` |
| Kernel name | `GEMMPerformance` |

## Optimization Notes

This example uses a 24-core A3 platform as the performance validation platform.

- **Core partitioning**: maximize parallelism by splitting work across Cube cores. Since `m`, `n`, and `k` are equal, prefer not splitting `k` within a single core, and split `m` and `n` across 24 cores. A `4 × 6` grouping yields `singleCoreM=1536`, `singleCoreK=6144`, `singleCoreN=1024` (chosen by this example).
- **Base block selection**: choose base blocks that maximize compute-to-memory ratio. For FP16, a common choice is `[baseM, baseN, baseK] = [128, 256, 64]`, which improves arithmetic intensity versus `[128, 128, 128]` while maintaining 512-byte-aligned GM writes.
- **L1 caching**: move multiple base blocks from GM to L1 per transfer to improve bandwidth utilization. This example sets `stepKa=stepKb=4` to cache four `k` blocks at a time.
- **Double buffering**: overlap DMA and compute by enabling double buffering in L1, L0A, and L0B.

## Tiling Parameters

| Parameter     | Value |
| ------------- | ----- |
| `m`           | 6144  |
| `k`           | 6144  |
| `n`           | 6144  |
| `singleCoreM` | 1536  |
| `singleCoreK` | 6144  |
| `singleCoreN` | 1024  |
| `baseM`       | 128   |
| `baseK`       | 64    |
| `baseN`       | 256   |
| `stepM`       | 1     |
| `stepKa`      | 4     |
| `stepKb`      | 4     |
| `stepN`       | 1     |

## Measured Performance (Reference)

On a 24-core Ascend A3, this example reports approximately:

- Cube utilization: 86.1%
- Scalar utilization: 8.1%
- MTE1 utilization: 68.1%
- MTE2 utilization: 95.2%
- MTE3 utilization: 3.1%
- AIC total cycles: 66725939
- Wall time: 1.506 ms

## Build and Run

1. Configure your Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Generate input + golden output:

```bash
cd ${git_clone_path}/kernels/gemm_performance
python3 scripts/gen_data.py
```

3. Run the example:

```bash
bash run.sh -r npu -v Ascend910B1
```

If the run succeeds, the output prints:

```text
test success
```

## Changelog

| Date       | Change |
| ---------- | ------ |
| 2025-12-15 | Adjusted example directory and added this README |
