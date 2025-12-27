# Basic GEMM Operator Example

## Overview

This example demonstrates how to implement a basic GEMM operator using PTO, including project setup, build, and execution.

## Supported AI Processors

- A2/A3

## Directory Layout

```
kernels/gemm_basic/
├── scripts/
│   └── gen_data.py              # Generates input and golden output
├── CMakeLists.txt               # Build configuration
├── gemm_basic_kernel.cpp        # Kernel implementation
├── main.cpp                     # Host-side entry point
└── run.sh                       # Convenience script
```

## Operator Description

### Function

This example implements GEMM with fixed dimensions `[m, k, n] = [512, 2048, 1536]`:

$$
C = A \times B
$$

Where:

- `A` shape: `[512, 2048]` (`m × k`)
- `B` shape: `[2048, 1536]` (`k × n`)
- `C` shape: `[512, 1536]` (`m × n`)

### Specification

| Item        | Value |
| ----------- | ----- |
| OpType      | `gemm` |
| Inputs      | `a`: `m×k`, `float16`, `ND`; `b`: `k×n`, `float16`, `DN` |
| Output      | `c`: `m×n`, `float`, `ND` |
| Kernel name | `gemm_basic_kernel` |

### Tiling Parameters

The validation platform has 24 cores. The workload is split across cores (prioritizing splitting `m` and `n`) using a `4 × 6` grouping: split `m` into 4 parts and `n` into 6 parts to fully utilize 24 cores.

Per-core shape:

- `singleCoreM = 128`, `singleCoreK = 2048`, `singleCoreN = 256`

Because the per-core tile still exceeds L0 capacity, `k` is further tiled into base blocks of size 64. The base block is:

- `baseM = 128`, `baseK = 64`, `baseN = 256`

| Parameter     | Value |
| ------------- | ----- |
| `m`           | 512   |
| `k`           | 2048  |
| `n`           | 1536  |
| `singleCoreM` | 128   |
| `singleCoreK` | 2048  |
| `singleCoreN` | 256   |
| `baseM`       | 128   |
| `baseK`       | 64    |
| `baseN`       | 256   |

## Implementation Notes

### Type definitions

The implementation defines matrix representations for GM, L1, and L0, then assigns backing storage for tiles. Example (simplified):

```cpp
using NDValidShapeA = TileShape2D<U, baseM, baseK>;
using NDsingleCoreShapeA = BaseShape2D<U, M, K>;
using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA>; // A in GM (ND)

using NDValidShapeB = TileShape2D<U, baseK, baseN, Layout::DN>;
using NDsingleCoreShapeB = BaseShape2D<U, K, N, Layout::DN>;
using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>; // B in GM (DN)

using NDValidShapeC = TileShape2D<T, baseM, baseN>;
using NDWholeShapeC = BaseShape2D<T, M, N>;
using GlobalDataOut = GlobalTensor<T, NDValidShapeC, NDWholeShapeC>; // C in GM
```

### Pipeline scheduling

This example overlaps data movement and compute using double buffering in L1 and L0 to improve utilization. Synchronization points ensure correct dependencies, including:

- Forward sync: `MTE2 -> MTE1`, `MTE1 -> MMAD`, `MMAD -> FIXPIPE`
- Reverse sync: `MTE1 -> MTE2`, `MMAD -> MTE1`

Pipeline overview:
Pipeline diagram: (to be added)

## Build and Run

1. Configure your Ascend CANN environment (example path):

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Run the example:

```bash
cd ${git_clone_path}/demos/baseline/gemm_basic
bash run.sh -r npu -v Ascend910B1
```

If the run succeeds, the output prints:

```text
test success
```
