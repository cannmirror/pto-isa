# Basic Topk Operator Example

## Overview

This example demonstrates how to implement a Topk operator using PTO, including project setup, build, and execution.

## Supported AI Processors

- A2/A3

## Directory Layout

```
kernels/topk/
├── scripts/
│   └── gen_data.py              # Generates input and golden output
├── CMakeLists.txt               # Build configuration
├── topk_kernel.cpp              # Kernel implementation
├── main.cpp                     # Host-side entry point
└── run.sh                       # Convenience script
```

## Operator Description

### Function

This example implements topk with fixed dimensions `[rows, cols] = [4800, 1024]`:

### Specification

| Item        | Value |
| ----------- | ----- |
| OpType      | `topk` |
| Inputs      | `[rows, cols] = [4800, 1024]` |
| Output      | `data`, `index` |
| Kernel name | `topk_kernel` |

### Tiling Parameters

The validation platform has 48 cores. The workload is split across cores.

Per-core shape:

- `rows = 100`, `cols = 1024`

## Implementation Notes

### Type definitions

The implementation defines topk representations. Load input data and index in GM to UB, use TSort32 to sort each 32 data, use TMrgsort for each tile. Extract data and index, then store back to gm seperately.

```cpp
    // data
    using DynShapeDim5 = Shape<1, 1, 1, singleLoopRow, validCol>;
    using DynStridDim5 = Stride<singleLoopRow * Cols, singleLoopRow * Cols, singleLoopRow * Cols, Cols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    // index
    using IndexShapeDim5 = Shape<1, 1, 1, 1, validCol>;
    using IndexStridDim5 = Stride<validCol, validCol, validCol, validCol, 1>;
    using IndexGlobalData = GlobalTensor<indexT, IndexShapeDim5, IndexStridDim5>;

    // sorted data and index
    using DstShapeDim5 = Shape<1, 1, 1, singleLoopRow, topk>;
    using DstStridDim5 = Stride<singleLoopRow * topk, singleLoopRow * topk, singleLoopRow * topk, topk, 1>;
    using DstDataGlobalData = GlobalTensor<T, DstShapeDim5, DstStridDim5>;
    using DstIdxGlobalData = GlobalTensor<indexT, DstShapeDim5, DstStridDim5>;
```

### Pipeline scheduling

This example overlaps data movement and compute using double buffering in L1 to improve utilization. In each iteration, two sets of operation are performed，TLOAD->TSORT32->TMRGSORT(include MRGSORT and MOV operation)->TSTORE. The pipeline dependece in each set is `MTE2->V->MTE1->V->MTE3`. TLOAD in the second sets can be performed before TSTORE in the first set is finished, so as others. Extra dependence `V->MTE2`is added to ensure that TLOAD in next iteration is performed after VEC operation is done in corresonding set.

## Build and Run

1. Configure your Ascend CANN environment (example path):

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Run the example:

```bash
cd ${git_clone_path}/demos/baseline/topk
bash run.sh -r npu -v Ascend910B1
```

If the run succeeds, the output prints:

```text
test success
```
