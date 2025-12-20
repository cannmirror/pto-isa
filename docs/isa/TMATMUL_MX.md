# TMATMUL_MX

## Introduction

Matrix multiply (GEMM) with additional scaling tiles for mixed-precision / quantized matmul on supported targets.

This instruction is currently implemented on A5 (see `include/pto/npu/a5/TMatmul.hpp`).

## Math Interpretation

Let:

- `M = aMatrix.GetValidRow()`
- `K = aMatrix.GetValidCol()`
- `N = bMatrix.GetValidCol()`

Conceptually, the result corresponds to a matrix multiply over the effective matmul domain (`0 <= i < M`, `0 <= j < N`), with the scaling tiles `aScaleMatrix` / `bScaleMatrix` configuring implementation-defined mixed-precision behavior:

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

The exact role of `aScaleMatrix` / `bScaleMatrix` (and any dequant/quant semantics) is target-defined.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous forms (conceptual):

```text
%c = tmatmul.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c_out = tmatmul.mx.acc %c_in, %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c = tmatmul.mx.bias %a, %a_scale, %b, %b_scale, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
    TileRightScale &bScaleMatrix, WaitEvents&... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
    TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents&... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
    TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A5)**:
  - `m/k/n` are taken from `aMatrix.GetValidRow()`, `aMatrix.GetValidCol()`, `bMatrix.GetValidCol()`.
  - Static legality checks are enforced via `CheckMadMxValid<...>()` (types, shapes, fractals, and scaling tile legality).
- **Bias form**:
  - `TileBias::DType` must be `float` and `TileBias::Loc == TileType::Bias` with `TileBias::Rows == 1` (A5 checks via `static_assert`).

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  // Exact tile types depend on the target’s MX matmul ABI; this is a schematic example.
}
```

