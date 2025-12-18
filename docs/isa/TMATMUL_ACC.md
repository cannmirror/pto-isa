# TMATMUL_ACC

## Introduction

Matrix multiply with accumulator input (fused accumulate).

## Math Interpretation

For matrix shapes `A` (MxK), `B` (KxN), `C0` (MxN), `C1` (MxN):

$$ \mathrm{C1}_{i,j} = \mathrm{C0}_{i,j} + \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

## IR Syntax

Synchronous form (from `docs/ir/PTO-IR.md`):

```mlir
%acc1 = pto.tile.matmul.acc %acc0, %a, %b
    : tile<MxNxTc, #pto.tile_info<loc=Acc, layout=Lc>>,
      tile<MxKxTa, #pto.tile_info<loc=Left,  layout=La>>,
      tile<KxNxTb, #pto.tile_info<loc=Right, layout=Lb>>
   -> tile<MxNxTc, #pto.tile_info<loc=Acc, layout=Lc>>
```

Asynchronous form:

```mlir
%acc1, %e = pto.tile.matmul.acc %acc0, %a, %b wait(%e0, %e1)
    : tile<MxNxTc, #pto.tile_info<loc=Acc, layout=Lc>>,
      tile<MxKxTa, #pto.tile_info<loc=Left,  layout=La>>,
      tile<KxNxTb, #pto.tile_info<loc=Right, layout=Lb>>
   -> tile<MxNxTc, #pto.tile_info<loc=Acc, layout=Lc>>,
      !pto.event<producer = #pto.op<TMATMUL_ACC>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix,
                                 WaitEvents&... events);
```

## Constraints

- All constraints from `TMATMUL` apply to the `(cOutMatrix, aMatrix, bMatrix)` triple.
- **Implementation notes (A2A3/A5)**:
  - `TMATMUL_ACC_IMPL` uses `aMatrix.GetValidRow()`, `aMatrix.GetValidCol()`, and `bMatrix.GetValidCol()` for `m/k/n`.
  - `cInMatrix` is not validated by explicit assertions in the current implementations (target-defined behavior).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c0, c1;
  TMATMUL_ACC(c1, c0, a, b);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c0, c1;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(c0, 0x3000);
  TASSIGN(c1, 0x4000);
  TMATMUL_ACC(c1, c0, a, b);
}
```
