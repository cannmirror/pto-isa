# TMATMUL_BIAS

## Introduction

Matrix multiply with bias add.

## Math Interpretation

For matrix shapes `A` (MxK), `B` (KxN), `Bias` (1xN), `C` (MxN):

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} + \mathrm{Bias}_{0,j} $$

Bias broadcasting behavior is implementation-defined.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%acc = tmatmul.bias %a, %b, %bias
    : !pto.tile<MxKxTa, #pto.tile_info<loc=Left,  layout=La>>,
      !pto.tile<KxNxTb, #pto.tile_info<loc=Right, layout=Lb>>,
      !pto.tile<1xNxTb2, #pto.tile_info<loc=Bias, layout=Lbias>>
   -> !pto.tile<MxNxTc, #pto.tile_info<loc=Acc, layout=Lc>>
```

Asynchronous form:

```text
%acc, %e = tmatmul.bias %a, %b, %bias wait(%e0, %e1)
    : !pto.tile<MxKxTa, #pto.tile_info<loc=Left,  layout=La>>,
      !pto.tile<KxNxTb, #pto.tile_info<loc=Right, layout=Lb>>,
      !pto.tile<1xNxTb2, #pto.tile_info<loc=Bias, layout=Lbias>>
   -> !pto.tile<MxNxTc, #pto.tile_info<loc=Acc, layout=Lc>>,
      !pto.event<producer = #pto.op<TMATMUL_BIAS>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_BIAS(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData,
                                  WaitEvents&... events);
```

## Constraints

- All constraints from `TMATMUL` apply to the `(cMatrix, aMatrix, bMatrix)` triple.
- **Bias constraints (A2A3)**:
  - `TileBias::DType` must match `TileRes::DType`.
  - `TileBias::Loc == TileType::Bias` and `TileBias::Rows == 1`.
- **Bias constraints (A5)**:
  - `TileBias::DType` must match `TileRes::DType`.
  - `TileBias::Loc == TileType::Bias`, `TileBias::Rows == 1`, and `TileBias::isRowMajor`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using Bias = Tile<TileType::Bias, half, 1, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  Bias bias;
  C c;
  TMATMUL_BIAS(c, a, b, bias);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using Bias = Tile<TileType::Bias, half, 1, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  Bias bias;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(bias, 0x3000);
  TASSIGN(c, 0x4000);
  TMATMUL_BIAS(c, a, b, bias);
}
```
