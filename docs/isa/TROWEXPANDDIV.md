# TROWEXPANDDIV

## Introduction

Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector `src1`.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{0,i}} $$

## IR Syntax

Synchronous form:

```mlir
%dst = pto.tile.rowexpanddiv %src0, %src1 : tile<...>, tile<...> -> tile<...>
```

Asynchronous form:

```mlir
%dst, %e = pto.tile.rowexpanddiv %src0, %src1 wait(%e0, %e1)
    : tile<...>, tile<...> -> tile<...>, !pto.event<producer = #pto.op<TROWEXPANDDIV>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDDIV(TileDataDst& dst, TileDataDst& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `TileDataDst::DType == TileDataSrc1::DType` (compile-time).
  - `TileDataDst::DType` must be one of: `half`, `float16_t`, `float`, `float32_t`.
  - Tile shape/layout constraint (compile-time): `TileDataDst::isRowMajor` and `!TileDataSrc1::isRowMajor` and `TileDataSrc1::Cols == 1`.
  - Runtime: `src1.GetValidRow() == 1` and `src1.GetValidCol() == dst.GetValidRow()`.
- **Implementation checks (A5)**:
  - `TileData::DType` must be `float` or `half`.
  - `TileData::isRowMajor` (dst/src0 tile layout) must be true.
  - No explicit runtime checks are enforced on `src1` shape/valid.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  using RowVecT = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox>;

  TileT src0, dst;
  RowVecT src1(16);
  TROWEXPANDDIV(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  using RowVecT = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox>;

  TileT src0, dst;
  RowVecT src1(16);
  TASSIGN(src0, 0x1000);
  TASSIGN(dst,  0x2000);
  TASSIGN(src1, 0x3000);
  TROWEXPANDDIV(dst, src0, src1);
}
```

