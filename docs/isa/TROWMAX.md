# TROWMAX

## Introduction

Reduce each row by taking the maximum across columns.

## Math Interpretation

$$
\\mathrm{dst}_{i,0} = \\max_j \\mathrm{src}_{i,j}
$$

## IR Syntax

Synchronous form:

```mlir
%dst = pto.tile.rowmax %src : tile<...> -> tile<...>
```

Asynchronous form:

```mlir
%dst, %e = pto.tile.rowmax %src wait(%e0)
    : tile<...> -> tile<...>, !pto.event<producer = #pto.op<TROWMAX>>
```

Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events);
```

## Constraints

Implementation checks (NPU):

- A2A3:
  - Tile location: `dst` and `src` must be `TileType::Vec`.
  - Tile layout: ND fractal (`isRowMajor` and `SLayout::NoneBox`).
  - Data types: `half` or `float`.
  - DType consistency: `dst.DType == src.DType`.
  - Runtime valid checks:
    - `srcValidCol != 0` and `srcValidRow != 0`.
    - `srcValidRow == dstValidRow` (the output valid row must match the input valid row).
- A5:
  - Data types: `half` or `float`.
  - DType consistency: `dst.DType == src.DType`.
  - No explicit runtime assertions on `validRow/validCol` in the implementation; the loops use `src.GetValidRow()` and `src.GetValidCol()`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWMAX(dst, src, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWMAX(dst, src, tmp);
}
```
