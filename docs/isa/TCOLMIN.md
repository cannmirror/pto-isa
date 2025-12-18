# TCOLMIN

## Introduction

Reduce each column by taking the minimum across rows.

## Math Interpretation

$$
\\mathrm{dst}_{0,j} = \\min_i \\mathrm{src}_{i,j}
$$

## IR Syntax

Synchronous form:

```mlir
%dst = pto.tile.colmin %src : tile<...> -> tile<...>
```

Asynchronous form:

```mlir
%dst, %e = pto.tile.colmin %src wait(%e0)
    : tile<...> -> tile<...>, !pto.event<producer = #pto.op<TCOLMIN>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMIN(TileDataOut& dst, TileDataIn& src, WaitEvents&... events);
```

## Constraints

Implementation checks (NPU):

- Tile location: `dst` and `src` must be `TileType::Vec`.
- Tile layout: both tiles must be ND fractal (`isRowMajor` and `SLayout::NoneBox`).
- Data types:
  - A2A3: `half`, `float`, `int16_t`, `int32_t`.
  - A5: `half`, `float`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `bfloat16_t`.
- DType consistency: `dst.DType == src.DType`.
- Runtime valid checks:
  - `src.GetValidCol() == dst.GetValidCol()`.
  - If `src.GetValidRow() == 0` or `src.GetValidCol() == 0`, the implementation returns early.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TCOLMIN(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCOLMIN(dst, src);
}
```
