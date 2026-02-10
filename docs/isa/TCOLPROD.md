# TCOLPROD

## Introduction

Reduce each column by prodming across rows.

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

$$ \mathrm{dst}_{0,j} = \prod_{i=0}^{R-1} \mathrm{src}_{i,j} $$

`isBinary` selects the implementation path (binary-tree accumulation vs. sequential accumulation).

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLPROD(TileDataOut& dst, TileDataIn& src, WaitEvents&... events);
```

## Constraints

Implementation checks (NPU):

- Tile location: `dst`, `src` must be `TileType::Vec`.
- Tile layout: all tiles must be ND fractal (`isRowMajor` and `SLayout::NoneBox`).
- DType consistency:
  - Must `dst.DType == src.DType`.
  - A2A3: `src.DType` must be one of `half`, `float`, `int16_t`, `int32_t`.
  - A5: `src.DType` must be one of `half`, `float`, `bfloat16`, `int16_t`, `int32_t`, `uint16_t`, `uint32_t`.
- Runtime valid checks:
  - `src.GetValidCol() == dst.GetValidCol()`.
  - Returns early if `src.GetValidRow() == 0` or `src.GetValidCol() == 0`.

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
  TCOLPROD(dst, src);
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
  TCOLPROD(dst, src);
}
```
