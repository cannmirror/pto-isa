# TROWEXPAND

## Introduction

Broadcast the first element of each source row across the destination row.

## Math Interpretation

$$
\\mathrm{dst}_{i,j} = \\mathrm{src}_{i,0}
$$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = trowexpand %src : !pto.tile<...> -> !pto.tile<...>
```

Asynchronous form:

```text
%dst, %e = trowexpand %src wait(%e0)
    : !pto.tile<...> -> !pto.tile<...>, !pto.event<producer = #pto.op<TROWEXPAND>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events);
```

## Constraints

Implementation checks (NPU):

- Tile location: `src` must be `TileType::Vec` and `dst` must be `TileType::Vec` (A5); A2A3 requires `src` to be Vec and requires ND layout for both.
- Tile layout: ND fractal (`isRowMajor` and `SLayout::NoneBox`) for both `src` and `dst`.
- Data type:
  - A2A3: element width must be 8/16/32-bit; `dst.DType == src.DType`.
  - A5: element width must be 8/16/32-bit; `dst.DType == src.DType`.
- Runtime valid checks:
  - A2A3: returns early if any of `dstValidRow`, `dstValidCol`, `srcValidRow`, `srcValidCol` is zero.
  - A5: asserts `srcValidRow == dstValidRow` and asserts `srcValidRow != 0 && srcValidCol != 0`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TROWEXPAND(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TROWEXPAND(dst, src);
}
```
