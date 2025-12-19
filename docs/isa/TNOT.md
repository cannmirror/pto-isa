# TNOT

## Introduction

Elementwise bitwise NOT of a tile.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \\mathrm{dst}_{i,j} = \\sim\\mathrm{src}_{i,j} $$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = tnot %src : !pto.tile<...>
```

Asynchronous form:

```text
%dst, %e = tnot %src wait(%e0)
    : !pto.tile<...>, !pto.event<producer = #pto.op<TNOT>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TNOT(TileData& dst, TileData& src, WaitEvents&... events);
```

## Constraints

- Intended for integral element types.
- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT x, out;
  TNOT(out, x);
}
```

