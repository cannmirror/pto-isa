# TORS

## Introduction

Elementwise bitwise OR of a tile and a scalar.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \\mathrm{dst}_{i,j} = \\mathrm{src}_{i,j} \\;|\\; \\mathrm{scalar} $$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = tors %src, %scalar : !pto.tile<...>, i32
```

Asynchronous form:

```text
%dst, %e = tors %src, %scalar wait(%e0)
    : !pto.tile<...>, i32, !pto.event<producer = #pto.op<TORS>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TORS(TileData& dst, TileData& src0, typename TileData::DType scalar, WaitEvents&... events);
```

## Constraints

- Intended for integral element types.
- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT x, out;
  TORS(out, x, 0x10u);
}
```

