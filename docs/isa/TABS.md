# TABS

## Introduction

Elementwise absolute value of a tile.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left|\mathrm{src}_{i,j}\right| $$

## IR Syntax

Synchronous form:

```mlir
%dst = pto.tile.abs %src : tile<...> -> tile<...>
```

Asynchronous form:

```mlir
%dst, %e = pto.tile.abs %src wait(%e0)
    : tile<...> -> tile<...>, !pto.event<producer = #pto.op<TABS>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileData& dst, TileData& src, WaitEvents&... events);
```

## Constraints

- **Implementation checks (CPU sim)**:
  - `TileData::DType` must be one of: `int32_t`, `int`, `int16_t`, `half`, `float`.
  - The implementation iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.
- **NPU support**:
  - No NPU implementation is currently included in `include/pto/common/pto_instr_impl.hpp` for `TABS`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TABS(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TABS(dst, src);
}
```

