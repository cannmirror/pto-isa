# TABS

## Introduction

Elementwise absolute value of a tile.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left|\mathrm{src}_{i,j}\right| $$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = tabs %src : !pto.tile<...> -> !pto.tile<...>
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileData& dst, TileData& src, WaitEvents&... events);
```

## Constraints

- **Implementation checks (CPU sim)**:
  - `TileData::DType` must be one of: `int32_t`, `int16_t`, `half`, `float`.
  - The implementation iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.
- **NPU support**:
  - Implemented on A2A3 (see `include/pto/npu/a2a3/TUnaryOp.hpp`).
  - A5 support is target-defined (no `include/pto/npu/a5/*` implementation is currently included for `TABS`).

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
