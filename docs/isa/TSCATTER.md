# TSCATTER

## Introduction

Scatter rows of a source tile into a destination tile using per-element row indices.

## Math Interpretation

For each source element `(i, j)`, write:

$$ \mathrm{dst}_{\mathrm{idx}_{i,j},\ j} = \mathrm{src}_{i,j} $$

If multiple elements map to the same destination location, the final value is implementation-defined (last writer wins in the current implementation).

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileData& dst, TileData& src, TileInd& indexes, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `TileData::Loc` must be `TileType::Vec`.
  - `TileData::DType` must be one of: `int32_t`, `int`, `int16_t`, `half`, `float16_t`, `float`, `float32_t`.
  - `TileInd::DType` must be `uint16_t` or `uint32_t`.
  - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
  - No bounds checks are enforced on `indexes` values.
  - The current implementation iterates over the full static tile shape (`Rows x Cols`), not the runtime valid region.
- **Implementation checks (A5)**:
  - No `TSCATTER_IMPL` implementation is currently included for this target in `include/pto/common/pto_instr_impl.hpp`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TSCATTER(dst, src, idx);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(idx, 0x3000);
  TSCATTER(dst, src, idx);
}
```

