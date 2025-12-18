# Tile Programming Model

PTO Tile Lib programs operate on **Tiles**: fixed-shape on-chip buffers that are the unit of computation and data movement for PTO instructions.

This document describes the C++ tile types in `include/pto/common/pto_tile.hpp` and `include/pto/common/memory.hpp`.

## `pto::Tile` type

Tiles are declared as a C++ template type:

```cpp
pto::Tile<
  pto::TileType Loc_,
  Element_,
  Rows_,
  Cols_,
  pto::BLayout BFractal_     = pto::BLayout::RowMajor,
  RowValid_                 = Rows_,
  ColValid_                 = Cols_,
  pto::SLayout SFractal_    = pto::SLayout::NoneBox,
  SFractalSize_             = pto::TileConfig::fractalABSize,
  pto::PadValue PadVal_     = pto::PadValue::Null
>;
```

### Location (`TileType`)

`TileType` encodes the logical/physical storage class of the tile (vector buffer vs matrix/cube buffers):

- `TileType::Vec` (UB / vector pipeline)
- `TileType::Mat` (matrix buffer)
- `TileType::Left`, `TileType::Right` (matmul operands)
- `TileType::Acc` (matmul accumulator)
- `TileType::Bias`, `TileType::Scaling` (auxiliary tiles for some matmul/move paths)

The location participates in instruction overload selection and in many implementation `static_assert` checks.

### Layout (`BLayout` and `SLayout`)

Two layout knobs are used:

- `BLayout` (`RowMajor`/`ColMajor`) controls the base (unboxed) interpretation and is exposed via `Tile::isRowMajor`.
- `SLayout` (`NoneBox`, `RowMajor`, `ColMajor`) controls whether the tile uses an additional boxed/fractal layout.

Many cube/matmul paths require specific combinations, for example:

- `TileRight` uses a col-major `SLayout` to match hardware expectations.
- `TileAcc` uses a dedicated accumulator fractal size (`TileConfig::fractalCSize`).

### Valid region (`RowValid_` / `ColValid_`)

Tiles have a **static capacity** `(Rows_, Cols_)` and a **valid region** `(validRow, validCol)` describing how much of the tile is meaningful for an operation.

- If `RowValid_ == Rows_` and `ColValid_ == Cols_`, the valid region is fully static.
- If either valid parameter is `pto::DYNAMIC` (`-1`), the runtime valid region is stored and queried via `GetValidRow()` / `GetValidCol()`.

Operations typically iterate over the valid region, but this is not universal; some implementations use the full static shape for performance or alignment reasons. Always check the instruction-level constraints (see `docs/isa/*`).

### Padding (`PadValue`)

`PadValue` is a compile-time policy used by some implementations when handling out-of-valid regions (e.g., select/copy/pad paths). Its effect is target- and op-dependent.

## Common aliases

`include/pto/common/pto_tile.hpp` provides convenience aliases for common matmul tiles:

- `pto::TileLeft<Element, Rows, Cols>`
- `pto::TileRight<Element, Rows, Cols>`
- `pto::TileAcc<Element, Rows, Cols>`

These aliases pick target-appropriate layout/fractal defaults and are preferred in examples for matmul-related instructions.

## Address binding (`TASSIGN`)

In manual placement flows, `TASSIGN(tile, addr)` binds a tile object to an implementation-defined address. In auto flows, `TASSIGN(tile, addr)` may be a no-op depending on build configuration.

See `docs/isa/TASSIGN.md` for details.

## Minimal example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  TADD(c, a, b);
}
```
