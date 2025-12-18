# Scalar Parameters and Enums

Many PTO intrinsics take scalar parameters in addition to tiles (e.g., comparison modes, rounding modes, atomic modes, or literal constants).

This document summarizes the scalar/enumeration types that appear in the public intrinsics in `include/pto/common/pto_instr.hpp`.

## Scalar values

Some instructions take scalar values as plain C++ types:

- `TADDS/TMULS/TDIVS/TEXPANDS`: scalar is `TileData::DType`.
- `TMINS`: scalar is a template type `T` and must be convertible to the tile element type.
- `TCI`: scalar `S` is a template type `T` and must match `TileData::DType` (enforced by `static_assert` in the implementation).

## Core enums

All enums below are available via `#include <pto/pto-inst.hpp>`.

### `pto::RoundMode`

Defined in `include/pto/common/constants.hpp`. Used by `TCVT` to specify rounding behavior (e.g., `RoundMode::CAST_RINT`).

### `pto::CmpMode`

Defined in `include/pto/common/type.hpp`. Used by `TCMPS` (and `TCMP`) for per-element comparisons (`EQ/NE/LT/GT/GE/LE`).

### `pto::MaskPattern`

Defined in `include/pto/common/type.hpp`. Used by the mask-pattern `TGATHER` variant to select a predefined 0/1 mask pattern.

### `pto::AtomicType`

Defined in `include/pto/common/constants.hpp`. Used as the template parameter to `TSTORE<..., AtomicType::AtomicAdd>` (or `AtomicNone`).

### `pto::AccToVecMode` and `pto::ReluPreMode`

Defined in `include/pto/common/constants.hpp`. Used by `TMOV` overloads when moving from accumulator tiles with optional quantization and/or ReLU behavior.

### `pto::PadValue`

Defined in `include/pto/common/constants.hpp`. Part of the `Tile<...>` template and used by some implementations to define how out-of-valid regions are treated (e.g., select/copy/pad paths).

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example(Tile<TileType::Vec, float, 16, 16>& dst,
             Tile<TileType::Vec, float, 16, 16>& src) {
  TCVT(dst, src, RoundMode::CAST_RINT);
  TMINS(dst, src, 0.0f);
}
```

