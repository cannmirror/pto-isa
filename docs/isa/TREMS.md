# TREMS

## Introduction

Elementwise remainder between a tile and a scalar: `fmod(src, scalar)` (or `%` for integers).

Depending on the operand order, the scalar can act as the dividend (scalar/tile) or the divisor (tile/scalar).

## Math Interpretation

For each element `(i, j)` in the valid region:

- Tile/scalar:
  - Integer types: $$\mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \bmod \mathrm{scalar}$$
  - Floating types: $$\mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src}_{i,j}, \mathrm{scalar})$$
- Scalar/tile:
  - Integer types: $$\mathrm{dst}_{i,j} = \mathrm{scalar} \bmod \mathrm{src}_{i,j}$$
  - Floating types: $$\mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{scalar}, \mathrm{src}_{i,j})$$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Tile/scalar form:

```text
%dst = trems %src, %scalar : !pto.tile<...>, f32
```

Scalar/tile form:

```text
%dst = trems %scalar, %src : f32, !pto.tile<...>
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileData& dst, TileData& src0, typename TileData::DType scalar, WaitEvents&... events);
```

Note: `include/pto/common/pto_instr.hpp` only exposes the tile/scalar form as a C++ overload today.

## Constraints

- Division-by-zero behavior is target-defined; the CPU simulator asserts in debug builds.
- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TREMS(out, x, 3.0f);
}
```
