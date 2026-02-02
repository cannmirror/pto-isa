# TGEMV_BIAS

## Introduction

Tile-based GEMV with bias add.

## See also

- Full GEMV family description (TGEMV / TGEMV_ACC / TGEMV_BIAS): `docs/isa/TGEMV.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
  WaitEvents&... events);
```
