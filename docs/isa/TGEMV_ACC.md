# TGEMV_ACC

## Introduction

Tile-based GEMV with explicit accumulator input tile (`cInMatrix`) and output tile (`cOutMatrix`).

## See also

- Full GEMV family description (TGEMV / TGEMV_ACC / TGEMV_BIAS): `docs/isa/TGEMV.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
  WaitEvents&... events);
```
