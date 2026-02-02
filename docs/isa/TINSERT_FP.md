# TINSERT_FP

## Introduction

Vector-quantization variant of `TINSERT` that also takes an `fp` (scaling) tile.

## See also

- TINSERT base instruction: `docs/isa/TINSERT.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                            uint16_t indexRow, uint16_t indexCol, WaitEvents&... events);
```
