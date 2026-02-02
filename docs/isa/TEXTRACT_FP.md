# TEXTRACT_FP

## Introduction

Extract a sub-tile from a source tile, while also providing an `fp` (scaling) tile used for vector quantization parameters (target/implementation-defined).

## See also

- TEXTRACT base instruction: `docs/isa/TEXTRACT.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                            uint16_t indexRow, uint16_t indexCol, WaitEvents&... events);
```
