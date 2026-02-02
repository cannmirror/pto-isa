# TQUANT

## Introduction

Quantize an FP32 tile into a lower-precision format (e.g. FP8), producing auxiliary exponent/scaling/max tiles. The quantization mode is a compile-time template parameter (`mode`).

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST RecordEvent TQUANT(TileDataSrc &src, TileDataExp &exp, TileDataOut &dst,
                            TileDataMax &max, TileDataSrc &scaling, WaitEvents&... events);
```

## Constraints

- This instruction is currently implemented for specific targets (see `include/pto/npu/*/TQuant.hpp`).
- Input type requirements and output tile types are mode/target-dependent.
