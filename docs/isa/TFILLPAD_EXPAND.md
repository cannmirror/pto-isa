# TFILLPAD_EXPAND

## Introduction

Expand fill/pad variant of TFILLPAD (allows dst to be larger than src; implementation-defined).

## See also

- TFILLPAD overview and constraints: `docs/isa/TFILLPAD.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData &dst, SrcTileData &src,
                            WaitEvents&... events);
```
