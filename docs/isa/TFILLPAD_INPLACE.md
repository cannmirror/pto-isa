# TFILLPAD_INPLACE

## Introduction

In-place fill/pad variant of TFILLPAD (implementation-defined).

## See also

- TFILLPAD overview and constraints: `docs/isa/TFILLPAD.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_INPLACE(DstTileData &dst, SrcTileData &src,
                            WaitEvents&... events);
```
