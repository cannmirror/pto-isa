# TSETFMATRIX

## Introduction

Set the FMATRIX register(s) used by IMG2COL-like operations from an `Img2colTileConfig` (target/implementation-defined).

## See also

- IMG2COL instruction: `docs/isa/TIMG2COL.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename T = uint64_t, typename... WaitEvents>
PTO_INST RecordEvent TSETFMATRIX(const Img2colTileConfig<T> &cfg = Img2colTileConfig<T>{}, WaitEvents&... events);
```
