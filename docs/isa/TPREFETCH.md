# TPREFETCH

## Introduction

Prefetch data from global memory into a tile-local cache/buffer (implementation-defined). This is typically used to reduce latency before a subsequent `TLOAD`.

Note: unlike most PTO instructions, `TPREFETCH` does **not** implicitly call `TSYNC(events...)` in the C++ wrapper.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

## Constraints

- Semantics and caching behavior are target/implementation-defined.
- Some targets may ignore prefetches or treat them as hints.
