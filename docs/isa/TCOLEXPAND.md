# TCOLEXPAND

## Introduction

Broadcast the first element of each source column across the destination column.

## Math Interpretation

$$
\\mathrm{dst}_{i,j} = \\mathrm{src}_{0,j}
$$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = tcolexpand %src : !pto.tile<...> -> !pto.tile<...>
```

Asynchronous form:

```text
%dst, %e = tcolexpand %src wait(%e0)
    : !pto.tile<...> -> !pto.tile<...>, !pto.event<producer = #pto.op<TCOLEXPAND>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events);
```

## Constraints

- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TCOLEXPAND(dst, src);
}
```

