# TGATHERB

## Introduction

Gather elements using byte offsets.

## Math Interpretation

For each element in the valid region:

$$ \mathrm{dst}_{i,j} = *\left(\mathrm{srcBase} + \mathrm{offset}_{i,j}\right) $$

Exact bounds behavior is implementation-defined.

## IR Syntax

Synchronous form:

```mlir
%dst = pto.tile.gatherb %src, %offsets : tile<...> -> tile<...>
```

Asynchronous form:

```mlir
%dst, %e = pto.tile.gatherb %src, %offsets wait(%e0)
    : tile<...> -> tile<...>, !pto.event<producer = #pto.op<TGATHERB>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset, typename... WaitEvents>
PTO_INST RecordEvent TGATHERB(TileDataDst& dst, TileDataSrc& src, TileDataOffset& offset, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - Destination layout must be row-major (`TileDataDst::isRowMajor`).
  - Destination element size must be `1`, `2`, or `4` bytes (enforced via `static_assert` in the helper).
- **Implementation checks (A5)**:
  - Destination element size must be `1`, `2`, or `4` bytes.
- **Offset interpretation**:
  - Offsets are interpreted as `uint32_t` values (byte offsets) by the implementation.
  - Offset bounds are not validated by explicit runtime assertions; out-of-range offsets are target-defined.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, uint8_t, 1, 256>;
  using OffT = Tile<TileType::Vec, uint32_t, 1, 256>;
  using DstT = Tile<TileType::Vec, uint8_t, 1, 256>;
  SrcT src;
  OffT off;
  DstT dst;
  TGATHERB(dst, src, off);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, uint8_t, 1, 256>;
  using OffT = Tile<TileType::Vec, uint32_t, 1, 256>;
  using DstT = Tile<TileType::Vec, uint8_t, 1, 256>;
  SrcT src;
  OffT off;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(off, 0x2000);
  TASSIGN(dst, 0x3000);
  TGATHERB(dst, src, off);
}
```
