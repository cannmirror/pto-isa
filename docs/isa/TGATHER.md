# TGATHER

## Introduction

Gather/select elements using either an index tile or a compile-time mask pattern.

## Math Interpretation

Index-based gather (conceptual):

$$ \mathrm{dst}_{i,j} = \mathrm{src0}\!\left[\mathrm{indices}_{i,j}\right] $$

Exact index interpretation and bounds behavior are implementation-defined.

Mask-pattern gather is an implementation-defined selection/reduction controlled by `pto::MaskPattern`.

## IR Syntax

Index-based gather:

```mlir
%dst = pto.tile.gather %src0, %indices : tile<...> -> tile<...>
```

Mask-pattern gather:

```mlir
%dst = pto.tile.gather %src {maskPattern = #pto.mask_pattern<P0101>}
    : tile<...> -> tile<...>
```

Asynchronous form:

```mlir
%dst, %e = pto.tile.gather %src0, %indices wait(%e0, %e1)
    : tile<...> -> tile<...>, !pto.event<producer = #pto.op<TGATHER>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/type.hpp`:

```cpp
template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD& dst, TileDataS0& src0, TileDataS1& src1, WaitEvents&... events);

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData& dst, SrcTileData& src, WaitEvents&... events);
```

## Constraints

- **Index-based gather: implementation checks (A2A3)**:
  - `sizeof(DstTileData::DType)` must be `2` or `4` bytes.
  - `sizeof(Src1TileData::DType)` must be `4` bytes.
  - `DstTileData::DType` must be the same type as `Src0TileData::DType`.
- **Index-based gather: implementation checks (A5)**:
  - `sizeof(DstTileData::DType)` must be `1`, `2`, or `4` bytes.
  - `sizeof(Src1TileData::DType)` must be `2` or `4` bytes.
  - `DstTileData::DType` must be the same type as `Src0TileData::DType`.
- **Mask-pattern gather: implementation checks (A2A3)**:
  - Source element size must be `2` or `4` bytes.
  - `dst` and `src` must both be `TileType::Vec` and row-major.
  - `sizeof(dst element) == sizeof(src element)` and `dst.GetValidCol() == DstTileData::Cols` (continuous dst storage).
- **Mask-pattern gather: implementation checks (A5)**:
  - `dst` and `src` must both be `TileType::Vec` and row-major.
  - Supported dtypes are restricted to a target-defined set (checked via `static_assert` in the implementation), and `sizeof(dst element) == sizeof(src element)`.
- **Bounds / validity**:
  - Index bounds are not validated by explicit runtime assertions; out-of-range indices are target-defined.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src0;
  IdxT idx;
  DstT dst;
  TGATHER(dst, src0, idx);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src);
}
```
